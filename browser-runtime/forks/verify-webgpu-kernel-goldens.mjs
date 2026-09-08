import { chromium } from "../../webui/node_modules/playwright/index.mjs";
import ts from "../../webui/node_modules/typescript/lib/typescript.js";
import { dirname, resolve } from "node:path";
import { createServer } from "node:http";
import { readFile, writeFile } from "node:fs/promises";
const [
  manifestPath,
  webllmRepository,
  reportPath = "/tmp/drowse-webgpu-kernel-goldens.json",
] = process.argv.slice(2);
if (!manifestPath || !webllmRepository)
  throw Error(
    "usage: node verify-webgpu-kernel-goldens.mjs MANIFEST WEBLLM_REPOSITORY [REPORT]",
  );
const root = dirname(resolve(manifestPath));
const manifest = JSON.parse(
  await readFile(resolve(manifestPath), "utf8"),
).filter((entry) => entry.family !== "sampling" && entry.family !== "controls");
for (const entry of manifest)
  entry.code = await readFile(resolve(root, entry.file), "utf8");
const importTypescript = async (name) => {
  const source = await readFile(
    resolve(webllmRepository, "src", name + ".ts"),
    "utf8",
  );
  const javascript = ts.transpileModule(source, {
    compilerOptions: {
      target: ts.ScriptTarget.ES2022,
      module: ts.ModuleKind.ES2022,
    },
  }).outputText;
  return import(
    "data:text/javascript;base64," + Buffer.from(javascript).toString("base64")
  );
};
const { drowseSamplingTopKShader } = await importTypescript("drowse_gpu_topk");
for (const capacity of [1, 8, 64, 1024])
  for (const merge of [false, true])
    manifest.push({
      file: `sampling-${capacity}-${merge}`,
      family: "sampling",
      capacity,
      merge,
      code: drowseSamplingTopKShader(capacity, merge),
    });
const { drowseCurveControlShader } = await importTypescript(
  "drowse_gpu_controls",
);
manifest.push({
  file: "curve-control-update",
  family: "controls",
  stride: 97,
  code: drowseCurveControlShader(97),
});
const server = createServer((req, res) =>
  res.end("<title>Numerical GPU validation</title>"),
);
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const browser = await chromium.launch({ channel: "chrome", headless: true });
try {
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  const result = await page.evaluate(async (manifest) => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw Error("WebGPU unavailable");
    const device = await adapter.requestDevice();
    let lost = null;
    device.lost.then((i) => (lost = i.message));
    const report = [];
    const data = (n, fn) => Float32Array.from({ length: n }, (_, i) => fn(i));
    function buffer(array) {
      const b = device.createBuffer({
        size: Math.max(16, array.byteLength),
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_DST |
          GPUBufferUsage.COPY_SRC,
      });
      device.queue.writeBuffer(b, 0, array);
      return b;
    }
    async function dispatch(entry, arrays, groups, args = {}) {
      device.pushErrorScope("validation");
      const module = device.createShaderModule({ code: entry.code });
      const info = await module.getCompilationInfo();
      if (info.messages.some((m) => m.type === "error"))
        throw Error(
          entry.file + ": " + info.messages.map((m) => m.message).join("\n"),
        );
      const pipeline = await device.createComputePipelineAsync({
        layout: "auto",
        compute: { module, entryPoint: "main_kernel" },
      });
      const entries = [],
        owned = [];
      for (const m of entry.code.matchAll(
        /@binding\((\d+)\) var<storage, [^>]+> (\w+)\s*:/g,
      )) {
        if (!arrays[m[2]]) throw Error("missing " + m[2]);
        const b = buffer(arrays[m[2]]);
        owned.push(b);
        entries.push({ binding: +m[1], resource: { buffer: b } });
      }
      const fields = [
        ...entry.code
          .match(/struct PODArgs \{([\s\S]*?)\}/)[1]
          .matchAll(/(\w+): [iu]32/g),
      ].map((m) => m[1]);
      const uniform = device.createBuffer({
        size: Math.max(16, fields.length * 4),
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(
        uniform,
        0,
        Int32Array.from(fields, (f) =>
          f === "packGridDimX" ? groups : (args[f] ?? 1),
        ),
      );
      owned.push(uniform);
      const ub = +entry.code.match(/@binding\((\d+)\) var<uniform>/)[1];
      entries.push({ binding: ub, resource: { buffer: uniform } });
      const bind = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries,
      });
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(groups);
      pass.end();
      const reads = [];
      for (const m of entry.code.matchAll(
        /@binding\((\d+)\) var<storage, read_write> (\w+)\s*:/g,
      )) {
        const a = arrays[m[2]],
          src = entries.find((e) => e.binding === +m[1]).resource.buffer;
        const read = device.createBuffer({
          size: Math.max(16, a.byteLength),
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        encoder.copyBufferToBuffer(src, 0, read, 0, a.byteLength);
        reads.push({ name: m[2], read, array: a });
      }
      device.queue.submit([encoder.finish()]);
      const output = {};
      for (const r of reads) {
        await r.read.mapAsync(GPUMapMode.READ);
        output[r.name] = new r.array.constructor(
          r.read.getMappedRange().slice(0, r.array.byteLength),
        );
        r.read.unmap();
        r.read.destroy();
      }
      const err = await device.popErrorScope();
      for (const b of owned) b.destroy();
      if (err) throw Error(entry.file + ": " + err.message);
      if (lost) throw Error("device lost " + lost);
      return output;
    }
    function compare(actual, expected, label, atol = 2e-5, rtol = 2e-5) {
      let max = 0;
      for (let i = 0; i < expected.length; i++) {
        const e = Math.abs(actual[i] - expected[i]);
        max = Math.max(max, e);
        if (
          !Number.isFinite(actual[i]) ||
          e > atol + rtol * Math.abs(expected[i])
        )
          throw Error(
            label +
              " mismatch " +
              i +
              ": " +
              actual[i] +
              " expected " +
              expected[i] +
              " error " +
              e,
          );
      }
      return max;
    }
    for (const entry of manifest.filter((e) => e.family === "affine")) {
      const { h, b, t } = entry;
      for (const mode of ["inactive", "push", "ablate", "mixed"]) {
        const residual = data(b * t * h, (i) => Math.sin(i * 0.13)),
          active = data(4, (i) => (mode === "inactive" ? 0 : i === 2 ? 0 : 1)),
          along = data(4, (i) => (i - 1.5) * 0.1),
          basis = data(4 * 8 * h, (i) => Math.sin(i * 0.27) / Math.sqrt(h)),
          neutral = data(4 * h, (i) => Math.cos(i * 0.19) * 0.1),
          target = data(4 * 8, (i) => Math.sin(i * 0.17)),
          kappa = data(4 * 8, (i) =>
            mode === "ablate"
              ? 1
              : mode === "mixed"
                ? i % 3 === 0
                  ? 1
                  : 0
                : 0,
          );
        const expected = Float64Array.from(residual);
        for (let tok = 0; tok < b * t; tok++)
          for (let g = 0; g < 4; g++)
            if (active[g] && along[g]) {
              const delta = new Float64Array(8);
              for (let r = 0; r < 8; r++) {
                let dot = 0;
                if (kappa[g * 8 + r])
                  for (let j = 0; j < h; j++)
                    dot +=
                      (expected[tok * h + j] - neutral[g * h + j]) *
                      basis[(g * 8 + r) * h + j];
                delta[r] = target[g * 8 + r] - kappa[g * 8 + r] * dot;
              }
              for (let j = 0; j < h; j++) {
                let shift = 0;
                for (let r = 0; r < 8; r++)
                  shift += delta[r] * basis[(g * 8 + r) * h + j];
                expected[tok * h + j] += active[g] * along[g] * shift;
              }
            }
        const output = await dispatch(
          entry,
          {
            residual_ptr: residual,
            active_ptr: active,
            along_ptr: along,
            basis_ptr: basis,
            neutral_ptr: neutral,
            target_ptr: target,
            kappa_ptr: kappa,
            output_ptr: new Float32Array(b * t * h),
          },
          b * t,
        );
        report.push({
          family: "affine",
          h,
          t,
          mode,
          maxError: compare(
            output.output_ptr,
            expected,
            entry.file + mode,
            mode === "inactive" ? 0 : 2e-5,
            mode === "inactive" ? 0 : 2e-5,
          ),
        });
      }
    }
    for (const entry of manifest.filter(
      (e) => e.family === "topk" && e.kernel === "drowse_exact_top8_tiles",
    )) {
      const { columns, rows } = entry;
      for (const mode of ["ties", "tails", "random"]) {
        const scores = data(rows * columns, (i) =>
          mode === "ties"
            ? 1
            : mode === "tails"
              ? i % columns === columns - 1
                ? 100
                : -2
              : Math.sin(i * 4.17),
        );
        let candidates = Math.ceil(columns / 256);
        let output = await dispatch(
          entry,
          {
            source_ptr: scores,
            output_values_ptr: new Float32Array(rows * candidates * 8),
            output_indices_ptr: new Int32Array(rows * candidates * 8),
          },
          rows * candidates,
        );
        const merge = manifest.find(
          (m) =>
            m.family === "topk" &&
            m.columns === columns &&
            m.kernel === "drowse_exact_top8_merge",
        );
        while (candidates > 1) {
          const blocks = Math.ceil(candidates / 32);
          output = await dispatch(
            merge,
            {
              source_values_ptr: output.output_values_ptr,
              source_indices_ptr: output.output_indices_ptr,
              output_values_ptr: new Float32Array(rows * blocks * 8),
              output_indices_ptr: new Int32Array(rows * blocks * 8),
            },
            rows * blocks,
            { candidates_per_row: candidates, cse_v3: blocks },
          );
          candidates = blocks;
        }
        for (let r = 0; r < rows; r++) {
          const ids = Array.from({ length: columns }, (_, i) => i)
            .sort(
              (a, b) =>
                scores[r * columns + b] - scores[r * columns + a] || a - b,
            )
            .slice(0, 8);
          for (let k = 0; k < 8; k++)
            if (
              output.output_indices_ptr[r * 8 + k] !== ids[k] ||
              output.output_values_ptr[r * 8 + k] !==
                scores[r * columns + ids[k]]
            )
              throw Error(
                entry.file +
                  mode +
                  " row " +
                  r +
                  " rank " +
                  k +
                  " got " +
                  output.output_indices_ptr[r * 8 + k] +
                  " expected " +
                  ids[k],
              );
        }
        report.push({ family: "topk", columns, rows, mode, exact: true });
      }
    }
    for (const columns of [17, 257, 4096, 70001]) {
      const rows = 2,
        candidates = Math.ceil(columns / 256),
        scores = data(rows * columns, (i) =>
          i % 7 === 0 ? 1 : Math.sin(i * 0.21),
        );
      const tile = manifest.find(
        (e) =>
          e.family === "topk-dynamic" && e.kernel === "drowse_exact_top8_tiles",
      );
      let output = await dispatch(
        tile,
        {
          source_ptr: scores,
          output_values_ptr: new Float32Array(rows * candidates * 8),
          output_indices_ptr: new Int32Array(rows * candidates * 8),
        },
        rows * candidates,
        { columns, cse_v1: candidates },
      );
      const merge = manifest.find(
        (e) =>
          e.family === "topk-dynamic" && e.kernel === "drowse_exact_top8_merge",
      );
      output = await dispatch(
        merge,
        {
          source_values_ptr: output.output_values_ptr,
          source_indices_ptr: output.output_indices_ptr,
          output_values_ptr: new Float32Array(rows * 8),
          output_indices_ptr: new Int32Array(rows * 8),
        },
        rows,
        { rows, candidates_per_row: candidates },
      );
      for (let r = 0; r < rows; r++) {
        const ids = Array.from({ length: columns }, (_, i) => i)
          .sort(
            (a, b) =>
              scores[r * columns + b] - scores[r * columns + a] || a - b,
          )
          .slice(0, 8);
        for (let k = 0; k < 8; k++)
          if (
            output.output_indices_ptr[r * 8 + k] !== ids[k] ||
            output.output_values_ptr[r * 8 + k] !== scores[r * columns + ids[k]]
          )
            throw Error(
              "dynamic " +
                columns +
                " row " +
                r +
                " rank " +
                k +
                " got " +
                output.output_indices_ptr[r * 8 + k] +
                " expected " +
                ids[k],
            );
      }
      report.push({ family: "topk-dynamic", columns, rows, exact: true });
    }
    for (const entry of manifest.filter((e) => e.family === "transport")) {
      const { h, layers } = entry;
      const source = data(layers * h, (i) => Math.sin(i * 0.17)),
        matrices = data(
          layers * h * h,
          (i) => Math.cos(i * 0.29) / Math.sqrt(h),
        );
      const expected = new Float64Array(layers * h);
      for (let l = 0; l < layers; l++)
        for (let j = 0; j < h; j++)
          for (let k = 0; k < h; k++)
            expected[l * h + j] +=
              source[l * h + k] * matrices[(l * h + j) * h + k];
      const output = await dispatch(
        entry,
        {
          source_ptr: source,
          matrices_ptr: matrices,
          output_ptr: new Float32Array(layers * h),
        },
        layers * Math.ceil(h / 4),
        { layers },
      );
      report.push({
        family: "transport",
        h,
        layers,
        maxError: compare(output.output_ptr, expected, entry.file),
      });
    }
    for (const entry of manifest.filter((e) => e.family === "curve")) {
      const { h, t, stride } = entry;
      for (const active of [0, 0.4, 1])
        for (const magnitude of [1, 100]) {
          const residual = data(t * h, (i) => Math.sin(i * 0.13)),
            basis = data(4 * 8 * h, (i) => Math.sin(i * 0.27) / Math.sqrt(h)),
            neutral = new Float32Array(4 * h),
            q = data(t * 8, (i) => Math.cos(i * 0.17)),
            coordinates = data(t * 8, (i) => Math.sin(i * 0.19) * magnitude),
            parameters = new Float32Array(4 * stride);
          parameters[0] = active;
          const expected = new Float64Array(t * h);
          for (let tok = 0; tok < t; tok++) {
            let norm0 = 0,
              norm1 = 0;
            for (let j = 0; j < h; j++) {
              let before = 0,
                after = 0;
              for (let r = 0; r < 8; r++) {
                before += q[tok * 8 + r] * basis[r * h + j];
                after += coordinates[tok * 8 + r] * basis[r * h + j];
              }
              expected[tok * h + j] = residual[tok * h + j] - before + after;
              norm0 += residual[tok * h + j] ** 2;
              norm1 += expected[tok * h + j] ** 2;
            }
            const scale = Math.min(
              (3 * Math.sqrt(norm0)) / Math.max(Math.sqrt(norm1), 1e-6),
              1,
            );
            for (let j = 0; j < h; j++)
              expected[tok * h + j] =
                active * expected[tok * h + j] * scale +
                (1 - active) * residual[tok * h + j];
          }
          const output = await dispatch(
            entry,
            {
              residual_buffer_ptr: residual,
              basis_buffer_ptr: basis,
              neutral_buffer_ptr: neutral,
              q_buffer_ptr: q,
              coordinate_buffer_ptr: coordinates,
              parameter_buffer_ptr: parameters,
              output_ptr: new Float32Array(t * h),
            },
            t,
          );
          report.push({
            family: "curve",
            h,
            t,
            active,
            magnitude,
            maxError: compare(
              output.output_ptr,
              expected,
              entry.file,
              3e-5,
              3e-5,
            ),
          });
        }
    }
    for (const capacity of [1, 8, 64, 1024])
      for (const columns of [17, 257, 50003, 262144]) {
        const count = Math.min(capacity, columns),
          scores = data(columns, (i) => (i % 7 === 0 ? 1 : Math.sin(i * 0.21)));
        let lists = Math.ceil(columns / Math.max(256, capacity));
        const tile = manifest.find(
          (e) => e.family === "sampling" && e.capacity === capacity && !e.merge,
        );
        let output = await dispatch(
          tile,
          {
            source_values: scores,
            output_values: new Float32Array(lists * capacity),
            output_indices: new Int32Array(lists * capacity),
          },
          lists,
          { columns, lists },
        );
        const merge = manifest.find(
          (e) => e.family === "sampling" && e.capacity === capacity && e.merge,
        );
        while (lists > 1) {
          const nextLists = Math.ceil(lists / 2);
          output = await dispatch(
            merge,
            {
              source_values: output.output_values,
              source_indices: output.output_indices,
              output_values: new Float32Array(nextLists * capacity),
              output_indices: new Int32Array(nextLists * capacity),
            },
            nextLists,
            { columns, lists },
          );
          lists = nextLists;
        }
        const ids = Array.from({ length: columns }, (_, i) => i)
          .sort((a, b) => scores[b] - scores[a] || a - b)
          .slice(0, count);
        for (let k = 0; k < count; k++)
          if (
            output.output_indices[k] !== ids[k] ||
            output.output_values[k] !== scores[ids[k]]
          )
            throw Error(
              "sampling " +
                columns +
                " capacity " +
                capacity +
                " rank " +
                k +
                " got " +
                output.output_indices[k] +
                " expected " +
                ids[k],
            );
        report.push({
          family: "sampling",
          columns,
          capacity,
          count,
          exact: true,
        });
      }
    for (const count of [4, 132]) {
      const entry = manifest.find((e) => e.family === "controls"),
        parameters = data(count * entry.stride, (i) => i * 0.01),
        active = data(count, (i) => i % 2),
        expected = parameters.slice();
      for (let i = 0; i < count; i++) expected[i * entry.stride] = active[i];
      const output = await dispatch(
        entry,
        { masks: active, parameters },
        Math.ceil(count / 64),
        { count },
      );
      report.push({
        family: "controls",
        count,
        exact:
          compare(output.parameters, expected, "sparse curve masks", 0, 0) ===
          0,
      });
    }
    device.destroy();
    return { adapter: adapter.info, checks: report };
  }, manifest);
  await writeFile(reportPath, JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result, null, 2));
} finally {
  await browser.close();
  await new Promise((r) => server.close(r));
}
