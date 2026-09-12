import { chromium } from 'playwright';
import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
if (!process.argv[2] || !process.argv[3]) throw Error('Usage: node pythia-projection-webgpu.mjs <model.wasm> <report.json> [local-harness-url]');
const wasm = await readFile(process.argv[2]);
const text = wasm.toString();
const start = text.indexOf('// Function: drowse_pairwise_output_projection_kernel');
const end = text.indexOf('\n}\n\n', start) + 4;
if (start < 0 || end < start) throw Error('Projection shader missing');
const shader = text.slice(start, end);
const browser = await chromium.launch({ channel: 'chrome', headless: true });
try {
  const page = await browser.newPage();
  await page.goto(process.argv[4] ?? 'http://127.0.0.1:4199');
  const result = await page.evaluate(async code => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter || adapter.info.isFallbackAdapter) throw Error('Hardware adapter required');
    const device = await adapter.requestDevice();
    try {
      const module = device.createShaderModule({ code });
      const info = await module.getCompilationInfo();
      if (info.messages.some(m => m.type === 'error')) throw Error(info.messages.map(m => m.message).join('\n'));
      const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'drowse_pairwise_output_projection_kernel' } });
      const results = [];
      for (const cancellation of [true, false]) {
        const n = 6, hidden = 512, v = 7;
        const inputs = new Float32Array(n * hidden), weights = new Float32Array(v * hidden);
        for (let i = 0; i < inputs.length; i++) inputs[i] = cancellation ? 1 : 20 + 10 * Math.sin(i * .137);
        for (let j = 0; j < v; j++) for (let k = 0; k < hidden; k++) {
          weights[j * hidden + k] = cancellation ? (k === 0 ? 1e8 : k === 32 ? .25 * (j + 1) : k === 64 ? -1e8 : 0) : .15 + .05 * Math.sin(k * .191) + .001 * Math.cos(j * 3.11 + k * .251);
        }
        const expected = new Array(n * v).fill(0);
        for (let i = 0; i < n; i++) for (let j = 0; j < v; j++) for (let k = 0; k < hidden; k++) expected[i * v + j] += inputs[i * hidden + k] * weights[j * hidden + k];
        const buffers = [];
        const allocate = (size, usage, data) => {
          const buffer = device.createBuffer({ size, usage }); buffers.push(buffer);
          if (data) device.queue.writeBuffer(buffer, 0, data);
          return buffer;
        };
        try {
          const output = allocate(n * v * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
          const source = allocate(inputs.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, inputs);
          const weight = allocate(weights.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, weights);
          const args = allocate(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, new Int32Array([n, v, Math.ceil(v / 4), 0]));
          const readback = allocate(n * v * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
          const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [output, source, weight, args].map((buffer, binding) => ({ binding, resource: { buffer } })) });
          const encoder = device.createCommandEncoder();
          const pass = encoder.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(Math.ceil(v / 4), n); pass.end();
          encoder.copyBufferToBuffer(output, 0, readback, 0, n * v * 4);
          device.queue.submit([encoder.finish()]);
          await readback.mapAsync(GPUMapMode.READ);
          const actual = Array.from(new Float32Array(readback.getMappedRange()));
          readback.unmap();
          const errors = actual.map((value, i) => Math.abs(value - expected[i]));
          const maxError = Math.max(...errors);
          let maxTV = 0;
          const softmax = row => { const m = Math.max(...row), p = row.map(x => Math.exp(x - m)), s = p.reduce((a, b) => a + b); return p.map(x => x / s); };
          for (let i = 0; i < n; i++) {
            const p = softmax(expected.slice(i * v, (i + 1) * v));
            const q = softmax(actual.slice(i * v, (i + 1) * v));
            maxTV = Math.max(maxTV, p.reduce((s, value, j) => s + Math.abs(value - q[j]), 0) / 2);
          }
          if ((cancellation && maxError !== 0) || !Number.isFinite(maxTV) || maxTV > 1e-4) throw Error(JSON.stringify({ cancellation, maxError, maxTV, actual, expected }));
          results.push({ cancellation, rows: n, hidden, vocab: v, maxError, maxTV });
        } finally { buffers.forEach(buffer => buffer.destroy()); }
      }
      return { userAgent: navigator.userAgent, adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture }, results };
    } finally { device.destroy(); }
  }, shader);
  result.modelLibrarySha256 = createHash('sha256').update(wasm).digest('hex');
  result.shaderSha256 = createHash('sha256').update(shader).digest('hex');
  result.passed = true;
  await writeFile(process.argv[3], JSON.stringify(result, null, 2), { flag: 'wx' });
  console.log(JSON.stringify(result));
} finally { await browser.close(); }
