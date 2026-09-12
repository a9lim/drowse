<script lang="ts">
  import { onMount as onInterfaceMount } from "svelte";
  import { registerInterfaceController } from "../../lib/workspaceController";
  import { authoredDraftSchema } from "../../lib/interfaceSchemas";
  import { ToolError } from "../../lib/webmcp/types";

  import MorphText from "../../lib/ui/MorphText.svelte";
  import FluentIcon from "../../lib/ui/FluentIcon.svelte";
  import { slide } from "svelte/transition";
  import { collapseIn, collapseOut } from "../../lib/motion";
  // Custom-nodes authoring — the historical path.
  //
  // The user brings labelled corpora.  The ``auto-domain`` switch decides
  // what happens to the geometry:
  //
  //   * off — pick a domain (box 1D/2D/3D with per-axis lo/hi + periodic,
  //     or sphere), place every node at hand-supplied coordinates, and
  //     submit through ``apiManifolds.create``.  Poisedness needs
  //     ``2n+1`` nodes.
  //   * on  — no domain, no coordinates: submit through
  //     ``apiManifolds.createDiscover`` and let the fitter derive the
  //     layout per-model via the same pca / spectral hyperparams the
  //     auto-generated tab exposes.  Only ≥2 nodes are required.

  import { tick } from "svelte";
  import { apiManifolds } from "../../lib/runtime/services";
  import { closeDrawer, openDrawer, refreshManifoldList } from "../../lib/stores.svelte";
  import { pushToast } from "../../lib/stores/toasts.svelte";
  import { userFacingError } from "../../lib/runtime/userFacingError";
  import { getRuntimeManifoldFitMaxIntrinsicDim } from "../../lib/runtime/registry";
  import { runtimeClient } from "../../lib/runtime/client";
  import { PER_NODE_ROLE_HELP } from "../../lib/manifolds/selectors";
  import { quotientDescription } from "../../lib/manifolds/surfaceGeometry";
  import type {
    AxisSpec,
    CreateDiscoverManifoldRequest,
    CreateManifoldRequest,
    ManifoldDomain,
  } from "../../lib/types";
  import Select from "../../lib/Select.svelte";
  import Checkbox from "../../lib/Checkbox.svelte";
  import NumberInput from "../../lib/NumberInput.svelte";
  import AdvancedSection from "../../lib/builder/AdvancedSection.svelte";
  import ValidationBlock from "../../lib/builder/ValidationBlock.svelte";
  import DiscoverTuningFields from "./DiscoverTuningFields.svelte";
  import FitMethodPicker from "./FitMethodPicker.svelte";
  import {
    defaultTuning,
    identitySlugs,
    slug,
    tuningHyperparams,
    tuningMessages,
    type ManifoldIdentity,
  } from "./shared";

  let { identity, oncomplete }: { identity: ManifoldIdentity; oncomplete?: () => void } = $props();

  let autoDomain = $state(false);
  const maxDimLimit = getRuntimeManifoldFitMaxIntrinsicDim();
  const browserMode = runtimeClient.mode !== "http";
  const tuning = $state(defaultTuning(maxDimLimit, browserMode));
  let advancedOpen = $state(false);
  let submitting = $state(false);
  let validationAttempted = $state(false);
  let formRegion: HTMLDivElement | null = $state(null);
  let addNodeButton: HTMLButtonElement | null = $state(null);

  // ---------- domain ----------

  type DomainKind = "box" | "sphere" | "klein" | "projective";
  let domainKind: DomainKind = $state("box");
  let boxDim = $state(2); // 1 | 2 | 3
  let sphereDim = $state(2);

  // Per-axis specs — three slots authored; only the first ``boxDim``
  // are used.  Defaults give a unit square that's easy to author on.
  interface AxisDraft {
    name: string;
    lo: number;
    hi: number;
    periodic: boolean;
  }
  let axisDrafts: AxisDraft[] = $state([
    { name: "x", lo: 0, hi: 1, periodic: false },
    { name: "y", lo: 0, hi: 1, periodic: false },
    { name: "z", lo: 0, hi: 1, periodic: false },
  ]);

  const intrinsicDim = $derived(domainKind === "box" ? boxDim : domainKind === "sphere" ? sphereDim : 2);
  const minNodes = $derived.by(() => domainKind === "projective" ? 6 : 2 * intrinsicDim + 1);

  /** Build the wire ManifoldDomain from the form state. */
  function buildDomain(): ManifoldDomain {
    if (!browserMode && domainKind === "klein") return { type: "klein" };
    if (!browserMode && domainKind === "projective") return { type: "projective", dim: 2 };
    if (domainKind === "sphere") {
      return { type: "sphere", dim: sphereDim };
    }
    const axes: AxisSpec[] = axisDrafts.slice(0, boxDim).map((a) => ({
      name: a.name,
      periodic: a.periodic,
      period: a.hi - a.lo,
      lo: a.lo,
      hi: a.hi,
    }));
    return { type: "box", axes };
  }

  function pickBoxDim(d: number): void {
    boxDim = d;
    domainKind = "box";
    reshapeNodeCoords();
  }
  function pickSphere(): void {
    domainKind = "sphere";
    reshapeNodeCoords();
  }
  function onSphereDim(d: number): void {
    sphereDim = d;
    reshapeNodeCoords();
  }

  // ---------- nodes ----------

  interface NodeDraft {
    label: string;
    coords: number[];
    statements: string;
    /** Optional per-node assistant-role substitution.  Empty string =
     *  "use the standard assistant baseline" (the legacy default).
     *  Validated client-side against the same slug regex the engine
     *  uses (`[a-z0-9._-]+`).  Persona manifolds use this — each node's
     *  centroid is pooled under its role's chat-template substitution. */
    role: string;
    expanded: boolean;
  }

  let nodes: NodeDraft[] = $state([]);
  const ROLE_SLUG_RE = /^[a-z0-9._-]+$/;

  /** Resize every node's coord array to the current intrinsic dim,
   *  preserving existing values, padding with zeros. */
  function reshapeNodeCoords(): void {
    const n = intrinsicDim;
    nodes = nodes.map((nd) => {
      const coords = nd.coords.slice(0, n);
      while (coords.length < n) coords.push(0);
      return { ...nd, coords };
    });
  }

  function addNode(): void {
    nodes = [
      ...nodes,
      {
        label: `node_${nodes.length + 1}`,
        coords: new Array(intrinsicDim).fill(0),
        statements: "",
        role: "",
        expanded: true,
      },
    ];
  }

  function removeNode(idx: number): void {
    nodes = nodes.filter((_, i) => i !== idx);
  }

  function setNodeField<K extends keyof NodeDraft>(
    idx: number,
    key: K,
    value: NodeDraft[K],
  ): void {
    nodes = nodes.map((nd, i) => (i === idx ? { ...nd, [key]: value } : nd));
  }

  function setNodeCoord(idx: number, ci: number, value: number): void {
    nodes = nodes.map((nd, i) => {
      if (i !== idx) return nd;
      const coords = nd.coords.slice();
      coords[ci] = value;
      return { ...nd, coords };
    });
  }

  // ---------- validation ----------

  /** Split a node's textarea body into trimmed non-empty statements. */
  function statementsOf(nd: NodeDraft): string[] {
    return nd.statements
      .split(/\r?\n/)
      .map((s) => s.trim())
      .filter(Boolean);
  }

  /** Check one coordinate vector against the domain.  Box: each coord
   *  in [lo, hi] (periodic axes accept anything — they wrap).  Sphere:
   *  no per-coord bound, the domain immerses the chart. */
  function coordsInDomain(coords: number[]): boolean {
    if (!coords.every(Number.isFinite)) return false;
    if (domainKind !== "box") return true;
    for (let i = 0; i < boxDim; i++) {
      const a = axisDrafts[i];
      if (a.periodic) continue;
      const v = coords[i];
      if (!Number.isFinite(v)) return false;
      if (v < a.lo || v > a.hi) return false;
    }
    return true;
  }

  const validation = $derived.by<{ ok: boolean; messages: string[] }>(() => {
    const messages: string[] = [];
    if (!slug(identity.name)) {
      messages.push("name required");
    }
    // Domain-shape validation only fires when the user is hand-authoring
    // coordinates.  auto-domain skips the box / sphere picker entirely
    // — the fitter derives the layout per-model.
    if (!autoDomain && domainKind === "box") {
      for (let i = 0; i < boxDim; i++) {
        const a = axisDrafts[i];
        if (a.hi <= a.lo) {
          messages.push(`axis "${a.name || i}": hi > lo`);
        }
      }
    }
    // Min-node count: hand-authored coords need ``2n+1`` for poisedness;
    // auto-domain only needs >=2 nodes (shared-structure requirement,
    // matching the auto-generated tab).
    if (autoDomain) {
      if (nodes.length < 2) {
        messages.push(`nodes: ${nodes.length} / 2`);
      }
    } else if (nodes.length < minNodes) {
      messages.push(`nodes: ${nodes.length} / ${minNodes}`);
    }
    const seenLabels = new Set<string>();
    for (const nd of nodes) {
      const lbl = slug(nd.label);
      if (!lbl) {
        messages.push("node label required");
      } else if (seenLabels.has(lbl)) {
        messages.push(`duplicate label "${lbl}"`);
      } else {
        seenLabels.add(lbl);
      }
      if (!autoDomain && !coordsInDomain(nd.coords)) {
        messages.push(`"${nd.label}": outside domain`);
      }
      if (statementsOf(nd).length === 0) {
        messages.push(`"${nd.label}": statement required`);
      }
      const r = nd.role.trim();
      if (r && !ROLE_SLUG_RE.test(r)) {
        messages.push(`"${nd.label}": invalid role "${r}"`);
      }
    }
    // auto-domain shares hyperparam validation with the auto-generated tab.
    if (autoDomain) messages.push(...tuningMessages(tuning, maxDimLimit));
    return { ok: messages.length === 0, messages };
  });

  function axisError(index: number): string | null {
    if (!validationAttempted || autoDomain || domainKind !== "box") return null;
    const axis = axisDrafts[index];
    return axis.hi <= axis.lo ? "The high value must be greater than the low value." : null;
  }

  function nodeLabelError(index: number): string | null {
    if (!validationAttempted) return null;
    const formatted = slug(nodes[index].label);
    if (!formatted) return "Enter a label for this node.";
    const matches = nodes.filter((node) => slug(node.label) === formatted).length;
    return matches > 1 ? "Each node label must be unique." : null;
  }

  function nodeCoordinateError(index: number): string | null {
    if (!validationAttempted || autoDomain || coordsInDomain(nodes[index].coords)) return null;
    return "Keep every coordinate inside the selected domain.";
  }

  function nodeStatementError(index: number): string | null {
    if (!validationAttempted || statementsOf(nodes[index]).length > 0) return null;
    return "Add at least one example statement.";
  }

  function nodeRoleError(index: number): string | null {
    if (!validationAttempted) return null;
    const role = nodes[index].role.trim();
    if (role && !ROLE_SLUG_RE.test(role)) {
      return "Use lowercase letters, numbers, dots, underscores, or hyphens.";
    }
    return null;
  }

  function nodeCountError(): string | null {
    if (!validationAttempted) return null;
    const required = autoDomain ? 2 : minNodes;
    return nodes.length < required
      ? `Add ${required - nodes.length} more ${required - nodes.length === 1 ? "node" : "nodes"}.`
      : null;
  }

  function sharedNameInput(): HTMLInputElement | null {
    const body = formRegion?.closest(".mb-form");
    return body?.querySelector<HTMLInputElement>(
      ":scope > .grid2 > .field:nth-child(2) input",
    ) ?? null;
  }

  function tuningInput(label: string): HTMLInputElement | null {
    const fields = formRegion?.querySelectorAll<HTMLElement>(".field") ?? [];
    for (const field of fields) {
      if (field.querySelector<HTMLElement>(".label")?.textContent?.trim() === label) {
        return field.querySelector<HTMLInputElement>("input");
      }
    }
    return null;
  }

  async function focusFirstInvalid(): Promise<void> {
    await tick();
    if (!slug(identity.name)) {
      sharedNameInput()?.focus();
      return;
    }
    if (!autoDomain && domainKind === "box") {
      const invalidAxis = axisDrafts.slice(0, boxDim).findIndex((axis) => axis.hi <= axis.lo);
      if (invalidAxis >= 0) {
        formRegion?.querySelector<HTMLInputElement>(`#authored-axis-${invalidAxis}-hi input`)?.focus();
        return;
      }
    }
    if (nodeCountError()) {
      addNodeButton?.focus();
      return;
    }
    for (let index = 0; index < nodes.length; index += 1) {
      const card = formRegion?.querySelector<HTMLElement>(`[data-node-index="${index}"]`);
      if (nodeLabelError(index)) {
        card?.querySelector<HTMLInputElement>(".node-label")?.focus();
        return;
      }
      if (nodeCoordinateError(index)) {
        card?.querySelector<HTMLInputElement>(".node-coords input")?.focus();
        return;
      }
      const roleError = nodeRoleError(index);
      const statementError = nodeStatementError(index);
      if (roleError || statementError) {
        if (!nodes[index].expanded) {
          setNodeField(index, "expanded", true);
          await tick();
        }
        const expandedCard = formRegion?.querySelector<HTMLElement>(`[data-node-index="${index}"]`);
        if (roleError) expandedCard?.querySelector<HTMLInputElement>(".node-role input")?.focus();
        else expandedCard?.querySelector<HTMLTextAreaElement>(".node-statements")?.focus();
        return;
      }
    }
    const tuningErrors = autoDomain ? tuningMessages(tuning, maxDimLimit) : [];
    if (tuningErrors.length === 0) return;
    advancedOpen = true;
    await tick();
    const label = tuning.maxDim < 1 || maxDimLimit !== null && tuning.maxDim > maxDimLimit
      ? "max dim"
      : "variance";
    tuningInput(label)?.focus();
  }

  // ---------- submit ----------

  async function save(): Promise<void> {
    if (submitting) return;
    validationAttempted = true;
    if (!validation.ok) {
      await focusFirstInvalid();
      return;
    }
    submitting = true;
    const { namespace, name, description } = identitySlugs(identity);
    // auto-domain split: bring-your-own-corpora discover (the fitter
    // derives coords per-model via pca / spectral) routes through
    // createDiscover; the historical authored path with hand-placed
    // coords keeps using create.
    if (autoDomain) {
      const req: CreateDiscoverManifoldRequest = {
        namespace,
        name,
        description,
        fit_mode: tuning.fitMode,
        hyperparams: tuningHyperparams(tuning),
        nodes: nodes.map((nd) => {
          const r = nd.role.trim();
          return {
            label: slug(nd.label),
            statements: statementsOf(nd),
            ...(r ? { role: r } : {}),
          };
        }),
      };
      try {
        await apiManifolds.createDiscover(req);
        await refreshManifoldList();
        pushToast(
          `Created ${namespace}/${name} (auto-domain, ${tuning.fitMode} fit). Open Manifolds to fit it.`,
          { kind: "info" },
        );
        if (oncomplete) oncomplete();
        else { closeDrawer(); openDrawer("manifolds"); }
      } catch (e) {
        pushToast(`Couldn't create the manifold: ${errorText(e)}`, {
          kind: "error",
          ttlMs: null,
        });
      } finally {
        submitting = false;
      }
      return;
    }
    const req: CreateManifoldRequest = {
      namespace,
      name,
      description,
      domain: buildDomain(),
      nodes: nodes.map((nd) => {
        const r = nd.role.trim();
        return {
          label: slug(nd.label),
          coords: nd.coords.slice(0, intrinsicDim),
          statements: statementsOf(nd),
          ...(r ? { role: r } : {}),
        };
      }),
    };
    try {
      const r = await apiManifolds.create(req);
      await refreshManifoldList();
      const advisories = r.advisories ?? [];
      if (advisories.length > 0) {
        pushToast(
          `Created ${namespace}/${name}. Check ${advisories.length} coordinate ${advisories.length === 1 ? "warning" : "warnings"}.`,
          { kind: "warning", detail: advisories.join("; "), ttlMs: 10000 },
        );
      } else {
        pushToast(`built manifold ${namespace}/${name}`, { kind: "info" });
      }
      if (oncomplete) oncomplete();
      else { closeDrawer(); openDrawer("manifolds"); }
    } catch (e) {
      pushToast(`Couldn't create the manifold: ${errorText(e)}`, {
        kind: "error",
        ttlMs: null,
      });
    } finally {
      submitting = false;
    }
  }

  function errorText(e: unknown): string {
    return userFacingError(e, "Unable to build this direction. Check the highlighted fields and try again.");
  }

  onInterfaceMount(() => registerInterfaceController("manifold_authored", {
    schema: authoredDraftSchema,
    read: () => ({ busy: submitting, values: { auto_domain: autoDomain, domain_kind: domainKind, box_dim: boxDim, sphere_dim: sphereDim, axes: axisDrafts, nodes, tuning, advanced_open: advancedOpen }, validation }),
    update: async (change) => {
      if (submitting) throw new ToolError("BUSY", "Wait for this interface operation to finish.");
      if (browserMode && ["klein", "projective"].includes(change.domain_kind as string)) throw new ToolError("UNAVAILABLE", "This domain is supported by the HTTP runtime only.");
      if (change.auto_domain !== undefined) autoDomain = change.auto_domain as boolean;
      if (change.domain_kind !== undefined) domainKind = change.domain_kind as DomainKind;
      if (change.box_dim !== undefined) boxDim = change.box_dim as number;
      if (change.sphere_dim !== undefined) sphereDim = change.sphere_dim as number;
      if (change.axes !== undefined) axisDrafts = structuredClone(change.axes) as AxisDraft[];
      if (change.nodes !== undefined) nodes = structuredClone(change.nodes) as NodeDraft[];
      if (change.tuning) Object.assign(tuning, change.tuning);
      if (change.advanced_open !== undefined) advancedOpen = change.advanced_open as boolean;
      if (change.nodes === undefined && ["domain_kind", "box_dim", "sphere_dim"].some(key => change[key] !== undefined)) reshapeNodeCoords();
    },
  }));
</script>

<div class="form-stack" bind:this={formRegion}>
  <!-- auto-domain switch: when on, skip the box/sphere picker and the
       per-node coord inputs; the fitter derives the layout per-model
       via pca / spectral.  When off, hand-author coords as before. -->
  <span class="auto-domain-toggle">
    <Checkbox bind:checked={autoDomain} label="auto-domain" />
  </span>

  {#if autoDomain}
    <!-- fit-method picker — mirrors the auto-generated tab's choice. -->
    <FitMethodPicker
      {tuning}
      linearOnly={browserMode}
      spectralNote="curved · best with ≥50 nodes"
      onchange={(fitMode) => (tuning.fitMode = fitMode)}
    />
  {:else}
    <section class="step">
      <h2 class="step-title">domain</h2>
      <div class="domain-kind" role="group" aria-label="Domain shape">
        <button
          type="button"
          class="kind-btn"
          class:active={domainKind === "box" && boxDim === 1}
          aria-pressed={domainKind === "box" && boxDim === 1}
          onclick={() => pickBoxDim(1)}
        >box 1D</button>
        <button
          type="button"
          class="kind-btn"
          class:active={domainKind === "box" && boxDim === 2}
          aria-pressed={domainKind === "box" && boxDim === 2}
          onclick={() => pickBoxDim(2)}
        >box 2D</button>
        <button
          type="button"
          class="kind-btn"
          class:active={domainKind === "box" && boxDim === 3}
          aria-pressed={domainKind === "box" && boxDim === 3}
          onclick={() => pickBoxDim(3)}
        >box 3D</button>
        <button
          type="button"
          class="kind-btn"
          class:active={domainKind === "sphere"}
          aria-pressed={domainKind === "sphere"}
          onclick={pickSphere}
        >sphere</button>
        {#if !browserMode}
          <button type="button" class="kind-btn" class:active={domainKind === "klein"}
            aria-pressed={domainKind === "klein"} onclick={() => { domainKind = "klein"; reshapeNodeCoords(); }}>Klein bottle</button>
          <button type="button" class="kind-btn" class:active={domainKind === "projective"}
            aria-pressed={domainKind === "projective"} onclick={() => { domainKind = "projective"; reshapeNodeCoords(); }}>Projective plane (RP²)</button>
        {/if}
      </div>

      {#if domainKind === "box"}
        <div class="axes">
          {#each axisDrafts.slice(0, boxDim) as axis, i (i)}
            <div class="axis-card">
              <label class="axis-field name-field">
                <span class="mini-label">axis</span>
                <input
                  type="text"
                  class="input mini"
                  value={axis.name}
                  oninput={(ev) => {
                    axisDrafts[i].name = (ev.currentTarget as HTMLInputElement).value;
                  }}
                  spellcheck="false"
                />
              </label>
              <label class="axis-field">
                <span class="mini-label">lo</span>
                <NumberInput
                  value={axis.lo}
                  step={0.1}
                  oninput={(v) => {
                    if (v !== null) axisDrafts[i].lo = v;
                  }}
                />
              </label>
              <label class="axis-field">
                <span class="mini-label">hi</span>
                <span id={`authored-axis-${i}-hi`}>
                  <NumberInput
                    value={axis.hi}
                    step={0.1}
                    invalid={axisError(i) !== null}
                    ariaLabel={`${axis.name || `Axis ${i + 1}`} high value`}
                    ariaDescribedby={axisError(i) ? `authored-axis-${i}-error` : undefined}
                    oninput={(v) => {
                      if (v !== null) axisDrafts[i].hi = v;
                    }}
                  />
                </span>
              </label>
              <span class="axis-check">
                <Checkbox
                  checked={axis.periodic}
                  label="periodic"
                  onchange={(v) => {
                    axisDrafts[i].periodic = v;
                  }}
                />
              </span>
              {#if axisError(i)}
                <p id={`authored-axis-${i}-error`} class="field-error axis-error">
                  {axisError(i)}
                </p>
              {/if}
            </div>
          {/each}
        </div>
      {:else}
        {#if domainKind === "sphere"}
        <label class="field sphere-field">
          <span class="label">sphere dim</span>
          <Select
            value={sphereDim}
            options={[
              { value: 1, label: "S¹ (circle)" },
              { value: 2, label: "S² (sphere)" },
              { value: 3, label: "S³" },
            ]}
            ariaLabel="Sphere dimension"
            onchange={onSphereDim}
          />
        </label>
        {:else}
          <p class="dim-note">{quotientDescription(buildDomain())} Spread nodes across the whole surface; the minimum count alone does not guarantee a stable fit.</p>
        {/if}
      {/if}
      <p class="dim-note">
        dim <strong><MorphText text={intrinsicDim} /></strong> · min <strong><MorphText text={minNodes} /></strong> nodes
      </p>
    </section>
  {/if}

  <!-- node editor -->
  <section class="step">
    <h2 class="step-title">nodes</h2>
    <p class="dim-note"><MorphText text={`${nodes.length} ${nodes.length === 1 ? "node" : "nodes"} · ${autoDomain ? "coordinates derived at fit" : `${intrinsicDim} dimensions`}`} /></p>
    {#if nodes.length === 0}
      <p class="muted">
        add ≥{autoDomain ? 2 : minNodes} nodes
      </p>
    {/if}
    <div class="node-list">
      {#each nodes as node, idx (idx)}
        <div class="node-card" data-node-index={idx}>
          <div class="node-head">
            <button
              type="button"
              class="node-expand"
              onclick={() => setNodeField(idx, "expanded", !node.expanded)}
              aria-expanded={node.expanded}
              aria-controls={`authored-node-${idx}-details`}
              aria-label={`${node.expanded ? "Collapse" : "Expand"} node ${node.label || idx + 1}`}
            >
              <span class="caret">{node.expanded ? "▾" : "▸"}</span>
            </button>
            <input
              type="text"
              class="input mini node-label"
              value={node.label}
              oninput={(ev) =>
                setNodeField(idx, "label", (ev.currentTarget as HTMLInputElement).value)}
              placeholder="label"
              spellcheck="false"
              aria-label={`Node ${idx + 1} label`}
              aria-invalid={nodeLabelError(idx) !== null}
              aria-describedby={nodeLabelError(idx) ? `authored-node-${idx}-label-error` : undefined}
            />
            {#if !autoDomain}
              <div class="node-coords">
                {#each node.coords as c, ci (ci)}
                  <span class="coord-cell">
                    <NumberInput
                      value={c}
                      step={0.1}
                      ariaLabel={`Node ${node.label || idx + 1}, coordinate ${ci + 1}`}
                      invalid={nodeCoordinateError(idx) !== null}
                      ariaDescribedby={nodeCoordinateError(idx) ? `authored-node-${idx}-coords-error` : undefined}
                      oninput={(v) => setNodeCoord(idx, ci, v ?? 0)}
                    />
                  </span>
                {/each}
              </div>
            {/if}
            <button
              type="button"
              class="node-remove"
              onclick={() => removeNode(idx)}
              aria-label="remove node {node.label}"
              {...{ "aria-description": "remove node" }}
            ><FluentIcon name="dismiss" /></button>
          </div>
          {#if nodeLabelError(idx)}
            <p id={`authored-node-${idx}-label-error`} class="field-error">
              {nodeLabelError(idx)}
            </p>
          {/if}
          {#if nodeCoordinateError(idx)}
            <p id={`authored-node-${idx}-coords-error`} class="field-error">
              {nodeCoordinateError(idx)}
            </p>
          {/if}
          {#if nodeRoleError(idx)}
            <p id={`authored-node-${idx}-role-error`} class="field-error">
              {nodeRoleError(idx)}
            </p>
          {/if}
          {#if nodeStatementError(idx)}
            <p id={`authored-node-${idx}-statements-error`} class="field-error">
              {nodeStatementError(idx)}
            </p>
          {/if}
          {#if node.expanded}
            <div id={`authored-node-${idx}-details`} class="node-details" in:slide={collapseIn()} out:slide={collapseOut()}>
            <label class="node-role">
              <span class="label">role</span>
              <input
                type="text"
                class="input mini"
                value={node.role}
                oninput={(ev) =>
                  setNodeField(
                    idx,
                    "role",
                    (ev.currentTarget as HTMLInputElement).value,
                  )}
                placeholder="pirate"
                autocomplete="off"
                spellcheck="false"
                aria-invalid={nodeRoleError(idx) !== null}
                aria-describedby={nodeRoleError(idx) ? `authored-node-${idx}-role-error` : undefined}
              />
            </label>
            <textarea
              class="node-statements"
              rows="4"
              value={node.statements}
              oninput={(ev) =>
                setNodeField(
                  idx,
                  "statements",
                  (ev.currentTarget as HTMLTextAreaElement).value,
                )}
              placeholder="one statement per line"
              aria-label={`Statements for node ${node.label || idx + 1}`}
              aria-invalid={nodeStatementError(idx) !== null}
              aria-describedby={nodeStatementError(idx) ? `authored-node-${idx}-statements-error` : undefined}
            ></textarea>
            </div>
          {/if}
        </div>
      {/each}
    </div>
    <button bind:this={addNodeButton} type="button" class="add-node" onclick={addNode}>
      + add node
    </button>
    {#if nodeCountError()}
      <p class="field-error">{nodeCountError()}</p>
    {/if}
    <p class="muted" {...{ "aria-description": (PER_NODE_ROLE_HELP) }}>
      Roles are optional. When nodes use different roles, Drowse follows the role of the nearest node.
    </p>
  </section>

  {#if autoDomain}
    <AdvancedSection bind:expanded={advancedOpen}>
      <DiscoverTuningFields {tuning} {maxDimLimit} />
    </AdvancedSection>
  {/if}

  <ValidationBlock
    verb="build"
    messages={validationAttempted ? validation.messages : []}
  />

  <button
    type="button"
    class="save-btn"
    disabled={submitting}
    onclick={save}
  >
    <MorphText text={submitting ? "building…" : autoDomain ? `build · ${tuning.fitMode}` : "build"} numbers={false} />
  </button>
</div>

<style>
  .field-error {
    margin: var(--space-1) 0 0;
    color: var(--accent-red);
    font-size: var(--text-xs);
    line-height: 1.4;
  }

  .axis-error {
    grid-column: 1 / -1;
  }

  input[aria-invalid="true"],
  textarea[aria-invalid="true"] {
    border-color: var(--accent-red);
  }
</style>
