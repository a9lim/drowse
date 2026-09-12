<script lang="ts">
  import { onMount as onInterfaceMount } from "svelte";
  import { registerInterfaceController } from "../lib/workspaceController";
  import { manifoldIdentitySchema } from "../lib/interfaceSchemas";
  import { ToolError } from "../lib/webmcp/types";

  // Manifold authoring shell — reached from "+ build manifold" in the
  // rack drawer.
  //
  // Three disjoint authoring paths share this drawer, and the only thing
  // they genuinely share is the identity (namespace / name / description)
  // and the mode tabs; everything below is per-path, so each is its own
  // sibling under ``drawers/manifold/``:
  //
  //   * auto      — DiscoverForm: hand the model a flat concept list, the
  //                 generator writes per-concept corpora, the fitter
  //                 derives coords per-model.  No coords, no domain.
  //   * template  — TemplatedForm: derive a manifold from a standalone
  //                 template (slot + values + multi-turn contexts).  The
  //                 tool for categories one references rather than
  //                 embodies (days, months, …).  Deterministic, no model.
  //   * custom    — AuthoredForm: user-supplied corpora, with coordinates
  //                 either hand-placed on a picked domain or derived by
  //                 the fitter (the ``auto-domain`` switch).
  //
  // All three build the same on-disk manifold artifact; the path shows up
  // as ``manifold.json::fit_mode``.  Inspector + steering are unchanged
  // from there on.
  //
  // The shared field/step/node styling lives in ``manifold/form.css``,
  // scoped under the ``.mb-form`` class on the body below, so the three
  // forms read as one surface without each carrying a copy.

  import DrawerCloseButton from "../lib/ui/DrawerCloseButton.svelte";
  import { untrack } from "svelte";
  import ModeTabs from "../lib/builder/ModeTabs.svelte";
  import { closeDrawer, openDrawer, steerRack, drawerState } from "../lib/stores.svelte";
  import { manifoldJobs } from "../lib/stores/manifoldJobs.svelte";
  import AuthoredForm from "./manifold/AuthoredForm.svelte";
  import DiscoverForm from "./manifold/DiscoverForm.svelte";
  import TemplatedForm from "./manifold/TemplatedForm.svelte";
  import { identitySlugs, type ManifoldIdentity } from "./manifold/shared";
  import "./manifold/form.css";

  let { params }: { params?: unknown } = $props();
  const returnToToken = $derived((params as { returnToToken?: unknown } | null)?.returnToToken);

  type AuthoringMode = "authored" | "discover" | "templated";
  let authoringMode: AuthoringMode = $state(untrack(() => ["authored", "templated"].includes((params as { mode?: string } | null)?.mode ?? "") ? (params as { mode: AuthoringMode }).mode : "discover"));
  let visited = $state<AuthoringMode[]>([]);
  $effect(() => { if (!visited.includes(authoringMode)) visited = [...visited, authoringMode]; });
  const busy = $derived(manifoldJobs.current !== null && ["running", "waiting", "cancelling"].includes(manifoldJobs.current.status));

  const identity: ManifoldIdentity = $state({
    namespace: "local",
    name: "",
    description: "",
  });

  function cancel(): void {
    if (returnToToken) {
      openDrawer("token_drilldown", returnToToken);
      return;
    }
    closeDrawer();
  }

  function complete(result?: { namespace: string; name: string }): void {
    if (drawerState.open !== "manifold_builder") return;
    const { namespace, name } = result ?? identitySlugs(identity);
    const created = steerRack.catalog.find(row => row.namespace === namespace && row.name === name);
    const mode = created?.resolved_fit_mode ?? created?.fit_mode;
    openDrawer(mode === "spectral" || mode === "authored" ? "manifolds" : "subspace", { returnToToken });
  }

  onInterfaceMount(() => registerInterfaceController("manifold_builder", {
    schema: manifoldIdentitySchema,
    read: () => ({ busy: busy, values: { mode: authoringMode, ...identity } }),
    update: async (change) => {
      if (busy) throw new ToolError("BUSY", "Wait for this interface operation to finish.");
      if (change.mode !== undefined) authoringMode = change.mode as AuthoringMode;
      for (const key of ["namespace", "name", "description"] as const) if (change[key] !== undefined) identity[key] = change[key] as string;
    },
  }));
</script>

<section class="drawer-shell" aria-label="Build manifold">
  <header class="header">
    <h2 class="title">Create a concept or scale</h2>
    <DrawerCloseButton onclick={cancel} />
  </header>

  <div class="body mb-form">
    <ModeTabs
      bind:value={authoringMode}
      tabs={[
        { value: "discover", label: "Generate examples" },
        { value: "templated", label: "Use a template" },
        { value: "authored", label: "Use your examples" },
      ]}
      ariaLabel="Authoring mode"
      disabled={busy}
    />

    <p class="intro">{authoringMode === "discover" ? "Describe at least two contrasting concepts, such as pirate and assistant. Drowse generates examples and learns a direction you can measure or steer."
      : authoringMode === "templated" ? "Use a prompt template for categories such as days, months, or directions. Its possible answers become the examples."
      : "Supply your own labeled examples. Choose coordinates for a scale, or let Drowse find the layout."}</p>

    <!-- identity — shared by all three paths -->
    <div class="grid2">
      <label class="field">
        <span class="label">Folder</span>
        <input
          type="text"
          class="input"
          bind:value={identity.namespace}
          placeholder="local"
          autocomplete="off"
          spellcheck="false"
          disabled={busy}
        />
      </label>
      <label class="field">
        <span class="label">Name (required)</span>
        <input
          type="text"
          class="input"
          bind:value={identity.name}
          required
          placeholder="e.g. pirate"
          autocomplete="off"
          spellcheck="false"
          disabled={busy}
        />
      </label>
    </div>
    <label class="field">
      <span class="label">Description (optional)</span>
      <input
        type="text"
        class="input"
        bind:value={identity.description}
        placeholder="What should this concept or scale measure?"
        autocomplete="off"
        disabled={busy}
      />
    </label>

    {#each visited as mode (mode)}
      <div class="authoring-form" hidden={authoringMode !== mode} inert={authoringMode !== mode}>
        {#if mode === "authored"}<AuthoredForm {identity} oncomplete={complete} />
        {:else if mode === "discover"}<DiscoverForm {identity} oncomplete={complete} />
        {:else}<TemplatedForm {identity} oncomplete={complete} />{/if}
      </div>
    {/each}
  </div>
</section>

<style>
  .drawer-shell {
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 0;
    color: var(--fg);
    font-family: var(--font-ui);
    font-size: var(--text);
  }
  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-sm);
    padding: var(--drawer-gutter-block) var(--drawer-gutter-inline);
  }
  .title {
    margin: 0;
    min-width: 0;
    text-wrap: balance;
    color: var(--accent);
    letter-spacing: 0;
    font-size: var(--text-md);
    font-weight: var(--weight-medium);
  }
  .body {
    flex: 1 1 auto;
    overflow-y: auto;
    padding: var(--drawer-gutter-block) var(--drawer-gutter-inline);
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
    min-height: 0;
  }
  .intro { margin: 0; color: var(--fg-dim); line-height: 1.5; font-size: var(--text-sm); text-wrap: pretty; }
  .authoring-form[hidden] { display: none; }
</style>
