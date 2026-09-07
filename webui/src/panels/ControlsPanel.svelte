<script lang="ts">
  import { DRAWERS } from "../drawers";
  import SaveConversationDrawer from "../drawers/SaveConversationDrawer.svelte";
  import { runtimeClient } from "../lib/runtime/client";
  import SegmentedTabs from "../lib/ui/SegmentedTabs.svelte";
  import InspectorPanel from "./InspectorPanel.svelte";
  import { sessionState } from "../lib/stores/session.svelte";

  export type ControlsSection = "response" | "model" | "chat";

  interface Props {
    section: ControlsSection;
  }

  let { section = $bindable() }: Props = $props();

  const generationLabel = $derived(sessionState.info?.is_base_model ? "Completion" : "Response");
  const sections = $derived<{ value: ControlsSection; label: string; title?: string }[]>([
    {
      value: "response",
      label: generationLabel,
      title: `Choose how the next ${generationLabel.toLowerCase()} is generated, shaped, and measured.`,
    },
    {
      value: "model",
      label: "Model",
      title: "Manage the active model, its tools, and local storage.",
    },
    { value: "chat", label: "Chat" },
  ]);
  const modelEntry = DRAWERS[runtimeClient.mode === "browser" ? "local_runtime" : "health"];
</script>

<section class="controls" aria-label={`${generationLabel}, model, and chat controls`}>
  <header class="controls-nav">
    <SegmentedTabs
      items={sections}
      bind:value={section}
      ariaLabel="Controls section"
    />
  </header>

  <div
    class="controls-body"
    class:chat-section={section === "chat"}
    role="tabpanel"
    aria-label={section === "response" ? `${generationLabel} controls` : section === "model" ? "Model controls" : "Chat controls"}
  >
    {#if section === "response"}
      <InspectorPanel />
    {:else if section === "model"}
      <modelEntry.component params={{ embedded: true }} />
    {:else}
      <SaveConversationDrawer embedded />
    {/if}
  </div>
</section>

<style>
  .controls {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    grid-template-rows: auto minmax(0, 1fr);
    height: 100%;
    min-width: 0;
    min-height: 0;
  }

  .controls-nav {
    display: flex;
    min-width: 0;
    align-items: center;
    min-height: calc(var(--control-target) + var(--space-4));
    padding: var(--panel-padding) var(--surface-gutter);
    border-bottom: 1px solid var(--grid-line);
    background: transparent;
  }

  .controls-nav :global(.sk-tabs) {
    width: fit-content;
    min-width: 0;
    padding: 0;
    background: transparent;
  }

  .controls-nav :global(.tab) {
    min-width: 7.5rem;
    border-radius: var(--radius-inset);
  }

  .controls-body {
    min-width: 0;
    min-height: 0;
    overflow: clip;
  }

  .controls-body.chat-section {
    overflow-y: auto;
  }

  @media (max-width: 680px) {
    .controls-nav {
      padding-inline: var(--panel-padding);
    }

    .controls-nav :global(.sk-tabs) {
      width: 100%;
      flex-wrap: nowrap;
    }

    .controls-nav :global(.tab) {
      flex: 1 1 auto;
      min-width: 0;
      padding-inline: var(--space-xs);
    }
  }
</style>
