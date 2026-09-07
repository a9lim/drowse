<script lang="ts">
  import DrawerCloseButton from "../lib/ui/DrawerCloseButton.svelte";
  import Disclosure from "../lib/Disclosure.svelte";
  import { closeDrawer } from "../lib/stores.svelte";
  import { getRuntimeManifoldFitMaxIntrinsicDim } from "../lib/runtime/registry";

  let _drawerProps: { params?: unknown } = $props();
  $effect(() => {
    void _drawerProps.params;
  });

  // Pre-derive the platform-appropriate modifier label so the shortcut
  // hints don't lie on Linux or Windows.  ``navigator`` may be undefined
  // in non-browser test environments — fall back to ``Cmd`` to match the
  // Mac-first development stance documented in CLAUDE.md.
  const modKey =
    typeof navigator !== "undefined" &&
    /Mac|iPhone|iPad|iPod/.test(navigator.platform)
      ? "Cmd"
      : "Ctrl";
  let shortcutsOpen = $state(false);
  let syntaxOpen = $state(false);
  const manifoldFitMaxDim = getRuntimeManifoldFitMaxIntrinsicDim();
</script>

<section class="drawer-shell" aria-label="Help drawer">
  <header class="header">
    <div>
      <h2 class="title">Help</h2>
    </div>
    <DrawerCloseButton onclick={closeDrawer} />
  </header>

  <!-- svelte-ignore a11y_no_noninteractive_tabindex (keyboard access for overflow content) -->
  <div class="body" role="region" aria-label="Help reference" tabindex="0">
    <section class="intro" aria-labelledby="help-start">
      <h2 id="help-start">What each area does</h2>
      <div class="area-grid">
        <article>
          <h3>Conversation</h3>
          <p>The chat appears here. Color on a model reply can show token surprisal, sampler entropy, or a saved reading.</p>
        </article>
        <article>
          <h3>Controls</h3>
          <p>Response controls shape and measure the next reply. Model controls manage the active model, its tools, and local storage. Chat controls let you name your chat, change its avatar, and save or download a copy.</p>
        </article>
        <article>
          <h3>Loom</h3>
          <p>Weave shows your current text beside its alternative continuations. Map shows where the paths divide; drag to move around or pinch to zoom.</p>
        </article>
        {#if manifoldFitMaxDim !== null}
          <article>
            <h3>Browser fitting</h3>
            <p>Browser fitting supports up to {manifoldFitMaxDim} dimensions. Larger fits need the Python/server runtime.</p>
          </article>
        {/if}
      </div>
    </section>

    <section class="block terms">
      <h2>Common terms</h2>
      <dl>
        <div><dt>Token</dt><dd>One unit of text for the model. It may be a whole word, part of one, or punctuation.</dd></div>
        <div><dt>Token surprisal</dt><dd>How unlikely the chosen token was after sampling settings were applied. It measures a word choice, not whether the reply is true.</dd></div>
        <div><dt>Sampler entropy</dt><dd>How spread out the next-token probabilities were after sampling settings. Low entropy can come from restrictive settings, even when the reply is wrong.</dd></div>
        <div><dt>Direction</dt><dd>An internal pattern used to steer a reply toward or away from a trait.</dd></div>
        <div><dt>Reading</dt><dd>A measurement of an internal pattern. It observes the reply without changing it.</dd></div>
        <div><dt>J-lens</dt><dd>A fitted projection of a layer's activations into token probabilities. Aggregate strength is the mean across fitted layers, not the final sampler probability or a record of the model's thoughts.</dd></div>
        <div><dt>SAE feature</dt><dd>A learned activation pattern. Its label is an interpretation, not proof of a concept or cause. Normalized strength compares activation with a reference maximum and can exceed 1.</dd></div>
        <div><dt>Captured or replayed</dt><dd>Captured readings were recorded during the original run. Replayed readings are computed later; their source and steering are shown beside them. An unsteered replay is a separate measurement, not the original capture.</dd></div>
        <div><dt>Loom</dt><dd>The conversation map, including its alternate continuations.</dd></div>
      </dl>
    </section>

    <Disclosure bind:expanded={shortcutsOpen} summary="Keyboard shortcuts">
      <div class="shortcut-help">
        <section>
          <h3>Conversation</h3>
          <table class="kb">
            <tbody>
              <tr><td><kbd>Esc</kbd></td><td>Stops generation or closes a panel</td></tr>
              <tr><td><kbd>Enter</kbd></td><td>Sends a message</td></tr>
              <tr><td><kbd>Shift</kbd> + <kbd>Enter</kbd></td><td>Starts a new line</td></tr>
              <tr><td><kbd>{modKey}</kbd> + <kbd>Enter</kbd></td><td>Adds text without generating</td></tr>
              <tr><td>Select a token</td><td>Opens its details</td></tr>
            </tbody>
          </table>
        </section>
        <section>
          <h3>Loom</h3>
          <p class="section-copy">These keys apply while the Loom has keyboard focus.</p>
          <table class="kb">
            <tbody>
              <tr><td><kbd>j</kbd> / <kbd>k</kbd></td><td>Moves between alternatives</td></tr>
              <tr><td><kbd>h</kbd> / <kbd>l</kbd></td><td>Moves to an earlier turn or its first continuation</td></tr>
              <tr><td><kbd>Enter</kbd></td><td>Opens the selected branch</td></tr>
              <tr><td><kbd>s</kbd></td><td>Stars the selected turn</td></tr>
              <tr><td><kbd>n</kbd></td><td>Adds a note</td></tr>
              <tr><td><kbd>/</kbd></td><td>Searches the conversation</td></tr>
              <tr><td><kbd>{modKey}</kbd> + <kbd>R</kbd></td><td>Generates another answer</td></tr>
              <tr><td><kbd>{modKey}</kbd> + <kbd>E</kbd></td><td>Edits the selected turn</td></tr>
              <tr><td><kbd>{modKey}</kbd> + <kbd>B</kbd></td><td>Creates an alternate branch</td></tr>
              <tr><td><kbd>{modKey}</kbd> + <kbd>N</kbd></td><td>Opens the map navigator</td></tr>
              <tr><td><kbd>{modKey}</kbd> + <kbd>D</kbd></td><td>Deletes the branch and its continuations</td></tr>
            </tbody>
          </table>
        </section>
      </div>
    </Disclosure>

    <Disclosure bind:expanded={syntaxOpen} summary="Steering expression reference">
      <div class="syntax-help">
        <p>Steering expressions are the text form of response controls. They describe directions and when those directions apply. The controls create them automatically; this section is only a technical reference.</p>
        <h3>Examples</h3>
        <div class="examples">
          <code>0.3 honest</code><span>A light push toward honesty</span>
          <code>0.4 warm@response</code><span>Warmth only during the reply</span>
          <code>0.5 personas%pirate</code><span>The pirate point on a persona manifold</span>
          <code>!sycophantic</code><span>The sycophantic direction removed</span>
          <code>0.3 a + 0.5 b</code><span>Two directions applied together</span>
        </div>
        <h3>Basic grammar</h3>
        <pre class="grammar" tabindex="0" role="region" aria-label="Steering expression grammar">{`expr      := term (("+" | "-") term)*
term      := [coeff "*"?] ["!"] selector ["@" trigger]
selector  := atom (("~" | "|") atom | "%" position)?
position  := signed_num ("," signed_num)* | label
atom      := [ns "/"] NAME ["." NAME] [":" variant] | "sae/" INT
trigger   := before | after | both | thinking | response
             | prompt | generated | when:<probe><op><num>
variant   := raw | sae | sae-<release>
             | role-<name> | from-<source>
`}</pre>
      </div>
    </Disclosure>
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
    padding: var(--drawer-gutter-block) var(--drawer-gutter-inline);
  }
  .title {
    color: var(--fg);
    font-size: var(--text-md);
    font-weight: var(--weight-structure);
  }
  .body {
    flex: 1 1 auto;
    overflow-y: auto;
    padding: var(--drawer-gutter-block) var(--drawer-gutter-inline);
    display: grid;
    grid-auto-rows: max-content;
    align-content: start;
    gap: var(--drawer-section-gap);
    min-height: 0;
  }
  h2 {
    margin: 0 0 var(--space-3);
    color: var(--fg-strong);
    font-family: var(--font-structure);
    font-size: var(--text-md);
    font-weight: var(--weight-structure);
  }
  .intro {
    padding: var(--surface-padding);
    border-radius: var(--radius-lg);
    background: var(--surface-sheen), color-mix(in srgb, var(--accent) 6%, var(--glass));
    box-shadow: var(--shadow-well);
  }
  .area-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: var(--space-3);
  }
  .area-grid article {
    padding: var(--surface-padding);
    border-radius: var(--radius-lg);
    background: var(--surface-sheen), var(--bg-elev);
  }
  .area-grid h3, .shortcut-help h3, .syntax-help h3 {
    margin: 0 0 var(--space-2);
    color: var(--fg);
    font-family: var(--font-structure);
    font-size: var(--text);
    font-weight: var(--weight-structure);
  }
  .area-grid p, .syntax-help p, .section-copy {
    margin: 0;
    color: var(--fg-dim);
    font-family: var(--font-reading);
    line-height: 1.5;
    text-wrap: pretty;
  }
  .kb {
    border-collapse: collapse;
    width: 100%;
    color: var(--fg-strong);
    font-size: var(--text-sm);
  }
  .kb td {
    padding: var(--space-2) var(--space-3);
    vertical-align: top;
  }
  .kb td:last-child {
    font-family: var(--font-reading);
  }
  .kb td:first-child {
    color: var(--fg-dim);
    white-space: nowrap;
    width: 9em;
  }
  kbd {
    background: var(--bg-elev);
    color: var(--fg-strong);
    padding: 0 var(--space-2);
    border-radius: var(--radius);
    font-family: inherit;
    font-size: var(--text-xs);
  }
  .grammar {
    background: var(--bg-deep);
    padding: var(--surface-padding);
    margin: var(--space-3) 0;
    color: var(--fg-strong);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    line-height: 1.4;
    overflow-x: auto;
    white-space: pre;
  }
  .terms dl {
    display: grid;
    gap: var(--space-3);
    margin: 0;
  }
  .terms dl div {
    display: grid;
    grid-template-columns: minmax(8rem, 0.35fr) 1fr;
    gap: var(--space-4);
    padding: var(--surface-padding);
    border-radius: var(--radius-lg);
    background: var(--surface-sheen), var(--glass);
  }
  .terms dt {
    color: var(--fg);
    font-family: var(--font-structure);
    font-weight: var(--weight-structure);
  }
  .terms dd {
    margin: 0;
    color: var(--fg-dim);
    font-family: var(--font-reading);
    line-height: 1.45;
  }
  .syntax-help {
    display: grid;
    gap: var(--space-4);
  }
  .shortcut-help {
    display: grid;
    gap: var(--space-6);
  }
  .examples {
    display: grid;
    grid-template-columns: minmax(12rem, auto) 1fr;
    gap: var(--space-2) var(--space-4);
    align-items: baseline;
  }
  .examples code {
    color: var(--accent-amber);
    font-family: var(--font-data);
  }
  .examples span {
    color: var(--fg-dim);
    font-family: var(--font-reading);
  }
  @media (max-width: 620px) {
    .area-grid { grid-template-columns: minmax(0, 1fr); }
    .terms dl div, .examples { grid-template-columns: minmax(0, 1fr); }
  }
</style>
