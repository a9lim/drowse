<script lang="ts">
  import MorphText from "./MorphText.svelte";
  import { tick } from "svelte";
  import { fly } from "svelte/transition";
  import Select from "../Select.svelte";
  import Button from "./Button.svelte";
  import FluentIcon from "./FluentIcon.svelte";
  import { contentIn, contentOut } from "../motion";
  import { CONTACT_ADDRESS, CONTACT_LIMITS, CONTACT_REASONS, contactTopics, type ContactSource } from "../contact";
  import { contactDrafts, resetContact, submitContact } from "../stores/contact.svelte";

  let { source = "feedback", ondone }: { source?: ContactSource; ondone?: () => void } = $props();
  const uid = $props.id();
  const draft = $derived(contactDrafts[source]);
  const topics = $derived(contactTopics(draft.fields.reason));
  const topicDisabled = $derived(draft.fields.reason === "other");
  const busy = $derived(draft.status === "sending");
  let form = $state<HTMLFormElement>();
  let result = $state<HTMLElement>();
  let copied = $state(false);
  let clearArmed = $state(false);
  const hasDraft = $derived(Boolean(draft.fields.title || draft.fields.body || draft.fields.email));

  $effect(() => {
    if (topics.length && !topics.some(option => option.value === draft.fields.topic)) {
      draft.fields.topic = topics[0].value;
      delete draft.errors.topic;
    }
  });

  async function submit(event: SubmitEvent) {
    event.preventDefault();
    copied = false;
    await submitContact(source);
    await tick();
    if (Object.keys(draft.errors).length) form?.querySelector<HTMLElement>('[aria-invalid="true"]')?.focus();
    else result?.focus({ preventScroll: true });
  }

  async function copyDraft() {
    const text = `${draft.fields.title}\n\n${draft.fields.body}\n\nTopic: ${topics.find(option => option.value === draft.fields.topic)?.label}\nReason: ${CONTACT_REASONS.find(option => option.value === draft.fields.reason)?.label}\nReply email: ${draft.fields.email || "Not provided"}\nReference: ${draft.requestId || "Not submitted"}`;
    try { await navigator.clipboard.writeText(text); copied = true; }
    catch { draft.message = "Clipboard access is unavailable. You can select and copy the text from the fields above."; }
  }

  function beforeLeave(event: BeforeUnloadEvent) {
    if ((hasDraft && draft.status !== "sent") || busy) { event.preventDefault(); event.returnValue = ""; }
  }
</script>

<svelte:window onbeforeunload={beforeLeave} />

{#if draft.status === "sent"}
  <section class="success" bind:this={result} tabindex="-1" aria-labelledby={`${uid}-success`} in:fly={contentIn(6)} out:fly={contentOut()}>
    <div class="success-icon"><FluentIcon name="check" size={24} /></div>
    <div role="status">
      <h2 id={`${uid}-success`}>Message sent</h2>
      <p>Your message is on its way to <a href={`mailto:${CONTACT_ADDRESS}`}>{CONTACT_ADDRESS}</a>.</p>
      <p>{draft.fields.email.trim() ? `We can reply to ${draft.fields.email.trim()}.` : "You didn't leave an email, so we can't reply."}</p>
    </div>
    <details class="reference"><summary>Message reference</summary><code>{draft.requestId}</code></details>
    <div class="actions">
      <Button onclick={() => resetContact(source)}>Send another message</Button>
      {#if ondone}<Button variant="solid" onclick={ondone}>Done</Button>{/if}
    </div>
  </section>
{:else}
  <form class="contact-form" bind:this={form} onsubmit={submit} novalidate aria-label={source === "feedback" ? "Feedback form" : "Contact form"} aria-busy={busy}>
    {#if source === "feedback"}<p class="intro">Tell us what could work better.</p>{/if}
    <div class="categories">
      {#if source === "contact"}
        <div class="field">
          <span class="label">Reason</span>
          <Select bind:value={draft.fields.reason} options={CONTACT_REASONS} ariaLabel="Reason" disabled={busy} />
        </div>
      {/if}
      <div class="field" class:is-disabled={topicDisabled}>
        <span class="label">Topic</span>
        <Select bind:value={draft.fields.topic} options={topics} ariaLabel="Topic" invalid={Boolean(draft.errors.topic)} ariaDescribedby={draft.errors.topic ? `${uid}-topic-error` : undefined} disabled={busy || topicDisabled} />
        {#if draft.errors.topic}<span id={`${uid}-topic-error`} class="error">{draft.errors.topic}</span>{/if}
      </div>
    </div>
    <label class="field">
      <span class="label" id={`${uid}-title-label`}>Title</span>
      <input name="title" bind:value={draft.fields.title} maxlength={CONTACT_LIMITS.title} disabled={busy} required aria-labelledby={`${uid}-title-label`} aria-invalid={Boolean(draft.errors.title)} aria-describedby={draft.errors.title ? `${uid}-title-error` : undefined} placeholder="A short summary" />
      {#if draft.errors.title}<span id={`${uid}-title-error`} class="error">{draft.errors.title}</span>{/if}
    </label>
    <label class="field">
      <span class="label-row"><span class="label" id={`${uid}-body-label`}>Message</span><span class="count" aria-hidden="true"><MorphText text={draft.fields.body.length.toLocaleString("en-US")} /> / <MorphText text={CONTACT_LIMITS.body.toLocaleString("en-US")} /></span></span>
      <textarea name="body" bind:value={draft.fields.body} maxlength={CONTACT_LIMITS.body} disabled={busy} required rows="6" aria-labelledby={`${uid}-body-label`} aria-invalid={Boolean(draft.errors.body)} aria-describedby={draft.errors.body ? `${uid}-body-error ${uid}-body-hint` : `${uid}-body-hint`} onkeydown={(event) => { if (event.key === "Enter" && (event.metaKey || event.ctrlKey) && !event.isComposing) { event.preventDefault(); form?.requestSubmit(); } }} placeholder={draft.fields.reason === "bug" ? "What happened, and what did you expect? Include the steps so we can try it too." : source === "feedback" ? "What happened, or what would you like to see?" : "What would you like to talk about?"}></textarea>
      <span id={`${uid}-body-hint`} class="sr-only">Up to <MorphText text={CONTACT_LIMITS.body.toLocaleString("en-US")} /> characters.</span>
      {#if draft.errors.body}<span id={`${uid}-body-error`} class="error">{draft.errors.body}</span>{/if}
    </label>
    <label class="field">
      <span class="label" id={`${uid}-email-label`}>Email <span class="optional">(optional)</span></span>
      <input type="email" name="email" autocomplete="email" inputmode="email" autocapitalize="none" spellcheck="false" bind:value={draft.fields.email} maxlength={CONTACT_LIMITS.email} disabled={busy} aria-labelledby={`${uid}-email-label`} aria-invalid={Boolean(draft.errors.email)} aria-describedby={`${uid}-email-hint`} placeholder="you@example.com" />
      <span id={`${uid}-email-hint`} class:error={Boolean(draft.errors.email)} class="hint">{draft.errors.email || "Add an email if you'd like a reply."}</span>
    </label>
    <div class="honeypot" aria-hidden="true" inert><label>Website<input name="website" bind:value={draft.fields.website} tabindex="-1" autocomplete="off" /></label></div>
    {#if draft.message}
      <div class="notice" bind:this={result} tabindex="-1" in:fly={contentIn()} out:fly={contentOut()}>
        <p role="alert">{draft.message}</p>
        {#if draft.requestId}<details class="reference"><summary>Message reference</summary><code>{draft.requestId}</code></details>{/if}
        <div class="recovery-actions"><Button onclick={copyDraft}><MorphText text={copied ? "Draft copied" : "Copy draft"} /></Button><a href={`mailto:${CONTACT_ADDRESS}`}>Email us directly</a></div>
        <span class="sr-only" role="status"><MorphText text={copied ? "Draft copied to clipboard." : ""} /></span>
      </div>
    {/if}
    <div class="send-section">
      <p class="privacy">Only the form fields are sent. Chats and model data aren't attached.</p>
      <div class="actions form-actions">
        <div class="secondary-action">{#if hasDraft}<Button variant="flat" disabled={busy} onclick={() => { if (clearArmed) { resetContact(source); clearArmed = false; } else clearArmed = true; }}>{clearArmed ? "Confirm clear draft" : "Clear draft"}</Button>{/if}</div>
        <Button type="submit" variant="solid" disabled={busy} {busy}><FluentIcon name="send" /><MorphText text={busy ? "Sending…" : draft.uncertain ? "Check delivery" : "Send message"} /></Button>
      </div>
      {#if hasDraft || busy}<p class="draft-note" role="status"><MorphText text={busy ? "Sending your message…" : "Your draft stays here while this page is open."} /></p>{/if}
    </div>
  </form>
{/if}

<style>
  .contact-form, .success { display: flex; flex-direction: column; gap: var(--space-md); min-width: 0; }
  p, h2 { margin: 0; }
  .intro, .privacy, .hint, .draft-note, .success p { color: var(--fg-dim); line-height: 1.5; text-wrap: pretty; }
  .intro { font-size: var(--text); }
  .field { display: flex; flex-direction: column; gap: var(--control-label-gap); min-width: 0; }
  .categories { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 14rem), 1fr)); gap: var(--space-sm); }
  .label { font-family: var(--font-structure); font-weight: var(--weight-structure); font-size: var(--text-sm); }
  .field.is-disabled .label { color: var(--fg-muted); }
  .optional { color: var(--fg-dim); font-weight: normal; }
  input, textarea { width: 100%; min-width: 0; min-height: var(--control-field); padding: var(--space-xs) var(--space-sm); border: 1px solid transparent; border-radius: var(--radius); background: var(--input-well); box-shadow: var(--shadow-well); color: var(--fg); font-family: var(--font-reading); font-size: var(--text); font-weight: var(--weight-reading); line-height: 1.5; transition: border-color var(--dur-fast) var(--ease-out), box-shadow var(--dur-fast) var(--ease-out); }
  textarea { resize: vertical; min-height: 10rem; max-height: 24rem; }
  input:focus-visible, textarea:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  input[aria-invalid="true"], textarea[aria-invalid="true"] { border-color: var(--accent-red); }
  input:disabled, textarea:disabled { opacity: 0.7; }
  input::placeholder, textarea::placeholder { color: var(--fg-muted); }
  .hint, .privacy, .draft-note, .reference { font-size: var(--text-xs); }
  .label-row { display: flex; align-items: baseline; justify-content: space-between; gap: var(--space-sm); }
  .count { flex: none; color: var(--fg-muted); font-family: var(--font-data); font-size: var(--text-xs); font-weight: var(--weight-data); font-variant-numeric: tabular-nums; }
  .error { color: var(--accent-red); font-size: var(--text-xs); }
  .notice { padding: var(--surface-padding); border-radius: var(--popup-radius); background: var(--danger-bg); display: grid; gap: var(--space-sm); }
  .notice > p:first-child { color: var(--accent-red); line-height: 1.5; }
  .reference { color: var(--fg-dim); min-width: 0; overflow-wrap: anywhere; }
  .reference summary { width: fit-content; max-width: 100%; min-height: var(--control-target); align-content: center; cursor: pointer; }
  .reference code { display: block; padding-block: var(--space-xs); font-size: inherit; user-select: all; }
  .actions, .recovery-actions { display: flex; flex-wrap: wrap; align-items: center; justify-content: flex-end; gap: var(--space-sm); }
  .recovery-actions { justify-content: flex-start; }
  .send-section { display: grid; gap: var(--space-sm); }
  .form-actions { justify-content: space-between; }
  .secondary-action { margin-inline-end: auto; }
  .draft-note { text-align: end; }
  .recovery-actions a { display: inline-flex; align-items: center; min-height: var(--control-target); }
  a { color: var(--accent); text-underline-offset: 3px; }
  a:focus-visible, summary:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 4px; border-radius: var(--radius-sm); }
  .success { padding-block: var(--space-lg); }
  .success [role="status"] { display: grid; gap: var(--space-sm); }
  .success-icon { display: grid; place-items: center; width: 48px; height: 48px; border-radius: 50%; background: var(--glass); color: var(--accent); }
  .success h2 { font-size: var(--text-lg); text-wrap: balance; }
  .honeypot, .sr-only { position: absolute; width: 1px; height: 1px; margin: -1px; padding: 0; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
  @media (max-width: 760px), (pointer: coarse) { input, textarea { font-size: var(--text-input-touch); } }
  @media (prefers-reduced-motion: reduce) { input, textarea { transition: none; } }
</style>
