<script lang="ts">
  import PageHeader from "./PageHeader.svelte";
  import PageFooter from "./PageFooter.svelte";
  import TabIdentity from "../../lib/ui/TabIdentity.svelte";
  import ContactForm from "../../lib/ui/ContactForm.svelte";
  import { CONTACT_ADDRESS } from "../../lib/contact";
</script>

<TabIdentity title="Contact us · Drowse" />
<svelte:head><meta name="description" content="Get in touch with the Drowse team. Ask for help, share feedback, or discuss working together." /></svelte:head>
<div class="contact-page">
  <a class="skip-link" href="#contact-main">Skip to content</a>
  <PageHeader />
  <main id="contact-main" tabindex="-1">
    <div class="heading">
      <h1>Contact us.</h1>
      <p class="intro">Have a question or an idea for Drowse? Send us a message.</p>
      <div class="email-option"><span>Prefer email?</span><a href={`mailto:${CONTACT_ADDRESS}`}>{CONTACT_ADDRESS}</a></div>
    </div>
    <div class="form-surface"><ContactForm source="contact" /></div>
  </main>
  <PageFooter current="contact" />
</div>

<style>
  .contact-page { min-height: 100svh; display: flex; flex-direction: column; color: var(--fg); background: var(--bg); }
  main { flex: 1; display: grid; grid-template-columns: minmax(0, 0.85fr) minmax(0, 1.15fr); align-items: start; gap: var(--space-2xl); width: min(100%, var(--page-max)); margin-inline: auto; padding: var(--space-2xl) var(--page-gutter) var(--section-space); }
  .heading { display: grid; gap: var(--space-md); padding-block: var(--surface-padding); min-width: 0; }
  h1 { margin: 0; font-family: var(--font-structure); font-weight: var(--weight-display); font-size: var(--text-section-display-compact); line-height: 1.05; letter-spacing: -0.05em; text-wrap: balance; }
  .intro { margin: 0; max-width: 28ch; font-size: var(--text-lg); color: var(--fg-dim); line-height: 1.5; }
  .email-option { display: grid; justify-items: start; gap: var(--space-xs); margin-top: var(--space-md); font-size: var(--text); color: var(--fg-dim); }
  .email-option a { display: inline-flex; align-items: center; min-height: var(--control-target); color: var(--accent); font-family: var(--font-structure); text-underline-offset: 4px; overflow-wrap: anywhere; }
  .email-option a:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 4px; border-radius: var(--radius-sm); }
  .form-surface { min-width: 0; padding: var(--surface-padding); border-radius: var(--popup-radius); background: var(--surface-card); box-shadow: var(--shadow-card); }
  .skip-link { position: fixed; inset: 8px auto auto 8px; z-index: 10; padding: var(--space-sm); background: var(--accent); color: var(--text-on-accent); border-radius: var(--radius); clip-path: inset(50%); transform: translateY(-160%); }
  .skip-link:focus { clip-path: none; transform: none; }
  @media (max-width: 1000px) {
    main { grid-template-columns: minmax(0, 1fr); max-width: 48rem; gap: var(--space-lg); padding-block: var(--space-lg) var(--space-2xl); }
    .heading { padding-block: 0; gap: var(--space-sm); }
    .intro { max-width: 36ch; }
    .email-option { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-xs) var(--space-sm); margin-top: 0; }
  }
  @media (max-width: 480px) {
    .form-surface { padding: var(--space-sm); }
  }
</style>
