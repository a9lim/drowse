# Drowse web UI

Svelte 5 + Vite source for the dashboard served by `drowse serve`. Production
builds go directly to `../drowse/web/dist/`; that committed directory is package
data and is the UI users receive from the wheel.

## Development

```bash
cd webui
npm ci

# In another terminal:
drowse serve <model>

npm run dev
```

Vite runs on `http://localhost:5173` and proxies `/drowse`, `/v1`, and `/api`
(including the native WebSocket) to `http://localhost:8000`.

Before committing a UI change:

```bash
npm run check
npm run build
git diff --exit-code ../drowse/web/dist
```

`npm run check` runs Svelte/TypeScript checks and the theme-token validator.
`npm run build` wipes and regenerates the committed production bundle. CI repeats
both commands and fails if the rebuilt bundle differs.

### Test the hosted app on an iPhone

An iPhone must load the LAN build over trusted HTTPS for WebGPU, workers, and
device storage to run in a secure context. Connect the Mac and iPhone to the same
Wi-Fi network, then run:

```bash
npm run dev:hosted:ios
```

The command discovers the Mac's LAN address, creates a local development CA and
a certificate containing the current LAN addresses, and starts the hosted Vite
app on `0.0.0.0:4173`. It prints the exact Safari URL and the path of the public
CA certificate to AirDrop. Private keys remain under the git-ignored
`../browser-runtime/.local-build/https/` directory and are never printed.

On the iPhone:

1. AirDrop the printed `.cer` file to the iPhone and accept it.
2. Open **Settings > General > VPN & Device Management**, select the downloaded
   **Drowse local development CA** profile, and install it.
3. Open **Settings > General > About > Certificate Trust Settings** and enable
   full trust for **Drowse local development CA**.
4. Open the printed `https://<LAN-IP>:4173/app` URL in Safari.
5. To install it as a web app, use **Share > Add to Home Screen > Add**.

Run the command again after the Mac's LAN address changes; the server certificate
is regenerated with the new address. Use `npm run dev:hosted:ios -- --port 4180`
to choose another port, or `--ip <address>` if automatic LAN discovery selects
the wrong interface. Safari 26 or newer is required for WebGPU. Passing the
compatibility screen establishes browser capabilities, but model inference still
needs validation on each physical iPhone model and available-memory class.

For an existing trusted certificate, the hosted Vite config also accepts
`DROWSE_HTTPS_CERT` and `DROWSE_HTTPS_KEY`; set both or neither. HTTPS
does not remove the hosted server's COOP/COEP isolation headers.

#### Physical iPhone release gate

Keep iPhone inference marked as a preview until a release candidate passes this
matrix in Safari on real hardware. Record the iPhone model, iOS build, available
storage, model variant, runtime identity, and context size with the result.

- Start from cleared site data, pass the graphics and OPFS write/read/delete
  checks over trusted HTTPS, and install the site from **Add to Home Screen**.
- Download each offered compact model and its required J-lens; pause, resume,
  reload, and verify hashes without restarting the download.
- Load and generate at the 2,048-token context and 256-token output limits;
  exercise Stop, retry, background/foreground, device loss, and a second-tab
  takeover while generation is active.
- Open a saved chat after killing and relaunching the web app. Confirm model,
  avatar, branches, response settings, and generated text all persist.
- Pan, zoom, branch, and inspect tokens in Loom in portrait, landscape, and with
  the software keyboard open. Enable J-lens live readings only on demand.
- On a compatible Gemma model, install the SAE separately and verify feature
  reads and steering. Models without an SAE must keep that surface unavailable.
- Run one 8 GB-class iPhone expected to load the compact model and one lower-
  memory device expected to fail safely, preserve downloaded data, and require
  an explicit retry rather than entering a reload loop.

## Application shape

The desktop shell has three permanent work areas:

- `LoomSidebar` — conversation tree, filtering, branching, regeneration, and
  node actions.
- `Chat` — role-plan composer, authored/generated turns, token highlights, and
  token-detail entry points.
- `InspectorPanel` — sampling plus the subspace, manifold, SAE, and J-lens
  steering/probe instruments.

The token-detail drawer has geometry, logits, SAE, and J-lens tabs over one
conversation-walking cursor. It uses captured measurement envelopes when they
exist and asks the replay endpoints for historical or newly attached readouts.

Below 1280 px those same areas become explicit `threads`, `chat`, and
`instruments` views. Dense tools open in a focus-trapped drawer. The command
palette is available from **Menu → All tools**; it has no global keyboard shortcut.

## Source map

```text
src/
  main.ts                    app entry point
  App.svelte                 shell, compact navigation, drawer host
  panels/
    Chat.svelte              turn surface and role-plan composer
    RawBuffer.svelte         raw transcript surface
    loom/                    conversation-tree components and controller
    InspectorPanel.svelte    instrument-tab host
    SteeringRack.svelte      subspace/manifold steering cards
    ProbeRack.svelte         attached geometry probes
    SaePanel.svelte          SAE source, steer, and probe surface
    JLensPanel.svelte        J-lens source, steer, and probe surface
    CommandPalette.svelte    global launcher
  drawers/                   authoring, analysis, admin, and export tools
    token/                   four token-detail tabs, cursor, and shared readout UI
  lib/
    api.ts                   typed HTTP, SSE, and WebSocket clients
    types.ts                 shared wire and UI types
    stores.svelte.ts         cross-cutting session/tree/stream state
    stores/                  focused palette, drawer, toast, and input slices
    expression.ts            steering-expression parser/serializer
    manifolds/               diagnostics helpers and renderer
    charts/                  small quantitative primitives
    ui/                      shared buttons, tabs, chips, and drawer chrome
    style/                   tokens, fonts, and global rules
```

The server-owned wire contract is documented beside the implementation in
`../drowse/server/AGENTS.md`; dashboard-specific ownership and interaction
contracts live in `../drowse/web/AGENTS.md`. Prefer those sources over copying a
route inventory into this file.

## State and component rules

- Put state in the smallest matching file under `lib/stores/`. Use
  `stores.svelte.ts` only for state shared across the WebSocket, loom, chat, and
  instruments.
- Svelte collection state uses `SvelteMap`/`SvelteSet`. Replace stored objects
  when mutating them so subscribers observe the change.
- Add shared primitives under `lib/ui/`; do not duplicate button, tab, card, or
  drawer chrome inside a feature.
- Keep `lib/types.ts` aligned with the Pydantic/native WebSocket schemas. New UI
  code reads the canonical `measurements` envelope on each token; the pre-5.x
  top-level `captured` / per-token readout aliases are gone.
- Long-running generate/fit/train work uses one updateable toast or the existing
  progress surface, not a stream of transient notifications.

## Adding a drawer

1. Add `src/drawers/FooDrawer.svelte` and accept `params: unknown` through
   `$props()`.
2. Add its name to `DrawerName` in `lib/types.ts` and to `NARROW_DRAWERS` in
   `App.svelte` if it is a form or picker.
3. Export it from `drawers/index.ts` and add the host branch in `App.svelte`.
4. Add a command-palette entry in `lib/commands.ts` or an inline launcher owned by
   the relevant panel.
5. Run the check/build sequence above and commit the regenerated bundle.

## Visual system

The dashboard ships paired light and dark themes. Hue identifies data space:
subspace/chrome is achromatic, manifold violet, SAE gold, J-lens and surprise
blue, live/positive green, and error/negative red. Roles do not carry hue.
Gradients encode depth or time only when direction is the data; shared chrome
uses depth/focus shadows, not decorative top-light or glow treatments. The
source of truth is `src/lib/style/tokens.css`; `src/lib/style/fonts.css`
self-hosts Wix Madefor for interface and reading text, with Martian Mono for
data.
