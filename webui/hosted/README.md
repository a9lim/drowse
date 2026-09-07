# Hosted Drowse shell

This Vite root is isolated from the dashboard bundled by `drowse serve`.

```bash
npm ci
npm run check:hosted
npm run test:hosted
npm run preview:hosted
```

Cloudflare Pages preview projects use `webui/` as the project root, `npm ci &&
npm run build:hosted` as the build command, and `dist-hosted` as the output
directory. The root `wrangler.jsonc` carries the equivalent static Pages
configuration.

`public-hosted/_headers` deliberately sends `X-Robots-Tag: noindex, nofollow`
while the hosted runtime remains an access-controlled feasibility preview.
The production project must use `npm ci && npm run build:hosted:release`. That
command fails unless the runtime, distribution, and model backend are verified,
then removes the preview-only noindex policy from the generated output. Source
preview files remain fail-closed.

The generated service worker precaches the shell only. Model weights and
instrument packs belong in verified OPFS storage and must never be added to
Workbox runtime caching.

The hosted application consumes compatible precomputed SAE and J-lens packs
on the GPU for steering, probes, gates, live readout, and token replay. It does
not train SAEs or fit J-lenses. Those authoring workflows remain available only
through the Python runtime; the browser worker rejects either request even if a
client constructs one manually. Browser manifold and template fitting is a
separate supported workflow and does not modify an installed SAE or J-lens.
