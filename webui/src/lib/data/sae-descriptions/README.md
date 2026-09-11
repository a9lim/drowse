# Published SAE descriptions

These indexes reproduce published Neuronpedia explanations for the exact
Gemma Scope 2 dictionaries used by Drowse's browser packs. They are generated
interpretations of features, not guaranteed semantic definitions.

| Model | Layer | Described features / 16,384 |
| --- | ---: | ---: |
| Gemma 3 270M IT | 12 | 15,931 |
| Gemma 3 1B IT | 13 | 15,914 |
| Gemma 3 4B IT | 17 | 15,700 |

The snapshot was retrieved on 2026-09-08 UTC (2026-09-07 in America/Detroit)
from the explanation exports documented at <https://docs.neuronpedia.org/api>.
`provenance.json` records each downloaded batch's URL path, SHA-256 and size,
plus the output hash and exact provider weights hash. No activation examples,
embeddings, or conversations are included.

To refresh from the current publisher exports, run from `webui/`:

```sh
node scripts/build-sae-descriptions.mjs
node scripts/sae-descriptions.test.mjs
npm run build
npm run build:hosted
```

The generator checks the current Neuronpedia model/source and HF dictionary
path against the provider's weights hash. It validates every export row's
model, source and feature id, then selects the newest nonempty explanation
per feature without rewriting its text. Each compact row is
`[description, explanationModelName]`. Missing IDs remain absent and can use
the live API fallback. Never copy explanations across layers or dictionaries
based on a shared feature id.
