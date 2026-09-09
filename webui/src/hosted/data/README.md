# Recorded landing preset

`neighbor-example.json` contains seven actual Gemma 3 4B Instruct generations from a locally verified WebGPU model/core-pack closure. It is a curated example, not a multi-prompt or multi-seed evaluation. The model and steering payload hashes, runtime bundle hash, prompt, sampling settings, complete replies, and token-level alternatives are retained in the file.

The probabilities are the runtime's sampling probabilities after temperature, with top-p 1 and no top-k cutoff. They are not raw logits, trait scores, or probabilities of whole sentences. Alternative lists can omit special tokens; the UI groups all unlisted probability mass together and includes the generated token when it is outside the leading alternatives.

To record again, supply a verified closure containing `runtime-lock.json`, `model/hosted-artifacts.json`, and `core/hosted-pack-artifacts.json`, with all referenced payload files present:

```sh
cd webui
node scripts/record-landing-demo.mjs <closure-directory> <recording-output.json>
node scripts/recorded-demo.test.mjs
```

The recorder requires Chrome with hardware WebGPU and verifies local payload sizes and hashes before loading. Review new outputs before replacing the preset. Do not hand-edit generated wording, token identities, or log probabilities. The identity describes the actual local recording closure, not an attestation that it is the currently published hosted model.

# Recorded 404 messages

`not-found-messages.json` contains 37 unique replies from 50 independent, unsteered Gemma 3 4B Instruct generations. The complete recording, including duplicate replies, seeds, sampling settings, token probabilities, alternatives, natural completion receipts, and verified artifact hashes, is served at `/recordings/404-gemma3-4b.json`. Its SHA-256 is pinned in the compact dataset.

The prompt requests one sentence beginning with `Error 404:` and explicitly excludes introductions and lists. Temperature is 1, top-p is 0.9, and the browser runtime's default top-k is 1024 (`top_k: 0` selects that default). Eight alternatives were requested per token; the post-filter distribution can contain fewer. The model uses the verified local `q4f32_1` closure, with no steering or live instruments. Each run uses a distinct recorded seed and the same independent prompt.

Deduplication uses the exact reply after trimming outside whitespace. Each unique message retains the token data from its first occurrence and the indices of every matching run. Wording and token probabilities are never edited or combined across runs. The page samples uniformly over unique messages and avoids an immediate repeat when session storage is available. The recording is static: the 404 page does not load model weights or run inference.

Highlights use the same monotonic surprise ramp as the chat UI. The detail card reports surprisal as `-log2(p)` in bits, from the recorded **post-temperature, post-top-k/top-p sampling probability**. It is not a raw model probability or a whole-sentence probability.

```sh
cd webui
node scripts/record-not-found.mjs <model-directory> <recording-output.json>
node scripts/prepare-not-found.mjs <recording-output.json>
node scripts/not-found-recording.test.mjs
```

The model directory must include `hosted-artifacts.json` and every referenced file. The recorder verifies all file sizes and SHA-256 values before generation. The preparation step requires 50 distinct seeds, natural EOS completion, exact text/token reconstruction, and valid probabilities. The UI recording download preserves all 50 original outputs.
