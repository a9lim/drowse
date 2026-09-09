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
