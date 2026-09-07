# Hosted instrument release corpora

`prepare_hosted_release_corpus.py` produces the bounded JSON inputs consumed by
the hosted SAE and J-lens builders. It is deliberately offline and fail closed:
the exact source snapshot, pinned license evidence, project word seeds, and all
release tokenizers must match the manifest before it writes anything. The
closure follows the launch-model lock exactly: Gemma 3 270M, Qwen3-1.7B, and
Qwen3-4B. Qwen3-0.6B and SmolLM2 are not launch models. Gemma uses published
provider J-lens and Gemma Scope SAE packs; browser release tooling does not fit
either instrument.

An approved run emits:

- `sae-corpus.json`
- `jlens-corpus.json`
- `jlens-words.json`, restricted to words that round-trip as one token in every
  pinned release tokenizer
- `release-corpus.json`, which records every input and output digest, source and
  license provenance, selection limits, tokenizer-specific token IDs, and the
  generator digest; `release-corpus.schema.json` is its public contract

The generator normalizes Unicode to NFC, collapses whitespace, preserves source
row order, packs only bounded documents, and writes canonical UTF-8 JSON. It
refuses an existing output directory. Parquet input requires `pyarrow`; tokenizer
validation requires `transformers`. Neither dependency may fetch from the Hub:
tokenizers are opened with `local_files_only=True` and `trust_remote_code=False`.

## Current candidates are intentionally blocked

`fineweb-sample-10bt-train.candidate.json` is the preferred technical
candidate. It pins one 574,962,194-byte train shard and the exact dataset card
at FineWeb commit `9bb295ddab0e05d785b879661af7260fed5140fc`. Its
file digest, 281,664-row Parquet metadata, ODC-By declaration, Common Crawl
terms notice, word seeds, and both release tokenizers are closed. The
Parquet reader streams row groups and stops after producing the bounded 320/128
document outputs, so the source is never materialized in memory. This candidate
remains blocked until a named reviewer approves the attribution and
redistribution notice; do not infer legal approval from the dataset card.

`wikitext-103-raw-test.candidate.json` records the exact cached WikiText shard,
but it is not a production manifest. The pinned dataset card's YAML license list
(`cc-by-sa-3.0`, `gfdl`) conflicts with its prose (`CC BY-SA 4.0`), the cached
file is the held-out test split, and no reviewed redistribution notice bundle is
available. The generator exits before touching the inputs while those blockers
remain.

Do not change `status` by itself. A production manifest needs an explicitly
approved non-test snapshot, an exact license-evidence file, a completed
attribution/redistribution notice, a named reviewer and review timestamp, and no
remaining blockers. Legal review is an external release requirement; this tool
enforces its recorded result but does not make it.

## Approved run

```bash
python3 webui/scripts/prepare_hosted_release_corpus.py \
  browser-runtime/release-corpus/approved-source.json \
  /immutable/source/shard.parquet \
  /immutable/source/README.md \
  browser-runtime/release-corpus/jlens-word-seeds.txt \
  /output/polythetic-release-corpus \
  --tokenizer qwen3-4b=/snapshots/qwen3-4b \
  --tokenizer qwen3-1.7b=/snapshots/qwen3-1.7b
```

Pass `sae-corpus.json` to `build-hosted-sae.mjs`. Pass
`jlens-corpus.json` and `jlens-words.json` to `build-hosted-jlens.mjs`.
Preserve `release-corpus.json` with the release evidence and instrument packs.
