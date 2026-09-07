#!/usr/bin/env python3
"""Opt-in MLC compiler entrypoint for unverified base-model candidates.

Run in the patched MLC/TVM environment with the usual compile or convert_weight
arguments. This produces local candidates, never signed catalog entries.
"""

import sys

from base_model_adapters import register_candidate_models


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("compile", "convert_weight", "gen_config"):
        raise SystemExit("Use compile, convert_weight, or gen_config with normal MLC arguments")
    hybrid = "--hybrid-state-abi" in sys.argv
    if hybrid:
        sys.argv.remove("--hybrid-state-abi")
    register_candidate_models(include_hybrid=hybrid)
    from mlc_llm.__main__ import main as mlc_main
    mlc_main()


if __name__ == "__main__":
    main()
