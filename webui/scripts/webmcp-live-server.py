"""Serve a local model in an isolated Drowse home for native WebMCP evidence."""

import argparse
import logging
import os
from pathlib import Path
import shutil
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--copy-home", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix="drowse-webmcp-live-server-"))
    os.environ["DROWSE_HOME"] = str(root / "drowse")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.copy_home:
        shutil.copytree(args.copy_home, root / "drowse")
    import torch
    import uvicorn
    from drowse import DrowseSession
    from drowse.server.app import create_app

    torch.set_num_threads(4)
    session = DrowseSession.from_pretrained(
        str(args.model), device=args.device,
        dtype=torch.float32, probes=[], max_tokens=32, compile=False,
        trust_remote_code=False, on_progress=logging.getLogger(__name__).warning,
    )
    try:
        uvicorn.run(create_app(session, api_key="", web=True), host="127.0.0.1", port=args.port, log_level="warning")
    finally:
        session.close()


if __name__ == "__main__":
    main()
