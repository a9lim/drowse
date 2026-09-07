"""Static web UI bundle and FastAPI mount helpers.

The Svelte+Vite source lives at the repo's ``webui/`` directory; the
pre-built bundle lands at ``drowse/web/dist/`` and ships in the wheel
via ``[tool.setuptools.package-data]``.  ``register_web_routes(app)``
mounts ``StaticFiles`` against that bundle and adds an SPA fallback.

Two paths to install / run:

* ``pip install drowse.ai`` is enough: FastAPI + uvicorn are base
  dependencies and the bundle ships in the wheel, so
  ``drowse serve <model>`` exposes the dashboard at ``/`` by default;
  add ``--no-web`` for API-only mode.
* From source: ``cd webui && npm ci && npm run build`` regenerates the
  bundle into ``drowse/web/dist/``.  CI verifies the committed bundle
  matches the source tree.
"""
from __future__ import annotations

from drowse.web.routes import dist_path, register_web_routes

__all__ = ["dist_path", "register_web_routes"]
