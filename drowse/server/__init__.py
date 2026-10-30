"""drowse HTTP server: OpenAI ``/v1``, Ollama ``/api``, and native ``/drowse/v1`` on one port."""

from drowse.server.app import create_app

__all__ = ["create_app"]
