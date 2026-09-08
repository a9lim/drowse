from typing import cast

from fastapi.testclient import TestClient

from drowse.core.session import DrowseSession
from drowse.server import create_app
from tests.test_server_loom import _StubSession


def test_dashboard_can_submit_feedback_without_broadening_other_connections():
    with TestClient(create_app(cast(DrowseSession, _StubSession()), web=True)) as client:
        response = client.get("/")
    policy = response.headers["content-security-policy"]
    connections = next(directive.strip().split()[1:] for directive in policy.split(";")
                       if directive.strip().startswith("connect-src "))
    assert connections == ["'self'", "https://www.neuronpedia.org", "https://drowse.ai/api/contact"]
    assert "frame-ancestors 'none'" in policy
    assert "form-action 'self'" in policy
