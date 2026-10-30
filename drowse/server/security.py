"""Request authentication, origin checks, body bounds, and response headers.

Only the private API prefixes (``/v1/``, ``/api/``, ``/drowse/v1/``) are
authenticated, and only here — before body parsing.  The dashboard shell and
its static assets are public: they hold no data, and the browser cannot send
a bearer header on a page navigation (the key is entered client-side and
rides each API call).  WebSockets authenticate in :func:`ws_auth_ok` before
the handshake is accepted.

Without an API key, the TCP peer must be loopback and the Host header must be
loopback or the ASGI server address (DNS-rebinding defence).  Browser origins
must match the request origin or an explicit ``cors_origins`` entry; wildcard
CORS grants nothing.
"""

from __future__ import annotations

import base64
import hmac
import ipaddress
from urllib.parse import parse_qsl, urlencode, urlsplit

from fastapi import HTTPException, Request, WebSocket
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from drowse.server.errors import PRIVATE_PREFIXES, detail_text, protocol_error

DEFAULT_MAX_REQUEST_BYTES = 64 * 1024 * 1024

_CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; font-src 'self' data:; "
    "connect-src 'self' https://www.neuronpedia.org https://drowse.ai/api/contact; "
    "frame-ancestors 'none'; base-uri 'none'; object-src 'none'; form-action 'self'"
)


def is_loopback_host(host: str) -> bool:
    """True for ``localhost`` and loopback IPs (IPv4-mapped IPv6 included)."""
    if host.lower() == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.is_loopback


def parse_origin(value: str) -> tuple[str, str, int] | None:
    """``scheme://host[:port]`` → ``(scheme, host, port)``; ``None`` if not a bare origin."""
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.path or parsed.query or parsed.fragment
            or any(char.isspace() for char in value) or "\\" in value
        ):
            return None
        port = parsed.port if parsed.port is not None else (443 if parsed.scheme == "https" else 80)
        return parsed.scheme, parsed.hostname.lower(), port
    except ValueError:
        return None


def _local_host_ok(conn: Request | WebSocket) -> bool:
    client = conn.scope.get("client")
    if client:
        try:
            ipaddress.ip_address(client[0])
        except ValueError:
            pass  # In-process ASGI transports can use non-IP client labels.
        else:
            if not is_loopback_host(client[0]):
                return False
    hosts = conn.headers.getlist("host")
    target = parse_origin("http://" + hosts[0]) if len(hosts) == 1 else None
    if target is None:
        return False
    server = conn.scope.get("server")
    return is_loopback_host(target[1]) or bool(server and target[1] == server[0].lower())


def _check_bearer(headers: Headers, expected: str) -> bool:
    """True iff a correct ``Authorization: Bearer <expected>`` header is present."""
    auth = headers.get("authorization")
    if not auth:
        return False
    scheme, _, token = auth.partition(" ")
    return scheme.lower() == "bearer" and hmac.compare_digest(token.encode(), expected.encode())


def _browser_origin_ok(conn: Request | WebSocket) -> bool:
    origins = conn.headers.getlist("origin")
    if not origins:
        return True
    origin = parse_origin(origins[0]) if len(origins) == 1 else None
    hosts = conn.headers.getlist("host")
    scheme = "https" if conn.scope["scheme"] in {"https", "wss"} else "http"
    target = parse_origin(scheme + "://" + hosts[0]) if len(hosts) == 1 else None
    return origin is not None and (origin == target or origin in conn.app.state.ws_origins)


def _authenticate(request: Request) -> None:
    """Raise 401/403 unless ``request`` may call the private API."""
    expected = request.app.state.api_key
    if not expected:
        if not _local_host_ok(request):
            raise HTTPException(403, "Untrusted host; configure an API key for remote access")
        return
    if not _check_bearer(request.headers, expected):
        raise HTTPException(401, "Invalid API key")


def _scrub_query_credentials(scope: Scope) -> None:
    """Drop ``token`` query parameters so the WebSocket access log never records one.

    URL credentials are not accepted, but a stale or third-party client may
    still send ``?token=<key>``; the server's access log reads this scope.
    """
    raw = scope.get("query_string", b"")
    if not raw:
        return
    query = parse_qsl(raw.decode("latin-1"), keep_blank_values=True)
    if any(name == "token" for name, _ in query):
        scope["query_string"] = urlencode([(name, value) for name, value in query if name != "token"]).encode()


def ws_auth_ok(websocket: WebSocket) -> bool:
    """Validate host, browser Origin, and credentials before accepting a WebSocket.

    The credential is an ``Authorization: Bearer`` header or exactly one
    ``drowse.auth.<base64url-utf8-key>`` subprotocol — never a URL parameter
    (one is scrubbed from the scope either way, before anything logs it).
    """
    _scrub_query_credentials(websocket.scope)
    expected = websocket.app.state.api_key
    if not expected and not _local_host_ok(websocket):
        return False
    if not _browser_origin_ok(websocket):
        return False
    if not expected:
        return True
    if _check_bearer(websocket.headers, expected):
        return True
    credentials = [
        protocol.removeprefix("drowse.auth.")
        for protocol in websocket.scope.get("subprotocols", [])
        if protocol.startswith("drowse.auth.")
    ]
    encoded = base64.urlsafe_b64encode(expected.encode()).rstrip(b"=")
    return len(credentials) == 1 and hmac.compare_digest(credentials[0].encode(), encoded)


class HttpSecurityMiddleware:
    """Pure-ASGI gate: auth + origin + body bounds before any handler runs.

    Pure ASGI (not ``BaseHTTPMiddleware``) so streaming cancellation and
    worker ownership pass through untouched.
    """

    def __init__(self, app: ASGIApp, *, max_request_bytes: int) -> None:
        self.app = app
        self.max_request_bytes = max_request_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope)
        path = scope["path"]
        private = path.startswith(PRIVATE_PREFIXES)

        async def secure_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["X-Content-Type-Options"] = "nosniff"
                headers["X-Frame-Options"] = "DENY"
                headers["Referrer-Policy"] = "no-referrer"
                headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=(), payment=(), usb=()"
                headers["Content-Security-Policy"] = _CONTENT_SECURITY_POLICY
                if private:
                    headers["Cache-Control"] = "no-store"
            await send(message)

        try:
            if private:
                _authenticate(request)
            check_origin = private or scope["method"] not in {"GET", "HEAD", "OPTIONS"}
            if check_origin and scope["method"] != "OPTIONS" and not _browser_origin_ok(request):
                raise HTTPException(403, "Untrusted request origin")
            lengths = request.headers.getlist("content-length")
            if lengths:
                if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit():
                    raise HTTPException(400, "Invalid Content-Length")
                length = lengths[0].lstrip("0") or "0"
                if len(length) > len(str(self.max_request_bytes)) or int(length) > self.max_request_bytes:
                    raise HTTPException(413, "Request body too large")
        except HTTPException as exc:
            response = protocol_error(path, exc.status_code, detail_text(exc.detail))
            await response(scope, receive, secure_send)
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_request_bytes:
                    raise HTTPException(413, "Request body too large")
            return message

        await self.app(scope, limited_receive, secure_send)
