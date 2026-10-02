"""`worldgen serve`: the generator behind a local web page.

The standard library's threading HTTP server, not a framework: the page is one user on
one machine, the API is a dozen routes, and a dependency the rest of the project does not
need would be the largest thing in it. Progress goes out as Server-Sent Events, which a
plain `EventSource` in the browser reads without a library either.

Every route is under `/api/`; everything else is a file from `static/`.
"""

import base64
import binascii
import hmac
import json
import mimetypes
import re
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yaml

from ..core.config import WorldConfig
from . import schema, views
from .jobs import Job, JobStore

STATIC = Path(__file__).resolve().parent / "static"

# The page's files, by the name a request asks for them by. Fixed when the module loads:
# the page is three files and does not change while the server runs.
_STATIC_FILES = {p.name: p for p in STATIC.iterdir() if p.is_file()}

# How long an event stream waits for news before sending a keep-alive, in seconds. Under
# the idle timeout of anything likely to sit between the page and the server.
_HEARTBEAT_S = 15.0

# What the browser's login prompt says. The username is not checked; only the password is.
_REALM = 'Basic realm="worldgen", charset="UTF-8"'

# The largest width or height, in hexes, a browser may ask for. 200x200 is the default map
# and takes minutes on a shared CPU; past it a single request could hold the one worker for
# as long as anyone liked. Generation runs on one core, so more hardware would not help.
DEFAULT_MAX_SIZE = 200

_JOB_ROUTE = re.compile(r"^/api/jobs/([\w-]+)(?:/([\w.]+))?$")


def _read_presets(directory: Path) -> list[dict]:
    """The presets `worldgen presets` lists: JSON files of overrides in ./presets."""
    out = []
    for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            out.append({"name": path.stem, "config": data})
    return out


def parse_config_text(text: str) -> dict:
    """A pasted or uploaded config file, YAML or JSON, as the form's full set of values.

    The path fields are dropped rather than refused: a user's own worldgen.yaml may well
    set them, and the rest of the file is still worth loading. The response says which.
    """
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"not valid YAML or JSON: {exc}") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("a config file must be a mapping of setting: value")
    data.pop("export", None)
    ignored = sorted(set(data) & schema.HIDDEN)
    for name in ignored:
        data.pop(name)
    config = WorldConfig.from_dict(schema.coerce(data))
    values = {k: v for k, v in asdict(config).items() if k not in schema.HIDDEN}
    return {"config": values, "ignored": ignored}


def password_matches(header: str | None, password: str) -> bool:
    """Whether an `Authorization: Basic ...` header carries *password*, any username.

    Compared in constant time, so the response time does not reveal how much of a guess
    was right.
    """
    if not header or not header.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(header[6:], validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return False
    _, _, given = decoded.partition(":")
    return hmac.compare_digest(given.encode("utf-8"), password.encode("utf-8"))


def check_size(config: WorldConfig, max_size: int) -> None:
    """Refuse a map wider or taller than *max_size* hexes, naming the side that is."""
    for side in ("width", "height"):
        value = getattr(config, side)
        if not 1 <= value <= max_size:
            raise ValueError(f"{side} must be between 1 and {max_size} hexes, got {value}")


def make_handler(
    store: JobStore,
    presets_dir: Path,
    password: str | None = None,
    max_size: int = DEFAULT_MAX_SIZE,
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "worldgen"

        def _authorised(self) -> bool:
            """Every route, the page included, sits behind the password when there is one.

            HTTP Basic rather than a login page: the browser asks once and then sends the
            password itself on every request, including the map images, the downloads and
            the progress stream, none of which could carry a session token without code.
            """
            if password is None or password_matches(self.headers.get("Authorization"), password):
                return True
            body = json.dumps({"error": "password required"}).encode("utf-8")
            self._send(HTTPStatus.UNAUTHORIZED, body, "application/json", WWW_Authenticate=_REALM)
            return False

        def log_message(self, format, *args):
            # Quiet: every map tile and progress poll would scroll the URL out of sight.
            pass

        # ---- responses ---------------------------------------------------------------

        def _send(self, status: int, body: bytes, content_type: str, **headers) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for name, value in headers.items():
                self.send_header(name.replace("_", "-"), value)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, status: int = HTTPStatus.OK, **headers) -> None:
            body = json.dumps(data).encode("utf-8")
            self._send(status, body, "application/json", **headers)

        def _error(self, status: int, message: str) -> None:
            self._json({"error": message}, status)

        def _download(self, name: str) -> dict:
            return {"Content_Disposition": f'attachment; filename="{name}"'}

        # ---- routing -----------------------------------------------------------------

        def do_GET(self):
            url = urlparse(self.path)
            # Before the password, so a host's health check needs no secret. It says only
            # that the process is up.
            if url.path == "/health":
                return self._json({"ok": True})
            if not self._authorised():
                return
            query = {k: v[-1] for k, v in parse_qs(url.query).items()}
            if url.path == "/api/schema":
                return self._json(
                    {"sections": schema.config_schema(), "limits": {"max_size": max_size}}
                )
            if url.path == "/api/presets":
                return self._json({"presets": _read_presets(presets_dir)})
            if match := _JOB_ROUTE.match(url.path):
                job = store.get(match.group(1))
                if job is None:
                    return self._error(HTTPStatus.NOT_FOUND, "no such world; it may have expired")
                return self._job_get(job, match.group(2), query)
            if url.path.startswith("/api/"):
                return self._error(HTTPStatus.NOT_FOUND, f"no route {url.path}")
            return self._static(url.path)

        def do_POST(self):
            if not self._authorised():
                return
            url = urlparse(self.path)
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length)
            if url.path == "/api/config/parse":
                try:
                    return self._json(parse_config_text(raw.decode("utf-8")))
                except (ValueError, UnicodeDecodeError) as exc:
                    return self._error(HTTPStatus.BAD_REQUEST, str(exc))
            if url.path == "/api/generate":
                return self._generate(raw)
            return self._error(HTTPStatus.NOT_FOUND, f"no route {url.path}")

        def _generate(self, raw: bytes) -> None:
            try:
                body = json.loads(raw or b"{}")
                seed = body.get("seed", 42)
                if isinstance(seed, bool) or not isinstance(seed, int):
                    raise ValueError(f"seed must be a whole number, got {seed!r}")
                config = WorldConfig.from_dict(schema.coerce(body.get("config", {})))
                check_size(config, max_size)
            except (ValueError, TypeError, AttributeError) as exc:
                return self._error(HTTPStatus.BAD_REQUEST, str(exc))
            job = store.submit(seed, config)
            return self._json(job.summary(), HTTPStatus.ACCEPTED)

        def _job_get(self, job: Job, action: str | None, query: dict) -> None:
            if action is None:
                return self._json(job.summary())
            if action == "events":
                return self._events(job)
            if job.state is None:
                return self._error(HTTPStatus.CONFLICT, "the world is not finished")
            state = job.state
            stem = f"world-{job.seed}"
            if action in ("map.svg", "map.png", "hex"):
                try:
                    view = views.resolve(job.config.model, query.get("view", "atlas"))
                except KeyError:
                    return self._error(HTTPStatus.BAD_REQUEST, "unknown view")
            if action == "map.svg":
                if view not in job.svgs:
                    job.svgs[view] = views.render_svg(state, view)
                headers = self._download(f"{stem}-{view}.svg") if "download" in query else {}
                body = job.svgs[view].encode("utf-8")
                return self._send(HTTPStatus.OK, body, "image/svg+xml", **headers)
            if action == "map.png":
                if view not in views.EXPORT_STYLES:
                    return self._error(HTTPStatus.BAD_REQUEST, "only a map style draws as PNG")
                body = views.render_png(state, view)
                return self._send(
                    HTTPStatus.OK, body, "image/png", **self._download(f"{stem}-{view}.png")
                )
            if action == "world.json":
                body = json.dumps(state.to_dict()).encode("utf-8")
                return self._send(
                    HTTPStatus.OK, body, "application/json", **self._download(f"{stem}.json")
                )
            if action == "config.json":
                body = json.dumps(asdict(job.config), indent=2).encode("utf-8")
                return self._send(
                    HTTPStatus.OK,
                    body,
                    "application/json",
                    **self._download(f"{stem}-config.json"),
                )
            if action == "hex":
                try:
                    x, y = float(query["x"]), float(query["y"])
                except (KeyError, ValueError):
                    return self._error(HTTPStatus.BAD_REQUEST, "hex needs numeric x and y")
                found = views.hex_at(state, view, x, y)
                if found is None:
                    return self._json({"hex": None})
                return self._json(
                    {
                        "hex": views.describe(state, found),
                        "outline": views.hex_outline(state, view, found),
                    }
                )
            return self._error(HTTPStatus.NOT_FOUND, f"no action {action!r}")

        def _events(self, job: Job) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            seen = 0
            try:
                while True:
                    fresh = job.wait(seen, _HEARTBEAT_S)
                    if not fresh:
                        self.wfile.write(b": keep-alive\n\n")
                    for event in fresh:
                        self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                    self.wfile.flush()
                    seen += len(fresh)
                    if job.finished and seen >= len(job.events):
                        return
            except (BrokenPipeError, ConnectionResetError):
                return

        def _static(self, path: str) -> None:
            # Looked up by name in a fixed list, never joined onto a directory: no request
            # spelling, `..` or otherwise, can name a file that is not in it.
            target = _STATIC_FILES.get("index.html" if path in ("", "/") else path.lstrip("/"))
            if target is None:
                return self._error(HTTPStatus.NOT_FOUND, "not found")
            content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            return self._send(HTTPStatus.OK, target.read_bytes(), content_type)

    return Handler


def make_server(
    host: str,
    port: int,
    presets_dir: Path,
    keep: int = 4,
    password: str | None = None,
    max_size: int = DEFAULT_MAX_SIZE,
) -> ThreadingHTTPServer:
    store = JobStore(keep=keep)
    server = ThreadingHTTPServer((host, port), make_handler(store, presets_dir, password, max_size))
    server.daemon_threads = True
    server.store = store  # type: ignore[attr-defined]
    return server
