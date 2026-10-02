"""The web interface: its config form, its hex lookup, and the server end to end."""

import json
import re
import threading
import urllib.error
import urllib.request
from dataclasses import asdict, fields
from pathlib import Path

import pytest

from worldgen.core.config import WorldConfig
from worldgen.web import schema, views
from worldgen.web.server import make_server, parse_config_text

# --- the form ---------------------------------------------------------------------------


def test_the_form_offers_every_setting_but_the_server_paths_once_each():
    offered = [f["name"] for section in schema.config_schema() for f in section["fields"]]
    declared = {f.name for f in fields(WorldConfig)}
    assert len(offered) == len(set(offered))
    assert set(offered) == declared - schema.HIDDEN


def test_every_setting_in_the_form_is_explained():
    unexplained = [
        f["name"] for s in schema.config_schema() for f in s["fields"] if not f["help"].strip()
    ]
    assert not unexplained


def test_a_default_is_always_one_of_its_choices():
    for section in schema.config_schema():
        for f in section["fields"]:
            if "choices" not in f:
                continue
            chosen = f["default"] if isinstance(f["default"], list) else [f["default"]]
            assert set(chosen) <= set(f["choices"]), f["name"]


def test_form_defaults_are_the_dataclass_defaults():
    defaults = asdict(WorldConfig())
    for section in schema.config_schema():
        for f in section["fields"]:
            expected = defaults[f["name"]]
            assert f["default"] == (list(expected) if isinstance(expected, tuple) else expected)


@pytest.mark.parametrize(
    "overrides",
    [
        {"width": "wide"},
        {"width": 12.5},
        {"width": True},
        {"continent_falloff": 1},
        {"wind_direction": [1.0]},
        {"continent_falloff_edges": ["up"]},
        {"naming_packs": ["no-such-pack"]},
        {"max_elevation_m": None},
    ],
)
def test_a_value_of_the_wrong_kind_is_refused_with_its_name(overrides):
    with pytest.raises(ValueError, match=next(iter(overrides))):
        schema.coerce(overrides)


def test_values_are_typed_to_their_fields():
    got = schema.coerce({"width": 40.0, "max_elevation_m": 900, "mean_temperature_c": None})
    assert got == {"width": 40, "max_elevation_m": 900.0, "mean_temperature_c": None}
    assert isinstance(got["width"], int)
    assert isinstance(got["max_elevation_m"], float)


@pytest.mark.parametrize("name", sorted(schema.HIDDEN))
def test_a_server_path_cannot_be_set_from_the_browser(name):
    with pytest.raises(ValueError, match="cannot be set"):
        schema.coerce({name: "/etc/passwd"})


def test_the_shipped_config_imports_as_the_defaults():
    text = (Path(schema.__file__).parent.parent / "default_config.yaml").read_text()
    parsed = parse_config_text(text)
    defaults = {k: v for k, v in asdict(WorldConfig()).items() if k not in schema.HIDDEN}
    assert json.loads(json.dumps(parsed["config"])) == json.loads(json.dumps(defaults))


def test_an_imported_config_drops_server_paths_and_says_so():
    parsed = parse_config_text("width: 30\nheightmap_path: /tmp/x.png\n")
    assert parsed["config"]["width"] == 30
    assert parsed["ignored"] == ["heightmap_path"]
    assert "heightmap_path" not in parsed["config"]


# --- the hex under a click --------------------------------------------------------------


def _polygon_centres(svg: str) -> list[tuple[float, float]]:
    centres = []
    for points in re.findall(r'<polygon points="([^"]+)"', svg):
        pts = [tuple(map(float, p.split(","))) for p in points.split()]
        if len(pts) == 6:
            centres.append((sum(p[0] for p in pts) / 6, sum(p[1] for p in pts) / 6))
    return centres


@pytest.mark.parametrize("view", ["atlas", "wargame", "elevation", "biome"])
def test_the_lookup_finds_the_hex_the_renderer_drew_there(small_state, view):
    """The first hexes each renderer draws are the world's first hexes, in order; a click
    on the centre of each, and just inside its edge, must come back as that hex."""
    drawn = _polygon_centres(views.render_svg(small_state, view))
    for h, (cx, cy) in zip(list(small_state.hexes.values())[:40], drawn, strict=False):
        assert views.hex_outline(small_state, view, h)["x"] == pytest.approx(cx, abs=0.01)
        assert views.hex_outline(small_state, view, h)["y"] == pytest.approx(cy, abs=0.01)
        size = views.hex_outline(small_state, view, h)["size"]
        for dx, dy in [(0, 0), (0.7 * size, 0), (0, -0.7 * size), (-0.5 * size, 0.4 * size)]:
            assert views.hex_at(small_state, view, cx + dx, cy + dy) is h


def test_a_click_off_the_grid_finds_nothing(small_state):
    assert views.hex_at(small_state, "atlas", -1000, -1000) is None


def test_a_hex_is_described_in_plain_json(small_state):
    settled = next(h for h in small_state.hexes.values() if h.settlement is not None)
    described = views.describe(small_state, settled)
    json.dumps(described)
    assert described["settlement"]["name"] == settled.settlement.name
    assert described["coord"] == list(settled.coord)


def test_an_unknown_view_is_refused(small_state):
    with pytest.raises(KeyError):
        views.render_svg(small_state, "no-such-plate")
    with pytest.raises(KeyError):
        views.render_png(small_state, "elevation")


# --- the server, end to end -------------------------------------------------------------


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    presets = tmp_path_factory.mktemp("presets")
    (presets / "tiny.json").write_text(json.dumps({"width": 16, "height": 16}))
    (presets / "broken.json").write_text("{not json")
    srv = make_server("127.0.0.1", 0, presets)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.store.shutdown()
    srv.server_close()


def _get(url: str) -> tuple[int, bytes, dict]:
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def _post(url: str, body) -> tuple[int, dict]:
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@pytest.fixture(scope="module")
def finished(server):
    """A 16x16 world generated through the API, with the events that reported it."""
    status, job = _post(
        f"{server}/api/generate", {"seed": 3, "config": {"width": 16, "height": 16}}
    )
    assert status == 202
    with urllib.request.urlopen(f"{server}/api/jobs/{job['id']}/events", timeout=300) as r:
        stream = r.read().decode()
    events = [json.loads(line[6:]) for line in stream.splitlines() if line.startswith("data: ")]
    return job["id"], events


def test_the_page_and_its_files_are_served(server):
    for path, kind in [("/", "text/html"), ("/app.js", "javascript"), ("/style.css", "css")]:
        status, _, headers = _get(server + path)
        assert status == 200 and kind in headers["Content-Type"]


@pytest.mark.parametrize("path", ["/../cli.py", "/%2e%2e/cli.py", "/%2e%2e/%2e%2e/pyproject.toml"])
def test_nothing_outside_the_static_folder_is_served(server, path):
    assert _get(server + path)[0] == 404


def test_presets_are_listed_and_a_broken_one_is_skipped(server):
    _, body, _ = _get(f"{server}/api/presets")
    assert [p["name"] for p in json.loads(body)["presets"]] == ["tiny"]


@pytest.mark.parametrize(
    "body",
    [
        {"seed": "seven"},
        {"seed": 1, "config": {"width": "wide"}},
        {"seed": 1, "config": {"widht": 16}},
        {"seed": 1, "config": {"heightmap_path": "/etc/passwd"}},
        {"seed": 1, "config": {"model": "baroque"}},
    ],
)
def test_a_bad_request_is_refused_before_anything_runs(server, body):
    status, reply = _post(f"{server}/api/generate", body)
    assert status == 400 and reply["error"]


def test_every_stage_is_reported_starting_and_finishing(finished):
    _, events = finished
    assert events[0]["type"] == "queued"
    assert events[-1]["type"] == "done"
    stages = [e for e in events if e["type"] == "stage"]
    total = stages[0]["total"]
    assert len(stages) == 2 * total
    for start, end in zip(stages[::2], stages[1::2], strict=True):
        assert start["elapsed"] is None and end["elapsed"] >= 0
        assert start["index"] == end["index"]
    assert [s["index"] for s in stages[::2]] == list(range(1, total + 1))


def test_a_late_listener_still_hears_the_whole_run(server, finished):
    job_id, events = finished
    _, stream, _ = _get(f"{server}/api/jobs/{job_id}/events")
    replay = [json.loads(line[6:]) for line in stream.decode().splitlines() if line[:6] == "data: "]
    assert [e["type"] for e in replay] == [e["type"] for e in events]


def test_the_finished_world_can_be_viewed_inspected_and_downloaded(server, finished):
    job_id, _ = finished
    base = f"{server}/api/jobs/{job_id}"
    _, body, _ = _get(base)
    summary = json.loads(body)
    assert summary["finished"] and summary["error"] is None
    assert summary["config"]["width"] == 16

    for view in summary["views"]:
        status, svg, headers = _get(f"{base}/map.svg?view={view}")
        assert status == 200 and headers["Content-Type"] == "image/svg+xml", view
        assert svg.startswith(b"<svg")

    status, png, headers = _get(f"{base}/map.png?view=atlas")
    assert status == 200 and png[:8] == b"\x89PNG\r\n\x1a\n"
    assert "attachment" in headers["Content-Disposition"]

    status, world, _ = _get(f"{base}/world.json")
    assert status == 200 and json.loads(world)["seed"] == 3

    status, body, _ = _get(f"{base}/hex?view=atlas&x=40&y=40")
    found = json.loads(body)
    assert status == 200 and found["hex"] is not None
    status, body, _ = _get(
        f"{base}/hex?view=atlas&x={found['outline']['x']}&y={found['outline']['y']}"
    )
    assert json.loads(body)["hex"]["coord"] == found["hex"]["coord"]


def test_an_unknown_world_or_view_is_an_error_not_a_crash(server, finished):
    job_id, _ = finished
    assert _get(f"{server}/api/jobs/nope")[0] == 404
    assert _get(f"{server}/api/jobs/{job_id}/map.svg?view=nope")[0] == 400
    assert _get(f"{server}/api/jobs/{job_id}/map.png?view=elevation")[0] == 400
    assert _get(f"{server}/api/jobs/{job_id}/hex?x=a")[0] == 400


def test_the_same_seed_through_the_server_is_the_same_world(server, finished):
    job_id, _ = finished
    _, again = _post(f"{server}/api/generate", {"seed": 3, "config": {"width": 16, "height": 16}})
    with urllib.request.urlopen(f"{server}/api/jobs/{again['id']}/events", timeout=300) as r:
        r.read()
    first = json.loads(_get(f"{server}/api/jobs/{job_id}/world.json")[1])
    second = json.loads(_get(f"{server}/api/jobs/{again['id']}/world.json")[1])
    assert first["hexes"] == second["hexes"]
    assert first["settlements"] == second["settlements"]


# --- the password -----------------------------------------------------------------------


def _basic(user: str, password: str) -> str:
    import base64

    return "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()


@pytest.mark.parametrize(
    ("header", "ok"),
    [
        (None, False),
        ("", False),
        ("Bearer sesame", False),
        ("Basic !!!not-base64", False),
        (_basic("anyone", "wrong"), False),
        (_basic("anyone", "open sesame"), True),
        (_basic("", "open sesame"), True),
        # A colon belongs to the password once the username has been split off.
        (_basic("a", "open sesame:extra"), False),
    ],
)
def test_only_the_right_password_is_accepted(header, ok):
    from worldgen.web.server import password_matches

    assert password_matches(header, "open sesame") is ok


@pytest.fixture(scope="module")
def locked(tmp_path_factory):
    srv = make_server("127.0.0.1", 0, tmp_path_factory.mktemp("p"), password="open sesame")
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.store.shutdown()
    srv.server_close()


def _request(url: str, auth: str | None = None, data: bytes | None = None) -> tuple[int, dict]:
    request = urllib.request.Request(url, data=data)
    if auth is not None:
        request.add_header("Authorization", auth)
    try:
        with urllib.request.urlopen(request, timeout=60) as r:
            return r.status, dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers)


@pytest.mark.parametrize("path", ["/", "/app.js", "/api/schema", "/api/presets", "/api/jobs/1-1"])
def test_every_page_asks_for_the_password(locked, path):
    status, headers = _request(locked + path)
    assert status == 401
    assert headers["WWW-Authenticate"].startswith("Basic ")
    assert _request(locked + path, _basic("x", "wrong"))[0] == 401


def test_generating_needs_the_password_too(locked):
    body = json.dumps({"seed": 1, "config": {"width": 8, "height": 8}}).encode()
    assert _request(f"{locked}/api/generate", data=body)[0] == 401


def test_the_right_password_opens_everything(locked):
    for path in ["/", "/app.js", "/api/schema"]:
        assert _request(locked + path, _basic("me", "open sesame"))[0] == 200


def test_serving_beyond_this_machine_needs_a_password():
    from click.testing import CliRunner

    from worldgen.cli import cli

    result = CliRunner().invoke(
        cli, ["serve", "--host", "0.0.0.0", "--no-open"], env={"WORLDGEN_PASSWORD": ""}
    )
    assert result.exit_code != 0
    assert "WORLDGEN_PASSWORD" in result.output


def test_the_health_check_needs_no_password(locked):
    assert _request(f"{locked}/health")[0] == 200


def test_a_view_name_is_the_servers_own_string():
    requested = "".join(["at", "las"])  # equal to, but not, the module's string
    assert views.resolve("organic", requested) is views.EXPORT_STYLES[0]
    with pytest.raises(KeyError):
        views.resolve("organic", "atlas\r\nSet-Cookie: x=1")


def test_a_header_cannot_be_smuggled_in_through_the_view(server, finished):
    job_id, _ = finished
    base = f"{server}/api/jobs/{job_id}"
    for action in ["map.svg", "map.png", "hex"]:
        status, _, headers = _get(f"{base}/{action}?view=atlas%0d%0aSet-Cookie:%20x=1&download=1")
        assert status == 400 and "Set-Cookie" not in headers
