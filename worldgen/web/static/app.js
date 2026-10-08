// The worldgen web interface. No framework and no build step: the server hands over the
// config schema, and every control is made from it.
"use strict";

const $ = (id) => document.getElementById(id);

// The settings worth having at hand without opening a section. They also appear in their
// sections; both controls edit the same value.
const BASICS = ["width", "height", "model", "grid_layout", "regional_climate", "max_elevation_m"];
const EXPORT_STYLES = ["atlas", "topographic", "wargame"];
const STORAGE_KEY = "worldgen.form";

const fieldsByName = {};
const defaults = {};
let values = {};
const setters = {}; // name -> [fn(value)], every control showing that field
const rows = {}; // name -> [row elements], to mark the field changed

let maxSize = null; // the server's cap on width and height, in hexes
let job = null;
let events = null;
let timer = null;
let view = "atlas";
const pan = { s: 1, tx: 0, ty: 0 };
let mapSize = null;

// ---- helpers -----------------------------------------------------------------------

const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

function overrides() {
  const out = {};
  for (const name of Object.keys(fieldsByName)) {
    if (!same(values[name], defaults[name])) out[name] = values[name];
  }
  return out;
}

async function api(path, options) {
  const response = await fetch(path, options);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || response.statusText);
  return body;
}

function el(tag, props = {}, children = []) {
  const { dataset, ...rest } = props;
  const node = Object.assign(document.createElement(tag), rest);
  Object.assign(node.dataset, dataset);
  for (const child of [].concat(children)) {
    if (child != null) node.append(child);
  }
  return node;
}

function label(name) {
  return name.replace(/_/g, " ");
}

function save() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ seed: seedValue(), overrides: overrides() }));
  } catch (_) {
    // Private windows and blocked storage: the form simply is not remembered.
  }
}

function restore() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
    if (!saved) return;
    if (Number.isInteger(saved.seed)) $("seed").value = saved.seed;
    for (const [name, value] of Object.entries(saved.overrides || {})) {
      if (name in fieldsByName) values[name] = value;
    }
  } catch (_) {
    // A stale or unreadable entry is ignored rather than trusted.
  }
}

function seedValue() {
  return Math.trunc(Number($("seed").value) || 0);
}

// ---- the form ----------------------------------------------------------------------

function setValue(name, value, source) {
  values[name] = value;
  for (const set of setters[name] || []) if (set !== source) set(value);
  const changed = !same(value, defaults[name]);
  for (const row of rows[name] || []) row.classList.toggle("changed", changed);
  refreshCounts();
  save();
}

function control(field) {
  const { name, kind, choices, nullable } = field;
  let input;
  let set;

  if (kind === "bool") {
    input = el("input", { type: "checkbox", id: `f-${name}-${Math.random()}` });
    input.addEventListener("change", () => setValue(name, input.checked, set));
    set = (v) => (input.checked = !!v);
  } else if (kind === "list") {
    input = el("div", { className: "checks" });
    const boxes = (choices || []).map((choice) => {
      const box = el("input", { type: "checkbox", value: choice });
      box.addEventListener("change", () => {
        // Kept in the order the choices are listed, which for packs is not meaningful
        // but is at least stable.
        const picked = boxes.filter((b) => b.checked).map((b) => b.value);
        setValue(name, picked, set);
      });
      input.append(el("label", {}, [box, choice]));
      return box;
    });
    set = (v) => boxes.forEach((b) => (b.checked = (v || []).includes(b.value)));
  } else if (kind === "pair") {
    const a = el("input", { type: "number", step: "any" });
    const b = el("input", { type: "number", step: "any" });
    const update = () => setValue(name, [Number(a.value) || 0, Number(b.value) || 0], set);
    a.addEventListener("change", update);
    b.addEventListener("change", update);
    input = el("div", { className: "inline" }, [a, b]);
    set = (v) => {
      a.value = v[0];
      b.value = v[1];
    };
  } else if (choices) {
    input = el("select");
    for (const choice of choices) input.append(el("option", { value: choice, textContent: choice || "(none)" }));
    input.addEventListener("change", () => setValue(name, input.value, set));
    set = (v) => (input.value = v);
  } else if (kind === "int" || kind === "float") {
    input = el("input", { type: "number", step: kind === "int" ? "1" : "any" });
    if (nullable) input.placeholder = "auto";
    input.addEventListener("change", () => {
      if (input.value === "") {
        setValue(name, nullable ? null : defaults[name], set);
        if (!nullable) set(defaults[name]);
        return;
      }
      const n = Number(input.value);
      setValue(name, kind === "int" ? Math.round(n) : n, set);
    });
    set = (v) => (input.value = v == null ? "" : v);
  } else {
    input = el("input", { type: "text" });
    input.addEventListener("change", () => setValue(name, input.value, set));
    set = (v) => (input.value = v ?? "");
  }

  (setters[name] ||= []).push(set);
  set(values[name]);
  return input;
}

function fieldRow(field, withHelp) {
  const input = control(field);
  const row = el("div", { className: "row" }, [
    el("label", { textContent: label(field.name), title: field.name }),
    input,
  ]);
  (rows[field.name] ||= []).push(row);
  if (!withHelp || !field.help) return row;
  const help = el("p", { className: "help", textContent: field.help, title: "Click to expand" });
  help.addEventListener("click", () => help.classList.toggle("open"));
  return el("div", { className: "field", dataset: { search: `${field.name} ${field.help}`.toLowerCase() } }, [row, help]);
}

function buildForm(sections) {
  for (const section of sections) {
    for (const field of section.fields) {
      fieldsByName[field.name] = field;
      defaults[field.name] = field.default;
    }
  }
  values = structuredClone(defaults);
  restore();

  for (const name of BASICS) {
    if (fieldsByName[name]) $("basics").append(fieldRow(fieldsByName[name], false));
  }
  for (const section of sections) {
    const count = el("span", { className: "count" });
    const details = el("details", {}, [el("summary", {}, [section.title, count])]);
    details.dataset.fields = section.fields.map((f) => f.name).join(" ");
    details.countNode = count;
    for (const field of section.fields) details.append(fieldRow(field, true));
    $("sections").append(details);
  }
  for (const name of Object.keys(fieldsByName)) setValue(name, values[name]);
}

function refreshCounts() {
  const changed = new Set(Object.keys(overrides()));
  $("changed-count").textContent = changed.size ? `${changed.size} changed` : "all defaults";
  for (const details of $("sections").children) {
    const n = details.dataset.fields.split(" ").filter((f) => changed.has(f)).length;
    details.countNode.textContent = n ? `· ${n} changed` : "";
  }
}

function applyConfig(config) {
  for (const [name, value] of Object.entries(config)) {
    if (name in fieldsByName) setValue(name, value);
  }
}

function resetForm() {
  for (const name of Object.keys(fieldsByName)) setValue(name, structuredClone(defaults[name]));
}

function showFormError(message) {
  $("form-error").textContent = message || "";
  $("form-error").hidden = !message;
}

async function loadConfigText(text, note) {
  const parsed = await api("/api/config/parse", { method: "POST", body: text });
  resetForm();
  applyConfig(parsed.config);
  const ignored = parsed.ignored.length ? ` (ignored ${parsed.ignored.join(", ")}: server paths)` : "";
  showFormError(ignored ? `${note}${ignored}` : "");
}

$("filter").addEventListener("input", () => {
  const needle = $("filter").value.trim().toLowerCase();
  for (const details of $("sections").children) {
    let any = false;
    for (const field of details.querySelectorAll(".field")) {
      const hit = !needle || field.dataset.search.includes(needle);
      field.hidden = !hit;
      any ||= hit;
    }
    details.hidden = !any;
    if (needle) details.open = any;
  }
});

$("dice").addEventListener("click", () => {
  $("seed").value = Math.floor(Math.random() * 2 ** 31);
  save();
});
$("seed").addEventListener("change", save);
$("seed").addEventListener("keydown", (e) => e.key === "Enter" && generate());
$("reset").addEventListener("click", () => {
  resetForm();
  $("preset").value = "";
  showFormError("");
});
$("import").addEventListener("change", async () => {
  const file = $("import").files[0];
  $("import").value = "";
  if (!file) return;
  try {
    await loadConfigText(await file.text(), `Loaded ${file.name}`);
  } catch (err) {
    showFormError(`${file.name}: ${err.message}`);
  }
});
$("preset").addEventListener("change", async () => {
  const preset = presets.find((p) => p.name === $("preset").value);
  if (!preset) return;
  try {
    await loadConfigText(JSON.stringify(preset.config), `Preset ${preset.name}`);
  } catch (err) {
    showFormError(`${preset.name}: ${err.message}`);
  }
});

let presets = [];

// ---- generating --------------------------------------------------------------------

function applyLimits(limits) {
  maxSize = limits?.max_size ?? null;
  if (maxSize == null) return;
  for (const name of ["width", "height"]) {
    for (const row of rows[name] || []) {
      const input = row.querySelector("input");
      input.min = 1;
      input.max = maxSize;
      input.title = `At most ${maxSize} hexes on this server`;
    }
  }
}

function sizeProblem() {
  if (maxSize == null) return null;
  for (const name of ["width", "height"]) {
    const v = values[name];
    if (!(v >= 1 && v <= maxSize)) return `${name} must be between 1 and ${maxSize} hexes on this server, got ${v}.`;
  }
  return null;
}

async function generate() {
  showFormError("");
  const problem = sizeProblem();
  if (problem) {
    showFormError(problem);
    return;
  }
  let started;
  try {
    started = await api("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ seed: seedValue(), config: overrides() }),
    });
  } catch (err) {
    showFormError(err.message);
    return;
  }
  follow(started);
}

function follow(started) {
  if (events) events.close();
  clearInterval(timer);
  job = started;
  $("generate").disabled = true;
  $("progress").hidden = false;
  $("stages").replaceChildren();
  $("bar-fill").style.width = "0";
  $("progress-label").textContent = "Queued";
  const t0 = performance.now();
  timer = setInterval(() => {
    $("progress-time").textContent = `${((performance.now() - t0) / 1000).toFixed(0)}s`;
  }, 500);

  const items = {};
  events = new EventSource(`/api/jobs/${job.id}/events`);
  events.onmessage = (message) => {
    const event = JSON.parse(message.data);
    if (event.type === "stage") {
      let li = items[event.index];
      if (!li) {
        li = items[event.index] = el("li", {}, [el("span", { textContent: event.name }), el("span")]);
        $("stages").append(li);
        li.scrollIntoView({ block: "nearest" });
      }
      const running = event.elapsed == null;
      li.className = running ? "running" : "done";
      li.lastChild.textContent = running ? "…" : `${event.elapsed.toFixed(1)}s`;
      const done = event.index - (running ? 1 : 0);
      $("bar-fill").style.width = `${(100 * done) / event.total}%`;
      $("progress-label").textContent = `${event.index}/${event.total} ${event.name}`;
    } else if (event.type === "done") {
      finish(`Done in ${event.elapsed.toFixed(1)}s`);
      loadWorld();
    } else if (event.type === "error") {
      finish("Failed");
      showFormError(event.message);
    }
  };
  events.onerror = () => {
    // The stream closes once the world is done; only a close before that is a failure.
    if (events && events.readyState === EventSource.CLOSED) finish("Lost the server");
  };
}

function finish(text) {
  events.close();
  events = null;
  clearInterval(timer);
  $("progress-label").textContent = text;
  $("generate").disabled = false;
  if (text.startsWith("Done")) $("bar-fill").style.width = "100%";
}

async function loadWorld() {
  const summary = await api(`/api/jobs/${job.id}`);
  const select = $("view");
  select.replaceChildren();
  const maps = el("optgroup", { label: "Maps" });
  const plates = el("optgroup", { label: "Debug plates" });
  for (const name of summary.views) {
    (EXPORT_STYLES.includes(name) ? maps : plates).append(el("option", { value: name, textContent: label(name) }));
  }
  select.append(maps, plates);
  if (!summary.views.includes(view)) view = "atlas";
  select.value = view;
  select.disabled = false;
  clearInspector();
  showMap();
}

// ---- the map -----------------------------------------------------------------------

function showMap() {
  const base = `/api/jobs/${job.id}`;
  const q = `view=${encodeURIComponent(view)}`;
  $("empty").hidden = true;
  $("loading").hidden = false;
  $("marker").toggleAttribute("hidden", true);
  $("map").src = `${base}/map.svg?${q}`;

  const link = (id, href, enabled = true) => {
    $(id).href = href;
    $(id).setAttribute("aria-disabled", String(!enabled));
  };
  link("dl-svg", `${base}/map.svg?${q}&download=1`);
  link("dl-png", `${base}/map.png?${q}`, EXPORT_STYLES.includes(view));
  link("dl-world", `${base}/world.json`);
  link("dl-config", `${base}/config.json`);
}

$("map").addEventListener("load", () => {
  $("loading").hidden = true;
  const img = $("map");
  const size = [img.naturalWidth, img.naturalHeight];
  // Refit only when the picture changes size: switching between two maps of one world
  // keeps the place you were looking at.
  if (!same(size, mapSize)) {
    mapSize = size;
    fit();
  }
});
$("map").addEventListener("error", () => {
  $("loading").hidden = true;
  showFormError("The map could not be drawn.");
});

$("view").addEventListener("change", () => {
  view = $("view").value;
  clearInspector();
  showMap();
});

function applyPan() {
  $("stage").style.transform = `translate(${pan.tx}px, ${pan.ty}px) scale(${pan.s})`;
}

function fit() {
  if (!mapSize) return;
  const box = $("viewport").getBoundingClientRect();
  pan.s = Math.min(box.width / mapSize[0], box.height / mapSize[1]) * 0.98;
  pan.tx = (box.width - mapSize[0] * pan.s) / 2;
  pan.ty = (box.height - mapSize[1] * pan.s) / 2;
  applyPan();
}

function zoomAt(factor, cx, cy) {
  const s = Math.min(Math.max(pan.s * factor, 0.02), 20);
  const f = s / pan.s;
  pan.tx = cx - (cx - pan.tx) * f;
  pan.ty = cy - (cy - pan.ty) * f;
  pan.s = s;
  applyPan();
}

function zoomCentre(factor) {
  const box = $("viewport").getBoundingClientRect();
  zoomAt(factor, box.width / 2, box.height / 2);
}

$("zoom-in").addEventListener("click", () => zoomCentre(1.5));
$("zoom-out").addEventListener("click", () => zoomCentre(1 / 1.5));
$("zoom-fit").addEventListener("click", fit);
window.addEventListener("resize", fit);

$("viewport").addEventListener(
  "wheel",
  (e) => {
    e.preventDefault();
    const box = $("viewport").getBoundingClientRect();
    zoomAt(Math.exp(-e.deltaY * 0.0015), e.clientX - box.left, e.clientY - box.top);
  },
  { passive: false },
);

let drag = null;
$("viewport").addEventListener("pointerdown", (e) => {
  if (!mapSize || e.button !== 0) return;
  drag = { x: e.clientX, y: e.clientY, tx: pan.tx, ty: pan.ty, moved: false };
  $("viewport").setPointerCapture(e.pointerId);
});
$("viewport").addEventListener("pointermove", (e) => {
  if (!drag) return;
  const dx = e.clientX - drag.x;
  const dy = e.clientY - drag.y;
  if (!drag.moved && Math.hypot(dx, dy) < 4) return;
  drag.moved = true;
  $("viewport").classList.add("dragging");
  pan.tx = drag.tx + dx;
  pan.ty = drag.ty + dy;
  applyPan();
});
$("viewport").addEventListener("pointerup", (e) => {
  if (!drag) return;
  const clicked = !drag.moved;
  drag = null;
  $("viewport").classList.remove("dragging");
  if (clicked) inspect(e.clientX, e.clientY);
});

// ---- the inspector -----------------------------------------------------------------

function clearInspector() {
  $("inspector-body").replaceChildren();
  $("inspector-empty").hidden = false;
  $("marker").toggleAttribute("hidden", true);
}

async function inspect(clientX, clientY) {
  if (!job) return;
  const box = $("viewport").getBoundingClientRect();
  const x = (clientX - box.left - pan.tx) / pan.s;
  const y = (clientY - box.top - pan.ty) / pan.s;
  let found;
  try {
    found = await api(`/api/jobs/${job.id}/hex?view=${encodeURIComponent(view)}&x=${x}&y=${y}`);
  } catch (err) {
    showFormError(err.message);
    return;
  }
  if (!found.hex) return clearInspector();
  drawMarker(found.outline);
  renderHex(found.hex);
}

function drawMarker({ x, y, size }) {
  const points = [];
  for (let i = 0; i < 6; i++) {
    const a = (Math.PI / 3) * i; // flat-top, as the renderers draw them
    points.push(`${x + size * Math.cos(a)},${y + size * Math.sin(a)}`);
  }
  const marker = $("marker");
  marker.setAttribute("width", mapSize[0]);
  marker.setAttribute("height", mapSize[1]);
  marker.querySelector("polygon").setAttribute("points", points.join(" "));
  marker.removeAttribute("hidden");
}

const FORMAT = {
  elevation: (v) => `${v.toFixed(0)} m`,
  temperature: (v) => `${v.toFixed(1)} °C`,
  moisture: (v) => `${v.toFixed(0)} mm`,
  wet_season_precip_mm: (v) => `${v.toFixed(0)} mm`,
  dry_season_precip_mm: (v) => `${v.toFixed(0)} mm`,
  slope: (v) => `${v.toFixed(1)} m`,
  relief: (v) => `${v.toFixed(0)} m`,
  catchment_km2: (v) => `${v.toFixed(0)} km²`,
  rural_population: (v) => v.toFixed(0),
};

// Shown first, in this order; everything else the hex carries follows.
const ORDER = [
  "coord", "terrain_class", "biome", "elevation", "slope", "relief", "temperature",
  "moisture", "wet_season_precip_mm", "dry_season_precip_mm", "rivers", "river_flow", "catchment_km2", "land_cover", "soil", "land_use",
  "cultivated", "rural_population", "territory_of", "roads", "tags",
];

function show(name, value) {
  if (value == null || (Array.isArray(value) && value.length === 0)) return "—";
  if (FORMAT[name] && typeof value === "number") return FORMAT[name](value);
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(2);
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (name === "coord") return `${value[0]}, ${value[1]}`;
  if (Array.isArray(value)) return value.join(", ");
  return String(value);
}

function renderHex(hex) {
  const body = [];
  if (hex.settlement) {
    const s = hex.settlement;
    body.push(el("p", { className: "place", textContent: s.name || "(unnamed)" }));
    body.push(el("p", { className: "muted", textContent: `${s.tier} · ${s.role} · ${s.population.toLocaleString()} people${s.culture ? ` · ${s.culture}` : ""}` }));
    if (s.etymology) body.push(el("p", { className: "etymology", textContent: s.etymology }));
  }
  const dl = el("dl");
  const keys = [...ORDER, ...Object.keys(hex).filter((k) => !ORDER.includes(k) && k !== "settlement")];
  for (const key of keys) {
    if (!(key in hex)) continue;
    dl.append(el("dt", { textContent: label(key) }), el("dd", { textContent: show(key, hex[key]) }));
  }
  body.push(dl);
  $("inspector-body").replaceChildren(...body);
  $("inspector-empty").hidden = true;
}

// ---- start -------------------------------------------------------------------------

$("generate").addEventListener("click", generate);

(async () => {
  try {
    const [{ sections, limits }, presetList] = await Promise.all([api("/api/schema"), api("/api/presets")]);
    buildForm(sections);
    applyLimits(limits);
    presets = presetList.presets;
    for (const p of presets) $("preset").append(el("option", { value: p.name, textContent: p.name }));
    $("preset").disabled = presets.length === 0;
  } catch (err) {
    showFormError(`Could not reach the server: ${err.message}`);
  }
})();
