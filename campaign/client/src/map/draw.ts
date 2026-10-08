/**
 * Drawing the map onto a canvas.
 *
 * Canvas rather than SVG or DOM: a 32x32 world is a thousand hexes and a 128x128 one is
 * sixteen thousand, which is more than enough to make per-frame DOM work stutter under a
 * moving cursor. Hover highlighting has to feel immediate or the sidebar is useless.
 *
 * Two layers. The terrain is painted once to an offscreen canvas and blitted, because it
 * only changes when the world or the fog does; everything that follows the cursor —
 * columns, the hover ring, reach — is redrawn each frame over the top. That keeps a hover
 * repaint to a blit and a handful of polygons rather than a thousand.
 *
 * Pure functions taking a context. No React, no state.
 */

import {
  axialToPixel,
  cornerId,
  cornerToPixel,
  DEFAULT_THEME,
  hexAt,
  key,
  riverClass,
  sideCorners,
  sideId,
  sideJoining,
  TAG_BRIDGE,
  TAG_FORD,
  type Corner,
  type Hex,
  type HexKey,
  type River,
  type RoadStyle,
  type Theme,
  type World,
  type WorldHex,
} from '@campaign/shared';

import {
  CITY,
  CITY_DISC,
  CITY_INK,
  drawEmblem,
  drawGlyph,
  drawRiverMark,
  ROLE_GLYPH,
  type RiverMarkKind,
} from './glyphs.js';
import { labelFont, placeLabels } from './labels.js';
import { drawSymbol, symbolSize, type SymbolSpec } from './symbols.js';

export type { SymbolSpec } from './symbols.js';

export interface View {
  /** Pixel radius of a hex. */
  readonly size: number;
  readonly offsetX: number;
  readonly offsetY: number;
}

/** Flat-top hex corners, at 0°, 60°, ... 300°. */
const CORNERS = Array.from({ length: 6 }, (_, i) => {
  const a = (Math.PI / 180) * (60 * i);
  return { x: Math.cos(a), y: Math.sin(a) };
});

export const toScreen = (h: Hex, view: View): { x: number; y: number } => {
  const p = axialToPixel(h, view.size);
  return { x: p.x + view.offsetX, y: p.y + view.offsetY };
};

/** A point given at a hex size of 1 — a corner, a side's middle — placed on screen. */
const pointToScreen = (p: { x: number; y: number }, view: View): { x: number; y: number } => ({
  x: p.x * view.size + view.offsetX,
  y: p.y * view.size + view.offsetY,
});

/** A corner on screen. */
const cornerToScreen = (c: Corner, view: View): { x: number; y: number } =>
  pointToScreen(cornerToPixel(c, 1), view);

/**
 * Add one hexagon to a path that is already open.
 *
 * Split out from `hexPath` because that function begins a new path, which makes it
 * useless for accumulating many hexes into one fill: called in a loop it discards
 * everything but the last hexagon. Anything drawing more than one hex at a time wants
 * this and a `Path2D`.
 */
export function hexSubPath(
  path: Path2D | CanvasRenderingContext2D,
  centre: { x: number; y: number },
  size: number,
): void {
  for (const [i, c] of CORNERS.entries()) {
    const x = centre.x + c.x * size;
    const y = centre.y + c.y * size;
    if (i === 0) path.moveTo(x, y);
    else path.lineTo(x, y);
  }
  path.closePath();
}

export function hexPath(ctx: CanvasRenderingContext2D, centre: { x: number; y: number }, size: number): void {
  ctx.beginPath();
  hexSubPath(ctx, centre, size);
}

/**
 * The fill for one hex.
 *
 * Mirrors the Python's `biome` colour mode, which is what `worldgen render` produces by
 * default, so the screen and a printed export agree. Fog wins over everything: an unseen
 * hex has no biome and no elevation to draw, only defaults standing in for the unknown.
 */
export function fillFor(hex: WorldHex, theme: Theme = DEFAULT_THEME): string {
  if (hex.tags.has('fog')) return theme.fog;
  if (hex.terrainClass === 'open_water') return theme.terrain.ocean ?? theme.fallback;
  if (hex.terrainClass === 'inland_water') return theme.terrain.lake ?? theme.fallback;
  if (hex.biome !== null) return theme.biome[hex.biome] ?? theme.fallback;
  return theme.terrain.flat ?? theme.fallback;
}

/**
 * What the map draws, as the reader has chosen it.
 *
 * The ground is one of three readings of the same hexes; everything else is an overlay that
 * can be switched off on its own. Units, riders, the wash and every pointing mode are not
 * here: they are the game, not the map, and are always drawn.
 */
export type Ground = 'terrain' | 'relief' | 'plain';

export const GROUNDS: readonly Ground[] = ['terrain', 'relief', 'plain'];

export interface MapLayers {
  readonly ground: Ground;
  readonly names: boolean;
  readonly settlements: boolean;
  readonly roads: boolean;
  readonly rivers: boolean;
  readonly crossings: boolean;
  readonly ports: boolean;
}

/** The overlays, in the order a control lists them. */
export const OVERLAYS = ['names', 'settlements', 'roads', 'rivers', 'crossings', 'ports'] as const;
export type Overlay = (typeof OVERLAYS)[number];

/** Everything on, over the terrain: the map as it was before there were layers. */
export const ALL_LAYERS: MapLayers = {
  ground: 'terrain',
  names: true,
  settlements: true,
  roads: true,
  rivers: true,
  crossings: true,
  ports: true,
};

/** Paper and water, for reading the road and river networks with nothing behind them. */
const PLAIN_LAND = '#f3ecd9';
const PLAIN_WATER = '#cfe0ef';
const PLAIN_ROAD_CASING = 'rgba(70,52,30,0.55)';

/**
 * Height colours, lowest first: the hypsometric tints of a printed atlas, green lowland
 * through tan upland to rock and snow. Each stop is a fraction of the world's land range.
 */
const RELIEF_STOPS: readonly (readonly [number, readonly [number, number, number]])[] = [
  [0, [120, 168, 104]],
  [0.25, [178, 196, 128]],
  [0.5, [222, 206, 146]],
  [0.72, [196, 156, 104]],
  [0.88, [150, 120, 96]],
  [1, [246, 244, 240]],
];

/** The lowest and highest land on a world, for scaling the relief tints to it. */
export function landRange(world: World): { lo: number; hi: number } {
  let lo = Infinity;
  let hi = -Infinity;
  for (const h of world.hexes.values()) {
    if (h.terrainClass === 'open_water' || h.terrainClass === 'inland_water' || h.tags.has('fog')) {
      continue;
    }
    lo = Math.min(lo, h.elevation);
    hi = Math.max(hi, h.elevation);
  }
  return Number.isFinite(lo) ? { lo, hi } : { lo: 0, hi: 1 };
}

/**
 * A hex's height as a tint, darkened on steep ground so slopes read as relief.
 *
 * Scaled to this world's own range rather than to fixed metres: a lowland map and an alpine
 * one each use the whole ramp, which is what makes the shape of the country legible.
 */
export function reliefColor(elevation: number, slope: number, range: { lo: number; hi: number }): string {
  const t = range.hi > range.lo ? Math.min(1, Math.max(0, (elevation - range.lo) / (range.hi - range.lo))) : 0;
  let i = 1;
  while (i < RELIEF_STOPS.length - 1 && RELIEF_STOPS[i]![0] < t) i++;
  const [t0, c0] = RELIEF_STOPS[i - 1]!;
  const [t1, c1] = RELIEF_STOPS[i]!;
  const f = t1 > t0 ? (t - t0) / (t1 - t0) : 0;
  // A slope of 300 m/km — twice Bad Going — takes a third of the light away.
  const shade = 1 - Math.min(0.35, Math.max(0, slope) / 900);
  const ch = (k: number) => Math.round((c0[k]! + (c1[k]! - c0[k]!) * f) * shade);
  return `rgb(${ch(0)},${ch(1)},${ch(2)})`;
}

/** The fill for one hex under the chosen ground. */
export function groundFill(
  hex: WorldHex,
  ground: Ground,
  range: { lo: number; hi: number },
  theme: Theme = DEFAULT_THEME,
): string {
  if (ground === 'terrain' || hex.tags.has('fog')) return fillFor(hex, theme);
  const wet = hex.terrainClass === 'open_water' || hex.terrainClass === 'inland_water';
  if (ground === 'plain') return wet ? PLAIN_WATER : PLAIN_LAND;
  return wet ? fillFor(hex, theme) : reliefColor(hex.elevation, hex.slope, range);
}

/** The extent of a world in pixels, for fitting it to a viewport. */
export function worldExtent(world: World, size: number): {
  minX: number;
  minY: number;
  width: number;
  height: number;
} {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const h of world.hexes.values()) {
    const p = axialToPixel(h.coord, size);
    minX = Math.min(minX, p.x);
    minY = Math.min(minY, p.y);
    maxX = Math.max(maxX, p.x);
    maxY = Math.max(maxY, p.y);
  }
  return { minX, minY, width: maxX - minX + size * 2, height: maxY - minY + size * 2 };
}

/** Paint the terrain, roads and rivers. Slow, and only redone when the world changes. */
export function drawTerrain(
  ctx: CanvasRenderingContext2D,
  world: World,
  view: View,
  theme: Theme = DEFAULT_THEME,
  layers: MapLayers = ALL_LAYERS,
): void {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  ctx.lineWidth = Math.max(0.4, view.size * 0.04);
  // Fainter on plain paper, where the grid would otherwise be the loudest thing drawn.
  ctx.strokeStyle = layers.ground === 'plain' ? 'rgba(0,0,0,0.07)' : 'rgba(0,0,0,0.18)';

  const range = layers.ground === 'relief' ? landRange(world) : { lo: 0, hi: 1 };
  for (const hex of world.hexes.values()) {
    hexPath(ctx, toScreen(hex.coord, view), view.size);
    ctx.fillStyle = groundFill(hex, layers.ground, range, theme);
    ctx.fill();
    ctx.stroke();
  }

  if (layers.rivers) drawRivers(ctx, world, view, theme);
  // The casing is pale so a road reads on forest and rock; on pale paper it vanishes, and
  // the road with it. A dark edge does the same job there.
  if (layers.roads) {
    drawRoads(ctx, world, view, layers.ground === 'plain' ? { ...theme, roadCasing: PLAIN_ROAD_CASING } : theme);
  }
  if (layers.crossings) {
    for (const c of crossingMarks(world)) {
      const p = pointToScreen(c.at, view);
      drawCrossing(ctx, c.kind, p.x, p.y, c.bearing, view.size);
    }
  }
  // Over the roads and the crossings, so a spring or white water on a forded reach still
  // shows.
  for (const mark of layers.rivers ? riverMarks(world) : []) {
    const p = pointToScreen(mark.at, view);
    drawRiverMark(
      ctx,
      mark.kind,
      p.x,
      p.y,
      mark.bearing,
      Math.max(2.5, view.size * 0.3),
      theme.river.major.color,
    );
  }
  if (layers.ports) {
    for (const c of anchoragePoints(world)) {
      const p = toScreen(c, view);
      drawAnchor(ctx, p.x, p.y, view.size);
    }
  }
  if (layers.settlements) drawSettlements(ctx, world, view);
  if (layers.names) drawLabels(ctx, world, view, theme, layers);
}

/** A river mark: where one rises, where it ends, or where it runs white. */
export interface RiverMark {
  /** Where it sits, at a hex size of 1: a corner, or the middle of a side. */
  readonly at: { readonly x: number; readonly y: number };
  readonly kind: RiverMarkKind;
  /** The river's bearing there, downstream, in radians on the flat-top layout. */
  readonly bearing: number;
}

/** The river's bearing along each side it runs, downstream, in radians. */
function sideBearings(world: World): Map<string, number> {
  const bearing = new Map<string, number>();
  for (const river of world.rivers) {
    river.corners.forEach((c, i) => {
      const next = river.corners[i + 1];
      if (next === undefined) return;
      const a = cornerToPixel(c, 1);
      const b = cornerToPixel(next, 1);
      bearing.set(sideId(sideJoining(c, next)), Math.atan2(b.y - a.y, b.x - a.x));
    });
  }
  return bearing;
}

/** The middle of a side, at a hex size of 1. */
function sideMiddle(side: Parameters<typeof sideCorners>[0]): { x: number; y: number } {
  const [a, b] = sideCorners(side).map((c) => cornerToPixel(c, 1));
  return { x: (a!.x + b!.x) / 2, y: (a!.y + b!.y) / 2 };
}

/**
 * Every river mark, read off what the generator tags: a source or an end on the corner a
 * course starts or stops at (`river_source`, `river_end`), white water at the middle of
 * its side (`cataract`, `rapids`), each turned along its river. Mirrors
 * `legend.river_marks` in the Python, so the campaign marks the same places the exported
 * map does.
 */
export function riverMarks(world: World): RiverMark[] {
  const bearing = sideBearings(world);
  const out: RiverMark[] = [];
  for (const river of world.rivers) {
    const n = river.corners.length;
    if (n < 2) continue;
    const first = river.corners[0]!;
    const last = river.corners[n - 1]!;
    const head = bearing.get(sideId(sideJoining(first, river.corners[1]!))) ?? 0;
    const tail = bearing.get(sideId(sideJoining(river.corners[n - 2]!, last))) ?? 0;
    if (world.riverCorners.get(cornerId(first))?.has('river_source')) {
      out.push({ at: cornerToPixel(first, 1), kind: 'source', bearing: head });
    }
    if (world.riverCorners.get(cornerId(last))?.has('river_end')) {
      out.push({ at: cornerToPixel(last, 1), kind: 'end', bearing: tail });
    }
  }
  for (const rs of world.riverSides.values()) {
    if (rs.tags.has('cataract') || rs.tags.has('rapids')) {
      out.push({
        at: sideMiddle(rs.side),
        kind: 'rapids',
        bearing: bearing.get(sideId(rs.side)) ?? 0,
      });
    }
  }
  return out;
}

/** A ford or a bridge, and the bearing of the river it crosses. */
export interface CrossingMark {
  /** The middle of the river side it crosses, at a hex size of 1. */
  readonly at: { readonly x: number; readonly y: number };
  readonly side: string;
  readonly kind: 'ford' | 'bridge';
  /** The river's bearing, in radians. The mark is laid square across it. */
  readonly bearing: number;
}

/**
 * Every ford and bridge, in stable order, at the middle of the river side it crosses.
 * Mirrors `legend.crossings` in the Python, so the campaign marks the crossings the
 * exported atlas does — and the ones the movement rules read, since both come off the
 * same side tags.
 */
export function crossingMarks(world: World): CrossingMark[] {
  const bearing = sideBearings(world);
  const out: CrossingMark[] = [];
  for (const [id, rs] of world.riverSides) {
    // A minor river is waded almost anywhere, so a ford on one marks nothing worth a
    // symbol; only a major river's fords are. As `legend.crossings` in the Python.
    const ford = rs.tags.has(TAG_FORD) && riverClass(rs, world) === 'major';
    const kind = rs.tags.has(TAG_BRIDGE) ? 'bridge' : ford ? 'ford' : null;
    if (kind === null) continue;
    out.push({
      at: sideMiddle(rs.side),
      side: id,
      kind,
      bearing: bearing.get(id) ?? 0,
    });
  }
  return out.sort((a, b) => a.at.x - b.at.x || a.at.y - b.at.y);
}

/**
 * Every land hex where a route takes to the water: a road's landing at a sea leg, and both
 * ends of every ferry. Mirrors `legend.anchorage_points`, which is where the atlas puts its
 * anchors — this is what a port looks like on the map.
 *
 * A landing counts only where a road reaches it, as in the Python: an anchor on a coast
 * with no road at it would be a port nobody can get to.
 */
export function anchoragePoints(world: World): Hex[] {
  const wet = (c: Hex): boolean => {
    const h = hexAt(world, c);
    return h === undefined || h.terrainClass === 'open_water' || h.terrainClass === 'inland_water';
  };
  const onRoad = new Set<string>();
  for (const e of world.roadEdges.values()) {
    if (wet(e.a) || wet(e.b)) continue;
    onRoad.add(key(e.a));
    onRoad.add(key(e.b));
  }
  const points = new Map<string, Hex>();
  for (const e of world.seaEdges.values()) {
    const aWet = wet(e.a);
    if (aWet === wet(e.b)) continue;
    const shore = aWet ? e.b : e.a;
    if (onRoad.has(key(shore))) points.set(key(shore), shore);
  }
  for (const f of world.ferries) {
    points.set(key(f.a), f.a);
    points.set(key(f.b), f.b);
  }
  return [...points.values()].sort((a, b) => a.q - b.q || a.r - b.r);
}

const CROSSING_INK = '#2b2118';
const ANCHOR_INK = '#1b3a5c';

/**
 * A ford or bridge, laid across the river rather than along it — the atlas's own marks
 * (`_crossing_marker` in `svg_export.py`), at the same proportion of the hex.
 *
 * Bridge: two parallel decks closed by abutments at each bank. Ford: the same span broken
 * into dashes, a way through the water rather than over it.
 */
function drawCrossing(
  ctx: CanvasRenderingContext2D,
  kind: 'ford' | 'bridge',
  cx: number,
  cy: number,
  bearing: number,
  hexSize: number,
): void {
  const s = (5 * hexSize) / 12;
  const half = 0.78 * s;
  const sep = 0.3 * s;
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(bearing + Math.PI / 2);
  ctx.strokeStyle = CROSSING_INK;
  ctx.lineWidth = Math.max(0.6, 0.26 * s);
  ctx.setLineDash(kind === 'ford' ? [0.32 * s, 0.28 * s] : []);
  ctx.beginPath();
  for (const side of [-1, 1]) {
    ctx.moveTo(-half, side * sep);
    ctx.lineTo(half, side * sep);
  }
  if (kind === 'bridge') {
    for (const side of [-1, 1]) {
      ctx.moveTo(side * half, -sep * 1.7);
      ctx.lineTo(side * half, sep * 1.7);
    }
  }
  ctx.stroke();
  ctx.restore();
}

/** An anchor where a route takes to the water: `_anchorage_marker` in `svg_export.py`. */
function drawAnchor(ctx: CanvasRenderingContext2D, cx: number, cy: number, hexSize: number): void {
  const s = (5 * hexSize) / 12;
  ctx.save();
  ctx.strokeStyle = ANCHOR_INK;
  ctx.lineWidth = Math.max(0.6, 0.28 * s);
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.arc(cx, cy - 0.8 * s, 0.26 * s, 0, Math.PI * 2);
  ctx.moveTo(cx, cy - 0.54 * s);
  ctx.lineTo(cx, cy + 0.92 * s);
  ctx.moveTo(cx - 0.6 * s, cy - 0.26 * s);
  ctx.lineTo(cx + 0.6 * s, cy - 0.26 * s);
  ctx.moveTo(cx - 0.72 * s, cy + 0.4 * s);
  ctx.quadraticCurveTo(cx, cy + 1.16 * s, cx + 0.72 * s, cy + 0.4 * s);
  ctx.stroke();
  ctx.restore();
}

/**
 * Place and river names, by the atlas's rules (`labels.ts`).
 *
 * Measured on this canvas, at this zoom: a name is placed against the size it will actually
 * be drawn, and one too small to read is left off. A pale halo under each keeps it legible
 * over a road or a forest.
 */
function drawLabels(
  ctx: CanvasRenderingContext2D,
  world: World,
  view: View,
  theme: Theme,
  layers: MapLayers,
): void {
  const measure = (text: string, size: number, bold: boolean): { w: number; h: number } => {
    ctx.font = labelFont(size, bold, false);
    return { w: ctx.measureText(text).width, h: size };
  };
  // A river's name means nothing without the river under it, so it goes with the rivers.
  const labels = placeLabels(world, (c) => toScreen(c, view), view.size, measure, {
    rivers: layers.rivers,
    markers: layers.settlements,
  });
  ctx.save();
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.lineJoin = 'round';
  for (const l of labels) {
    ctx.save();
    ctx.translate(l.x, l.y);
    if (l.angle !== 0) ctx.rotate(l.angle);
    ctx.font = labelFont(l.size, l.bold, l.italic);
    ctx.lineWidth = Math.max(2, l.size * 0.28);
    ctx.strokeStyle = 'rgba(255,255,255,0.85)';
    ctx.strokeText(l.text, 0, 0);
    ctx.fillStyle = l.river ? theme.river.major.color : '#1e1e1e';
    ctx.fillText(l.text, 0, 0);
    ctx.restore();
  }
  ctx.restore();
}

/** A stretch of one river drawn in one style. */
export interface RiverRun {
  readonly cls: 'major' | 'minor';
  readonly corners: readonly Corner[];
}

/**
 * Split a river into stretches by the rules' Major/Minor class, side by side.
 *
 * A side the masked world has no record of — the fog cut it — counts as minor: the
 * generator drew a watercourse there, so something is drawn.
 */
export function riverRuns(river: River, world: World): RiverRun[] {
  const runs: { cls: 'major' | 'minor'; corners: Corner[] }[] = [];
  river.corners.forEach((c, i) => {
    const next = river.corners[i + 1];
    if (next === undefined) return;
    const side = world.riverSides.get(sideId(sideJoining(c, next)));
    const cls = riverClass(side, world) === 'major' ? 'major' : 'minor';
    const last = runs.at(-1);
    if (last !== undefined && last.cls === cls) last.corners.push(next);
    else runs.push({ cls, corners: [c, next] });
  });
  return runs;
}

function drawRivers(ctx: CanvasRenderingContext2D, world: World, view: View, theme: Theme): void {
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  for (const river of world.rivers) {
    for (const run of riverRuns(river, world)) {
      const style = theme.river[run.cls];
      ctx.strokeStyle = style.color;
      ctx.lineWidth = Math.max(run.cls === 'major' ? 2.5 : 1, view.size * style.width);
      ctx.beginPath();
      run.corners.forEach((c, i) => {
        const p = cornerToScreen(c, view);
        if (i === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      });
      ctx.stroke();
    }
  }
}

function drawRoads(
  ctx: CanvasRenderingContext2D,
  world: World,
  view: View,
  theme: Theme,
): void {
  ctx.lineCap = 'round';
  const tiers = (['track', 'secondary', 'primary'] as const).flatMap((tier) => {
    const style = theme.road[tier];
    return style === undefined ? [] : [{ tier, style }];
  });
  const widthOf = (style: RoadStyle): number => Math.max(1, view.size * 0.06 * style.width);
  const dashOf = (style: RoadStyle): number[] =>
    style.dash ? style.dash.map((d) => d * view.size * 0.15) : [];
  const pathOf = (tier: string): void => {
    ctx.beginPath();
    for (const edge of world.roadEdges.values()) {
      if (edge.tier !== tier) continue;
      const a = toScreen(edge.a, view);
      const b = toScreen(edge.b, view);
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
    }
  };

  // Every casing before any road, so one road's casing never cuts across another road
  // where they meet. A track's casing is dashed with it: solid, it outweighs the track's
  // own pale dashes and every track reads as a white road.
  ctx.strokeStyle = theme.roadCasing;
  for (const { tier, style } of tiers) {
    ctx.lineWidth = widthOf(style) + Math.max(1.5, view.size * 0.05);
    ctx.setLineDash(dashOf(style));
    pathOf(tier);
    ctx.stroke();
  }

  // Ascending tier order, so a primary road is never overdrawn by a track.
  for (const { tier, style } of tiers) {
    ctx.strokeStyle = style.color;
    ctx.lineWidth = widthOf(style);
    ctx.setLineDash(dashOf(style));
    pathOf(tier);
    ctx.stroke();
  }
  ctx.setLineDash([]);
}

function drawSettlements(ctx: CanvasRenderingContext2D, world: World, view: View): void {
  const r = Math.max(2, view.size * 0.28);
  for (const s of world.settlements) {
    const p = toScreen(s.coord, view);
    // A city is a church among roofs on a gold disc, larger again than a trade's tool.
    if (s.tier === 'city') {
      drawEmblem(ctx, CITY, p.x, p.y, Math.max(4, r * 1.15), CITY_DISC, CITY_INK);
      continue;
    }
    // A mine or a lumber camp is drawn as its tool, a little larger so the tool reads.
    const glyph = ROLE_GLYPH[s.role];
    if (glyph !== undefined) {
      drawGlyph(ctx, glyph, p.x, p.y, Math.max(3, r * 0.8));
      continue;
    }
    ctx.fillStyle = '#fff';
    ctx.strokeStyle = '#222';
    ctx.lineWidth = 1;
    ctx.beginPath();
    // A town is a square and a village a smaller disc, as on the atlas — the two read
    // apart at a glance, which matters on a map that is mostly towns.
    if (s.tier === 'village') ctx.arc(p.x, p.y, r * 0.36, 0, Math.PI * 2);
    else ctx.rect(p.x - r / 2, p.y - r / 2, r, r);
    ctx.fill();
    ctx.stroke();
  }
}

/**
 * How much of the observation wash to show.
 *
 * `three` is the honest picture and the default. `two` collapses remembered ground into
 * the dark, which answers the narrower question — *where are my eyes right now* — more
 * loudly. `off` is for reading the terrain itself, and for deciding how dark is too dark.
 */
export type WashMode = 'three' | 'two' | 'off';

/**
 * What is known about one hex, as far as the wash is concerned.
 *
 * Three claims a commander's map has to keep apart:
 *
 * - `observed` — somebody of their is looking at it now. What is drawn here is true.
 * - `surveyed` — their troops have covered it. They know the ground; they know nothing about
 *   who is standing on it, and the map is silent rather than empty.
 * - `unseen` — never covered. The terrain is drawn because `terrainFog` is off, which is
 *   a convenience of this campaign's settings and not something they have earned.
 */
export type WashBand = 'observed' | 'surveyed' | 'unseen';

/**
 * Which band a hex falls in.
 *
 * Pure, and separated from the drawing so the classification can be tested without a
 * canvas. `visible` wins over `surveyed` unconditionally: the two sets are maintained
 * independently, and a hex they are looking at right now is observed whether or not the
 * survey bookkeeping has caught up with it.
 *
 * An empty `visible` means the viewer has no eyes on the map at all — a referee, who is
 * sent no sets because they are not standing anywhere. It must read as *wash nothing*, not
 * as *they see nothing*, or the one screen meant to see everything goes black. The guard
 * lives here rather than in the renderer so the two cannot come to different conclusions.
 */
export function washBand(
  k: HexKey,
  visible: ReadonlySet<HexKey>,
  surveyed: ReadonlySet<HexKey>,
  mode: WashMode,
): WashBand {
  if (mode === 'off' || visible.size === 0) return 'observed';
  if (visible.has(k)) return 'observed';
  if (mode === 'three' && surveyed.has(k)) return 'surveyed';
  return 'unseen';
}

/**
 * The wash over ground nobody is watching.
 *
 * On its own canvas rather than in the overlay, because it changes when the clock moves
 * and not when the cursor does. Painting it per frame would put a pass over every hex of
 * the map back into the hover path, which is the one thing the two-layer split exists to
 * prevent.
 *
 * One fill per band rather than one per hex, following `drawRoads`: the path accumulates
 * every hex of a band and is rasterised once. On a 128x128 world that is two fills
 * instead of sixteen thousand.
 *
 * A referee is sent empty sets, so `washBand` puts everything in `observed` and this
 * draws nothing at all. The early return below is only an optimisation; the decision
 * itself belongs to the classifier.
 */
export function drawWash(
  ctx: CanvasRenderingContext2D,
  world: World,
  view: View,
  opts: {
    visible: ReadonlySet<HexKey>;
    surveyed: ReadonlySet<HexKey>;
    mode: WashMode;
  },
  theme: Theme = DEFAULT_THEME,
): void {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  if (opts.mode === 'off' || opts.visible.size === 0) return;

  const bands: readonly WashBand[] = ['surveyed', 'unseen'];
  for (const band of bands) {
    const alpha = theme.wash[band];
    if (alpha <= 0) continue;

    const path = new Path2D();
    let any = false;
    for (const hex of world.hexes.values()) {
      if (washBand(key(hex.coord), opts.visible, opts.surveyed, opts.mode) !== band) continue;
      // A hair over the true size. Neighbouring hexes share an edge, and at exactly
      // `view.size` the antialiased seams between them let a lattice of bright lines
      // through the wash.
      hexSubPath(path, toScreen(hex.coord, view), view.size + 0.5);
      any = true;
    }
    if (!any) continue;

    ctx.save();
    ctx.fillStyle = theme.wash.color;
    ctx.globalAlpha = alpha;
    // Every hex is a separate subpath and neighbours overlap by the half pixel above.
    // Nonzero so those overlaps stay filled; even-odd would punch them out and the wash
    // would come back as a lattice of holes.
    ctx.fill(path, 'nonzero');
    ctx.restore();
  }
}

/**
 * What a mark stands for, which decides how it is drawn.
 *
 * Three kinds, and they must be told apart at a glance, because on a commander's map they
 * are three different claims about the world:
 *
 * - `live` — a formation they are standing next to. This is true now.
 * - `reported` — one of their own, where a despatch says it stood. True at an hour that has
 *   passed, and the older it is the less it means.
 * - `contact` — an enemy somebody saw. Barely anything is known and it may have marched.
 *
 * An earlier version drew the last two identically as small faint dots, which was both
 * illegible and a lie: it made "my own cavalry, an hour ago" and "an enemy column, seen
 * by somebody else" look like the same class of fact.
 */
export type MarkKind = 'live' | 'reported' | 'contact';

/**
 * Something to draw on the map: an id, the ground it covers, and its symbol.
 *
 * Deliberately not a unit. The symbol spec carries only what is drawn — the arm and size
 * as far as they are known, and no further — so the drawing code never has the enemy's
 * record within reach in the first place. A contact whose arm is not known arrives here
 * with `kind: null`, not with the arm and an instruction not to draw it.
 */
export interface Mark {
  readonly id: string;
  /** The ground it occupies. One hex for anything not observed directly. */
  readonly column: readonly Hex[];
  readonly kind: MarkKind;
  readonly symbol: SymbolSpec;
  /**
   * Drawn deployed: three parallel lines rather than a column ribbon.
   *
   * Set only for a formation actually observed. How an enemy is standing is intelligence
   * — a corps in line is about to fight and a corps in column is about to march — and a
   * sighting carries no such thing, so a reported or contact mark is never deployed here
   * even when the formation behind it is.
   */
  readonly deployed: boolean;
}

/**
 * A despatch rider, on the referee's map and on nobody else's.
 *
 * The route is where the addressee actually is, so drawing it for a commander would hand
 * them the position of their own detached corps and end the game. For a referee it is the
 * best thing on the screen: they can watch a rider cross the country between two armies and
 * see, before the dice do, that the order is about to pass a picket.
 *
 * `ridden` is the ground behind them and `ahead` the ground in front — different weights,
 * because where a rider has got to is the fact and where they are going is a plan.
 */
export interface Rider {
  readonly id: string;
  readonly at: Hex;
  readonly ridden: readonly Hex[];
  readonly ahead: readonly Hex[];
  readonly color: string;
}

/**
 * Ground being fought over.
 *
 * Deliberately not faction-coloured. A battlefield belongs to nobody while it is being
 * fought over — that is what makes it one — and tinting it with a side's colour would
 * announce an outcome the map has no business predicting. So it is drawn as heat: a wash
 * and a hatch, in a colour no faction uses.
 */
function drawBattleGround(
  ctx: CanvasRenderingContext2D,
  view: View,
  battle: ReadonlySet<HexKey> | undefined,
): void {
  if (battle === undefined || battle.size === 0) return;

  ctx.save();
  for (const k of battle) {
    const [q, r] = k.split(',').map(Number);
    const p = toScreen({ q: q!, r: r! }, view);

    ctx.fillStyle = 'rgba(200,64,40,0.22)';
    hexPath(ctx, p, view.size * 0.96);
    ctx.fill();

    ctx.strokeStyle = 'rgba(220,90,60,0.7)';
    ctx.lineWidth = Math.max(1, view.size * 0.06);
    ctx.stroke();
  }
  ctx.restore();
}

/** Everything that follows the cursor. Redrawn every frame; must stay cheap. *//** Everything that follows the cursor. Redrawn every frame; must stay cheap. */
export function drawOverlay(
  ctx: CanvasRenderingContext2D,
  view: View,
  opts: {
    marks: readonly Mark[];
    hovered: Hex | null;
    hoveredUnitId: string | null;
    selectedUnitId: string | null;
    reach?: ReadonlyMap<string, number> | undefined;
    riders?: readonly Rider[] | undefined;
    /** A hex the referee is about to choose as a destination. */
    picking?: Hex | null | undefined;
    /** Ground the referee has pointed at so far, in the order they pointed at it. */
    route?: readonly Hex[] | undefined;
    /** Where every marching column is going, as the engine would route it now. */
    plans?: readonly Plan[] | undefined;
    /** Ground being fought over. */
    battle?: ReadonlySet<HexKey> | undefined;
  },
): void {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  // Under everything. A battlefield is ground, not a thing standing on it, and the
  // formations in the fighting have to stay readable over the top of it.
  drawBattleGround(ctx, view, opts.battle);

  if (opts.reach !== undefined) {
    ctx.fillStyle = 'rgba(255,255,255,0.16)';
    for (const k of opts.reach.keys()) {
      const [q, r] = k.split(',').map(Number);
      hexPath(ctx, toScreen({ q: q!, r: r! }, view), view.size * 0.96);
      ctx.fill();
    }
  }

  // Under the formations, because they are plans rather than facts: where the columns are
  // going, and the ground the referee has pointed at but not yet committed.
  for (const plan of opts.plans ?? []) drawPlan(ctx, view, plan);
  drawPickedRoute(ctx, view, opts.route ?? []);

  // Riders under the formations: a courier is one rider on a horse and a division is a corps,
  // and where the two are on the same hex the corps is the thing to see.
  for (const rider of opts.riders ?? []) drawRider(ctx, view, rider);

  // Live formations first, so a report or a contact standing on the same ground is drawn
  // over them rather than hidden beneath.
  const order: MarkKind[] = ['live', 'reported', 'contact'];
  for (const kind of order) {
    for (const mark of opts.marks) {
      if (mark.kind !== kind) continue;
      const emphasised = mark.id === opts.hoveredUnitId || mark.id === opts.selectedUnitId;
      drawMark(ctx, view, mark, emphasised);
    }
  }

  if (opts.hovered !== null) {
    // While a destination is being chosen the ring is the accent colour and thicker: the
    // next click does something irreversible, and the cursor should say so.
    const picking = opts.picking !== undefined && opts.picking !== null;
    ctx.strokeStyle = picking ? '#cba135' : '#ffffff';
    ctx.lineWidth = Math.max(picking ? 2.5 : 1.5, view.size * (picking ? 0.18 : 0.12));
    hexPath(ctx, toScreen(opts.hovered, view), view.size * 0.94);
    ctx.stroke();
  }
}

/** A column's route ahead, in its own colour. */
export interface Plan {
  readonly unitId: string;
  readonly color: string;
  readonly route: readonly Hex[];
}

/**
 * Where a column is going.
 *
 * Thin, dotted and under everything else. This is the referee's own screen and they have
 * every column on it at once, so a plan has to be legible as a direction without becoming
 * the thing the eye lands on — the formations are the map, the routes are an annotation.
 *
 * A small square on the destination, because a route that simply stops leaves "is that
 * where they are going, or is that as far as I drew it" unanswerable.
 */
function drawPlan(ctx: CanvasRenderingContext2D, view: View, plan: Plan): void {
  const points = plan.route.map((h) => toScreen(h, view));
  if (points.length < 2) return;

  ctx.save();
  ctx.globalAlpha = 0.55;
  ctx.strokeStyle = plan.color;
  ctx.lineWidth = Math.max(1, view.size * 0.08);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.setLineDash([view.size * 0.28, view.size * 0.36]);

  ctx.beginPath();
  ctx.moveTo(points[0]!.x, points[0]!.y);
  for (const p of points.slice(1)) ctx.lineTo(p.x, p.y);
  ctx.stroke();

  const end = points.at(-1)!;
  const r = Math.max(1.5, view.size * 0.2);
  ctx.setLineDash([]);
  ctx.globalAlpha = 0.8;
  ctx.fillStyle = plan.color;
  ctx.fillRect(end.x - r, end.y - r, r * 2, r * 2);
  ctx.restore();
}

/**
 * The places a referee has pointed at, in the order they pointed at them.
 *
 * Straight segments between the picks rather than the route itself. The route is the
 * engine's to work out — drawing a guess at it here would be the client claiming to know
 * ground it has no cost model for, and it would be wrong the moment a road it cannot see
 * turns out to be quicker. Numbers, because "by way of A then B" and "by way of B then A"
 * are different orders and an undecorated chain of rings does not say which this is.
 */
function drawPickedRoute(ctx: CanvasRenderingContext2D, view: View, route: readonly Hex[]): void {
  if (route.length === 0) return;
  const points = route.map((h) => toScreen(h, view));

  ctx.save();
  ctx.strokeStyle = '#cba135';
  ctx.lineWidth = Math.max(1.5, view.size * 0.12);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  if (points.length > 1) {
    ctx.save();
    ctx.globalAlpha = 0.7;
    ctx.setLineDash([view.size * 0.5, view.size * 0.4]);
    ctx.beginPath();
    ctx.moveTo(points[0]!.x, points[0]!.y);
    for (const p of points.slice(1)) ctx.lineTo(p.x, p.y);
    ctx.stroke();
    ctx.restore();
  }

  points.forEach((p, i) => {
    hexPath(ctx, p, view.size * 0.9);
    ctx.stroke();

    // The last pick is where they are to end up; the rest are only on the way. Labelling
    // it as such stops a referee counting rings to find their destination.
    const label = i === points.length - 1 ? '×' : String(i + 1);
    ctx.font = `${Math.max(8, view.size * 0.9).toFixed(0)}px system-ui, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.lineWidth = Math.max(2, view.size * 0.22);
    ctx.strokeStyle = 'rgba(0,0,0,0.65)';
    ctx.strokeText(label, p.x, p.y);
    ctx.fillStyle = '#cba135';
    ctx.fillText(label, p.x, p.y);
    ctx.strokeStyle = '#cba135';
    ctx.lineWidth = Math.max(1.5, view.size * 0.12);
  });

  ctx.restore();
}

/**
 * One rider: the road behind them solid and thin, the road ahead dotted, and a small mark
 * where they have actually got to.
 *
 * Small on purpose. There may be a dozen of these on a busy evening and they must not
 * compete with the formations — a rider is a fact about communication rather than about
 * ground, and the eye should find them only when it goes looking.
 */
function drawRider(ctx: CanvasRenderingContext2D, view: View, rider: Rider): void {
  const line = (path: readonly Hex[], dash: boolean, alpha: number): void => {
    if (path.length < 2) return;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = rider.color;
    ctx.lineWidth = Math.max(1, view.size * 0.1);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    if (dash) ctx.setLineDash([view.size * 0.35, view.size * 0.4]);
    ctx.beginPath();
    path.forEach((h, i) => {
      const p = toScreen(h, view);
      if (i === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    });
    ctx.stroke();
    ctx.restore();
  };

  line(rider.ridden, false, 0.7);
  line(rider.ahead, true, 0.35);

  // A pale ring rather than a dark one. The column ribbons are faction-coloured lines
  // too, so a faction-coloured dot on a faction-coloured trail disappears into it; the
  // ring is what separates "one rider on a horse" from "eighteen kilometres of cavalry".
  const p = toScreen(rider.at, view);
  const r = Math.max(3.5, view.size * 0.3);
  ctx.save();
  ctx.beginPath();
  ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
  ctx.fillStyle = rider.color;
  ctx.fill();
  ctx.lineWidth = Math.max(1.4, r * 0.4);
  ctx.strokeStyle = '#e6e8ef';
  ctx.stroke();
  ctx.restore();
}

/**
 * One mark: a NATO symbol at the head, and behind it the ground the column covers.
 *
 * The trail is the point of the whole model. A division is not a counter on a hex — at its
 * own spacing it is between two and eighteen kilometres of road, exposed along all of it,
 * and a reader should see that without consulting the sidebar. So the symbol says what the
 * formation is and the trail says how much country it is standing on.
 *
 * Only a formation actually observed has a trail. A reported position and an enemy
 * sighting are single hexes: a despatch says where something was, not how it was strung
 * out, and drawing a column from either would invent a report nobody made.
 */
function drawMark(
  ctx: CanvasRenderingContext2D,
  view: View,
  mark: Mark,
  emphasised: boolean,
): void {
  const { column, symbol } = mark;
  if (column.length === 0) return;

  const head = toScreen(column[0]!, view);

  if (mark.deployed) {
    drawDeployed(ctx, view, mark, emphasised);
  } else if (column.length > 1) {
    ctx.save();
    ctx.strokeStyle = symbol.color;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    // Capped as well as scaled. Zoomed in, an uncapped ribbon becomes a wall that hides
    // the ground it is meant to be lying on.
    ctx.lineWidth = Math.max(2, Math.min(11, view.size * (emphasised ? 0.4 : 0.28)));
    // Behind the symbol rather than through it, so the frame stays readable.
    ctx.globalAlpha = 0.85;
    ctx.beginPath();
    column.forEach((c, i) => {
      const p = toScreen(c, view);
      if (i === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    });
    ctx.stroke();

    // A cap at the tail, so the end of the column is a place rather than a fade.
    const tail = toScreen(column[column.length - 1]!, view);
    ctx.beginPath();
    ctx.arc(tail.x, tail.y, Math.max(1.5, Math.min(7, view.size * 0.16)), 0, Math.PI * 2);
    ctx.fillStyle = symbol.color;
    ctx.fill();
    ctx.restore();
  }

  drawSymbol(ctx, head.x, head.y, symbolSize(view.size), { ...symbol, emphasised });
}

/**
 * A formation standing in line of battle: three parallel lines.
 *
 * A column ribbon would be wrong here in the most misleading way available. A deployed
 * division covers about a kilometre — one hex — so its ribbon is a dot, and a reader
 * would see the formation that is about to fight as the smallest thing on the map. The
 * three lines say the opposite, and say it in the language the period used for itself:
 * ranks, drawn across the front rather than along the road.
 *
 * They run along the frontage and stack across it. Where the front is a single hex there
 * is no direction to take from the ground, so they lie flat — the orientation carries no
 * information the model has, and a made-up facing would imply one it does not.
 */
function drawDeployed(
  ctx: CanvasRenderingContext2D,
  view: View,
  mark: Mark,
  emphasised: boolean,
): void {
  const { column, symbol } = mark;
  const head = toScreen(column[0]!, view);
  const tail = toScreen(column[column.length - 1]!, view);

  // The axis of the front. A single-hex front has none, so it lies flat.
  const dx = tail.x - head.x;
  const dy = tail.y - head.y;
  const span = Math.hypot(dx, dy);
  const [ax, ay] = span > 0.5 ? [dx / span, dy / span] : [1, 0];
  // Perpendicular, which is the direction the ranks stack in.
  const [px, py] = [-ay, ax];

  const cx = (head.x + tail.x) / 2;
  const cy = (head.y + tail.y) / 2;
  const half = Math.max(view.size * 0.55, span / 2 + view.size * 0.3);
  const gap = Math.max(2.5, view.size * 0.2);

  ctx.save();
  ctx.strokeStyle = symbol.color;
  ctx.lineCap = 'round';
  ctx.globalAlpha = 0.85;
  ctx.lineWidth = Math.max(1.5, Math.min(6, view.size * (emphasised ? 0.16 : 0.11)));

  for (const rank of [-1, 0, 1]) {
    const ox = px * gap * rank;
    const oy = py * gap * rank;
    ctx.beginPath();
    ctx.moveTo(cx + ox - ax * half, cy + oy - ay * half);
    ctx.lineTo(cx + ox + ax * half, cy + oy + ay * half);
    ctx.stroke();
  }
  ctx.restore();
}

/**
 * What is on this hex, for hit-testing the cursor against a column.
 *
 * Returns an id rather than a record. The map draws marks and knows nothing about what
 * they stand for — one of them is a division whose every statistic this browser holds,
 * and another is a smudge on the horizon that was an enemy an hour ago. Resolving the id
 * is the console's job, and keeping that out of here is what stops the two being
 * conflated in the one place where conflating them would leak.
 */
export function markAtHex(marks: readonly Mark[], c: Hex): string | null {
  const k = key(c);
  for (const { id, column } of marks) {
    if (column.some((h) => key(h) === k)) return id;
  }
  return null;
}

/** How long a place pointed at from the places list flashes for, in milliseconds. */
export const FLASH_MS = 2400;
const FLASH_PULSES = 3;

/**
 * One frame of the flash on a place chosen from the places list, `elapsed` ms in.
 *
 * Rings that swell out from the hex and fade, three of them, so the eye is drawn to the
 * spot even when the map has just moved under it. With `still` — the reader has asked for
 * less motion — one steady ring is drawn for the same time instead.
 */
export function drawFlash(
  ctx: CanvasRenderingContext2D,
  view: View,
  hex: Hex,
  elapsed: number,
  still: boolean,
  color: string,
): void {
  if (elapsed < 0 || elapsed >= FLASH_MS) return;
  const p = toScreen(hex, view);
  const base = Math.max(8, view.size);
  ctx.save();
  ctx.strokeStyle = color;
  if (still) {
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(p.x, p.y, base * 1.3, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
    return;
  }
  const phase = ((elapsed / FLASH_MS) * FLASH_PULSES) % 1;
  // The last pulse fades out rather than stopping dead.
  const fade = 1 - elapsed / FLASH_MS;
  ctx.globalAlpha = (1 - phase) * (0.35 + 0.65 * fade);
  ctx.lineWidth = 3 + 2 * (1 - phase);
  ctx.beginPath();
  ctx.arc(p.x, p.y, base * (0.8 + 1.8 * phase), 0, Math.PI * 2);
  ctx.stroke();
  // A fixed inner ring, so the spot itself is marked between pulses.
  ctx.globalAlpha = 0.9 * fade + 0.1;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(p.x, p.y, base * 0.75, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}
