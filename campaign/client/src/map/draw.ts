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
  DEFAULT_THEME,
  key,
  type Hex,
  type HexKey,
  type Theme,
  type World,
  type WorldHex,
} from '@campaign/shared';

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
): void {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  ctx.lineWidth = Math.max(0.4, view.size * 0.04);
  ctx.strokeStyle = 'rgba(0,0,0,0.18)';

  for (const hex of world.hexes.values()) {
    hexPath(ctx, toScreen(hex.coord, view), view.size);
    ctx.fillStyle = fillFor(hex, theme);
    ctx.fill();
    ctx.stroke();
  }

  drawRivers(ctx, world, view);
  drawRoads(ctx, world, view, theme);
  drawSettlements(ctx, world, view);
}

function drawRivers(ctx: CanvasRenderingContext2D, world: World, view: View): void {
  ctx.strokeStyle = '#3a7ad9';
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  for (const river of world.rivers) {
    if (river.hexes.length < 2) continue;
    ctx.lineWidth = Math.max(1, view.size * 0.18);
    ctx.beginPath();
    river.hexes.forEach((c, i) => {
      const p = toScreen(c, view);
      if (i === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    });
    ctx.stroke();
  }
}

function drawRoads(
  ctx: CanvasRenderingContext2D,
  world: World,
  view: View,
  theme: Theme,
): void {
  ctx.lineCap = 'round';
  // Ascending tier order, so a primary road is never overdrawn by a track.
  for (const tier of ['track', 'secondary', 'primary'] as const) {
    const style = theme.road[tier];
    if (style === undefined) continue;
    ctx.strokeStyle = style.color;
    ctx.lineWidth = Math.max(1, view.size * 0.06 * style.width);
    ctx.setLineDash(style.dash ? style.dash.map((d) => d * view.size * 0.15) : []);

    ctx.beginPath();
    for (const edge of world.roadEdges.values()) {
      if (edge.tier !== tier) continue;
      const a = toScreen(edge.a, view);
      const b = toScreen(edge.b, view);
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
    }
    ctx.stroke();
  }
  ctx.setLineDash([]);
}

function drawSettlements(ctx: CanvasRenderingContext2D, world: World, view: View): void {
  const r = Math.max(2, view.size * 0.28);
  for (const s of world.settlements) {
    const p = toScreen(s.coord, view);
    ctx.fillStyle = '#fff';
    ctx.strokeStyle = '#222';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.rect(p.x - r / 2, p.y - r / 2, r, r);
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
 * - `observed` — somebody of his is looking at it now. What is drawn here is true.
 * - `surveyed` — his men have covered it. He knows the ground; he knows nothing about
 *   who is standing on it, and the map is silent rather than empty.
 * - `unseen` — never covered. The terrain is drawn because `terrainFog` is off, which is
 *   a convenience of this campaign's settings and not something he has earned.
 */
export type WashBand = 'observed' | 'surveyed' | 'unseen';

/**
 * Which band a hex falls in.
 *
 * Pure, and separated from the drawing so the classification can be tested without a
 * canvas. `visible` wins over `surveyed` unconditionally: the two sets are maintained
 * independently, and a hex he is looking at right now is observed whether or not the
 * survey bookkeeping has caught up with it.
 *
 * An empty `visible` means the viewer has no eyes on the map at all — a referee, who is
 * sent no sets because he is not standing anywhere. It must read as *wash nothing*, not
 * as *he sees nothing*, or the one screen meant to see everything goes black. The guard
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
 * - `live` — a formation he is standing next to. This is true now.
 * - `reported` — one of his own, where a despatch says it stood. True at an hour that has
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
}

/**
 * A despatch rider, on the referee's map and on nobody else's.
 *
 * The route is where the addressee actually is, so drawing it for a commander would hand
 * him the position of his own detached corps and end the game. For a referee it is the
 * best thing on the screen: he can watch a rider cross the country between two armies and
 * see, before the dice do, that the order is about to pass a picket.
 *
 * `ridden` is the ground behind him and `ahead` the ground in front — different weights,
 * because where a rider has got to is the fact and where he is going is a plan.
 */
export interface Rider {
  readonly id: string;
  readonly at: Hex;
  readonly ridden: readonly Hex[];
  readonly ahead: readonly Hex[];
  readonly color: string;
}

/** Everything that follows the cursor. Redrawn every frame; must stay cheap. */
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
  },
): void {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  if (opts.reach !== undefined) {
    ctx.fillStyle = 'rgba(255,255,255,0.16)';
    for (const k of opts.reach.keys()) {
      const [q, r] = k.split(',').map(Number);
      hexPath(ctx, toScreen({ q: q!, r: r! }, view), view.size * 0.96);
      ctx.fill();
    }
  }

  // Riders under the formations: a courier is a man on a horse and a division is a corps,
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

/**
 * One rider: the road behind him solid and thin, the road ahead dotted, and a small mark
 * where he has actually got to.
 *
 * Small on purpose. There may be a dozen of these on a busy evening and they must not
 * compete with the formations — a rider is a fact about communication rather than about
 * ground, and the eye should find him only when it goes looking.
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
  // ring is what separates "a man on a horse" from "eighteen kilometres of cavalry".
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

  if (column.length > 1) {
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
