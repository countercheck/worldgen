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
  type Theme,
  type World,
  type WorldHex,
} from '@campaign/shared';

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

export function hexPath(ctx: CanvasRenderingContext2D, centre: { x: number; y: number }, size: number): void {
  ctx.beginPath();
  for (const [i, c] of CORNERS.entries()) {
    const x = centre.x + c.x * size;
    const y = centre.y + c.y * size;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
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
 * Something to draw on the map: an id, the ground it covers, and what kind of claim it is.
 *
 * Deliberately not a unit. Keeping this shape thin is what stops the drawing code from
 * having the enemy's record in reach at all.
 */
export interface Mark {
  readonly id: string;
  readonly column: readonly Hex[];
  readonly color: string;
  readonly kind: MarkKind;
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
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = Math.max(1.5, view.size * 0.12);
    hexPath(ctx, toScreen(opts.hovered, view), view.size * 0.94);
    ctx.stroke();
  }
}

/**
 * One mark, drawn according to what it claims.
 *
 * A live formation is a solid disc at the head of a solid ribbon along the ground it
 * occupies — the length is the point, and a reader should see a division is a column of
 * hexes without consulting the sidebar. A report is a hollow ring: something was there.
 * A contact is a diamond, a different shape entirely, because it is not one of yours.
 *
 * The two uncertain kinds are drawn *larger* than the certain one, not smaller. They are
 * what a commander is actually reasoning about, and the instinct to make uncertainty faint
 * had them nearly invisible on a map whose whole subject is uncertainty.
 */
function drawMark(
  ctx: CanvasRenderingContext2D,
  view: View,
  mark: Mark,
  emphasised: boolean,
): void {
  const { column, color, kind } = mark;
  if (column.length === 0) return;

  const head = toScreen(column[0]!, view);
  const r = view.size * (emphasised ? 0.58 : 0.46);
  ctx.lineWidth = Math.max(1.5, view.size * 0.1);

  if (kind === 'live') {
    if (column.length > 1) {
      ctx.strokeStyle = color;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.lineWidth = Math.max(2, view.size * (emphasised ? 0.42 : 0.3));
      ctx.beginPath();
      column.forEach((c, i) => {
        const p = toScreen(c, view);
        if (i === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      });
      ctx.stroke();
      ctx.lineWidth = Math.max(1.5, view.size * 0.1);
    }

    ctx.beginPath();
    ctx.arc(head.x, head.y, r * 0.9, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = emphasised ? '#ffffff' : 'rgba(0,0,0,0.55)';
    ctx.stroke();
    return;
  }

  if (kind === 'reported') {
    // A hollow ring, dashed: a position somebody vouched for, not one you can see.
    ctx.beginPath();
    ctx.arc(head.x, head.y, r, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(0,0,0,0.45)';
    ctx.fill();
    ctx.setLineDash([view.size * 0.28, view.size * 0.22]);
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(2, view.size * 0.16);
    ctx.stroke();
    ctx.setLineDash([]);

    if (emphasised) {
      ctx.beginPath();
      ctx.arc(head.x, head.y, r * 1.45, 0, Math.PI * 2);
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = Math.max(1.5, view.size * 0.08);
      ctx.stroke();
    }
    return;
  }

  // A contact: a diamond, so it cannot be mistaken for one of yours at a glance.
  const d = r * 1.15;
  ctx.beginPath();
  ctx.moveTo(head.x, head.y - d);
  ctx.lineTo(head.x + d, head.y);
  ctx.lineTo(head.x, head.y + d);
  ctx.lineTo(head.x - d, head.y);
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.8;
  ctx.fill();
  ctx.globalAlpha = 1;
  ctx.setLineDash([view.size * 0.25, view.size * 0.2]);
  ctx.strokeStyle = emphasised ? '#ffffff' : 'rgba(255,255,255,0.75)';
  ctx.lineWidth = Math.max(1.5, view.size * 0.11);
  ctx.stroke();
  ctx.setLineDash([]);
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
