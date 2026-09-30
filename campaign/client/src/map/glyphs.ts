/**
 * Icons for settlements: a pickaxe for a mine, an axe for a lumber camp, a church among
 * roofs for a city.
 *
 * Mirrors `worldgen/render/glyphs.py`, which draws the same shapes into the SVG and PNG
 * exports — `tests/test_glyphs.py` compares the two files, so change both together.
 * Coordinates are in a unit square, x right and y down, -1 to 1 either way; a handle is
 * drawn as a thick line and a head as a filled polygon, inside a white disc.
 */

export interface Glyph {
  readonly handle: readonly [readonly [number, number], readonly [number, number]];
  readonly head: readonly (readonly [number, number])[];
  /** Handle thickness, as a share of the disc's radius. */
  readonly handleWidth: number;
}

export const AXE: Glyph = {
  handle: [[-0.6, 0.82], [0.38, -0.46]],
  head: [[0.41, -0.47], [1.08, -0.24], [0.97, 0.12], [0.64, 0.4], [0.19, -0.19]],
  handleWidth: 0.2,
};

export const PICKAXE: Glyph = {
  handle: [[-0.6, 0.82], [0.25, -0.27]],
  head: [[-0.45, -0.92], [-0.13, -0.87], [0.2, -0.76], [0.49, -0.59], [0.73, -0.36], [0.93, -0.07], [1.07, 0.22], [0.85, 0.04], [0.59, -0.16], [0.32, -0.37], [0.05, -0.56], [-0.22, -0.76]],
  handleWidth: 0.2,
};

/** The icon for a settlement role, or undefined for a settlement drawn plain. */
export const ROLE_GLYPH: Readonly<Record<string, Glyph>> = {
  mining: PICKAXE,
  lumber: AXE,
};

/**
 * Draw *glyph* in a white disc of radius *r* at (x, y). The icon fills 80% of the disc,
 * so its outline stays clear.
 */
export function drawGlyph(
  ctx: CanvasRenderingContext2D,
  glyph: Glyph,
  x: number,
  y: number,
  r: number,
): void {
  const s = r * 0.8;
  const at = (p: readonly [number, number]): [number, number] => [x + p[0] * s, y + p[1] * s];

  ctx.fillStyle = '#fff';
  ctx.strokeStyle = '#222';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  const [a, b] = [at(glyph.handle[0]), at(glyph.handle[1])];
  ctx.strokeStyle = '#3b2a1a';
  ctx.lineWidth = Math.max(1, glyph.handleWidth * r);
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(a[0], a[1]);
  ctx.lineTo(b[0], b[1]);
  ctx.stroke();
  ctx.lineCap = 'butt';

  ctx.fillStyle = '#2b2b2b';
  ctx.beginPath();
  glyph.head.forEach((p, i) => {
    const [px, py] = at(p);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.closePath();
  ctx.fill();
}

/** A silhouette: filled polygons, drawn dark on a coloured disc. */
export interface Emblem {
  readonly shapes: readonly (readonly (readonly [number, number])[])[];
}

/** A city: a church tower and spire between two houses, the right one set a little lower. */
export const CITY: Emblem = {
  shapes: [
    [[-0.85, 0.75], [-0.85, 0.2], [-0.55, -0.15], [-0.25, 0.2], [-0.25, 0.75]],
    [[-0.18, 0.75], [-0.18, -0.35], [0.0, -1.0], [0.18, -0.35], [0.18, 0.75]],
    [[0.25, 0.75], [0.25, 0.3], [0.55, 0.0], [0.85, 0.3], [0.85, 0.75]],
  ],
};

/** The gold of a city's disc, and the ink of its silhouette. */
export const CITY_DISC = '#e8c547';
export const CITY_INK = '#2b2b2b';

/**
 * Draw *emblem* on a disc of radius *r* at (x, y), filled *disc*. Like `drawGlyph`, the
 * silhouette fills 80% of the disc.
 */
export function drawEmblem(
  ctx: CanvasRenderingContext2D,
  emblem: Emblem,
  x: number,
  y: number,
  r: number,
  disc: string,
  ink: string,
): void {
  const s = r * 0.8;
  ctx.fillStyle = disc;
  ctx.strokeStyle = '#222';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = ink;
  for (const shape of emblem.shapes) {
    ctx.beginPath();
    shape.forEach(([px, py], i) => {
      if (i === 0) ctx.moveTo(x + px * s, y + py * s);
      else ctx.lineTo(x + px * s, y + py * s);
    });
    ctx.closePath();
    ctx.fill();
  }
}

/*
 * Marks on a river, drawn in the river's own frame: x runs downstream, y across the water,
 * one unit a mark's size. Mirrors the RIVER_ constants in glyphs.py.
 */

/** A source is a spring: a ring with a dot. */
export const RIVER_SOURCE_RING = 0.7;
export const RIVER_SOURCE_DOT = 0.3;

/** An end is an arrowhead pointing where the water goes. */
export const RIVER_END_ARROW: readonly (readonly [number, number])[] = [[1.1, 0.0], [-0.6, 0.85], [-0.2, 0.0], [-0.6, -0.85]];

/** White water is three bars across the river, white on a dark casing. */
export const RAPIDS_BARS: readonly (readonly [readonly [number, number], readonly [number, number]])[] = [[[-0.6, -0.9], [-0.6, 0.9]], [[0.0, -0.9], [0.0, 0.9]], [[0.6, -0.9], [0.6, 0.9]]];
export const RAPIDS_INK = '#ffffff';
export const RAPIDS_CASING = '#1f3b57';

export type RiverMarkKind = 'source' | 'end' | 'rapids';

/**
 * Draw a river mark at (x, y), *size* pixels to the unit, turned so x runs along
 * *bearing* (radians, screen y down).
 */
export function drawRiverMark(
  ctx: CanvasRenderingContext2D,
  kind: RiverMarkKind,
  x: number,
  y: number,
  bearing: number,
  size: number,
  riverColor: string,
): void {
  const ca = Math.cos(bearing);
  const sa = Math.sin(bearing);
  const at = ([px, py]: readonly [number, number]): [number, number] => [
    x + (px * ca - py * sa) * size,
    y + (px * sa + py * ca) * size,
  ];

  if (kind === 'source') {
    ctx.fillStyle = '#fff';
    ctx.strokeStyle = riverColor;
    ctx.lineWidth = Math.max(1, 0.35 * size);
    ctx.beginPath();
    ctx.arc(x, y, RIVER_SOURCE_RING * size, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = riverColor;
    ctx.beginPath();
    ctx.arc(x, y, RIVER_SOURCE_DOT * size, 0, Math.PI * 2);
    ctx.fill();
    return;
  }

  if (kind === 'end') {
    ctx.fillStyle = riverColor;
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = Math.max(0.8, 0.2 * size);
    ctx.lineJoin = 'round';
    ctx.beginPath();
    RIVER_END_ARROW.forEach((p, i) => {
      const [px, py] = at(p);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    return;
  }

  ctx.lineCap = 'round';
  for (const [colour, width] of [
    [RAPIDS_CASING, 0.55 * size],
    [RAPIDS_INK, 0.28 * size],
  ] as const) {
    ctx.strokeStyle = colour;
    ctx.lineWidth = Math.max(1, width);
    for (const [a, b] of RAPIDS_BARS) {
      const [ax, ay] = at(a);
      const [bx, by] = at(b);
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
    }
  }
  ctx.lineCap = 'butt';
}
