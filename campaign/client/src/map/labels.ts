/**
 * Where the names go on the map.
 *
 * A port of `worldgen/export/labels.py`, by the same three rules, so a name sits in the
 * same place on the campaign map as on the atlas the generator exported:
 *
 * - **Size says rank.** A city's name is set larger and bolder than a town's, a town's than
 *   a village's.
 * - **Rivers are italic and follow the water,** set along the middle reach and turned to
 *   the river's heading.
 * - **Nothing overlaps.** Each name tries above, below, right and left of its place and
 *   takes the first spot clear of every marker and every name already set, or is left off.
 *   Names are set in order of rank, so a crowded map drops villages before cities.
 *
 * One thing the atlas does not have to do: zoom. A name too small to read at the current
 * scale is not set at all, and so is no obstacle to the names that are.
 */

import { cornerToPixel, type Hex, type World } from '@campaign/shared';

/** Font size as a fraction of the hex size. */
const TIER_SCALE: Readonly<Record<string, number>> = { city: 0.95, town: 0.75, village: 0.58 };
const RIVER_SCALE = 0.62;
const TIER_ORDER: Readonly<Record<string, number>> = { city: 0, town: 1, village: 3 };
const RIVER_ORDER = 2;
/** How far from a settlement's centre its name stands, and how much room its marker takes. */
const GAP = 0.55;
const MARKER = 0.45;
/** Smaller than this, in pixels, and a name is not set. */
export const MIN_LABEL_PX = 7;

type Box = readonly [number, number, number, number];
type Point = { readonly x: number; readonly y: number };

/** (text, font size, bold) -> width and height in pixels. */
export type Measure = (text: string, size: number, bold: boolean) => { w: number; h: number };

export interface PlacedLabel {
  readonly text: string;
  /** Centre of the text, in canvas pixels. */
  readonly x: number;
  readonly y: number;
  readonly size: number;
  readonly bold: boolean;
  readonly italic: boolean;
  /** Radians clockwise from horizontal; only rivers turn. */
  readonly angle: number;
  readonly river: boolean;
}

/** A width for text when there is no canvas to measure it with, as the Python estimates. */
export const estimateWidth: Measure = (text, size, bold) => ({
  w: text.length * size * (bold ? 0.62 : 0.56),
  h: size,
});

const clear = (box: Box, taken: readonly Box[]): boolean =>
  taken.every(([a0, b0, a1, b1]) => box[2] <= a0 || box[0] >= a1 || box[3] <= b0 || box[1] >= b1);

/** The axis-aligned box round a `w` x `h` rectangle centred on (cx, cy), turned. */
function rotatedBox(cx: number, cy: number, w: number, h: number, angle: number): Box {
  const bw = Math.abs(w * Math.cos(angle)) + Math.abs(h * Math.sin(angle));
  const bh = Math.abs(w * Math.sin(angle)) + Math.abs(h * Math.cos(angle));
  return [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2];
}

type Job =
  | { order: readonly number[]; kind: 'settlement'; s: World['settlements'][number] }
  | { order: readonly number[]; kind: 'river'; name: string; reach: readonly Point[] };

const byOrder = (a: readonly number[], b: readonly number[]): number => {
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    const d = (a[i] ?? 0) - (b[i] ?? 0);
    if (d !== 0) return d;
  }
  return 0;
};

/**
 * Every name that fits, positioned, in the order to draw them.
 *
 * `toPixel` maps a hex to its centre on the canvas. `markers` says whether settlement markers
 * are drawn — they are obstacles if so — and `rivers` whether rivers are, since a river's
 * name means nothing without the river under it. Both default to on, as in the Python.
 */
export function placeLabels(
  world: World,
  toPixel: (c: Hex) => Point,
  hexSize: number,
  measure: Measure = estimateWidth,
  opts: { rivers?: boolean; markers?: boolean } = {},
): PlacedLabel[] {
  const taken: Box[] = [];
  const m = MARKER * hexSize;
  if (opts.markers ?? true) {
    for (const s of world.settlements) {
      const { x, y } = toPixel(s.coord);
      taken.push([x - m, y - m, x + m, y + m]);
    }
  }

  const jobs: Job[] = [];
  for (const s of world.settlements) {
    if (s.name === '') continue;
    jobs.push({
      order: [TIER_ORDER[s.tier] ?? 3, -s.population, s.coord.q, s.coord.r],
      kind: 'settlement',
      s,
    });
  }
  // A river's course is its corners, which never stand in water; they are placed from
  // the same origin `toPixel` puts the hexes on.
  const origin = toPixel({ q: 0, r: 0 });
  for (const river of opts.rivers === false ? [] : world.rivers) {
    const reach = river.corners.map((c) => {
      const p = cornerToPixel(c, hexSize);
      return { x: p.x + origin.x, y: p.y + origin.y };
    });
    if (river.name !== '' && reach.length >= 3) {
      const first = river.corners[0]!;
      jobs.push({
        order: [RIVER_ORDER, -reach.length, first.q, first.r, first.k],
        kind: 'river',
        name: river.name,
        reach,
      });
    }
  }
  jobs.sort((a, b) => byOrder(a.order, b.order));

  const placed: PlacedLabel[] = [];
  for (const job of jobs) {
    const label =
      job.kind === 'settlement'
        ? placeSettlement(job.s, toPixel, hexSize, measure, taken)
        : placeRiver(job.name, job.reach, hexSize, measure, taken);
    if (label !== null) placed.push(label);
  }
  return placed;
}

function placeSettlement(
  s: World['settlements'][number],
  toPixel: (c: Hex) => Point,
  hexSize: number,
  measure: Measure,
  taken: Box[],
): PlacedLabel | null {
  const size = (TIER_SCALE[s.tier] ?? TIER_SCALE.village!) * hexSize;
  if (size < MIN_LABEL_PX) return null;
  const bold = s.tier === 'city';
  const { w, h } = measure(s.name, size, bold);
  const { x, y } = toPixel(s.coord);
  const gap = GAP * hexSize;
  for (const [cx, cy] of [
    [x, y - gap - h / 2],
    [x, y + gap + h / 2],
    [x + gap + w / 2, y],
    [x - gap - w / 2, y],
  ] as const) {
    const box: Box = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2];
    if (clear(box, taken)) {
      taken.push(box);
      return { text: s.name, x: cx, y: cy, size, bold, italic: false, angle: 0, river: false };
    }
  }
  return null;
}

function placeRiver(
  name: string,
  reach: readonly Point[],
  hexSize: number,
  measure: Measure,
  taken: Box[],
): PlacedLabel | null {
  const size = RIVER_SCALE * hexSize;
  if (size < MIN_LABEL_PX) return null;
  const { w, h } = measure(name, size, false);
  const n = reach.length;
  // The middle of the reach first, then either third: the middle is where a reader's eye
  // expects it, and the thirds are where there is often more room. Each angle is read
  // across two sides either way, which smooths the zigzag of a hexside course.
  for (const i of new Set([Math.floor(n / 2), Math.floor(n / 3), Math.floor((2 * n) / 3)])) {
    const a = reach[Math.max(0, i - 2)]!;
    const b = reach[Math.min(n - 1, i + 2)]!;
    let angle = Math.atan2(b.y - a.y, b.x - a.x);
    // Never upside down: a name reads left to right whichever way the water runs.
    if (angle > Math.PI / 2) angle -= Math.PI;
    else if (angle <= -Math.PI / 2) angle += Math.PI;
    const { x, y } = reach[i]!;
    // Beside the line rather than on it, on either bank.
    const nx = -Math.sin(angle);
    const ny = Math.cos(angle);
    const off = h * 0.5 + hexSize * 0.2;
    for (const side of [-1, 1]) {
      const cx = x + side * nx * off;
      const cy = y + side * ny * off;
      const box = rotatedBox(cx, cy, w, h, angle);
      if (clear(box, taken)) {
        taken.push(box);
        return { text: name, x: cx, y: cy, size, bold: false, italic: true, angle, river: true };
      }
    }
  }
  return null;
}

/** The canvas font for a label. */
export const labelFont = (size: number, bold: boolean, italic: boolean): string =>
  `${italic ? 'italic ' : ''}${bold ? '600 ' : ''}${size.toFixed(1)}px system-ui, sans-serif`;
