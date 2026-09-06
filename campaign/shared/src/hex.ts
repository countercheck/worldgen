/**
 * Axial hex math, ported from `worldgen/core/hex_grid.py`.
 *
 * The Python is the reference implementation. This file must agree with it case for
 * case, which `test/hex.conformance.test.ts` enforces against a fixture the Python
 * generates — if either side drifts, one of the two suites goes red.
 *
 * Flat-top orientation throughout. Both grid layouts store hexes under axial
 * coordinates; only the *set* of hexes a world is built from differs, so adjacency,
 * distance and pathfinding are unaffected by the layout.
 */

export interface Hex {
  readonly q: number;
  readonly r: number;
}

/** A `Map`-safe identity for a hex. Hexes are compared by value everywhere. */
export type HexKey = string;

export const key = (h: Hex): HexKey => `${h.q},${h.r}`;

export function unkey(k: HexKey): Hex {
  const comma = k.indexOf(',');
  return { q: Number(k.slice(0, comma)), r: Number(k.slice(comma + 1)) };
}

export const AXIAL = 'axial';
export const OFFSET = 'offset';
export type Layout = typeof AXIAL | typeof OFFSET;

/** Odd-q offset column/row to axial (flat-top layout). */
export function offsetToAxial(col: number, row: number): Hex {
  return { q: col, r: row - (col - (col & 1)) / 2 };
}

/** Axial to odd-q offset column/row (flat-top layout). */
export function axialToOffset(h: Hex): { col: number; row: number } {
  return { col: h.q, row: h.r + (h.q - (h.q & 1)) / 2 };
}

/** The hex coordinate `layout` stores at grid column/row. */
export function gridCoord(layout: Layout, col: number, row: number): Hex {
  return layout === OFFSET ? offsetToAxial(col, row) : { q: col, r: row };
}

/** The grid column/row `layout` stores `h` at — the inverse of `gridCoord`. */
export function gridIndex(layout: Layout, h: Hex): { col: number; row: number } {
  return layout === OFFSET ? axialToOffset(h) : { col: h.q, row: h.r };
}

/**
 * The six axial directions, in the order `hex_grid.py` lists them.
 *
 * The order is load-bearing: `ring` walks them in sequence to close a loop, and the
 * conformance fixture compares `neighbors` element by element.
 */
export const DIRECTIONS: readonly Hex[] = [
  { q: 1, r: 0 },
  { q: 1, r: -1 },
  { q: 0, r: -1 },
  { q: -1, r: 0 },
  { q: -1, r: 1 },
  { q: 0, r: 1 },
];

/** Six neighbours in axial coordinates. */
export function neighbors(h: Hex): Hex[] {
  return DIRECTIONS.map((d) => ({ q: h.q + d.q, r: h.r + d.r }));
}

/** Distance in axial coordinates. */
export function distance(a: Hex, b: Hex): number {
  return (
    (Math.abs(a.q - b.q) + Math.abs(a.r - b.r) + Math.abs(a.q + a.r - (b.q + b.r))) / 2
  );
}

/**
 * All hexes at exactly `radius` distance from `center`.
 *
 * Walks the ring one corner at a time: start `radius` steps along one direction, then
 * take `radius` steps along each of the six in turn, which closes the loop exactly.
 */
export function ring(center: Hex, radius: number): Hex[] {
  if (radius <= 0) return [center];

  const start = DIRECTIONS[4]!;
  let q = center.q + start.q * radius;
  let r = center.r + start.r * radius;

  const results: Hex[] = [];
  for (const d of DIRECTIONS) {
    for (let i = 0; i < radius; i++) {
      results.push({ q, r });
      q += d.q;
      r += d.r;
    }
  }
  return results;
}

/** All hexes within `radius` distance of `center`. The recon primitive. */
export function hexRange(center: Hex, radius: number): Hex[] {
  const results: Hex[] = [];
  for (let r = 0; r <= radius; r++) results.push(...ring(center, r));
  return results;
}

const SQRT3 = Math.sqrt(3);

/** Axial to pixel (flat-top layout). */
export function axialToPixel(h: Hex, hexSize: number): { x: number; y: number } {
  return {
    x: hexSize * ((3 / 2) * h.q),
    y: hexSize * ((SQRT3 / 2) * h.q + SQRT3 * h.r),
  };
}

/** Pixel to axial (flat-top layout). */
export function pixelToAxial(x: number, y: number, hexSize: number): Hex {
  return roundAxial(
    ((2 / 3) * x) / hexSize,
    ((-1 / 3) * x + (SQRT3 / 3) * y) / hexSize,
  );
}

/**
 * Round half to even — Python's `round()`, which is NOT `Math.round`.
 *
 * `Math.round(0.5)` is 1 and `Math.round(-0.5)` is -0; Python gives 0 and 0. The
 * difference only shows on an exact .5, which is precisely what a point on a hex
 * boundary produces, so using `Math.round` here would put boundary clicks in a
 * different hex than the Python does and break the conformance fixture.
 */
function pyRound(x: number): number {
  const floor = Math.floor(x);
  const diff = x - floor;
  if (diff > 0.5) return floor + 1;
  if (diff < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

/**
 * Collapse negative zero.
 *
 * JavaScript has a -0 that Python's integers do not, and hex arithmetic produces it
 * readily: `-rr - rs` is -0 whenever both terms are zero. It compares equal to 0 under
 * `===` so most code never notices, but `Object.is` and deep-equality do not, which
 * makes coordinates that are arithmetically identical test as different. Normalising
 * here keeps a `Hex` a single canonical value for every position.
 */
const nz = (x: number): number => (x === 0 ? 0 : x);

/** Round fractional axial coordinates to the nearest hex. */
export function roundAxial(q: number, r: number): Hex {
  const s = -q - r;
  let rq = pyRound(q);
  let rr = pyRound(r);
  const rs = pyRound(s);

  const qDiff = Math.abs(rq - q);
  const rDiff = Math.abs(rr - r);
  const sDiff = Math.abs(rs - s);

  if (qDiff > rDiff && qDiff > sDiff) rq = -rr - rs;
  else if (rDiff > sDiff) rr = -rq - rs;

  return { q: nz(rq), r: nz(rr) };
}

/**
 * A binary heap ordered the way Python's `heapq` orders `(f, coord)` tuples: by score,
 * then by q, then by r.
 *
 * The tie-break is not decoration. Equal-scored frontier entries are common on a hex
 * grid, and without a total order the path returned would depend on insertion order —
 * so two runs of the same search could differ, and the TS could differ from the Python.
 */
class Frontier {
  private readonly items: { f: number; h: Hex }[] = [];

  private static before(a: { f: number; h: Hex }, b: { f: number; h: Hex }): boolean {
    if (a.f !== b.f) return a.f < b.f;
    if (a.h.q !== b.h.q) return a.h.q < b.h.q;
    return a.h.r < b.h.r;
  }

  get size(): number {
    return this.items.length;
  }

  push(f: number, h: Hex): void {
    const items = this.items;
    items.push({ f, h });
    let i = items.length - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (!Frontier.before(items[i]!, items[parent]!)) break;
      [items[i], items[parent]] = [items[parent]!, items[i]!];
      i = parent;
    }
  }

  pop(): Hex | undefined {
    const items = this.items;
    const top = items[0];
    if (top === undefined) return undefined;
    const last = items.pop()!;
    if (items.length > 0) {
      items[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let best = i;
        if (l < items.length && Frontier.before(items[l]!, items[best]!)) best = l;
        if (r < items.length && Frontier.before(items[r]!, items[best]!)) best = r;
        if (best === i) break;
        [items[i], items[best]] = [items[best]!, items[i]!];
        i = best;
      }
    }
    return top.h;
  }
}

/**
 * A* over a hex grid. `nodeCost` is the cost of entering a hex (Infinity = impassable);
 * `edgeCost` adds an optional per-edge term.
 *
 * Returns the path including both endpoints, or null if the goal cannot be reached.
 */
export function astar<T>(
  grid: ReadonlyMap<HexKey, T>,
  start: Hex,
  goal: Hex,
  nodeCost: (hex: T, coord: Hex) => number,
  edgeCost?: (from: T, to: T, fromCoord: Hex, toCoord: Hex) => number,
): Hex[] | null {
  const startKey = key(start);
  const goalKey = key(goal);
  if (!grid.has(startKey) || !grid.has(goalKey)) return null;

  const frontier = new Frontier();
  frontier.push(0, start);
  const cameFrom = new Map<HexKey, HexKey | null>([[startKey, null]]);
  const gScore = new Map<HexKey, number>([[startKey, 0]]);
  const visited = new Set<HexKey>();

  while (frontier.size > 0) {
    const current = frontier.pop()!;
    const currentKey = key(current);
    if (visited.has(currentKey)) continue;
    visited.add(currentKey);

    if (currentKey === goalKey) {
      const path: Hex[] = [];
      let node: HexKey | null = goalKey;
      while (node !== null) {
        path.push(unkey(node));
        node = cameFrom.get(node) ?? null;
      }
      return path.reverse();
    }

    const currentHex = grid.get(currentKey)!;
    for (const next of neighbors(current)) {
      const nextKey = key(next);
      const nextHex = grid.get(nextKey);
      if (nextHex === undefined || visited.has(nextKey)) continue;

      let cost = nodeCost(nextHex, next);
      if (edgeCost !== undefined) cost += edgeCost(currentHex, nextHex, current, next);

      // Checked after the edge term so an impassable *edge* is skipped too. Adding
      // Infinity and pushing the node instead would let the search reach the goal with
      // an infinite score and return a path straight through the forbidden edge.
      if (!Number.isFinite(cost)) continue;

      const tentative = gScore.get(currentKey)! + cost;
      const known = gScore.get(nextKey);
      if (known === undefined || tentative < known) {
        cameFrom.set(nextKey, currentKey);
        gScore.set(nextKey, tentative);
        frontier.push(tentative + distance(next, goal), next);
      }
    }
  }

  return null;
}
