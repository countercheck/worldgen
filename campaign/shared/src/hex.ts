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

/**
 * A direction turned sixty degrees.
 *
 * The cube rotation `(x, y, z) -> (-z, -x, -y)`, written in axial. Used to find the flank
 * of a column: a formation that spreads sideways spreads perpendicular to the road it
 * came in on, and the only thing that knows where that road ran is the direction between
 * two consecutive hexes of its path.
 *
 * Rotation preserves distance, so a unit direction stays a unit direction.
 */
export const rotate60 = (d: Hex): Hex => ({ q: -d.r, r: d.q + d.r });

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

// ---- corners and sides --------------------------------------------------
//
// A port of the corner and side names in `hex_grid.py`; see there for the scheme. In
// short: corner k of a flat-top hex sits at 60·k degrees (y down), side i joins corner i
// to corner i + 1, and a hex owns its corners 0 and 1 and its sides 0, 1 and 2, so every
// corner and every side has exactly one name.

/** A corner where three hexes meet: corner `k` (0 or 1) of the hex that owns it. */
export interface Corner {
  readonly q: number;
  readonly r: number;
  readonly k: number;
}

/** A side between two hexes: side `s` (0, 1 or 2) of the hex that owns it. */
export interface Side {
  readonly q: number;
  readonly r: number;
  readonly s: number;
}

/** `"q,r,k"` — the same string the Python writes, so events and JSON can name a corner. */
export const cornerId = (c: Corner): string => `${c.q},${c.r},${c.k}`;

/** `"q,r,s"` — the same string the Python writes, so events and JSON can name a side. */
export const sideId = (s: Side): string => `${s.q},${s.r},${s.s}`;

/** The neighbour across each side, in side order. Not the same order as `DIRECTIONS`. */
const SIDE_DIRECTIONS: readonly Hex[] = [
  { q: 1, r: 0 },
  { q: 0, r: 1 },
  { q: -1, r: 1 },
  { q: -1, r: 0 },
  { q: 0, r: -1 },
  { q: 1, r: -1 },
];

/** Corner k of a hex as an offset to its owner, and the owner's corner index. */
const CORNER_OWNERS: readonly [number, number, number][] = [
  [0, 0, 0],
  [0, 0, 1],
  [-1, 1, 0],
  [-1, 0, 1],
  [-1, 0, 0],
  [0, -1, 1],
];

const mod6 = (i: number): number => ((i % 6) + 6) % 6;

/** The name of corner `k` (0–5) of a hex. */
export function cornerOf(h: Hex, k: number): Corner {
  const [dq, dr, owned] = CORNER_OWNERS[mod6(k)]!;
  return { q: h.q + dq, r: h.r + dr, k: owned };
}

/** The name of side `i` (0–5) of a hex — the side it shares with that neighbour. */
export function sideOf(h: Hex, i: number): Side {
  const j = mod6(i);
  if (j < 3) return { q: h.q, r: h.r, s: j };
  const d = SIDE_DIRECTIONS[j]!;
  return { q: h.q + d.q, r: h.r + d.r, s: j - 3 };
}

/** A hex's six corners, in corner order. */
export const hexCorners = (h: Hex): Corner[] => [0, 1, 2, 3, 4, 5].map((k) => cornerOf(h, k));

/** A hex's six sides, in side order. */
export const hexSides = (h: Hex): Side[] => [0, 1, 2, 3, 4, 5].map((i) => sideOf(h, i));

/** The three hexes meeting at a corner: its owner first. */
export function cornerHexes(c: Corner): Hex[] {
  return c.k === 0
    ? [
        { q: c.q, r: c.r },
        { q: c.q + 1, r: c.r },
        { q: c.q + 1, r: c.r - 1 },
      ]
    : [
        { q: c.q, r: c.r },
        { q: c.q + 1, r: c.r },
        { q: c.q, r: c.r + 1 },
      ];
}

/** The three corners one side away, in step with `cornerSides`. */
export function cornerNeighbors(c: Corner): Corner[] {
  return c.k === 0
    ? [
        { q: c.q, r: c.r, k: 1 },
        { q: c.q, r: c.r - 1, k: 1 },
        { q: c.q + 1, r: c.r - 1, k: 1 },
      ]
    : [
        { q: c.q, r: c.r, k: 0 },
        { q: c.q - 1, r: c.r + 1, k: 0 },
        { q: c.q, r: c.r + 1, k: 0 },
      ];
}

/** The side two neighbouring hexes share; the same whichever is named first. */
export function sideBetween(a: Hex, b: Hex): Side {
  const dq = b.q - a.q;
  const dr = b.r - a.r;
  const i = SIDE_DIRECTIONS.findIndex((d) => d.q === dq && d.r === dr);
  if (i < 0) throw new Error(`${key(a)} and ${key(b)} are not neighbours`);
  return sideOf(a, i);
}

/** The two hexes either side of a side: its owner first. */
export function sideHexes(s: Side): Hex[] {
  const d = SIDE_DIRECTIONS[s.s]!;
  return [
    { q: s.q, r: s.r },
    { q: s.q + d.q, r: s.r + d.r },
  ];
}

/** A side's two end corners, in the owner's clockwise order. */
export function sideCorners(s: Side): Corner[] {
  const owner = { q: s.q, r: s.r };
  return [cornerOf(owner, s.s), cornerOf(owner, s.s + 1)];
}

/** The three sides meeting at a corner, in step with `cornerNeighbors`. */
export function cornerSides(c: Corner): Side[] {
  const mine = cornerHexes(c);
  return cornerNeighbors(c).map((n) => {
    // Two neighbouring corners share exactly two hexes; the side between them joins the
    // two corners.
    const theirs = new Set(cornerHexes(n).map(key));
    const [a, b] = mine.filter((h) => theirs.has(key(h)));
    return sideBetween(a!, b!);
  });
}

/** The side running from corner `a` to neighbouring corner `b`. */
export function sideJoining(a: Corner, b: Corner): Side {
  // Two neighbouring corners share exactly two hexes; the side between them joins the
  // two corners.
  const theirs = new Set(cornerHexes(b).map(key));
  const shared = cornerHexes(a).filter((h) => theirs.has(key(h)));
  if (shared.length !== 2) throw new Error(`${cornerId(a)} and ${cornerId(b)} are not neighbouring corners`);
  return sideBetween(shared[0]!, shared[1]!);
}

/** Pixel position of a corner (flat-top layout). */
export function cornerToPixel(c: Corner, hexSize: number): { x: number; y: number } {
  const { x, y } = axialToPixel({ q: c.q, r: c.r }, hexSize);
  const angle = (Math.PI / 3) * c.k;
  return { x: x + hexSize * Math.cos(angle), y: y + hexSize * Math.sin(angle) };
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
 * `heuristic` estimates the remaining cost and defaults to hex distance, which is what
 * the Python does and what the conformance fixture pins. **It must never overestimate**,
 * or the path returned is not the cheapest one. The default is only admissible when no
 * step can cost less than 1.0 — callers working in other units (hours, say, where a
 * courier on a highway crosses a hex in a tenth of one) must scale it down accordingly.
 *
 * Returns the path including both endpoints, or null if the goal cannot be reached.
 */
export function astar<T>(
  grid: ReadonlyMap<HexKey, T>,
  start: Hex,
  goal: Hex,
  nodeCost: (hex: T, coord: Hex) => number,
  edgeCost?: (from: T, to: T, fromCoord: Hex, toCoord: Hex) => number,
  heuristic: (from: Hex, to: Hex) => number = distance,
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
        frontier.push(tentative + heuristic(next, goal), next);
      }
    }
  }

  return null;
}
