/**
 * A faction's map: the world reduced to what it knows.
 *
 * The output is a `world.json` in the generator's own schema, so a commander's map opens
 * in every tool that reads a world — including the Python renderer:
 *
 *     worldgen render --input fog.json --attribute biome --output fog.svg
 *
 * This works on the **raw document** rather than on the parsed `World`. The parser keeps
 * only the fields the rules need, roughly a third of what a hex carries, and masking has
 * to preserve all of them — moisture, soil, land use, habitability — or the file it
 * produces is a different kind of world rather than a partial one.
 *
 * ## Unknown hexes are nulled, not omitted
 *
 * Two verified reasons, and both are easy to get wrong:
 *
 * 1. `WorldState.from_dict` requires `elevation`, `moisture`, `temperature`,
 *    `terrain_class`, `river_flow` and `cultivated` on every hex. An omitted hex does not
 *    load at all.
 * 2. Every Python renderer takes min/max over the hexes present to size its canvas. Drop
 *    hexes and each faction's map comes out a different size and differently centred, so
 *    two commanders' maps could not be laid over one another — or over the referee's.
 *
 * So an unknown hex keeps its entry, carries default values, and is tagged `fog`.
 *
 * **This is a small lie by construction.** An unseen hex serialises as flat land at
 * elevation zero, so a consumer that ignores tags reads ground where there is sea. The
 * alternatives are worse: a new `terrain_class` value breaks every existing reader, and
 * omission breaks both loading and the canvas bounds. The tag is the contract, and
 * `metadata.fog` states it in the file itself.
 */

import { key, type Hex, type HexKey } from './hex.js';

/** The tag an unseen hex carries. Nothing should hand-write this string. */
export const FOG_TAG = 'fog';

/** Known, but not currently in view — drawn as memory rather than as observation. */
export const REMEMBERED_TAG = 'remembered';

export interface MaskOptions {
  /** Every hex the faction has ever observed. */
  readonly seen: ReadonlySet<HexKey>;
  /** What it can see right now — a subset of `seen`. */
  readonly visible: ReadonlySet<HexKey>;
  readonly faction: string;
  readonly clockHours: number;
  /**
   * When false, memory is discarded and only what is currently in view survives.
   *
   * The rules have no such mode; it exists for a referee who wants to show a player
   * exactly what their pickets see this hour.
   */
  readonly remember?: boolean;
}

type Doc = Record<string, unknown>;

const coordKey = (q: unknown, r: unknown): HexKey => `${q as number},${r as number}`;

/**
 * A hex with nothing in it.
 *
 * Every field the schema requires, at the dataclass defaults `worldgen` would use. The
 * `fog` tag is the only thing that distinguishes it from genuine flat ground, which is
 * why it must never be dropped.
 */
function blankHex(q: number, r: number): Doc {
  return {
    q,
    r,
    elevation: 0.0,
    moisture: 0.0,
    temperature: 0.0,
    biome: null,
    terrain_class: 'land',
    slope: 0.0,
    relief: 0.0,
    alluvium: 0.0,
    land_cover: null,
    soil: null,
    land_use: null,
    rural_population: 0.0,
    river_flow: 0.0,
    catchment_km2: 0.0,
    habitability_city: 0.0,
    habitability_town: 0.0,
    habitability_village: 0.0,
    cultivated: false,
    territory: null,
    territory_cost: 0.0,
    tags: [FOG_TAG],
    road_connections: [],
  };
}

/**
 * Reduce a world document to one faction's knowledge.
 *
 * Pure: the input document is not touched.
 */
export function maskWorld(doc: Doc, opts: MaskOptions): Doc {
  const remember = opts.remember ?? true;
  const known: ReadonlySet<HexKey> = remember ? opts.seen : opts.visible;

  const rawHexes = Array.isArray(doc.hexes) ? (doc.hexes as Doc[]) : [];

  const hexes = rawHexes.map((h) => {
    const k = coordKey(h.q, h.r);
    if (!known.has(k)) return blankHex(h.q as number, h.r as number);

    const tags = Array.isArray(h.tags) ? [...(h.tags as string[])] : [];
    if (!opts.visible.has(k) && !tags.includes(REMEMBERED_TAG)) tags.push(REMEMBERED_TAG);

    return {
      ...h,
      tags: tags.sort(),
      // Pruned to known neighbours. Without this the file leaks the shape of a road
      // running on into country the faction has never seen — a commander could read the
      // enemy's lines of communication off their own map.
      road_connections: (Array.isArray(h.road_connections) ? h.road_connections : []).filter(
        (c) => known.has(coordKey((c as number[])[0], (c as number[])[1])),
      ),
    };
  });

  const bothKnown = (e: Doc): boolean => {
    const a = e.a as number[];
    const b = e.b as number[];
    return known.has(coordKey(a[0], a[1])) && known.has(coordKey(b[0], b[1]));
  };

  const edges = (field: string): Doc[] =>
    (Array.isArray(doc[field]) ? (doc[field] as Doc[]) : []).filter(bothKnown);

  const settlements = (Array.isArray(doc.settlements) ? (doc.settlements as Doc[]) : []).filter(
    (s) => {
      const c = s.coord as number[];
      return known.has(coordKey(c[0], c[1]));
    },
  );

  return {
    ...doc,
    hexes,
    rivers: maskRivers(Array.isArray(doc.rivers) ? (doc.rivers as Doc[]) : [], known),
    settlements,
    road_edges: edges('road_edges'),
    sea_edges: edges('sea_edges'),
    ferries: edges('ferries'),
    metadata: {
      ...((doc.metadata as Doc) ?? {}),
      fog: {
        faction: opts.faction,
        clock_hours: opts.clockHours,
        remembered: remember,
        fog_tag: FOG_TAG,
        remembered_tag: REMEMBERED_TAG,
        known: known.size,
        visible: opts.visible.size,
        total: rawHexes.length,
        note:
          'Hexes tagged "fog" were never observed by this faction. Their values are ' +
          'defaults, not measurements — do not read them as terrain.',
      },
    },
  };
}

/**
 * Split each river into the runs of it the faction has actually seen.
 *
 * A river is a chain, and keeping only the known hexes of one would leave a renderer
 * drawing a straight line between two banks either side of unseen country. Runs shorter
 * than two hexes are dropped: a single hex of river is a dot, not a watercourse, and the
 * Python's own river drawing assumes a polyline.
 */
function maskRivers(rivers: Doc[], known: ReadonlySet<HexKey>): Doc[] {
  const out: Doc[] = [];

  for (const river of rivers) {
    const hexes = Array.isArray(river.hexes) ? (river.hexes as number[][]) : [];
    let run: number[][] = [];

    const flush = (): void => {
      if (run.length >= 2) out.push({ ...river, hexes: run });
      run = [];
    };

    for (const c of hexes) {
      if (known.has(coordKey(c[0], c[1]))) run.push(c);
      else flush();
    }
    flush();
  }

  return out;
}

/** Whether a parsed hex record is unobserved ground rather than real terrain. */
export const isFog = (tags: Iterable<string>): boolean => [...tags].includes(FOG_TAG);

/** Convenience: the set of keys for a list of hexes. */
export const keysOf = (coords: Iterable<Hex>): Set<HexKey> => {
  const out = new Set<HexKey>();
  for (const c of coords) out.add(key(c));
  return out;
};
