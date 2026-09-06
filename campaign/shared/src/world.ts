/**
 * The generated world, as the campaign layer sees it.
 *
 * This parses `world.json` from `worldgen`'s `WorldState.to_dict()` — schema 1.8 at the
 * time of writing. The world is strictly read-only here: nothing in the campaign layer
 * ever writes a hex back. Play state lives entirely separately, which is what lets a
 * world stay the reproducible artefact the generator promises.
 *
 * The parser validates rather than casts. A `world.json` arrives over an upload endpoint
 * from whoever is refereeing, so it is untrusted input, and a missing field surfacing as
 * `undefined` three layers into a movement calculation is far worse than a clear error at
 * the door.
 */

import { AXIAL, key, OFFSET, type Hex, type HexKey, type Layout } from './hex.js';

/**
 * Schema versions this parser accepts.
 *
 * Mirrors `SUPPORTED_SCHEMA_VERSIONS` in `worldgen/core/world_state.py`. The generator
 * migrates older files on load, so it accepts a wider range than we do — we only ever
 * see what it wrote most recently. Widen this deliberately, having checked that the
 * fields below actually survive the older shape.
 */
export const SUPPORTED_SCHEMA_VERSIONS = new Set(['1.8']);

export type TerrainClass = 'open_water' | 'inland_water' | 'coast' | 'land';

export type LandCover =
  | 'open_water'
  | 'bog'
  | 'marsh'
  | 'dense_forest'
  | 'woodland'
  | 'scrub'
  | 'open'
  | 'tundra'
  | 'desert'
  | 'alpine'
  | 'bare_rock';

export type RoadTier = 'primary' | 'secondary' | 'track';

/** Draw and precedence order, mirroring `ROAD_TIER_RANK`. Higher wins a shared edge. */
export const ROAD_TIER_RANK: Readonly<Record<RoadTier, number>> = {
  track: 0,
  secondary: 1,
  primary: 2,
};

/** Hex tags the campaign rules care about. The generator writes others; these we read. */
export const TAG_FORD = 'ford';
export const TAG_BRIDGE = 'bridge';
export const TAG_RIVER = 'river';

export interface WorldHex {
  readonly coord: Hex;
  readonly elevation: number;
  readonly slope: number;
  readonly relief: number;
  readonly terrainClass: TerrainClass;
  readonly landCover: LandCover | null;
  readonly biome: string | null;
  readonly riverFlow: number;
  /** Physical upstream area. The river-size measure; `riverFlow` is a drawing rank. */
  readonly catchmentKm2: number;
  readonly settlementName: string | null;
  readonly tags: ReadonlySet<string>;
  readonly roadConnections: readonly Hex[];
}

export interface RoadEdge {
  readonly a: Hex;
  readonly b: Hex;
  readonly tier: RoadTier;
  /** Elevation gained from `a` to `b`. The on-road grade term, already computed. */
  readonly deltaElevationM: number;
}

export interface River {
  readonly hexes: readonly Hex[];
  readonly flowVolume: number;
}

export interface Settlement {
  readonly coord: Hex;
  readonly tier: string;
  readonly role: string;
  readonly population: number;
  readonly name: string;
}

export interface Ferry {
  readonly a: Hex;
  readonly b: Hex;
}

/**
 * The subset of `WorldConfig` the campaign rules read.
 *
 * These arrive inside `metadata.config`, which matters: the river rules must use the
 * thresholds the world was *generated* with, not today's defaults. A world generated
 * before a threshold moved would otherwise have its rivers silently reclassified.
 */
export interface WorldConfigSubset {
  /** Discharge at which a watercourse floats a barge — the Major/Minor river line. */
  readonly navigableMinDischarge: number;
  readonly fordMaxCatchmentKm2: number;
  readonly crossingReliefM: number;
  readonly meanPrecipMm: number;
  readonly model: string;
}

export interface World {
  readonly schemaVersion: string;
  readonly seed: number;
  readonly width: number;
  readonly height: number;
  readonly layout: Layout;
  readonly hexes: ReadonlyMap<HexKey, WorldHex>;
  readonly rivers: readonly River[];
  readonly settlements: readonly Settlement[];
  /** Keyed by `edgeKey(a, b)`, so a lookup does not care which way a unit is moving. */
  readonly roadEdges: ReadonlyMap<string, RoadEdge>;
  readonly seaEdges: ReadonlyMap<string, RoadEdge>;
  readonly ferries: readonly Ferry[];
  readonly config: WorldConfigSubset;
}

/**
 * The canonical undirected key for an edge between two hexes.
 *
 * Mirrors `road_edge_key` in `world_state.py`: the pair is sorted, so an edge has one
 * key however it is approached. Movement asks about the edge it is crossing without
 * knowing which end the generator wrote first.
 */
export function edgeKey(a: Hex, b: Hex): string {
  return a.q < b.q || (a.q === b.q && a.r <= b.r)
    ? `${a.q},${a.r}|${b.q},${b.r}`
    : `${b.q},${b.r}|${a.q},${a.r}`;
}

export class WorldParseError extends Error {
  override readonly name = 'WorldParseError';
}

const fail = (msg: string): never => {
  throw new WorldParseError(msg);
};

function num(v: unknown, where: string): number {
  if (typeof v !== 'number' || !Number.isFinite(v)) {
    fail(`${where}: expected a finite number, got ${JSON.stringify(v)}`);
  }
  return v as number;
}

function str(v: unknown, where: string): string {
  if (typeof v !== 'string') fail(`${where}: expected a string, got ${JSON.stringify(v)}`);
  return v as string;
}

function coord(v: unknown, where: string): Hex {
  if (!Array.isArray(v) || v.length !== 2) {
    fail(`${where}: expected a [q, r] pair, got ${JSON.stringify(v)}`);
  }
  const arr = v as unknown[];
  return { q: num(arr[0], `${where}.q`), r: num(arr[1], `${where}.r`) };
}

function oneOf<T extends string>(v: unknown, allowed: readonly T[], where: string): T {
  const s = str(v, where);
  if (!allowed.includes(s as T)) {
    fail(`${where}: ${JSON.stringify(s)} is not one of ${allowed.join(', ')}`);
  }
  return s as T;
}

const TERRAIN_CLASSES = ['open_water', 'inland_water', 'coast', 'land'] as const;
const ROAD_TIERS = ['primary', 'secondary', 'track'] as const;
const LAND_COVERS = [
  'open_water', 'bog', 'marsh', 'dense_forest', 'woodland', 'scrub',
  'open', 'tundra', 'desert', 'alpine', 'bare_rock',
] as const;

function parseEdges(raw: unknown, where: string): Map<string, RoadEdge> {
  const out = new Map<string, RoadEdge>();
  if (raw === undefined || raw === null) return out;
  if (!Array.isArray(raw)) fail(`${where}: expected an array`);

  for (const [i, entry] of (raw as Record<string, unknown>[]).entries()) {
    const a = coord(entry.a, `${where}[${i}].a`);
    const b = coord(entry.b, `${where}[${i}].b`);
    out.set(edgeKey(a, b), {
      a,
      b,
      tier: oneOf(entry.tier, ROAD_TIERS, `${where}[${i}].tier`),
      deltaElevationM: num(entry.delta_elevation_m, `${where}[${i}].delta_elevation_m`),
    });
  }
  return out;
}

/** Parse a `world.json` document. Throws `WorldParseError` on anything malformed. */
export function parseWorld(raw: unknown): World {
  if (typeof raw !== 'object' || raw === null) fail('world.json: expected an object');
  const d = raw as Record<string, unknown>;

  const version = str(d.version, 'version');
  if (!SUPPORTED_SCHEMA_VERSIONS.has(version)) {
    fail(
      `world.json schema ${version} is not supported — this build reads ` +
        `${[...SUPPORTED_SCHEMA_VERSIONS].join(', ')}. Regenerate the world, or widen ` +
        `SUPPORTED_SCHEMA_VERSIONS once the fields campaign/ reads are known to survive.`,
    );
  }

  const layout = oneOf(d.layout ?? AXIAL, [AXIAL, OFFSET] as const, 'layout');

  if (!Array.isArray(d.hexes)) fail('hexes: expected an array');
  const hexes = new Map<HexKey, WorldHex>();
  for (const [i, entry] of (d.hexes as Record<string, unknown>[]).entries()) {
    const c: Hex = { q: num(entry.q, `hexes[${i}].q`), r: num(entry.r, `hexes[${i}].r`) };
    const tags = Array.isArray(entry.tags) ? (entry.tags as unknown[]).map(String) : [];
    const roads = Array.isArray(entry.road_connections)
      ? (entry.road_connections as unknown[]).map((rc, j) =>
          coord(rc, `hexes[${i}].road_connections[${j}]`),
        )
      : [];

    hexes.set(key(c), {
      coord: c,
      elevation: num(entry.elevation, `hexes[${i}].elevation`),
      slope: num(entry.slope ?? 0, `hexes[${i}].slope`),
      relief: num(entry.relief ?? 0, `hexes[${i}].relief`),
      terrainClass: oneOf(entry.terrain_class, TERRAIN_CLASSES, `hexes[${i}].terrain_class`),
      landCover:
        entry.land_cover == null
          ? null
          : oneOf(entry.land_cover, LAND_COVERS, `hexes[${i}].land_cover`),
      biome: entry.biome == null ? null : str(entry.biome, `hexes[${i}].biome`),
      riverFlow: num(entry.river_flow ?? 0, `hexes[${i}].river_flow`),
      catchmentKm2: num(entry.catchment_km2 ?? 0, `hexes[${i}].catchment_km2`),
      settlementName: null,
      tags: new Set(tags),
      roadConnections: roads,
    });
  }

  const settlements: Settlement[] = Array.isArray(d.settlements)
    ? (d.settlements as Record<string, unknown>[]).map((s, i) => ({
        coord: coord(s.coord, `settlements[${i}].coord`),
        tier: str(s.tier, `settlements[${i}].tier`),
        role: str(s.role, `settlements[${i}].role`),
        population: num(s.population, `settlements[${i}].population`),
        name: str(s.name, `settlements[${i}].name`),
      }))
    : [];

  // The generator stores a settlement on the hex as well as in the list; the JSON only
  // carries the list, so the back-reference is rebuilt here. Movement and recon both ask
  // "is there a town on this hex" far more often than they walk the settlement list.
  for (const s of settlements) {
    const h = hexes.get(key(s.coord));
    if (h !== undefined) {
      hexes.set(key(s.coord), { ...h, settlementName: s.name });
    }
  }

  const rivers: River[] = Array.isArray(d.rivers)
    ? (d.rivers as Record<string, unknown>[]).map((r, i) => ({
        hexes: Array.isArray(r.hexes)
          ? (r.hexes as unknown[]).map((h, j) => coord(h, `rivers[${i}].hexes[${j}]`))
          : [],
        flowVolume: num(r.flow_volume ?? 0, `rivers[${i}].flow_volume`),
      }))
    : [];

  const ferries: Ferry[] = Array.isArray(d.ferries)
    ? (d.ferries as Record<string, unknown>[]).map((f, i) => ({
        a: coord(f.a, `ferries[${i}].a`),
        b: coord(f.b, `ferries[${i}].b`),
      }))
    : [];

  const meta = (d.metadata ?? {}) as Record<string, unknown>;
  const cfg = (meta.config ?? {}) as Record<string, unknown>;

  return {
    schemaVersion: version,
    seed: num(d.seed, 'seed'),
    width: num(d.width, 'width'),
    height: num(d.height, 'height'),
    layout,
    hexes,
    rivers,
    settlements,
    roadEdges: parseEdges(d.road_edges, 'road_edges'),
    seaEdges: parseEdges(d.sea_edges, 'sea_edges'),
    ferries,
    config: {
      // Defaults mirror worldgen/core/config.py. A world generated before a field
      // existed still has to yield a usable cost model, so these fall back rather than
      // throwing — unlike the hex data above, which cannot be guessed.
      navigableMinDischarge: typeof cfg.navigable_min_discharge === 'number'
        ? cfg.navigable_min_discharge
        : 60000,
      fordMaxCatchmentKm2: typeof cfg.ford_max_catchment_km2 === 'number'
        ? cfg.ford_max_catchment_km2
        : 60,
      crossingReliefM: typeof cfg.crossing_relief_m === 'number' ? cfg.crossing_relief_m : 60,
      meanPrecipMm: typeof cfg.mean_precip_mm === 'number' ? cfg.mean_precip_mm : 800,
      model: typeof cfg.model === 'string' ? cfg.model : 'classic',
    },
  };
}

/** Look a hex up by coordinate. */
export const hexAt = (w: World, c: Hex): WorldHex | undefined => w.hexes.get(key(c));

/** Whether a hex is water, and so impassable to a land unit. */
export const isWater = (h: WorldHex): boolean =>
  h.terrainClass === 'open_water' || h.terrainClass === 'inland_water';

/** The road edge joining two hexes, if the generator built one. */
export const roadBetween = (w: World, a: Hex, b: Hex): RoadEdge | undefined =>
  w.roadEdges.get(edgeKey(a, b));
