/**
 * A demonstration campaign, so the console has something in it on first load.
 *
 * Built through the real engine rather than by hand: every unit here is added by an
 * `add_unit` command that goes through `check` and `apply` like any other, so if the
 * rules would refuse one of these placements the console says so rather than showing a
 * unit the engine does not believe in.
 *
 * The units are chosen to show the things worth seeing. A guards cavalry division with
 * scouts is eighteen kilometres of column sweeping a two-hex corridor; a line infantry
 * division is two and a half and sees one hex. Putting them side by side is the quickest
 * way to understand why column length matters.
 */

import {
  advanceColumn,
  applyAll,
  distance,
  key,
  neighbors,
  unkey,
  DEFAULT_CONFIG,
  EMPTY_STATE,
  KIND_DEFAULTS,
  parseWorld,
  planMarch,
  type CampaignState,
  type Command,
  type Faction,
  type Hex,
  type Trait,
  type Unit,
  type UnitKind,
  type World,
  type WorldRef,
} from '@campaign/shared';

import worldDoc from '../../shared/test/fixtures/world-32x32.json';

export const RED: Faction = { id: 'red', name: 'Armée du Nord', color: '#d1495b' };
export const BLUE: Faction = { id: 'blue', name: 'Coalition', color: '#3d6fd1' };

interface Spec {
  id: string;
  faction: string;
  kind: UnitKind;
  name: string;
  effectives: number;
  experience: -2 | -1 | 0 | 1 | 2;
  traits: Trait[];
  spacingMultiplier: number;
  guns: number;
  corps: string | null;
  /** How far to march it from its start, so it has a column rather than a point. */
  marchTo?: (world: World, from: Hex) => Hex | null;
}

function makeUnit(spec: Spec, at: Hex): Unit {
  const defaults = KIND_DEFAULTS[spec.kind];
  return {
    id: spec.id,
    faction: spec.faction,
    kind: spec.kind,
    effectives: spec.effectives,
    fatigue: 0,
    experience: spec.experience,
    morale: [10, 20, 30, 40, 50][spec.experience + 2]!,
    provisions: 40,
    maxProvisions: 40,
    equipment: 30,
    maxEquipment: 30,
    guns: spec.guns,
    marchSpeedKmh: defaults.marchSpeedKmh,
    spacingM: defaults.spacingM,
    spacingMultiplier: spec.spacingMultiplier,
    traits: spec.traits,
    formation: 'march',
    column: [at],
    hoursMarchedToday: 0,
    corps: spec.corps,
  };
}

/** Human-readable names, kept beside the units rather than inside them. */
export const UNIT_NAMES: Record<string, string> = {};

const SPECS: Spec[] = [
  {
    id: 'red-1',
    faction: 'red',
    kind: 'infantry',
    name: '1re Division',
    effectives: 6400,
    experience: 0,
    traits: ['long_tail'],
    spacingMultiplier: 1.3,
    guns: 12,
    corps: 'I Corps',
  },
  {
    id: 'red-2',
    faction: 'red',
    kind: 'cavalry',
    name: 'Cuirassiers de la Garde',
    effectives: 4200,
    experience: 1,
    traits: ['scout', 'heavy'],
    spacingMultiplier: 1.5,
    guns: 6,
    corps: 'Cavalry Reserve',
  },
  {
    id: 'red-3',
    faction: 'red',
    kind: 'hq',
    name: "Quartier Général",
    effectives: 900,
    experience: 2,
    traits: ['fast'],
    spacingMultiplier: 1,
    guns: 0,
    corps: null,
  },
  {
    id: 'blue-1',
    faction: 'blue',
    kind: 'infantry',
    name: '3rd Division',
    effectives: 5200,
    experience: 1,
    traits: [],
    spacingMultiplier: 1.2,
    guns: 8,
    corps: 'II Corps',
  },
  {
    id: 'blue-2',
    faction: 'blue',
    kind: 'cavalry',
    name: 'Light Brigade',
    effectives: 4000,
    experience: 0,
    traits: ['scout'],
    spacingMultiplier: 1.4,
    guns: 0,
    corps: 'II Corps',
  },
];

/**
 * The largest connected body of dry land.
 *
 * Everything below picks from this rather than from every land hex, because a generated
 * world has islands and spits, and farthest-point sampling is drawn to exactly those:
 * the point furthest from everything else is usually a rock in the sea. A unit placed
 * there can march nowhere, which is how the first version of this produced a division
 * that would not move.
 */
function mainland(world: World): Hex[] {
  const land = new Set(
    [...world.hexes.values()].filter((h) => h.terrainClass === 'land').map((h) => key(h.coord)),
  );

  let best: Hex[] = [];
  const seen = new Set<string>();

  for (const start of land) {
    if (seen.has(start)) continue;
    const component: Hex[] = [];
    const queue = [unkey(start)];
    seen.add(start);

    while (queue.length > 0) {
      const c = queue.pop()!;
      component.push(c);
      for (const n of neighbors(c)) {
        const k = key(n);
        if (land.has(k) && !seen.has(k)) {
          seen.add(k);
          queue.push(n);
        }
      }
    }
    if (component.length > best.length) best = component;
  }

  return best.sort((a, b) => a.q - b.q || a.r - b.r);
}

/**
 * Starting positions spread across the land mass.
 *
 * Farthest-point sampling rather than every nth hex of a sorted list. Sorting by
 * coordinate and stepping through it puts every unit on roughly one line, and they then
 * march along roughly one route — which is what the first version of this did, and it
 * made a demo where five formations looked like one. Picking each start as far as
 * possible from those already chosen spreads them over the actual shape of the country.
 */
function landStarts(world: World, count: number): Hex[] {
  const land = mainland(world);
  if (land.length === 0) return [];

  const picked: Hex[] = [land[Math.floor(land.length / 2)]!];
  while (picked.length < count && picked.length < land.length) {
    let best: Hex | null = null;
    let bestDist = -1;
    for (const c of land) {
      const nearest = Math.min(...picked.map((p) => distance(p, c)));
      if (nearest > bestDist) {
        bestDist = nearest;
        best = c;
      }
    }
    if (best === null) break;
    picked.push(best);
  }
  return picked;
}

/**
 * Somewhere for a unit to have marched from, so its column is a line of real ground.
 *
 * Each unit gets its own objective — the land hex furthest from it in a direction
 * nobody else has taken — so the demo shows several routes over several kinds of
 * country rather than a queue along one road.
 */
function objectiveFor(land: readonly Hex[], from: Hex, taken: readonly Hex[]): Hex | null {
  let best: Hex | null = null;
  let bestScore = -Infinity;

  for (const c of land) {
    const reach = distance(from, c);
    if (reach < 6) continue;
    // Prefer a bearing no other unit is already marching on, but never let that rule out
    // every candidate — a demo that silently leaves a unit standing still is worse than
    // one where two routes run near each other.
    const clearance = taken.length === 0 ? 0 : Math.min(...taken.map((t) => distance(t, c)));
    const score = clearance * 2 + Math.min(reach, 20);
    if (score > bestScore) {
      bestScore = score;
      best = c;
    }
  }
  return best;
}

export interface Demo {
  readonly world: World;
  readonly worldDoc: unknown;
  readonly state: CampaignState;
  readonly names: Record<string, string>;
}

export function buildDemo(): Demo {
  const world = parseWorld(worldDoc);

  const ref: WorldRef = {
    seed: world.seed,
    width: world.width,
    height: world.height,
    layout: world.layout,
    schemaVersion: world.schemaVersion,
    hash: 'sha256:demo',
  };

  const land = mainland(world);
  const starts = landStarts(world, SPECS.length);
  const names: Record<string, string> = {};

  const commands: Command[] = [
    { kind: 'create_campaign', name: 'Demonstration', world: ref, seed: 20260906 },
    { kind: 'add_faction', faction: RED },
    { kind: 'add_faction', faction: BLUE },
  ];

  SPECS.forEach((spec, i) => {
    const at = starts[i] ?? starts[0]!;
    names[spec.id] = spec.name;
    commands.push({ kind: 'add_unit', unit: makeUnit(spec, at) });
  });

  const out = applyAll(commands, EMPTY_STATE, world, 'strict');
  if (!out.ok) {
    throw new Error(
      `the demo scenario is not legal: ${out.violations.map((v) => v.message).join('; ')}`,
    );
  }

  // Give each unit a marched path, so its column is a line of ground rather than a dot.
  // Done through the real router, so the tail follows a route the unit could actually
  // have taken — including going round water and preferring roads.
  let state = out.state;
  const units = new Map(state.units);
  const objectives: Hex[] = [];

  for (const unit of [...state.units.values()].sort((a, b) => (a.id < b.id ? -1 : 1))) {
    const start = unit.column[0]!;
    const goal = objectiveFor(land, start, objectives);
    if (goal === null) continue;

    const path = planMarch(world, DEFAULT_CONFIG, unit, goal);
    if (path === null || path.length < 2) continue;
    objectives.push(goal);

    let marched = unit;
    for (const step of path.slice(1, 26)) {
      marched = { ...marched, column: advanceColumn(marched, step) };
    }
    units.set(unit.id, marched);
  }
  state = { ...state, units };

  return { world, worldDoc, state, names };
}
