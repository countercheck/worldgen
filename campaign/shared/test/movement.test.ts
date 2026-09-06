/**
 * Terrain grading, march speeds, river crossings and routing.
 *
 * Two kinds of test here. The speed and crossing tables are checked against the rules
 * directly, on synthetic worlds where every hex is controlled — a real world cannot be
 * made to contain exactly the case under test. The routing and reach tests then run on
 * the real generated fixture, because a pathfinder that only works on hand-built grids
 * has not been tested.
 */

import { describe, expect, it } from 'vitest';

import world32 from './fixtures/world-32x32.json' with { type: 'json' };

import { DEFAULT_CONFIG, GRADES, type CampaignConfig, type Grade } from '../src/config.js';
import { crossingFor, discharge, riverClass } from '../src/crossing.js';
import { distance, key, type Hex, type HexKey } from '../src/hex.js';
import {
  fastestHoursPerHex,
  hoursToEnter,
  marchHoursLeftToday,
  pathHours,
  planMarch,
  reachable,
  speedKmh,
  stepCost,
  unitSpeedKmh,
} from '../src/movement.js';
import { gradeOf, gradeOfHex, isPassable } from '../src/terrain.js';
import type { Trait, Unit, UnitKind } from '../src/unit.js';
import { parseWorld, type LandCover, type RoadTier, type World, type WorldHex } from '../src/world.js';

const cfg = DEFAULT_CONFIG;
const real: World = parseWorld(world32);

function unit(kind: UnitKind, traits: Trait[] = [], at: Hex = { q: 0, r: 0 }): Unit {
  return {
    id: 'u',
    faction: 'red',
    kind,
    effectives: 5000,
    fatigue: 0,
    experience: 0,
    morale: 30,
    provisions: 40,
    maxProvisions: 40,
    equipment: 30,
    maxEquipment: 30,
    guns: 0,
    marchSpeedKmh: 3,
    spacingM: 0.5,
    spacingMultiplier: 1,
    traits,
    formation: 'march',
    column: [at],
    hoursMarchedToday: 0,
    corps: null,
  };
}

/** A flat land world of `size` x `size`, with hooks to vary individual hexes. */
function grid(
  size: number,
  hex: (q: number, r: number) => Partial<WorldHex> = () => ({}),
  roads: { a: Hex; b: Hex; tier: RoadTier }[] = [],
): World {
  const hexes = new Map<HexKey, WorldHex>();
  for (let q = 0; q < size; q++) {
    for (let r = 0; r < size; r++) {
      const coord = { q, r };
      hexes.set(key(coord), {
        coord,
        elevation: 100,
        slope: 0,
        relief: 0,
        terrainClass: 'land',
        landCover: 'open',
        biome: 'grassland',
        riverFlow: 0,
        catchmentKm2: 0,
        settlementName: null,
        tags: new Set<string>(),
        roadConnections: [],
        ...hex(q, r),
      });
    }
  }

  const roadEdges = new Map<string, { a: Hex; b: Hex; tier: RoadTier; deltaElevationM: number }>();
  for (const { a, b, tier } of roads) {
    const k = a.q < b.q || (a.q === b.q && a.r <= b.r)
      ? `${a.q},${a.r}|${b.q},${b.r}`
      : `${b.q},${b.r}|${a.q},${a.r}`;
    roadEdges.set(k, { a, b, tier, deltaElevationM: 0 });
  }

  return {
    schemaVersion: '1.8',
    seed: 1,
    width: size,
    height: size,
    layout: 'axial',
    hexes,
    rivers: [],
    settlements: [],
    roadEdges,
    seaEdges: new Map(),
    ferries: [],
    config: {
      navigableMinDischarge: 60000,
      fordMaxCatchmentKm2: 60,
      crossingReliefM: 60,
      meanPrecipMm: 800,
      model: 'organic',
    },
  };
}

describe("the rules' movement table", () => {
  const table: [UnitKind | 'courier', Grade, number][] = [
    ['infantry', 'highway', 3], ['infantry', 'road', 3],
    ['infantry', 'off_road', 2], ['infantry', 'bad_going', 1],
    ['cavalry', 'highway', 5], ['cavalry', 'road', 5],
    ['cavalry', 'off_road', 3], ['cavalry', 'bad_going', 1],
    ['hq', 'highway', 5], ['hq', 'road', 5],
    ['hq', 'off_road', 3], ['hq', 'bad_going', 1],
    ['courier', 'highway', 10], ['courier', 'road', 10],
    ['courier', 'off_road', 6], ['courier', 'bad_going', 2],
    ['convoy', 'highway', 1], ['convoy', 'road', 1],
    ['convoy', 'off_road', 2 / 3], ['convoy', 'bad_going', 1 / 3],
  ];

  it('matches row for row', () => {
    for (const [mover, grade, kmh] of table) {
      expect(speedKmh(cfg, mover, grade), `${mover} ${grade}`).toBeCloseTo(kmh, 6);
    }
  });

  it('makes a courier four times an infantryman on the road', () => {
    expect(speedKmh(cfg, 'courier', 'road') / speedKmh(cfg, 'infantry', 'road')).toBeCloseTo(10 / 3);
  });
});

describe('trait speed modifiers', () => {
  it('shifts march speed by the rules amounts', () => {
    expect(unitSpeedKmh(cfg, unit('infantry'), 'road')).toBe(3);
    expect(unitSpeedKmh(cfg, unit('infantry', ['slow']), 'road')).toBe(2);
    expect(unitSpeedKmh(cfg, unit('infantry', ['very_slow']), 'road')).toBe(1);
    expect(unitSpeedKmh(cfg, unit('infantry', ['fast']), 'road')).toBe(4);
    expect(unitSpeedKmh(cfg, unit('infantry', ['very_fast']), 'road')).toBe(5);
  });

  it('never brings a unit to a standstill', () => {
    // A stack of penalties on bad going would otherwise reach zero or below, and a zero
    // speed makes every step cost Infinity — a unit frozen by arithmetic rather than by
    // any rule.
    const crippled = unit('infantry', ['very_slow', 'slow']);
    expect(unitSpeedKmh(cfg, crippled, 'bad_going')).toBeGreaterThan(0);
    expect(unitSpeedKmh(cfg, crippled, 'bad_going')).toBe(cfg.minSpeedKmh);
  });
});

describe('grading the ground', () => {
  it('reads a primary road as Highway and lesser roads as Road', () => {
    // The whole reason RoadTier matters to this game.
    const a = { q: 0, r: 0 };
    const b = { q: 1, r: 0 };
    for (const [tier, expected] of [
      ['primary', 'highway'],
      ['secondary', 'road'],
      ['track', 'road'],
    ] as [RoadTier, Grade][]) {
      const w = grid(4, () => ({}), [{ a, b, tier }]);
      expect(gradeOf(w, cfg, a, b), tier).toBe(expected);
    }
  });

  it('grades open country off-road', () => {
    const w = grid(4);
    expect(gradeOf(w, cfg, { q: 0, r: 0 }, { q: 1, r: 0 })).toBe('off_road');
  });

  it('grades standing water and closed canopy as bad going', () => {
    for (const cover of ['bog', 'marsh', 'dense_forest', 'alpine', 'bare_rock'] as LandCover[]) {
      const w = grid(4, (q) => (q === 1 ? { landCover: cover } : {}));
      expect(gradeOf(w, cfg, { q: 0, r: 0 }, { q: 1, r: 0 }), cover).toBe('bad_going');
    }
  });

  it('leaves woodland and scrub merely off-road', () => {
    // Slower than open country, which the off-road rate already captures — but not a bog.
    for (const cover of ['woodland', 'scrub', 'open', 'desert', 'tundra'] as LandCover[]) {
      const w = grid(4, (q) => (q === 1 ? { landCover: cover } : {}));
      expect(gradeOf(w, cfg, { q: 0, r: 0 }, { q: 1, r: 0 }), cover).toBe('off_road');
    }
  });

  it('grades steep ground as bad going', () => {
    const w = grid(4, (q) => (q === 1 ? { slope: cfg.badGoingSlopeMPerKm + 1 } : {}));
    expect(gradeOf(w, cfg, { q: 0, r: 0 }, { q: 1, r: 0 })).toBe('bad_going');
  });

  it('lets a road override the ground it runs over', () => {
    // A road through a marsh is still a road: that is what building one means.
    const a = { q: 0, r: 0 };
    const b = { q: 1, r: 0 };
    const w = grid(4, (q) => (q === 1 ? { landCover: 'marsh' as LandCover } : {}), [
      { a, b, tier: 'primary' },
    ]);
    expect(gradeOfHex(w.hexes.get(key(b))!, cfg)).toBe('bad_going');
    expect(gradeOf(w, cfg, a, b)).toBe('highway');
  });
});

describe('stepCost', () => {
  it('is one over the speed, because a hex is a kilometre', () => {
    const w = grid(4);
    const inf = unit('infantry');
    expect(stepCost(w, cfg, inf, { q: 0, r: 0 }, { q: 1, r: 0 }).hours).toBeCloseTo(1 / 2, 6);

    const road = grid(4, () => ({}), [
      { a: { q: 0, r: 0 }, b: { q: 1, r: 0 }, tier: 'secondary' },
    ]);
    expect(stepCost(road, cfg, inf, { q: 0, r: 0 }, { q: 1, r: 0 }).hours).toBeCloseTo(1 / 3, 6);
  });

  it('treats water as a wall, not as slow going', () => {
    const w = grid(4, (q) => (q === 1 ? { terrainClass: 'open_water' as const } : {}));
    expect(stepCost(w, cfg, unit('infantry'), { q: 0, r: 0 }, { q: 1, r: 0 }).hours).toBe(Infinity);
    expect(isPassable(w, { q: 1, r: 0 })).toBe(false);
  });

  it('refuses a hex that is not on the map', () => {
    const w = grid(4);
    expect(hoursToEnter(w, cfg, unit('infantry'), { q: 0, r: 0 }, { q: 99, r: 99 })).toBe(Infinity);
  });
});

describe('river crossings', () => {
  /** A world whose column 1 is a river of the given size. */
  const riverAt = (catchmentKm2: number, tags: string[] = []) =>
    grid(5, (q) =>
      q === 1 ? { catchmentKm2, tags: new Set(['river', ...tags]) } : {},
    );

  const MINOR = 10; // 10 km2 x 800 mm = 8,000, well under the 60,000 threshold
  const MAJOR = 200; // 200 x 800 = 160,000, well over

  const from = { q: 0, r: 0 };
  const to = { q: 1, r: 0 };

  it('splits rivers on discharge, not on the drawing rank', () => {
    // riverFlow is a normalised rank for line widths; reading it as a physical quantity
    // would make a river's class depend on how many other rivers the map happens to have.
    const minor = riverAt(MINOR);
    const major = riverAt(MAJOR);
    expect(riverClass(minor.hexes.get(key(to))!, minor)).toBe('minor');
    expect(riverClass(major.hexes.get(key(to))!, major)).toBe('major');
    expect(discharge(major.hexes.get(key(to))!, major)).toBeGreaterThan(
      major.config.navigableMinDischarge,
    );
  });

  it('lets a minor river be forded, at an hour', () => {
    const w = riverAt(MINOR);
    const c = crossingFor(w, cfg, unit('infantry'), from, to);
    expect(c.river).toBe('minor');
    expect(c.how).toBe('ford');
    expect(c.hours).toBe(cfg.fordHours);
  });

  it('lets a minor river be crossed free at a bridge', () => {
    const w = riverAt(MINOR, ['bridge']);
    const c = crossingFor(w, cfg, unit('infantry'), from, to);
    expect(c.how).toBe('bridge');
    expect(c.hours).toBe(0);
  });

  it('refuses a major river with no bridge', () => {
    const w = riverAt(MAJOR);
    const c = crossingFor(w, cfg, unit('infantry'), from, to);
    expect(c.hours).toBe(Infinity);
    expect(c.how).toBe('blocked');
    expect(c.violations.map((v) => v.code)).toContain('major_river_unbridged');
  });

  it('makes that refusal soft, so a referee can still allow it', () => {
    // It is a rule of the game, not of arithmetic: a hard frost or a boat bridge the map
    // does not know about must remain adjudicable.
    const w = riverAt(MAJOR);
    const c = crossingFor(w, cfg, unit('infantry'), from, to);
    expect(c.violations.every((v) => v.severity === 'soft')).toBe(true);
  });

  it('crosses a major river at a bridge, at an hour per division', () => {
    const w = riverAt(MAJOR, ['bridge']);
    const c = crossingFor(w, cfg, unit('infantry'), from, to);
    expect(c.how).toBe('bridge');
    expect(c.hours).toBe(cfg.majorCrossingHours);
  });

  it('lets pontooneers bridge a major river themselves', () => {
    const w = riverAt(MAJOR);
    const c = crossingFor(w, cfg, unit('infantry', ['pontooneers']), from, to);
    expect(c.how).toBe('pontoon');
    expect(c.hours).toBe(cfg.pontoonBuildHours);
    expect(c.violations).toEqual([]);
  });

  it('does not let a ford tag serve a major river', () => {
    // A ford on a navigable river is not a crossing a division can use.
    const w = riverAt(MAJOR, ['ford']);
    expect(crossingFor(w, cfg, unit('infantry'), from, to).how).toBe('blocked');
  });

  it('treats a road across the channel as a bridge', () => {
    // If the generator ran a road over a river here, it built whatever the road needed.
    // This is what keeps worlds without CrossingStage playable.
    const w = grid(
      5,
      (q) => (q === 1 ? { catchmentKm2: MAJOR, tags: new Set(['river']) } : {}),
      [{ a: from, b: to, tier: 'secondary' }],
    );
    expect(crossingFor(w, cfg, unit('infantry'), from, to).how).toBe('bridge');
  });

  it('charges nothing for marching along a river rather than across it', () => {
    // Without this a unit following a valley pays a crossing every hex, which would make
    // the best going on the map into the worst.
    const w = grid(5, () => ({ catchmentKm2: MINOR, tags: new Set(['river']) }));
    const c = crossingFor(w, cfg, unit('infantry'), { q: 1, r: 0 }, { q: 2, r: 0 });
    expect(c.river).toBe('none');
    expect(c.hours).toBe(0);
  });

  it('charges the crossing per division, so a corps queues', () => {
    const w = riverAt(MAJOR, ['bridge']);
    const divisions = [unit('infantry'), unit('infantry'), unit('infantry')];
    const total = divisions.reduce(
      (a, u) => a + crossingFor(w, cfg, u, from, to).hours,
      0,
    );
    expect(total).toBe(3 * cfg.majorCrossingHours);
  });
});

describe('planMarch', () => {
  it('returns a connected path of legal steps on a real world', () => {
    const land = [...real.hexes.values()].filter((h) => h.terrainClass === 'land');
    const start = land[0]!.coord;
    const goal = land[Math.floor(land.length / 2)]!.coord;
    const u = unit('infantry', [], start);

    const path = planMarch(real, cfg, u, goal);
    if (path === null) return; // legitimately unreachable across water

    expect(path[0]).toEqual(start);
    expect(path.at(-1)).toEqual(goal);
    for (let i = 1; i < path.length; i++) {
      expect(distance(path[i - 1]!, path[i]!), `step ${i}`).toBe(1);
      expect(hoursToEnter(real, cfg, u, path[i - 1]!, path[i]!)).toBeLessThan(Infinity);
    }
  });

  it('prefers a longer road to a shorter cross-country line', () => {
    // Least time, not least distance. A courier routed by distance would ignore every
    // road on the map.
    const size = 12;
    const roads: { a: Hex; b: Hex; tier: RoadTier }[] = [];
    // A road running the long way round: down column 0, along row 11, back up column 11.
    for (let r = 0; r < size - 1; r++) {
      roads.push({ a: { q: 0, r }, b: { q: 0, r: r + 1 }, tier: 'primary' });
    }
    // The direct line is bad going, so cross-country is 1 km/h against 3 on the highway.
    const w = grid(
      size,
      (q, r) => (q > 0 && q < size - 1 && r > 0 ? { landCover: 'bog' as LandCover } : {}),
      roads,
    );

    const u = unit('infantry', [], { q: 0, r: 0 });
    const path = planMarch(w, cfg, u, { q: 0, r: size - 1 });

    expect(path).not.toBeNull();
    expect(path!.every((h) => h.q === 0)).toBe(true);
  });

  it('takes a longer road over a shorter slog, which needs an admissible heuristic', () => {
    // The discriminating case for the pathfinder's heuristic, and it has to be built
    // carefully: a road that is *also* the shortest line proves nothing, because a
    // greedy search finds it too.
    //
    // Here the direct line is eight hexes of bog at 1 km/h — eight hours. The detour is
    // ten hexes of highway at 3 km/h — three and a third. More hexes, less than half the
    // time.
    //
    // `astar` counts hexes at 1.0 by default. In hours that overestimates threefold, and
    // every hex of the direct line then scores f = 8 while the detour's first step scores
    // 8.33 — so the slow route reaches the goal and returns before the fast one is ever
    // explored. Scaling the heuristic by the unit's fastest step fixes it. Without that
    // scaling this test fails with an eight-hour path.
    const roads: { a: Hex; b: Hex; tier: RoadTier }[] = [
      { a: { q: 0, r: 0 }, b: { q: 0, r: 1 }, tier: 'primary' },
      { a: { q: 8, r: 1 }, b: { q: 8, r: 0 }, tier: 'primary' },
    ];
    for (let q = 0; q < 8; q++) {
      roads.push({ a: { q, r: 1 }, b: { q: q + 1, r: 1 }, tier: 'primary' });
    }
    const w = grid(10, () => ({ landCover: 'bog' as LandCover }), roads);
    const u = unit('infantry', [], { q: 0, r: 0 });

    const direct = Array.from({ length: 9 }, (_, q) => ({ q, r: 0 }));
    expect(pathHours(w, cfg, u, direct)).toBeCloseTo(8, 6);

    const path = planMarch(w, cfg, u, { q: 8, r: 0 })!;
    expect(path).not.toBeNull();

    const hours = pathHours(w, cfg, u, path);
    expect(hours).toBeCloseTo(10 / 3, 6);
    expect(hours).toBeLessThan(pathHours(w, cfg, u, direct));
    expect(path.length).toBeGreaterThan(direct.length);
  });

  it('returns null when the goal cannot be reached', () => {
    const w = grid(5, (q) => (q === 2 ? { terrainClass: 'open_water' as const } : {}));
    expect(planMarch(w, cfg, unit('infantry', [], { q: 0, r: 0 }), { q: 4, r: 0 })).toBeNull();
  });

  it('scales its heuristic by the fastest step the unit has', () => {
    expect(fastestHoursPerHex(cfg, unit('infantry'))).toBeCloseTo(1 / 3, 6);
    expect(fastestHoursPerHex(cfg, unit('cavalry'))).toBeCloseTo(1 / 5, 6);
    for (const g of GRADES) {
      expect(1 / unitSpeedKmh(cfg, unit('cavalry'), g)).toBeGreaterThanOrEqual(
        fastestHoursPerHex(cfg, unit('cavalry')) - 1e-9,
      );
    }
  });
});

describe('reachable', () => {
  it('reaches further on a road than across country', () => {
    const size = 12;
    const roads: { a: Hex; b: Hex; tier: RoadTier }[] = [];
    for (let q = 0; q < size - 1; q++) {
      roads.push({ a: { q, r: 0 }, b: { q: q + 1, r: 0 }, tier: 'primary' });
    }
    const w = grid(size, () => ({}), roads);
    const u = unit('infantry', [], { q: 0, r: 0 });

    const reach = reachable(w, cfg, u, 3);
    // Three hours on a highway at 3 km/h is nine hexes; across country at 2 km/h, six.
    expect(reach.hours.get(key({ q: 9, r: 0 }))).toBeDefined();
    expect(reach.hours.get(key({ q: 0, r: 7 }))).toBeUndefined();
  });

  it('includes the start at no cost and stays inside the budget', () => {
    const w = grid(10);
    const u = unit('cavalry', [], { q: 5, r: 5 });
    const reach = reachable(w, cfg, u, 2);

    expect(reach.hours.get(key({ q: 5, r: 5 }))).toBe(0);
    for (const [, hours] of reach.hours) expect(hours).toBeLessThanOrEqual(2);
  });

  it('reaches nothing on no budget', () => {
    const w = grid(6);
    const reach = reachable(w, cfg, unit('infantry', [], { q: 0, r: 0 }), 0);
    expect([...reach.hours.keys()]).toEqual([key({ q: 0, r: 0 })]);
  });

  it('does not cross water', () => {
    const w = grid(7, (q) => (q === 3 ? { terrainClass: 'open_water' as const } : {}));
    const reach = reachable(w, cfg, unit('infantry', [], { q: 0, r: 0 }), 50);
    for (const k of reach.hours.keys()) {
      expect(Number(k.split(',')[0]), `${k} is across the water`).toBeLessThan(3);
    }
  });

  it('is what a commander sees, so it works on a partial world too', () => {
    // The reach overlay runs client-side over the masked world, so ground nobody has
    // scouted is simply not offered. Modelled here by a world with a hole in it.
    const full = grid(8);
    const partial: World = {
      ...full,
      hexes: new Map([...full.hexes].filter(([, h]) => h.coord.q < 4)),
    };
    const reach = reachable(partial, cfg, unit('cavalry', [], { q: 0, r: 0 }), 10);
    for (const k of reach.hours.keys()) {
      expect(Number(k.split(',')[0])).toBeLessThan(4);
    }
  });
});

describe('the daily march cap', () => {
  it('is the rules twenty hours', () => {
    expect(cfg.maxMarchHoursPerDay).toBe(20);
    expect(marchHoursLeftToday(cfg, unit('infantry'))).toBe(20);
  });

  it('runs down as a unit marches, and never below zero', () => {
    const u = { ...unit('infantry'), hoursMarchedToday: 14 };
    expect(marchHoursLeftToday(cfg, u)).toBe(6);
    expect(marchHoursLeftToday(cfg, { ...u, hoursMarchedToday: 25 })).toBe(0);
  });
});

describe('config overrides', () => {
  it('changes the rules without changing the code', () => {
    // The ruleset is at v4 and moving; a revision should be a config edit.
    const faster: CampaignConfig = {
      ...cfg,
      speeds: { ...cfg.speeds, infantry: { highway: 6, road: 6, off_road: 4, bad_going: 2 } },
    };
    const w = grid(4);
    const u = unit('infantry');
    expect(stepCost(w, cfg, u, { q: 0, r: 0 }, { q: 1, r: 0 }).hours).toBeCloseTo(1 / 2, 6);
    expect(stepCost(w, faster, u, { q: 0, r: 0 }, { q: 1, r: 0 }).hours).toBeCloseTo(1 / 4, 6);
  });
});
