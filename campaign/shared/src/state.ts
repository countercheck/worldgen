/**
 * Campaign state, and the fold that derives it from the log.
 *
 * `reduce` is pure, total and free of randomness: it never rejects an event and never
 * consults a generator. Everything that could refuse has already happened in `check`,
 * before the event existed. That is what makes replay trustworthy — a log always folds,
 * and always to the same thing.
 *
 * The log is deliberately *not* part of the state. State is derived; the log is the
 * record. Keeping the log out means `reduce` cannot accidentally read history it should
 * not, and the store stays free to page or truncate what it keeps in memory.
 */

import { key, type Hex, type HexKey } from './hex.js';
import type { Faction, LoggedEvent, WorldRef } from './events.js';
import type { Unit } from './unit.js';

/**
 * One faction's picture of the war.
 *
 * `seen` is memory and grows monotonically — the one exception being a referee's
 * `hexes_concealed`. What is visible *right now* is derived from unit positions each time
 * it is needed and is never stored, because a stored copy is a second fact that can
 * disagree with the units.
 */
export interface FactionKnowledge {
  readonly faction: string;
  readonly seen: ReadonlySet<HexKey>;
  /** Campaign hour each hex was last observed. Drives how stale a memory reads. */
  readonly lastSeenHours: ReadonlyMap<HexKey, number>;
}

export interface CampaignState {
  readonly name: string;
  readonly world: WorldRef;
  /** Seeds every roll, via `rngFor(seed, seq)`. */
  readonly seed: number;
  /** Hours since the scenario epoch. */
  readonly clockHours: number;
  /** The sequence number the next event will take. */
  readonly nextSeq: number;
  readonly factions: ReadonlyMap<string, Faction>;
  readonly units: ReadonlyMap<string, Unit>;
  readonly knowledge: ReadonlyMap<string, FactionKnowledge>;
}

const EMPTY_WORLD: WorldRef = {
  seed: 0,
  width: 0,
  height: 0,
  layout: 'axial',
  schemaVersion: '0',
  hash: '',
};

/**
 * The state before any event.
 *
 * Not usable as a campaign — `campaign_created` fills it in. It exists so `reduce` has a
 * total starting point and replay needs no special case for the first event.
 */
export const EMPTY_STATE: CampaignState = {
  name: '',
  world: EMPTY_WORLD,
  seed: 0,
  clockHours: 0,
  nextSeq: 0,
  factions: new Map(),
  units: new Map(),
  knowledge: new Map(),
};

const withUnit = (s: CampaignState, unit: Unit): CampaignState => ({
  ...s,
  units: new Map(s.units).set(unit.id, unit),
});

function knowledgeFor(s: CampaignState, faction: string): FactionKnowledge {
  return (
    s.knowledge.get(faction) ?? {
      faction,
      seen: new Set<HexKey>(),
      lastSeenHours: new Map<HexKey, number>(),
    }
  );
}

function withKnowledge(s: CampaignState, k: FactionKnowledge): CampaignState {
  return { ...s, knowledge: new Map(s.knowledge).set(k.faction, k) };
}

function observe(
  s: CampaignState,
  faction: string,
  coords: readonly Hex[],
  atHours: number,
): CampaignState {
  const k = knowledgeFor(s, faction);
  const seen = new Set(k.seen);
  const lastSeen = new Map(k.lastSeenHours);
  for (const c of coords) {
    const kk = key(c);
    seen.add(kk);
    lastSeen.set(kk, atHours);
  }
  return withKnowledge(s, { faction, seen, lastSeenHours: lastSeen });
}

function forget(s: CampaignState, faction: string, coords: readonly Hex[]): CampaignState {
  const k = knowledgeFor(s, faction);
  const seen = new Set(k.seen);
  const lastSeen = new Map(k.lastSeenHours);
  for (const c of coords) {
    const kk = key(c);
    seen.delete(kk);
    lastSeen.delete(kk);
  }
  return withKnowledge(s, { faction, seen, lastSeenHours: lastSeen });
}

/**
 * Fold one event into the state.
 *
 * Total by construction: every branch returns a state, and an event naming something that
 * has since vanished is ignored rather than throwing. A log that cannot be folded is a
 * campaign that cannot be opened, which is a far worse failure than an event that turns
 * out to be a no-op.
 */
export function reduce(state: CampaignState, event: LoggedEvent): CampaignState {
  const s: CampaignState = { ...state, nextSeq: Math.max(state.nextSeq, event.seq + 1) };
  const p = event.payload;

  switch (p.kind) {
    case 'campaign_created':
      return {
        ...s,
        name: p.name,
        world: p.world,
        seed: p.seed,
        clockHours: p.startHours,
      };

    case 'faction_added':
      return {
        ...s,
        factions: new Map(s.factions).set(p.faction.id, p.faction),
        knowledge: new Map(s.knowledge).set(p.faction.id, {
          faction: p.faction.id,
          seen: new Set(),
          lastSeenHours: new Map(),
        }),
      };

    case 'unit_added':
      return withUnit(s, p.unit);

    case 'unit_removed': {
      const units = new Map(s.units);
      units.delete(p.unitId);
      return { ...s, units };
    }

    case 'clock_advanced':
      return { ...s, clockHours: p.toHours };

    case 'unit_teleported': {
      const unit = s.units.get(p.unitId);
      if (unit === undefined) return s;
      return withUnit(s, { ...unit, column: p.column });
    }

    case 'hexes_revealed':
      return observe(s, p.faction, p.coords, event.clockHours);

    case 'hexes_concealed':
      return forget(s, p.faction, p.coords);

    case 'unit_stat_set': {
      const unit = s.units.get(p.unitId);
      if (unit === undefined) return s;
      // Spread only the keys actually present, so an absent field means "leave alone"
      // rather than "set to undefined" — the difference matters for `corps`, which is
      // legitimately null.
      const changes = Object.fromEntries(
        Object.entries(p.changes).filter(([, v]) => v !== undefined),
      );
      return withUnit(s, { ...unit, ...changes });
    }
  }
}

/** Fold a whole log. The only way a campaign is ever loaded. */
export const replay = (events: Iterable<LoggedEvent>, from = EMPTY_STATE): CampaignState => {
  let s = from;
  for (const e of events) s = reduce(s, e);
  return s;
};

/** Units belonging to one faction, in a stable order. */
export const unitsOf = (s: CampaignState, faction: string): Unit[] =>
  [...s.units.values()].filter((u) => u.faction === faction).sort((a, b) => (a.id < b.id ? -1 : 1));

/**
 * Factions in a fixed order.
 *
 * Sorted, not insertion-ordered: turn order and resolution order must not depend on the
 * order a referee happened to add factions in, for the same reason `CrossingStage` sorts
 * before it picks bridge sites.
 */
export const factionIds = (s: CampaignState): string[] => [...s.factions.keys()].sort();

/** Whether a faction has ever observed a hex. */
export const hasSeen = (s: CampaignState, faction: string, c: Hex): boolean =>
  s.knowledge.get(faction)?.seen.has(key(c)) ?? false;
