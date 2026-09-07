/**
 * The one place campaign state is turned into something a client may see.
 *
 * It lives in the shared engine rather than in the server because it is pure — state in,
 * a plain object out — and because the client has to name what it receives. A client that
 * declared its own copy of `ClientView` would be free to drift from what is actually
 * sent, and the shape of a fog boundary is the last thing that should be described twice.
 * Only the server ever calls it: nothing in the browser has ground truth to mask.
 *
 * This game is about incomplete information, so the fog is not a display convention — it
 * is the product. If a commander's browser receives everything and hides part of it,
 * anyone who opens the developer tools has the whole picture and the game is over.
 * Filtering happens here, before anything is serialised.
 *
 * Two rules make that enforceable rather than aspirational:
 *
 * 1. **Every read path goes through `viewFor`.** There is no other function that returns
 *    campaign state, and no route may reach for `state` directly. `test/leakage.test.ts`
 *    asserts against the *serialised bytes* of real responses, not against object graphs,
 *    because a `toJSON` or an accidentally-enumerable field leaks through a structural
 *    assertion that looks perfectly convincing.
 *
 * 2. **Redaction happens where data is built, not where it is drawn.** An enemy contact
 *    is constructed already stripped of what its intelligence grade does not earn, so a
 *    poor sighting never carries the corps identity in the first place.
 *
 * ## What is hidden, now that terrain fog is off
 *
 * The ground is accurate and public. The secrets are where the enemy is, and — the one
 * that carries the game — where your own detached formations are. A commander sees the
 * formation he rides with in full and everything else beneath him only as a dated report.
 */

import { formationsUnder, subordinates, type Commander } from './commander.js';
import { DEFAULT_CONFIG, type CampaignConfig } from './config.js';
import type { Faction } from './events.js';
import { key, type Hex, type HexKey } from './hex.js';
import { maskWorld } from './mask.js';
import { commanderVisible, spottedUnder, type Contact } from './recon.js';
import type { CampaignState } from './state.js';
import { echelonOf, type Echelon, type Formation, type Unit, type UnitKind } from './unit.js';
import { parseWorld, type World } from './world.js';

/** Who is asking. */
export type Role =
  | { readonly kind: 'referee' }
  | { readonly kind: 'commander'; readonly id: string };

export const REFEREE_ROLE: Role = { kind: 'referee' };
export const commanderRole = (id: string): Role => ({ kind: 'commander', id });

/** Public facts about a faction. Never the join token, which is not in state at all. */
export interface PublicFaction {
  readonly id: string;
  readonly name: string;
  readonly color: string;
}

/** Public facts about a commander: who he is, not what he knows. */
export interface PublicCommander {
  readonly id: string;
  readonly name: string;
  readonly faction: string;
  readonly unitId: string;
  readonly superiorId: string | null;
}

/**
 * A formation as its commander last heard of it.
 *
 * Deliberately not a `Unit`. A dated snapshot and a live record are different things, and
 * a client handed a `Unit` will draw it as though it were true now — which is exactly the
 * belief this design exists to deny. Everything here is what a despatch would carry.
 */
export interface UnitReport {
  readonly unitId: string;
  readonly name: string;
  readonly faction: string;
  /** Your own formation, so its arm and size are not in doubt — only its position is. */
  readonly kind: UnitKind;
  readonly echelon: Echelon;
  /** The hour the report describes, which is not the hour it arrived. */
  readonly atHours: number;
  readonly head: Hex;
  readonly effectives: number;
  readonly fatigue: number;
  readonly formation: Formation;
  readonly provisions: number;
  readonly corps: string | null;
}

export interface ClientView {
  readonly role: 'referee' | 'commander';
  readonly commander: PublicCommander | null;
  readonly campaign: {
    readonly id: string;
    readonly name: string;
    readonly clockHours: number;
    readonly seq: number;
  };
  readonly factions: readonly PublicFaction[];
  /** Commanders this role may know of: everyone on his own side, or all of them. */
  readonly commanders: readonly PublicCommander[];
  /** A `world.json` document. Masked only when `terrainFog` is on. */
  readonly world: unknown;
  /**
   * Formations held in full.
   *
   * A referee gets every unit on the map. A commander gets exactly one: the formation he
   * rides with, which is the only thing he can actually look at.
   */
  readonly units: readonly Unit[];
  /** Formations beneath him, as last reported. Empty for a referee, who has the units. */
  readonly reports: readonly UnitReport[];
  /** Enemies, as last seen and no better. Empty for a referee, who sees units instead. */
  readonly contacts: readonly Contact[];
  /** Ground his formations have covered. Terrain memory; says nothing about the enemy. */
  readonly surveyed: readonly HexKey[];
  /** What he can see from where he stands, right now. */
  readonly visible: readonly HexKey[];
}

const publicFaction = (f: Faction): PublicFaction => ({
  id: f.id,
  name: f.name,
  color: f.color,
});

const publicCommander = (c: Commander): PublicCommander => ({
  id: c.id,
  name: c.name,
  faction: c.faction,
  unitId: c.unitId,
  superiorId: c.superiorId,
});

/**
 * Snapshot a formation as of a given hour.
 *
 * Until riders exist this is taken at the current hour, which makes reports instantaneous
 * — the interim rule stated in `observe.ts`. The type and the plumbing are the finished
 * ones, so step 2 changes when the snapshot is captured and nothing else.
 */
export const reportOf = (unit: Unit, atHours: number): UnitReport => ({
  unitId: unit.id,
  name: unit.name,
  faction: unit.faction,
  kind: unit.kind,
  echelon: echelonOf(unit),
  atHours,
  head: unit.column[0] ?? { q: 0, r: 0 },
  effectives: unit.effectives,
  fatigue: unit.fatigue,
  formation: unit.formation,
  provisions: unit.provisions,
  corps: unit.corps,
});

export interface ViewInput {
  readonly campaignId: string;
  readonly state: CampaignState;
  /** The world as generated. Masked before sending only when `terrainFog` is on. */
  readonly worldDoc: unknown;
  readonly world: World;
  readonly cfg?: CampaignConfig;
}

/**
 * Everything, and only everything, this role is entitled to.
 *
 * A referee gets ground truth. A commander gets the formation he rides with, dated
 * reports of everything beneath him, contacts for the enemies his command has actually
 * spotted, and nothing else — in particular no record of an enemy nobody has observed and
 * no live position for any formation but his own.
 */
export function viewFor(input: ViewInput, role: Role): ClientView {
  const cfg = input.cfg ?? DEFAULT_CONFIG;
  const { state, world } = input;

  const campaign = {
    id: input.campaignId,
    name: state.name,
    clockHours: state.clockHours,
    seq: state.nextSeq,
  };
  const factions = [...state.factions.values()]
    .sort((a, b) => (a.id < b.id ? -1 : 1))
    .map(publicFaction);

  const byId = (a: { id: string }, b: { id: string }): number => (a.id < b.id ? -1 : 1);

  if (role.kind === 'referee') {
    return {
      role: 'referee',
      commander: null,
      campaign,
      factions,
      commanders: [...state.commanders.values()].map(publicCommander).sort(byId),
      world: input.worldDoc,
      units: [...state.units.values()].sort(byId),
      reports: [],
      contacts: [],
      surveyed: [],
      visible: [],
    };
  }

  const me = state.commanders.get(role.id);
  const surveyed = state.knowledge.get(role.id)?.surveyed ?? new Set<HexKey>();

  // A commander who has been removed — killed, captured, relieved — keeps a view so his
  // client does not crash mid-session, but it is empty of everything an appointment
  // carries. Failing open here would be the worst possible direction to fail.
  if (me === undefined) {
    return {
      role: 'commander',
      commander: null,
      campaign,
      factions,
      commanders: [],
      world: maskedWorld(input, cfg, new Set<HexKey>(), new Set<HexKey>(), role.id),
      units: [],
      reports: [],
      contacts: [],
      surveyed: [],
      visible: [],
    };
  }

  const visible = commanderVisible(state, world, cfg, role.id);
  const own = state.units.get(me.unitId);

  // Everything under him except his own formation, which he has in full. `formationsUnder`
  // includes it, so it is filtered out rather than sent twice in two different shapes.
  const reports = formationsUnder(state, role.id)
    .filter((u) => u.id !== me.unitId)
    .map((u) => reportOf(u, state.clockHours))
    .sort((a, b) => (a.unitId < b.unitId ? -1 : 1));

  // Contacts are built by `spottedUnder`, which constructs each one already stripped to
  // what the sighting earned. Nothing here has to remember to redact.
  const contacts = [...spottedUnder(state, world, cfg, role.id).values()].sort((a, b) =>
    a.unitId < b.unitId ? -1 : 1,
  );

  return {
    role: 'commander',
    commander: publicCommander(me),
    campaign,
    factions,
    // His own side's chain of command. Knowing who commands the enemy's II Corps is
    // intelligence, and it arrives by sighting or not at all.
    commanders: [...state.commanders.values()]
      .filter((c) => c.faction === me.faction)
      .map(publicCommander)
      .sort(byId),
    world: maskedWorld(input, cfg, surveyed, visible, me.faction),
    units: own === undefined ? [] : [own],
    reports,
    contacts,
    surveyed: [...surveyed].sort(),
    visible: [...visible].sort(),
  };
}

/**
 * The world document as this role should receive it.
 *
 * With `terrainFog` off — the default — everybody gets the same accurate map, which is
 * both the historical picture and what makes a two-hex sight radius playable. The masking
 * path is kept and still exercised by its own tests so that turning fog back on with the
 * issued map is a switch rather than a rebuild.
 */
function maskedWorld(
  input: ViewInput,
  cfg: CampaignConfig,
  surveyed: ReadonlySet<HexKey>,
  visible: ReadonlySet<HexKey>,
  faction: string,
): unknown {
  if (!cfg.terrainFog) return input.worldDoc;
  return maskWorld(input.worldDoc as Record<string, unknown>, {
    seen: surveyed,
    visible,
    faction,
    clockHours: input.state.clockHours,
  });
}

/**
 * The world alone, for the export endpoint.
 *
 * Goes through the same path as `viewFor` so the file a referee downloads and the map a
 * commander sees can never disagree.
 */
export function exportFor(input: ViewInput, role: Role): unknown {
  return viewFor(input, role).world;
}

/**
 * A cheap check that a view really is masked, used by the leakage tests and by the read
 * routes as a last line of defence.
 *
 * Belt and braces: `viewFor` is meant to guarantee these, and a bug in it is precisely
 * the kind that ships quietly, so the properties are asserted at the boundary as well as
 * in a test.
 */
export function assertMasked(view: ClientView, cfg: CampaignConfig = DEFAULT_CONFIG): void {
  if (view.role === 'referee') return;

  const faction = view.commander?.faction ?? null;

  // At most one live formation, and it must be the one he rides with. This is the
  // assertion that matters now that the ground is public: a second unit here is somebody
  // else's position leaking as fact rather than as a dated report.
  if (view.units.length > 1) {
    throw new Error(`leak: a commander was sent ${view.units.length} live formations`);
  }
  for (const u of view.units) {
    if (faction !== null && u.faction !== faction) {
      throw new Error(`leak: unit ${u.id} belongs to ${u.faction}, not ${faction}`);
    }
    if (view.commander !== null && u.id !== view.commander.unitId) {
      throw new Error(`leak: ${u.id} is not the formation ${view.commander.id} rides with`);
    }
  }

  for (const r of view.reports) {
    if (faction !== null && r.faction !== faction) {
      throw new Error(`leak: report on ${r.unitId}, which is ${r.faction}, not ${faction}`);
    }
  }

  for (const c of view.commanders) {
    if (faction !== null && c.faction !== faction) {
      throw new Error(`leak: ${c.id} commands for ${c.faction}, not ${faction}`);
    }
  }

  if (!cfg.terrainFog) return;

  const doc = view.world as { hexes?: { q: number; r: number; tags?: string[] }[] };
  const known = new Set(view.surveyed);
  for (const h of doc.hexes ?? []) {
    const k = key({ q: h.q, r: h.r });
    if (!(h.tags ?? []).includes('fog') && !known.has(k)) {
      throw new Error(
        `fog leak: hex ${k} is unmasked but ${view.commander?.id} has not surveyed it. ` +
          `This is a data-exposure bug, not a rendering one — refusing to serve.`,
      );
    }
  }
}

/** Reparse a view's world, for callers that want it as a `World` rather than a document. */
export const asWorld = (view: ClientView): World => parseWorld(view.world);

export { subordinates };
