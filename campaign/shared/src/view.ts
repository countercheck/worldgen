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
 * formation they ride with in full and everything else beneath them only as a dated report.
 */

import { formationsUnder, subordinates, type Commander } from './commander.js';
import { DEFAULT_CONFIG, DEFAULT_RULESET, type CampaignConfig } from './config.js';
import {
  addresseeCopy,
  captorCopy,
  isSuperseded,
  senderCopy,
  type CapturedDespatch,
  type Despatch,
  type ReceivedDespatch,
  type SentDespatch,
} from './despatch.js';
import type { Faction, LoggedEvent } from './events.js';
import { key, type HexKey } from './hex.js';
import { maskWorld } from './mask.js';
import { commanderVisible, publicContact, type PublicContact } from './recon.js';
import {
  capturedBy,
  contactsOf,
  despatchesFrom,
  inboxOf,
  type CampaignState,
} from './state.js';
import type { PendingDecision, Task } from './task.js';
import type { Unit, UnitReport } from './unit.js';
import { parseWorld, projectWorld, type World } from './world.js';

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

/** Public facts about a commander: who they are, not what they know. */
export interface PublicCommander {
  readonly id: string;
  readonly name: string;
  readonly faction: string;
  readonly unitId: string;
  /**
   * The name of that formation: "1re Division".
   *
   * Sent rather than looked up, because a commander's view carries exactly one live unit —
   * their own — and would otherwise have no way to say whose formation a lateral peer
   * rides with. It leaks nothing: which formation an officer commands is the order of
   * battle, known to every staff on their own side, and a name is not a position. Their
   * *side's* order of battle only — `commanders` is already filtered to their faction, so
   * the enemy's establishment still arrives by sighting or not at all.
   */
  readonly unitName: string;
  readonly superiorId: string | null;
}

export interface ClientView {
  readonly role: 'referee' | 'commander';
  readonly commander: PublicCommander | null;
  readonly campaign: {
    readonly id: string;
    readonly name: string;
    readonly clockHours: number;
    readonly seq: number;
    /** Which named set of rules this campaign is played under. */
    readonly ruleset: string;
  };
  /**
   * The numbers this campaign actually runs on.
   *
   * Sent, not assumed. The console works out reach, march rates, column lengths and what a
   * patrol costs, and a client computing those off its own bundled defaults would quietly
   * disagree with the server the moment a campaign was started under anything but the
   * standard rules. Not secret: both sides play by the same table and are entitled to read
   * it, which is what makes it safe to send to a commander.
   */
  readonly config: CampaignConfig;
  readonly factions: readonly PublicFaction[];
  /** Commanders this role may know of: everyone on their own side, or all of them. */
  readonly commanders: readonly PublicCommander[];
  /** A `world.json` document. Masked only when `terrainFog` is on. */
  readonly world: unknown;
  /**
   * Formations held in full.
   *
   * A referee gets every unit on the map. A commander gets exactly one: the formation they
   * ride with, which is the only thing they can actually look at.
   */
  readonly units: readonly Unit[];
  /** Formations beneath them, as last reported. Empty for a referee, who has the units. */
  readonly reports: readonly UnitReport[];
  /**
   * Enemies, as last heard of and no better. Empty for a referee, who sees units instead.
   *
   * Each carries their own label and never the observed unit's id — a commander who had that
   * could correlate two sightings hours apart for free, which these rules make them buy
   * with a patrol.
   */
  readonly contacts: readonly PublicContact[];
  /** Ground their formations have covered. Terrain memory; says nothing about the enemy. */
  readonly surveyed: readonly HexKey[];
  /** What they can see from where they stand, right now. */
  readonly visible: readonly HexKey[];
  /**
   * Ground being fought over.
   *
   * A referee sees every battlefield. A commander sees only the ones on ground they can
   * currently observe — gunfire carries, but this is the campaign map, and a battle they
   * cannot see is a battle they have to be told about by a rider like anything else.
   */
  readonly battle: readonly HexKey[];

  // ---- the post -------------------------------------------------------
  /**
   * What they have written, as they may see it: no route, and no fate.
   *
   * The absence of a fate is the mechanic rather than an omission. A commander who could
   * see that their rider had been taken would know their order never arrived, and no
   * commander in 1815 knew that without being told. Empty for a referee, who has the
   * despatches themselves.
   */
  readonly sent: readonly SentDespatch[];
  /** What is actually in their hand. Delivered only — nothing in transit toward them. */
  readonly received: readonly ReceivedDespatch[];
  /** Enemy paper their side has taken off a rider. */
  readonly captured: readonly CapturedDespatch[];
  /** What the formation they ride with is doing. They set out on it; they know. */
  readonly task: Task | null;

  // ---- the referee's, and only their ------------------------------------
  /** Every despatch in the campaign, routes and fates and all. Empty for a commander. */
  readonly despatches: readonly Despatch[];
  /** The queue of things somebody has to decide. Empty for a commander. */
  readonly decisions: readonly PendingDecision[];
  /** What every formation is doing. Empty for a commander, who has `task`. */
  readonly tasks: readonly Task[];
}

const publicFaction = (f: Faction): PublicFaction => ({
  id: f.id,
  name: f.name,
  color: f.color,
});

const publicCommander = (
  c: Commander,
  units: ReadonlyMap<string, Unit>,
): PublicCommander => ({
  id: c.id,
  name: c.name,
  faction: c.faction,
  unitId: c.unitId,
  // The id is a poor name and a good fallback: a commander riding with a formation that
  // has just been removed should still be legible in an inbox rather than blank.
  unitName: units.get(c.unitId)?.name ?? c.unitId,
  superiorId: c.superiorId,
});

export interface ViewInput {
  readonly campaignId: string;
  readonly state: CampaignState;
  /** The world as generated. Masked before sending only when `terrainFog` is on. */
  readonly worldDoc: unknown;
  readonly world: World;
  readonly cfg?: CampaignConfig;
  /** The named set of rules this campaign was started under. */
  readonly ruleset?: string;
}

/**
 * Everything, and only everything, this role is entitled to.
 *
 * A referee gets ground truth. A commander gets the formation they ride with, dated
 * reports of everything beneath them, contacts for the enemies their command has actually
 * spotted, and nothing else — in particular no record of an enemy nobody has observed and
 * no live position for any formation but their own.
 */
export function viewFor(input: ViewInput, role: Role): ClientView {
  const cfg = input.cfg ?? DEFAULT_CONFIG;
  const { state, world } = input;

  const campaign = {
    id: input.campaignId,
    name: state.name,
    clockHours: state.clockHours,
    seq: state.nextSeq,
    ruleset: input.ruleset ?? DEFAULT_RULESET,
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
      config: cfg,
      factions,
      commanders: [...state.commanders.values()]
        .map((c) => publicCommander(c, state.units))
        .sort(byId),
      // Projected, like every other path out of this function. A referee sees the whole
      // map and no masking applies to them, which is exactly why this line read
      // `input.worldDoc` and quietly sent twelve fields nobody reads — the saving was
      // real for commanders and absent for the one role that loads the map most.
      world: projectWorld(input.worldDoc),
      units: [...state.units.values()].sort(byId),
      reports: [],
      contacts: [],
      surveyed: [],
      visible: [],
      battle: [...state.battle].sort(),
      sent: [],
      received: [],
      captured: [],
      task: null,
      despatches: [...state.despatches.values()].sort(
        (a, b) => a.sentAtHours - b.sentAtHours || (a.id < b.id ? -1 : 1),
      ),
      decisions: [...state.decisions.values()].sort(
        (a, b) => a.atHours - b.atHours || (a.id < b.id ? -1 : 1),
      ),
      tasks: [...state.tasks.values()].sort((a, b) => (a.unitId < b.unitId ? -1 : 1)),
    };
  }

  const me = state.commanders.get(role.id);
  const surveyed = state.knowledge.get(role.id)?.surveyed ?? new Set<HexKey>();

  // A commander who has been removed — killed, captured, relieved — keeps a view so their
  // client does not crash mid-session, but it is empty of everything an appointment
  // carries. Failing open here would be the worst possible direction to fail.
  if (me === undefined) {
    return {
      role: 'commander',
      commander: null,
      campaign,
      config: cfg,
      factions,
      commanders: [],
      world: maskedWorld(input, cfg, new Set<HexKey>(), new Set<HexKey>(), role.id),
      units: [],
      battle: [],
      reports: [],
      contacts: [],
      surveyed: [],
      visible: [],
      sent: [],
      received: [],
      captured: [],
      task: null,
      despatches: [],
      decisions: [],
      tasks: [],
    };
  }

  const visible = commanderVisible(state, world, cfg, role.id);
  const own = state.units.get(me.unitId);

  // What they actually hold, not what is true. Taken from their knowledge rather than
  // snapshotted from the units, which is the difference between a design about not
  // knowing where your own corps is and a list of exactly where it is.
  //
  // Filtered to formations under them: knowledge accumulates reports of anyone who has
  // written to them, and a peer's position is their own business.
  const under = new Set(formationsUnder(state, role.id).map((u) => u.id));
  const filed = state.knowledge.get(role.id)?.reports ?? new Map<string, UnitReport>();
  const reports = [...filed.values()]
    .filter((r) => under.has(r.unitId) && r.unitId !== me.unitId)
    .sort((a, b) => (a.unitId < b.unitId ? -1 : 1));

  // What they have been told, not what their columns can see this instant. Held knowledge,
  // like their reports and for the same reason: an enemy that walks out of view stops being
  // current, it does not stop having been there. `spottedBy` at request time would make a
  // contact blink out the moment a picket looked away.
  //
  // Two redactions, both by construction rather than by deletion. `contactFrom` built each
  // sighting already stripped to what its intelligence grade earned, and `publicContact`
  // drops the observed unit's id on the way out.
  const contacts = contactsOf(state, role.id).map(publicContact);

  // Their outbox, stripped by construction. An acknowledgement that has come back is the
  // one and only thing they ever learn about a despatch's fate, so it is computed from
  // their own inbox rather than from the despatch they sent.
  const held = inboxOf(state, role.id);
  const acknowledged = new Set(
    held.filter((d) => d.kind === 'acknowledgement').map((d) => d.inReplyTo),
  );
  const sent = despatchesFrom(state, role.id).map((d) => senderCopy(d, acknowledged.has(d.id)));
  const received = held.map((d) => addresseeCopy(d, isSuperseded(d, held)));

  return {
    role: 'commander',
    commander: publicCommander(me, state.units),
    campaign,
    config: cfg,
    factions,
    // Their own side's chain of command. Knowing who commands the enemy's II Corps is
    // intelligence, and it arrives by sighting or not at all.
    commanders: [...state.commanders.values()]
      .filter((c) => c.faction === me.faction)
      .map((c) => publicCommander(c, state.units))
      .sort(byId),
    world: maskedWorld(input, cfg, surveyed, visible, me.faction),
    units: own === undefined ? [] : [own],
    reports,
    contacts,
    surveyed: [...surveyed].sort(),
    visible: [...visible].sort(),
    battle: [...state.battle].filter((k) => visible.has(k)).sort(),
    sent,
    received,
    captured: capturedBy(state, me.faction).map(captorCopy),
    task: state.tasks.get(me.unitId) ?? null,
    despatches: [],
    decisions: [],
    tasks: [],
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
  // Projected before anything else, so the fields the engine never reads are gone before
  // they can be masked, serialised, pushed down a socket, or held in a browser. See
  // `projectWorld`: it is the difference between sending 23.6 MB and sending 9.1 MB.
  //
  // Before masking rather than after, because `maskWorld` spreads each hex through and
  // would happily carry twelve dead fields into the masked copy as well.
  const doc = projectWorld(input.worldDoc);
  if (!cfg.terrainFog) return doc;
  return maskWorld(doc as Record<string, unknown>, {
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

  // At most one live formation, and it must be the one they ride with. This is the
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

  // The despatch record itself carries the rider's route and the paper's fate, and both
  // are ground truth about where a formation is and what happened out of sight. A
  // commander is sent the ledgers instead, and never the record.
  if (view.despatches.length > 0) {
    throw new Error(`leak: a commander was sent ${view.despatches.length} raw despatches`);
  }
  if (view.decisions.length > 0 || view.tasks.length > 0) {
    throw new Error("leak: the referee's queue was sent to a commander");
  }

  // Belt and braces over `senderCopy` and `addresseeCopy`. Those build by construction
  // rather than by deletion, so this should be unreachable — which is exactly why it is
  // worth asserting, because the day it is reachable nobody will be watching.
  for (const d of [...view.sent, ...view.received] as unknown as Record<string, unknown>[]) {
    for (const forbidden of ['route', 'fate', 'progress', 'etaHours']) {
      if (forbidden in d) {
        throw new Error(`leak: despatch ${String(d['id'])} carries ${forbidden}`);
      }
    }
  }

  for (const d of view.received) {
    if (!Number.isFinite(d.receivedAtHours)) {
      throw new Error(`leak: despatch ${d.id} is in a commander's hand but never arrived`);
    }
  }

  for (const d of view.captured) {
    if (faction !== null && d.faction === faction) {
      throw new Error(`leak: ${d.id} is ${faction}'s own paper, not a capture`);
    }
    // The sender's own return travels on every despatch, and `captorCopy` cuts it down to
    // `capturedReport` on the way out. `unitId` surviving that is the whole of the
    // correlation a patrol is supposed to buy, handed over with the paper.
    const report = d.body.unitReport as Record<string, unknown> | undefined;
    if (report !== undefined && 'unitId' in report) {
      throw new Error(`leak: captured despatch ${d.id} names the formation that sent it`);
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

/**
 * A log entry as a commander may read it.
 *
 * The log is the single richest thing in the campaign: every march in order, every
 * rider's path, every contact anyone filed. Filtering it by who *acted* is not enough,
 * because one command produces events that happened to other commanders — a cascaded order
 * carries a route to the addressee's subordinate, stamped with the original sender's
 * name — so the filter is on the fact rather than on the actor.
 *
 * What survives is their outbox: the despatches they themselves wrote, in the shape
 * `senderCopy` already defines. Everything else is somebody else's business and is dropped
 * rather than trimmed.
 */
export interface LoggedAction {
  readonly seq: number;
  readonly clockHours: number;
  readonly despatch: SentDespatch;
}

export function logFor(
  events: readonly LoggedEvent[],
  commanderId: string,
  acknowledged: ReadonlySet<string | null> = new Set(),
): LoggedAction[] {
  const out: LoggedAction[] = [];
  for (const e of events) {
    if (e.payload.kind !== 'despatch_sent') continue;
    const d = e.payload.despatch;
    if (d.from !== commanderId) continue;
    out.push({
      seq: e.seq,
      clockHours: e.clockHours,
      despatch: senderCopy(d, acknowledged.has(d.id)),
    });
  }
  return out;
}

/** Reparse a view's world, for callers that want it as a `World` rather than a document. */
export const asWorld = (view: ClientView): World => parseWorld(view.world);

export { subordinates };
