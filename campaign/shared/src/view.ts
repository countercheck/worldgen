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
 * is the product. If a commander's browser receives the whole world and hides part of it,
 * anyone who opens the developer tools has the entire map and every enemy position, and
 * the game is over. Filtering has to happen here, before anything is serialised.
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
 *    is constructed already stripped of everything its intelligence grade does not earn,
 *    so a low-quality sighting never carries the corps identity in the first place.
 */

import { DEFAULT_CONFIG, type CampaignConfig } from './config.js';
import { key, type HexKey } from './hex.js';
import { maskWorld } from './mask.js';
import { factionVisible, spotted, type Contact } from './recon.js';
import type { Faction } from './events.js';
import type { CampaignState } from './state.js';
import type { Unit } from './unit.js';
import { parseWorld, type World } from './world.js';

/** Who is asking. */
export type Role = { readonly kind: 'referee' } | { readonly kind: 'faction'; readonly id: string };

export const REFEREE_ROLE: Role = { kind: 'referee' };
export const factionRole = (id: string): Role => ({ kind: 'faction', id });

/** Public facts about a faction. Never the join token, which is not in state at all. */
export interface PublicFaction {
  readonly id: string;
  readonly name: string;
  readonly color: string;
}

export interface ClientView {
  readonly role: 'referee' | 'faction';
  readonly faction: string | null;
  readonly campaign: {
    readonly id: string;
    readonly name: string;
    readonly clockHours: number;
    readonly seq: number;
  };
  readonly factions: readonly PublicFaction[];
  /** A `world.json` document — the whole world for a referee, a masked one otherwise. */
  readonly world: unknown;
  /** Own units in full. A faction never receives another faction's unit records. */
  readonly units: readonly Unit[];
  /** Enemies, as last seen and no better. Empty for a referee, who sees units instead. */
  readonly contacts: readonly Contact[];
  readonly seen: readonly HexKey[];
  readonly visible: readonly HexKey[];
}

const publicFaction = (f: Faction): PublicFaction => ({
  id: f.id,
  name: f.name,
  color: f.color,
});

export interface ViewInput {
  readonly campaignId: string;
  readonly state: CampaignState;
  /** The world as generated. Never sent to a faction unmasked. */
  readonly worldDoc: unknown;
  readonly world: World;
  readonly cfg?: CampaignConfig;
}

/**
 * Everything, and only everything, this role is entitled to.
 *
 * A referee gets ground truth. A faction gets its own units, a world masked to what it
 * has seen, and contacts for the enemies it has actually spotted — nothing else, and in
 * particular no record of an enemy it has never observed.
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

  if (role.kind === 'referee') {
    return {
      role: 'referee',
      faction: null,
      campaign,
      factions,
      world: input.worldDoc,
      units: [...state.units.values()].sort((a, b) => (a.id < b.id ? -1 : 1)),
      contacts: [],
      seen: [],
      visible: [],
    };
  }

  const id = role.id;
  const seen = state.knowledge.get(id)?.seen ?? new Set<HexKey>();
  const visible = factionVisible(state, world, cfg, id);

  // Contacts are built by `spotted`, which constructs each one already stripped to what
  // the sighting earned. Nothing here has to remember to redact.
  const contacts = [...spotted(state, world, cfg, id).values()].sort((a, b) =>
    a.unitId < b.unitId ? -1 : 1,
  );

  return {
    role: 'faction',
    faction: id,
    campaign,
    factions,
    world: maskWorld(input.worldDoc as Record<string, unknown>, {
      seen,
      visible,
      faction: id,
      clockHours: state.clockHours,
    }),
    // Filtered by faction, not merely marked. A unit belonging to somebody else must not
    // be in this array at all.
    units: [...state.units.values()]
      .filter((u) => u.faction === id)
      .sort((a, b) => (a.id < b.id ? -1 : 1)),
    contacts,
    seen: [...seen].sort(),
    visible: [...visible].sort(),
  };
}

/**
 * The masked world alone, for the export endpoint.
 *
 * Goes through the same masking as `viewFor` so the file a referee downloads and the map
 * a commander sees can never disagree.
 */
export function exportFor(input: ViewInput, role: Role): unknown {
  return viewFor(input, role).world;
}

/**
 * A cheap check that a view really is masked, used by the leakage tests and by the
 * export route as a last line of defence.
 *
 * Belt and braces: `viewFor` is meant to guarantee this, and a bug in it is precisely the
 * kind that ships quietly, so the property is also asserted at the boundary rather than
 * only in a test.
 */
export function assertMasked(view: ClientView): void {
  if (view.role === 'referee') return;

  const doc = view.world as { hexes?: { q: number; r: number; tags?: string[] }[] };
  const seen = new Set(view.seen);

  for (const h of doc.hexes ?? []) {
    const k = key({ q: h.q, r: h.r });
    const fogged = (h.tags ?? []).includes('fog');
    if (!fogged && !seen.has(k)) {
      throw new Error(
        `fog leak: hex ${k} is unmasked but ${view.faction} has not seen it. ` +
          `This is a data-exposure bug, not a rendering one — refusing to serve.`,
      );
    }
  }

  for (const u of view.units) {
    if (u.faction !== view.faction) {
      throw new Error(`fog leak: unit ${u.id} belongs to ${u.faction}, not ${view.faction}`);
    }
  }
}

/** Reparse a masked world, for callers that want it as a `World` rather than a document. */
export const asWorld = (view: ClientView): World => parseWorld(view.world);
