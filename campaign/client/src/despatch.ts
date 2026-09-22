/**
 * The post, as a commander's console needs it.
 *
 * Pure and separate from the components, for the same reason `board.ts` is: who may be
 * written to, and how long a rider is likely to take, are worth testing without a
 * browser.
 *
 * ## The estimate is the interesting part
 *
 * The server never sends a delivery estimate, and must not: an estimate is a distance,
 * and a distance is a position. So the *client* computes one — from the commander's own
 * last report of that formation, over the map they actually hold, with the same routing
 * code the server uses for the real ride.
 *
 * That is the shared engine earning its keep. The number is honestly wrong in exactly the
 * way the commander's knowledge is wrong: if their III Corps has marched twenty kilometres
 * since the report, their estimate is twenty kilometres out, and neither they nor this
 * function has any way to know it. It is labelled as a guess wherever it is shown, and
 * carries the hour it was computed from so a reader can judge how much to trust it.
 */

import {
  planRide,
  rideHours,
  type CampaignConfig,
  type ClientView,
  type Hex,
  type ReceivedDespatch,
  type SentDespatch,
  type World,
} from '@campaign/shared';

/**
 * How an addressee stands to the sender, which is also why they may be written to at all.
 *
 * The first two are the chain of command, one link either way. The third is anyone else
 * on their own side whose column they can see from where they stand.
 */
export type Relation = 'superior' | 'subordinate' | 'in_sight';

/** Somebody a commander may write to, and why. */
export interface Correspondent {
  readonly id: string;
  readonly name: string;
  readonly unitId: string;
  /**
   * The formation they ride with, and whose side they are on, for the addressee list.
   *
   * An officer's name is not enough to address a despatch by. Two generals of the same
   * rank are told apart by what they command, and a new player has learned the formations
   * on the map long before they have learned which marshal rides with which — so the list
   * says both. Neither is a secret and neither is a position: it is the order of battle,
   * which every staff on that side already holds.
   */
  readonly unitName: string;
  readonly faction: string;
  readonly relation: Relation;
}

/**
 * The address a note to the referee is sent to, in place of a commander's id.
 *
 * Not a commander, and never one: an id with a character no commander id can contain, so
 * no officer can ever be named into it — `idFor` makes ids from letters, digits and
 * hyphens only.
 */
export const REFEREE_ADDRESS = '@referee';

const RELATION_ORDER: Record<Relation, number> = { superior: 0, subordinate: 1, in_sight: 2 };

/**
 * Who this commander may write to right now, in the order a form should offer them.
 *
 * Exactly who the server says, from `view.addressees`, which is computed by the same
 * function `check` enforces — so the list cannot offer somebody a despatch would then be
 * refused for. Their superior first, then those beneath them, then anyone else in sight.
 */
export function correspondents(view: ClientView, senderId?: string): Correspondent[] {
  // The viewer, normally. A referee writing on a commander's behalf names the sender
  // explicitly — they have no seat of their own, and the question "who may this commander write
  // to" is about the commander, not about who is looking.
  const from = senderId ?? view.commander?.id;
  if (from === undefined) return [];

  const sender = view.commanders.find((c) => c.id === from);
  const allowed = new Set(view.addressees[from] ?? []);

  return view.commanders
    .filter((c) => c.id !== from && allowed.has(c.id))
    .map((c) => ({
      id: c.id,
      name: c.name,
      unitId: c.unitId,
      unitName: c.unitName,
      faction: c.faction,
      relation:
        sender?.superiorId === c.id
          ? ('superior' as const)
          : c.superiorId === from
            ? ('subordinate' as const)
            : ('in_sight' as const),
    }))
    .sort(
      (a, b) =>
        RELATION_ORDER[a.relation] - RELATION_ORDER[b.relation] || (a.name < b.name ? -1 : 1),
    );
}

/**
 * How long a rider would take, over the ground as it actually is.
 *
 * The referee's, and only their. A commander is shown none of this: they do not know where
 * the addressee is, so they cannot know how long the ride will be, and a number — even a
 * hedged one — is a distance, and a distance is a position.
 */
export interface Estimate {
  readonly hours: number;
}

export function estimateRide(
  view: ClientView,
  world: World,
  cfg: CampaignConfig,
  toCommanderId: string,
  via: readonly Hex[] = [],
  fromCommanderId?: string,
): Estimate | null {
  const sender = fromCommanderId ?? view.commander?.id;
  if (sender === undefined) return null;

  const from = placeOf(view, sender);
  const to = placeOf(view, toCommanderId);
  if (from === null || to === null) return null;

  const route = planRide(world, cfg, from, to, via);
  return route === null ? null : { hours: rideHours(world, cfg, route) };
}

/**
 * Where a commander's formation actually stands.
 *
 * Live units only, and null for anything the view does not carry one of. That is the whole
 * of the restriction: a commander's view holds exactly one live unit — their own — so they can
 * estimate nothing, and the referee's holds them all.
 */
function placeOf(view: ClientView, commanderId: string): Hex | null {
  const commander = view.commanders.find((c) => c.id === commanderId);
  if (commander === undefined) return null;
  return view.units.find((u) => u.id === commander.unitId)?.column[0] ?? null;
}

/** The inbox, newest information first — by the hour described, not the hour it landed. */
export const inbox = (view: ClientView): ReceivedDespatch[] =>
  [...view.received].sort((a, b) => b.sentAtHours - a.sentAtHours);

/** The outbox, newest first. */
export const outbox = (view: ClientView): SentDespatch[] =>
  [...view.sent].sort((a, b) => b.sentAtHours - a.sentAtHours);

/** What a forwarded despatch says, so two lags stack rather than one fact being retyped. */
export const forwardOf = (
  d: ReceivedDespatch,
): { text?: string; contacts?: readonly unknown[] } => {
  const preface = `Forwarded from ${d.from}, written at hour ${d.sentAtHours}.`;
  return {
    text: d.body.text === undefined ? preface : `${preface}\n\n${d.body.text}`,
    ...(d.body.contacts === undefined ? {} : { contacts: d.body.contacts }),
  };
};

/**
 * An officer as the post has to name them: who, what they command, and which side.
 *
 * One function rather than three lookups at each of the four places a commander is named
 * in the post — the inbox, the outbox, the addressee list and the referee's log — so that
 * an officer is described the same way wherever they appear. A despatch that says only
 * "From Ney" is asking the reader to remember an order of battle; one that says only
 * "1re Division" has lost who put their name to it.
 */
export interface CommanderLabel {
  readonly name: string;
  readonly unit: string;
  /** The side's name as it is shown, not its id. */
  readonly faction: string;
}

/**
 * How to name any commander this view knows of.
 *
 * Falls back to the id on all three counts, which is what a commander riding with a
 * formation that has just been destroyed comes back as. An inbox that went blank would be
 * worse: the despatch is still in their hand and still says something.
 */
export function labelling(view: ClientView): (commanderId: string) => CommanderLabel {
  const byId = new Map(view.commanders.map((c) => [c.id, c]));
  const factions = new Map(view.factions.map((f) => [f.id, f.name]));

  return (commanderId) => {
    const c = byId.get(commanderId);
    if (c === undefined) return { name: commanderId, unit: commanderId, faction: '' };
    return {
      name: c.name,
      unit: c.unitName,
      faction: factions.get(c.faction) ?? c.faction,
    };
  };
}
