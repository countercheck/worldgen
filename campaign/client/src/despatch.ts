/**
 * The post, as a commander's console needs it.
 *
 * Pure and separate from the components, for the same reason `board.ts` is: the rules
 * about who may be written to, what may be ordered, and how long a rider is likely to
 * take are worth testing without a browser.
 *
 * ## The estimate is the interesting part
 *
 * The server never sends a delivery estimate, and must not: an estimate is a distance,
 * and a distance is a position. So the *client* computes one — from the commander's own
 * last report of that formation, over the map he actually holds, with the same routing
 * code the server uses for the real ride.
 *
 * That is the shared engine earning its keep. The number is honestly wrong in exactly the
 * way the commander's knowledge is wrong: if his III Corps has marched twenty kilometres
 * since the report, his estimate is twenty kilometres out, and neither he nor this
 * function has any way to know it. It is labelled as a guess wherever it is shown, and
 * carries the hour it was computed from so a reader can judge how much to trust it.
 */

import {
  planRide,
  rideHours,
  type CampaignConfig,
  type ClientView,
  type Hex,
  type PublicCommander,
  type ReceivedDespatch,
  type SentDespatch,
  type World,
} from '@campaign/shared';

/** Somebody a commander may write to, and whether he may give them orders. */
export interface Correspondent {
  readonly id: string;
  readonly name: string;
  readonly unitId: string;
  /**
   * Whether an order may be sent, as opposed to merely a message.
   *
   * Orders travel downward only. Lateral coordination between corps commanders was real
   * and mattered enormously, so writing sideways is allowed and ordering sideways is not
   * — and the difference is worth showing in the form rather than discovering on a
   * refusal.
   */
  readonly mayOrder: boolean;
}

/** Everyone beneath a commander, at any depth. Cycle-guarded, like the engine's version. */
export function descendantsOf(
  commanders: readonly PublicCommander[],
  id: string,
): Set<string> {
  const children = new Map<string, string[]>();
  for (const c of commanders) {
    if (c.superiorId === null) continue;
    children.set(c.superiorId, [...(children.get(c.superiorId) ?? []), c.id]);
  }

  const found = new Set<string>();
  const queue = [id];
  while (queue.length > 0) {
    for (const child of children.get(queue.shift()!) ?? []) {
      if (found.has(child) || child === id) continue;
      found.add(child);
      queue.push(child);
    }
  }
  return found;
}

/**
 * Who this commander may write to, in the order a form should offer them.
 *
 * Subordinates first, because most despatches are orders and most orders go down. Then
 * everyone else on his own side. Never himself, and never the enemy.
 */
export function correspondents(view: ClientView): Correspondent[] {
  const me = view.commander;
  if (me === null) return [];

  const under = descendantsOf(view.commanders, me.id);
  return view.commanders
    .filter((c) => c.id !== me.id)
    .map((c) => ({ id: c.id, name: c.name, unitId: c.unitId, mayOrder: under.has(c.id) }))
    .sort((a, b) => {
      if (a.mayOrder !== b.mayOrder) return a.mayOrder ? -1 : 1;
      return a.name < b.name ? -1 : 1;
    });
}

/**
 * A commander's own guess at how long a rider would take.
 *
 * Null when he has no idea where the man is — which is the common case for a peer he has
 * never had a report of, and is worth saying rather than papering over with a number.
 */
export interface Estimate {
  readonly hours: number;
  /** The hour the report this was computed from describes. How stale the guess is. */
  readonly fromHours: number;
}

export function estimateRide(
  view: ClientView,
  world: World,
  cfg: CampaignConfig,
  toCommanderId: string,
  via: readonly Hex[] = [],
): Estimate | null {
  const me = view.commander;
  const from = view.units[0]?.column[0];
  if (me === null || from === undefined) return null;

  const target = view.commanders.find((c) => c.id === toCommanderId);
  if (target === undefined) return null;

  // The last thing he heard about where that formation was. Not where it is.
  const report = view.reports.find((r) => r.unitId === target.unitId);
  if (report === undefined) return null;

  const route = planRide(world, cfg, from, report.head, via);
  if (route === null) return null;

  return { hours: rideHours(world, cfg, route), fromHours: report.atHours };
}

/** The inbox, newest information first — by the hour described, not the hour it landed. */
export const inbox = (view: ClientView): ReceivedDespatch[] =>
  [...view.received].sort((a, b) => b.sentAtHours - a.sentAtHours);

/** The outbox, newest first. */
export const outbox = (view: ClientView): SentDespatch[] =>
  [...view.sent].sort((a, b) => b.sentAtHours - a.sentAtHours);

/**
 * Whether a received despatch still wants an answer.
 *
 * An acknowledgement is the only feedback channel in the game, so the console has to make
 * it obvious which despatches have had one and which have not — a sender learns nothing
 * at all unless somebody clicks this.
 */
export const isAcknowledged = (view: ClientView, despatchId: string): boolean =>
  view.sent.some((d) => d.kind === 'acknowledgement' && d.inReplyTo === despatchId);

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
