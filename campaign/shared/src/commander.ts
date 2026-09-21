/**
 * Commanders, and the chain of command.
 *
 * Every formation has one. Most are run by the referee and some are held by players via a
 * join link, and there is no structural difference between the two — handing somebody a
 * token is the whole of putting them in a seat.
 *
 * A commander is not a unit and does not have a position of their own. They ride with a
 * formation, and that formation is three things at once: where they are, what they can see,
 * and where a despatch rider must go to find them. Everything else here follows from
 * keeping those three attached to the unit rather than to the commander.
 *
 * The hierarchy is one field. `superiorId` points up; who a commander commands is everyone
 * whose `superiorId` leads back to them. A single parent per commander means the structure
 * cannot become a graph by accident, and the tree is cheap enough to walk that caching it
 * would only be a second fact to keep in step.
 */

import type { CampaignState } from './state.js';
import type { Unit } from './unit.js';

export interface Commander {
  readonly id: string;
  /** "Marshal Ney". Shown wherever the commander rather than the formation is meant. */
  readonly name: string;
  readonly faction: string;
  /**
   * The formation they ride with: their position, their eyes, and their address.
   *
   * More than one commander may ride with the same formation — a corps commander whose
   * own headquarters has been destroyed falls back on one of their divisions — so this is
   * not a key and units are not indexed by it.
   */
  readonly unitId: string;
  /** Who they answer to. Null for the army commander, and for nobody else. */
  readonly superiorId: string | null;
  /**
   * Whether an arriving despatch is passed straight down to subordinates.
   *
   * True for a commander the referee is running, so that twenty formations do not become
   * twenty pieces of paperwork a campaign day. False for a player, who writes their own.
   */
  readonly autoCascade: boolean;
}

/** Commanders in a fixed order, so nothing depends on the order they were added in. */
export const commanderIds = (s: CampaignState): string[] => [...s.commanders.keys()].sort();

/** Everyone who answers to this commander directly. */
export const directSubordinates = (s: CampaignState, id: string): Commander[] =>
  [...s.commanders.values()]
    .filter((c) => c.superiorId === id)
    .sort((a, b) => (a.id < b.id ? -1 : 1));

/**
 * Everyone beneath this commander, at any depth, excluding themselves.
 *
 * Breadth-first and guarded against cycles. A cycle should be impossible — `check`
 * refuses to create one — but a state loaded from a log written by an older or buggier
 * build is not something a tree walk should hang on.
 */
export function subordinates(s: CampaignState, id: string): Commander[] {
  const found: Commander[] = [];
  const seen = new Set<string>([id]);
  const queue: string[] = [id];

  while (queue.length > 0) {
    for (const child of directSubordinates(s, queue.shift()!)) {
      if (seen.has(child.id)) continue;
      seen.add(child.id);
      found.push(child);
      queue.push(child.id);
    }
  }
  return found;
}

/** The chain upward, nearest superior first. Ends at the army commander. */
export function superiors(s: CampaignState, id: string): Commander[] {
  const chain: Commander[] = [];
  const seen = new Set<string>([id]);

  let at = s.commanders.get(id)?.superiorId ?? null;
  while (at !== null && !seen.has(at)) {
    const next = s.commanders.get(at);
    if (next === undefined) break;
    seen.add(at);
    chain.push(next);
    at = next.superiorId;
  }
  return chain;
}

/**
 * Whether `id` may give orders to `targetId`.
 *
 * Orders travel downward only. A message may go to anyone on the same side — lateral
 * coordination between corps commanders was real and mattered enormously — but an order
 * is an instruction, and an instruction that can go sideways is not a chain of command.
 */
export const mayOrder = (s: CampaignState, id: string, targetId: string): boolean =>
  id !== targetId && subordinates(s, id).some((c) => c.id === targetId);

/**
 * Whether `id` may send any despatch at all to `targetId`.
 *
 * Same faction, and not themselves. Writing to the enemy is not a despatch, and if it ever
 * becomes a mechanic it will be a different one with its own rules.
 */
export function mayWriteTo(s: CampaignState, id: string, targetId: string): boolean {
  const from = s.commanders.get(id);
  const to = s.commanders.get(targetId);
  if (from === undefined || to === undefined) return false;
  return id !== targetId && from.faction === to.faction;
}

/** The formation a commander rides with, if it still exists. */
export const formationOf = (s: CampaignState, id: string): Unit | undefined => {
  const commander = s.commanders.get(id);
  return commander === undefined ? undefined : s.units.get(commander.unitId);
};

/**
 * The formations a commander is responsible for: their own, and their subordinates'.
 *
 * Sorted and deduplicated, because two commanders riding with the same formation would
 * otherwise list it twice.
 */
export function formationsUnder(s: CampaignState, id: string): Unit[] {
  const ids = new Set<string>();
  const self = s.commanders.get(id);
  if (self !== undefined) ids.add(self.unitId);
  for (const sub of subordinates(s, id)) ids.add(sub.unitId);

  return [...ids]
    .map((unitId) => s.units.get(unitId))
    .filter((u): u is Unit => u !== undefined)
    .sort((a, b) => (a.id < b.id ? -1 : 1));
}

/** Whether making `superiorId` the superior of `id` would close a loop. */
export function wouldCycle(
  s: CampaignState,
  id: string,
  superiorId: string | null,
): boolean {
  if (superiorId === null) return false;
  if (superiorId === id) return true;
  return superiors(s, superiorId).some((c) => c.id === id);
}
