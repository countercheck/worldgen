/**
 * Turning what formations can see into events.
 *
 * A commander's knowledge has to be *remembered*, not recomputed. What a division can see
 * right now follows from where it stands, but what its commander knows follows from
 * everywhere his formations have ever stood — and that is history, so it belongs in the
 * log like any other fact. Recomputing it from current positions would mean a man forgot
 * a valley the moment his column marched out of it.
 *
 * So after every command the server asks what each commander has now covered, and appends
 * a `hexes_surveyed` event for anything new. Nothing is emitted when nothing was learned,
 * which keeps a log of a hundred marches from carrying a hundred thousand redundant
 * coordinates.
 *
 * ## The interim rule, stated rather than assumed
 *
 * A commander's knowledge here is the union of what **all his formations** can see, not
 * only the one he rides with. That is not the finished model: the design is that he sees
 * through his own formation and learns the rest by despatch rider, hours late and
 * sometimes never.
 *
 * Until riders exist, the rule in force is **every formation reports to its superior the
 * instant it sees anything**. That is a coherent intermediate rule rather than a leak —
 * the reports are real, dated, and attributed; they simply travel at infinite speed. Step
 * 2 gives them a rider and a delay, and nothing else about this changes.
 *
 * `commanderVisible` is already the finished thing and is what a client draws as "what I
 * can see from here". The distinction is live in the code before it is live in the rules,
 * which is the cheap way round.
 */

import type { CampaignConfig } from './config.js';
import { commanderIds } from './commander.js';
import type { EventPayload } from './events.js';
import { unkey, type HexKey } from './hex.js';
import { commandVisible } from './recon.js';
import type { CampaignState } from './state.js';
import type { World } from './world.js';

/**
 * Events for everything each commander has newly learned.
 *
 * Commanders are walked in sorted order so the events a command produces do not depend on
 * the order a referee happened to add them — the same reason the generator sorts before
 * it places bridges.
 */
export function observationEvents(
  state: CampaignState,
  world: World,
  cfg: CampaignConfig,
): EventPayload[] {
  const out: EventPayload[] = [];

  for (const commanderId of commanderIds(state)) {
    const known = state.knowledge.get(commanderId)?.surveyed ?? new Set<HexKey>();
    const fresh: HexKey[] = [];

    for (const k of commandVisible(state, world, cfg, commanderId)) {
      if (!known.has(k)) fresh.push(k);
    }

    if (fresh.length === 0) continue;
    // Sorted, so two runs of the same campaign produce byte-identical events.
    fresh.sort();
    out.push({
      kind: 'hexes_surveyed',
      commanderId,
      coords: fresh.map(unkey),
    });
  }

  return out;
}

/** Whether anybody would learn anything new right now. */
export const hasNewObservations = (
  state: CampaignState,
  world: World,
  cfg: CampaignConfig,
): boolean => observationEvents(state, world, cfg).length > 0;
