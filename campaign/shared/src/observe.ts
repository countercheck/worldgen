/**
 * Turning what units can see into events.
 *
 * A faction's knowledge has to be *remembered*, not recomputed. What a division can see
 * right now follows from where it stands, but what its army knows follows from everywhere
 * it has ever stood — and that is history, so it belongs in the log like any other fact.
 * Recomputing it from current positions would mean a faction forgot a valley the moment
 * it marched out of it, which is the opposite of fog of war.
 *
 * So after every command the server asks what each faction can now see, and appends a
 * `hexes_revealed` event for anything newly observed. Nothing is emitted when nothing was
 * learned, which keeps a log of a hundred marches from carrying a hundred thousand
 * redundant coordinates.
 *
 * This lives in `shared` rather than in the server because the rule — what counts as
 * seen — is part of the game, and the client needs the same answer to draw a reach
 * preview over its own knowledge.
 */

import type { CampaignConfig } from './config.js';
import type { EventPayload } from './events.js';
import { unkey, type HexKey } from './hex.js';
import { factionVisible } from './recon.js';
import { factionIds, type CampaignState } from './state.js';
import type { World } from './world.js';

/**
 * Events for everything each faction has newly observed.
 *
 * Factions are walked in sorted order so the events a command produces do not depend on
 * the order a referee happened to add them — the same reason the generator sorts before
 * it places bridges.
 */
export function observationEvents(
  state: CampaignState,
  world: World,
  cfg: CampaignConfig,
): EventPayload[] {
  const out: EventPayload[] = [];

  for (const faction of factionIds(state)) {
    const known = state.knowledge.get(faction)?.seen ?? new Set<HexKey>();
    const fresh: HexKey[] = [];

    for (const k of factionVisible(state, world, cfg, faction)) {
      if (!known.has(k)) fresh.push(k);
    }

    if (fresh.length === 0) continue;
    // Sorted, so two runs of the same campaign produce byte-identical events.
    fresh.sort();
    out.push({
      kind: 'hexes_revealed',
      faction,
      coords: fresh.map(unkey),
    });
  }

  return out;
}

/** Whether a faction would learn anything new right now. */
export const hasNewObservations = (
  state: CampaignState,
  world: World,
  cfg: CampaignConfig,
): boolean => observationEvents(state, world, cfg).length > 0;
