/**
 * Turning what formations can see into events.
 *
 * A commander's knowledge has to be *remembered*, not recomputed. What a division can see
 * right now follows from where it stands, but what its commander knows follows from
 * everywhere their formations have ever stood — and that is history, so it belongs in the
 * log like any other fact. Recomputing it from current positions would mean a commander forgot
 * a valley the moment their column marched out of it.
 *
 * So after every command the server asks what each commander has now covered, and appends
 * a `hexes_surveyed` event for anything new. Nothing is emitted when nothing was learned,
 * which keeps a log of a hundred marches from carrying a hundred thousand redundant
 * coordinates.
 *
 * ## Two passes, and why they are different
 *
 * `observationEvents` records **ground**, which is public and never stale — terrain fog is
 * off, so this only feeds the memory the issued map will want when it arrives.
 *
 * `sightingEvents` records **the enemy**, under each commander's own labels — see
 * `knowledge.ts`, which decides whether a fresh sighting continues a contact or starts one.
 *
 * `reportEvents` records **formations**, which is the fog that carries the game. It fires
 * only where a commander needs no rider: the formation they are standing next to, one whose
 * column is touching their own, and a one-off seed for a formation they have never had word of
 * — because they wrote the order of battle and knows where they put their divisions. Everything
 * else arrives by despatch, hours late, or never.
 *
 * Both are the store's job rather than the engine's, because both are consequences of a
 * command rather than part of one. A state built by `applyAll` alone has commanders who
 * know nothing, which is why the tests run this pass themselves.
 */

import { commanderIds, formationOf, formationsUnder } from './commander.js';
import type { CampaignConfig } from './config.js';
import { formationsTouch } from './despatch.js';
import type { EventPayload } from './events.js';
import { sightingEvents } from './knowledge.js';
import { unkey, type HexKey } from './hex.js';
import { commandVisible } from './recon.js';
import type { CampaignState } from './state.js';
import { reportOf } from './unit.js';
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

/**
 * Reports a commander does not need a rider for.
 *
 * Three cases, and no others. They are standing next to the formation, so they see it. Their
 * own column is touching it, so word crosses in minutes — the same free traffic that lets two
 * touching formations hand despatches over. Or they have never had a report of it at all, in
 * which case they are given one at the current hour, because they wrote the order of battle and
 * knows where they put their divisions before anybody marched anywhere.
 *
 * Everything else waits for a rider, which is the point. The third case is a seed rather
 * than a rule: it fires once per formation, and from then on that hour only moves when a
 * despatch arrives or the columns close up.
 */
export function reportEvents(state: CampaignState, cfg: CampaignConfig): EventPayload[] {
  const out: EventPayload[] = [];

  for (const commanderId of commanderIds(state)) {
    const own = formationOf(state, commanderId);
    const held = state.knowledge.get(commanderId)?.reports;

    for (const unit of formationsUnder(state, commanderId)) {
      const inHand =
        own !== undefined &&
        (unit.id === own.id || formationsTouch(own, unit, cfg.footprint));

      if (!inHand && held?.has(unit.id) === true) continue;
      out.push({
        kind: 'report_filed',
        commanderId,
        report: reportOf(unit, state.clockHours),
      });
    }
  }

  return out;
}

/** Everything a command produced that nobody had to be told. */
export const knowledgeEvents = (
  state: CampaignState,
  world: World,
  cfg: CampaignConfig,
): EventPayload[] => [
  ...observationEvents(state, world, cfg),
  ...reportEvents(state, cfg),
  ...sightingEvents(state, world, cfg),
];

/** Whether anybody would learn anything new right now. */
export const hasNewObservations = (
  state: CampaignState,
  world: World,
  cfg: CampaignConfig,
): boolean => observationEvents(state, world, cfg).length > 0;
