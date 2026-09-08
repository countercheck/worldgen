/**
 * Filing what a commander has been told about the enemy.
 *
 * One question, and everything here exists to answer it: when word of an enemy column
 * arrives, is it the contact he is already holding, or a new one?
 *
 * ## Why that is not obvious
 *
 * The engine knows which formation was seen. The commander must not — a contact carries
 * his own label and never the observed unit's id, because a man who could correlate two
 * sightings hours apart has been handed for free what these rules make him buy with a
 * patrol. So identity has to be decided *here*, on his behalf, and the decision has to be
 * one his staff could plausibly have made.
 *
 * ## The rule is eyes on, not elapsed time
 *
 * A column his own men have not lost sight of is one contact, and each fresh look updates
 * it in place: that is a picket keeping watch, and nothing is being given away because he
 * watched it happen. The moment it goes out of view the contact is marked lost, and if it
 * reappears later it gets a **new** label — with the old sighting left on his map as a
 * separate mark. Whether the two are the same corps is then his judgement, which is the
 * judgement the period turned on and exactly what a contact number must not make for him.
 *
 * An earlier version of this compared *hours since the contact was last filed* against a
 * window. That measured the wrong thing entirely. Filing happens when a command is
 * processed, so a referee advancing the clock in six-hour steps minted a new contact every
 * step — for an enemy standing still in front of a division that had never once stopped
 * looking at it. `inSight` is a fact about the world; a gap between filings is an artefact
 * of when somebody clicked.
 *
 * ## Second-hand word always starts a new contact
 *
 * A report from a subordinate mints its own label even when the recipient's own pickets
 * are watching the same column. That is correct rather than untidy: a headquarters
 * receiving word of an enemy on its flank has no way to know it is the same body of troops
 * its own screen can see, and deciding that they are is precisely the work of a staff.
 */

import { formationOf } from './commander.js';
import type { CampaignConfig } from './config.js';
import type { EventPayload } from './events.js';
import { spottedBy, type Contact, type Sighting } from './recon.js';
import type { CampaignState } from './state.js';
import type { World } from './world.js';

/** Where a sighting came from, which decides whether it may continue a contact. */
export type Provenance =
  /** His own formation is looking at it. May continue a contact still in sight. */
  | 'own_eyes'
  /** Somebody wrote to him about it. Always a fresh label — see the module comment. */
  | 'reported';

/**
 * Events filing a batch of sightings for one commander.
 *
 * The counter is tracked locally as the batch is built, because state is not folded
 * between the payloads of a single pass — two new contacts in one tick would otherwise be
 * minted with the same label, and the second would overwrite the first.
 */
export function fileSightings(
  state: CampaignState,
  cfg: CampaignConfig,
  commanderId: string,
  sightings: Iterable<Sighting>,
  atHours: number,
  provenance: Provenance = 'own_eyes',
): EventPayload[] {
  const knowledge = state.knowledge.get(commanderId);
  const held = knowledge?.contacts;
  let next = knowledge?.nextContactNo ?? 1;

  const out: EventPayload[] = [];
  const mintedThisPass = new Map<string, string>();

  for (const sighting of sightings) {
    const watching =
      provenance === 'own_eyes' ? contactInSight(held, sighting.unitId) : undefined;
    const existing = mintedThisPass.get(sighting.unitId) ?? watching?.id;

    const id = existing ?? `c${next}`;
    if (existing === undefined) {
      next += 1;
      mintedThisPass.set(sighting.unitId, id);
    }

    const contact: Contact = { ...sighting, id, inSight: provenance === 'own_eyes' };

    // Quiet unless something changed. A division watching a stationary enemy for a day
    // would otherwise write an identical event on every command, and the log is a record
    // of what happened rather than of what was still true.
    if (watching !== undefined && !worthFiling(watching, contact, cfg)) continue;
    out.push({ kind: 'contact_filed', commanderId, contact });
  }
  return out;
}

/** A contact of this formation that the commander has not lost sight of. */
function contactInSight(
  held: ReadonlyMap<string, Contact> | undefined,
  unitId: string,
): Contact | undefined {
  if (held === undefined) return undefined;
  for (const contact of held.values()) {
    if (contact.unitId === unitId && contact.inSight) return contact;
  }
  return undefined;
}

/**
 * Whether a fresh look is worth writing down.
 *
 * It moved, or he learned more about it, or enough time has passed that the hour on his
 * map has drifted from the hour he is actually looking at it. `contactRefreshHours` is a
 * logging cadence and nothing else — no rule keys off it, and changing it changes only how
 * chatty the log is and how stale a watched contact's hour is allowed to read.
 */
function worthFiling(held: Contact, fresh: Contact, cfg: CampaignConfig): boolean {
  if (held.coord.q !== fresh.coord.q || held.coord.r !== fresh.coord.r) return true;
  if (held.intelLevel !== fresh.intelLevel) return true;
  if (!held.inSight) return true;
  return fresh.seenAtHours - held.seenAtHours >= cfg.contactRefreshHours;
}

/**
 * What each commander's own formation can see, filed under his own labels — and what it
 * has just lost sight of.
 *
 * His own eyes only. What his subordinates see reaches him as sightings attached to a
 * report, hours later or never; merging their vision into his here is the telepathy this
 * design exists to deny.
 */
export function sightingEvents(
  state: CampaignState,
  world: World,
  cfg: CampaignConfig,
): EventPayload[] {
  const out: EventPayload[] = [];

  // Sorted, so the events a command produces do not depend on the order a referee
  // happened to appoint commanders in — the same reason the generator sorts before it
  // places bridges.
  for (const commanderId of [...state.commanders.keys()].sort()) {
    const own = formationOf(state, commanderId);
    if (own === undefined) continue;

    const seen = spottedBy(state, world, cfg, own);
    const sightings = [...seen.values()].sort((a, b) => (a.unitId < b.unitId ? -1 : 1));
    out.push(...fileSightings(state, cfg, commanderId, sightings, state.clockHours));

    // Anything he was watching and can no longer see. The contact stays on his map at the
    // hex he last saw it; what changes is that a later sighting will be a new contact
    // rather than a continuation, because he did lose it.
    const held = state.knowledge.get(commanderId)?.contacts;
    for (const contact of [...(held?.values() ?? [])].sort((a, b) => (a.id < b.id ? -1 : 1))) {
      if (!contact.inSight || seen.has(contact.unitId)) continue;
      out.push({
        kind: 'contact_lost',
        commanderId,
        contactId: contact.id,
        atHours: state.clockHours,
      });
    }
  }
  return out;
}
