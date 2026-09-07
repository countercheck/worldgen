/**
 * What a formation can see, and therefore what the man riding with it knows.
 *
 * The rules keep this deliberately simple: a unit observes a zone extending one hex from
 * its march column in all directions, or two hexes if it has the Scout trait. There is no
 * line of sight, no ray-casting, no terrain occlusion — a ridge does not hide anything.
 *
 * The interesting part is not the radius, it is what the radius is measured from. A
 * division is a column of hexes, not a point, so a scouting cavalry division eighteen
 * kilometres long sweeps a corridor two hexes wide and twenty-two long. Recon is a
 * consequence of the column, which is why `column.ts` had to come first.
 */

import { occupied } from './column.js';
import { formationsUnder } from './commander.js';
import type { CampaignConfig, Grade } from './config.js';
import { distance, hexRange, key, type Hex, type HexKey } from './hex.js';
import type { CampaignState } from './state.js';
import { echelonOf, hasTrait, isDivision, type Echelon, type Unit } from './unit.js';
import { hexAt, type World } from './world.js';

/** How far a unit sees: one hex, or two with scouts. */
export const reconRadius = (cfg: CampaignConfig, unit: Unit): number =>
  hasTrait(unit, 'scout') ? cfg.scoutReconRadius : cfg.reconRadius;

/**
 * The hexes a single unit observes.
 *
 * Measured from every hex of the column, not from the head — the rules say "from march
 * column", and a column strung across twenty kilometres of road sees the country beside
 * all of it.
 */
export function reconZone(
  world: World,
  cfg: CampaignConfig,
  unit: Unit,
  grade: Grade = 'road',
): Set<HexKey> {
  const radius = reconRadius(cfg, unit);
  const seen = new Set<HexKey>();
  for (const c of occupied(unit, grade)) {
    for (const h of hexRange(c, radius)) {
      if (hexAt(world, h) !== undefined) seen.add(key(h));
    }
  }
  return seen;
}

/**
 * What the commander himself can see, from where he stands.
 *
 * The formation he rides with, and nothing else. Not his corps, not his side — a man on a
 * horse at Fleurus can see about a kilometre, whatever else is marching under his name
 * forty kilometres away. Everything beyond this reaches him by despatch or not at all,
 * which is the whole subject of the ruleset.
 *
 * Derived from the column's position every time it is asked, never stored. A stored copy
 * is a second fact that can disagree with where the unit actually stands, and the one
 * thing a fog-of-war engine cannot afford is two answers to "what can he see".
 */
export function commanderVisible(
  state: CampaignState,
  world: World,
  cfg: CampaignConfig,
  commanderId: string,
): Set<HexKey> {
  const commander = state.commanders.get(commanderId);
  if (commander === undefined) return new Set<HexKey>();
  const unit = state.units.get(commander.unitId);
  if (unit === undefined) return new Set<HexKey>();
  return reconZone(world, cfg, unit);
}

/**
 * Everything the formations under a commander can currently see between them.
 *
 * Emphatically **not** what the commander knows — that is `commanderVisible` plus what
 * has been reported to him. This is the observing side of the transaction: the ground his
 * divisions are actually looking at, from which reports are generated. Keeping the two
 * functions distinct, and distinctly named, is what stops the second quietly becoming the
 * first the next time somebody needs "what can this command see".
 */
export function commandVisible(
  state: CampaignState,
  world: World,
  cfg: CampaignConfig,
  commanderId: string,
): Set<HexKey> {
  const seen = new Set<HexKey>();
  for (const unit of formationsUnder(state, commanderId)) {
    for (const k of reconZone(world, cfg, unit)) seen.add(k);
  }
  return seen;
}

/**
 * What a faction learns about an enemy unit it has spotted.
 *
 * `intelLevel` follows the rules' patrol table, where the lowest die read decides how
 * much was learned: 2 is presence and location, 3-4 adds direction of march, 5 adds the
 * number of divisions, 6 adds the identity of the corps. Standard reconnaissance — a unit
 * simply seeing another — is presence and location, so it enters at 2.
 *
 * Recording *how well* an enemy was seen, rather than merely that it was, is what lets a
 * commander's map show a vague contact differently from a positively identified corps.
 */
export interface Contact {
  readonly unitId: string;
  readonly faction: string;
  readonly coord: Hex;
  readonly intelLevel: IntelLevel;
  /** Campaign hour of the sighting. What makes a contact go stale. */
  readonly seenAtHours: number;
  /** Only known at intel 5 and above. */
  readonly kind: Unit['kind'] | null;
  /**
   * Roughly how large it is. Only known at intel 4 and above.
   *
   * The rules' patrol table gives "the number of divisions" at 4, which is the same
   * question as how big the thing in front of you is. Below that a sighting draws as an
   * empty frame, which is exactly what NATO symbology means by it.
   */
  readonly echelon: Echelon | null;
  /** Only known at intel 6. */
  readonly corps: string | null;
}

export type IntelLevel = 1 | 2 | 3 | 4 | 5 | 6;

/** Presence and location — what simply seeing a unit tells you. */
export const SIGHTING_INTEL: IntelLevel = 2;

/**
 * Build a contact at a given quality, revealing only what that quality earns.
 *
 * The redaction happens here rather than at the point of display, so a low-grade contact
 * never carries the identity of the corps in the first place. Anything a commander is not
 * entitled to know should not reach their client at all — filtering it out on the way to
 * the screen is how it leaks.
 */
export function contactFrom(unit: Unit, intel: IntelLevel, atHours: number): Contact {
  const at = unit.column[0];
  return {
    unitId: unit.id,
    faction: unit.faction,
    coord: at ?? { q: 0, r: 0 },
    intelLevel: intel,
    seenAtHours: atHours,
    kind: intel >= 5 ? unit.kind : null,
    echelon: intel >= 4 ? echelonOf(unit) : null,
    corps: intel >= 6 ? unit.corps : null,
  };
}

/**
 * Enemy units one formation can see, and how well.
 *
 * A unit is spotted when any part of its column stands in the observer's recon zone. The
 * whole column counts — a corps whose head is hidden but whose baggage is strung across
 * open country has been seen.
 *
 * Per observing formation rather than per side, because a sighting is something a
 * particular division made at a particular hour, and it is that division which has to get
 * word back. A contact with no observer attached cannot be reported, cannot be dated
 * honestly, and cannot be told apart from a rumour.
 */
export function spottedBy(
  state: CampaignState,
  world: World,
  cfg: CampaignConfig,
  observer: Unit,
): Map<string, Contact> {
  const zone = reconZone(world, cfg, observer);
  const found = new Map<string, Contact>();

  for (const unit of state.units.values()) {
    if (unit.faction === observer.faction) continue;
    const seenHex = occupied(unit).find((c) => zone.has(key(c)));
    if (seenHex === undefined) continue;

    found.set(unit.id, {
      ...contactFrom(unit, SIGHTING_INTEL, state.clockHours),
      // The hex it was actually seen on, which may be its tail rather than its head.
      coord: seenHex,
    });
  }
  return found;
}

/**
 * Every enemy the formations under a commander can see, merged.
 *
 * Until riders exist, every formation reports to its superior the instant it sees
 * anything — see `observe.ts` for why that is a stated rule rather than a leak. When two
 * of them see the same enemy, the better-informed sighting wins, which is what a
 * headquarters comparing two despatches would conclude.
 */
export function spottedUnder(
  state: CampaignState,
  world: World,
  cfg: CampaignConfig,
  commanderId: string,
): Map<string, Contact> {
  const merged = new Map<string, Contact>();

  for (const observer of formationsUnder(state, commanderId)) {
    for (const [id, contact] of spottedBy(state, world, cfg, observer)) {
      const held = merged.get(id);
      if (held === undefined || contact.intelLevel > held.intelLevel) merged.set(id, contact);
    }
  }
  return merged;
}

/**
 * Extra dice a unit adds to a patrol's or courier's roll when it is the one being passed.
 *
 * The rules' modifiers, shared by courier interception and patrol contact: a division is
 * harder to slip past than a detachment, cavalry harder than infantry, and a scouting
 * formation hardest of all because looking outward is what it is for.
 */
export function detectionDice(cfg: CampaignConfig, unit: Unit): number {
  let dice = 0;
  if (isDivision(unit)) dice += cfg.interceptDiceDivision;
  if (unit.kind === 'cavalry') dice += cfg.interceptDiceCavalry;
  if (hasTrait(unit, 'scout')) dice += cfg.interceptDiceScout;
  return dice;
}

/**
 * Whether gunfire at `from` can be heard at `to`.
 *
 * The rules give thirty kilometres, and add that forests and hills mute it. The muting is
 * not modelled: it is the one place the ruleset asks for terrain occlusion, and there is
 * no line-of-sight machinery in this engine to hang it on. Distance only, for now.
 */
export const hearsGunfire = (cfg: CampaignConfig, from: Hex, to: Hex): boolean =>
  distance(from, to) <= cfg.gunfireRangeKm;
