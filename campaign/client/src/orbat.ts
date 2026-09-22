/**
 * Building an order of battle.
 *
 * The referee's preparation for a game: both sides raised, placed, and only then handed
 * out as links. The shaping lives here rather than in the form so that what a new
 * formation *is* can be tested — a unit with the wrong spacing or a morale above its
 * ceiling is a bug nobody notices until a column turns out to be eighteen kilometres long.
 *
 * Everything derivable is derived. A referee raising a division says what it is and how
 * many troops it has; the speed it marches at, the space it takes on a road and the morale it
 * can hold all follow from the rules, and asking them for them would be asking them to
 * restate the tables they are already playing under.
 */

import {
  MAX_MORALE,
  type CampaignConfig,
  type Commander,
  type Echelon,
  type Experience,
  type Hex,
  type Trait,
  type Unit,
  type UnitKind,
} from '@campaign/shared';

import { copy } from './copy.js';

export interface UnitDraft {
  readonly id: string;
  readonly name: string;
  readonly faction: string;
  readonly kind: UnitKind;
  readonly paperStrength: number;
  readonly experience: Experience;
  readonly traits: readonly Trait[];
  readonly corps: string | null;
  readonly echelon: Echelon | null;
  readonly at: Hex | null;
  /**
   * Who commands it, by name, and who they answer to.
   *
   * Part of the draft rather than a second form, because a formation nobody commands
   * cannot be ordered and cannot report — it is a hole in the chain of command rather
   * than a piece on the board. Raising one and appointing to it is a single act here for
   * the same reason a unit is raised with its supply full: the referee is describing a
   * formation that exists, and a formation that exists has an officer at its head.
   */
  readonly commanderName: string;
  /** Empty for army command, which is where the first formation of a side necessarily sits. */
  readonly superiorId: string;
}

export const emptyDraft = (faction: string): UnitDraft => ({
  id: '',
  name: '',
  faction,
  kind: 'infantry',
  paperStrength: 5000,
  experience: 0,
  traits: [],
  corps: null,
  echelon: null,
  at: null,
  commanderName: '',
  superiorId: '',
});

/**
 * An id from a name, where the referee has not given one.
 *
 * Ids are typed into commands and read out of the log, so they want to be short and
 * legible rather than random. A suffix is added only when it has to be, because
 * "1re-division-2" is a worse name than "1re-division" and should not be paid for
 * speculatively.
 */
export function idFor(name: string, taken: ReadonlySet<string>): string {
  const base =
    name
      .toLowerCase()
      .normalize('NFD')
      .replace(/[̀-ͯ]/g, '')
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-|-$/g, '')
      .slice(0, 40) || copy.orbat.fallbackId;

  if (!taken.has(base)) return base;
  for (let n = 2; ; n++) {
    const candidate = `${base}-${n}`;
    if (!taken.has(candidate)) return candidate;
  }
}

/** Everything wrong with a draft, in the order a reader would find it. */
export function draftProblems(
  draft: UnitDraft,
  cfg: CampaignConfig,
  taken: ReadonlySet<string>,
): string[] {
  const out: string[] = [];
  if (draft.name.trim() === '') out.push(copy.orbat.needsName);
  if (draft.id !== '' && taken.has(draft.id)) out.push(copy.orbat.idTaken(draft.id));
  if (!Number.isFinite(draft.paperStrength) || draft.paperStrength < 0) {
    out.push(copy.orbat.strengthNegative);
  }
  if (draft.commanderName.trim() === '') out.push(copy.orbat.needsCommander);
  if (draft.at === null) out.push(copy.orbat.needsGround);
  return out;
}

/**
 * The formation a draft describes.
 *
 * Fatigue at nothing and supply full: a formation is raised fresh and marched into
 * trouble, rather than arriving already in it. Morale starts at its ceiling for the same
 * reason — the ceiling is what experience buys, and a unit that has not been fought has
 * not lost any of it.
 */
export function unitFrom(draft: UnitDraft, cfg: CampaignConfig, at: Hex): Unit {
  const defaults = cfg.kindDefaults[draft.kind];
  const morale = (cfg.maxMorale ?? MAX_MORALE)[draft.experience];

  return {
    id: draft.id,
    name: draft.name.trim(),
    faction: draft.faction,
    kind: draft.kind,
    paperStrength: draft.paperStrength,
    fatigue: 0,
    experience: draft.experience,
    morale,
    provisions: 40,
    maxProvisions: 40,
    equipment: 30,
    maxEquipment: 30,
    guns: draft.kind === 'artillery_reserve' ? 24 : 6,
    marchSpeedKmh: defaults.marchSpeedKmh,
    spacingM: defaults.spacingM,
    // A long tail is the trait's whole meaning, so raising a unit with it should produce
    // one rather than leaving the multiplier at 1 and the trait decorative.
    spacingMultiplier: draft.traits.includes('long_tail') ? 1.3 : 1,
    traits: draft.traits,
    formation: 'march',
    formationChange: null,
    column: [at],
    hoursMarchedToday: 0,
    corps: draft.corps,
    parentUnitId: null,
    ...(draft.echelon === null ? {} : { echelon: draft.echelon }),
  };
}

/** The commander a draft appoints to the formation it raises. */
export function commanderFrom(
  draft: UnitDraft,
  unitId: string,
  taken: ReadonlySet<string>,
): Commander {
  return {
    id: idFor(draft.commanderName, taken),
    name: draft.commanderName.trim(),
    faction: draft.faction,
    unitId,
    superiorId: draft.superiorId === '' ? null : draft.superiorId,
  };
}
