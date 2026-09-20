/**
 * The column: a division's physical length on the ground.
 *
 * This is what makes the game unusual. A division of 4,000 at half a metre a man is two
 * kilometres of road before any baggage; a cavalry division at three metres a horse is
 * twelve. On a 1 km hex grid that means a unit is not a marker on a hex — it is a line of
 * hexes, and its tail is still in the last village when its head reaches the next.
 *
 * Two consequences the rules lean on. The column is exposed along its whole length, so
 * recon and interception both work against the path rather than against a point. And the
 * tail has to catch up before the unit is concentrated enough to do anything, which is
 * `catchupHours` — 4.5 hours for the cavalry division in the rules' own example.
 */

import { DIRECTIONS, distance, key, rotate60, type Hex } from './hex.js';
import type { Grade } from './config.js';
import { presentUnderArms, type Formation, type Unit } from './unit.js';

/**
 * Column length in kilometres.
 *
 * The rules give this as `(PaperStrength * spacing + guns * 0.05) * spacing multiplier`,
 * with spacing in metres, so the whole bracket is metres and the result converts to km.
 *
 * The guns term reads oddly — five centimetres a gun is not a gun, and it was probably
 * meant as 0.05 km — but taken literally it reproduces both of the rules' worked
 * examples to the digit, and taken as kilometres it does not. Implemented as written;
 * if the intent turns out to be 50 m a gun, this is the one line to change.
 */
export function columnLengthKm(unit: Unit): number {
  const metres = (unit.paperStrength * unit.spacingM + unit.guns * 0.05) * unit.spacingMultiplier;
  return metres / 1000;
}

/**
 * How long the rear takes to reach where the head is now.
 *
 * The rules' `Column Length / March Speed`. What a commander pays to concentrate.
 */
export const catchupHours = (unit: Unit, speedKmh: number): number =>
  speedKmh <= 0 ? Infinity : columnLengthKm(unit) / speedKmh;

/**
 * Highway march halves the column, per the rules' movement table.
 *
 * A metalled road takes a column several abreast where a track takes it in file, so the
 * same men occupy half the distance. Applied when asking how much ground a unit covers,
 * never stored — the unit has not changed, only the road it is on.
 */
export const effectiveLengthKm = (unit: Unit, grade: Grade): number =>
  grade === 'highway' ? columnLengthKm(unit) / 2 : columnLengthKm(unit);

/**
 * How many hexes the column covers, at 1 km per hex.
 *
 * Rounded up, and never less than one: a unit is always somewhere, and a column that
 * spills a hundred metres into the next hex is still in it.
 */
export const columnHexes = (unit: Unit, grade: Grade = 'road'): number =>
  Math.max(1, Math.ceil(effectiveLengthKm(unit, grade)));

/**
 * The ground a formation stands on, as a fold of its column of march.
 *
 * A column is long and thin because it is on a road. Everything it does other than march
 * gathers it up: the same men off the road take less length and more width, which is why
 * the tail needs `catchupHours` to come in before the unit is concentrated. The footprint
 * is that fold expressed on the grid — divide the march length by `foldsInto`, then stand
 * `widthHexes` abreast.
 */
export interface FootprintShape {
  /** How many hexes of march column gather into one hex of frontage. */
  readonly foldsInto: number;
  /** How many hexes across the formation stands. 1 is a file: a line, as marched. */
  readonly widthHexes: number;
  /**
   * Men per hex of frontage, where the ground comes from the formation's strength rather
   * than from the length of its column. Null to fold the column instead.
   *
   * Deployment does not care how long the road was. A division that arrives strung over
   * eighteen kilometres and one that arrives concentrated form the same line, because
   * what sets a frontage is how many men there are to stand in it — about a kilometre per
   * ten thousand. Folding the column would make the cavalry division four times the
   * frontage of the infantry division beside it, which is an artefact of horse spacing on
   * a road and has nothing to do with deploying.
   */
  readonly menPerHex: number | null;
}

/**
 * What each formation folds into. See `cfg.footprint` for the live table.
 *
 * Camp is the one that matters so far. A division halts and builds one, and the two hours
 * it costs are the men coming off the road and pitching: a quarter the length, twice the
 * width. Twelve kilometres of cavalry column becomes a camp three hexes long and two
 * across, which is a thicker line than the tail it replaced, and that is what a bivouac
 * looks like from a hilltop.
 *
 * Occupation folds harder and stays narrow, because it is a garrison gone into a town
 * rather than a formation standing in a field.
 *
 * Battle is measured the other way, from strength: a kilometre of frontage per ten
 * thousand men. That makes an ordinary division a single hex, which is why deployment
 * needs no facing — there is no shape on this grid to orient. What happens inside that
 * hex is below the resolution of a 1 km map and belongs to whatever resolves a battle.
 *
 * March and rout are the column as marched.
 */
export const FOOTPRINT: Readonly<Record<Formation, FootprintShape>> = {
  march: { foldsInto: 1, widthHexes: 1, menPerHex: null },
  battle: { foldsInto: 1, widthHexes: 1, menPerHex: 10000 },
  rest: { foldsInto: 4, widthHexes: 2, menPerHex: null },
  occupation: { foldsInto: 8, widthHexes: 1, menPerHex: null },
  rout: { foldsInto: 1, widthHexes: 1, menPerHex: null },
};

/**
 * How many hexes of the marched path the formation still stretches along.
 *
 * Always at least one: a unit is always somewhere.
 */
export const spineHexes = (
  unit: Unit,
  grade: Grade = 'road',
  shapes: Readonly<Record<Formation, FootprintShape>> = FOOTPRINT,
): number => {
  const shape = shapes[unit.formation] ?? FOOTPRINT.march;

  // Frontage is measured in men, not in road. Present under arms rather than paper
  // strength, because what sets a frontage is who is standing in the line: the rules
  // already define the fatigued remainder as the number that fights, and a division worn
  // down to half covers half the ground.
  if (shape.menPerHex != null && shape.menPerHex > 0) {
    return Math.max(1, Math.ceil(presentUnderArms(unit) / shape.menPerHex));
  }
  return Math.max(1, Math.ceil(columnHexes(unit, grade) / Math.max(1, shape.foldsInto)));
};

/**
 * Which way the column was heading at the `i`th hex of its path.
 *
 * `unit.column` is head-first, so the hex behind is the one *after* in the array and the
 * direction of march points away from it. Derived from the path rather than stored: a
 * facing that was written down once would be wrong the moment the column turned a corner,
 * and the path already knows.
 */
function marchDirection(path: readonly Hex[], i: number): Hex {
  const here = path[i];
  const behind = path[i + 1];
  const ahead = path[i - 1];
  if (here === undefined) return DIRECTIONS[0]!;
  if (behind !== undefined) return { q: here.q - behind.q, r: here.r - behind.r };
  if (ahead !== undefined) return { q: ahead.q - here.q, r: ahead.r - here.r };
  // A unit just placed, with no path behind it. It has no direction of march, so the
  // flank is arbitrary — but it must be the *same* arbitrary one on every replay.
  return DIRECTIONS[0]!;
}

/**
 * The hexes the formation actually occupies.
 *
 * `unit.column` is the marched path with the head first, and may be longer than the unit
 * is — a unit that has marched twenty hexes still only occupies the last few. It may also
 * be shorter, when a unit has just been placed and has no history; then it occupies what
 * it has.
 *
 * The head is always first in the result, because callers read index 0 as the tip of the
 * column and a formation is engaged at its tip whatever shape it is standing in.
 *
 * Widening is geometric, not geographic: flank hexes are not checked for water or for
 * being on the map. A camp pitched on a lake shore will claim a hex of the lake, and the
 * rules that care — crossing, passability — test the ground themselves. Worth revisiting
 * the day a camp is attacked across the water it is standing in.
 */
export function occupied(
  unit: Unit,
  grade: Grade = 'road',
  shapes: Readonly<Record<Formation, FootprintShape>> = FOOTPRINT,
): Hex[] {
  const shape = shapes[unit.formation] ?? FOOTPRINT.march;
  const spine = unit.column.slice(0, spineHexes(unit, grade, shapes));
  if (shape.widthHexes <= 1) return spine;

  // Insertion-ordered so the head stays at index 0, and keyed so a flank hex that lands
  // on the spine — which happens wherever the column turned — is counted once.
  const out = new Map<string, Hex>();
  for (const h of spine) out.set(key(h), h);

  for (let i = 0; i < spine.length; i++) {
    const at = spine[i]!;
    const flank = rotate60(marchDirection(unit.column, i));
    for (let w = 1; w < shape.widthHexes; w++) {
      const h = { q: at.q + flank.q * w, r: at.r + flank.r * w };
      out.set(key(h), h);
    }
  }
  return [...out.values()];
}

/**
 * Push a new head onto the column, keeping only as much tail as the unit is long.
 *
 * Trimming here rather than on read keeps the stored path bounded — a unit marching for
 * a week would otherwise accumulate an unbounded history that serialises into every
 * save.
 */
export function advanceColumn(unit: Unit, to: Hex, grade: Grade = 'road'): Hex[] {
  // Cap the retained path a little beyond the column's own length, so that a unit
  // marching onto a highway (which halves its length) and back off again does not
  // permanently lose the tail it would have had.
  const keep = Math.max(columnHexes(unit, grade), columnHexes(unit, 'road')) + 1;
  return [to, ...unit.column].slice(0, keep);
}

/** Whether any part of the formation stands on a hex. */
export const columnCovers = (
  unit: Unit,
  c: Hex,
  grade: Grade = 'road',
  shapes: Readonly<Record<Formation, FootprintShape>> = FOOTPRINT,
): boolean => occupied(unit, grade, shapes).some((h) => h.q === c.q && h.r === c.r);

/**
 * Whether the column is a connected chain of adjacent hexes.
 *
 * An invariant rather than a rule: a column that teleports mid-length is a bug in
 * whatever moved it. Referee teleports set the whole column at once and so stay
 * connected or are deliberately not.
 */
export const columnIsConnected = (unit: Unit): boolean =>
  unit.column.every((h, i) => i === 0 || distance(unit.column[i - 1]!, h) === 1);
