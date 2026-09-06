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

import { distance, type Hex } from './hex.js';
import type { Grade } from './config.js';
import type { Unit } from './unit.js';

/**
 * Column length in kilometres.
 *
 * The rules give this as `(Effectives * spacing + guns * 0.05) * spacing multiplier`,
 * with spacing in metres, so the whole bracket is metres and the result converts to km.
 *
 * The guns term reads oddly — five centimetres a gun is not a gun, and it was probably
 * meant as 0.05 km — but taken literally it reproduces both of the rules' worked
 * examples to the digit, and taken as kilometres it does not. Implemented as written;
 * if the intent turns out to be 50 m a gun, this is the one line to change.
 */
export function columnLengthKm(unit: Unit): number {
  const metres = (unit.effectives * unit.spacingM + unit.guns * 0.05) * unit.spacingMultiplier;
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
 * The hexes the column actually occupies: its head, and the path behind it as far as its
 * length reaches.
 *
 * `unit.column` is the marched path with the head first, and may be longer than the
 * column is — a unit that has marched twenty hexes still only occupies the last few. It
 * may also be shorter, when a unit has just been placed and has no history; then it
 * occupies what it has.
 */
export const occupied = (unit: Unit, grade: Grade = 'road'): Hex[] =>
  unit.column.slice(0, columnHexes(unit, grade));

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

/** Whether any part of the column stands on a hex. */
export const columnCovers = (unit: Unit, c: Hex, grade: Grade = 'road'): boolean =>
  occupied(unit, grade).some((h) => h.q === c.q && h.r === c.r);

/**
 * Whether the column is a connected chain of adjacent hexes.
 *
 * An invariant rather than a rule: a column that teleports mid-length is a bug in
 * whatever moved it. Referee teleports set the whole column at once and so stay
 * connected or are deliberately not.
 */
export const columnIsConnected = (unit: Unit): boolean =>
  unit.column.every((h, i) => i === 0 || distance(unit.column[i - 1]!, h) === 1);
