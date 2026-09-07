/**
 * A `ClientView` turned into the things the console draws and reads.
 *
 * Pure, and separate from the components on purpose: this is where the three kinds of
 * thing on the map are told apart, and it is worth being able to test that without a
 * browser. Nothing here filters, hides or redacts — by the time a view reaches this
 * function everything in it is already something the holder is entitled to.
 *
 * The three kinds, and why they must not collapse into each other:
 *
 * - **A unit** is live and complete. A commander has exactly one: the formation he rides
 *   with. A referee has all of them.
 * - **A report** is a formation of his own as he last heard of it. It has an hour on it,
 *   and that hour is usually not now.
 * - **A contact** is an enemy somebody saw. It carries almost nothing, by design.
 *
 * An earlier version of the console fabricated a `Unit` for an enemy contact so the map
 * had something to draw. It worked perfectly and put the entire enemy order of battle in
 * the page. The same mistake is available here twice over now, so all three stay distinct
 * all the way to the screen.
 */

import {
  key,
  occupied,
  parseWorld,
  type ClientView,
  type Contact,
  type HexKey,
  type PublicCommander,
  type PublicFaction,
  type Theme,
  type Unit,
  type UnitReport,
  type World,
} from '@campaign/shared';

import type { Mark } from './map/draw.js';

export interface Board {
  readonly world: World;
  readonly marks: readonly Mark[];
  /** Live formations: one for a commander, all of them for a referee. */
  readonly units: ReadonlyMap<string, Unit>;
  /** Own formations as last reported. Empty for a referee, who has the units. */
  readonly reports: ReadonlyMap<string, UnitReport>;
  readonly contacts: ReadonlyMap<string, Contact>;
  readonly factions: ReadonlyMap<string, PublicFaction>;
  readonly commanders: ReadonlyMap<string, PublicCommander>;
  readonly surveyed: ReadonlySet<HexKey>;
  readonly visible: ReadonlySet<HexKey>;
}

export function boardFrom(view: ClientView, theme: Theme): Board {
  const world = parseWorld(view.world);
  const factions = new Map(view.factions.map((f) => [f.id, f]));
  const colour = (f: string): string => factions.get(f)?.color ?? theme.fallback;

  const units = new Map(view.units.map((u) => [u.id, u]));
  const reports = new Map(view.reports.map((r) => [r.unitId, r]));
  const contacts = new Map(view.contacts.map((c) => [c.unitId, c]));

  const marks: Mark[] = [
    // Live formations occupy the length of their column; the whole point of a division
    // here is that it is a line of ground rather than a counter on a hex.
    ...view.units.map((u) => ({
      id: u.id,
      column: occupied(u),
      color: colour(u.faction),
      kind: 'live' as const,
    })),
    // A reported formation is one hex: where it was said to be. Drawing its column would
    // claim knowledge of how it is strung out, which no despatch carries.
    ...view.reports.map((r) => ({
      id: r.unitId,
      column: [r.head],
      color: colour(r.faction),
      kind: 'reported' as const,
    })),
    // A contact is one hex too. Knowing an enemy was in a village is not knowing how far
    // back its baggage was; a column here would invent a report nobody made.
    ...view.contacts.map((c) => ({
      id: c.unitId,
      column: [c.coord],
      color: colour(c.faction),
      kind: 'contact' as const,
    })),
  ];

  return {
    world,
    marks,
    units,
    reports,
    contacts,
    factions,
    commanders: new Map(view.commanders.map((c) => [c.id, c])),
    surveyed: new Set(view.surveyed),
    visible: new Set(view.visible),
  };
}

/** How old a dated fact is, in campaign hours. Never negative. */
export const ageHours = (atHours: number, clockHours: number): number =>
  Math.max(0, clockHours - atHours);

/**
 * Age as a commander would read it.
 *
 * The gap between the hour a fact describes and the hour it is now is the whole of the
 * fog in this design, so it is worth one function that always says it the same way.
 */
export function ageLabel(atHours: number, clockHours: number): string {
  const age = ageHours(atHours, clockHours);
  if (age === 0) return 'now';
  if (age < 1) return `${Math.round(age * 60)} min ago`;
  return `${age.toFixed(age < 10 ? 1 : 0)} h ago`;
}

/** Whether a hex is currently watched, as opposed to merely covered at some point. */
export const isWatched = (board: Board, coord: { q: number; r: number }): boolean =>
  board.visible.has(key(coord));
