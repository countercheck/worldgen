/**
 * A `ClientView` turned into the things the console draws and reads.
 *
 * Pure, and separate from the components on purpose: this is where the distinction
 * between a unit and a contact is made, and it is worth being able to test it without a
 * browser. The server has already decided what may be in the view — nothing here filters,
 * hides or redacts anything, because by the time a view reaches this function everything
 * in it is already something the holder is entitled to.
 *
 * The one rule that matters: a contact never becomes a `Unit`. An earlier version of the
 * console faked one out of the enemy's real record so the map could draw it, which worked
 * perfectly and meant the whole enemy order of battle was in the page. Contacts stay
 * contacts, and the sidebar has a different panel for them.
 */

import {
  key,
  occupied,
  parseWorld,
  type ClientView,
  type Contact,
  type HexKey,
  type PublicFaction,
  type Theme,
  type Unit,
  type World,
} from '@campaign/shared';

import type { Mark } from './map/draw.js';

export interface Board {
  /** The world as this role knows it: ground truth for a referee, masked otherwise. */
  readonly world: World;
  readonly marks: readonly Mark[];
  readonly units: ReadonlyMap<string, Unit>;
  readonly contacts: ReadonlyMap<string, Contact>;
  readonly factions: ReadonlyMap<string, PublicFaction>;
  readonly seen: ReadonlySet<HexKey>;
  readonly visible: ReadonlySet<HexKey>;
}

export function boardFrom(view: ClientView, theme: Theme): Board {
  const world = parseWorld(view.world);
  const factions = new Map(view.factions.map((f) => [f.id, f]));
  const colour = (id: string): string => factions.get(id)?.color ?? theme.fallback;

  const units = new Map(view.units.map((u) => [u.id, u]));
  const contacts = new Map(view.contacts.map((c) => [c.unitId, c]));

  const marks: Mark[] = [
    // Own units occupy the length of their column; the whole point of a division here is
    // that it is a line of ground rather than a counter on a hex.
    ...view.units.map((u) => ({
      id: u.id,
      column: occupied(u),
      color: colour(u.faction),
      visible: true,
    })),
    // A contact is one hex: where it was seen. Knowing an enemy was in a village is not
    // knowing how far back its baggage was strung out, and drawing a column there would
    // invent an intelligence report nobody made.
    ...view.contacts.map((c) => ({
      id: c.unitId,
      column: [c.coord],
      color: colour(c.faction),
      visible: false,
    })),
  ];

  return {
    world,
    marks,
    units,
    contacts,
    factions,
    seen: new Set(view.seen),
    visible: new Set(view.visible),
  };
}

/** How stale a sighting is, in campaign hours. */
export const contactAgeHours = (contact: Contact, clockHours: number): number =>
  Math.max(0, clockHours - contact.seenAtHours);

/** Whether a hex is currently watched, as opposed to merely remembered. */
export const isWatched = (board: Board, coord: { q: number; r: number }): boolean =>
  board.visible.has(key(coord));
