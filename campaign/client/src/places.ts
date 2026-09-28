/**
 * The places list: every settlement the viewer's map holds, grouped by size.
 *
 * What is listed is what the world carries, and nothing more. A commander's world has
 * already been masked by the server (`mask.ts`) down to the places their side knows of,
 * so a list built from it cannot tell them about a town they have never heard of.
 */

import type { Settlement } from '@campaign/shared';

export interface PlaceGroup {
  readonly tier: string;
  readonly places: readonly Settlement[];
}

/** Biggest first: a reader looking for somewhere usually means a town, not a hamlet. */
const TIER_ORDER = ['city', 'town', 'village'];

/**
 * Lower-case with the accents taken off, so "orleans" finds Orléans and "dun" finds Dûn.
 * A reader typing a name they heard read aloud cannot be expected to type its diacritics.
 */
export function foldName(s: string): string {
  return s.normalize('NFD').replace(/\p{Diacritic}/gu, '').toLowerCase();
}

/**
 * The settlements matching *filter*, grouped by tier and sorted by name within each.
 *
 * A tier the generator adds later still shows, after the three it knows, rather than
 * vanishing from the list. Empty groups are dropped.
 */
export function placeGroups(settlements: readonly Settlement[], filter: string): PlaceGroup[] {
  const wanted = foldName(filter.trim());
  const byTier = new Map<string, Settlement[]>();
  for (const s of settlements) {
    if (wanted !== '' && !foldName(s.name).includes(wanted)) continue;
    const list = byTier.get(s.tier);
    if (list === undefined) byTier.set(s.tier, [s]);
    else list.push(s);
  }
  const rank = (tier: string): number => {
    const i = TIER_ORDER.indexOf(tier);
    return i === -1 ? TIER_ORDER.length : i;
  };
  return [...byTier.entries()]
    .sort(([a], [b]) => rank(a) - rank(b) || a.localeCompare(b))
    .map(([tier, places]) => ({
      tier,
      places: [...places].sort((a, b) => a.name.localeCompare(b.name)),
    }));
}
