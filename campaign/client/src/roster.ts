/**
 * What the order-of-battle drawer lists, and how it is grouped.
 *
 * Kept out of the component so it can be tested, because the interesting part is not the
 * table — it is the decision about *which* facts a given role is shown. A referee reads
 * formations; a commander reads his own formation and his memory of everyone else's, and
 * the two must never be built by filtering one list, because a filtered list is how the
 * remembered rows quietly become live ones.
 */

import type { Unit, UnitReport } from '@campaign/shared';

/** One row, from either source, reduced to what both can say. */
export interface RosterLine {
  readonly unitId: string;
  readonly name: string;
  readonly faction: string;
  readonly at: { q: number; r: number };
  readonly paperStrength: number;
  readonly fatigue: number;
  readonly formation: string;
  readonly corps: string | null;
  /** The hour this describes, or null when it is simply true now. */
  readonly asOfHours: number | null;
  /** Present for a live row. A report cannot carry one, and that is the whole difference. */
  readonly unit: Unit | null;
}

export interface RosterGroup {
  readonly title: string;
  readonly note: string | null;
  readonly lines: readonly RosterLine[];
}

const fromUnit = (u: Unit): RosterLine => ({
  unitId: u.id,
  name: u.name,
  faction: u.faction,
  at: u.column[0] ?? { q: 0, r: 0 },
  paperStrength: u.paperStrength,
  fatigue: u.fatigue,
  formation: u.formation,
  corps: u.corps,
  asOfHours: null,
  unit: u,
});

const fromReport = (r: UnitReport): RosterLine => ({
  unitId: r.unitId,
  name: r.name,
  faction: r.faction,
  at: r.head,
  paperStrength: r.paperStrength,
  fatigue: r.fatigue,
  formation: r.formation,
  corps: r.corps,
  asOfHours: r.atHours,
  unit: null,
});

export function rosterGroups(input: {
  role: 'referee' | 'commander';
  /** Live formations: all of them for a referee, his own for a commander. */
  units: readonly Unit[];
  /** Dated reports. Empty for a referee, who has no need of them. */
  reports: readonly UnitReport[];
  factionName: (id: string) => string;
}): RosterGroup[] {
  const live = [...input.units].sort((a, b) => (a.id < b.id ? -1 : 1)).map(fromUnit);

  // A report for a formation whose live state is already listed would be the same
  // formation twice, once true and once remembered — and a reader would have no way to
  // tell which of the two rows to believe.
  const seen = new Set(live.map((l) => l.unitId));
  const heard = input.reports
    .filter((r) => !seen.has(r.unitId))
    .sort((a, b) => (a.unitId < b.unitId ? -1 : 1))
    .map(fromReport);

  if (input.role === 'referee') {
    return [...new Set(live.map((l) => l.faction))].sort().map((f) => ({
      title: input.factionName(f),
      note: null,
      lines: live.filter((l) => l.faction === f),
    }));
  }

  return [
    { title: 'With you', note: null, lines: live },
    {
      title: 'Under your command',
      note:
        heard.length === 0
          ? null
          : 'As you last heard. Every hour below is when word reached you, not where they ' +
            'are now.',
      lines: heard,
    },
  ];
}
