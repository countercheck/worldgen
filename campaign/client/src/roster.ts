/**
 * What the order-of-battle drawer shows: the chain of command, and the formations on it.
 *
 * Kept out of the component so it can be tested, because the interesting part is not the
 * drawing — it is the decision about *which* facts a given role is shown. A referee reads
 * formations; a commander reads their own formation and their memory of everyone else's, and
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

// ---- the chain of command ----------------------------------------------

/**
 * A formation as the tree shows it.
 *
 * `line` is what is known of it, and from where: live for a referee or for the formation a
 * commander rides with, dated when it came by rider, and null when nothing has been heard
 * at all — their superior's division, or a peer's, is a name in the order of battle and
 * nothing more. The name is always there; the facts are not.
 */
export interface FormationNode {
  readonly unitId: string;
  readonly name: string;
  readonly line: RosterLine | null;
  /**
   * The other officers riding with this formation, by name.
   *
   * Normally none. A corps commander whose headquarters has gone falls back on one of their
   * divisions, and the division then appears under both of them, each saying who else is
   * there — otherwise one of the two rows looks like the only one.
   */
  readonly alsoRiding: readonly string[];
  /** Detachments off it: patrols, which have no officer of their own. */
  readonly patrols: readonly FormationNode[];
}

/**
 * A commander, the formation they ride with, and everyone who answers to them.
 *
 * Two different relationships, drawn two different ways. `formation` is where they are —
 * the column they ride with, which is their eyes and their address. `subordinates` is who
 * they command. A marshal rides with one division and commands a corps, and a tree that
 * blurred the two would show the corps as belonging to the division.
 */
export interface CommandNode {
  readonly id: string;
  readonly name: string;
  readonly formation: FormationNode;
  readonly subordinates: readonly CommandNode[];
}

/** One side's chain of command, and whatever nobody commands. */
export interface CommandTree {
  readonly faction: string;
  /** Army command, normally one. More when a side has not been joined up yet. */
  readonly roots: readonly CommandNode[];
  /**
   * Formations with nobody riding with them, and patrols whose parent is gone.
   *
   * A hole in the chain of command rather than a piece on the board — nobody can order
   * them and nobody reports for them — so they are listed apart, where they cannot be
   * mistaken for part of anybody's command.
   */
  readonly uncommanded: readonly FormationNode[];
}

/** The commander fields the tree needs: `PublicCommander` fits, and so does `Commander`. */
interface Officer {
  readonly id: string;
  readonly name: string;
  readonly faction: string;
  readonly unitId: string;
  readonly unitName?: string;
  readonly superiorId: string | null;
}

const byName = <T extends { name: string; id?: string; unitId?: string }>(a: T, b: T): number =>
  a.name.localeCompare(b.name) || ((a.id ?? a.unitId ?? '') < (b.id ?? b.unitId ?? '') ? -1 : 1);

/**
 * The order of battle as a chain of command, one tree per side.
 *
 * Built from what the role holds and nothing else: live units, dated reports, and the
 * commanders of their own side. A referee passes every side and every unit; a commander
 * passes their own side, the one formation they ride with, and their reports — and a
 * formation they have no word of comes out with a name and a null `line`, never a guess.
 *
 * Total over bad data. A superior on the other side, or one who has been removed, makes a
 * commander a root rather than an orphan nobody can find; a loop in the chain, which
 * `check` refuses but an older log might hold, is broken at its lowest id rather than
 * walked forever.
 */
export function commandTrees(input: {
  /** The sides to build, in the order to show them. */
  factions: readonly string[];
  commanders: readonly Officer[];
  /** Live formations: all of them for a referee, their own for a commander. */
  units: readonly Unit[];
  /** Dated reports. Empty for a referee. */
  reports: readonly UnitReport[];
}): CommandTree[] {
  // Live beats remembered: a formation reporting itself would otherwise be known twice.
  const lines = new Map<string, RosterLine>();
  for (const r of input.reports) lines.set(r.unitId, fromReport(r));
  for (const u of input.units) lines.set(u.id, fromUnit(u));

  const patrolsOf = new Map<string, Unit[]>();
  for (const u of input.units) {
    if (u.parentUnitId === null) continue;
    patrolsOf.set(u.parentUnitId, [...(patrolsOf.get(u.parentUnitId) ?? []), u]);
  }

  const ridersOf = new Map<string, Officer[]>();
  for (const c of input.commanders) {
    ridersOf.set(c.unitId, [...(ridersOf.get(c.unitId) ?? []), c]);
  }

  const formation = (unitId: string, fallbackName: string, rider: string | null): FormationNode => {
    const line = lines.get(unitId) ?? null;
    return {
      unitId,
      name: line?.name ?? fallbackName,
      line,
      alsoRiding: (ridersOf.get(unitId) ?? [])
        .filter((c) => c.id !== rider)
        .map((c) => c.name)
        .sort(),
      patrols: (patrolsOf.get(unitId) ?? [])
        .map((p) => formation(p.id, p.name, null))
        .sort(byName),
    };
  };

  return input.factions.map((faction) => {
    const side = input.commanders.filter((c) => c.faction === faction);
    const ids = new Set(side.map((c) => c.id));
    const childrenOf = new Map<string, Officer[]>();
    for (const c of side) {
      if (c.superiorId === null || !ids.has(c.superiorId)) continue;
      childrenOf.set(c.superiorId, [...(childrenOf.get(c.superiorId) ?? []), c]);
    }

    const placed = new Set<string>();
    const node = (c: Officer): CommandNode => {
      placed.add(c.id);
      return {
        id: c.id,
        name: c.name,
        formation: formation(c.unitId, c.unitName ?? c.unitId, c.id),
        subordinates: (childrenOf.get(c.id) ?? [])
          .filter((sub) => !placed.has(sub.id))
          .sort(byName)
          .map(node),
      };
    };

    const roots = side
      .filter((c) => c.superiorId === null || !ids.has(c.superiorId))
      .sort(byName)
      .map(node);
    // Whatever a walk from the roots never reached is a loop. Break it at its lowest id.
    for (const c of [...side].sort((a, b) => (a.id < b.id ? -1 : 1))) {
      if (!placed.has(c.id)) roots.push(node(c));
    }

    const commanded = new Set(side.map((c) => c.unitId));
    const known = new Set(lines.keys());
    const uncommanded = [...lines.values()]
      .filter((l) => l.faction === faction && !commanded.has(l.unitId))
      .filter((l) => {
        // A patrol belongs under its parent, wherever that parent is listed. Only an
        // orphan — a patrol whose formation is gone — stands on its own.
        const parent = l.unit?.parentUnitId ?? null;
        return parent === null || !known.has(parent);
      })
      .map((l) => formation(l.unitId, l.name, null))
      .sort(byName);

    return { faction, roots, uncommanded };
  });
}
