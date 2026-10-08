/**
 * An order of battle, read from a YAML file.
 *
 * The same thing the Roster's raise form builds, many at once: a referee preparing a
 * forty-formation campaign writes it out once rather than clicking through forty forms.
 * The file is a tree, because the chain of command is one, and each node becomes the same
 * `UnitDraft` the form produces so a unit raised from a file is indistinguishable from
 * one raised by hand.
 *
 * Nothing is sent until the whole file is understood. A half-imported order of battle is
 * worse than none: it leaves the referee deleting formations one at a time to try again.
 * So every problem is collected, with where in the file it is, before any command exists.
 * The engine still has the last word — it may refuse what this accepts — but a typo is
 * caught here, all of them at once.
 */

import { parse } from 'yaml';

import {
  ECHELONS,
  EMPTY_STATE,
  EXPERIENCES,
  TRAITS,
  UNIT_KINDS,
  applyAll,
  hexAt,
  type CampaignConfig,
  type Command,
  type Commander,
  type Echelon,
  type Experience,
  type Faction,
  type Hex,
  type Trait,
  type UnitKind,
  type World,
} from '@campaign/shared';

import { copy } from './copy.js';
import { commanderFrom, emptyDraft, idFor, startingColor, unitFrom, type UnitDraft } from './orbat.js';

/** What the campaign already holds, so the file adds to it rather than colliding with it. */
export interface OobExisting {
  readonly factions: readonly Faction[];
  readonly unitIds: ReadonlySet<string>;
  readonly commanderIds: ReadonlySet<string>;
}

export const NOTHING_YET: OobExisting = {
  factions: [],
  unitIds: new Set(),
  commanderIds: new Set(),
};

export type OobResult =
  | {
      readonly ok: true;
      /** Sides the file names that the campaign does not have yet. */
      readonly newFactions: readonly Faction[];
      /**
       * Every unit, then every commander, superiors before those who answer to them — the
       * order the engine needs. Sides are not in here: a new campaign is created with them,
       * and an existing one is sent `add_faction` for `newFactions` first.
       */
      readonly commands: readonly Command[];
    }
  | { readonly ok: false; readonly problems: readonly string[] };

const SIDE_KEYS = ['name', 'id', 'color', 'army'] as const;
const NODE_KEYS = [
  'unit',
  'commander',
  'id',
  'commanderId',
  'kind',
  'strength',
  'guns',
  'experience',
  'echelon',
  'corps',
  'traits',
  'at',
  'subordinates',
] as const;

const COLOUR = /^#(?:[0-9a-f]{3}|[0-9a-f]{6})$/i;

type Mapping = Record<string, unknown>;
const isMapping = (v: unknown): v is Mapping =>
  typeof v === 'object' && v !== null && !Array.isArray(v);

/** `army:` and `subordinates:` take one formation or a list of them. */
const asList = (v: unknown): unknown[] => (Array.isArray(v) ? v : v == null ? [] : [v]);

export function parseOob(
  text: string,
  world: World,
  cfg: CampaignConfig,
  existing: OobExisting = NOTHING_YET,
): OobResult {
  const problems: string[] = [];
  const at = (path: readonly string[], problem: string): void => {
    problems.push(copy.oob.at(path.join(copy.oob.sep), problem));
  };

  let doc: unknown;
  try {
    doc = parse(text);
  } catch (err) {
    return { ok: false, problems: [copy.oob.notYaml(String((err as Error).message ?? err))] };
  }
  if (!isMapping(doc) || !Array.isArray(doc.sides)) {
    return { ok: false, problems: [copy.oob.needsSides] };
  }

  const known = new Map(existing.factions.map((f) => [f.id, f]));
  const sideIds = new Set<string>();
  const unitIds = new Set(existing.unitIds);
  const commanderIds = new Set(existing.commanderIds);
  const newFactions: Faction[] = [];
  const units: Command[] = [];
  const commanders: Command[] = [];

  const textOf = (node: Mapping, k: string, path: readonly string[]): string | undefined => {
    const v = node[k];
    if (v == null) return undefined;
    if (typeof v !== 'string' && typeof v !== 'number') {
      at(path, copy.oob.notText(k));
      return undefined;
    }
    return String(v).trim() || undefined;
  };

  const unknownKeys = (node: Mapping, allowed: readonly string[], path: readonly string[]) => {
    for (const k of Object.keys(node)) {
      if (!allowed.includes(k)) at(path, copy.oob.unknownKey(k, allowed));
    }
  };

  /**
   * An explicit id is checked; a generated one is made unique.
   *
   * A clash between two explicit ids is reported rather than silently suffixed: the
   * referee chose both and meant them.
   */
  const claim = (
    given: string | undefined,
    name: string,
    taken: Set<string>,
    path: readonly string[],
  ): string => {
    let id: string;
    if (given !== undefined) {
      id = given;
      if (taken.has(id)) {
        at(path, existing.unitIds.has(id) || existing.commanderIds.has(id) ? copy.oob.idTaken(id) : copy.oob.idTwice(id));
      }
    } else {
      id = idFor(name, taken);
    }
    taken.add(id);
    return id;
  };

  const formation = (
    raw: unknown,
    faction: string,
    superiorId: string,
    path: readonly string[],
  ): void => {
    if (!isMapping(raw)) {
      at(path, copy.oob.notAMapping);
      return;
    }
    const name = textOf(raw, 'unit', path);
    const commanderName = textOf(raw, 'commander', path);
    // Named once both are known, so the path reads as the file does.
    const here = [...path.slice(0, -1), name ?? commanderName ?? path[path.length - 1]!];
    unknownKeys(raw, NODE_KEYS, here);
    if (name === undefined) at(here, copy.oob.needs('unit'));
    if (commanderName === undefined) at(here, copy.oob.needs('commander'));

    const base = emptyDraft(faction);

    const kind = oneOf<UnitKind>(raw, 'kind', UNIT_KINDS, base.kind, here);
    const experience = oneOf<Experience>(raw, 'experience', EXPERIENCES, base.experience, here);
    const echelonRaw = raw.echelon == null ? null : oneOf<Echelon>(raw, 'echelon', ECHELONS, 'none', here);

    const wholeOr = <T,>(k: string, fallback: T): number | T => {
      const v = raw[k];
      if (v == null) return fallback;
      if (typeof v === 'number' && Number.isInteger(v) && v >= 0) return v;
      at(here, copy.oob.notWhole(k));
      return fallback;
    };
    const paperStrength = wholeOr('strength', base.paperStrength);
    // Any formation may have guns, not only an artillery reserve: a division's own
    // batteries are part of what it is, and the file is where a referee says how many.
    const guns = wholeOr('guns', base.guns);

    const traits: Trait[] = [];
    if (raw.traits != null) {
      if (!Array.isArray(raw.traits)) at(here, copy.oob.notAList);
      else {
        for (const t of raw.traits) {
          if ((TRAITS as readonly unknown[]).includes(t)) {
            if (!traits.includes(t as Trait)) traits.push(t as Trait);
          } else {
            at(here, copy.oob.notOneOf('traits', String(t), TRAITS));
          }
        }
      }
    }

    let hex: Hex | null = null;
    const a = raw.at;
    if (a == null) at(here, copy.oob.needs('at'));
    else if (!isMapping(a) || !Number.isInteger(a.q) || !Number.isInteger(a.r)) {
      at(here, copy.oob.notHex);
    } else {
      hex = { q: a.q as number, r: a.r as number };
      if (hexAt(world, hex) === undefined) at(here, copy.oob.offMap(hex.q, hex.r));
    }

    const id = claim(textOf(raw, 'id', here), name ?? '', unitIds, here);
    const draft: UnitDraft = {
      ...base,
      id,
      name: name ?? '',
      kind,
      paperStrength,
      experience,
      traits,
      corps: textOf(raw, 'corps', here) ?? null,
      echelon: echelonRaw,
      guns,
      at: hex,
      commanderName: commanderName ?? '',
      superiorId,
    };

    const givenCommanderId = textOf(raw, 'commanderId', here);
    let commander: Commander;
    if (givenCommanderId === undefined) {
      commander = commanderFrom(draft, id, commanderIds);
      commanderIds.add(commander.id);
    } else {
      commander = {
        ...commanderFrom(draft, id, commanderIds),
        id: claim(givenCommanderId, '', commanderIds, here),
      };
    }

    if (hex !== null) units.push({ kind: 'add_unit', unit: unitFrom(draft, cfg, hex) });
    commanders.push({ kind: 'add_commander', commander });

    if (raw.subordinates != null && !isMapping(raw.subordinates) && !Array.isArray(raw.subordinates)) {
      at(here, copy.oob.notAList);
    }
    asList(raw.subordinates).forEach((sub, i) =>
      formation(sub, faction, commander.id, [...here, copy.oob.formation(i + 1)]),
    );
  };

  /** A value that must be one of a fixed set, or the default when absent. */
  function oneOf<T>(
    node: Mapping,
    k: string,
    allowed: readonly T[],
    fallback: T,
    path: readonly string[],
  ): T {
    const v = node[k];
    if (v == null) return fallback;
    if ((allowed as readonly unknown[]).includes(v)) return v as T;
    at(path, copy.oob.notOneOf(k, String(v), allowed.map(String)));
    return fallback;
  }

  doc.sides.forEach((rawSide: unknown, i: number) => {
    const path = [copy.oob.side(i + 1)];
    if (!isMapping(rawSide)) {
      at(path, copy.oob.notAMapping);
      return;
    }
    const name = textOf(rawSide, 'name', path);
    const here = [name ?? path[0]!];
    unknownKeys(rawSide, SIDE_KEYS, here);
    if (name === undefined) at(here, copy.oob.needs('name'));

    // Matched by name too, so a file written for a campaign whose sides were added by hand
    // finds them without the referee having to look up their ids.
    const id =
      textOf(rawSide, 'id', here) ??
      existing.factions.find((f) => f.name === name)?.id ??
      idFor(name ?? '', new Set());
    if (sideIds.has(id)) at(here, copy.oob.sideTwice(id));
    sideIds.add(id);

    // A side the campaign already has is added to, not redefined: its colour is on every
    // marker already drawn, and a file is not the place to change it.
    if (!known.has(id)) {
      const color =
        textOf(rawSide, 'color', here) ??
        startingColor(existing.factions.length + newFactions.length);
      if (!COLOUR.test(color)) at(here, copy.oob.notColour(color));
      newFactions.push({ id, name: name ?? id, color });
    }

    if (rawSide.army != null && !isMapping(rawSide.army) && !Array.isArray(rawSide.army)) {
      at(here, copy.oob.notAList);
    }
    asList(rawSide.army).forEach((root, j) =>
      formation(root, id, '', [...here, copy.oob.formation(j + 1)]),
    );
  });

  if (problems.length > 0) return { ok: false, problems };
  return { ok: true, newFactions, commands: [...units, ...commanders] };
}

/**
 * What the engine would say to this order of battle in a new campaign, before there is one.
 *
 * The same engine the server runs, over the same commands, at the strictness a new
 * campaign starts at. Starting a campaign creates it first and fills it after, so a
 * refusal found only when the server sends it leaves a campaign half raised; found here,
 * nothing has been created yet and the referee fixes the file and tries again.
 */
export function rehearse(
  world: World,
  out: Extract<OobResult, { ok: true }>,
): readonly string[] {
  const run = applyAll(
    [
      {
        kind: 'create_campaign',
        name: '',
        world: {
          seed: world.seed,
          width: world.width,
          height: world.height,
          layout: world.layout,
          schemaVersion: world.schemaVersion,
          hash: '',
        },
        seed: 0,
      },
      ...out.newFactions.map((faction) => ({ kind: 'add_faction' as const, faction })),
      ...out.commands,
    ],
    EMPTY_STATE,
    world,
    'strict',
  );
  return run.violations.map((v) => v.message);
}
