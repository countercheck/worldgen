/**
 * The referee's hand on a unit: every stat, set outright.
 *
 * For what the engine did not see — the casualties of a fight settled at the table, a
 * night march the log has no record of, a division reinforced between sessions. Only the
 * values that were changed are sent, so an edit to morale does not also write back a
 * fatigue that moved while the form was open.
 */

import { useEffect, useState } from 'react';

import {
  ECHELONS,
  EXPERIENCE_NAMES,
  EXPERIENCES,
  FORMATIONS,
  isPatrol,
  roadHoursWithin,
  TRAITS,
  UNIT_KINDS,
  unitStatProblems,
  type Echelon,
  type Experience,
  type Formation,
  type Trait,
  type Unit,
  type UnitKind,
  type UnitStatChanges,
} from '@campaign/shared';

import { copy, prettify } from '../copy.js';

import { echelonLabel } from './Orbat.jsx';

/** The number fields, in the order the form shows them. */
const NUMBERS = [
  'paperStrength',
  'fatigue',
  'morale',
  'provisions',
  'maxProvisions',
  'equipment',
  'maxEquipment',
  'guns',
  'marchSpeedKmh',
  'spacingM',
  'spacingMultiplier',
  'hoursMarchedToday',
  'roadHoursLast24',
] as const;

type NumberField = (typeof NUMBERS)[number];

interface Draft {
  readonly name: string;
  readonly kind: UnitKind;
  readonly experience: Experience;
  readonly formation: Formation;
  /** The change under way: what it becomes, or '' for none, and hours from now to finish. */
  readonly changeTo: Formation | '';
  readonly changeIn: string;
  readonly echelon: Echelon | null;
  readonly corps: string;
  readonly traits: readonly Trait[];
  /** Held as typed, so a half-typed number is not snapped to something else. */
  readonly numbers: Readonly<Record<NumberField, string>>;
}

/** Two decimals at most: enough for hours and km/h, and what a referee would type. */
const shown = (n: number): string => String(Math.round(n * 100) / 100);

function draftOf(unit: Unit, clockHours: number): Draft {
  const current: Record<NumberField, number> = {
    paperStrength: unit.paperStrength,
    fatigue: unit.fatigue,
    morale: unit.morale,
    provisions: unit.provisions,
    maxProvisions: unit.maxProvisions,
    equipment: unit.equipment,
    maxEquipment: unit.maxEquipment,
    guns: unit.guns,
    marchSpeedKmh: unit.marchSpeedKmh,
    spacingM: unit.spacingM,
    spacingMultiplier: unit.spacingMultiplier,
    hoursMarchedToday: unit.hoursMarchedToday,
    roadHoursLast24: roadHoursWithin(unit, clockHours),
  };
  return {
    name: unit.name,
    kind: unit.kind,
    experience: unit.experience,
    formation: unit.formation,
    changeTo: unit.formationChange?.to ?? '',
    changeIn:
      unit.formationChange == null ? '' : shown(unit.formationChange.completesAtHours - clockHours),
    echelon: unit.echelon ?? null,
    corps: unit.corps ?? '',
    traits: unit.traits,
    numbers: Object.fromEntries(
      NUMBERS.map((f) => [f, shown(current[f])]),
    ) as Record<NumberField, string>,
  };
}

/** What the draft changes, against what the unit is now. Empty when nothing. */
function changesFrom(draft: Draft, was: Draft, clockHours: number): UnitStatChanges {
  const out: Record<string, unknown> = {};
  if (draft.name !== was.name) out.name = draft.name;
  if (draft.kind !== was.kind) out.kind = draft.kind;
  if (draft.experience !== was.experience) out.experience = draft.experience;
  if (draft.formation !== was.formation) out.formation = draft.formation;
  if (draft.changeTo !== was.changeTo || draft.changeIn !== was.changeIn) {
    out.formationChange =
      draft.changeTo === ''
        ? null
        : {
            to: draft.changeTo,
            completesAtHours:
              clockHours + (draft.changeIn.trim() === '' ? NaN : Number(draft.changeIn)),
          };
  }
  if (draft.echelon !== was.echelon && draft.echelon !== null) out.echelon = draft.echelon;
  if (draft.corps !== was.corps) out.corps = draft.corps.trim() === '' ? null : draft.corps;
  const traits = [...draft.traits].sort().join();
  if (traits !== [...was.traits].sort().join()) out.traits = draft.traits;
  for (const f of NUMBERS) {
    if (draft.numbers[f] !== was.numbers[f]) {
      // Blank is not zero: it is refused as not a number rather than quietly emptying a field.
      out[f] = draft.numbers[f].trim() === '' ? NaN : Number(draft.numbers[f]);
    }
  }
  return out as UnitStatChanges;
}

export function UnitEdit({
  unit,
  clockHours,
  formations,
  onSave,
  onReassign,
}: {
  unit: Unit;
  clockHours: number;
  /** The formations of its side a patrol could answer to. */
  formations: readonly Unit[];
  onSave: (changes: UnitStatChanges) => Promise<string | null>;
  /** Give a patrol to another formation. */
  onReassign: (parentUnitId: string) => Promise<string | null>;
}) {
  const was = draftOf(unit, clockHours);
  const held = JSON.stringify(was);
  const [draft, setDraft] = useState<Draft>(was);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setDraft(draftOf(unit, clockHours));
    // Keyed on the content: a fresh view carries an equal unit in a new object every time
    // the clock moves, and resetting on that would eat what the referee is typing.
  }, [held]);

  const [parent, setParent] = useState(unit.parentUnitId ?? '');
  useEffect(() => setParent(unit.parentUnitId ?? ''), [unit.parentUnitId]);

  const changes = changesFrom(draft, was, clockHours);
  const changed = Object.keys(changes).length > 0;
  const problems = unitStatProblems(changes);

  const number = (f: NumberField) => (
    <label key={f}>
      <span>{copy.unitEdit.fields[f]}</span>
      <input
        type="number"
        inputMode="decimal"
        value={draft.numbers[f]}
        disabled={busy}
        onChange={(e) => setDraft({ ...draft, numbers: { ...draft.numbers, [f]: e.target.value } })}
      />
    </label>
  );

  return (
    <details className="panel-section unit-edit">
      <summary>
        <h3>{copy.unitEdit.heading}</h3>
      </summary>
      <div className="hours-form">
        <p className="muted small">{copy.unitEdit.blurb}</p>
        <div className="hours-grid">
          <label>
            <span>{copy.unitEdit.fields.name}</span>
            <input
              value={draft.name}
              disabled={busy}
              onChange={(e) => setDraft({ ...draft, name: e.target.value })}
            />
          </label>
          <label>
            <span>{copy.unitEdit.fields.kind}</span>
            <select
              value={draft.kind}
              disabled={busy}
              onChange={(e) => setDraft({ ...draft, kind: e.target.value as UnitKind })}
            >
              {UNIT_KINDS.map((k) => (
                <option key={k} value={k}>
                  {prettify(k)}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>{copy.unitEdit.fields.echelon}</span>
            <select
              value={draft.echelon ?? ''}
              disabled={busy}
              onChange={(e) => setDraft({ ...draft, echelon: e.target.value as Echelon })}
            >
              {draft.echelon === null && <option value="">{copy.orbat.echelonGuessed}</option>}
              {ECHELONS.filter((e) => e !== 'none').map((e) => (
                <option key={e} value={e}>
                  {echelonLabel(e)}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>{copy.unitEdit.fields.experience}</span>
            <select
              value={draft.experience}
              disabled={busy}
              onChange={(e) =>
                setDraft({ ...draft, experience: Number(e.target.value) as Experience })
              }
            >
              {EXPERIENCES.map((x) => (
                <option key={x} value={x}>
                  {EXPERIENCE_NAMES[x]}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>{copy.unitEdit.fields.formation}</span>
            <select
              value={draft.formation}
              disabled={busy}
              onChange={(e) => setDraft({ ...draft, formation: e.target.value as Formation })}
            >
              {FORMATIONS.map((f) => (
                <option key={f} value={f}>
                  {prettify(f)}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>{copy.unitEdit.fields.changeTo}</span>
            <select
              value={draft.changeTo}
              disabled={busy}
              onChange={(e) => {
                const changeTo = e.target.value as Formation | '';
                // A change needs an hour to finish at: offer now, rather than a blank that
                // would be refused.
                const changeIn = changeTo === '' ? '' : draft.changeIn === '' ? '0' : draft.changeIn;
                setDraft({ ...draft, changeTo, changeIn });
              }}
            >
              <option value="">{copy.unitEdit.noChange}</option>
              {FORMATIONS.filter((f) => f !== draft.formation).map((f) => (
                <option key={f} value={f}>
                  {prettify(f)}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>{copy.unitEdit.fields.changeIn}</span>
            <input
              type="number"
              inputMode="decimal"
              min={0}
              value={draft.changeIn}
              disabled={busy || draft.changeTo === ''}
              onChange={(e) => setDraft({ ...draft, changeIn: e.target.value })}
            />
          </label>
          <label>
            <span>{copy.unitEdit.fields.corps}</span>
            <input
              value={draft.corps}
              placeholder={copy.orbat.corpsPlaceholder}
              disabled={busy}
              onChange={(e) => setDraft({ ...draft, corps: e.target.value })}
            />
          </label>
          {NUMBERS.map(number)}
        </div>

        <div className="field">
          <div className="row-label">{copy.unitEdit.fields.traits}</div>
          <div className="tags">
            {TRAITS.map((t) => (
              <button
                key={t}
                className={`tag${draft.traits.includes(t) ? ' on' : ''}`}
                disabled={busy}
                onClick={() =>
                  setDraft({
                    ...draft,
                    traits: draft.traits.includes(t)
                      ? draft.traits.filter((x) => x !== t)
                      : [...draft.traits, t],
                  })
                }
              >
                {prettify(t)}
              </button>
            ))}
          </div>
        </div>

        <p className="muted small">{copy.unitEdit.hoursHint}</p>

        {isPatrol(unit) && (
          <div className="hours-grid">
            <label>
              <span>{copy.unitEdit.fields.parent}</span>
              <select value={parent} disabled={busy} onChange={(e) => setParent(e.target.value)}>
                {formations.map((f) => (
                  <option key={f.id} value={f.id}>
                    {f.name}
                  </option>
                ))}
              </select>
            </label>
            <div className="despatch-actions">
              <button
                disabled={busy || parent === unit.parentUnitId}
                onClick={() => {
                  setBusy(true);
                  setError(null);
                  void onReassign(parent)
                    .then(setError, (err: Error) => setError(err.message))
                    .finally(() => setBusy(false));
                }}
              >
                {copy.unitEdit.reassign}
              </button>
            </div>
          </div>
        )}

        <div className="despatch-actions">
          <button
            className="primary"
            disabled={busy || !changed || problems.length > 0}
            onClick={() => {
              setBusy(true);
              setError(null);
              // A refusal comes back as a sentence; anything else comes back thrown, and
              // has to be shown just the same.
              void onSave(changes)
                .then(setError, (err: Error) => setError(err.message))
                .finally(() => setBusy(false));
            }}
          >
            {copy.unitEdit.save}
          </button>
          <button disabled={busy || !changed} onClick={() => setDraft(was)}>
            {copy.unitEdit.reset}
          </button>
        </div>

        {problems.length > 0 && (
          <ul className="muted small orbat-problems">
            {problems.map((p) => (
              <li key={p}>{p}</li>
            ))}
          </ul>
        )}
        {error !== null && <p className="bad small">{error}</p>}
      </div>
    </details>
  );
}
