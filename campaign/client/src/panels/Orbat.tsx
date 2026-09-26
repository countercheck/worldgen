/**
 * Raising formations and appointing officers — the referee's preparation for a game.
 *
 * Opened from the order of battle, at the place in the chain of command the new piece is
 * going: "+ subordinate" under a commander raises a formation and the officer at its head,
 * already answering to them; "+ officer" on a formation appoints somebody to ride with it.
 * So neither form asks which side or whose command — the tree already said so, by where the
 * referee clicked.
 *
 * Placing is done by pointing at the map, not by typing coordinates. A referee setting up
 * a scenario is looking at ground — a river, a road junction, the town they mean — and
 * asking them to read a hex off it and type two numbers is asking them to do the map's job.
 */

import { useState } from 'react';

import {
  ECHELON_MARKS,
  ECHELONS,
  EXPERIENCE_NAMES,
  EXPERIENCES,
  TRAITS,
  UNIT_KINDS,
  type CampaignConfig,
  type Commander,
  type Echelon,
  type Experience,
  type Hex,
  type UnitKind,
  type Unit,
} from '@campaign/shared';

import { copy, prettify } from '../copy.js';
import {
  commanderFrom,
  draftProblems,
  emptyDraft,
  idFor,
  unitFrom,
  type UnitDraft,
} from '../orbat.js';


/** A commander as these forms need one. */
interface Officer {
  readonly id: string;
  readonly name: string;
  readonly faction: string;
}

/**
 * Raise a formation, and the officer at its head, under a commander or as army command.
 *
 * One act, because a formation nobody commands cannot be ordered and cannot report — it is
 * a hole in the chain of command rather than a piece on the board.
 */
export function RaiseForm({
  faction,
  superior,
  units,
  commanders,
  cfg,
  placing,
  onPlace,
  onRaise,
  onCancel,
  busy,
}: {
  faction: string;
  /** Who the new officer answers to, or null for army command. */
  superior: Officer | null;
  units: readonly Unit[];
  commanders: readonly Officer[];
  cfg: CampaignConfig;
  /** The hex the referee last pointed at while this was open, if any. */
  placing: Hex | null;
  /** Ask for a hex. The drawer hides, the map takes a click, and `placing` comes back. */
  onPlace: () => void;
  /**
   * Raise the formation and appoint its commander.
   *
   * Two commands go to the server, in that order and only in that order — a commander must
   * have a formation to ride with before they can be appointed to it — but it is one thing
   * the referee did, and a failure halfway through leaves a formation with nobody at its
   * head, which the caller has to say out loud.
   */
  onRaise: (unit: Unit, commander: Commander) => void;
  onCancel: () => void;
  busy: boolean;
}) {
  const [draft, setDraft] = useState<UnitDraft>(() => ({
    ...emptyDraft(faction),
    superiorId: superior?.id ?? '',
  }));

  const taken = new Set(units.map((u) => u.id));
  const at = draft.at ?? placing;
  const id = draft.id === '' ? idFor(draft.name, taken) : draft.id;
  const problems = draftProblems({ ...draft, id, at }, cfg, taken);

  return (
    <div className="orbat-form">
      <p className="muted small">
        {superior === null ? copy.orbat.asArmyCommand : copy.orbat.under(superior.name)}
      </p>
      <div className="orbat-grid">
        <label>
          <span>{copy.orbat.commander}</span>
          <input
            value={draft.commanderName}
            placeholder={copy.orbat.commanderNamePlaceholder}
            onChange={(e) => setDraft({ ...draft, commanderName: e.target.value })}
            disabled={busy}
          />
        </label>

        <label>
          <span>{copy.orbat.ridingWith}</span>
          <input
            value={draft.name}
            placeholder={copy.orbat.namePlaceholder}
            onChange={(e) => setDraft({ ...draft, name: e.target.value })}
            disabled={busy}
          />
        </label>

        <label>
          <span>{copy.orbat.arm}</span>
          <select
            value={draft.kind}
            onChange={(e) => setDraft({ ...draft, kind: e.target.value as UnitKind })}
            disabled={busy}
          >
            {UNIT_KINDS.map((k) => (
              <option key={k} value={k}>
                {prettify(k)}
              </option>
            ))}
          </select>
        </label>

        <label>
          <span>{copy.orbat.echelon}</span>
          <select
            value={draft.echelon ?? ''}
            onChange={(e) =>
              setDraft({
                ...draft,
                echelon: e.target.value === '' ? null : (e.target.value as Echelon),
              })
            }
            disabled={busy}
          >
            {/* Left unset, the map guesses from strength and arm — which is right for a
                division and wrong for an army headquarters of a few hundred staff. */}
            <option value="">{copy.orbat.echelonGuessed}</option>
            {ECHELONS.filter((e) => e !== 'none').map((e) => (
              <option key={e} value={e}>
                {echelonLabel(e)}
              </option>
            ))}
          </select>
        </label>

        <label>
          <span>{copy.orbat.paperStrength}</span>
          <input
            type="number"
            min={0}
            step={100}
            value={draft.paperStrength}
            onChange={(e) =>
              setDraft({ ...draft, paperStrength: Number(e.target.value) })
            }
            disabled={busy}
          />
        </label>

        <label>
          <span>{copy.orbat.experience}</span>
          <select
            value={draft.experience}
            onChange={(e) =>
              setDraft({ ...draft, experience: Number(e.target.value) as Experience })
            }
            disabled={busy}
          >
            {EXPERIENCES.map((x) => (
              <option key={x} value={x}>
                {EXPERIENCE_NAMES[x]}
              </option>
            ))}
          </select>
        </label>

        <label>
          <span>{copy.orbat.corps}</span>
          <input
            value={draft.corps ?? ''}
            placeholder={copy.orbat.corpsPlaceholder}
            onChange={(e) =>
              setDraft({ ...draft, corps: e.target.value === '' ? null : e.target.value })
            }
            disabled={busy}
          />
        </label>
      </div>

      <div className="field">
        <div className="row-label">{copy.orbat.traits}</div>
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

      <div className="despatch-actions">
        <button onClick={onPlace} disabled={busy}>
          {at === null ? copy.orbat.pointAtGround : copy.orbat.standingAt(at.q, at.r)}
        </button>
        <button
          className="primary"
          disabled={busy || problems.length > 0 || at === null}
          onClick={() => {
            if (at === null) return;
            onRaise(
              unitFrom({ ...draft, id }, cfg, at),
              commanderFrom({ ...draft, id }, id, new Set(commanders.map((c) => c.id))),
            );
          }}
        >
          {copy.orbat.raise}
        </button>
        <button onClick={onCancel} disabled={busy}>
          {copy.orbat.cancel}
        </button>
      </div>

      {problems.length > 0 && (
        <ul className="muted small orbat-problems">
          {problems.map((p) => (
            <li key={p}>{p}</li>
          ))}
        </ul>
      )}
      {problems.length === 0 && (
        <p className="muted small">
          {copy.orbat.willBeRaisedBefore} <code>{id}</code>
          {copy.orbat.willBeRaisedAfter}
        </p>
      )}
    </div>
  );
}

/**
 * Appoint an officer to ride with a formation that already exists.
 *
 * To fill a vacancy — a formation whose commander was killed or relieved — or to put a
 * second officer with it: a corps commander whose own headquarters has gone.
 */
export function AppointForm({
  unit,
  commanders,
  onAppoint,
  onCancel,
  busy,
}: {
  unit: { readonly id: string; readonly name: string; readonly faction: string };
  commanders: readonly Officer[];
  onAppoint: (commander: Commander) => void;
  onCancel: () => void;
  busy: boolean;
}) {
  const [name, setName] = useState('');
  const [superiorId, setSuperiorId] = useState('');

  return (
    <div className="orbat-form">
      <p className="muted small">{copy.orbat.toRideWith(unit.name)}</p>
      <div className="orbat-grid">
        <label>
          <span>{copy.orbat.name}</span>
          <input
            value={name}
            placeholder={copy.orbat.commanderNamePlaceholder}
            onChange={(e) => setName(e.target.value)}
            disabled={busy}
          />
        </label>

        <label>
          <span>{copy.orbat.answersTo}</span>
          <select
            value={superiorId}
            onChange={(e) => setSuperiorId(e.target.value)}
            disabled={busy}
          >
            <option value="">{copy.orbat.noSuperior}</option>
            {/* Their own side only. A chain of command that crosses the lines is not a
                chain of command. */}
            {commanders
              .filter((c) => c.faction === unit.faction)
              .map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name}
                </option>
              ))}
          </select>
        </label>
      </div>

      <div className="despatch-actions">
        <button
          className="primary"
          disabled={busy || name.trim() === ''}
          onClick={() =>
            onAppoint({
              id: idFor(name, new Set(commanders.map((c) => c.id))),
              name: name.trim(),
              faction: unit.faction,
              unitId: unit.id,
              superiorId: superiorId === '' ? null : superiorId,
            })
          }
        >
          {copy.orbat.appoint}
        </button>
        <button onClick={onCancel} disabled={busy}>
          {copy.orbat.cancel}
        </button>
      </div>

      <p className="muted small">{copy.orbat.appointBlurb}</p>
    </div>
  );
}

/** The size mark a formation will carry, for a referee choosing an echelon. */
export const echelonLabel = (e: keyof typeof ECHELON_MARKS): string =>
  ECHELON_MARKS[e] === '' ? prettify(e) : `${prettify(e)} (${ECHELON_MARKS[e]})`;
