/**
 * Raising, placing and editing formations — the referee's preparation for a game.
 *
 * Kept in the order-of-battle drawer rather than the sidebar because it is the same work
 * as reading the order of battle: you look at what you have, and you add what is missing.
 * The sidebar is about the one formation under the cursor, and a form for creating a
 * second one does not belong there.
 *
 * Placing is done by pointing at the map, not by typing coordinates. A referee setting up
 * a scenario is looking at ground — a river, a road junction, the town they mean — and
 * asking them to read a hex off it and type two numbers is asking them to do the map's job.
 */

import { useState } from 'react';

import {
  ECHELON_MARKS,
  EXPERIENCE_NAMES,
  type CampaignConfig,
  type Commander,
  type Experience,
  type Hex,
  type PublicFaction,
  type Trait,
  type Unit,
  type UnitKind,
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

const KINDS: UnitKind[] = [
  'infantry',
  'cavalry',
  'hq',
  'artillery_reserve',
  'garrison',
  'convoy',
];

const TRAITS: Trait[] = [
  'scout',
  'heavy',
  'fast',
  'very_fast',
  'slow',
  'very_slow',
  'long_tail',
  'foraging',
  'pontooneers',
];

const EXPERIENCES: Experience[] = [-2, -1, 0, 1, 2];

export function Orbat({
  factions,
  units,
  commanders,
  cfg,
  placing,
  onPlace,
  onRaise,
  onAppoint,
  busy,
  error,
}: {
  factions: readonly PublicFaction[];
  units: readonly Unit[];
  commanders: readonly { id: string; name: string; faction: string; unitId: string }[];
  cfg: CampaignConfig;
  /** The hex the referee last pointed at while this was open, if any. */
  placing: Hex | null;
  /** Ask for a hex. The drawer closes, the map takes a click, and `placing` comes back. */
  onPlace: () => void;
  /**
   * Raise the formation and appoint its commander, as one act.
   *
   * Two commands go to the server, in that order and only in that order — a commander must
   * have a formation to ride with before they can be appointed to it — but it is one thing
   * the referee did, and a failure halfway through leaves a formation with nobody at its
   * head, which the caller has to say out loud.
   */
  onRaise: (unit: Unit, commander: Commander) => void;
  onAppoint: (commander: Commander) => void;
  busy: boolean;
  error: string | null;
}) {
  const [open, setOpen] = useState<'unit' | 'commander' | null>(null);
  const [draft, setDraft] = useState<UnitDraft>(() => emptyDraft(factions[0]?.id ?? 'red'));
  const [draftCommander, setDraftCommander] = useState({ name: '', unitId: '', superiorId: '' });

  const taken = new Set(units.map((u) => u.id));
  const at = draft.at ?? placing;
  const id = draft.id === '' ? idFor(draft.name, taken) : draft.id;
  const problems = draftProblems({ ...draft, id, at }, cfg, taken);

  const ofFaction = (faction: string) => units.filter((u) => u.faction === faction);

  return (
    <section className="orbat">
      <div className="despatch-actions">
        <button
          className={open === 'unit' ? 'primary' : ''}
          onClick={() => setOpen(open === 'unit' ? null : 'unit')}
        >
          {copy.orbat.raiseFormation}
        </button>
        <button
          className={open === 'commander' ? 'primary' : ''}
          onClick={() => setOpen(open === 'commander' ? null : 'commander')}
        >
          {copy.orbat.appointCommander}
        </button>
      </div>

      {open === 'unit' && (
        <div className="orbat-form">
          <div className="orbat-grid">
            <label>
              <span>{copy.orbat.name}</span>
              <input
                value={draft.name}
                placeholder={copy.orbat.namePlaceholder}
                onChange={(e) => setDraft({ ...draft, name: e.target.value })}
                disabled={busy}
              />
            </label>

            <label>
              <span>{copy.orbat.side}</span>
              <select
                value={draft.faction}
                onChange={(e) =>
                  // The superior goes with it. Keeping it would offer Wellington as Ney's
                  // superior for exactly as long as it took somebody to press Raise.
                  setDraft({ ...draft, faction: e.target.value, superiorId: '' })
                }
                disabled={busy}
              >
                {factions.map((f) => (
                  <option key={f.id} value={f.id}>
                    {f.name}
                  </option>
                ))}
              </select>
            </label>

            <label>
              <span>{copy.orbat.arm}</span>
              <select
                value={draft.kind}
                onChange={(e) => setDraft({ ...draft, kind: e.target.value as UnitKind })}
                disabled={busy}
              >
                {KINDS.map((k) => (
                  <option key={k} value={k}>
                    {prettify(k)}
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

            {/* Asked for here rather than in the appointment form below, because a
                formation nobody commands cannot be ordered and cannot report. The other
                form is for a second officer riding with an existing formation — a corps
                commander whose headquarters has gone — not for filling a vacancy this one
                left open. */}
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
              <span>{copy.orbat.answersTo}</span>
              <select
                value={draft.superiorId}
                onChange={(e) => setDraft({ ...draft, superiorId: e.target.value })}
                disabled={busy}
              >
                <option value="">{copy.orbat.noSuperior}</option>
                {/* Their own side only, and it changes with the side above: a chain of
                    command that crosses the lines is not a chain of command. */}
                {commanders
                  .filter((c) => c.faction === draft.faction)
                  .map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.name}
                    </option>
                  ))}
              </select>
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
                setDraft(emptyDraft(draft.faction));
              }}
            >
              {copy.orbat.raise}
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
      )}

      {open === 'commander' && (
        <div className="orbat-form">
          <div className="orbat-grid">
            <label>
              <span>{copy.orbat.name}</span>
              <input
                value={draftCommander.name}
                placeholder={copy.orbat.commanderNamePlaceholder}
                onChange={(e) => setDraftCommander({ ...draftCommander, name: e.target.value })}
                disabled={busy}
              />
            </label>

            <label>
              <span>{copy.orbat.ridesWith}</span>
              <select
                value={draftCommander.unitId}
                onChange={(e) => setDraftCommander({ ...draftCommander, unitId: e.target.value, superiorId: '' })}
                disabled={busy}
              >
                <option value="">{copy.orbat.none}</option>
                {factions.map((f) => (
                  <optgroup key={f.id} label={f.name}>
                    {ofFaction(f.id).map((u) => (
                      <option key={u.id} value={u.id}>
                        {u.name}
                      </option>
                    ))}
                  </optgroup>
                ))}
              </select>
            </label>

            <label>
              <span>{copy.orbat.answersTo}</span>
              <select
                value={draftCommander.superiorId}
                onChange={(e) => setDraftCommander({ ...draftCommander, superiorId: e.target.value })}
                disabled={busy || draftCommander.unitId === ''}
              >
                <option value="">{copy.orbat.noSuperior}</option>
                {/* Their own side only. A chain of command that crosses the lines is not a
                    chain of command. */}
                {commanders
                  .filter(
                    (c) =>
                      c.faction === units.find((u) => u.id === draftCommander.unitId)?.faction,
                  )
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
              disabled={busy || draftCommander.name.trim() === '' || draftCommander.unitId === ''}
              onClick={() => {
                const unit = units.find((u) => u.id === draftCommander.unitId);
                if (unit === undefined) return;
                onAppoint({
                  id: idFor(draftCommander.name, new Set(commanders.map((c) => c.id))),
                  name: draftCommander.name.trim(),
                  faction: unit.faction,
                  unitId: draftCommander.unitId,
                  superiorId: draftCommander.superiorId === '' ? null : draftCommander.superiorId,
                });
                setDraftCommander({ name: '', unitId: '', superiorId: '' });
              }}
            >
              {copy.orbat.appoint}
            </button>
          </div>

          <p className="muted small">{copy.orbat.appointBlurb}</p>
        </div>
      )}

      {error !== null && <p className="error">{error}</p>}
    </section>
  );
}

/** The size mark a formation will carry, for a referee choosing an echelon. */
export const echelonLabel = (e: keyof typeof ECHELON_MARKS): string =>
  ECHELON_MARKS[e] === '' ? e : `${e} (${ECHELON_MARKS[e]})`;
