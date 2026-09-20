/**
 * Raising, placing and editing formations — the referee's preparation for a game.
 *
 * Kept in the order-of-battle drawer rather than the sidebar because it is the same work
 * as reading the order of battle: you look at what you have, and you add what is missing.
 * The sidebar is about the one formation under the cursor, and a form for creating a
 * second one does not belong there.
 *
 * Placing is done by pointing at the map, not by typing coordinates. A referee setting up
 * a scenario is looking at ground — a river, a road junction, the town he means — and
 * asking him to read a hex off it and type two numbers is asking him to do the map's job.
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

import { draftProblems, emptyDraft, idFor, unitFrom, type UnitDraft } from '../orbat.js';

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
  onRaise: (unit: Unit) => void;
  onAppoint: (commander: Commander) => void;
  busy: boolean;
  error: string | null;
}) {
  const [open, setOpen] = useState<'unit' | 'commander' | null>(null);
  const [draft, setDraft] = useState<UnitDraft>(() => emptyDraft(factions[0]?.id ?? 'red'));
  const [man, setMan] = useState({ name: '', unitId: '', superiorId: '' });

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
          Raise a formation
        </button>
        <button
          className={open === 'commander' ? 'primary' : ''}
          onClick={() => setOpen(open === 'commander' ? null : 'commander')}
        >
          Appoint a commander
        </button>
      </div>

      {open === 'unit' && (
        <div className="orbat-form">
          <div className="orbat-grid">
            <label>
              <span>Name</span>
              <input
                value={draft.name}
                placeholder="1re Division"
                onChange={(e) => setDraft({ ...draft, name: e.target.value })}
                disabled={busy}
              />
            </label>

            <label>
              <span>Side</span>
              <select
                value={draft.faction}
                onChange={(e) => setDraft({ ...draft, faction: e.target.value })}
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
              <span>Arm</span>
              <select
                value={draft.kind}
                onChange={(e) => setDraft({ ...draft, kind: e.target.value as UnitKind })}
                disabled={busy}
              >
                {KINDS.map((k) => (
                  <option key={k} value={k}>
                    {k.replace(/_/g, ' ')}
                  </option>
                ))}
              </select>
            </label>

            <label>
              <span>Paper strength</span>
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
              <span>Experience</span>
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
              <span>Corps</span>
              <input
                value={draft.corps ?? ''}
                placeholder="I Corps"
                onChange={(e) =>
                  setDraft({ ...draft, corps: e.target.value === '' ? null : e.target.value })
                }
                disabled={busy}
              />
            </label>
          </div>

          <div className="field">
            <div className="row-label">Traits</div>
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
                  {t.replace(/_/g, ' ')}
                </button>
              ))}
            </div>
          </div>

          <div className="despatch-actions">
            <button onClick={onPlace} disabled={busy}>
              {at === null ? 'Point at the ground' : `Standing at ${at.q}, ${at.r} — move`}
            </button>
            <button
              className="primary"
              disabled={busy || problems.length > 0 || at === null}
              onClick={() => {
                if (at === null) return;
                onRaise(unitFrom({ ...draft, id }, cfg, at));
                setDraft(emptyDraft(draft.faction));
              }}
            >
              Raise
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
              It will be raised as <code>{id}</code>, fresh and fully supplied.
            </p>
          )}
        </div>
      )}

      {open === 'commander' && (
        <div className="orbat-form">
          <div className="orbat-grid">
            <label>
              <span>Name</span>
              <input
                value={man.name}
                placeholder="Marshal Ney"
                onChange={(e) => setMan({ ...man, name: e.target.value })}
                disabled={busy}
              />
            </label>

            <label>
              <span>Rides with</span>
              <select
                value={man.unitId}
                onChange={(e) => setMan({ ...man, unitId: e.target.value, superiorId: '' })}
                disabled={busy}
              >
                <option value="">—</option>
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
              <span>Answers to</span>
              <select
                value={man.superiorId}
                onChange={(e) => setMan({ ...man, superiorId: e.target.value })}
                disabled={busy || man.unitId === ''}
              >
                <option value="">nobody — army command</option>
                {/* His own side only. A chain of command that crosses the lines is not a
                    chain of command. */}
                {commanders
                  .filter(
                    (c) =>
                      c.faction === units.find((u) => u.id === man.unitId)?.faction,
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
              disabled={busy || man.name.trim() === '' || man.unitId === ''}
              onClick={() => {
                const unit = units.find((u) => u.id === man.unitId);
                if (unit === undefined) return;
                onAppoint({
                  id: idFor(man.name, new Set(commanders.map((c) => c.id))),
                  name: man.name.trim(),
                  faction: unit.faction,
                  unitId: man.unitId,
                  superiorId: man.superiorId === '' ? null : man.superiorId,
                  // Run by the referee until a seat is issued for him, so an arriving
                  // order cascades rather than waiting on a player who does not exist yet.
                  autoCascade: true,
                });
                setMan({ name: '', unitId: '', superiorId: '' });
              }}
            >
              Appoint
            </button>
          </div>

          <p className="muted small">
            Appointing a man does not give anybody a seat. Issue him a link when you want
            somebody to play him.
          </p>
        </div>
      )}

      {error !== null && <p className="error">{error}</p>}
    </section>
  );
}

/** The size mark a formation will carry, for a referee choosing an echelon. */
export const echelonLabel = (e: keyof typeof ECHELON_MARKS): string =>
  ECHELON_MARKS[e] === '' ? e : `${e} (${ECHELON_MARKS[e]})`;
