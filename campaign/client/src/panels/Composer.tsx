/**
 * Writing a despatch.
 *
 * Small on purpose. Orders in this game are prose — the referee reads what was written
 * and decides what the addressee makes of it — so there is nothing to build here beyond
 * an addressee, a box, and one instruction to the rider. Almost all the design effort of
 * a commander's interface belongs in the reading rather than the writing.
 *
 * Three things this form has to get right:
 *
 * **Ordering and writing are different acts.** A commander may write to anyone on their own
 * side, but may only *order* those beneath them. The addressee list says which is which
 * before they choose, rather than letting them discover it on a refusal.
 *
 * **The waypoints are the rider's, not the column's.** "Send my rider via Ligny" and
 * "march on Ligny" are unrelated routes and a form that blurred them would be misread the
 * first time somebody was in a hurry. So they are worded as an instruction to the rider
 * carrying the paper, and they live next to the addressee rather than next to the text.
 *
 * **The estimate is their own guess and says so.** It is computed here, from their last
 * report of that formation, over the map they hold — never from the server, which knows
 * where the addressee actually is and must never let that leak back through a number.
 */

import { useEffect, useMemo, useState } from 'react';

import type { DespatchKind } from '@campaign/shared';

import { dayHour } from '../board.js';
import { copy } from '../copy.js';

import type { Correspondent, Estimate } from '../despatch.js';

export interface Draft {
  readonly to: string;
  readonly despatchKind: DespatchKind;
  readonly text: string;
  /** Who it is from. Absent for a commander, who can only be themselves. */
  readonly from?: string;
}

export function Composer({
  correspondents,
  factionName,
  estimateFor,
  clockHours,
  busy,
  error,
  onSend,
  onCancel,
  initial,
  senders,
  from,
  onFrom,
}: {
  correspondents: readonly Correspondent[];
  /** A side's name from its id, so an addressee's affiliation reads rather than decodes. */
  factionName: (faction: string) => string;
  /**
   * How long a rider would take, where the asker is entitled to know.
   *
   * Absent for a commander, and that is the rule rather than an omission: they do not know
   * where the addressee is, so they cannot know how long the ride will be. A number here —
   * even a hedged one — is a distance, and a distance is a position.
   */
  estimateFor?: (commanderId: string) => Estimate | null;
  clockHours: number;
  busy: boolean;
  error: string | null;
  onSend: (draft: Draft) => void;
  onCancel: () => void;
  initial?: Partial<Draft>;
  /**
   * Whose name this may be written in.
   *
   * Only a referee has more than one. They run most of the commanders on the map and takes
   * dictation from the players who hold the rest, so writing as a commander is their ordinary
   * work rather than an impersonation — and the log records that they did it.
   */
  senders?: readonly { id: string; name: string; unitName: string; faction: string }[];
  from?: string;
  onFrom?: (commanderId: string) => void;
}) {
  const [to, setTo] = useState(initial?.to ?? correspondents[0]?.id ?? '');
  const [text, setText] = useState(initial?.text ?? '');

  // The addressee list changes with the sender: a commander may write to their own side only,
  // and order only those beneath them. Keeping a stale addressee would send Ney's order to
  // Wellington the moment the referee switched seats.
  useEffect(() => {
    if (!correspondents.some((c) => c.id === to)) setTo(correspondents[0]?.id ?? '');
  }, [correspondents, to]);

  const addressee = correspondents.find((c) => c.id === to) ?? null;
  // An order if they may give one, a message if they may not. Not a control: making the
  // reader choose between two words for the same box would only invite the wrong one.
  const despatchKind: DespatchKind =
    initial?.despatchKind ?? (addressee?.mayOrder === true ? 'order' : 'report');

  const estimate = useMemo(
    () => (to === '' || estimateFor === undefined ? null : estimateFor(to)),
    [to, estimateFor],
  );

  return (
    <section className="panel-section composer">
      <h3>{copy.composer.heading}</h3>

      {senders !== undefined && senders.length > 0 && (
        <label className="field">
          <div className="row-label">{copy.composer.from}</div>
          <select
            value={from ?? senders[0]?.id ?? ''}
            onChange={(e) => onFrom?.(e.target.value)}
            disabled={busy}
          >
            {senders.map((c) => (
              <option key={c.id} value={c.id}>
                {copy.composer.correspondent(c.name, c.unitName, factionName(c.faction))}
              </option>
            ))}
          </select>
        </label>
      )}

      <label className="field">
        <div className="row-label">{copy.composer.to}</div>
        <select value={to} onChange={(e) => setTo(e.target.value)} disabled={busy}>
          {correspondents.map((c) => (
            <option key={c.id} value={c.id}>
              {copy.composer.correspondent(c.name, c.unitName, factionName(c.faction))}
              {c.mayOrder ? '' : copy.composer.messageOnly}
            </option>
          ))}
        </select>
      </label>

      <textarea
        className="prose"
        rows={6}
        value={text}
        placeholder={
          despatchKind === 'order'
            ? copy.composer.orderPlaceholder
            : copy.composer.reportPlaceholder
        }
        onChange={(e) => setText(e.target.value)}
        disabled={busy}
      />

      <p className="muted">
        {senders !== undefined
          ? copy.composer.asReferee
          : despatchKind === 'order'
            ? copy.composer.isOrder
            : copy.composer.isMessage}
      </p>

      {estimateFor === undefined ? (
        <p className="muted">{copy.composer.noEstimate}</p>
      ) : estimate === null ? (
        <p className="muted">{copy.composer.nothingToRideTo}</p>
      ) : (
        <p className="muted guess">
          <strong>{copy.composer.rideLabel}</strong>
          {copy.composer.rideEstimate(
            estimate.hours.toFixed(1),
            dayHour(clockHours + estimate.hours),
          )}
        </p>
      )}

      {error !== null && <p className="error">{error}</p>}

      <div className="composer-actions">
        <button
          className="primary"
          disabled={busy || to === '' || text.trim() === ''}
          onClick={() =>
            onSend({ to, despatchKind, text, ...(from === undefined ? {} : { from }) })
          }
        >
          {busy ? copy.composer.sending : copy.composer.send}
        </button>
        <button onClick={onCancel} disabled={busy}>
          {copy.composer.cancel}
        </button>
      </div>
    </section>
  );
}
