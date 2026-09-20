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
 * **Ordering and writing are different acts.** A commander may write to anyone on his own
 * side, but may only *order* those beneath him. The addressee list says which is which
 * before he chooses, rather than letting him discover it on a refusal.
 *
 * **The waypoints are the rider's, not the column's.** "Send my rider via Ligny" and
 * "march on Ligny" are unrelated routes and a form that blurred them would be misread the
 * first time somebody was in a hurry. So they are worded as an instruction to the man
 * carrying the paper, and they live next to the addressee rather than next to the text.
 *
 * **The estimate is his own guess and says so.** It is computed here, from his last
 * report of that formation, over the map he holds — never from the server, which knows
 * where the addressee actually is and must never let that leak back through a number.
 */

import { useEffect, useMemo, useState } from 'react';

import type { DespatchKind } from '@campaign/shared';

import { dayHour } from '../board.js';

import type { Correspondent, Estimate } from '../despatch.js';

export interface Draft {
  readonly to: string;
  readonly despatchKind: DespatchKind;
  readonly text: string;
  /** Who it is from. Absent for a commander, who can only be himself. */
  readonly from?: string;
}

export function Composer({
  correspondents,
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
  /**
   * How long a rider would take, where the asker is entitled to know.
   *
   * Absent for a commander, and that is the rule rather than an omission: he does not know
   * where the addressee is, so he cannot know how long the ride will be. A number here —
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
   * Only a referee has more than one. He runs most of the commanders on the map and takes
   * dictation from the players who hold the rest, so writing as a man is his ordinary work
   * rather than an impersonation — and the log records that he did it.
   */
  senders?: readonly { id: string; name: string }[];
  from?: string;
  onFrom?: (commanderId: string) => void;
}) {
  const [to, setTo] = useState(initial?.to ?? correspondents[0]?.id ?? '');
  const [text, setText] = useState(initial?.text ?? '');

  // The addressee list changes with the sender: a man may write to his own side only, and
  // order only those beneath him. Keeping a stale addressee would send Ney's order to
  // Wellington the moment the referee switched seats.
  useEffect(() => {
    if (!correspondents.some((c) => c.id === to)) setTo(correspondents[0]?.id ?? '');
  }, [correspondents, to]);

  const addressee = correspondents.find((c) => c.id === to) ?? null;
  // An order if he may give one, a message if he may not. Not a control: making the
  // reader choose between two words for the same box would only invite the wrong one.
  const despatchKind: DespatchKind =
    initial?.despatchKind ?? (addressee?.mayOrder === true ? 'order' : 'report');

  const estimate = useMemo(
    () => (to === '' || estimateFor === undefined ? null : estimateFor(to)),
    [to, estimateFor],
  );

  return (
    <section className="panel-section composer">
      <h3>Write a despatch</h3>

      {senders !== undefined && senders.length > 0 && (
        <label className="field">
          <div className="row-label">From</div>
          <select
            value={from ?? senders[0]?.id ?? ''}
            onChange={(e) => onFrom?.(e.target.value)}
            disabled={busy}
          >
            {senders.map((c) => (
              <option key={c.id} value={c.id}>
                {c.name}
              </option>
            ))}
          </select>
        </label>
      )}

      <label className="field">
        <div className="row-label">To</div>
        <select value={to} onChange={(e) => setTo(e.target.value)} disabled={busy}>
          {correspondents.map((c) => (
            <option key={c.id} value={c.id}>
              {c.name}
              {c.mayOrder ? '' : ' — message only'}
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
            ? 'Move on Quatre Bras with all speed; I expect you astride the crossroads by noon.'
            : 'What you have seen, and when you saw it.'
        }
        onChange={(e) => setText(e.target.value)}
        disabled={busy}
      />

      <p className="muted">
        {senders !== undefined
          ? 'Written in his name, and logged as yours. What the addressee makes of it is ' +
            'still a decision when it arrives.'
          : despatchKind === 'order'
            ? 'This is an order. The referee will read it and decide what your subordinate makes of it.'
            : 'This is a message. You may write to anyone on your own side; only those beneath you take orders.'}
      </p>

      {estimateFor === undefined ? (
        <p className="muted">
          How long he takes is not yours to know. He will ride until he finds them, and
          nobody will tell you when he did.
        </p>
      ) : estimate === null ? (
        <p className="muted">
          Nothing on the map to ride to yet. Send him anyway — he will find them.
        </p>
      ) : (
        <p className="muted guess">
          <strong>The ride:</strong> about {estimate.hours.toFixed(1)} h over the ground as
          it stands, arriving around {dayHour(clockHours + estimate.hours)}. He may be
          stopped on the way, and the sender is never told whether he was.
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
          {busy ? 'Sealing…' : 'Send by rider'}
        </button>
        <button onClick={onCancel} disabled={busy}>
          Cancel
        </button>
      </div>
    </section>
  );
}
