/**
 * Writing a despatch.
 *
 * Small on purpose. A despatch in this game is prose — orders are whatever is written in
 * one, and the referee reads it and decides what the addressee makes of it — so there is
 * nothing to build here beyond an addressee, a box, and one instruction to the rider.
 * Almost all the design effort of a commander's interface belongs in the reading rather
 * than the writing.
 *
 * Three things this form has to get right:
 *
 * **The list is who a rider can be sent to, and says why.** Their superior, those directly
 * beneath them, and anyone on their side they can see. Everyone else is reached through
 * somebody on that list, and the form says so rather than letting them find out on a
 * refusal. The referee is on it too, for a commander — but as a note out of the game,
 * not a despatch, and the form says that as well.
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

import { useEffect, useMemo, useRef, useState } from 'react';

import { dayHour } from '../board.js';
import { copy } from '../copy.js';
import { useNarrow } from '../layout.js';

import { REFEREE_ADDRESS, type Correspondent, type Estimate } from '../despatch.js';

export interface Draft {
  /** A commander's id, or `REFEREE_ADDRESS` for a note to the referee. */
  readonly to: string;
  readonly text: string;
  /** Who it is from. Absent for a commander, who can only be themselves. */
  readonly from?: string;
}

export function Composer({
  correspondents,
  factionName,
  toReferee = false,
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
  /** Offer the referee as an addressee: a commander's form, not a referee's own. */
  toReferee?: boolean;
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
   * Only a referee has more than one. They run most of the commanders on the map and take
   * dictation from the players who hold the rest, so writing as a commander is their ordinary
   * work rather than an impersonation — and the log records that they did it.
   */
  senders?: readonly { id: string; name: string; unitName: string; faction: string }[];
  from?: string;
  onFrom?: (commanderId: string) => void;
}) {
  const first = correspondents[0]?.id ?? (toReferee ? REFEREE_ADDRESS : '');
  const [to, setTo] = useState(initial?.to ?? first);
  const [text, setText] = useState(initial?.text ?? '');
  const toTheReferee = to === REFEREE_ADDRESS;
  // A dropdown has room for one line, and on a phone that line is cut off long before the
  // formation that tells two generals apart. So a phone gets every addressee as a row.
  const narrow = useNarrow();

  // Opened from a button further down the post, the form lands above where the reader is
  // looking — off the top of a phone entirely, because the browser holds the button it was
  // opened from still and pushes everything inserted above it out of view. So the form
  // brings itself into view once, a frame after opening, when the browser has done that.
  const form = useRef<HTMLElement>(null);
  useEffect(() => {
    const frame = requestAnimationFrame(() => form.current?.scrollIntoView?.({ block: 'start' }));
    return () => cancelAnimationFrame(frame);
  }, []);

  // The addressee list changes with the sender, and with the ground: somebody in sight a
  // moment ago may have marched out of it. Keeping a stale addressee would offer a despatch
  // the server is about to refuse.
  useEffect(() => {
    const still = correspondents.some((c) => c.id === to) || (toReferee && toTheReferee);
    if (!still) setTo(first);
  }, [correspondents, to, toReferee, toTheReferee, first]);

  const estimate = useMemo(
    () => (to === '' || toTheReferee || estimateFor === undefined ? null : estimateFor(to)),
    [to, toTheReferee, estimateFor],
  );

  return (
    <section ref={form} className="panel-section composer">
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

      {narrow ? (
        <fieldset className="choices addressees" disabled={busy}>
          <legend>{copy.composer.to}</legend>
          {correspondents.map((c) => (
            <label key={c.id} className="choice">
              <input
                type="radio"
                name="to"
                checked={to === c.id}
                onChange={() => setTo(c.id)}
              />
              <span className="choice-label">
                <strong>{c.name}</strong>
                <span className="muted small">
                  {copy.composer.correspondentLine(c.unitName, factionName(c.faction))}
                </span>
              </span>
              {copy.composer.relationTag[c.relation] !== '' && (
                <span className="choice-tag">{copy.composer.relationTag[c.relation]}</span>
              )}
            </label>
          ))}
          {toReferee && (
            <label className="choice">
              <input
                type="radio"
                name="to"
                checked={toTheReferee}
                onChange={() => setTo(REFEREE_ADDRESS)}
              />
              <span className="choice-label muted">{copy.composer.theReferee}</span>
            </label>
          )}
        </fieldset>
      ) : (
      <label className="field">
        <div className="row-label">{copy.composer.to}</div>
        <select value={to} onChange={(e) => setTo(e.target.value)} disabled={busy}>
          {correspondents.map((c) => (
            <option key={c.id} value={c.id}>
              {copy.composer.correspondent(c.name, c.unitName, factionName(c.faction))}
              {copy.composer.relation[c.relation]}
            </option>
          ))}
          {toReferee && <option value={REFEREE_ADDRESS}>{copy.composer.theReferee}</option>}
        </select>
      </label>
      )}

      <textarea
        className="prose"
        rows={6}
        value={text}
        placeholder={
          toTheReferee ? copy.composer.notePlaceholder : copy.composer.despatchPlaceholder
        }
        onChange={(e) => setText(e.target.value)}
        disabled={busy}
      />

      <p className="muted">
        {senders !== undefined
          ? copy.composer.asReferee
          : toTheReferee
            ? copy.composer.isNote
            : copy.composer.whoMayBeWritten}
      </p>

      {toTheReferee ? null : estimateFor === undefined ? (
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
          onClick={() => onSend({ to, text, ...(from === undefined ? {} : { from }) })}
        >
          {busy
            ? copy.composer.sending
            : toTheReferee
              ? copy.composer.sendNote
              : copy.composer.send}
        </button>
        <button onClick={onCancel} disabled={busy}>
          {copy.composer.cancel}
        </button>
      </div>
    </section>
  );
}
