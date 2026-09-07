/**
 * The post: what has reached a commander, and what he has sent.
 *
 * This is where the design either reads or does not, and the whole of it is in which hour
 * is shown first.
 *
 * **A despatch is dated by the hour it describes.** A report that took six hours to
 * arrive is telling him about hour four, not about hour ten. Leading with the arrival
 * time — the natural thing for an inbox to do — would be a lie about what he knows, and
 * would make a stale report look fresh at exactly the moment it mattered. So the hour
 * written is the headline, the hour it landed is the footnote, and the gap between them
 * is stated rather than left to be worked out.
 *
 * **The outbox is deliberately unhelpful.** It says what he wrote and when, and nothing
 * about whether it arrived, because that is the mechanic rather than a missing feature.
 * The one thing that can change is an acknowledgement coming back, which is the only
 * feedback channel in the game and is therefore the loudest thing on a sent despatch.
 *
 * **Superseded orders are marked, not hidden.** An order that turned up after a later one
 * is disregarded by correct staff practice, and watching that happen is half of
 * understanding why a corps did what it did.
 */

import type { ReceivedDespatch, SentDespatch } from '@campaign/shared';

import { ageLabel } from '../board.js';

const hour = (h: number): string => `hour ${h % 1 === 0 ? h : h.toFixed(1)}`;

export function Post({
  received,
  sent,
  clockHours,
  nameOf,
  acknowledged,
  onAcknowledge,
  onForward,
  onWrite,
  busyId,
}: {
  received: readonly ReceivedDespatch[];
  sent: readonly SentDespatch[];
  clockHours: number;
  nameOf: (commanderId: string) => string;
  acknowledged: (despatchId: string) => boolean;
  onAcknowledge: (d: ReceivedDespatch) => void;
  onForward: (d: ReceivedDespatch) => void;
  onWrite: () => void;
  busyId: string | null;
}) {
  return (
    <>
      <section className="panel-section">
        <h3>In my hand</h3>
        <button className="primary wide" onClick={onWrite}>
          Write a despatch
        </button>

        {received.length === 0 ? (
          <p className="muted">
            Nothing has reached you. Anything on the road toward you is invisible until a
            rider puts it in your hand.
          </p>
        ) : (
          <ul className="post">
            {received.map((d) => (
              <li
                key={d.id}
                className={`despatch ${d.superseded ? 'superseded' : ''} kind-${d.kind}`}
              >
                <div className="despatch-head">
                  <span className="despatch-from">{nameOf(d.from)}</span>
                  <span className="despatch-kind">{d.kind}</span>
                </div>

                {/* The hour it describes, first and large. Everything else is smaller. */}
                <div className="despatch-when">
                  Written {hour(d.sentAtHours)} · {ageLabel(d.sentAtHours, clockHours)}
                </div>
                <div className="muted small">
                  Reached you {hour(d.receivedAtHours)}, after{' '}
                  {(d.receivedAtHours - d.sentAtHours).toFixed(1)} h on the road
                  {d.forwardedFrom === null ? '' : ` · forwarded from ${nameOf(d.forwardedFrom)}`}
                </div>

                {d.superseded && (
                  <div className="flag">
                    Overtaken by a later order you already hold. Disregarded.
                  </div>
                )}

                {d.body.text !== undefined && <p className="prose-read">{d.body.text}</p>}

                {d.body.contacts !== undefined && d.body.contacts.length > 0 && (
                  <div className="muted small">
                    {d.body.contacts.length === 1
                      ? '1 sighting attached'
                      : `${d.body.contacts.length} sightings attached`}
                    {d.body.contacts.map((c) => (
                      <span key={`${c.coord.q},${c.coord.r}`} className="tag">
                        {c.coord.q}, {c.coord.r}
                      </span>
                    ))}
                  </div>
                )}

                <div className="despatch-actions">
                  {acknowledged(d.id) ? (
                    <span className="muted small">Acknowledged</span>
                  ) : (
                    <button disabled={busyId === d.id} onClick={() => onAcknowledge(d)}>
                      Acknowledge
                    </button>
                  )}
                  <button disabled={busyId === d.id} onClick={() => onForward(d)}>
                    Forward
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="panel-section">
        <h3>Sent</h3>
        {sent.length === 0 ? (
          <p className="muted">You have written nothing yet.</p>
        ) : (
          <ul className="post">
            {sent.map((d) => (
              <li key={d.id} className={`despatch sent kind-${d.kind}`}>
                <div className="despatch-head">
                  <span className="despatch-from">To {nameOf(d.to)}</span>
                  <span className="despatch-kind">{d.kind}</span>
                </div>
                <div className="despatch-when">
                  Written {hour(d.sentAtHours)} · {ageLabel(d.sentAtHours, clockHours)}
                </div>

                {d.body.text !== undefined && <p className="prose-read">{d.body.text}</p>}

                <div className="muted small">
                  {d.handed
                    ? 'Handed over on the spot — their column was touching yours.'
                    : d.acknowledged
                      ? 'Acknowledged. It arrived.'
                      : 'No acknowledgement. You have no way of knowing whether it arrived.'}
                  {d.via.length > 0 && ` · Rider sent via ${d.via.length} waypoint(s) of yours.`}
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>
    </>
  );
}
