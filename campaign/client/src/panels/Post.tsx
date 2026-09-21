/**
 * The post: what has reached a commander, and what they have sent.
 *
 * This is where the design either reads or does not, and the whole of it is in which hour
 * is shown first.
 *
 * **A despatch is dated by the hour it describes.** A report that took six hours to
 * arrive is telling them about hour four, not about hour ten. Leading with the arrival
 * time — the natural thing for an inbox to do — would be a lie about what they know, and
 * would make a stale report look fresh at exactly the moment it mattered. So the hour
 * written is the headline, the hour it landed is the footnote, and the gap between them
 * is stated rather than left to be worked out.
 *
 * **The outbox is deliberately unhelpful.** It says what they wrote and when, and nothing
 * about whether it arrived, because that is the mechanic rather than a missing feature.
 * The one thing that can change is an acknowledgement coming back, which is the only
 * feedback channel in the game and is therefore the loudest thing on a sent despatch.
 *
 * **Superseded orders are marked, not hidden.** An order that turned up after a later one
 * is disregarded by correct staff practice, and watching that happen is half of
 * understanding why a corps did what it did.
 */

import type { ReceivedDespatch, SentDespatch } from '@campaign/shared';

import { ageLabel, dayHour } from '../board.js';
import { copy } from '../copy.js';



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
        <h3>{copy.post.inboxHeading}</h3>
        <button className="primary wide" onClick={onWrite}>
          {copy.post.write}
        </button>

        {received.length === 0 ? (
          <p className="muted">{copy.post.emptyInbox}</p>
        ) : (
          <ul className="post">
            {received.map((d) => (
              <li
                key={d.id}
                className={`despatch ${d.superseded ? 'superseded' : ''} kind-${d.kind}`}
              >
                <div className="despatch-head">
                  <span className="despatch-from">{nameOf(d.from)}</span>
                </div>

                {/* The hour it describes, first and large. Everything else is smaller. */}
                <div className="despatch-when">
                  {copy.post.written(
                    dayHour(d.sentAtHours),
                    ageLabel(d.sentAtHours, clockHours),
                  )}
                </div>
                <div className="muted small">
                  {copy.post.reached(
                    dayHour(d.receivedAtHours),
                    (d.receivedAtHours - d.sentAtHours).toFixed(1),
                  )}
                  {d.forwardedFrom === null ? '' : copy.post.forwardedFrom(nameOf(d.forwardedFrom))}
                </div>

                {d.superseded && <div className="flag">{copy.post.superseded}</div>}

                {d.body.text !== undefined && <p className="prose-read">{d.body.text}</p>}

                {d.body.contacts !== undefined && d.body.contacts.length > 0 && (
                  <div className="muted small">
                    {copy.post.sightingsAttached(d.body.contacts.length)}
                    {d.body.contacts.map((c) => (
                      <span key={`${c.coord.q},${c.coord.r}`} className="tag">
                        {c.coord.q}, {c.coord.r}
                      </span>
                    ))}
                  </div>
                )}

                <div className="despatch-actions">
                  {acknowledged(d.id) ? (
                    <span className="muted small">{copy.post.acknowledged}</span>
                  ) : (
                    <button disabled={busyId === d.id} onClick={() => onAcknowledge(d)}>
                      {copy.post.acknowledge}
                    </button>
                  )}
                  <button disabled={busyId === d.id} onClick={() => onForward(d)}>
                    {copy.post.forward}
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="panel-section">
        <h3>{copy.post.sentHeading}</h3>
        {sent.length === 0 ? (
          <p className="muted">{copy.post.emptyOutbox}</p>
        ) : (
          <ul className="post">
            {sent.map((d) => (
              <li key={d.id} className={`despatch sent kind-${d.kind}`}>
                <div className="despatch-head">
                  <span className="despatch-from">{copy.post.to(nameOf(d.to))}</span>
                </div>
                <div className="despatch-when">
                  {copy.post.written(
                    dayHour(d.sentAtHours),
                    ageLabel(d.sentAtHours, clockHours),
                  )}
                </div>

                {d.body.text !== undefined && <p className="prose-read">{d.body.text}</p>}

                <div className="muted small">
                  {d.handed
                    ? copy.post.handed
                    : d.acknowledged
                      ? copy.post.arrived
                      : copy.post.unknownFate}
                  {d.via.length > 0 && copy.post.viaWaypoints(d.via.length)}
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>
    </>
  );
}
