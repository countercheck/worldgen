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
 * They learn it arrived when somebody writes back, which is another despatch in the inbox
 * above, on another rider who can also be stopped.
 */

import type { ReceivedDespatch, SentDespatch } from '@campaign/shared';

import { ageLabel, dayHour } from '../board.js';
import { copy } from '../copy.js';
import type { CommanderLabel } from '../despatch.js';



export function Post({
  received,
  sent,
  clockHours,
  labelOf,
  onForward,
  onWrite,
  busyId,
}: {
  received: readonly ReceivedDespatch[];
  sent: readonly SentDespatch[];
  clockHours: number;
  /**
   * How to name an officer: who they are, what they command, which side.
   *
   * Both halves are shown, on two lines rather than one. The officer is who the despatch is
   * from — they put their name to it — and the formation is which body of troops it is
   * therefore about, which is what a reader actually plans against.
   */
  labelOf: (commanderId: string) => CommanderLabel;
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
              <li key={d.id} className="despatch">
                <div className="despatch-head">
                  <span className="despatch-from">{labelOf(d.from).name}</span>
                  <span className="despatch-commands">
                    {copy.post.commands(labelOf(d.from).unit, labelOf(d.from).faction)}
                  </span>
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
                  {d.forwardedFrom === null ? '' : copy.post.forwardedFrom(labelOf(d.forwardedFrom).name)}
                </div>

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
              <li key={d.id} className="despatch sent">
                <div className="despatch-head">
                  <span className="despatch-from">{copy.post.to(labelOf(d.to).name)}</span>
                  <span className="despatch-commands">
                    {copy.post.commands(labelOf(d.to).unit, labelOf(d.to).faction)}
                  </span>
                </div>
                <div className="despatch-when">
                  {copy.post.written(
                    dayHour(d.sentAtHours),
                    ageLabel(d.sentAtHours, clockHours),
                  )}
                </div>

                {d.body.text !== undefined && <p className="prose-read">{d.body.text}</p>}

                <div className="muted small">
                  {d.handed ? copy.post.handed : copy.post.unknownFate}
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
