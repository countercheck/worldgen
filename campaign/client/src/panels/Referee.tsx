/**
 * The referee's console.
 *
 * Under prose orders the referee's cost is reading everybody's post and turning it into
 * marches, so the one thing this screen has to do is make that **one motion rather than
 * two**. A despatch arrives, they read it, and the control that sets the addressee's task
 * is on the same card — not on another tab, not after finding the unit in a list. If that
 * motion is two actions the game is tiring to run, and a game that is tiring to run does
 * not get run.
 *
 * So the queue is the primary object here, not the despatch log. A decision names the
 * commander it belongs to and the formation that ran into it, and carries the button that
 * resolves it: point at the ground, or say it needed nothing.
 *
 * The log below is a reference rather than a workspace. It is the only place in the system
 * where routes and fates are visible — a rider's path betrays where their addressee actually
 * is, which is why no commander ever sees one — so it is also the referee's window on the
 * thing their players cannot see and are most curious about.
 */

import { contestants } from '@campaign/shared';

import type { Despatch, PendingDecision, Task, Unit } from '@campaign/shared';

import { ageLabel, dayHour } from '../board.js';
import { copy, triggerLabel } from '../copy.js';
import type { CommanderLabel } from '../despatch.js';

export function DecisionQueue({
  decisions,
  clockHours,
  nameOf,
  unitOf,
  taskOf,
  orderingUnitId,
  onOrder,
  onResolve,
  busyId,
}: {
  decisions: readonly PendingDecision[];
  clockHours: number;
  nameOf: (commanderId: string) => string;
  unitOf: (unitId: string) => Unit | undefined;
  taskOf: (unitId: string) => Task | undefined;
  /** The formation whose destination is currently being pointed at, if any. */
  orderingUnitId: string | null;
  onOrder: (unitId: string) => void;
  onResolve: (decision: PendingDecision, favouring?: string) => void;
  busyId: string | null;
}) {
  const open = decisions.filter((d) => d.resolvedAtHours === null);

  return (
    <section className="panel-section">
      <h3>{copy.referee.queueHeading}</h3>

      {open.length === 0 ? (
        <p className="muted">{copy.referee.queueEmpty}</p>
      ) : (
        <ul className="post">
          {open.map((d) => {
            const unit = unitOf(d.unitId);
            const task = taskOf(d.unitId);
            const ordering = orderingUnitId === d.unitId;

            return (
              <li key={d.id} className={`despatch decision trigger-${d.trigger}`}>
                <div className="despatch-head">
                  {/* Traffic carries no commander: two columns meeting on a road is a
                      fact about the ground rather than about what anyone believes. */}
                  <span className="despatch-from">
                    {d.commanderId === null ? copy.referee.theGround : nameOf(d.commanderId)}
                  </span>
                  <span className="despatch-kind">{d.trigger.replace(/_/g, ' ')}</span>
                </div>

                <div className="despatch-when">
                  {unit?.name ?? d.unitId}{' '}
                  {triggerLabel(d.trigger)}
                </div>
                <div className="muted small">
                  {dayHour(d.atHours)} · {ageLabel(d.atHours, clockHours)}
                  {task === undefined
                    ? copy.referee.noStandingTask
                    : task.complete
                      ? copy.referee.taskHalted
                      : copy.referee.taskMarching(task.destination.q, task.destination.r)}
                </div>

                <Context decision={d} />

                {/* A tie for a hex is the one decision with a ruling to make rather than
                    just an acknowledgement, so it gets the two names to choose between.
                    "Dealt with" stays available and deliberately settles nothing: the
                    columns meet again and ask again, which is what not deciding means. */}
                {d.trigger === 'column_contested' && (
                  <div className="despatch-actions">
                    {[d.unitId, ...contestants(d)].map((id) => (
                      <button
                        key={id}
                        className="primary"
                        disabled={busyId === d.id}
                        onClick={() => onResolve(d, id)}
                      >
                        {copy.referee.giveItTo(unitOf(id)?.name ?? id)}
                      </button>
                    ))}
                  </div>
                )}

                <div className="despatch-actions">
                  <button
                    className={ordering ? 'primary' : ''}
                    disabled={unit === undefined}
                    onClick={() => onOrder(d.unitId)}
                  >
                    {ordering ? copy.referee.pointing : copy.referee.march}
                  </button>
                  <button disabled={busyId === d.id} onClick={() => onResolve(d)}>
                    {copy.referee.dealtWith}
                  </button>
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}

/**
 * The caption on a decision.
 *
 * `context` is a plain record rather than a union — every trigger wants to say something
 * different and typing that would be agreeing a schema for what is essentially a caption
 * — so this reads the few keys it knows about and says nothing about the rest.
 */
function Context({ decision }: { decision: PendingDecision }) {
  const c = decision.context as {
    contacts?: readonly { coord: { q: number; r: number }; intelLevel: number }[];
    destination?: { q: number; r: number };
    at?: { q: number; r: number };
    from?: string;
    sentAtHours?: number;
    withUnitId?: string;
    patrolUnitId?: string;
    hostile?: boolean;
    dice?: number;
  };

  if (c.contacts !== undefined && c.contacts.length > 0) {
    return (
      <p className="muted small">
        {copy.referee.columnsSeen(
          c.contacts.length,
          c.contacts.map((x) => `${x.coord.q}, ${x.coord.r}`).join(' · '),
        )}
      </p>
    );
  }

  // A patrol meeting something is the one decision that comes with a die roll attached,
  // so the pool the rules call for is worked out here rather than left to be counted off
  // the modifiers by hand at the moment it matters.
  if (decision.trigger === 'patrol_contact') {
    return (
      <p className="muted small">
        {copy.referee.patrolMet(
          c.hostile === true ? copy.referee.patrolHostile : copy.referee.patrolFriendly,
          c.withUnitId ?? copy.referee.patrolSomething,
          c.at?.q,
          c.at?.r,
        )}
        {c.dice === undefined ? '' : copy.referee.patrolRoll(c.dice)}
      </p>
    );
  }

  if (decision.trigger === 'despatch_arrived') {
    return (
      <p className="muted small">
        {copy.referee.despatchArrived(
          c.from ?? copy.referee.somebody,
          dayHour(c.sentAtHours ?? decision.atHours),
        )}
      </p>
    );
  }

  if (c.at !== undefined) {
    return (
      <p className="muted small">
        {copy.referee.stoppedShort(c.at.q, c.at.r, c.destination?.q, c.destination?.r)}
      </p>
    );
  }

  return null;
}

/**
 * Every despatch in the campaign, and the only place a route or a fate is ever shown.
 *
 * In flight first: those are the ones whose outcome is not settled and the only ones the
 * referee can still do anything about — reroute a rider, or simply know that an order is
 * about to pass a picket and watch what happens.
 */
export function DespatchLog({
  despatches,
  clockHours,
  labelOf,
}: {
  despatches: readonly Despatch[];
  clockHours: number;
  /**
   * How to name an officer, formation and side.
   *
   * The referee's log spans both armies, so the side is doing real work here rather than
   * repeating what the reader already knows: two lines of traffic crossing in the same list
   * are only legible if each says whose it is.
   */
  labelOf: (commanderId: string) => CommanderLabel;
}) {
  const ordered = [...despatches].sort((a, b) => {
    const flying = (d: Despatch): number => (d.fate.kind === 'in_transit' ? 0 : 1);
    return flying(a) - flying(b) || b.sentAtHours - a.sentAtHours;
  });

  return (
    <section className="panel-section">
      <h3>{copy.referee.logHeading}</h3>
      {ordered.length === 0 ? (
        <p className="muted">{copy.referee.logEmpty}</p>
      ) : (
        <ul className="post">
          {ordered.map((d) => (
            <li key={d.id} className={`despatch log fate-${d.fate.kind} kind-${d.kind}`}>
              <div className="despatch-head">
                <span className="despatch-from">
                  {labelOf(d.from).name} → {labelOf(d.to).name}
                </span>
                <span className="despatch-commands">
                  {copy.post.commands(labelOf(d.from).unit, labelOf(d.from).faction)}
                  {copy.referee.towards}
                  {copy.post.commands(labelOf(d.to).unit, labelOf(d.to).faction)}
                </span>
              </div>
              <div className="despatch-when">
                {dayHour(d.sentAtHours)} · {ageLabel(d.sentAtHours, clockHours)}
              </div>
              <div className="muted small">
                <Fate d={d} />
              </div>
              {d.body.text !== undefined && <p className="prose-read">{d.body.text}</p>}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function Fate({ d }: { d: Despatch }) {
  const fate = d.fate;
  switch (fate.kind) {
    case 'in_transit': {
      const done = Math.floor(d.progress);
      return (
        <>
          {copy.referee.riding(done, d.route.length - 1)}{' '}
          {d.handed ? copy.referee.ridingHanded : copy.referee.ridingBlind}
        </>
      );
    }
    case 'delivered':
      return (
        <>
          {copy.referee.delivered(
            dayHour(fate.atHours),
            (fate.atHours - d.sentAtHours).toFixed(1),
          )}
          {d.handed ? copy.referee.deliveredHanded : ''}
        </>
      );
    case 'lost':
      return (
        <>
          {copy.referee.lost(fate.by, dayHour(fate.atHours), fate.dice.join(', '))}
        </>
      );
    case 'captured':
      return (
        <>
          <strong className="bad">{copy.referee.capturedLabel}</strong>
          {copy.referee.captured(fate.by, dayHour(fate.atHours), fate.dice.join(', '))}
        </>
      );
  }
}
