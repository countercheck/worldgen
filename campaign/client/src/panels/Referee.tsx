/**
 * The referee's console.
 *
 * Under prose orders the referee's cost is reading everybody's post and turning it into
 * marches, so the one thing this screen has to do is make that **one motion rather than
 * two**. A despatch arrives, he reads it, and the control that sets the addressee's task
 * is on the same card — not on another tab, not after finding the unit in a list. If that
 * motion is two actions the game is tiring to run, and a game that is tiring to run does
 * not get run.
 *
 * So the queue is the primary object here, not the despatch log. A decision names the
 * commander it belongs to and the formation that ran into it, and carries the button that
 * resolves it: point at the ground, or say it needed nothing.
 *
 * The log below is a reference rather than a workspace. It is the only place in the system
 * where routes and fates are visible — a rider's path betrays where his addressee actually
 * is, which is why no commander ever sees one — so it is also the referee's window on the
 * thing his players cannot see and are most curious about.
 */

import type { Despatch, PendingDecision, Task, Unit } from '@campaign/shared';

import { ageLabel } from '../board.js';

const hour = (h: number): string => (h % 1 === 0 ? String(h) : h.toFixed(1));

/** What each trigger means, in the referee's language rather than the engine's. */
const TRIGGERS: Readonly<Record<string, string>> = {
  enemy_contact: 'has come into contact',
  crossing_impassable: 'cannot get across',
  gunfire_heard: 'hears guns',
  objective_reached: 'has arrived',
  despatch_arrived: 'has received a despatch',
  out_of_provisions: 'is out of provisions',
  attacked: 'is under attack',
};

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
  onResolve: (decision: PendingDecision) => void;
  busyId: string | null;
}) {
  const open = decisions.filter((d) => d.resolvedAtHours === null);

  return (
    <section className="panel-section">
      <h3>Wants a decision</h3>

      {open.length === 0 ? (
        <p className="muted">
          Nothing is waiting on you. Run the clock until something is.
        </p>
      ) : (
        <ul className="post">
          {open.map((d) => {
            const unit = unitOf(d.unitId);
            const task = taskOf(d.unitId);
            const ordering = orderingUnitId === d.unitId;

            return (
              <li key={d.id} className={`despatch decision trigger-${d.trigger}`}>
                <div className="despatch-head">
                  <span className="despatch-from">{nameOf(d.commanderId)}</span>
                  <span className="despatch-kind">{d.trigger.replace(/_/g, ' ')}</span>
                </div>

                <div className="despatch-when">
                  {unit?.name ?? d.unitId} {TRIGGERS[d.trigger] ?? 'needs a decision'}
                </div>
                <div className="muted small">
                  Hour {hour(d.atHours)} · {ageLabel(d.atHours, clockHours)}
                  {task === undefined
                    ? ' · no standing task'
                    : task.complete
                      ? ' · halted'
                      : ` · marching on ${task.destination.q}, ${task.destination.r}`}
                </div>

                <Context decision={d} />

                <div className="despatch-actions">
                  <button
                    className={ordering ? 'primary' : ''}
                    disabled={unit === undefined}
                    onClick={() => onOrder(d.unitId)}
                  >
                    {ordering ? 'Pointing…' : 'March them somewhere'}
                  </button>
                  <button disabled={busyId === d.id} onClick={() => onResolve(d)}>
                    Dealt with
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
  };

  if (c.contacts !== undefined && c.contacts.length > 0) {
    return (
      <p className="muted small">
        {c.contacts.length === 1 ? 'A column' : `${c.contacts.length} columns`} seen at{' '}
        {c.contacts.map((x) => `${x.coord.q}, ${x.coord.r}`).join(' · ')}
      </p>
    );
  }

  if (decision.trigger === 'despatch_arrived') {
    return (
      <p className="muted small">
        From {c.from ?? 'somebody'}, written hour {hour(c.sentAtHours ?? decision.atHours)}.
        Read it in his seat, then tell his formation where to go.
      </p>
    );
  }

  if (c.at !== undefined) {
    return (
      <p className="muted small">
        Stopped at {c.at.q}, {c.at.r}, short of {c.destination?.q}, {c.destination?.r}.
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
  nameOf,
}: {
  despatches: readonly Despatch[];
  clockHours: number;
  nameOf: (commanderId: string) => string;
}) {
  const ordered = [...despatches].sort((a, b) => {
    const flying = (d: Despatch): number => (d.fate.kind === 'in_transit' ? 0 : 1);
    return flying(a) - flying(b) || b.sentAtHours - a.sentAtHours;
  });

  return (
    <section className="panel-section">
      <h3>The post</h3>
      {ordered.length === 0 ? (
        <p className="muted">Nobody has written to anybody.</p>
      ) : (
        <ul className="post">
          {ordered.map((d) => (
            <li key={d.id} className={`despatch log fate-${d.fate.kind} kind-${d.kind}`}>
              <div className="despatch-head">
                <span className="despatch-from">
                  {nameOf(d.from)} → {nameOf(d.to)}
                </span>
                <span className="despatch-kind">{d.kind}</span>
              </div>
              <div className="despatch-when">
                Hour {hour(d.sentAtHours)} · {ageLabel(d.sentAtHours, clockHours)}
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
          Riding — {done} of {d.route.length - 1} hexes.{' '}
          {d.handed ? 'Handed over.' : 'The sender has no idea.'}
        </>
      );
    }
    case 'delivered':
      return (
        <>
          Delivered hour {hour(fate.atHours)}, after{' '}
          {(fate.atHours - d.sentAtHours).toFixed(1)} h.
          {d.handed ? ' Handed over on the spot.' : ''}
        </>
      );
    case 'lost':
      return (
        <>
          Rider stopped by {fate.by} at hour {hour(fate.atHours)} — dice [{fate.dice.join(', ')}].
          The paper went with him.
        </>
      );
    case 'captured':
      return (
        <>
          <strong className="bad">Captured</strong> by {fate.by} at hour {hour(fate.atHours)} —
          dice [{fate.dice.join(', ')}]. They have read it; the sender has not been told.
        </>
      );
  }
}
