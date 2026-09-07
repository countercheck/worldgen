/**
 * What a commander has, and how long ago he heard about it.
 *
 * The piece that makes the whole model legible in one glance: one formation he is
 * standing next to, and beneath it a list of his own corps, each with an hour on it and
 * most of those hours old. Nothing here is a surprise once the design is understood, and
 * everything here is a surprise the first time somebody expects a wargame.
 *
 * The age is the point, so it is the thing on the right of every row — the column a
 * reader's eye runs down. A formation reported four hours ago could be anywhere within
 * twelve kilometres of where it says it is, and the panel says so rather than leaving a
 * confident-looking hex to be believed.
 */

import type { Unit, UnitReport } from '@campaign/shared';

import { ageHours, ageLabel } from '../board.js';

export function Command({
  own,
  reports,
  clockHours,
  colorOf,
  onSelect,
  taskLine,
}: {
  own: Unit | null;
  reports: readonly UnitReport[];
  clockHours: number;
  colorOf: (faction: string) => string;
  onSelect: (unitId: string) => void;
  /** What the formation he rides with is doing, if anything. He set out on it; he knows. */
  taskLine: string | null;
}) {
  return (
    <section className="panel-section">
      <h3>Under my command</h3>

      {own !== null && (
        <button className="formation with-me" onClick={() => onSelect(own.id)}>
          <span className="swatch small" style={{ background: colorOf(own.faction) }} />
          <span className="formation-name">{own.name}</span>
          <span className="formation-age now">with me</span>
        </button>
      )}

      {taskLine !== null && <p className="muted small">{taskLine}</p>}

      {reports.length === 0 ? (
        <p className="muted">Nobody else answers to you.</p>
      ) : (
        <>
          <ul className="unit-list">
            {reports.map((r) => (
              <li key={r.unitId}>
                <button className="formation" onClick={() => onSelect(r.unitId)}>
                  <span
                    className="swatch small ghost"
                    style={{ background: colorOf(r.faction) }}
                  />
                  <span className="formation-name">{r.name}</span>
                  <span className="formation-age">{ageLabel(r.atHours, clockHours)}</span>
                </button>
              </li>
            ))}
          </ul>
          <p className="muted small">
            Every hour above is when you last heard, not where they are. At infantry pace
            the oldest of them could be {Math.round(worstDrift(reports, clockHours))} km from
            the hex you are looking at.
          </p>
        </>
      )}
    </section>
  );
}

/**
 * How far the stalest report could be wrong by, in kilometres.
 *
 * Infantry march at three km/h on a road and one hex is one kilometre, so the number is
 * simply three times the age of the oldest report. Deliberately generous rather than
 * precise: it is there to stop a hex being believed, not to be planned against.
 */
function worstDrift(reports: readonly UnitReport[], clockHours: number): number {
  const oldest = Math.max(...reports.map((r) => ageHours(r.atHours, clockHours)));
  return oldest * 3;
}
