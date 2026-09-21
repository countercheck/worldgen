/**
 * The order of battle, in a drawer.
 *
 * The sidebar answers "what is this one formation", and answers it well. This answers the
 * other question a referee asks constantly and could not ask at all — "what is everything
 * doing" — without giving up the map to a table.
 *
 * ## Two very different lists
 *
 * A referee gets every formation as it is. A commander gets their own as it is, and every
 * other one *as they last heard*: a `UnitReport` carries the hour it describes, not the hour
 * it arrived, and half the design is that those are different. So a commander's rows are
 * dated and a referee's are not, and the two are built from different data rather than one
 * filtered — a filtered list is how the dated ones quietly become live ones.
 *
 * A report carries less than a unit does, and deliberately: a rider knows where a
 * formation was, roughly how strong and how tired, and does not know its ammunition. The
 * columns here are the intersection, so a commander is never shown a blank where a referee
 * sees a number and left to wonder which it is.
 */

import {
  isPatrol,
  maxMorale,
  presentUnderArms,
  type CampaignConfig,
  type Unit,
  type UnitReport,
} from '@campaign/shared';

import type { ReactNode } from 'react';

import { ageLabel, dayHour } from '../board.js';
import { copy } from '../copy.js';
import { rosterGroups, type RosterLine } from '../roster.js';

export function Roster({
  open,
  onClose,
  role,
  units,
  reports,
  ownUnitId,
  clockHours,
  cfg,
  factionName,
  colorOf,
  selectedId,
  onSelect,
  taskOf,
  editor,
}: {
  open: boolean;
  onClose: () => void;
  role: 'referee' | 'commander';
  /** Live formations: all of them for a referee, their own for a commander. */
  units: readonly Unit[];
  /** Dated reports. Empty for a referee, who has no need of them. */
  reports: readonly UnitReport[];
  ownUnitId: string | null;
  clockHours: number;
  cfg: CampaignConfig;
  factionName: (id: string) => string;
  colorOf: (faction: string) => string;
  selectedId: string | null;
  onSelect: (unitId: string) => void;
  taskOf: (unitId: string) => string | null;
  /** The referee's order-of-battle controls. Absent for a commander, who raises nothing. */
  editor?: ReactNode;
}) {
  if (!open) return null;

  const groups = rosterGroups({ role, units, reports, factionName });

  return (
    <aside className="roster" aria-label={copy.roster.label}>
      <header className="roster-head">
        <h2>{copy.roster.heading}</h2>
        <button className="dismiss" onClick={onClose}>
          {copy.roster.close}
        </button>
      </header>

      <div className="roster-body">
        {editor}

        {groups.map((group) => (
          <section key={group.title} className="roster-group">
            <h3>
              {group.title}
              <span className="muted"> · {group.lines.length}</span>
            </h3>
            {group.note !== null && <p className="muted small">{group.note}</p>}

            {group.lines.length === 0 ? (
              <p className="muted small">{copy.roster.empty}</p>
            ) : (
              <table className="roster-table">
                <thead>
                  <tr>
                    <th>{copy.roster.columnFormation}</th>
                    <th>{copy.roster.columnWhere}</th>
                    <th>{copy.roster.columnStrength}</th>
                    <th>{copy.roster.columnFatigue}</th>
                    <th>{copy.roster.columnDoing}</th>
                  </tr>
                </thead>
                <tbody>
                  {group.lines.map((line: RosterLine) => {
                    const task = taskOf(line.unitId);
                    const patrol = line.unit !== null && isPatrol(line.unit);
                    return (
                      <tr
                        key={line.unitId}
                        className={
                          (line.unitId === selectedId ? 'selected ' : '') +
                          (line.asOfHours === null ? '' : 'remembered')
                        }
                        onClick={() => onSelect(line.unitId)}
                      >
                        <td>
                          <span
                            className={`swatch small${line.asOfHours === null ? '' : ' ghost'}`}
                            style={{ background: colorOf(line.faction) }}
                          />
                          {line.name}
                          {line.corps !== null && <span className="muted"> · {line.corps}</span>}
                          {patrol && <span className="muted">{copy.roster.patrol}</span>}
                        </td>
                        <td className="num">
                          {line.at.q}, {line.at.r}
                          {line.asOfHours !== null && (
                            <div className="muted small" title={dayHour(line.asOfHours)}>
                              {ageLabel(line.asOfHours, clockHours)}
                            </div>
                          )}
                        </td>
                        <td className="num">
                          {/* A patrol has no strength in the sense this column means, and
                              a zero here would read as a formation destroyed. */}
                          {patrol ? (
                            <span className="muted">{copy.roster.detachment}</span>
                          ) : (
                            <>
                              {line.paperStrength.toLocaleString()}
                              {line.unit !== null && (
                                <div className="muted small">
                                  {copy.roster.underArms(
                                    presentUnderArms(line.unit).toLocaleString(),
                                  )}
                                </div>
                              )}
                            </>
                          )}
                        </td>
                        <td className="num">
                          {patrol ? (
                            <span className="muted">{copy.roster.notTracked}</span>
                          ) : (
                            <>
                              {line.fatigue}
                              {line.unit !== null && (
                                <div className="muted small">
                                  {copy.roster.morale(
                                    line.unit.morale,
                                    maxMorale(line.unit, cfg.maxMorale),
                                  )}
                                </div>
                              )}
                            </>
                          )}
                        </td>
                        <td>
                          {line.formation}
                          {task !== null && <div className="muted small">{task}</div>}
                          {line.unitId === ownUnitId && (
                            <div className="muted small">{copy.roster.withYou}</div>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </section>
        ))}
      </div>
    </aside>
  );
}
