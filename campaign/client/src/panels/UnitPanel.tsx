/**
 * A unit's state.
 *
 * Everything the rules track, plus the things they make you compute: present under arms,
 * column length, catch-up time, and what the unit's speed actually is on each grade after
 * its traits. Those derived figures are the ones a commander needs and the ones nobody
 * wants to work out by hand — which is the whole reason for a console rather than a
 * spreadsheet.
 */

import {
  catchupHours,
  columnHexes,
  occupied,
  columnLengthKm,
  DEFAULT_CONFIG,
  type CampaignConfig,
  EXPERIENCE_NAMES,
  GRADES,
  isBroken,
  isPatrol,
  isStarving,
  marchHoursLeftToday,
  maxMorale,
  presentUnderArms,
  reconRadius,
  unitSpeedKmh,
  type Unit,
} from '@campaign/shared';

import { copy, hexes, prettify } from '../copy.js';

import { Bar, Field, Row, Section } from './parts.js';

export function UnitPanel({
  unit,
  name,
  factionName,
  color,
  patrolsOut,
  parent,
  cfg = DEFAULT_CONFIG,
}: {
  unit: Unit;
  name: string;
  factionName: string;
  color: string;
  /** How many patrols this formation has in the field. Undefined where it is not known. */
  patrolsOut?: number;
  /** The formation this one was detached from, for a patrol. Its stats stand in for the patrol's. */
  parent?: Unit;
  /** The campaign's numbers, as the server resolved them. */
  cfg?: CampaignConfig;
}) {
  const lengthKm = columnLengthKm(unit);
  const speed = unitSpeedKmh(cfg, unit, 'road');

  return (
    <>
      <Section title={name}>
        <div className="unit-head">
          <span className="swatch" style={{ background: color }} />
          <div>
            <div className="unit-kind">
              {prettify(unit.kind)} · {EXPERIENCE_NAMES[unit.experience]}
            </div>
            <div className="muted">
              {factionName}
              {unit.corps !== null && ` · ${unit.corps}`}
            </div>
          </div>
        </div>
        {isBroken(unit) && <p className="bad">{copy.unit.broken}</p>}
        {isStarving(unit) && <p className="bad">{copy.unit.starving}</p>}
      </Section>

      <Section title={copy.unit.strengthHeading}>
        <Row label={copy.unit.paperStrength} value={unit.paperStrength.toLocaleString()} />
        {/* A patrol is not a formation in miniature. It carries no morale, no supply and
            no fatigue, and its zeroes mean "not tracked" — drawn as empty bars they would
            read as a division on the point of collapse. */}
        {/* A patrol has none of its own, and is immune to all of it. What is shown is the
            parent's, read off the parent rather than copied onto the patrol — a copy would
            be right at the hour it was detached and wrong by the afternoon. */}
        {isPatrol(unit) ? (
          parent === undefined ? (
            <p className="muted small">{copy.unit.patrolOrphaned}</p>
          ) : (
            <>
              <p className="muted small">{copy.unit.patrolOf(parent.name)}</p>
              <Bar label={copy.unit.fatigue} value={parent.fatigue} max={100} invert />
              <Bar
                label={copy.unit.morale}
                value={parent.morale}
                max={maxMorale(parent, cfg.maxMorale)}
              />
              <Bar
                label={copy.unit.provisions}
                value={parent.provisions}
                max={parent.maxProvisions}
              />
              <Bar
                label={copy.unit.equipment}
                value={parent.equipment}
                max={parent.maxEquipment}
              />
            </>
          )
        ) : (
          <>
            <Row
              label={copy.unit.presentUnderArms}
              value={presentUnderArms(unit).toLocaleString()}
              hint={copy.unit.presentUnderArmsHint}
            />
            <Row label={copy.unit.guns} value={String(unit.guns)} />
            <Bar label={copy.unit.fatigue} value={unit.fatigue} max={100} invert />
            <Bar
              label={copy.unit.morale}
              value={unit.morale}
              max={maxMorale(unit, cfg.maxMorale)}
            />
            <Bar
              label={copy.unit.provisions}
              value={unit.provisions}
              max={unit.maxProvisions}
            />
            <Bar label={copy.unit.equipment} value={unit.equipment} max={unit.maxEquipment} />
          </>
        )}
      </Section>

      <Section title={copy.unit.columnHeading}>
        <Row
          label={copy.unit.length}
          value={`${lengthKm.toFixed(1)} km`}
          hint={copy.unit.lengthHint}
        />
        {copy.unit.fold[unit.formation] === undefined ? (
          <Row
            label={copy.unit.occupies}
            value={hexes(occupied(unit, 'road', cfg.footprint).length)}
          />
        ) : (
          <Row
            label={copy.unit.occupies}
            value={hexes(occupied(unit, 'road', cfg.footprint).length)}
            hint={copy.unit.occupiesHint(columnHexes(unit), copy.unit.fold[unit.formation]!)}
          />
        )}
        <Row
          label={copy.unit.catchup}
          value={`${catchupHours(unit, speed).toFixed(2)} h`}
          hint={copy.unit.catchupHint}
        />
        <Row
          label={copy.unit.spacing}
          value={copy.unit.spacingValue(unit.spacingM, unit.spacingMultiplier)}
        />
      </Section>

      <Section title={copy.unit.marchHeading}>
        <Row label={copy.unit.formation} value={prettify(unit.formation)} />
        <Row
          label={copy.unit.marchedToday}
          value={copy.unit.marchedTodayValue(
            unit.hoursMarchedToday.toFixed(1),
            cfg.maxMarchHoursPerDay,
          )}
        />
        <Row
          label={copy.unit.remaining}
          value={`${marchHoursLeftToday(cfg, unit).toFixed(1)} h`}
        />
        <Field label={copy.unit.speedByGoing}>
          <table className="grid">
            <tbody>
              {GRADES.map((g) => (
                <tr key={g}>
                  <td>{prettify(g)}</td>
                  <td>{unitSpeedKmh(cfg, unit, g).toFixed(2)} km/h</td>
                  <td className="muted">{(1 / unitSpeedKmh(cfg, unit, g)).toFixed(2)} h/hex</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Field>
      </Section>

      <Section title={copy.unit.reconHeading}>
        <Row label={copy.unit.sees} value={copy.unit.seesValue(reconRadius(cfg, unit))} />
        {unit.traits.includes('scout') && (
          <Row
            label={copy.unit.patrols}
            value={
              patrolsOut === undefined
                ? copy.unit.patrolsFree(cfg.freePatrols)
                : copy.unit.patrolsOut(patrolsOut, cfg.freePatrols) +
                  (patrolsOut > cfg.freePatrols
                    ? copy.unit.patrolsPaid(patrolsOut - cfg.freePatrols)
                    : '')
            }
          />
        )}
        {parent !== undefined && (
          <Row label={copy.unit.detachedFrom} value={parent.name} />
        )}
      </Section>

      {unit.traits.length > 0 && (
        <Section title={copy.unit.traitsHeading}>
          <p className="tags">
            {[...unit.traits].sort().map((t) => (
              <span className="tag" key={t}>
                {prettify(t)}
              </span>
            ))}
          </p>
        </Section>
      )}
    </>
  );
}
