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

import { Bar, Field, Row, Section } from './parts.js';

const GRADE_LABEL: Record<string, string> = {
  highway: 'Highway',
  road: 'Road',
  off_road: 'Off-road',
  bad_going: 'Bad going',
};

/**
 * Why the ground a formation stands on is not the length of its column.
 *
 * Absent for the formations that *are* the column, so no hint is shown for them.
 */
const FOLD_HINT: Partial<Record<string, string>> = {
  battle: 'deployed at a kilometre of frontage per ten thousand men',
  rest: 'gathered into camp',
  occupation: 'gone into quarters',
};

const pretty = (s: string): string =>
  s.replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase());

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
              {pretty(unit.kind)} · {EXPERIENCE_NAMES[unit.experience]}
            </div>
            <div className="muted">
              {factionName}
              {unit.corps !== null && ` · ${unit.corps}`}
            </div>
          </div>
        </div>
        {isBroken(unit) && <p className="bad">Broken — must rout.</p>}
        {isStarving(unit) && <p className="bad">Out of provisions — cannot march or fight.</p>}
      </Section>

      <Section title="Strength">
        <Row label="PaperStrength" value={unit.paperStrength.toLocaleString()} />
        {/* A patrol is not a formation in miniature. It carries no morale, no supply and
            no fatigue, and its zeroes mean "not tracked" — drawn as empty bars they would
            read as a division on the point of collapse. */}
        {/* A patrol has none of its own, and is immune to all of it. What is shown is the
            parent's, read off the parent rather than copied onto the patrol — a copy would
            be right at the hour it was detached and wrong by the afternoon. */}
        {isPatrol(unit) ? (
          parent === undefined ? (
            <p className="muted small">A detachment. Its parent is out of sight.</p>
          ) : (
            <>
              <p className="muted small">
                A detachment of {parent.name}, and immune to all of it. These are the
                parent&rsquo;s.
              </p>
              <Bar label="Fatigue" value={parent.fatigue} max={100} invert />
              <Bar label="Morale" value={parent.morale} max={maxMorale(parent, cfg.maxMorale)} />
              <Bar label="Provisions" value={parent.provisions} max={parent.maxProvisions} />
              <Bar label="Equipment" value={parent.equipment} max={parent.maxEquipment} />
            </>
          )
        ) : (
          <>
            <Row
              label="Present under arms"
              value={presentUnderArms(unit).toLocaleString()}
              hint="PaperStrength reduced by fatigue"
            />
            <Row label="Guns" value={String(unit.guns)} />
            <Bar label="Fatigue" value={unit.fatigue} max={100} invert />
            <Bar label="Morale" value={unit.morale} max={maxMorale(unit, cfg.maxMorale)} />
            <Bar label="Provisions" value={unit.provisions} max={unit.maxProvisions} />
            <Bar label="Equipment" value={unit.equipment} max={unit.maxEquipment} />
          </>
        )}
      </Section>

      <Section title="Column">
        <Row
          label="Length"
          value={`${lengthKm.toFixed(1)} km`}
          hint="PaperStrength × spacing × multiplier"
        />
        {FOLD_HINT[unit.formation] === undefined ? (
          <Row label="Occupies" value={`${occupied(unit, 'road', cfg.footprint).length} hexes`} />
        ) : (
          <Row
            label="Occupies"
            value={`${occupied(unit, 'road', cfg.footprint).length} hexes`}
            hint={`${columnHexes(unit)} strung out on the march, ${FOLD_HINT[unit.formation]}`}
          />
        )}
        <Row
          label="Catch-up"
          value={`${catchupHours(unit, speed).toFixed(2)} h`}
          hint="For the rear to reach the head, at road speed"
        />
        <Row label="Spacing" value={`${unit.spacingM} m × ${unit.spacingMultiplier}`} />
      </Section>

      <Section title="March">
        <Row label="Formation" value={pretty(unit.formation)} />
        <Row
          label="Marched today"
          value={`${unit.hoursMarchedToday.toFixed(1)} h of ${cfg.maxMarchHoursPerDay} h`}
        />
        <Row label="Remaining" value={`${marchHoursLeftToday(cfg, unit).toFixed(1)} h`} />
        <Field label="Speed by going">
          <table className="grid">
            <tbody>
              {GRADES.map((g) => (
                <tr key={g}>
                  <td>{GRADE_LABEL[g]}</td>
                  <td>{unitSpeedKmh(cfg, unit, g).toFixed(2)} km/h</td>
                  <td className="muted">{(1 / unitSpeedKmh(cfg, unit, g)).toFixed(2)} h/hex</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Field>
      </Section>

      <Section title="Reconnaissance">
        <Row
          label="Sees"
          value={`${reconRadius(cfg, unit)} hex${reconRadius(cfg, unit) === 1 ? '' : 'es'} from the column`}
        />
        {unit.traits.includes('scout') && (
          <Row
            label="Patrols"
            value={
              patrolsOut === undefined
                ? `up to ${cfg.freePatrols} without cost`
                : `${patrolsOut} out of ${cfg.freePatrols} free` +
                  (patrolsOut > cfg.freePatrols ? ` · ${patrolsOut - cfg.freePatrols} paid for` : '')
            }
          />
        )}
        {parent !== undefined && <Row label="Detached from" value={parent.name} />}
      </Section>

      {unit.traits.length > 0 && (
        <Section title="Traits">
          <p className="tags">
            {[...unit.traits].sort().map((t) => (
              <span className="tag" key={t}>
                {pretty(t)}
              </span>
            ))}
          </p>
        </Section>
      )}
    </>
  );
}
