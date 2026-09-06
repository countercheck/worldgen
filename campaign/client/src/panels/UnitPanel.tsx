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
  columnLengthKm,
  DEFAULT_CONFIG,
  EXPERIENCE_NAMES,
  GRADES,
  isBroken,
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

const pretty = (s: string): string =>
  s.replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase());

export function UnitPanel({
  unit,
  name,
  factionName,
  color,
}: {
  unit: Unit;
  name: string;
  factionName: string;
  color: string;
}) {
  const cfg = DEFAULT_CONFIG;
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
        <Row label="Effectives" value={unit.effectives.toLocaleString()} />
        <Row
          label="Present under arms"
          value={presentUnderArms(unit).toLocaleString()}
          hint="Effectives reduced by fatigue"
        />
        <Row label="Guns" value={String(unit.guns)} />
        <Bar label="Fatigue" value={unit.fatigue} max={100} invert />
        <Bar label="Morale" value={unit.morale} max={maxMorale(unit)} />
        <Bar label="Provisions" value={unit.provisions} max={unit.maxProvisions} />
        <Bar label="Equipment" value={unit.equipment} max={unit.maxEquipment} />
      </Section>

      <Section title="Column">
        <Row
          label="Length"
          value={`${lengthKm.toFixed(1)} km`}
          hint="Effectives × spacing × multiplier"
        />
        <Row label="Occupies" value={`${columnHexes(unit)} hexes`} />
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
          <Row label="Patrols" value={`up to ${cfg.freePatrols} without cost`} />
        )}
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
