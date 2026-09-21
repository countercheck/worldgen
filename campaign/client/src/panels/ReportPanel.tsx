/**
 * One of your own formations, as you last heard of it.
 *
 * A separate panel from `UnitPanel`, and the separation is the point. A unit panel shows
 * what is true; this shows what was true at an hour that has passed, and a reader who
 * cannot tell those apart at a glance will make plans on the wrong one. So the hour comes
 * first, before the numbers, and the numbers are stated in the past tense.
 *
 * There is no fatigue bar or provisions bar here even though the report carries both.
 * A bar reads as a gauge — a live reading of a current quantity — and that is precisely
 * the wrong impression. Figures with an hour attached read as a despatch.
 */

import { ageHours, ageLabel } from '../board.js';
import { copy, prettify } from '../copy.js';

import type { PublicFaction, UnitReport } from '@campaign/shared';

import { Row, Section } from './parts.js';

export function ReportPanel({
  report,
  faction,
  clockHours,
}: {
  report: UnitReport;
  faction: PublicFaction | undefined;
  clockHours: number;
}) {
  const age = ageHours(report.atHours, clockHours);

  return (
    <Section title={report.name}>
      <div className="unit-head">
        <span className="swatch ghost" style={{ background: faction?.color ?? '#888' }} />
        <div>
          <div className="unit-kind">{copy.report.lastReport}</div>
          <div className="muted">
            {faction?.name ?? report.faction}
            {report.corps === null ? '' : ` · ${report.corps}`}
          </div>
        </div>
      </div>

      <Row
        label={copy.report.reportingHour}
        value={`${report.atHours} · ${ageLabel(report.atHours, clockHours)}`}
        hint={copy.report.reportingHourHint}
      />
      <Row label={copy.report.stoodAt} value={`${report.head.q}, ${report.head.r}`} />
      <Row label={copy.report.paperStrength} value={report.paperStrength.toLocaleString()} />
      <Row
        label={copy.report.fatigue}
        value={copy.report.fatigueValue(Math.round(report.fatigue))}
      />
      <Row label={copy.report.provisions} value={String(report.provisions)} />
      <Row label={copy.report.formation} value={prettify(report.formation)} />

      {age > 0 && <p className="muted">{copy.report.drift(Math.round(age * 3))}</p>}
    </Section>
  );
}
