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

import type { PublicFaction, UnitReport } from '@campaign/shared';

import { Row, Section } from './parts.js';

const pretty = (s: string): string => s.replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase());

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
          <div className="unit-kind">Last report</div>
          <div className="muted">
            {faction?.name ?? report.faction}
            {report.corps === null ? '' : ` · ${report.corps}`}
          </div>
        </div>
      </div>

      <Row
        label="Reporting hour"
        value={`${report.atHours} · ${ageLabel(report.atHours, clockHours)}`}
        hint="The hour this describes, which is not necessarily the hour it reached you."
      />
      <Row label="Stood at" value={`${report.head.q}, ${report.head.r}`} />
      <Row label="PaperStrength" value={report.paperStrength.toLocaleString()} />
      <Row label="Fatigue" value={`${Math.round(report.fatigue)} of 100`} />
      <Row label="Provisions" value={String(report.provisions)} />
      <Row label="Formation" value={pretty(report.formation)} />

      {age > 0 && (
        <p className="muted">
          Where they are now is not something you know. At infantry pace they could be
          anywhere within {Math.round(age * 3)} km of that hex by now.
        </p>
      )}
    </Section>
  );
}
