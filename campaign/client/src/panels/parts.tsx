/** Small presentational pieces the panels share. */

import type { ReactNode } from 'react';

export function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="panel-section">
      <h3>{title}</h3>
      {children}
    </section>
  );
}

export function Row({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="row" title={hint}>
      <span className="row-label">{label}</span>
      <span className="row-value">{value}</span>
    </div>
  );
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="field">
      <div className="row-label">{label}</div>
      {children}
    </div>
  );
}

/**
 * A proportion, shown as a bar.
 *
 * `invert` is for quantities where more is worse — fatigue — so the colour still reads
 * as "good" on the left and "bad" on the right without the reader having to remember
 * which way round each one goes.
 */
export function Bar({
  label,
  value,
  max,
  invert = false,
}: {
  label: string;
  value: number;
  max: number;
  invert?: boolean;
}) {
  const frac = max > 0 ? Math.max(0, Math.min(1, value / max)) : 0;
  const health = invert ? 1 - frac : frac;
  const hue = Math.round(health * 110); // 0 red through 110 green
  return (
    <div className="bar-row">
      <span className="row-label">{label}</span>
      <span className="bar">
        <span
          className="bar-fill"
          style={{ width: `${frac * 100}%`, background: `hsl(${hue} 60% 45%)` }}
        />
      </span>
      <span className="bar-value">
        {Math.round(value)}
        <span className="muted"> / {Math.round(max)}</span>
      </span>
    </div>
  );
}
