/**
 * An enemy, as far as anybody knows.
 *
 * A separate panel rather than a unit panel with fields blanked out, because those are
 * two different things and only one of them is honest. A blanked-out unit panel implies
 * the numbers exist somewhere in the page and are merely being withheld; this panel says
 * what was actually reported and no more, and it can say nothing else because there is
 * nothing else in a `Contact` to say.
 *
 * What each grade of intelligence earns comes from the rules' patrol table: a distant
 * sighting is a presence, and it takes a patrol closing to identify a formation.
 */

import type { Contact, IntelLevel } from '@campaign/shared';

import { Row, Section } from './parts.js';

/** What a report at each grade actually told you. Straight from the patrol table. */
const INTEL_MEANING: Record<IntelLevel, string> = {
  1: 'Something is there. Nothing more.',
  2: 'Presence and position.',
  3: 'Presence, position and the direction of march.',
  4: 'Rough strength.',
  5: 'The arm — horse, foot or guns.',
  6: 'The formation identified by name.',
};

const pretty = (s: string): string => s.replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase());

export function ContactPanel({
  contact,
  factionName,
  color,
  clockHours,
}: {
  contact: Contact;
  factionName: string;
  color: string;
  clockHours: number;
}) {
  const age = Math.max(0, clockHours - contact.seenAtHours);

  return (
    <Section title="Enemy contact">
      <div className="unit-head">
        <span className="swatch" style={{ background: color }} />
        <div>
          <div className="unit-name">{contact.corps ?? 'Unidentified'}</div>
          <div className="muted">{factionName}</div>
        </div>
      </div>

      <Row label="Seen at" value={`${contact.coord.q}, ${contact.coord.r}`} />
      <Row label="Reported" value={age === 0 ? 'Just now' : `${age.toFixed(1)} h ago`} />
      <Row
        label="Arm"
        value={contact.kind === null ? 'Unknown' : pretty(contact.kind)}
        hint="Only a close patrol reports whether a formation is horse, foot or guns."
      />
      <Row label="Report grade" value={`${contact.intelLevel} of 6`} />

      <p className="muted">{INTEL_MEANING[contact.intelLevel]}</p>
      {age > 0 && (
        <p className="muted">
          This is where it was, not where it is. At infantry pace it could be anywhere
          within {Math.round(age * 3)} km of that hex by now.
        </p>
      )}
    </Section>
  );
}
