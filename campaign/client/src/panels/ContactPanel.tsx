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

import type { PublicContact } from '@campaign/shared';

import { copy, prettify } from '../copy.js';

import { Row, Section } from './parts.js';

export function ContactPanel({
  contact,
  factionName,
  color,
  clockHours,
}: {
  contact: PublicContact;
  factionName: string;
  color: string;
  clockHours: number;
}) {
  const age = Math.max(0, clockHours - contact.seenAtHours);

  return (
    <Section title={copy.contact.title(contact.id)}>
      <div className="unit-head">
        <span className="swatch" style={{ background: color }} />
        <div>
          <div className="unit-name">{contact.corps ?? copy.contact.unidentified}</div>
          <div className="muted">{factionName}</div>
        </div>
      </div>

      <Row label={copy.contact.seenAt} value={`${contact.coord.q}, ${contact.coord.r}`} />
      <Row
        label={copy.contact.reported}
        value={age === 0 ? copy.contact.justNow : copy.contact.ago(age.toFixed(1))}
      />
      <Row
        label={copy.contact.arm}
        value={contact.kind === null ? copy.contact.armUnknown : prettify(contact.kind)}
        hint={copy.contact.armHint}
      />
      <Row label={copy.contact.grade} value={copy.contact.gradeValue(contact.intelLevel)} />

      <p className="muted">{copy.contact.intel[contact.intelLevel]}</p>
      {age > 0 && <p className="muted">{copy.contact.drift(Math.round(age * 3))}</p>}
      <p className="muted small">{copy.contact.yourOwnNumber}</p>
    </Section>
  );
}
