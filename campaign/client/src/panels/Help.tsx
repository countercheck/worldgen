/**
 * How to use the console, behind the Help button.
 *
 * A dialog over whatever is open rather than a page of its own: the question "what does
 * this button do" is asked while looking at the button, and a help page that navigates
 * away from it answers the question somewhere the reader can no longer check it.
 *
 * It opens on the topic for the reader's seat and keeps the others a tab away. The words
 * are all in `copy.help`; this only lays them out.
 */

import { useEffect, useState } from 'react';

import { copy, type HelpSection, type HelpTopic } from '../copy.js';

const TOPICS = Object.keys(copy.help.topics) as HelpTopic[];

/** Where a reader lands: their own half of the game, or the front door if they have no seat. */
export function firstTopic(role: 'referee' | 'commander' | null): HelpTopic {
  return role === 'referee' ? 'referee' : role === 'commander' ? 'command' : 'start';
}

export function Help({
  role,
  onClose,
}: {
  role: 'referee' | 'commander' | null;
  onClose: () => void;
}) {
  const [topic, setTopic] = useState<HelpTopic>(() => firstTopic(role));

  useEffect(() => {
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const sections: readonly HelpSection[] = copy.help.guide[topic];

  return (
    <div className="sheet-backdrop help-backdrop" onClick={onClose}>
      <section
        className="more-sheet help-sheet"
        role="dialog"
        aria-label={copy.help.heading}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="more-head">
          <h2>{copy.help.heading}</h2>
          <button className="dismiss" onClick={onClose}>
            {copy.help.close}
          </button>
        </div>

        <div className="help-topics" role="tablist" aria-label={copy.help.topicsLabel}>
          {TOPICS.map((t) => (
            <button
              key={t}
              role="tab"
              aria-selected={topic === t}
              className={topic === t ? 'on' : ''}
              onClick={() => setTopic(t)}
            >
              {copy.help.topics[t]}
            </button>
          ))}
        </div>

        <div className="help-body" role="tabpanel">
          {sections.map((s) => (
            <section key={s.heading}>
              <h3>{s.heading}</h3>
              {s.body.map((p) => (
                <p key={p}>{p}</p>
              ))}
              {s.terms !== undefined && (
                <dl>
                  {s.terms.map(([term, meaning]) => (
                    <div key={term}>
                      <dt>{term}</dt>
                      <dd>{meaning}</dd>
                    </div>
                  ))}
                </dl>
              )}
            </section>
          ))}
        </div>
      </section>
    </div>
  );
}
