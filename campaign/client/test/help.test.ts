import { describe, expect, it } from 'vitest';

import { copy, type HelpTopic } from '../src/copy.js';
import { firstTopic } from '../src/panels/Help.jsx';

describe('help', () => {
  it('opens on the reader’s own half of the game', () => {
    expect(firstTopic('referee')).toBe('referee');
    expect(firstTopic('commander')).toBe('command');
    expect(firstTopic(null)).toBe('start');
  });

  it('has something to say under every tab it offers', () => {
    for (const topic of Object.keys(copy.help.topics) as HelpTopic[]) {
      const sections = copy.help.guide[topic];
      expect(sections.length).toBeGreaterThan(0);
      for (const s of sections) {
        expect(s.body.length + (s.terms?.length ?? 0)).toBeGreaterThan(0);
      }
    }
  });

  it('gives no two sections of a tab the same heading', () => {
    // Headings key the sections when they are drawn, so a repeat would lose one.
    for (const sections of Object.values(copy.help.guide)) {
      const headings = sections.map((s) => s.heading);
      expect(new Set(headings).size).toBe(headings.length);
    }
  });
});
