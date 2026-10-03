import { describe, expect, it } from 'vitest';

import { copy } from '../src/copy.js';

describe('the message carrying every commander’s link', () => {
  const seats = [
    { label: 'Marshal Ney', link: 'https://example.test/#/j/abc/ney-token' },
    { label: 'The Earl of Uxbridge', link: 'https://example.test/#/j/abc/uxbridge-token' },
  ];

  it('names the campaign, then gives each seat its own line with its whole link', () => {
    const lines = copy.join.allLinks('Demonstration', seats).split('\n');
    expect(lines[0]).toContain('Demonstration');
    for (const { label, link } of seats) {
      expect(lines).toContain(`${label}: ${link}`);
    }
  });

  it('carries only the links it was given — never the referee’s', () => {
    const message = copy.join.allLinks('Demonstration', seats);
    expect(message.match(/https:\/\//g)).toHaveLength(seats.length);
  });

  it('keeps a name’s own capitals: "of" stays lower case', () => {
    expect(copy.join.allLinks('D', seats)).toContain('The Earl of Uxbridge:');
  });

  it('reads a sighting by what is known of it, keeping the staff’s number', () => {
    expect(copy.idle.contactLabel('c1', null)).toBe('Unidentified column · c1');
    expect(copy.idle.contactLabel('c2', 'Cavalry')).toBe('Cavalry column · c2');
  });
});
