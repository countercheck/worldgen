import { describe, expect, it } from 'vitest';

import { panesFor } from '../src/layout.js';

describe('phone tabs', () => {
  it('always start on the map', () => {
    expect(panesFor('commander')[0]).toBe('map');
    expect(panesFor('referee')[0]).toBe('map');
  });

  it('give a commander their command, and a referee none', () => {
    expect(panesFor('commander')).toContain('command');
    expect(panesFor('referee')).not.toContain('command');
  });

  it('give both the post and the order of battle', () => {
    for (const role of ['commander', 'referee'] as const) {
      expect(panesFor(role)).toContain('post');
      expect(panesFor(role)).toContain('orbat');
    }
  });
});
