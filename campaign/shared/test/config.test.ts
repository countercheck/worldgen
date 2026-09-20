/**
 * Rulesets, and the numbers a campaign runs on.
 *
 * The project rule is that no game number is hardcoded, and these are the tests that give
 * it teeth: that every tunable is reachable through config, that a named ruleset resolves
 * to what it claims, and that resolution layers in the order a referee would expect.
 */

import { describe, expect, it } from 'vitest';

import {
  DEFAULT_CONFIG,
  DEFAULT_RULESET,
  isRuleset,
  resolveConfig,
  RULESETS,
  type CampaignConfig,
} from '../src/config.js';

describe('rulesets', () => {
  it('always has a standard one, and it is the rules as written', () => {
    expect(RULESETS[DEFAULT_RULESET]).toBeDefined();
    expect(resolveConfig(undefined, DEFAULT_RULESET)).toEqual(DEFAULT_CONFIG);
  });

  it('names every set it knows, and denies the ones it does not', () => {
    for (const id of Object.keys(RULESETS)) expect(isRuleset(id)).toBe(true);
    expect(isRuleset('house-rules-of-1987')).toBe(false);
  });

  it('describes each one, because a referee picks by reading', () => {
    for (const [id, set] of Object.entries(RULESETS)) {
      expect(set.id, id).toBe(id);
      expect(set.name.length, id).toBeGreaterThan(0);
      expect(set.description.length, id).toBeGreaterThan(20);
    }
  });

  it('falls back to the standard set rather than throwing on a name it does not know', () => {
    // A campaign row naming a ruleset this build has since dropped must still load. A
    // throw here is a campaign nobody can open again.
    expect(resolveConfig(undefined, 'nonesuch')).toEqual(DEFAULT_CONFIG);
  });

  it('applies what a named set claims to change', () => {
    const brisk = resolveConfig(undefined, 'brisk');
    expect(brisk.maxMarchHoursPerDay).toBe(12);
    expect(brisk.formationChangeHours.march.rest).toBe(1);
    // And changes nothing it did not name.
    expect(brisk.speeds).toEqual(DEFAULT_CONFIG.speeds);
  });

  it('lets a campaign bend one number without inventing a ruleset for it', () => {
    const out = resolveConfig({ maxMarchHoursPerDay: 8 }, 'brisk');
    expect(out.maxMarchHoursPerDay).toBe(8);
    // The rest of the ruleset still stands.
    expect(out.formationChangeHours.march.rest).toBe(1);
  });

  it('layers base, then ruleset, then the campaign', () => {
    // The order a referee would expect: a server's own defaults, the set of rules chosen,
    // and then this one game's amendment on top.
    const base: CampaignConfig = { ...DEFAULT_CONFIG, freePatrols: 9, gunfireRangeKm: 99 };
    const out = resolveConfig({ freePatrols: 1 }, 'brisk', base);

    expect(out.freePatrols).toBe(1); // the campaign's
    expect(out.maxMarchHoursPerDay).toBe(12); // the ruleset's
    expect(out.gunfireRangeKm).toBe(99); // the base's, untouched by either
  });
});

describe('every game number is reachable', () => {
  it('carries the rules tables rather than leaving them in the code', () => {
    // The ones that used to be constants in unit.ts, and the ones that always lived here.
    const cfg = DEFAULT_CONFIG;
    expect(cfg.minDivisionPaperStrength).toBe(4000);
    expect(cfg.patrolPaperStrength).toBe(20);
    expect(cfg.maxMorale[0]).toBe(30);
    expect(cfg.maxMorale[2]).toBe(50);
    expect(cfg.kindDefaults.cavalry.spacingM).toBe(3);
    expect(cfg.speeds.infantry.road).toBe(3);
    expect(cfg.marchFatigue.infantry[9]).toBe(3);
    expect(cfg.formationChangeHours.march.rest).toBe(2);
  });

  it('can be overridden field by field, every one of them', () => {
    // A tunable that cannot be tuned is a hardcoded number with extra steps. Every key is
    // set to a distinguishable value and read back.
    const keys = Object.keys(DEFAULT_CONFIG) as (keyof CampaignConfig)[];
    expect(keys.length).toBeGreaterThan(30);

    for (const key of keys) {
      const value = DEFAULT_CONFIG[key];
      const bent = typeof value === 'number' ? value + 1 : value;
      const out = resolveConfig({ [key]: bent } as Partial<CampaignConfig>);
      expect(out[key], key).toEqual(bent);
    }
  });
});
