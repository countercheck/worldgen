/**
 * What a march costs the troops.
 *
 * The table is checked against the rules directly, band by band, because every one of
 * these numbers is a game number somebody will want to argue about — and because the curve
 * is held cumulatively while the rules state it as bands, which is exactly the kind of
 * transcription that goes wrong silently.
 */

import { describe, expect, it } from 'vitest';

import { catchupHours } from '../src/column.js';
import { DEFAULT_CONFIG } from '../src/config.js';
import {
  campBreak,
  campDay,
  columnMotionWindow,
  darkHoursBetween,
  isDark,
  marchFatigueAt,
  marchFatigueBetween,
  nightFatigue,
} from '../src/fatigue.js';
import { presentUnderArms, type Experience, type Unit, type UnitKind } from '../src/unit.js';

const cfg = DEFAULT_CONFIG;

const unit = (kind: UnitKind = 'infantry', experience: Experience = 0): Unit => ({
  id: 'u',
  name: 'u',
  faction: 'red',
  kind,
  paperStrength: 4000,
  fatigue: 0,
  experience,
  morale: 30,
  provisions: 40,
  maxProvisions: 40,
  equipment: 30,
  maxEquipment: 30,
  guns: 0,
  marchSpeedKmh: 3,
  spacingM: 0.5,
  spacingMultiplier: 1,
  traits: [],
  formation: 'march',
  formationChange: null,
  column: [{ q: 0, r: 0 }],
  hoursMarchedToday: 0,
  corps: null,
});

describe("the rules' fatigue table", () => {
  it('matches the infantry row band for band', () => {
    // 0-4h: 0; then 1, 1, 2, 2, 3, 4, 5, 6 through the twelfth hour.
    const expected: [number, number][] = [
      [0, 0], [1, 0], [2, 0], [3, 0], [4, 0],
      [5, 1], [6, 1], [7, 2], [8, 2], [9, 3], [10, 4], [11, 5], [12, 6],
    ];
    for (const [hours, fatigue] of expected) {
      expect(marchFatigueAt(cfg, unit('infantry'), hours), `${hours}h`).toBe(fatigue);
    }
  });

  it('matches the cavalry row, which runs an hour ahead of the infantry', () => {
    const expected: [number, number][] = [
      [0, 1], [1, 1], [2, 1], [3, 1], [4, 1],
      [5, 1], [6, 2], [7, 2], [8, 3], [9, 4], [10, 5], [11, 6],
    ];
    for (const [hours, fatigue] of expected) {
      expect(marchFatigueAt(cfg, unit('cavalry'), hours), `${hours}h`).toBe(fatigue);
    }
  });

  it('adds two an hour past the end of the table', () => {
    const inf = unit('infantry');
    expect(marchFatigueAt(cfg, inf, 13)).toBe(8);
    expect(marchFatigueAt(cfg, inf, 14)).toBe(10);
    expect(marchFatigueAt(cfg, inf, 20)).toBe(22);
  });

  it('shifts the column by experience, left for a veteran and right for a raw one', () => {
    // The rules' own mechanism, and the reason experience is worth having: it does not
    // make troops march faster, it makes them arrive able to fight.
    const at = (xp: Experience) => marchFatigueAt(cfg, unit('infantry', xp), 9);
    expect(at(0)).toBe(3);
    expect(at(1)).toBe(2);
    expect(at(2)).toBe(2);
    expect(at(-1)).toBe(4);
    expect(at(-2)).toBe(5);
  });

  it('charges a stretch of road by subtraction, so the bands compose', () => {
    // The whole reason the curve is held cumulatively: marching from the eighth hour to
    // the eleventh costs the same whether it is charged in one step or six.
    const u = unit('infantry');
    expect(marchFatigueBetween(cfg, u, 8, 11)).toBe(3);

    let piecemeal = 0;
    for (let h = 8; h < 11; h += 0.5) piecemeal += marchFatigueBetween(cfg, u, h, h + 0.5);
    expect(piecemeal).toBe(3);
  });

  it('charges nothing for standing still', () => {
    expect(marchFatigueBetween(cfg, unit('infantry'), 6, 6)).toBe(0);
  });
});

describe('darkness', () => {
  it('is between sunset and sunrise, whatever day it is', () => {
    expect(isDark(cfg, 12)).toBe(false);
    expect(isDark(cfg, 6)).toBe(false);
    expect(isDark(cfg, 5.99)).toBe(true);
    expect(isDark(cfg, 18)).toBe(true);
    expect(isDark(cfg, 23)).toBe(true);
    expect(isDark(cfg, 24 * 3 + 2)).toBe(true);
  });

  it('measures a window that straddles sunset', () => {
    // A step that crosses the boundary is charged for the part of it that was dark, not
    // all or nothing on where its midpoint happened to fall.
    expect(darkHoursBetween(cfg, 17, 19)).toBeCloseTo(1, 9);
    expect(darkHoursBetween(cfg, 5, 7)).toBeCloseTo(1, 9);
  });

  it('measures a window spanning a whole night', () => {
    expect(darkHoursBetween(cfg, 18, 30)).toBeCloseTo(12, 9);
    expect(darkHoursBetween(cfg, 12, 12)).toBe(0);
    expect(darkHoursBetween(cfg, 8, 16)).toBe(0);
  });

  it('measures a window spanning more than one night', () => {
    // Twenty-four hours contains exactly one night, wherever it starts.
    expect(darkHoursBetween(cfg, 10, 34)).toBeCloseTo(12, 9);
    expect(darkHoursBetween(cfg, 0, 48)).toBeCloseTo(24, 9);
  });
});

describe('the tail, and why it matters', () => {
  it('leaves the column on the road after the head has stopped', () => {
    const u = unit('infantry');
    const window = columnMotionWindow(u, 10, 14, 3);
    expect(window.from).toBe(10);
    // Two kilometres of division at three km/h is forty minutes behind the head.
    expect(window.to).toBeCloseTo(14 + catchupHours(u, 3), 9);
    expect(window.to).toBeGreaterThan(14);
  });

  it('makes a march that ends at dusk a night march for the rear', () => {
    // The head camps in the last of the light and pays nothing. The troops at the back are
    // still walking half an hour later, in the dark, and the rules charge for that.
    const u = unit('infantry');
    const headStopped = cfg.sunsetHour;
    const window = columnMotionWindow(u, headStopped - 4, headStopped, 3);

    expect(darkHoursBetween(cfg, headStopped - 4, headStopped)).toBe(0);
    expect(nightFatigue(cfg, headStopped, window.to)).toBeGreaterThan(0);
  });

  it('charges a longer column more of the night than a shorter one', () => {
    const small = unit('infantry');
    const long = { ...unit('infantry'), paperStrength: 12000 };
    const at = cfg.sunsetHour;
    const of = (u: Unit) => nightFatigue(cfg, at, columnMotionWindow(u, at - 1, at, 3).to);
    expect(of(long)).toBeGreaterThan(of(small));
  });
});

describe('a day in camp', () => {
  const tired = (fatigue: number, morale: number): Unit => ({ ...unit(), fatigue, morale });

  it('sheds the configured fatigue, and brings in the troops it stood for', () => {
    const day = campDay(cfg, tired(80, 30));
    expect(day.fatigue).toBe(cfg.campFatigueRecoveryPerDay);
    expect(day.recoveredTroops).toBe(Math.round((4000 * cfg.campFatigueRecoveryPerDay) / 100));
  });

  it('counts the missing and the ill among them, in the configured shares', () => {
    const day = campDay(cfg, tired(80, 30));
    expect(day.missing).toBe(Math.round(day.recoveredTroops * cfg.campMissingShare));
    expect(day.ill).toBe(Math.round(day.recoveredTroops * cfg.campIllShare));
    expect(day.missing + day.ill).toBeLessThanOrEqual(day.recoveredTroops);
  });

  it('never sheds more fatigue than there is', () => {
    const day = campDay(cfg, tired(5, 30));
    expect(day.fatigue).toBe(5);
    expect(campDay(cfg, tired(0, 30))).toMatchObject({ fatigue: 0, recoveredTroops: 0, missing: 0, ill: 0 });
  });

  it('restores morale up to the experience ceiling and no further', () => {
    expect(campDay(cfg, tired(0, 0)).morale).toBe(cfg.campMoraleRecoveryPerDay);
    // A regular's ceiling is thirty.
    expect(campDay(cfg, tired(0, 25)).morale).toBe(5);
    expect(campDay(cfg, tired(0, 30)).morale).toBe(0);
    // Above it, by a referee's hand, is left where it is rather than pulled down.
    expect(campDay(cfg, tired(0, 45)).morale).toBe(0);
  });

  const rest = (u: Unit): Unit => {
    const day = campDay(cfg, u);
    return {
      ...u,
      fatigue: u.fatigue - day.fatigue,
      morale: u.morale + day.morale,
      campMissing: (u.campMissing ?? 0) + day.missing,
      campIll: (u.campIll ?? 0) + day.ill,
    };
  };

  it('recovers even a broken formation', () => {
    // Morale 0, fatigue 80, two days: morale 20, fatigue 40.
    const u = rest(rest(tired(80, 0)));
    expect(u.morale).toBe(20);
    expect(u.fatigue).toBe(40);
  });

  it('puts exactly the troops who came back into the line, day after day', () => {
    let u = tired(80, 30);
    for (let d = 0; d < 4; d++) {
      const day = campDay(cfg, u);
      const next = rest(u);
      expect(presentUnderArms(next) - presentUnderArms(u)).toBe(
        day.recoveredTroops - day.missing - day.ill,
      );
      expect(next.fatigue).toBe(u.fatigue - cfg.campFatigueRecoveryPerDay);
      // The rolls do not move while the camp stands.
      expect(next.paperStrength).toBe(u.paperStrength);
      u = next;
    }
  });

  it('matches the worked example', () => {
    // Four thousand at fatigue 80: eight hundred fit. Each day brings 800 in, 240 of them lost.
    const one = rest(tired(80, 30));
    expect(one.fatigue).toBe(60);
    expect(presentUnderArms(one)).toBe(1360);
    const two = rest(one);
    expect(two.fatigue).toBe(40);
    expect(presentUnderArms(two)).toBe(1920);
  });
});

describe('breaking camp', () => {
  // Two days' rest from fatigue 80 on four thousand: 480 found missing or ill.
  const camped: Unit = { ...unit(), fatigue: 40, campMissing: 160, campIll: 320 };
  const broken = (u: Unit): Unit => {
    const b = campBreak(u);
    return { ...u, paperStrength: u.paperStrength - b.missing - b.ill, fatigue: b.fatigue, campMissing: 0, campIll: 0 };
  };

  it('strikes the missing and the ill off the rolls', () => {
    expect(campBreak(camped)).toMatchObject({ missing: 160, ill: 320 });
    expect(broken(camped).paperStrength).toBe(3520);
  });

  it('leaves present under arms where it was', () => {
    expect(presentUnderArms(broken(camped))).toBe(presentUnderArms(camped));
  });

  it('reads the still-fatigued as a share of the smaller rolls', () => {
    expect(campBreak(camped).fatigue).toBeCloseTo((1600 / 3520) * 100, 9);
    expect(campBreak({ ...camped, fatigue: 0 }).fatigue).toBe(0);
  });

  it('costs nothing when the camp found nobody', () => {
    expect(campBreak({ ...unit(), fatigue: 30 })).toEqual({ missing: 0, ill: 0, fatigue: 30 });
  });
});
