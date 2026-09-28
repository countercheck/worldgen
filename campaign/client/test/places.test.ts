import { describe, expect, it } from 'vitest';

import type { Settlement } from '@campaign/shared';

import { centreOn, onScreen, zoomAt } from '../src/map/gesture.js';
import { foldName, placeGroups } from '../src/places.js';

const place = (name: string, tier: string, q = 0, r = 0): Settlement => ({
  coord: { q, r },
  tier,
  role: 'market',
  population: 1000,
  name,
  culture: '',
  etymology: '',
});

const SETTLEMENTS = [
  place('Wick', 'village', 1, 0),
  place('Orléans', 'city', 2, 0),
  place('Ashford', 'town', 3, 0),
  place('Barton', 'village', 4, 0),
  place('Khazad-dûm', 'city', 5, 0),
  place('Brough', 'town', 6, 0),
];

describe('the places list', () => {
  it('groups cities, then towns, then villages, each sorted by name', () => {
    const groups = placeGroups(SETTLEMENTS, '');
    expect(groups.map((g) => g.tier)).toEqual(['city', 'town', 'village']);
    expect(groups[0]!.places.map((p) => p.name)).toEqual(['Khazad-dûm', 'Orléans']);
    expect(groups[2]!.places.map((p) => p.name)).toEqual(['Barton', 'Wick']);
  });

  it('lists every settlement exactly once', () => {
    const listed = placeGroups(SETTLEMENTS, '').flatMap((g) => g.places);
    expect(listed).toHaveLength(SETTLEMENTS.length);
  });

  it('finds a name typed without its accents, in any case', () => {
    expect(placeGroups(SETTLEMENTS, 'orleans').flatMap((g) => g.places.map((p) => p.name))).toEqual([
      'Orléans',
    ]);
    expect(placeGroups(SETTLEMENTS, 'DUM').flatMap((g) => g.places.map((p) => p.name))).toEqual([
      'Khazad-dûm',
    ]);
    expect(foldName('Dûn Éadig')).toBe('dun eadig');
  });

  it('drops a tier with no match, and returns nothing for no match at all', () => {
    expect(placeGroups(SETTLEMENTS, 'ash').map((g) => g.tier)).toEqual(['town']);
    expect(placeGroups(SETTLEMENTS, 'zzz')).toEqual([]);
  });

  it('keeps a tier it does not know, after the ones it does', () => {
    const groups = placeGroups([...SETTLEMENTS, place('Holt', 'hamlet')], '');
    expect(groups.map((g) => g.tier)).toEqual(['city', 'town', 'village', 'hamlet']);
  });
});

describe('bringing a place into view', () => {
  const box = { w: 800, h: 600 };

  it('knows a point inside the box, and one within the margin of its edge', () => {
    expect(onScreen({ x: 400, y: 300 }, box)).toBe(true);
    expect(onScreen({ x: -1, y: 300 }, box)).toBe(false);
    expect(onScreen({ x: 795, y: 300 }, box, 10)).toBe(false);
  });

  /** Where the map draws ground that sits at `fitted` in the fitted view (see `zoomAt`). */
  const screenOf = (fitted: { x: number; y: number }, c: { zoom: number; pan: { x: number; y: number } }) => ({
    x: (fitted.x - box.w / 2) * c.zoom + c.pan.x + box.w / 2,
    y: (fitted.y - box.h / 2) * c.zoom + c.pan.y + box.h / 2,
  });

  it('puts the ground in the middle of the box, at any zoom', () => {
    for (const zoom of [0.5, 1, 3.7]) {
      const camera = { zoom, pan: { x: 123, y: -45 } };
      const fitted = { x: 1500, y: -200 };
      const next = centreOn(camera, fitted, box);
      expect(next.zoom).toBe(zoom);
      const at = screenOf(fitted, next);
      expect(at.x).toBeCloseTo(box.w / 2);
      expect(at.y).toBeCloseTo(box.h / 2);
    }
  });

  it('agrees with zoomAt about where ground is drawn', () => {
    // Zooming about the middle keeps the middle where it is; so a place centred and then
    // zoomed about the middle is still in the middle.
    const centred = centreOn({ zoom: 2, pan: { x: 0, y: 0 } }, { x: 50, y: 70 }, box);
    const zoomed = zoomAt(centred, { x: box.w / 2, y: box.h / 2 }, box, 5);
    const at = screenOf({ x: 50, y: 70 }, zoomed);
    expect(at.x).toBeCloseTo(box.w / 2);
    expect(at.y).toBeCloseTo(box.h / 2);
  });
});
