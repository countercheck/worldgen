/**
 * The map's reading of a hand.
 *
 * The property that matters for zooming is that the ground under the pointer stays under
 * it. It is checked the way the map places a hex — `c·zoom + pan + box/2` — rather than
 * against numbers worked out by hand.
 */

import { describe, expect, it } from 'vitest';

import {
  clampZoom,
  isTap,
  MAX_ZOOM,
  midpoint,
  MIN_ZOOM,
  TAP_SLOP,
  zoomAt,
  type Camera,
} from '../src/map/gesture.js';

const box = { w: 400, h: 800 };

/** Where the map draws a point whose zoom-independent offset is `c`. */
function onScreen(camera: Camera, c: { x: number; y: number }) {
  return {
    x: c.x * camera.zoom + camera.pan.x + box.w / 2,
    y: c.y * camera.zoom + camera.pan.y + box.h / 2,
  };
}

/** The `c` of whatever is drawn at screen point `p` under `camera`. */
function under(camera: Camera, p: { x: number; y: number }) {
  return {
    x: (p.x - camera.pan.x - box.w / 2) / camera.zoom,
    y: (p.y - camera.pan.y - box.h / 2) / camera.zoom,
  };
}

describe('taps', () => {
  it('a press that lifts where it landed is a tap', () => {
    expect(isTap({ x: 10, y: 10 }, { x: 10, y: 10 })).toBe(true);
  });

  it('a fingertip may roll a little and still tap', () => {
    expect(isTap({ x: 0, y: 0 }, { x: TAP_SLOP * 0.6, y: TAP_SLOP * 0.6 })).toBe(true);
  });

  it('a drag is never a tap', () => {
    expect(isTap({ x: 0, y: 0 }, { x: TAP_SLOP + 1, y: 0 })).toBe(false);
  });
});

describe('zoom', () => {
  it('is kept inside its limits', () => {
    expect(clampZoom(100)).toBe(MAX_ZOOM);
    expect(clampZoom(0)).toBe(MIN_ZOOM);
    expect(clampZoom(1.5)).toBe(1.5);
  });

  it('keeps the ground under the pointer where it is', () => {
    const from: Camera = { zoom: 1.3, pan: { x: -40, y: 25 } };
    for (const at of [
      { x: 0, y: 0 },
      { x: 200, y: 400 },
      { x: 390, y: 12 },
    ]) {
      const c = under(from, at);
      const to = zoomAt(from, at, box, 2.7);
      const landed = onScreen(to, c);
      expect(landed.x).toBeCloseTo(at.x, 6);
      expect(landed.y).toBeCloseTo(at.y, 6);
    }
  });

  it('changes nothing when the zoom does not change', () => {
    const from: Camera = { zoom: 2, pan: { x: 13, y: -7 } };
    const to = zoomAt(from, { x: 50, y: 60 }, box, 2);
    expect(to.pan.x).toBeCloseTo(from.pan.x, 9);
    expect(to.pan.y).toBeCloseTo(from.pan.y, 9);
  });

  it('holds the pointer even when the zoom is clamped', () => {
    const from: Camera = { zoom: MAX_ZOOM * 0.9, pan: { x: 0, y: 0 } };
    const at = { x: 80, y: 700 };
    const c = under(from, at);
    const to = zoomAt(from, at, box, MAX_ZOOM * 5);
    expect(to.zoom).toBe(MAX_ZOOM);
    expect(onScreen(to, c).x).toBeCloseTo(at.x, 6);
  });

  it('a pinch zooms about the middle of the two fingers', () => {
    expect(midpoint({ x: 0, y: 0 }, { x: 100, y: 50 })).toEqual({ x: 50, y: 25 });
  });
});
