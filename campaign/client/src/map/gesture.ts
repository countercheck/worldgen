/**
 * What a hand on the map means, worked out apart from the map.
 *
 * A mouse and a finger ask different things of the same surface. A mouse can hover, so a
 * press can mean "select this" the instant it lands. A finger cannot, and every pan
 * starts with a press — select on press and every drag across the map would first
 * deselect whatever the reader was looking at. So a press is only a tap once it lifts
 * again without having travelled, and the distance it may travel is the slop below.
 *
 * Pure, so it can be tested without a canvas or a DOM.
 */

export type Point = { readonly x: number; readonly y: number };

/** Where the camera is: a zoom factor over the fitted view, and a pan in CSS pixels. */
export type Camera = { readonly zoom: number; readonly pan: Point };

/**
 * How far a press may wander and still be a tap, in CSS pixels.
 *
 * Big enough for a fingertip that rolls as it lifts, small enough that a deliberate drag
 * is never read as one.
 */
export const TAP_SLOP = 8;

export const MIN_ZOOM = 0.4;
export const MAX_ZOOM = 8;

export function clampZoom(zoom: number): number {
  return Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoom));
}

export function distance(a: Point, b: Point): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

export function midpoint(a: Point, b: Point): Point {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

/** Whether a press that went down at `from` and is now at `to` is still a tap. */
export function isTap(from: Point, to: Point, slop = TAP_SLOP): boolean {
  return distance(from, to) <= slop;
}

/**
 * Zoom to `zoom` while keeping the ground under `at` where it is on screen.
 *
 * `at` is relative to the map's own box. The map places a hex at
 * `c·zoom + pan + box/2`, where `c` depends only on the hex and the fitted view; holding
 * the screen position of the point under `at` fixed across a change of zoom gives the pan
 * below. Zooming about the pointer, or about the middle of two fingers, is what makes a
 * pinch feel like it has hold of the map rather than of the middle of the window.
 */
export function zoomAt(camera: Camera, at: Point, box: { w: number; h: number }, zoom: number): Camera {
  const next = clampZoom(zoom);
  const ratio = next / camera.zoom;
  const cx = at.x - box.w / 2;
  const cy = at.y - box.h / 2;
  return {
    zoom: next,
    pan: {
      x: cx - (cx - camera.pan.x) * ratio,
      y: cy - (cy - camera.pan.y) * ratio,
    },
  };
}
