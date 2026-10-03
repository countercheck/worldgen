/**
 * The map, and the cursor.
 *
 * Two stacked canvases. The lower one holds terrain and is repainted only when the world
 * or the view changes; the upper holds units and the hover ring and is repainted on every
 * pointer move. Without that split, moving the mouse would repaint a thousand hexes per
 * frame and the hover would lag behind the cursor — which would make the sidebar feel
 * broken even though it was correct.
 *
 * A tap and a click are the same act here, and both land when the press lifts rather than
 * when it goes down (see `gesture.ts`): a finger has to press to pan, and a map that
 * selected on every press would deselect on every drag. Two fingers pinch, about the point
 * between them.
 *
 * Hit-testing goes through `pixelToAxial`, the same conversion the Python uses and the
 * one the conformance fixture pins. That matters more than it looks: the rounding at a
 * hex boundary decides which hex a cursor on an edge belongs to, and getting it wrong
 * gives a cursor that reports the neighbour of what it is visibly over.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import {
  axialToPixel,
  key,
  pixelToAxial,
  type Hex,
  type HexKey,
  type Theme,
  type World,
} from '@campaign/shared';

import { copy } from '../copy.js';
import { coarsePointer } from '../layout.js';

import {
  ALL_LAYERS,
  drawFlash,
  drawOverlay,
  drawTerrain,
  drawWash,
  markAtHex,
  worldExtent,
  FLASH_MS,
  type MapLayers,
  type Mark,
  type Plan,
  type Rider,
  type View,
  type WashMode,
} from './draw.js';

import {
  centreOn,
  distance,
  isTap,
  midpoint,
  onScreen,
  zoomAt,
  type Camera,
  type Point,
} from './gesture.js';

export type { Mark, Rider, WashMode } from './draw.js';

/**
 * A place to draw the eye to, from outside the map: the places list. `seq` changes with
 * every request, so choosing the same place twice flashes it twice.
 */
export type Focus = { readonly hex: Hex; readonly seq: number };

/** A press in progress: where it went down, and whether it has stopped being a tap. */
type Press = { start: Point; button: number; moved: boolean };

export function HexMap({
  world,
  marks,
  theme,
  hovered,
  onHover,
  selectedId,
  onSelect,
  reach,
  riders,
  onPick,
  route,
  plans,
  battle,
  visible,
  surveyed,
  washMode,
  focus,
  layers = ALL_LAYERS,
}: {
  world: World;
  marks: readonly Mark[];
  theme: Theme;
  hovered: Hex | null;
  /** The hex under the cursor and what stands on it, by id. */
  onHover: (hex: Hex | null, markId: string | null) => void;
  selectedId: string | null;
  onSelect: (markId: string | null) => void;
  reach?: ReadonlyMap<string, number> | undefined;
  riders?: readonly Rider[] | undefined;
  /**
   * Choosing a hex rather than a unit.
   *
   * When set, a click names ground instead of selecting what is standing on it — which is
   * how a referee sets a destination. Two modes on one surface is usually a mistake, but
   * the alternative is a coordinate box, and a referee reading a despatch that says
   * "Quatre Bras" wants to point at Quatre Bras.
   */
  onPick?: ((hex: Hex) => void) | undefined;
  /** Ground already pointed at in this order, drawn as the referee builds the chain. */
  route?: readonly Hex[] | undefined;
  /** Where every marching column is going. Empty for a commander, who is sent no tasks. */
  plans?: readonly Plan[] | undefined;
  /** Ground being fought over. */
  battle?: ReadonlySet<HexKey> | undefined;
  /**
   * Ground under observation from the formation the viewer rides with, and ground their
   * command has covered at some point. Both empty for a referee, who sees everything.
   */
  visible?: ReadonlySet<HexKey> | undefined;
  surveyed?: ReadonlySet<HexKey> | undefined;
  washMode?: WashMode | undefined;
  /**
   * Flash this place, and bring it into view if it is not. Off screen, the map pans to put
   * it in the middle at the zoom it already has; on screen, it stays put, because moving
   * a map the reader can already see the place on only loses them.
   */
  focus?: Focus | null | undefined;
  /** What the map draws besides the game. Everything, unless the reader has said otherwise. */
  layers?: MapLayers | undefined;
}) {
  const wrap = useRef<HTMLDivElement>(null);
  const terrainRef = useRef<HTMLCanvasElement>(null);
  const washRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const flashRef = useRef<HTMLCanvasElement>(null);

  const [box, setBox] = useState({ w: 800, h: 600 });
  const [camera, setCamera] = useState<Camera>({ zoom: 1, pan: { x: 0, y: 0 } });
  const { zoom, pan } = camera;
  const [hoveredUnitId, setHoveredUnitId] = useState<string | null>(null);
  // The place being flashed and when it started, in `performance.now()` time.
  const [flash, setFlash] = useState<{ hex: Hex; started: number } | null>(null);
  // Every pointer currently down, where it was last seen. One is a pan or a tap; two are a
  // pinch. Refs, not state: they change on every move and nothing renders from them.
  const pointers = useRef(new Map<number, Point>());
  const press = useRef<Press | null>(null);

  // Fit the world to the viewport once, then let zoom and pan work from there.
  const base = useMemo(() => {
    const probe = worldExtent(world, 10);
    const scale = Math.min(box.w / probe.width, box.h / probe.height) * 10;
    const size = Math.max(3, scale);
    const extent = worldExtent(world, size);
    return {
      size,
      offsetX: (box.w - extent.width) / 2 - extent.minX + size,
      offsetY: (box.h - extent.height) / 2 - extent.minY + size,
    };
  }, [world, box.w, box.h]);

  const view: View = useMemo(
    () => ({
      size: base.size * zoom,
      offsetX: base.offsetX * zoom + pan.x + (box.w * (1 - zoom)) / 2,
      offsetY: base.offsetY * zoom + pan.y + (box.h * (1 - zoom)) / 2,
    }),
    [base, zoom, pan, box.w, box.h],
  );

  // Keyed on `seq` alone: the view changes as the map pans, and a pan must not re-run the
  // request that caused it.
  useEffect(() => {
    if (focus === null || focus === undefined) return;
    const at = axialToPixel(focus.hex, view.size);
    const screen = { x: at.x + view.offsetX, y: at.y + view.offsetY };
    // A hex's width in from the edge, so a place half off the map counts as off it.
    if (!onScreen(screen, box, view.size)) {
      const fitted = axialToPixel(focus.hex, base.size);
      setCamera((c) =>
        centreOn(c, { x: fitted.x + base.offsetX, y: fitted.y + base.offsetY }, box),
      );
    }
    setFlash({ hex: focus.hex, started: performance.now() });
  }, [focus?.seq]);

  // Its own canvas and its own frame loop, so the flash animates without repainting the
  // terrain or the units under it. Restarted when the view moves, and picks up where it
  // was, since the time is measured from when the flash began.
  useEffect(() => {
    const canvas = flashRef.current;
    if (canvas === null) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = box.w * dpr;
    canvas.height = box.h * dpr;
    const ctx = canvas.getContext('2d');
    if (ctx === null || flash === null) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const still = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches === true;
    const color =
      getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() ||
      '#cba135';
    let frame = 0;
    const draw = (): void => {
      const elapsed = performance.now() - flash.started;
      ctx.clearRect(0, 0, box.w, box.h);
      if (elapsed >= FLASH_MS) {
        setFlash(null);
        return;
      }
      drawFlash(ctx, view, flash.hex, elapsed, still, color);
      frame = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(frame);
  }, [flash, view, box]);

  useEffect(() => {
    const el = wrap.current;
    if (el === null) return;
    const observer = new ResizeObserver(() => {
      setBox({ w: el.clientWidth, h: el.clientHeight });
    });
    observer.observe(el);
    setBox({ w: el.clientWidth, h: el.clientHeight });
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = terrainRef.current;
    if (canvas === null) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = box.w * dpr;
    canvas.height = box.h * dpr;
    const ctx = canvas.getContext('2d');
    if (ctx === null) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawTerrain(ctx, world, view, theme, layers);
  }, [world, view, theme, box, layers]);

  // Its own pass, and deliberately not in the overlay's dependency list: the wash moves
  // when the clock does, so a hover must not repaint it.
  useEffect(() => {
    const canvas = washRef.current;
    if (canvas === null) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = box.w * dpr;
    canvas.height = box.h * dpr;
    const ctx = canvas.getContext('2d');
    if (ctx === null) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawWash(
      ctx,
      world,
      view,
      {
        visible: visible ?? new Set<HexKey>(),
        surveyed: surveyed ?? new Set<HexKey>(),
        mode: washMode ?? 'three',
      },
      theme,
    );
  }, [world, view, theme, box, visible, surveyed, washMode]);

  useEffect(() => {
    const canvas = overlayRef.current;
    if (canvas === null) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = box.w * dpr;
    canvas.height = box.h * dpr;
    const ctx = canvas.getContext('2d');
    if (ctx === null) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawOverlay(ctx, view, {
      marks,
      hovered,
      hoveredUnitId,
      selectedUnitId: selectedId,
      reach,
      riders,
      picking: onPick === undefined ? null : hovered,
      route,
      plans,
      battle,
    });
  }, [
    marks,
    hovered,
    hoveredUnitId,
    selectedId,
    view,
    box,
    reach,
    riders,
    onPick,
    route,
    plans,
    battle,
  ]);

  const local = useCallback((e: React.PointerEvent | React.WheelEvent): Point => {
    const rect = overlayRef.current?.getBoundingClientRect();
    return rect === undefined
      ? { x: e.clientX, y: e.clientY }
      : { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }, []);

  const hexAt = useCallback(
    (p: Point): Hex | null => {
      const c = pixelToAxial(p.x - view.offsetX, p.y - view.offsetY, view.size);
      return world.hexes.has(key(c)) ? c : null;
    },
    [view, world],
  );

  const hoverAt = useCallback(
    (p: Point) => {
      const hex = hexAt(p);
      const id = hex === null ? null : markAtHex(marks, hex);
      setHoveredUnitId(id);
      onHover(hex, id);
      return { hex, id };
    },
    [hexAt, marks, onHover],
  );

  const onPointerMove = useCallback(
    (e: React.PointerEvent) => {
      const at = local(e);
      const last = pointers.current.get(e.pointerId);

      // Nothing pressed: a mouse moving over the map, which is a hover.
      if (last === undefined) {
        if (e.pointerType === 'mouse') hoverAt(at);
        return;
      }

      // Still a tap: leave the pointer where it went down, so that when it does leave the
      // slop the pan takes in the whole way it has come rather than trailing by the slop.
      const p = press.current;
      if (p !== null && !p.moved && isTap(p.start, at)) return;

      const others = [...pointers.current.entries()].filter(([id]) => id !== e.pointerId);
      pointers.current.set(e.pointerId, at);

      const other = others[0]?.[1];
      if (other !== undefined) {
        // A pinch: zoom by how far the fingers spread, about the point between them, and
        // pan by how far that point moved.
        const before = distance(last, other);
        const after = distance(at, other);
        const from = midpoint(last, other);
        const to = midpoint(at, other);
        setCamera((c) => {
          const zoomed = before > 0 ? zoomAt(c, from, box, c.zoom * (after / before)) : c;
          return {
            zoom: zoomed.zoom,
            pan: { x: zoomed.pan.x + to.x - from.x, y: zoomed.pan.y + to.y - from.y },
          };
        });
        return;
      }

      if (p !== null) p.moved = true;
      setCamera((c) => ({
        zoom: c.zoom,
        pan: { x: c.pan.x + at.x - last.x, y: c.pan.y + at.y - last.y },
      }));
    },
    [local, hoverAt, box],
  );

  const release = useCallback((e: React.PointerEvent) => {
    pointers.current.delete(e.pointerId);
    if (pointers.current.size === 0) press.current = null;
    const el = e.currentTarget as HTMLElement;
    if (el.hasPointerCapture(e.pointerId)) el.releasePointerCapture(e.pointerId);
  }, []);

  return (
    <div
      ref={wrap}
      className="map"
      onPointerDown={(e) => {
        const at = local(e);
        pointers.current.set(e.pointerId, at);
        // A second finger makes the gesture a pinch, and a pinch is never a tap.
        if (pointers.current.size === 1) {
          press.current = { start: at, button: e.button, moved: false };
        } else if (press.current !== null) {
          press.current.moved = true;
        }
        e.currentTarget.setPointerCapture(e.pointerId);
      }}
      onPointerUp={(e) => {
        const p = press.current;
        const tapped = p !== null && !p.moved && p.button === 0 && pointers.current.size === 1;
        release(e);
        if (!tapped) return;

        // A finger has no hover, so its tap is also the hover: the sidebar reads the ground
        // or the column it landed on, exactly as a mouse resting there would make it.
        const { hex, id } = e.pointerType === 'mouse'
          ? { hex: hexAt(local(e)), id: null as string | null }
          : hoverAt(local(e));
        if (onPick !== undefined) {
          if (hex !== null) onPick(hex);
        } else {
          onSelect(hex === null ? null : (id ?? markAtHex(marks, hex)));
        }
      }}
      onPointerCancel={release}
      onPointerMove={onPointerMove}
      onPointerLeave={(e) => {
        // A lifted finger leaves too, and clearing on that would throw away what the tap
        // just put in the sidebar.
        if (e.pointerType !== 'mouse') return;
        setHoveredUnitId(null);
        onHover(null, null);
      }}
      onWheel={(e) => {
        const at = local(e);
        setCamera((c) => zoomAt(c, at, box, c.zoom * (e.deltaY < 0 ? 1.12 : 1 / 1.12)));
      }}
    >
      <canvas ref={terrainRef} style={{ width: box.w, height: box.h }} />
      {/* Between the ground and the troops on it: the wash dims terrain, never a symbol. */}
      <canvas ref={washRef} style={{ width: box.w, height: box.h }} />
      <canvas ref={overlayRef} style={{ width: box.w, height: box.h }} />
      <canvas ref={flashRef} className="map-flash" style={{ width: box.w, height: box.h }} />
      {/* Only while nothing is being pointed at. Every pointing mode has a banner above the
          map saying what a click will do and how to get out, and a second line down here
          could only repeat it or, as it once did, contradict it. */}
      {onPick === undefined && (
        <div className="map-hint">{coarsePointer() ? copy.map.hintTouch : copy.map.hint}</div>
      )}
    </div>
  );
}
