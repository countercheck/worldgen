/**
 * The map, and the cursor.
 *
 * Two stacked canvases. The lower one holds terrain and is repainted only when the world
 * or the view changes; the upper holds units and the hover ring and is repainted on every
 * pointer move. Without that split, moving the mouse would repaint a thousand hexes per
 * frame and the hover would lag behind the cursor — which would make the sidebar feel
 * broken even though it was correct.
 *
 * Hit-testing goes through `pixelToAxial`, the same conversion the Python uses and the
 * one the conformance fixture pins. That matters more than it looks: the rounding at a
 * hex boundary decides which hex a cursor on an edge belongs to, and getting it wrong
 * gives a cursor that reports the neighbour of what it is visibly over.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import {
  key,
  occupied,
  pixelToAxial,
  type Hex,
  type Theme,
  type Unit,
  type World,
} from '@campaign/shared';

import { drawOverlay, drawTerrain, unitAtHex, worldExtent, type View } from './draw.js';

export interface DrawableUnit {
  readonly unit: Unit;
  readonly color: string;
  /** False for an enemy known only by an old sighting. */
  readonly visible: boolean;
}

export function HexMap({
  world,
  units,
  theme,
  hovered,
  onHover,
  selectedId,
  onSelect,
  reach,
}: {
  world: World;
  units: readonly DrawableUnit[];
  theme: Theme;
  hovered: Hex | null;
  onHover: (hex: Hex | null, unit: Unit | null) => void;
  selectedId: string | null;
  onSelect: (unit: Unit | null) => void;
  reach?: ReadonlyMap<string, number> | undefined;
}) {
  const wrap = useRef<HTMLDivElement>(null);
  const terrainRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);

  const [box, setBox] = useState({ w: 800, h: 600 });
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [hoveredUnitId, setHoveredUnitId] = useState<string | null>(null);
  const dragging = useRef<{ x: number; y: number } | null>(null);

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

  const drawable = useMemo(
    () => units.map((u) => ({ ...u, column: occupied(u.unit) })),
    [units],
  );

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
    drawTerrain(ctx, world, view, theme);
  }, [world, view, theme, box]);

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
      units: drawable,
      hovered,
      hoveredUnitId,
      selectedUnitId: selectedId,
      reach,
    });
  }, [drawable, hovered, hoveredUnitId, selectedId, view, box, reach]);

  const hexAtPointer = useCallback(
    (e: React.PointerEvent): Hex | null => {
      const canvas = overlayRef.current;
      if (canvas === null) return null;
      const rect = canvas.getBoundingClientRect();
      const c = pixelToAxial(
        e.clientX - rect.left - view.offsetX,
        e.clientY - rect.top - view.offsetY,
        view.size,
      );
      return world.hexes.has(key(c)) ? c : null;
    },
    [view, world],
  );

  const onPointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (dragging.current !== null) {
        setPan((p) => ({
          x: p.x + e.clientX - dragging.current!.x,
          y: p.y + e.clientY - dragging.current!.y,
        }));
        dragging.current = { x: e.clientX, y: e.clientY };
        return;
      }
      const hex = hexAtPointer(e);
      const unit = hex === null ? null : unitAtHex(drawable, hex);
      setHoveredUnitId(unit?.id ?? null);
      onHover(hex, unit);
    },
    [hexAtPointer, drawable, onHover],
  );

  return (
    <div
      ref={wrap}
      className="map"
      onPointerDown={(e) => {
        if (e.button === 0) {
          const hex = hexAtPointer(e);
          onSelect(hex === null ? null : unitAtHex(drawable, hex));
        }
        dragging.current = { x: e.clientX, y: e.clientY };
        (e.target as HTMLElement).setPointerCapture(e.pointerId);
      }}
      onPointerUp={(e) => {
        dragging.current = null;
        (e.target as HTMLElement).releasePointerCapture(e.pointerId);
      }}
      onPointerMove={onPointerMove}
      onPointerLeave={() => {
        setHoveredUnitId(null);
        onHover(null, null);
      }}
      onWheel={(e) => {
        setZoom((z) => Math.max(0.4, Math.min(8, z * (e.deltaY < 0 ? 1.12 : 1 / 1.12))));
      }}
    >
      <canvas ref={terrainRef} style={{ width: box.w, height: box.h }} />
      <canvas ref={overlayRef} style={{ width: box.w, height: box.h }} />
      <div className="map-hint">scroll to zoom · drag to pan · click a unit to select</div>
    </div>
  );
}
