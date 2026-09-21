/**
 * NATO unit symbols, drawn on a canvas.
 *
 * APP-6 in the small: a frame whose shape says what kind of claim this is, an interior
 * icon saying what arm the formation is, and size marks above saying how large. The rest
 * of the standard — status, mobility, higher formation, the whole field of modifiers — is
 * not here and is not wanted.
 *
 * ## Why the standard fits this game unusually well
 *
 * NATO symbology already has a vocabulary for *not knowing*, and it lines up almost
 * exactly with the rules' patrol table:
 *
 * - A **frame with nothing in it** means something is there and you cannot say what. That
 *   is a plain sighting — intel 2, presence and location.
 * - The **interior icon** appears once you know the arm, which the table grants at 5.
 * - The **size marks** appear once you know roughly how big it is, which is 4.
 * - A **dashed frame** means a position that is suspected or reported rather than
 *   observed, which is precisely what a despatch carries and what a commander holds about
 *   every formation but the one they are standing next to.
 *
 * So the symbol degrades in exactly the way the intelligence does, and a reader who knows
 * the standard can tell how good a report is without reading a word.
 *
 * ## Two deliberate departures
 *
 * The frame is coloured by **faction**, not by the standard's affiliation blue/red. A
 * two-sided historical game has its own colours and a referee looking at ground truth has
 * no side to be affiliated to; shape carries the friend-or-foe distinction instead, which
 * is the part of the standard that survives being printed in one colour.
 *
 * Symbols are drawn at a **legible size rather than a scaled one**. They are annotations
 * on the map, not features of it, so they hold their size as the map zooms out and only
 * the column ribbon shrinks. A symbol that scaled honestly would be an unreadable smudge
 * at the zoom that shows a whole theatre, which is the zoom people actually use.
 */

import { ECHELON_MARKS, type Echelon, type UnitKind } from '@campaign/shared';

/** What the frame's shape says. */
export type Affiliation = 'friend' | 'hostile';

export interface SymbolSpec {
  readonly affiliation: Affiliation;
  /** The arm, if it is known. Null draws an empty frame, which is the standard's way. */
  readonly kind: UnitKind | null;
  /** How large, if it is known. */
  readonly echelon: Echelon | null;
  readonly color: string;
  /** A reported or suspected position rather than an observed one. */
  readonly dashed: boolean;
  readonly emphasised: boolean;
}

/**
 * Frame height in pixels for a given hex size: legible first, proportionate second.
 *
 * The floor is what keeps a symbol readable at the zoom that shows a whole theatre, which
 * is the zoom people actually use. The ceiling stops it becoming a billboard when zoomed
 * right in on a few hexes — past a point a bigger symbol says nothing more.
 */
export const symbolSize = (hexSize: number): number =>
  Math.max(15, Math.min(46, hexSize * 1.5));

/**
 * Draw one symbol centred on a point.
 *
 * The centre is the formation's head hex. A headquarters in the full standard hangs from
 * a staff whose foot marks the location, which would make an HQ anchor differently from
 * everything else; here the staff is drawn as a flag below the frame and the frame stays
 * centred, so every symbol on the map means its own hex.
 */
export function drawSymbol(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  size: number,
  spec: SymbolSpec,
): void {
  const { affiliation, kind, echelon, color, dashed, emphasised } = spec;

  const halfH = size / 2;
  const halfW = affiliation === 'friend' ? size * 0.72 : size * 0.66;

  ctx.save();
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';

  // A dark plate under the frame, so a symbol stays readable over pale farmland as well
  // as over water. Without it the interior icon disappears on light ground.
  ctx.beginPath();
  framePath(ctx, x, y, halfW, halfH, affiliation);
  ctx.fillStyle = 'rgba(12, 13, 18, 0.66)';
  ctx.fill();

  ctx.strokeStyle = color;
  // A floor of two pixels: a dashed frame at the smallest size is half gaps, and a
  // hairline that is half gaps is not a symbol, it is a smudge.
  ctx.lineWidth = Math.max(2, size * (emphasised ? 0.14 : 0.11));
  if (dashed) ctx.setLineDash([size * 0.26, size * 0.16]);
  ctx.beginPath();
  framePath(ctx, x, y, halfW, halfH, affiliation);
  ctx.stroke();
  ctx.setLineDash([]);

  if (emphasised) {
    ctx.beginPath();
    framePath(ctx, x, y, halfW * 1.28, halfH * 1.3, affiliation);
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = Math.max(1.2, size * 0.07);
    ctx.stroke();
  }

  if (kind !== null) drawBranch(ctx, x, y, halfW * 0.66, halfH * 0.62, kind, color, size);
  if (echelon !== null && echelon !== 'none') {
    // Clear of the frame by a visible gap. Sitting closer, the marks read as part of the
    // frame's top edge rather than as a size — which is the one thing they must not do.
    drawEchelon(ctx, x, y - halfH - size * 0.36, size, echelon, color);
  }

  ctx.restore();
}

/** Friendly formations are rectangles; hostile ones are diamonds. */
function framePath(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  halfW: number,
  halfH: number,
  affiliation: Affiliation,
): void {
  if (affiliation === 'friend') {
    ctx.rect(x - halfW, y - halfH, halfW * 2, halfH * 2);
    return;
  }
  ctx.moveTo(x, y - halfH * 1.25);
  ctx.lineTo(x + halfW * 1.2, y);
  ctx.lineTo(x, y + halfH * 1.25);
  ctx.lineTo(x - halfW * 1.2, y);
  ctx.closePath();
}

/**
 * The arm, drawn inside the frame.
 *
 * Infantry is the crossed belts, cavalry the single slash of reconnaissance, artillery a
 * filled round shot. A headquarters gets a staff and pennant below the frame rather than
 * an interior mark, which is how the standard distinguishes a command post from the
 * troops it commands.
 */
function drawBranch(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  halfW: number,
  halfH: number,
  kind: UnitKind,
  color: string,
  size: number,
): void {
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = Math.max(1.3, size * 0.085);
  ctx.beginPath();

  switch (kind) {
    case 'infantry':
    case 'garrison':
      ctx.moveTo(x - halfW, y - halfH);
      ctx.lineTo(x + halfW, y + halfH);
      ctx.moveTo(x + halfW, y - halfH);
      ctx.lineTo(x - halfW, y + halfH);
      ctx.stroke();
      if (kind === 'garrison') {
        // A bar beneath the cross: troops holding a place rather than manoeuvring.
        ctx.beginPath();
        ctx.moveTo(x - halfW, y + halfH * 1.15);
        ctx.lineTo(x + halfW, y + halfH * 1.15);
        ctx.stroke();
      }
      return;

    case 'cavalry':
      ctx.moveTo(x - halfW, y + halfH);
      ctx.lineTo(x + halfW, y - halfH);
      ctx.stroke();
      return;

    case 'artillery_reserve':
      ctx.arc(x, y, Math.max(1.8, halfH * 0.55), 0, Math.PI * 2);
      ctx.fill();
      return;

    case 'convoy':
      // Two bars, one above the other: the train rather than the troops.
      ctx.moveTo(x - halfW, y - halfH * 0.4);
      ctx.lineTo(x + halfW, y - halfH * 0.4);
      ctx.moveTo(x - halfW, y + halfH * 0.6);
      ctx.lineTo(x + halfW, y + halfH * 0.6);
      ctx.stroke();
      return;

    case 'hq': {
      // Staff and pennant, hanging below the frame from its lower-left corner.
      const footY = y + halfH * 2.4;
      ctx.moveTo(x - halfW * 1.45, y + halfH);
      ctx.lineTo(x - halfW * 1.45, footY);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(x - halfW * 1.45, y + halfH * 1.15);
      ctx.lineTo(x - halfW * 0.3, y + halfH * 1.5);
      ctx.lineTo(x - halfW * 1.45, y + halfH * 1.85);
      ctx.closePath();
      ctx.fill();
      return;
    }
  }
}

/**
 * Size marks above the frame: `XX` for a division, `X` for a brigade, `III` for a
 * regiment, and so on up to `XXX` for a corps.
 *
 * Drawn as strokes rather than text. Canvas text at fourteen pixels is a grey blur, and
 * these are three shapes rather than three letters.
 */
function drawEchelon(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  size: number,
  echelon: Echelon,
  color: string,
): void {
  const marks = ECHELON_MARKS[echelon];
  if (marks === '') return;

  const glyphs = [...marks];
  const w = size * 0.22;
  const gap = size * 0.1;
  const total = glyphs.length * w + (glyphs.length - 1) * gap;
  let left = x - total / 2;

  ctx.strokeStyle = color;
  ctx.lineWidth = Math.max(1.2, size * 0.075);
  ctx.beginPath();

  for (const glyph of glyphs) {
    const cx = left + w / 2;
    if (glyph === 'X') {
      ctx.moveTo(cx - w / 2, y - w / 2);
      ctx.lineTo(cx + w / 2, y + w / 2);
      ctx.moveTo(cx + w / 2, y - w / 2);
      ctx.lineTo(cx - w / 2, y + w / 2);
    } else {
      ctx.moveTo(cx, y - w / 2);
      ctx.lineTo(cx, y + w / 2);
    }
    left += w + gap;
  }
  ctx.stroke();
}
