/**
 * The header's controls, for a screen too narrow to hold them.
 *
 * On a wide screen the seat switch, the shading and the reach toggle sit in the header
 * beside the clock. On a phone the header has room for the campaign's name and the hour and
 * little else, so they move here, behind one button, as a sheet over the map. They are
 * the same controls doing the same things; only where they sit changes.
 */

import { copy } from '../copy.js';

import type { WashMode } from '../map/draw.js';

export type Identity = { id: string; label: string; token: string; color: string | undefined };

const WASHES: readonly { mode: WashMode; label: string; blurb: string }[] = [
  { mode: 'three', label: copy.console.washThree, blurb: copy.more.washThreeBlurb },
  { mode: 'two', label: copy.console.washTwo, blurb: copy.more.washTwoBlurb },
  { mode: 'off', label: copy.console.washNone, blurb: copy.more.washNoneBlurb },
];

export function More({
  identities,
  activeToken,
  onSwitch,
  washMode,
  onWash,
  showReach,
  reachDisabled,
  onReach,
  onHome,
  onHelp,
  onClose,
}: {
  identities: readonly Identity[];
  activeToken: string;
  onSwitch: (token: string) => void;
  /** Null for a referee, who sees everything and so has nothing to shade. */
  washMode: WashMode | null;
  onWash: (mode: WashMode) => void;
  showReach: boolean;
  reachDisabled: boolean;
  onReach: (on: boolean) => void;
  onHome: () => void;
  onHelp: () => void;
  onClose: () => void;
}) {
  return (
    <div className="sheet-backdrop" onClick={onClose}>
      <section
        className="more-sheet"
        role="dialog"
        aria-label={copy.more.heading}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="more-head">
          <h2>{copy.more.heading}</h2>
          <button className="dismiss" onClick={onClose}>
            {copy.more.close}
          </button>
        </div>

        {identities.length > 0 && (
          <fieldset className="choices">
            <legend>{copy.more.seat}</legend>
            {identities.map((identity) => (
              <label key={identity.id} className="choice">
                <input
                  type="radio"
                  name="seat"
                  checked={identity.token === activeToken}
                  onChange={() => onSwitch(identity.token)}
                />
                <span
                  className="swatch small"
                  style={{ background: identity.color ?? 'transparent' }}
                />
                <span className="choice-label">{identity.label}</span>
              </label>
            ))}
            <p className="muted small">{copy.more.seatBlurb}</p>
          </fieldset>
        )}

        {washMode !== null && (
          <fieldset className="choices">
            <legend>{copy.more.shading}</legend>
            {WASHES.map((w) => (
              <label key={w.mode} className="choice">
                <input
                  type="radio"
                  name="wash"
                  checked={washMode === w.mode}
                  onChange={() => onWash(w.mode)}
                />
                <span className="choice-label">
                  {w.label}
                  <span className="muted small">{w.blurb}</span>
                </span>
              </label>
            ))}
          </fieldset>
        )}

        <label className="choice switch">
          <input
            type="checkbox"
            checked={showReach}
            disabled={reachDisabled}
            onChange={(e) => onReach(e.target.checked)}
          />
          <span className="choice-label">
            {copy.console.reachToggle}
            {reachDisabled && <span className="muted small">{copy.more.reachNeedsSelection}</span>}
          </span>
        </label>

        <button className="more-home" onClick={onHome}>
          {copy.console.home}
          <span className="muted small">{copy.console.homeHint}</span>
        </button>

        <button className="more-home" onClick={onHelp}>
          {copy.help.open}
          <span className="muted small">{copy.help.heading}</span>
        </button>
      </section>
    </div>
  );
}
