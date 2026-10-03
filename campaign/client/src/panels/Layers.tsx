/**
 * What the map shows: one reading of the ground, and the overlays on it.
 *
 * The same fieldsets on every screen. On a wide one they open from the header beside the
 * map; on a phone they sit in the More sheet with the other header controls.
 */

import { copy } from '../copy.js';
import { ALL_LAYERS, GROUNDS, OVERLAYS, type MapLayers } from '../map/draw.js';

/** Plain paper with only the networks on it — the view the button below asks for. */
export const NETWORKS_ONLY: MapLayers = {
  ground: 'plain',
  names: false,
  settlements: false,
  roads: true,
  rivers: true,
  crossings: true,
  ports: false,
};

export function LayerControls({
  layers,
  onChange,
}: {
  layers: MapLayers;
  onChange: (layers: MapLayers) => void;
}) {
  return (
    <div className="layer-controls">
      <fieldset className="choices">
        <legend>{copy.layers.ground}</legend>
        {GROUNDS.map((g) => (
          <label key={g} className="choice">
            <input
              type="radio"
              name="ground"
              checked={layers.ground === g}
              onChange={() => onChange({ ...layers, ground: g })}
            />
            <span className="choice-label">
              {copy.layers.grounds[g]}
              <span className="muted small">{copy.layers.groundBlurbs[g]}</span>
            </span>
          </label>
        ))}
      </fieldset>

      <fieldset className="choices">
        <legend>{copy.layers.overlays}</legend>
        {OVERLAYS.map((o) => (
          <label key={o} className="choice">
            <input
              type="checkbox"
              checked={layers[o]}
              onChange={(e) => onChange({ ...layers, [o]: e.target.checked })}
            />
            <span className="choice-label">{copy.layers.overlayNames[o]}</span>
          </label>
        ))}
      </fieldset>

      <div className="despatch-actions">
        <button onClick={() => onChange(NETWORKS_ONLY)}>{copy.layers.networks}</button>
        <button onClick={() => onChange(ALL_LAYERS)}>{copy.layers.everything}</button>
      </div>
      <p className="muted small">{copy.layers.blurb}</p>
    </div>
  );
}
