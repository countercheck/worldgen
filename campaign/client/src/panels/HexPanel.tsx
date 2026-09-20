/**
 * What the ground under the cursor is.
 *
 * The point of the panel is to answer the questions a commander actually asks of a hex —
 * how fast can I cross it, can I get over that river, what is it — rather than to dump
 * every field the generator stored. So it shows derived answers next to the raw values
 * they come from: the movement grade beside the land cover that decided it, the river
 * class beside the catchment it was computed from.
 */

import {
  crossingFor,
  DEFAULT_CONFIG,
  type CampaignConfig,
  discharge,
  gradeOfHex,
  isWater,
  neighbors,
  riverClass,
  roadBetween,
  speedKmh,
  type Hex,
  type Unit,
  type World,
  type WorldHex,
} from '@campaign/shared';

import { Field, Row, Section } from './parts.js';

const TITLE: Record<string, string> = {
  highway: 'Highway',
  road: 'Road',
  off_road: 'Off-road',
  bad_going: 'Bad going',
};

const pretty = (s: string | null): string =>
  s === null ? '—' : s.replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase());

export function HexPanel({
  world,
  hex,
  coord,
  selected,
  cfg = DEFAULT_CONFIG,
}: {
  world: World;
  hex: WorldHex;
  coord: Hex;
  selected: Unit | null;
  /** The campaign's numbers, as the server resolved them. */
  cfg?: CampaignConfig;
}) {
  const fog = hex.tags.has('fog');

  if (fog) {
    return (
      <Section title={`Hex ${coord.q}, ${coord.r}`}>
        <p className="muted">
          Never observed. Nothing is known about this ground — what is stored here are
          defaults standing in for the unknown, not measurements.
        </p>
      </Section>
    );
  }

  const water = isWater(hex);
  const grade = gradeOfHex(hex, cfg);
  const river = riverClass(hex, world);

  // Roads leaving this hex, by tier — the reason a hex may be quick to cross even
  // though the ground is not.
  const roads = neighbors(coord)
    .map((n) => roadBetween(world, coord, n))
    .filter((e): e is NonNullable<typeof e> => e !== undefined);
  const tiers = [...new Set(roads.map((e) => e.tier))].sort();

  return (
    <>
      <Section title={`Hex ${coord.q}, ${coord.r}`}>
        <Row label="Terrain" value={pretty(hex.terrainClass)} />
        <Row label="Biome" value={pretty(hex.biome)} />
        <Row label="Cover" value={pretty(hex.landCover)} />
        {hex.settlementName !== null && <Row label="Settlement" value={hex.settlementName} />}
        {hex.tags.has('remembered') && (
          <Row label="Status" value="Remembered — not currently observed" />
        )}
      </Section>

      <Section title="Relief">
        <Row label="Elevation" value={`${Math.round(hex.elevation)} m`} />
        <Row label="Slope" value={`${Math.round(hex.slope)} m / km`} />
        <Row label="Relief" value={`${Math.round(hex.relief)} m`} />
      </Section>

      <Section title="Going">
        {water ? (
          <p className="muted">Water. Impassable to every unit in this ruleset.</p>
        ) : (
          <>
            <Row label="Grade" value={TITLE[grade] ?? grade} />
            {tiers.length > 0 && (
              <Row label="Roads" value={tiers.map(pretty).join(', ')} />
            )}
            <Field label="Hours to enter, by arm">
              <table className="grid">
                <thead>
                  <tr>
                    <th />
                    <th>Off road</th>
                    <th>On road</th>
                  </tr>
                </thead>
                <tbody>
                  {(['infantry', 'cavalry', 'courier'] as const).map((mover) => (
                    <tr key={mover}>
                      <td>{pretty(mover)}</td>
                      <td>{(1 / speedKmh(cfg, mover, grade)).toFixed(2)} h</td>
                      <td>{(1 / speedKmh(cfg, mover, 'road')).toFixed(2)} h</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Field>
          </>
        )}
      </Section>

      {river !== 'none' && (
        <Section title="Watercourse">
          <Row
            label="Class"
            value={river === 'major' ? 'Major — navigable' : 'Minor — fordable'}
          />
          <Row label="Catchment" value={`${Math.round(hex.catchmentKm2)} km²`} />
          <Row
            label="Discharge"
            value={`${Math.round(discharge(hex, world)).toLocaleString()} of ${world.config.navigableMinDischarge.toLocaleString()}`}
          />
          {hex.tags.has('bridge') && <Row label="Crossing" value="Bridge" />}
          {hex.tags.has('ford') && !hex.tags.has('bridge') && (
            <Row label="Crossing" value="Ford" />
          )}
          {selected !== null && <CrossingFor world={world} unit={selected} to={coord} cfg={cfg} />}
        </Section>
      )}

      {hex.tags.size > 0 && (
        <Section title="Tags">
          <p className="tags">
            {[...hex.tags].sort().map((t) => (
              <span className="tag" key={t}>
                {t}
              </span>
            ))}
          </p>
        </Section>
      )}
    </>
  );
}

/** What crossing here would cost the unit currently selected. */
function CrossingFor({
  world,
  unit,
  to,
  cfg,
}: {
  world: World;
  unit: Unit;
  to: Hex;
  cfg: CampaignConfig;
}) {
  // Approached from a neighbour that is not itself in the channel, which is what a unit
  // arriving at the bank would be doing.
  const from =
    neighbors(to).find((n) => {
      const h = world.hexes.get(`${n.q},${n.r}`);
      return h !== undefined && !h.tags.has('river');
    }) ?? to;

  const c = crossingFor(world, cfg, unit, from, to);

  return (
    <Field label={`For ${unit.id}`}>
      {Number.isFinite(c.hours) ? (
        <span>
          {c.how === 'none' ? 'No crossing needed' : `${c.how} — ${c.hours} h`}
        </span>
      ) : (
        <span className="bad">
          Cannot cross. {c.violations.map((v) => v.message).join(' ')}
        </span>
      )}
    </Field>
  );
}
