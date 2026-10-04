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
  riversBeside,
  roadBetween,
  speedKmh,
  type Hex,
  type Unit,
  type World,
  type WorldHex,
} from '@campaign/shared';

import { copy, prettify, prettyOrDash } from '../copy.js';

import { Field, Row, Section } from './parts.js';

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
      <Section title={copy.hex.title(coord.q, coord.r)}>
        <p className="muted">{copy.hex.neverObserved}</p>
      </Section>
    );
  }

  const water = isWater(hex);
  const grade = gradeOfHex(hex, cfg);
  // Rivers run along hexsides: what this hex has is the rivers beside it, largest first.
  const beside = riversBeside(world, coord);
  const main = beside[0];
  const river = riverClass(main?.side, world);

  // Roads leaving this hex, by tier — the reason a hex may be quick to cross even
  // though the ground is not.
  const roads = neighbors(coord)
    .map((n) => roadBetween(world, coord, n))
    .filter((e): e is NonNullable<typeof e> => e !== undefined);
  const tiers = [...new Set(roads.map((e) => e.tier))].sort();

  return (
    <>
      <Section title={copy.hex.title(coord.q, coord.r)}>
        <Row label={copy.hex.terrain} value={prettyOrDash(hex.terrainClass)} />
        <Row label={copy.hex.biome} value={prettyOrDash(hex.biome)} />
        <Row label={copy.hex.cover} value={prettyOrDash(hex.landCover)} />
        {hex.settlementName !== null && (
          <Row label={copy.hex.settlement} value={hex.settlementName} />
        )}
        {hex.tags.has('remembered') && (
          <Row label={copy.hex.status} value={copy.hex.remembered} />
        )}
      </Section>

      <Section title={copy.hex.reliefHeading}>
        <Row label={copy.hex.elevation} value={copy.hex.metres(Math.round(hex.elevation))} />
        <Row label={copy.hex.slope} value={copy.hex.metresPerKm(Math.round(hex.slope))} />
        <Row label={copy.hex.relief} value={copy.hex.metres(Math.round(hex.relief))} />
      </Section>

      <Section title={copy.hex.goingHeading}>
        {water ? (
          <p className="muted">{copy.hex.water}</p>
        ) : (
          <>
            <Row label={copy.hex.grade} value={prettify(grade)} />
            {tiers.length > 0 && (
              <Row label={copy.hex.roads} value={tiers.map(prettyOrDash).join(', ')} />
            )}
            <Field label={copy.hex.hoursToEnter}>
              <table className="grid">
                <thead>
                  <tr>
                    <th />
                    <th>{copy.hex.offRoadColumn}</th>
                    <th>{copy.hex.onRoadColumn}</th>
                  </tr>
                </thead>
                <tbody>
                  {(['infantry', 'cavalry', 'courier'] as const).map((mover) => (
                    <tr key={mover}>
                      <td>{prettify(mover)}</td>
                      <td>{copy.hex.hours((1 / speedKmh(cfg, mover, grade)).toFixed(2))}</td>
                      <td>{copy.hex.hours((1 / speedKmh(cfg, mover, 'road')).toFixed(2))}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Field>
          </>
        )}
      </Section>

      {main !== undefined && river !== 'none' && (
        <Section title={copy.hex.watercourseHeading}>
          <Row
            label={copy.hex.riverClass}
            value={river === 'major' ? copy.hex.riverMajor : copy.hex.riverMinor}
          />
          <Row
            label={copy.hex.catchment}
            value={copy.hex.catchmentValue(Math.round(main.side.catchmentKm2))}
          />
          <Row
            label={copy.hex.discharge}
            value={copy.hex.dischargeValue(
              Math.round(discharge(main.side, world)).toLocaleString(),
              world.config.navigableMinDischarge.toLocaleString(),
            )}
          />
          {beside.some((b) => b.side.tags.has('bridge')) && (
            <Row label={copy.hex.crossing} value={copy.hex.bridge} />
          )}
          {beside.some((b) => b.side.tags.has('ford')) &&
            !beside.some((b) => b.side.tags.has('bridge')) && (
              <Row label={copy.hex.crossing} value={copy.hex.ford} />
            )}
          {selected !== null && (
            <CrossingFor world={world} unit={selected} from={coord} to={main.across} cfg={cfg} />
          )}
        </Section>
      )}

      {hex.tags.size > 0 && (
        <Section title={copy.hex.tagsHeading}>
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

/** What crossing the largest river beside this hex would cost the unit selected. */
function CrossingFor({
  world,
  unit,
  from,
  to,
  cfg,
}: {
  world: World;
  unit: Unit;
  from: Hex;
  to: Hex;
  cfg: CampaignConfig;
}) {
  const c = crossingFor(world, cfg, unit, from, to);

  return (
    <Field label={copy.hex.crossingFor(unit.id)}>
      {Number.isFinite(c.hours) ? (
        <span>
          {c.how === 'none'
            ? copy.hex.noCrossingNeeded
            : copy.hex.crossingCost(c.how, c.hours)}
        </span>
      ) : (
        <span className="bad">
          {copy.hex.cannotCross(c.violations.map((v) => v.message).join(' '))}
        </span>
      )}
    </Field>
  );
}
