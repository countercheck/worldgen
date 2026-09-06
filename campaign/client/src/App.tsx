/**
 * The console.
 *
 * A referee's view of the whole campaign, with a role switch that shows what any one
 * commander can see. Hovering the map reports the ground and, over a column, the unit
 * standing on it; clicking selects a unit so its reach and its crossing costs can be read
 * against the hex under the cursor.
 *
 * **Fog here is drawn, not enforced.** The role switch filters what this page renders,
 * and the whole world is in the browser. That is right for a referee's own machine and
 * wrong for a commander's: a player given this page could read the entire map out of
 * memory. The masking that a player will actually receive happens on the server, which is
 * not built yet — until it is, this is a referee tool and says so on screen.
 */

import { useMemo, useState } from 'react';

import {
  DEFAULT_CONFIG,
  DEFAULT_THEME,
  factionVisible,
  key,
  maskWorld,
  occupied,
  parseWorld,
  reachable,
  reconZone,
  spotted,
  type Hex,
  type HexKey,
  type Unit,
} from '@campaign/shared';

import { HexMap, type DrawableUnit } from './map/HexMap.js';
import { HexPanel } from './panels/HexPanel.js';
import { UnitPanel } from './panels/UnitPanel.js';
import { buildDemo } from './scenario.js';

const demo = buildDemo();
const cfg = DEFAULT_CONFIG;

type Role = 'referee' | 'red' | 'blue';

export default function App() {
  const [role, setRole] = useState<Role>('referee');
  const [hovered, setHovered] = useState<Hex | null>(null);
  const [hoveredUnit, setHoveredUnit] = useState<Unit | null>(null);
  const [selected, setSelected] = useState<Unit | null>(null);
  const [showReach, setShowReach] = useState(false);

  const state = demo.state;

  /** What this role can see. The referee sees everything. */
  const seen: ReadonlySet<HexKey> | null = useMemo(() => {
    if (role === 'referee') return null;
    // Everywhere any of this faction's units has been, plus what they see now — the
    // demo has no event history, so memory is approximated by the marched columns.
    const memory = new Set<HexKey>(factionVisible(state, demo.world, cfg, role));
    for (const unit of state.units.values()) {
      if (unit.faction !== role) continue;
      for (const k of reconZone(demo.world, cfg, unit)) memory.add(k);
      for (const c of unit.column) memory.add(key(c));
    }
    return memory;
  }, [role, state]);

  const visible: ReadonlySet<HexKey> = useMemo(
    () => (role === 'referee' ? new Set() : factionVisible(state, demo.world, cfg, role)),
    [role, state],
  );

  /** The world as this role knows it. */
  const world = useMemo(() => {
    if (seen === null) return demo.world;
    return parseWorld(
      maskWorld(demo.worldDoc as Record<string, unknown>, {
        seen,
        visible,
        faction: role,
        clockHours: state.clockHours,
      }),
    );
  }, [seen, visible, role, state.clockHours]);

  /** Own units in full; enemies only where they have actually been spotted. */
  const units: DrawableUnit[] = useMemo(() => {
    const colour = (f: string): string =>
      state.factions.get(f)?.color ?? DEFAULT_THEME.fallback;

    if (role === 'referee') {
      return [...state.units.values()].map((unit) => ({
        unit,
        color: colour(unit.faction),
        visible: true,
      }));
    }

    const contacts = spotted(state, demo.world, cfg, role);
    return [...state.units.values()]
      .filter((u) => u.faction === role || contacts.has(u.id))
      .map((unit) => ({
        unit:
          unit.faction === role
            ? unit
            : // An enemy is drawn only where it was seen, not along its whole column:
              // knowing where a formation was is not knowing how it is strung out.
              { ...unit, column: [contacts.get(unit.id)!.coord] },
        color: colour(unit.faction),
        visible: unit.faction === role,
      }));
  }, [role, state]);

  const reach = useMemo(() => {
    if (!showReach || selected === null) return undefined;
    const live = state.units.get(selected.id);
    if (live === undefined) return undefined;
    return reachable(world, cfg, live, 10).hours;
  }, [showReach, selected, world, state]);

  const hoveredHex = hovered === null ? undefined : world.hexes.get(key(hovered));
  const shown = hoveredUnit ?? selected;
  const shownIsOwn = shown === null || role === 'referee' || shown.faction === role;

  return (
    <div className="app">
      <header>
        <h1>Campaign</h1>
        <div className="roles">
          {(['referee', 'red', 'blue'] as const).map((r) => (
            <button
              key={r}
              className={role === r ? 'active' : ''}
              onClick={() => setRole(r)}
              style={
                r === 'referee'
                  ? undefined
                  : { borderColor: state.factions.get(r)?.color ?? undefined }
              }
            >
              {r === 'referee' ? 'Referee' : (state.factions.get(r)?.name ?? r)}
            </button>
          ))}
        </div>
        <label className="toggle">
          <input
            type="checkbox"
            checked={showReach}
            onChange={(e) => setShowReach(e.target.checked)}
            disabled={selected === null}
          />
          Reach of selected, 10 h
        </label>
        <div className="clock">Hour {state.clockHours}</div>
      </header>

      {role !== 'referee' && (
        <div className="warning">
          Fog is drawn here, not enforced — the whole world is in this page. A commander’s
          map will be masked on the server before it is sent.
        </div>
      )}

      <div className="body">
        <HexMap
          world={world}
          units={units}
          theme={DEFAULT_THEME}
          hovered={hovered}
          onHover={(hex, unit) => {
            setHovered(hex);
            setHoveredUnit(unit);
          }}
          selectedId={selected?.id ?? null}
          onSelect={setSelected}
          reach={reach}
        />

        <aside className="sidebar">
          {shown !== null && shownIsOwn && (
            <UnitPanel
              unit={shown}
              name={demo.names[shown.id] ?? shown.id}
              factionName={state.factions.get(shown.faction)?.name ?? shown.faction}
              color={state.factions.get(shown.faction)?.color ?? '#888'}
            />
          )}

          {shown !== null && !shownIsOwn && (
            <section className="panel-section">
              <h3>Enemy contact</h3>
              <p className="muted">
                Last seen here. Strength, arm and formation are not known from a sighting
                alone — a patrol has to close before any of that is reported.
              </p>
            </section>
          )}

          {hovered !== null && hoveredHex !== undefined && (
            <HexPanel world={world} hex={hoveredHex} coord={hovered} selected={selected} />
          )}

          {hovered === null && shown === null && (
            <section className="panel-section">
              <h3>Nothing under the cursor</h3>
              <p className="muted">
                Move over the map to read the ground, or over a column to read the unit
                standing on it. Click a unit to keep it in view.
              </p>
              <h3>Units</h3>
              <ul className="unit-list">
                {[...state.units.values()]
                  .filter((u) => role === 'referee' || u.faction === role)
                  .map((u) => (
                    <li key={u.id}>
                      <button onClick={() => setSelected(u)}>
                        <span
                          className="swatch small"
                          style={{ background: state.factions.get(u.faction)?.color }}
                        />
                        {demo.names[u.id] ?? u.id}
                        <span className="muted">
                          {' '}
                          · {occupied(u).length} {occupied(u).length === 1 ? 'hex' : 'hexes'}
                        </span>
                      </button>
                    </li>
                  ))}
              </ul>
            </section>
          )}
        </aside>
      </div>
    </div>
  );
}
