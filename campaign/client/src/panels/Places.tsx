/**
 * Every settlement on the map, in a drawer, to find one by name.
 *
 * A despatch names a place, and a reader who has not memorised the map has to hunt for it.
 * This is the gazetteer: pick a name and the map flashes the place, bringing it into view
 * first if it is off the screen (see `HexMap`'s `focus`).
 *
 * On a wide screen the drawer sits over the sidebar and the map stays visible beside it,
 * so a reader can go down the list pointing at one place after another. On a phone it
 * covers the map, so choosing a place closes it — the reader asked to see where it is.
 */

import { useState } from 'react';

import type { Settlement } from '@campaign/shared';

import { copy } from '../copy.js';
import { placeGroups } from '../places.js';

export function Places({
  settlements,
  onShow,
  onClose,
}: {
  settlements: readonly Settlement[];
  onShow: (settlement: Settlement) => void;
  onClose: () => void;
}) {
  const [filter, setFilter] = useState('');
  const groups = placeGroups(settlements, filter);

  return (
    <aside className="places" aria-label={copy.places.heading}>
      <div className="roster-head places-head">
        <h2>{copy.places.heading}</h2>
        <button className="dismiss" onClick={onClose}>
          {copy.places.close}
        </button>
      </div>
      <div className="places-filter">
        <input
          type="search"
          value={filter}
          placeholder={copy.places.filter}
          aria-label={copy.places.filter}
          onChange={(e) => setFilter(e.target.value)}
        />
      </div>
      <div className="roster-body">
        {settlements.length === 0 && <p className="muted small">{copy.places.none}</p>}
        {settlements.length > 0 && groups.length === 0 && (
          <p className="muted small">{copy.places.noMatch(filter.trim())}</p>
        )}
        {groups.map((group) => (
          <section key={group.tier} className="roster-group">
            <h3>{copy.places.tier(group.tier, group.places.length)}</h3>
            <ul className="places-list">
              {group.places.map((s) => (
                <li key={`${s.coord.q},${s.coord.r}`}>
                  <button
                    className="place"
                    data-hex={`${s.coord.q},${s.coord.r}`}
                    title={s.etymology === '' ? undefined : s.etymology}
                    aria-label={copy.places.show(s.name)}
                    onClick={() => onShow(s)}
                  >
                    <span className="place-name">{s.name}</span>
                    <span className="place-meta muted small">
                      {s.culture === '' ? '' : `${s.culture} · `}
                      {copy.places.population(s.population)}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          </section>
        ))}
      </div>
    </aside>
  );
}
