/**
 * The hours of the day: whether the sun is up, when it rises and sets, and when a column's
 * head is on the road.
 *
 * Darkness is a rule here rather than a mood: it costs fatigue, and a column still on the
 * road after sunset pays for it at the tail. So the clock carries a sun or a moon, and
 * the referee who moves the season moves the sun everyone else is shown.
 */

import { useEffect, useState } from 'react';

import {
  isDark,
  standingOrdersProblems,
  type CampaignConfig,
  type StandingOrders,
} from '@campaign/shared';

import { timeOfDay } from '../board.js';
import { copy } from '../copy.js';

/** A sun by day and a moon by night, beside the clock. The words are in the title. */
export function DayNight({ cfg, hours }: { cfg: CampaignConfig; hours: number }) {
  const dark = isDark(cfg, hours);
  const label = dark
    ? copy.daylight.night(timeOfDay(cfg.sunriseHour))
    : copy.daylight.day(timeOfDay(cfg.sunsetHour));

  return (
    <span className={`day-night ${dark ? 'night' : 'day'}`} title={label} role="img" aria-label={label}>
      <svg width="15" height="15" viewBox="0 0 24 24" aria-hidden="true">
        {dark ? (
          <path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z" />
        ) : (
          <>
            <circle cx="12" cy="12" r="4.5" />
            {[0, 45, 90, 135, 180, 225, 270, 315].map((a) => (
              <line
                key={a}
                x1="12"
                y1="2.5"
                x2="12"
                y2="5"
                transform={`rotate(${a} 12 12)`}
              />
            ))}
          </>
        )}
      </svg>
    </span>
  );
}

/** Hours of the day as a list of choices, with `—` for no limit where one is allowed. */
function HourSelect({
  value,
  from,
  to,
  allowNone,
  disabled,
  onChange,
}: {
  value: number | null;
  from: number;
  to: number;
  allowNone: boolean;
  disabled: boolean;
  onChange: (hour: number | null) => void;
}) {
  const hours = Array.from({ length: to - from + 1 }, (_, i) => from + i);
  return (
    <select
      value={value === null ? '' : String(value)}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value === '' ? null : Number(e.target.value))}
    >
      {allowNone && <option value="">{copy.standing.any}</option>}
      {hours.map((h) => (
        <option key={h} value={h}>
          {/* 24:00 rather than 00:00: "off the road by midnight" is the end of this day. */}
          {h === 24 ? '24:00' : timeOfDay(h)}
        </option>
      ))}
    </select>
  );
}

/**
 * The referee's sunrise and sunset.
 *
 * Whole hours: the clock has nothing finer, and a sunset at 19:48 would be charged exactly
 * as one at 20:00 by every hour it touches.
 */
export function DaylightControl({
  cfg,
  onSet,
}: {
  cfg: CampaignConfig;
  onSet: (sunriseHour: number, sunsetHour: number) => Promise<string | null>;
}) {
  const [sunrise, setSunrise] = useState(Math.round(cfg.sunriseHour));
  const [sunset, setSunset] = useState(Math.round(cfg.sunsetHour));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Follow the campaign when somebody else moves the sun, or the view catches up.
  useEffect(() => {
    setSunrise(Math.round(cfg.sunriseHour));
    setSunset(Math.round(cfg.sunsetHour));
  }, [cfg.sunriseHour, cfg.sunsetHour]);

  const unchanged = sunrise === cfg.sunriseHour && sunset === cfg.sunsetHour;

  return (
    <div className="hours-form">
      <p className="muted small">{copy.daylight.blurb}</p>
      <div className="hours-grid">
        <label>
          <span>{copy.daylight.sunrise}</span>
          <HourSelect
            value={sunrise}
            from={0}
            to={23}
            allowNone={false}
            disabled={busy}
            onChange={(h) => setSunrise(h ?? 0)}
          />
        </label>
        <label>
          <span>{copy.daylight.sunset}</span>
          <HourSelect
            value={sunset}
            from={1}
            to={24}
            allowNone={false}
            disabled={busy}
            onChange={(h) => setSunset(h ?? 24)}
          />
        </label>
      </div>
      <div className="despatch-actions">
        <button
          className="primary"
          disabled={busy || unchanged || sunrise >= sunset}
          onClick={() => {
            setBusy(true);
            setError(null);
            void onSet(sunrise, sunset)
              .then(setError)
              .finally(() => setBusy(false));
          }}
        >
          {copy.daylight.set}
        </button>
      </div>
      {error !== null && <p className="bad small">{error}</p>}
    </div>
  );
}

/** Standing orders as one line: what a reader needs before deciding to change them. */
export function standingSummary(orders: StandingOrders | undefined): string {
  if (orders === undefined) return copy.standing.none;
  const parts: string[] = [];
  if (orders.startHour !== null) parts.push(copy.standing.start(timeOfDay(orders.startHour)));
  if (orders.latestHour !== null) {
    parts.push(copy.standing.latest(orders.latestHour === 24 ? '24:00' : timeOfDay(orders.latestHour)));
  }
  if (orders.maxHoursOnRoad !== null) parts.push(copy.standing.max(orders.maxHoursOnRoad));
  return parts.length === 0 ? copy.standing.none : copy.standing.summary(parts);
}

const NONE: StandingOrders = { startHour: null, latestHour: null, maxHoursOnRoad: null };

/**
 * When the head of a column steps off, when it must be off the road, and how long it may
 * spend on it.
 *
 * The same form for a commander and a referee. What differs is whose formation it lands on,
 * and the server decides that rather than the form.
 */
export function StandingOrdersPanel({
  orders,
  cfg,
  forReferee,
  onSave,
}: {
  orders: StandingOrders | undefined;
  cfg: CampaignConfig;
  forReferee: boolean;
  onSave: (orders: StandingOrders | null) => Promise<string | null>;
}) {
  const [draft, setDraft] = useState<StandingOrders>(orders ?? NONE);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const held = JSON.stringify(orders ?? NONE);
  useEffect(() => {
    setDraft(orders ?? NONE);
    // Keyed on the content rather than the object: a fresh view carries equal orders in a
    // new object every time the clock moves, and resetting on that would eat a draft.
  }, [held]);

  const problems = standingOrdersProblems(draft, cfg);
  const unchanged = JSON.stringify(draft) === held;
  const empty =
    draft.startHour === null && draft.latestHour === null && draft.maxHoursOnRoad === null;

  const save = (next: StandingOrders | null): void => {
    setBusy(true);
    setError(null);
    void onSave(next)
      .then(setError)
      .finally(() => setBusy(false));
  };

  return (
    <section className="panel-section">
      <h3>{copy.standing.heading}</h3>
      <p className="muted small">{standingSummary(orders)}</p>

      <div className="hours-form">
        <p className="muted small">
          {forReferee ? copy.standing.blurbReferee : copy.standing.blurbOwn}
        </p>
        <div className="hours-grid">
          <label>
            <span>{copy.standing.startHour}</span>
            <HourSelect
              value={draft.startHour}
              from={0}
              to={23}
              allowNone
              disabled={busy}
              onChange={(h) => setDraft({ ...draft, startHour: h })}
            />
          </label>
          <label>
            <span>{copy.standing.latestHour}</span>
            <HourSelect
              value={draft.latestHour}
              from={1}
              to={24}
              allowNone
              disabled={busy}
              onChange={(h) => setDraft({ ...draft, latestHour: h })}
            />
          </label>
          <label>
            <span>{copy.standing.maxHoursOnRoad}</span>
            <input
              type="number"
              min={1}
              max={cfg.maxMarchHoursPerDay}
              step={1}
              placeholder={copy.standing.any}
              value={draft.maxHoursOnRoad ?? ''}
              disabled={busy}
              onChange={(e) =>
                setDraft({
                  ...draft,
                  maxHoursOnRoad: e.target.value === '' ? null : Number(e.target.value),
                })
              }
            />
          </label>
        </div>

        {problems.length > 0 && (
          <ul className="muted small orbat-problems">
            {problems.map((p) => (
              <li key={p}>{p}</li>
            ))}
          </ul>
        )}

        <div className="despatch-actions">
          <button
            className="primary"
            disabled={busy || unchanged || problems.length > 0}
            // All three blank is the same order as lifting them, and is sent as that.
            onClick={() => save(empty ? null : draft)}
          >
            {copy.standing.save}
          </button>
          {orders !== undefined && (
            <button disabled={busy} onClick={() => save(null)}>
              {copy.standing.lift}
            </button>
          )}
        </div>
        {error !== null && <p className="bad small">{error}</p>}
      </div>
    </section>
  );
}
