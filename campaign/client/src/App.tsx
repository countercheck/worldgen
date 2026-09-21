/**
 * The console.
 *
 * Everything on this page came out of one HTTP response. There is no world bundled in the
 * client, no state assembled locally and no second path with more in it: the browser is
 * handed a `ClientView` and draws it. That is the difference between fog as a display
 * convention and fog as a rule — a commander who opens the developer tools finds only
 * what their own troops have seen, because that is all that was ever sent.
 *
 * The role switch is the clearest case. It used to filter what this page rendered from a
 * world it already held, which made it a toggle on a lie. It now swaps the token the
 * client is using and refetches, so switching to a commander means genuinely asking the
 * server as that commander and receiving genuinely less. A referee can do it because they
 * were handed every side's link when they created the campaign; nobody else has the
 * tokens to try.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';

import {
  DEFAULT_CONFIG,
  DEFAULT_THEME,
  key,
  occupied,
  pathHours,
  planMarch,
  reachable,
  viaAhead,
  type ClientView,
} from '@campaign/shared';

import {
  addCommander,
  addUnit,
  advanceClock,
  clearTask,
  detachPatrol,
  fetchView,
  resolveDecision,
  sendDespatch,
  setFormation,
  setTask,
  subscribe,
  teleportUnit,
  declareBattle,
  endBattle,
  type Session,
} from './api.js';
import { ageLabel, boardFrom, dayHour, timeOfDay } from './board.js';
import { copy, hexes, prettify, triggerLabel } from './copy.js';
import {
  correspondents as correspondentsOf,
  estimateRide,
  forwardOf,
  inbox,
  isAcknowledged,
  outbox,
} from './despatch.js';
import { Join, type Joined } from './Join.jsx';
import { HexMap } from './map/HexMap.js';
import { Command } from './panels/Command.jsx';
import { Composer, type Draft } from './panels/Composer.jsx';
import { ContactPanel } from './panels/ContactPanel.jsx';
import { HexPanel } from './panels/HexPanel.js';
import { Orbat } from './panels/Orbat.jsx';
import { Post } from './panels/Post.jsx';
import { DecisionQueue, DespatchLog } from './panels/Referee.jsx';
import { ReportPanel } from './panels/ReportPanel.jsx';
import { Roster } from './panels/Roster.jsx';
import { UnitPanel } from './panels/UnitPanel.js';
import {
  clearSession,
  joinLink,
  loadSession,
  loadWash,
  nextWash,
  saveSession,
  saveWash,
} from './session.js';

import type { WashMode } from './map/draw.js';

import type {
  CampaignConfig,
  Formation,
  Hex,
  PendingDecision,
  ReceivedDespatch,
  Task,
  Unit,
} from '@campaign/shared';

/**
 * The numbers to compute with, when there is no view yet.
 *
 * Everything below reads `view.config` — the campaign's own table, sent by the server —
 * because a console working out reach and march rates off its own bundled defaults would
 * quietly disagree with the engine the moment a campaign ran under anything but the
 * standard rules. This is only what the first paint uses before the view lands.
 */
const FALLBACK_CONFIG = DEFAULT_CONFIG;

/**
 * The pointing mode is normally a unit id — "this formation is to march there".
 *
 * Raising one has no unit yet, so it borrows the same machinery under a name no unit can
 * have. One pointing mode rather than two: the map has exactly one way of asking for a
 * hex, and a second would be a second set of ways to get stuck in it.
 */
const ORDER_OF_BATTLE = '\u0000orbat';

/**
 * Pointing at ground to put a formation on it, rather than to march it there.
 *
 * A teleport breaks every movement rule at once, which is why it is the referee's alone
 * and why it is logged as what it is. Setting up a scenario and correcting a mistake are
 * the same act.
 */
const PLACE_PREFIX = '\u0000place:';

/**
 * Painting ground as being fought over.
 *
 * Unlike every other pointing mode this one does not end on the first hex. A battlefield
 * is several hexes more often than it is one, and making the referee re-enter the mode
 * for each would be a worse version of holding the button down. Pointing at ground
 * already in the fighting takes it back out, so the same gesture draws and erases.
 */
const DECLARE_BATTLE = '\u0000battle';

export default function App() {
  const [joined, setJoined] = useState<Joined | null>(() => {
    const stored = loadSession();
    return stored === null
      ? null
      : {
          session: stored.session,
          held: stored.held,
          ownToken: stored.ownToken,
          seats: stored.seats ?? {},
        };
  });

  if (joined === null) {
    return (
      <Join
        onJoined={(j) => {
          saveSession(j);
          setJoined(j);
        }}
      />
    );
  }

  return (
    <Console
      joined={joined}
      onSwitch={(session) => {
        const next = { ...joined, session };
        saveSession(next);
        setJoined(next);
      }}
      onLeave={() => {
        clearSession();
        setJoined(null);
      }}
    />
  );
}

function Console({
  joined,
  onSwitch,
  onLeave,
}: {
  joined: Joined;
  onSwitch: (session: Session) => void;
  onLeave: () => void;
}) {
  const { session } = joined;

  const [view, setView] = useState<ClientView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [live, setLive] = useState<'open' | 'closed' | 'error'>('closed');

  const [hovered, setHovered] = useState<{ q: number; r: number } | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [showReach, setShowReach] = useState(false);
  const [washMode, setWashMode] = useState<WashMode>(() => loadWash() ?? 'three');

  // The composer, and whichever despatch is currently waiting on the server. Kept here
  // rather than in the panels so that a view arriving over the socket mid-write does not
  // throw away what somebody was typing.
  const [writing, setWriting] = useState<Partial<Draft> | null>(null);
  const [sending, setSending] = useState(false);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [postError, setPostError] = useState<string | null>(null);

  // The referee's half: which formation is having its destination pointed at, the ground
  // he has pointed at so far, and what the clock last stopped for. All transient, and none
  // of it belongs in the view — an order half-composed is not a fact about the campaign.
  const [roster, setRoster] = useState(false);
  const [writingAs, setWritingAs] = useState<string | null>(null);
  // Where a formation being raised is to stand. Collected by the same pointing mode the
  // march orders use, because a referee setting up a scenario is looking at ground rather
  // than at coordinates.
  const [placing, setPlacing] = useState<Hex | null>(null);
  const [ordering, setOrdering] = useState<string | null>(null);
  const [picked, setPicked] = useState<readonly Hex[]>([]);
  const [halted, setHalted] = useState<PendingDecision | null>(null);

  // Fetched once so the page has something immediately, then kept current by the socket.
  // The socket sends a full view on connect too, so this is only about the gap: a first
  // paint that waits on a WebSocket handshake looks like a broken page.
  useEffect(() => {
    let cancelled = false;
    setView(null);
    fetchView(session)
      .then((v) => {
        if (!cancelled) setView(v);
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message);
      });

    const stop = subscribe(session, {
      onView: (v) => {
        if (!cancelled) setView(v);
      },
      onStatus: (s) => {
        if (!cancelled) setLive(s);
      },
    });

    return () => {
      cancelled = true;
      stop();
    };
  }, [session.campaignId, session.token]);

  // Escape leaves the pointing mode. A map that has silently changed what a click does,
  // with no way out but clicking somewhere, is a trap.
  useEffect(() => {
    if (ordering === null) return;
    const onKey = (e: KeyboardEvent): void => {
      if (e.key !== 'Escape') return;
      setOrdering(null);
      setPicked([]);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [ordering]);

  // Escape closes the drawer. Bound separately from the pointing handler, which is only
  // alive while a destination is being chosen — and pointing wins when both are open,
  // because the map has silently changed what a click does and that is the trap to leave.
  useEffect(() => {
    if (!roster) return;
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === 'Escape' && ordering === null) setRoster(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [roster, ordering]);

  // `v` cycles how much of the map is washed. Separate from the Escape handler above,
  // which is only bound while a destination is being pointed at.
  useEffect(() => {
    const onKey = (e: KeyboardEvent): void => {
      if (e.key !== 'v' || e.metaKey || e.ctrlKey || e.altKey) return;
      // A commander writing "v" in a despatch is writing, not asking for the map.
      const el = e.target as HTMLElement | null;
      const tag = el?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || el?.isContentEditable === true) return;
      setWashMode((m) => {
        const next = nextWash(m);
        saveWash(next);
        return next;
      });
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const board = useMemo(
    () => (view === null ? null : boardFrom(view, DEFAULT_THEME)),
    [view],
  );

  // The campaign's own numbers, as the server resolved them. Never the client's copy:
  // reach, march rates and what a patrol costs all have to agree with the engine.
  const cfg = view?.config ?? FALLBACK_CONFIG;

  const selectedUnit = board?.units.get(selectedId ?? '') ?? null;

  /**
   * Where every marching column is actually going, worked out rather than stored.
   *
   * A task names a destination and the engine re-routes each hex, so there is no path on
   * the wire to draw — and storing one would be drawing a plan that went stale the moment
   * the ground turned out not to be what the map said. Recomputing it here from the same
   * cost model the scheduler uses gives the referee the route his columns will actually
   * take, as of now.
   *
   * Referee only, and it needs no guard: a commander's view carries no `tasks` at all, so
   * this is empty for him. The map he would compute it on is masked anyway.
   */
  const plans = useMemo(() => {
    if (board === null || view === null) return [];
    const out: { unitId: string; color: string; route: readonly Hex[]; hours: number }[] = [];

    for (const task of view.tasks) {
      if (task.complete) continue;
      const unit = board.units.get(task.unitId);
      const head = unit?.column[0];
      if (unit === undefined || head === undefined) continue;

      const route = planMarch(board.world, cfg, unit, task.destination, head, viaAhead(task, head));
      if (route === null || route.length < 2) continue;

      out.push({
        unitId: task.unitId,
        color: board.factions.get(unit.faction)?.color ?? '#888',
        route,
        hours: pathHours(board.world, cfg, unit, route),
      });
    }
    return out;
  }, [board, view]);

  /**
   * The patrols a formation has in the field.
   *
   * Counted off the units the viewer was actually sent, which is the honest answer to the
   * question being asked: a referee sees every patrol, and a commander sees his own. A
   * count taken from anywhere else would be a number nobody can check against the map.
   */
  const patrolsOf = useCallback(
    (unitId: string) =>
      board === null ? [] : [...board.units.values()].filter((u) => u.parentUnitId === unitId),
    [board],
  );

  const planFor = useCallback(
    (unitId: string) => plans.find((p) => p.unitId === unitId),
    [plans],
  );

  const reach = useMemo(() => {
    if (!showReach || board === null || selectedUnit === null) return undefined;
    // Computed over the world this role was sent, which for a commander is their own
    // masked map. A plan is only as good as the country you know about.
    return reachable(board.world, cfg, selectedUnit, 10).hours;
  }, [showReach, selectedUnit, board]);

  const advance = useCallback(
    (hours: number, untilDecision = false) => {
      setHalted(null);
      advanceClock(session, hours, untilDecision)
        .then((result) => setHalted(result.halted ?? null))
        .catch((err: Error) => setError(err.message));
    },
    [session],
  );

  /**
   * Put a despatch on the road.
   *
   * A refusal comes back as violations rather than as a thrown error — "he does not
   * answer to you, so that is a message rather than an order" is the game working — so it
   * is shown in the form the commander is still looking at.
   */
  const write = useCallback(
    async (
      to: string,
      despatchKind: 'order' | 'report' | 'acknowledgement',
      body: { text?: string; contacts?: readonly unknown[] },
      extra: { inReplyTo?: string; forwardedFrom?: string; from?: string } = {},
    ): Promise<boolean> => {
      setPostError(null);
      const result = await sendDespatch(session, {
        to,
        despatchKind,
        body: body as { text?: string },
        ...extra,
      }).catch((err: Error) => {
        setPostError(err.message);
        return null;
      });

      if (result === null) return false;
      if (!result.ok) {
        setPostError(result.violations?.map((v) => v.message).join('; ') ?? 'refused');
        return false;
      }
      return true;
    },
    [session],
  );

  /**
   * Send a formation to the last place pointed at, by way of the rest.
   *
   * The chain is built by clicking and only sent when the referee says so, because one
   * click can no longer mean both "and then here" and "go". The common case is unchanged
   * in substance: point once, press March.
   */
  const order = useCallback(
    (unitId: string, route: readonly Hex[]) => {
      const destination = route.at(-1);
      if (destination === undefined) return;

      setOrdering(null);
      setPicked([]);
      setPostError(null);
      void setTask(session, unitId, destination, { via: route.slice(0, -1) }).then((result) => {
        if (!result.ok) {
          setPostError(
            result.violations?.map((v) => v.message).join('; ') ?? copy.orders.refused,
          );
        }
      });
    },
    [session],
  );

  if (error !== null) {
    return (
      <div className="join">
        <h1>{copy.console.unreadableTitle}</h1>
        <p className="error">{error}</p>
        <button onClick={onLeave}>{copy.console.useAnotherLink}</button>
      </div>
    );
  }

  if (view === null || board === null) {
    return (
      <div className="join">
        <h1>{copy.console.loadingTitle}</h1>
        <p className="busy">{copy.console.loading}</p>
      </div>
    );
  }

  const isReferee = view.role === 'referee';
  const clock = view.campaign.clockHours;
  const shownId = hoveredId ?? selectedId;
  // Three lookups, in the order of how much they claim to know. A mark is a unit, or a
  // dated report of one of ours, or a sighting of somebody else's — never more than one.
  const shownUnit = shownId === null ? null : (board.units.get(shownId) ?? null);
  const shownReport = shownId === null ? null : (board.reports.get(shownId) ?? null);
  const shownContact = shownId === null ? null : (board.contacts.get(shownId) ?? null);
  const hoveredHex = hovered === null ? undefined : board.world.hexes.get(key(hovered));

  // Whose name the referee is currently writing in. A commander has only his own and
  // never sees the control.
  const senders = isReferee
    ? [...view.commanders].sort((a, b) => (a.name < b.name ? -1 : 1))
    : [];
  const sender = writingAs ?? senders[0]?.id ?? null;
  const correspondents = correspondentsOf(view, sender ?? undefined);
  const nameOf = (id: string): string => board.commanders.get(id)?.name ?? id;
  const ownFormation =
    view.commander === null ? null : (board.units.get(view.commander.unitId) ?? null);

  /**
   * What the formation he rides with is doing.
   *
   * His own task and nobody else's. A referee's task list is the referee's; a commander
   * learns what his subordinates were told to do only from the copies of his own orders
   * — and from whether they turn up where he asked.
   */
  const taskLine =
    view.task === null
      ? null
      : view.task.complete
        ? copy.command.halted(view.task.destination.q, view.task.destination.r)
        : copy.command.marchingOn(
            view.task.destination.q,
            view.task.destination.r,
            dayHour(view.task.setAtHours),
          );

  /**
   * Every identity this browser actually holds a token for.
   *
   * Only a referee who created the campaign has the faction tokens, so only they get a
   * switcher — and it is a real one. A commander sent a single link has exactly one
   * identity and no way to ask for another, which is not an interface decision.
   */
  const identities: { id: string; label: string; token: string; color: string | undefined }[] =
    Object.keys(joined.held).length === 0
      ? []
      : [
          {
            id: 'referee',
            label: copy.console.refereeSeat,
            token: joined.ownToken,
            color: undefined,
          },
          ...Object.entries(joined.held).map(([commanderId, token]) => {
            // The man's name, not his id. A referee switching seats is choosing a person
            // to be, and "kellermann" is the engine's bookkeeping.
            const seat = joined.seats[commanderId];
            const faction = seat?.faction ?? board.commanders.get(commanderId)?.faction;
            return {
              id: commanderId,
              label: seat?.name ?? board.commanders.get(commanderId)?.name ?? commanderId,
              token,
              color: faction === undefined ? undefined : board.factions.get(faction)?.color,
            };
          }),
        ];

  return (
    <div className="app">
      <header>
        <h1>{view.campaign.name}</h1>

        {identities.length > 0 && (
          <div className="roles">
            {identities.map((identity) => (
              <button
                key={identity.id}
                className={session.token === identity.token ? 'active' : ''}
                onClick={() => onSwitch({ campaignId: session.campaignId, token: identity.token })}
                style={identity.color === undefined ? undefined : { borderColor: identity.color }}
              >
                {identity.label}
              </button>
            ))}
          </div>
        )}

        <button
          className={`wash-toggle${roster ? ' active' : ''}`}
          onClick={() => setRoster((r) => !r)}
          title={copy.console.orderOfBattleHint}
        >
          {copy.console.orderOfBattle(
            isReferee ? view.units.length : view.reports.length + view.units.length,
          )}
        </button>

        <label className="toggle">
          <input
            type="checkbox"
            checked={showReach}
            onChange={(e) => setShowReach(e.target.checked)}
            disabled={selectedUnit === null}
          />
          {copy.console.reachToggle}
        </label>

        {/* Only where there is something to wash. A referee is sent no visible set, so
            offering him the control would be offering him a switch that does nothing. */}
        {!isReferee && (
          <button
            className="wash-toggle"
            title={copy.console.washHint}
            onClick={() => {
              const next = nextWash(washMode);
              saveWash(next);
              setWashMode(next);
            }}
          >
            {washMode === 'three'
              ? copy.console.washThree
              : washMode === 'two'
                ? copy.console.washTwo
                : copy.console.washNone}
          </button>
        )}

        <div className="clock">
          {dayHour(view.campaign.clockHours)}
          {isReferee && (
            <span className="clock-controls">
              <button onClick={() => advance(1)}>{copy.console.advanceHour}</button>
              <button onClick={() => advance(6)}>{copy.console.advanceSix}</button>
              {/* The control a referee actually uses: run forward and stop the moment
                  something needs a human, rather than guessing at an interval and finding
                  out afterwards that two corps met each other ninety minutes in. */}
              <button
                className="primary"
                title={copy.console.runHint}
                onClick={() => advance(48, true)}
              >
                {copy.console.run}
              </button>
              {/* Beside the clock, because declaring a battle is a thing a referee does
                  the moment the clock stops for a contact. */}
              <button
                className={ordering === DECLARE_BATTLE ? 'primary' : undefined}
                title={copy.console.battleHint}
                onClick={() =>
                  setOrdering((o) => (o === DECLARE_BATTLE ? null : DECLARE_BATTLE))
                }
              >
                {ordering === DECLARE_BATTLE ? copy.console.battleDone : copy.console.battle}
                {board.battle.size > 0 && ordering !== DECLARE_BATTLE
                  ? ` · ${board.battle.size}`
                  : ''}
              </button>
            </span>
          )}
        </div>

        <span className={`live live-${live}`} title={copy.console.liveHint(live)}>
          {live === 'open' ? copy.console.liveOpen : copy.console.liveReconnecting}
        </span>
      </header>

      {isReferee && halted !== null && (
        <div className="notice halted">
          {copy.notices.clockStopped(
            dayHour(halted.atHours),
            halted.commanderId === null ? '' : `${nameOf(halted.commanderId)}'s `,
            board.units.get(halted.unitId)?.name ?? halted.unitId,
            triggerLabel(halted.trigger),
          )}
          <button className="dismiss" onClick={() => setHalted(null)}>
            {copy.console.dismiss}
          </button>
        </div>
      )}

      {isReferee && ordering === DECLARE_BATTLE && (
        <div className="notice picking">{copy.notices.pickBattle}</div>
      )}

      {isReferee && ordering === ORDER_OF_BATTLE && (
        <div className="notice picking">{copy.notices.pickForRaise}</div>
      )}

      {isReferee && ordering !== null && ordering.startsWith(PLACE_PREFIX) && (
        <div className="notice picking">
          {copy.notices.pickForPlace(
            board.units.get(ordering.slice(PLACE_PREFIX.length))?.name ?? 'it',
          )}
        </div>
      )}

      {isReferee &&
        ordering !== null &&
        ordering !== ORDER_OF_BATTLE &&
        ordering !== DECLARE_BATTLE &&
        !ordering.startsWith(PLACE_PREFIX) && (
        <div className="notice picking">
          {copy.notices.pickForMarch(board.units.get(ordering)?.name ?? ordering)}
        </div>
      )}

      {!isReferee && view.commander !== null && (
        <div className="notice">
          {copy.notices.whoYouAre(
            view.commander.name,
            board.units.get(view.commander.unitId)?.name ?? view.commander.unitId,
            board.visible.size,
          )}
        </div>
      )}

      <div className="body">
      <Roster
        open={roster}
        onClose={() => setRoster(false)}
        role={isReferee ? 'referee' : 'commander'}
        units={view.units}
        reports={isReferee ? [] : view.reports}
        ownUnitId={view.commander?.unitId ?? null}
        clockHours={clock}
        cfg={cfg}
        factionName={(id) => board.factions.get(id)?.name ?? id}
        colorOf={(f) => board.factions.get(f)?.color ?? '#888'}
        selectedId={selectedId}
        onSelect={setSelectedId}
        editor={
          isReferee ? (
            <Orbat
              factions={view.factions}
              units={view.units}
              commanders={view.commanders}
              cfg={cfg}
              placing={placing}
              busy={sending}
              error={postError}
              onPlace={() => {
                setRoster(false);
                setOrdering(ORDER_OF_BATTLE);
                setPicked([]);
              }}
              onRaise={(unit) => {
                setSending(true);
                setPostError(null);
                void addUnit(session, unit)
                  .then((result) => {
                    if (!result.ok) {
                      setPostError(
                        result.violations?.map((v) => v.message).join('; ') ??
                          copy.orders.refused,
                      );
                    } else {
                      setPlacing(null);
                    }
                  })
                  .finally(() => setSending(false));
              }}
              onAppoint={(commander) => {
                setSending(true);
                setPostError(null);
                void addCommander(session, commander)
                  .then((result) => {
                    if (!result.ok) {
                      setPostError(
                        result.violations?.map((v) => v.message).join('; ') ??
                          copy.orders.refused,
                      );
                    }
                  })
                  .finally(() => setSending(false));
              }}
            />
          ) : undefined
        }
        taskOf={(unitId) => {
          const task = view.tasks.find((t) => t.unitId === unitId);
          if (task === undefined) return view.task?.unitId === unitId ? taskLine : null;
          if (task.complete) {
            return copy.orders.taskArrived(task.destination.q, task.destination.r);
          }
          return copy.orders.taskMarching(task.destination.q, task.destination.r);
        }}
      />

        <HexMap
          world={board.world}
          marks={board.marks}
          theme={DEFAULT_THEME}
          hovered={hovered}
          onHover={(hex, id) => {
            setHovered(hex);
            setHoveredId(id);
          }}
          selectedId={selectedId}
          onSelect={setSelectedId}
          reach={reach}
          riders={board.riders}
          plans={plans}
          battle={board.battle}
          onPick={
            ordering === null
              ? undefined
              : (hex) => {
                  if (ordering.startsWith(PLACE_PREFIX)) {
                    const unitId = ordering.slice(PLACE_PREFIX.length);
                    setOrdering(null);
                    setPostError(null);
                    void teleportUnit(session, unitId, [hex]).then((result) => {
                      if (!result.ok) {
                        setPostError(
                          result.violations?.map((v) => v.message).join('; ') ??
                            copy.orders.refused,
                        );
                      }
                    });
                    return;
                  }
                  if (ordering === DECLARE_BATTLE) {
                    // Stays in the mode: a field is painted, not pointed at once.
                    const fighting = board.battle.has(key(hex));
                    setPostError(null);
                    void (fighting ? endBattle : declareBattle)(session, [hex]).then(
                      (result) => {
                        if (!result.ok) {
                          setPostError(
                            result.violations?.map((v) => v.message).join('; ') ??
                              copy.orders.refused,
                          );
                        }
                      },
                    );
                    return;
                  }
                  if (ordering === ORDER_OF_BATTLE) {
                    setPlacing(hex);
                    setOrdering(null);
                    setRoster(true);
                    return;
                  }
                  setPicked((r) =>
                    // Pointing twice at the same ground is a slip of the hand, not an
                    // instruction to go there and then go there again.
                    key(hex) === key(r.at(-1) ?? { q: NaN, r: NaN }) ? r : [...r, hex],
                  );
                }
          }
          route={picked}
          visible={board.visible}
          surveyed={board.surveyed}
          washMode={washMode}
        />

        <aside className="sidebar">
          {isReferee && (
            <>
              {postError !== null && (
                <section className="panel-section">
                  <p className="error">{postError}</p>
                </section>
              )}

              <DecisionQueue
                decisions={view.decisions}
                clockHours={clock}
                nameOf={nameOf}
                unitOf={(id) => board.units.get(id)}
                taskOf={(id) => view.tasks.find((t) => t.unitId === id)}
                orderingUnitId={ordering}
                onOrder={setOrdering}
                onResolve={(d, favouring) => {
                  setBusyId(d.id);
                  void resolveDecision(session, d.id, undefined, favouring).finally(() =>
                    setBusyId(null),
                  );
                }}
                busyId={busyId}
              />

              {writing === null ? (
                <section className="panel-section">
                  <h3>{copy.referee.despatchesHeading}</h3>
                  <div className="despatch-actions">
                    <button onClick={() => setWriting({})}>
                      {copy.referee.writeOnBehalf}
                    </button>
                  </div>
                  <p className="muted small">{copy.referee.writeOnBehalfBlurb}</p>
                </section>
              ) : (
                <Composer
                  correspondents={correspondents}
                  estimateFor={(id) =>
                    estimateRide(view, board.world, cfg, id, [], sender ?? undefined)
                  }
                  clockHours={clock}
                  busy={sending}
                  error={postError}
                  initial={writing}
                  senders={senders}
                  {...(sender === null ? {} : { from: sender })}
                  onFrom={setWritingAs}
                  onCancel={() => {
                    setWriting(null);
                    setPostError(null);
                  }}
                  onSend={(draft) => {
                    if (draft.from === undefined) return;
                    setSending(true);
                    void write(draft.to, draft.despatchKind, { text: draft.text }, {
                      from: draft.from,
                    })
                      .then((ok) => {
                        if (ok) setWriting(null);
                      })
                      .finally(() => setSending(false));
                  }}
                />
              )}

              <DespatchLog
                despatches={view.despatches}
                clockHours={clock}
                nameOf={nameOf}
              />
            </>
          )}

          {!isReferee && view.commander !== null && (
            <>
              {writing !== null && (
                <Composer
                  correspondents={correspondents}
                  clockHours={clock}
                  busy={sending}
                  error={postError}
                  initial={writing}
                  onCancel={() => {
                    setWriting(null);
                    setPostError(null);
                  }}
                  onSend={(draft) => {
                    setSending(true);
                    void write(draft.to, draft.despatchKind, { text: draft.text })
                      .then((ok) => {
                        if (ok) setWriting(null);
                      })
                      .finally(() => setSending(false));
                  }}
                />
              )}

              <Post
                received={inbox(view)}
                sent={outbox(view)}
                clockHours={clock}
                nameOf={nameOf}
                acknowledged={(id) => isAcknowledged(view, id)}
                busyId={busyId}
                onWrite={() => setWriting({})}
                onAcknowledge={(d: ReceivedDespatch) => {
                  setBusyId(d.id);
                  // An acknowledgement is itself a despatch, so it takes a rider and can
                  // itself be lost. That recursion is the whole of the feedback channel.
                  void write(
                    d.from,
                    'acknowledgement',
                    { text: copy.post.acknowledgementText(dayHour(d.sentAtHours)) },
                    { inReplyTo: d.id },
                  ).finally(() => setBusyId(null));
                }}
                onForward={(d: ReceivedDespatch) => {
                  // Opens the composer rather than sending: forwarding is a choice of
                  // addressee, and the man he wants is rarely the first in the list.
                  setWriting({ despatchKind: 'report', text: forwardOf(d).text ?? '' });
                }}
              />

              <Command
                own={ownFormation}
                reports={view.reports}
                clockHours={clock}
                colorOf={(f) => board.factions.get(f)?.color ?? '#888'}
                onSelect={setSelectedId}
                taskLine={taskLine}
              />
            </>
          )}

          {isReferee && shownUnit !== null && (
            <section className="panel-section">
              {/* Not the unit's name: the panel below already carries that, and a heading
                  repeated twice reads as two sections about different things. */}
              <h3>{copy.orders.heading}</h3>
              <div className="despatch-actions">
                <button
                  className={ordering === shownUnit.id ? 'primary' : ''}
                  onClick={() => {
                    setOrdering(shownUnit.id);
                    setPicked([]);
                  }}
                >
                  {ordering === shownUnit.id ? copy.orders.pointing : copy.orders.march}
                </button>
                {view.tasks.some((t) => t.unitId === shownUnit.id) && (
                  <button onClick={() => void clearTask(session, shownUnit.id)}>
                    {copy.orders.halt}
                  </button>
                )}
                {/* The rules give patrols to Scout. The button says what the next one
                    costs, because the fourth is not free and the cost is permanent. */}
                <button
                  onClick={() => {
                    setOrdering(PLACE_PREFIX + shownUnit.id);
                    setPicked([]);
                  }}
                  title={copy.orders.placeHint}
                >
                  {copy.orders.place}
                </button>
                {shownUnit.traits.includes('scout') && shownUnit.parentUnitId == null && (
                  <button
                    onClick={() => {
                      setPostError(null);
                      void detachPatrol(session, shownUnit.id).then((result) => {
                        if (!result.ok) {
                          setPostError(
                            result.violations?.map((v) => v.message).join('; ') ??
                              copy.orders.refused,
                          );
                        }
                      });
                    }}
                  >
                    {copy.orders.sendPatrol}
                    {patrolsOf(shownUnit.id).length >= cfg.freePatrols
                      ? copy.orders.patrolCost(cfg.extraPatrolCost)
                      : ''}
                  </button>
                )}
              </div>

              {ordering === shownUnit.id && (
                <div className="picked-route">
                  {picked.length === 0 ? (
                    <p className="muted">{copy.orders.nowhereNamed}</p>
                  ) : (
                    <ol>
                      {picked.map((hex, i) => (
                        <li key={key(hex)} className={i === picked.length - 1 ? 'destination' : ''}>
                          {key(hex)}
                          {i === picked.length - 1
                            ? copy.orders.destinationSuffix
                            : copy.orders.waypointSuffix}
                        </li>
                      ))}
                    </ol>
                  )}
                  <div className="despatch-actions">
                    <button
                      className="primary"
                      disabled={picked.length === 0}
                      onClick={() => order(shownUnit.id, picked)}
                    >
                      {copy.orders.confirmMarch}
                    </button>
                    <button
                      disabled={picked.length === 0}
                      onClick={() => setPicked((r) => r.slice(0, -1))}
                    >
                      {copy.orders.undoLast}
                    </button>
                    <button
                      onClick={() => {
                        setOrdering(null);
                        setPicked([]);
                      }}
                    >
                      {copy.orders.cancel}
                    </button>
                  </div>
                </div>
              )}

              <TaskLine
                task={view.tasks.find((t) => t.unitId === shownUnit.id)}
                plan={planFor(shownUnit.id)}
              />

              <FormationControl
                unit={shownUnit}
                clockHours={clock}
                cfg={cfg}
                onSet={(formation) => {
                  setPostError(null);
                  void setFormation(session, shownUnit.id, formation).then((result) => {
                    if (!result.ok) {
                      setPostError(
                        result.violations?.map((v) => v.message).join('; ') ??
                          copy.orders.refused,
                      );
                    }
                  });
                }}
              />
            </section>
          )}

          {shownUnit !== null && (
            <UnitPanel
              unit={shownUnit}
              name={shownUnit.name}
              factionName={board.factions.get(shownUnit.faction)?.name ?? shownUnit.faction}
              color={board.factions.get(shownUnit.faction)?.color ?? '#888'}
              cfg={cfg}
              patrolsOut={patrolsOf(shownUnit.id).length}
              {...(() => {
                const parent =
                  shownUnit.parentUnitId == null
                    ? undefined
                    : board.units.get(shownUnit.parentUnitId);
                return parent === undefined ? {} : { parent };
              })()}
            />
          )}

          {shownUnit === null && shownReport !== null && (
            <ReportPanel
              report={shownReport}
              faction={board.factions.get(shownReport.faction)}
              clockHours={clock}
            />
          )}

          {shownUnit === null && shownReport === null && shownContact !== null && (
            <ContactPanel
              contact={shownContact}
              factionName={
                board.factions.get(shownContact.faction)?.name ?? shownContact.faction
              }
              color={board.factions.get(shownContact.faction)?.color ?? '#888'}
              clockHours={clock}
            />
          )}

          {hovered !== null && hoveredHex !== undefined && (
            <HexPanel
              world={board.world}
              hex={hoveredHex}
              coord={hovered}
              selected={selectedUnit}
              cfg={cfg}
            />
          )}

          {hovered === null && shownId === null && (
            <section className="panel-section">
              <h3>{copy.idle.heading}</h3>
              <p className="muted">{copy.idle.blurb}</p>

              {/* A commander has his formations above, in the panel that also carries
                  their hours. Repeating them here would be the same list twice, once
                  without the thing that makes it mean anything. */}
              {isReferee && <h3>{copy.idle.formations}</h3>}
              <ul className="unit-list">
                {(isReferee ? [...board.units.values()] : []).map((u) => (
                  <li key={u.id}>
                    <button onClick={() => setSelectedId(u.id)}>
                      <span
                        className="swatch small"
                        style={{ background: board.factions.get(u.faction)?.color }}
                      />
                      {u.name}
                      <span className="muted">
                        {' '}
                        · {hexes(occupied(u, 'road', cfg.footprint).length)}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>

              {board.contacts.size > 0 && (
                <>
                  <h3>{copy.idle.contacts}</h3>
                  <ul className="unit-list">
                    {[...board.contacts.values()].map((c) => (
                      <li key={c.id}>
                        <button onClick={() => setSelectedId(c.id)}>
                          <span
                            className="swatch small ghost"
                            style={{ background: board.factions.get(c.faction)?.color }}
                          />
                          {c.corps ?? copy.idle.contactLabel(c.id)}
                          <span className="muted">
                            {' '}
                            · {ageLabel(c.seenAtHours, clock)}
                          </span>
                        </button>
                      </li>
                    ))}
                  </ul>
                </>
              )}

              <h3>{copy.idle.campaignHeading}</h3>
              <p className="muted">
                {copy.idle.yourLink} <code className="link">{joinLink(session)}</code>
              </p>
              <button onClick={onLeave}>{copy.idle.leave}</button>
            </section>
          )}
        </aside>
      </div>
    </div>
  );
}

/**
 * What a formation is doing, for the referee's eye.
 *
 * Its destination and the hour it was set, which together are the audit trail from a
 * piece of prose to a column on a road. A commander sees the same for his own formation
 * and for no other.
 */
/**
 * What a formation is, and what the referee can make it.
 *
 * The hours are on the buttons because they are the whole decision. Forming for battle
 * costs an hour and making camp costs two, and a referee choosing between them with the
 * numbers hidden is choosing blind — the cost *is* the rule.
 */
function FormationControl({
  unit,
  clockHours,
  cfg,
  onSet,
}: {
  unit: Unit;
  clockHours: number;
  cfg: CampaignConfig;
  onSet: (formation: Formation) => void;
}) {
  const change = unit.formationChange;
  const offered: Formation[] = ['march', 'battle', 'rest', 'occupation'];

  return (
    <div className="formation-control">
      {change != null && (
        <p className="muted small">
          {copy.formation.changing(
            prettify(unit.formation),
            prettify(change.to),
            timeOfDay(change.completesAtHours),
            (change.completesAtHours - clockHours).toFixed(1),
          )}
        </p>
      )}

      <div className="despatch-actions">
        {offered.map((to) => {
          const hours = cfg.formationChangeHours[unit.formation][to];
          const current = unit.formation === to && change == null;
          return (
            <button
              key={to}
              className={current ? 'primary' : ''}
              disabled={current}
              onClick={() => onSet(to)}
              title={
                current ? copy.formation.alreadyIn : copy.formation.changeHint(hours)
              }
            >
              {prettify(to)}
              {current ? '' : copy.formation.cost(hours)}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function TaskLine({
  task,
  plan,
}: {
  task: Task | undefined;
  plan: { route: readonly Hex[]; hours: number } | undefined;
}) {
  if (task === undefined) {
    return <p className="muted small">{copy.orders.noTask}</p>;
  }
  if (task.complete) {
    return (
      <p className="muted small">
        {copy.orders.arrived(task.destination.q, task.destination.r)}
      </p>
    );
  }

  const ahead = task.via.slice(task.viaIndex);

  return (
    <>
      <p className="muted small">
        {copy.orders.marchingOn(
          task.destination.q,
          task.destination.r,
          dayHour(task.setAtHours),
        )}
      </p>

      {/* The route as it stands, not as it was ordered: it is recomputed from where the
          column is now, so it answers "where will they be" rather than "what did I say". */}
      {plan !== undefined && (
        <p className="muted small">
          {copy.orders.routeAhead(plan.route.length - 1, plan.hours.toFixed(1))}
          {ahead.length > 0 &&
            copy.orders.byWayOf(ahead.map((h) => `${h.q}, ${h.r}`).join('; then '))}
        </p>
      )}
    </>
  );
}
