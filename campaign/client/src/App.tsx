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
  reachable,
  type ClientView,
} from '@campaign/shared';

import {
  advanceClock,
  clearTask,
  fetchView,
  resolveDecision,
  sendDespatch,
  setTask,
  subscribe,
  type Session,
} from './api.js';
import { ageLabel, boardFrom } from './board.js';
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
import { Post } from './panels/Post.jsx';
import { DecisionQueue, DespatchLog } from './panels/Referee.jsx';
import { ReportPanel } from './panels/ReportPanel.jsx';
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

import type { Hex, PendingDecision, ReceivedDespatch, Task } from '@campaign/shared';

const cfg = DEFAULT_CONFIG;

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

  // The referee's half: which formation is having its destination pointed at, and what
  // the clock last stopped for. Both are transient and neither belongs in the view.
  const [ordering, setOrdering] = useState<string | null>(null);
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
      if (e.key === 'Escape') setOrdering(null);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [ordering]);

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

  const selectedUnit = board?.units.get(selectedId ?? '') ?? null;

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
      extra: { inReplyTo?: string; forwardedFrom?: string } = {},
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

  /** Point at ground and send a formation to it. The referee's whole job, in one motion. */
  const order = useCallback(
    (unitId: string, destination: Hex) => {
      setOrdering(null);
      setPostError(null);
      void setTask(session, unitId, destination).then((result) => {
        if (!result.ok) {
          setPostError(result.violations?.map((v) => v.message).join('; ') ?? 'refused');
        }
      });
    },
    [session],
  );

  if (error !== null) {
    return (
      <div className="join">
        <h1>Cannot read that campaign</h1>
        <p className="error">{error}</p>
        <button onClick={onLeave}>Use a different link</button>
      </div>
    );
  }

  if (view === null || board === null) {
    return (
      <div className="join">
        <h1>Campaign</h1>
        <p className="busy">Fetching your map…</p>
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

  const correspondents = correspondentsOf(view);
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
        ? `Halted at ${view.task.destination.q}, ${view.task.destination.r} — the march is done.`
        : `Marching on ${view.task.destination.q}, ${view.task.destination.r}, ordered at hour ${view.task.setAtHours}.`;

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
          { id: 'referee', label: 'Referee', token: joined.ownToken, color: undefined },
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

        <label className="toggle">
          <input
            type="checkbox"
            checked={showReach}
            onChange={(e) => setShowReach(e.target.checked)}
            disabled={selectedUnit === null}
          />
          Reach of selected, 10 h
        </label>

        {/* Only where there is something to wash. A referee is sent no visible set, so
            offering him the control would be offering him a switch that does nothing. */}
        {!isReferee && (
          <button
            className="wash-toggle"
            title="What is under observation right now (v)"
            onClick={() => {
              const next = nextWash(washMode);
              saveWash(next);
              setWashMode(next);
            }}
          >
            {washMode === 'three'
              ? 'Watched · marched · unknown'
              : washMode === 'two'
                ? 'Watched only'
                : 'No shading'}
          </button>
        )}

        <div className="clock">
          Hour {view.campaign.clockHours}
          {isReferee && (
            <span className="clock-controls">
              <button onClick={() => advance(1)}>+1 h</button>
              <button onClick={() => advance(6)}>+6 h</button>
              {/* The control a referee actually uses: run forward and stop the moment
                  something needs a human, rather than guessing at an interval and finding
                  out afterwards that two corps met each other ninety minutes in. */}
              <button
                className="primary"
                title="Advance until something needs a decision"
                onClick={() => advance(48, true)}
              >
                Run
              </button>
            </span>
          )}
        </div>

        <span className={`live live-${live}`} title={`Live updates ${live}`}>
          {live === 'open' ? 'live' : 'reconnecting'}
        </span>
      </header>

      {isReferee && halted !== null && (
        <div className="notice halted">
          The clock stopped at hour {halted.atHours}: {nameOf(halted.commanderId)}
          {"'s "}
          {board.units.get(halted.unitId)?.name ?? halted.unitId}{' '}
          {halted.trigger.replace(/_/g, ' ')}. It is in the queue below.
          <button className="dismiss" onClick={() => setHalted(null)}>
            Dismiss
          </button>
        </div>
      )}

      {isReferee && ordering !== null && (
        <div className="notice picking">
          Point at the ground {board.units.get(ordering)?.name ?? ordering} is to march to.
          A destination, not a route — they will find their own way, and discover what is in
          it when they get there. Escape to think again.
        </div>
      )}

      {!isReferee && view.commander !== null && (
        <div className="notice">
          You are {view.commander.name}, riding with{' '}
          {board.units.get(view.commander.unitId)?.name ?? view.commander.unitId}. You can
          see {board.visible.size} hexes from where you stand, and the enemy only within
          them. Every other formation below — your own corps included — is where it was
          when you last had word, which is not where it is now.
        </div>
      )}

      <div className="body">
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
          onPick={ordering === null ? undefined : (hex) => order(ordering, hex)}
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
                onResolve={(d) => {
                  setBusyId(d.id);
                  void resolveDecision(session, d.id).finally(() => setBusyId(null));
                }}
                busyId={busyId}
              />

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
                  estimateFor={(id) => estimateRide(view, board.world, cfg, id)}
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
                    { text: `Received your despatch of hour ${d.sentAtHours}.` },
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
              <h3>Orders</h3>
              <div className="despatch-actions">
                <button
                  className={ordering === shownUnit.id ? 'primary' : ''}
                  onClick={() => setOrdering(shownUnit.id)}
                >
                  {ordering === shownUnit.id ? 'Pointing…' : 'March them somewhere'}
                </button>
                {view.tasks.some((t) => t.unitId === shownUnit.id) && (
                  <button onClick={() => void clearTask(session, shownUnit.id)}>Halt</button>
                )}
              </div>
              <TaskLine task={view.tasks.find((t) => t.unitId === shownUnit.id)} />
            </section>
          )}

          {shownUnit !== null && (
            <UnitPanel
              unit={shownUnit}
              name={shownUnit.name}
              factionName={board.factions.get(shownUnit.faction)?.name ?? shownUnit.faction}
              color={board.factions.get(shownUnit.faction)?.color ?? '#888'}
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
            />
          )}

          {hovered === null && shownId === null && (
            <section className="panel-section">
              <h3>Nothing under the cursor</h3>
              <p className="muted">
                Move over the map to read the ground, or over a column to read the unit
                standing on it. Click a unit to keep it in view.
              </p>

              {/* A commander has his formations above, in the panel that also carries
                  their hours. Repeating them here would be the same list twice, once
                  without the thing that makes it mean anything. */}
              {isReferee && <h3>Formations</h3>}
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
                        · {occupied(u).length} {occupied(u).length === 1 ? 'hex' : 'hexes'}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>

              {board.contacts.size > 0 && (
                <>
                  <h3>Contacts</h3>
                  <ul className="unit-list">
                    {[...board.contacts.values()].map((c) => (
                      <li key={c.id}>
                        <button onClick={() => setSelectedId(c.id)}>
                          <span
                            className="swatch small ghost"
                            style={{ background: board.factions.get(c.faction)?.color }}
                          />
                          {c.corps ?? `Contact ${c.id}`}
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

              <h3>This campaign</h3>
              <p className="muted">
                Your link:{' '}
                <code className="link">{joinLink(session)}</code>
              </p>
              <button onClick={onLeave}>Leave</button>
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
function TaskLine({ task }: { task: Task | undefined }) {
  if (task === undefined) {
    return <p className="muted small">No standing task. They stay where they are.</p>;
  }
  if (task.complete) {
    return (
      <p className="muted small">
        Arrived at {task.destination.q}, {task.destination.r}. Nothing further ordered.
      </p>
    );
  }
  return (
    <p className="muted small">
      Marching on {task.destination.q}, {task.destination.r}, ordered at hour{' '}
      {task.setAtHours}
      {task.arrivesAtHours === null
        ? '.'
        : ` · next hex at hour ${task.arrivesAtHours.toFixed(1)}.`}
    </p>
  );
}
