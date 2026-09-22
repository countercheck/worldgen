/**
 * The campaign store: an append-only log, and state derived from it.
 *
 * Loading a campaign means folding its events. Snapshots exist only so that folding does
 * not get slower forever, and are always reconstructible — deleting the snapshot table is
 * a valid recovery, and the tests do exactly that to prove it.
 *
 * The store owns one rule the engine does not: after any command, every commander is
 * asked what they can now see for themselves, and anything newly learned is appended as its own
 * event. That is what makes knowledge persist rather than being recomputed — without it a
 * commander would forget a valley the moment their column marched out of the far side, and a
 * report of their own corps would always read "now".
 */

import { createHash, randomBytes } from 'node:crypto';

import {
  apply,
  DEFAULT_CONFIG,
  DEFAULT_RULESET,
  isRuleset,
  resolveConfig,
  type ConfigOverrides,
  EMPTY_STATE,
  knowledgeEvents,
  parseWorld,
  projectWorld,
  reduce,
  replay,
  type CampaignConfig,
  type CampaignState,
  type Commander,
  type Command,
  type Contact,
  type Despatch,
  type EventPayload,
  type LoggedEvent,
  type PendingDecision,
  type Task,
  type Unit,
  type UnitReport,
  commanderRole,
  REFEREE_ROLE,
  type Role,
  type Strictness,
  type Violation,
  type World,
} from '@campaign/shared';

import type { Db } from './db.js';

/**
 * Bring a stored payload up to the shape the engine now expects.
 *
 * The log is the campaign, and it is append-only: an event written last month cannot be
 * rewritten because a field was renamed this month. So the rename is honoured on the way
 * in instead — `effectives` became `paperStrength`, and a payload that predates that still
 * folds to the same state it always did.
 *
 * Renames only. Anything that changes what an event *means* is a new event kind, because
 * a reader of the log has to be able to trust that an old entry still says what it said.
 */
const RENAMED_FIELDS: readonly (readonly [string, string])[] = [
  ['effectives', 'paperStrength'],
  ['costEffectives', 'costPaperStrength'],
];

function upgrade<T>(value: T): T {
  if (Array.isArray(value)) return value.map((v) => upgrade(v)) as unknown as T;
  if (value === null || typeof value !== 'object') return value;

  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
    const renamed = RENAMED_FIELDS.find(([from]) => from === k)?.[1] ?? k;
    // A payload already carrying the new name wins: an upgrade must never undo itself.
    if (renamed !== k && renamed in (value as Record<string, unknown>)) continue;
    out[renamed] = upgrade(v);
  }
  return out as T;
}

/** How many events between snapshots. Small enough to matter, large enough not to churn. */
const SNAPSHOT_EVERY = 50;

export interface CampaignRow {
  readonly id: string;
  readonly name: string;
  readonly worldHash: string;
  readonly worldDoc: unknown;
  readonly world: World;
  readonly strictness: Strictness;
  /** Which named set of rules it was started under. */
  readonly ruleset: string;
  /**
   * The numbers this campaign actually runs on, resolved at creation and stored.
   *
   * Stored rather than re-resolved on every read, so that editing a ruleset tomorrow
   * cannot silently re-tune a game already in progress. The ruleset name beside it says
   * where the numbers came from; these are what they were.
   */
  readonly config: CampaignConfig;
}

export const sha256 = (s: string): string =>
  'sha256:' + createHash('sha256').update(s).digest('hex');

/** A join token, and the hash that is all the database ever holds. */
export const newToken = (): string => randomBytes(32).toString('base64url');
export const hashToken = (t: string): string => createHash('sha256').update(t).digest('hex');

/**
 * How many campaigns keep their parsed world in memory.
 *
 * `campaign()` is called on every authenticated request — `authorise` does it before any
 * route sees the caller — and without a cache each call reads the whole world blob out of
 * SQLite, `JSON.parse`s it, and rebuilds every hex. On a 200x200 map that is 56 ms and a
 * fresh allocation of the entire map, per request, per commander watching.
 *
 * Bounded because the alternative is a process that has served a thousand campaigns
 * holding a thousand worlds. Four is the active set of a referee's evening with room over;
 * raise it for a server running several games at once and a container with the memory for
 * them.
 */
const CACHED_CAMPAIGNS = 4;

export class CampaignStore {
  /**
   * Parsed campaigns, most recently used last.
   *
   * Safe to hold without invalidation because a campaign row is written once and never
   * altered: there is no `UPDATE campaigns` or `DELETE FROM campaigns` in this file, and
   * `cache.test.ts` asserts there is not. Everything that changes about a campaign is an
   * event, and events are not in this row.
   *
   * A `Map` rather than an LRU library: insertion order is iteration order in JavaScript,
   * so re-inserting on read makes the first key the least recently used one, which is the
   * whole of the eviction policy.
   */
  private readonly cache = new Map<string, CampaignRow>();

  constructor(
    private readonly db: Db,
    /**
     * A fallback for a campaign that has no stored numbers of its own.
     *
     * Every campaign created since rulesets landed carries its own resolved config, and
     * that is what runs. This is only what an older row falls back to.
     */
    private readonly cfg: CampaignConfig = DEFAULT_CONFIG,
  ) {}

  /**
   * Create a campaign from a world document, and mint the referee's link.
   *
   * Only the referee's. A join link names a commander, and there are no commanders yet —
   * they need formations to ride with, and those arrive afterwards. The referee sets up
   * the order of battle and then issues a link per seat with `issueToken`, which is also
   * the order a real game is prepared in.
   *
   * The world is stored with the campaign rather than referenced, so a campaign is
   * self-contained and a regenerated `world.json` on disk cannot silently invalidate
   * every coordinate in a saved campaign.
   */
  create(opts: {
    id: string;
    name: string;
    worldDoc: unknown;
    factions: readonly { id: string; name: string; color: string }[];
    seed?: number;
    strictness?: Strictness;
    /** A named set of rules. Unknown names fall back to the standard one. */
    ruleset?: string;
    /** This campaign's own amendments, on top of the ruleset. */
    config?: ConfigOverrides;
  }): { campaign: CampaignRow; refereeToken: string } {
    const blob = JSON.stringify(opts.worldDoc);
    const world = parseWorld(opts.worldDoc);
    const strictness = opts.strictness ?? 'strict';
    const ruleset = opts.ruleset !== undefined && isRuleset(opts.ruleset)
      ? opts.ruleset
      : DEFAULT_RULESET;
    // Built on the store's own numbers rather than on the rules as written, so a server
    // run with amended defaults hands them to the campaigns it creates.
    const config = resolveConfig(opts.config, ruleset, this.cfg);

    const hash = sha256(blob);

    // `OR IGNORE`, and the saving is the whole point: a referee running five games on one
    // map stores that map once. At 23 MB for a 200x200 world, a thousand campaigns over
    // fifty maps is 225 MB rather than 23 GB.
    //
    // Before the campaign row rather than after, so that a campaign never exists pointing
    // at a world that does not.
    this.db
      .prepare(`INSERT OR IGNORE INTO worlds (hash, blob) VALUES (?, ?)`)
      .run(hash, blob);

    this.db
      .prepare(
        `INSERT INTO campaigns
           (id, name, world_hash, strictness, ruleset, config_json, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        opts.id,
        opts.name,
        hash,
        strictness,
        ruleset,
        JSON.stringify(config),
        new Date().toISOString(),
      );

    const refereeToken = newToken();
    this.db
      .prepare(
        `INSERT INTO roles (campaign_id, token_hash, role_kind, commander_id) VALUES (?, ?, ?, ?)`,
      )
      .run(opts.id, hashToken(refereeToken), 'referee', null);

    const campaign: CampaignRow = {
      id: opts.id,
      name: opts.name,
      worldHash: hash,
      // Projected here as well as on read, so a campaign is the same object whether it
      // was just created or just loaded. The full document is already in `blob` and on
      // its way to disk; what is kept in hand is what a client can use.
      worldDoc: projectWorld(opts.worldDoc),
      world,
      strictness,
      ruleset,
      config,
    };

    const commands: Command[] = [
      {
        kind: 'create_campaign',
        name: opts.name,
        world: {
          seed: world.seed,
          width: world.width,
          height: world.height,
          layout: world.layout,
          schemaVersion: world.schemaVersion,
          hash,
        },
        seed: opts.seed ?? Math.floor(Math.random() * 2 ** 31),
      },
      ...opts.factions.map((f): Command => ({ kind: 'add_faction', faction: f })),
    ];

    for (const cmd of commands) {
      const result = this.execute(campaign, cmd, REFEREE_ROLE);
      if (!result.ok) {
        throw new Error(
          `could not create campaign: ${result.violations.map((v) => v.message).join('; ')}`,
        );
      }
    }

    // Warmed rather than left for the next request to load from disk: the referee is
    // about to look at the map they have just uploaded.
    return { campaign: this.remember(campaign), refereeToken };
  }

  /**
   * Mint a join link for one commander's seat.
   *
   * Returned once and never stored in the clear, so a lost link is reissued rather than
   * looked up. Calling this again for the same seat is legitimate and mints a second
   * working token — which is how a referee replaces a link somebody forwarded to the
   * wrong person, and why revocation is `revokeTokens` rather than an overwrite.
   */
  issueToken(campaignId: string, commanderId: string): string {
    const token = newToken();
    this.db
      .prepare(
        `INSERT INTO roles (campaign_id, token_hash, role_kind, commander_id) VALUES (?, ?, ?, ?)`,
      )
      .run(campaignId, hashToken(token), 'commander', commanderId);
    return token;
  }

  /** Invalidate every link to a seat. The other half of reissuing one. */
  revokeTokens(campaignId: string, commanderId: string): void {
    this.db
      .prepare(`DELETE FROM roles WHERE campaign_id = ? AND commander_id = ?`)
      .run(campaignId, commanderId);
  }

  /** Put a parsed campaign at the fresh end, evicting the stalest if the cache is full. */
  private remember(row: CampaignRow): CampaignRow {
    this.cache.delete(row.id);
    this.cache.set(row.id, row);
    // One at a time rather than a loop: exactly one has just gone in, so at most one is
    // now surplus. `keys().next()` is the oldest, insertion order being iteration order.
    if (this.cache.size > CACHED_CAMPAIGNS) {
      const stalest = this.cache.keys().next().value;
      if (stalest !== undefined) this.cache.delete(stalest);
    }
    return row;
  }

  campaign(id: string): CampaignRow | null {
    const cached = this.cache.get(id);
    // Re-inserted on a hit as well as a miss, so that reading a campaign is what keeps it
    // warm. Otherwise the four most recently *created* would be held and the one actually
    // being played would be evicted from under it.
    if (cached !== undefined) return this.remember(cached);

    const row = this.db
      .prepare(
        `SELECT c.id, c.name, c.world_hash, w.blob AS world_blob,
                c.strictness, c.ruleset, c.config_json
         FROM campaigns c
         JOIN worlds w ON w.hash = c.world_hash
         WHERE c.id = ?`,
      )
      .get(id) as
      | {
          id: string;
          name: string;
          world_hash: string;
          world_blob: string;
          strictness: string;
          ruleset: string | null;
          config_json: string | null;
        }
      | undefined;
    if (row === undefined) return null;

    // Projected on the way in, so what is held is the nine megabytes a client could
    // actually use rather than the twenty-three the generator wrote. `parseWorld` ignores
    // everything dropped — `world.projection.test.ts` proves the parsed worlds are equal —
    // and `viewFor` projects again on the way out, which is idempotent.
    //
    // The full document stays in SQLite, unread. Nothing needs it today, and a field that
    // moves from ignored to read tomorrow is then a cache that reloads rather than a
    // thousand campaigns that must be uploaded again.
    const worldDoc = projectWorld(JSON.parse(row.world_blob));
    const ruleset = row.ruleset ?? DEFAULT_RULESET;
    return this.remember({
      id: row.id,
      name: row.name,
      worldHash: row.world_hash,
      worldDoc,
      world: parseWorld(worldDoc),
      strictness: row.strictness as Strictness,
      ruleset,
      // A campaign written before rulesets existed has no stored numbers. Resolving them
      // now gives it the rules as written, which is what it has been playing under all
      // along — there was nothing else to play under.
      config:
        row.config_json === null
          ? resolveConfig(undefined, ruleset)
          : (JSON.parse(row.config_json) as CampaignConfig),
    });
  }

  /** Resolve a join token to what it may do. Unknown tokens resolve to nothing. */
  roleFor(campaignId: string, token: string | undefined): Role | null {
    if (token === undefined || token === '') return null;
    const row = this.db
      .prepare(`SELECT role_kind, commander_id FROM roles WHERE campaign_id = ? AND token_hash = ?`)
      .get(campaignId, hashToken(token)) as
      | { role_kind: string; commander_id: string | null }
      | undefined;
    if (row === undefined) return null;
    return row.role_kind === 'referee' ? REFEREE_ROLE : commanderRole(row.commander_id!);
  }

  events(campaignId: string, fromSeq = 0): LoggedEvent[] {
    const rows = this.db
      .prepare(
        `SELECT seq, clock_hours, payload_json, actor_json, forced, strictness, bypassed_json
         FROM events WHERE campaign_id = ? AND seq >= ? ORDER BY seq`,
      )
      .all(campaignId, fromSeq) as {
      seq: number;
      clock_hours: number;
      payload_json: string;
      actor_json: string;
      forced: number;
      strictness: string;
      bypassed_json: string;
    }[];

    return rows.map((r) => ({
      seq: r.seq,
      clockHours: r.clock_hours,
      actor: JSON.parse(r.actor_json),
      payload: upgrade(JSON.parse(r.payload_json)) as EventPayload,
      forced: r.forced === 1,
      strictness: r.strictness as Strictness,
      bypassed: JSON.parse(r.bypassed_json) as Violation[],
    }));
  }

  /**
   * The current state, from the newest snapshot plus the events after it.
   *
   * `upToSeq` replays only a prefix, which is how the referee's rewind works — there is
   * no undo, only a shorter log.
   */
  state(campaignId: string, upToSeq?: number): CampaignState {
    if (upToSeq !== undefined) {
      return replay(this.events(campaignId).filter((e) => e.seq < upToSeq));
    }

    const snap = this.db
      .prepare(
        `SELECT seq, state_json FROM snapshots WHERE campaign_id = ? ORDER BY seq DESC LIMIT 1`,
      )
      .get(campaignId) as { seq: number; state_json: string } | undefined;

    const base = snap === undefined ? EMPTY_STATE : deserialise(snap.state_json);
    const from = snap === undefined ? 0 : snap.seq;
    return replay(this.events(campaignId, from), base);
  }

  /**
   * Run a command, append what it produced, and record what everyone now sees.
   *
   * The observation pass is the reason this is not simply `apply`. It runs after the
   * command's own events so that a unit which has just moved observes from where it now
   * stands, not from where it was.
   */
  execute(
    campaign: CampaignRow,
    command: Command,
    role: Role,
    opts: { force?: boolean; strictness?: Strictness } = {},
  ): { ok: boolean; events: LoggedEvent[]; violations: readonly Violation[] } {
    const actor =
      role.kind === 'referee'
        ? ({ kind: 'referee' } as const)
        : ({ kind: 'commander', id: role.id } as const);

    // The campaign's own numbers, not the store's. Two campaigns in one database may be
    // played under different rules, which is the whole point of naming them.
    const cfg = campaign.config ?? this.cfg;

    let state = this.state(campaign.id);
    const outcome = apply(command, state, campaign.world, campaign.strictness, {
      actor,
      cfg,
      ...(opts.force !== undefined ? { force: opts.force } : {}),
      ...(opts.strictness !== undefined ? { strictness: opts.strictness } : {}),
    });

    if (!outcome.ok) return { ok: false, events: [], violations: outcome.violations };

    const all = [...outcome.events];
    state = outcome.state;

    for (const payload of knowledgeEvents(state, campaign.world, cfg)) {
      const event: LoggedEvent = {
        seq: state.nextSeq,
        clockHours: state.clockHours,
        actor: { kind: 'referee' },
        payload,
        forced: false,
        strictness: campaign.strictness,
        bypassed: [],
      };
      all.push(event);
      state = reduce(state, event);
    }

    this.append(campaign.id, all);
    this.maybeSnapshot(campaign.id, state);

    return { ok: true, events: all, violations: outcome.violations };
  }

  private append(campaignId: string, events: readonly LoggedEvent[]): void {
    const stmt = this.db.prepare(
      `INSERT INTO events
         (campaign_id, seq, clock_hours, kind, payload_json, actor_json,
          forced, strictness, bypassed_json, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    );
    const now = new Date().toISOString();
    // One transaction, so a command is all-or-nothing in the log. A half-written command
    // would fold into a state no sequence of commands could have produced.
    this.db.exec('BEGIN');
    try {
      for (const e of events) {
        stmt.run(
          campaignId,
          e.seq,
          e.clockHours,
          e.payload.kind,
          JSON.stringify(e.payload),
          JSON.stringify(e.actor),
          e.forced ? 1 : 0,
          e.strictness,
          JSON.stringify(e.bypassed),
          now,
        );
      }
      this.db.exec('COMMIT');
    } catch (err) {
      this.db.exec('ROLLBACK');
      throw err;
    }
  }

  /**
   * Write a snapshot once the log has grown `SNAPSHOT_EVERY` events past the last one.
   *
   * Measured as a distance from the previous snapshot rather than as `nextSeq % 50`. A
   * command emits as many events as it produced — an advance through a day of marching
   * emits dozens — so a modulus is only ever hit by coincidence, and in practice never
   * was: after setup no snapshot was written at all and every read replayed the whole log.
   */
  private maybeSnapshot(campaignId: string, state: CampaignState): void {
    const last = this.db
      .prepare(`SELECT MAX(seq) AS seq FROM snapshots WHERE campaign_id = ?`)
      .get(campaignId) as { seq: number | null } | undefined;

    const since = state.nextSeq - (last?.seq ?? 0);
    if (since < SNAPSHOT_EVERY) return;

    this.db
      .prepare(`INSERT OR REPLACE INTO snapshots (campaign_id, seq, state_json) VALUES (?, ?, ?)`)
      .run(campaignId, state.nextSeq, serialise(state));
  }

  /** Drop every snapshot. State must still load, by replaying from the beginning. */
  clearSnapshots(campaignId: string): void {
    this.db.prepare(`DELETE FROM snapshots WHERE campaign_id = ?`).run(campaignId);
  }
}

/**
 * State to JSON and back.
 *
 * `Map` and `Set` do not survive `JSON.stringify`, so they are written as arrays and
 * rebuilt on the way in. A snapshot that silently lost what a commander had surveyed
 * would be indistinguishable from a commander who forgot the campaign.
 */
export function serialise(state: CampaignState): string {
  return JSON.stringify({
    ...state,
    factions: [...state.factions.values()],
    commanders: [...state.commanders.values()],
    units: [...state.units.values()],
    knowledge: [...state.knowledge.values()].map((k) => ({
      commanderId: k.commanderId,
      surveyed: [...k.surveyed],
      lastSurveyedHours: [...k.lastSurveyedHours],
      reports: [...k.reports.values()],
      contacts: [...k.contacts.values()],
      nextContactNo: k.nextContactNo,
    })),
    despatches: [...state.despatches.values()],
    tasks: [...state.tasks.values()],
    decisions: [...state.decisions.values()],
    battle: [...state.battle],
  });
}

export function deserialise(json: string): CampaignState {
  const d = upgrade(JSON.parse(json)) as {
    name: string;
    world: CampaignState['world'];
    seed: number;
    clockHours: number;
    nextSeq: number;
    factions: CampaignState['factions'] extends ReadonlyMap<string, infer F> ? F[] : never;
    commanders: Commander[];
    // `formationChange` and `parentUnitId` postdate the first snapshots, like the task
    // and decision fields.
    units: (Omit<Unit, 'formationChange' | 'parentUnitId'> &
      Partial<Pick<Unit, 'formationChange' | 'parentUnitId'>>)[];
    knowledge: {
      commanderId: string;
      surveyed: string[];
      lastSurveyedHours: [string, number][];
      reports?: UnitReport[];
      contacts?: Contact[];
      nextContactNo?: number;
    }[];
    despatches?: Despatch[];
    // Waypoints postdate the first snapshots, so they are optional on the way in even
    // though `Task` requires them on the way out.
    tasks?: (Omit<Task, 'via' | 'viaIndex'> & Partial<Pick<Task, 'via' | 'viaIndex'>>)[];
    // `favouring` postdates the first snapshots, like the task fields above.
    decisions?: (Omit<PendingDecision, 'favouring'> &
      Partial<Pick<PendingDecision, 'favouring'>>)[];
    // Battlefields postdate the first snapshots. Absent means no fighting anywhere, which
    // is what a campaign written before they existed had.
    battle?: string[];
  };

  return {
    name: d.name,
    world: d.world,
    seed: d.seed,
    clockHours: d.clockHours,
    nextSeq: d.nextSeq,
    factions: new Map(d.factions.map((f) => [f.id, f])),
    // Tolerated as absent: a snapshot written before commanders existed still folds, and
    // the events after it will rebuild what it lacks.
    commanders: new Map((d.commanders ?? []).map((c) => [c.id, c])),
    units: new Map(
      d.units.map((u) => [
        u.id,
        { ...u, formationChange: u.formationChange ?? null, parentUnitId: u.parentUnitId ?? null },
      ]),
    ),
    knowledge: new Map(
      (d.knowledge ?? []).map((k) => [
        k.commanderId,
        {
          commanderId: k.commanderId,
          surveyed: new Set(k.surveyed),
          lastSurveyedHours: new Map(k.lastSurveyedHours),
          reports: new Map((k.reports ?? []).map((r) => [r.unitId, r])),
          contacts: new Map((k.contacts ?? []).map((c) => [c.id, c])),
          nextContactNo: k.nextContactNo ?? 1,
        },
      ]),
    ),
    // Tolerated as absent for the same reason as `commanders`: a snapshot written before
    // riders existed still folds, and the events after it rebuild what it lacks.
    despatches: new Map((d.despatches ?? []).map((x) => [x.id, x])),
    // `via`/`viaIndex` likewise: a snapshot written before waypoints existed has neither,
    // and a march with no waypoints behind it is exactly what such a task was.
    tasks: new Map(
      (d.tasks ?? []).map((t) => [t.unitId, { ...t, via: t.via ?? [], viaIndex: t.viaIndex ?? 0 }]),
    ),
    decisions: new Map(
      (d.decisions ?? []).map((k) => [k.id, { ...k, favouring: k.favouring ?? null }]),
    ),
    battle: new Set(d.battle ?? []),
  };
}
