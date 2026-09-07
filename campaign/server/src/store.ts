/**
 * The campaign store: an append-only log, and state derived from it.
 *
 * Loading a campaign means folding its events. Snapshots exist only so that folding does
 * not get slower forever, and are always reconstructible — deleting the snapshot table is
 * a valid recovery, and the tests do exactly that to prove it.
 *
 * The store owns one rule the engine does not: after any command, each faction is asked
 * what it can now see, and anything newly observed is appended as its own event. That is
 * what makes knowledge persist. Without it a faction would forget a valley the moment it
 * marched out of the far side.
 */

import { createHash, randomBytes } from 'node:crypto';

import {
  apply,
  DEFAULT_CONFIG,
  EMPTY_STATE,
  observationEvents,
  parseWorld,
  reduce,
  replay,
  type CampaignConfig,
  type CampaignState,
  type Commander,
  type Command,
  type Despatch,
  type LoggedEvent,
  type PendingDecision,
  type Task,
  commanderRole,
  REFEREE_ROLE,
  type Role,
  type Strictness,
  type Violation,
  type World,
} from '@campaign/shared';

import type { Db } from './db.js';

/** How many events between snapshots. Small enough to matter, large enough not to churn. */
const SNAPSHOT_EVERY = 50;

export interface CampaignRow {
  readonly id: string;
  readonly name: string;
  readonly worldHash: string;
  readonly worldDoc: unknown;
  readonly world: World;
  readonly strictness: Strictness;
}

export const sha256 = (s: string): string =>
  'sha256:' + createHash('sha256').update(s).digest('hex');

/** A join token, and the hash that is all the database ever holds. */
export const newToken = (): string => randomBytes(32).toString('base64url');
export const hashToken = (t: string): string => createHash('sha256').update(t).digest('hex');

export class CampaignStore {
  constructor(
    private readonly db: Db,
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
  }): { campaign: CampaignRow; refereeToken: string } {
    const blob = JSON.stringify(opts.worldDoc);
    const world = parseWorld(opts.worldDoc);
    const strictness = opts.strictness ?? 'strict';

    this.db
      .prepare(
        `INSERT INTO campaigns (id, name, world_hash, world_blob, strictness, created_at)
         VALUES (?, ?, ?, ?, ?, ?)`,
      )
      .run(opts.id, opts.name, sha256(blob), blob, strictness, new Date().toISOString());

    const refereeToken = newToken();
    this.db
      .prepare(
        `INSERT INTO roles (campaign_id, token_hash, role_kind, commander_id) VALUES (?, ?, ?, ?)`,
      )
      .run(opts.id, hashToken(refereeToken), 'referee', null);

    const campaign: CampaignRow = {
      id: opts.id,
      name: opts.name,
      worldHash: sha256(blob),
      worldDoc: opts.worldDoc,
      world,
      strictness,
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
          hash: sha256(blob),
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

    return { campaign, refereeToken };
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

  campaign(id: string): CampaignRow | null {
    const row = this.db
      .prepare(`SELECT id, name, world_hash, world_blob, strictness FROM campaigns WHERE id = ?`)
      .get(id) as
      | { id: string; name: string; world_hash: string; world_blob: string; strictness: string }
      | undefined;
    if (row === undefined) return null;

    const worldDoc = JSON.parse(row.world_blob) as unknown;
    return {
      id: row.id,
      name: row.name,
      worldHash: row.world_hash,
      worldDoc,
      world: parseWorld(worldDoc),
      strictness: row.strictness as Strictness,
    };
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
      payload: JSON.parse(r.payload_json),
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

    let state = this.state(campaign.id);
    const outcome = apply(command, state, campaign.world, campaign.strictness, {
      actor,
      cfg: this.cfg,
      ...(opts.force !== undefined ? { force: opts.force } : {}),
      ...(opts.strictness !== undefined ? { strictness: opts.strictness } : {}),
    });

    if (!outcome.ok) return { ok: false, events: [], violations: outcome.violations };

    const all = [...outcome.events];
    state = outcome.state;

    for (const payload of observationEvents(state, campaign.world, this.cfg)) {
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

  private maybeSnapshot(campaignId: string, state: CampaignState): void {
    if (state.nextSeq % SNAPSHOT_EVERY !== 0) return;
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
 * would be indistinguishable from a man who forgot the campaign.
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
    })),
    despatches: [...state.despatches.values()],
    tasks: [...state.tasks.values()],
    decisions: [...state.decisions.values()],
  });
}

export function deserialise(json: string): CampaignState {
  const d = JSON.parse(json) as {
    name: string;
    world: CampaignState['world'];
    seed: number;
    clockHours: number;
    nextSeq: number;
    factions: CampaignState['factions'] extends ReadonlyMap<string, infer F> ? F[] : never;
    commanders: Commander[];
    units: CampaignState['units'] extends ReadonlyMap<string, infer U> ? U[] : never;
    knowledge: {
      commanderId: string;
      surveyed: string[];
      lastSurveyedHours: [string, number][];
    }[];
    despatches?: Despatch[];
    tasks?: Task[];
    decisions?: PendingDecision[];
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
    units: new Map(d.units.map((u) => [u.id, u])),
    knowledge: new Map(
      (d.knowledge ?? []).map((k) => [
        k.commanderId,
        {
          commanderId: k.commanderId,
          surveyed: new Set(k.surveyed),
          lastSurveyedHours: new Map(k.lastSurveyedHours),
        },
      ]),
    ),
    // Tolerated as absent for the same reason as `commanders`: a snapshot written before
    // riders existed still folds, and the events after it rebuild what it lacks.
    despatches: new Map((d.despatches ?? []).map((x) => [x.id, x])),
    tasks: new Map((d.tasks ?? []).map((t) => [t.unitId, t])),
    decisions: new Map((d.decisions ?? []).map((k) => [k.id, k])),
  };
}
