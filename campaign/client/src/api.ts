/**
 * The client's half of the wire.
 *
 * Everything the browser knows about a campaign arrives through here, and there is
 * nothing else: no bundled world, no locally assembled state, no second path that
 * happens to have more in it. The console draws whatever `ClientView` it was handed, so
 * what a commander can see is decided on the server and cannot be argued with from the
 * developer tools.
 *
 * `ClientView` is imported from the shared engine rather than declared here. A second
 * copy of that shape would be free to drift from the one the server actually sends, and
 * the description of a fog boundary is the last thing worth writing down twice.
 */

import type {
  ClientView,
  Command,
  DespatchBody,
  Commander,
  Faction,
  Formation,
  Hex,
  PendingDecision,
  Strictness,
  Unit,
  UnitStatChanges,
} from '@campaign/shared';

/** A campaign and the token that says who you are in it. */
export interface Session {
  readonly campaignId: string;
  readonly token: string;
}

export interface CreateResult {
  readonly id: string;
  /**
   * The referee's own link, and the only one that exists yet.
   *
   * A join link names a commander's seat, and there are no seats until formations are on
   * the map and commanders appointed to them. The referee sets up the order of battle and then
   * calls `issueSeatToken` for each seat somebody is to play.
   */
  readonly refereeToken: string;
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function request<T>(path: string, init: RequestInit = {}, token?: string): Promise<T> {
  const headers: Record<string, string> = {};
  // Declared only when there is actually a body. Fastify parses by content-type, so
  // announcing JSON and sending nothing is a 400 — which is how issuing a join link
  // failed, the request having neither a body nor any reason to want one.
  if (init.body !== undefined) headers['content-type'] = 'application/json';
  // The token travels in a header rather than the query string: a URL ends up in browser
  // history, in a screenshot and in any log the request passes through, and this one is
  // the whole of a player's identity.
  if (token !== undefined) headers['x-campaign-token'] = token;

  const res = await fetch(path, { ...init, headers: { ...headers, ...init.headers } });
  const text = await res.text();
  const body: unknown = text === '' ? null : JSON.parse(text);

  if (!res.ok) {
    const message =
      (body as { error?: string } | null)?.error ?? `${res.status} ${res.statusText}`;
    throw new ApiError(res.status, message);
  }
  return body as T;
}

/**
 * Compress a request body, where the browser can and the saving is worth having.
 *
 * `CompressionStream` is the platform's own gzip and needs nothing installed. It is not
 * universal — an older browser, or any page not served over a secure context — so this
 * returns the string unchanged when it is missing and the server accepts either. A
 * capability test rather than a version test: the feature is present or it is not.
 */
async function gzipped(body: string): Promise<BodyInit> {
  if (typeof CompressionStream === 'undefined') return body;
  try {
    const stream = new Blob([body]).stream().pipeThrough(new CompressionStream('gzip'));
    return await new Response(stream).blob();
  } catch {
    // Compressing is an optimisation and never the point. A browser that has the class
    // but fails on it should still be able to start a campaign.
    return body;
  }
}

/**
 * The one request worth compressing.
 *
 * A create carries a generated world and nothing else here comes close: 32 MB for a
 * 200x200 map against a few hundred bytes for an order. Gzipped that is 4.6 MB, which on
 * a domestic upstream is the difference between a few seconds and most of a minute.
 *
 * The server reads either. `@fastify/compress` decompresses the request when the header
 * says to, and enforces the body limit against the *decompressed* size — checked with a
 * 0.29 MB body declaring 300 MB, which is refused with a 413 rather than allocated.
 */
export async function createCampaign(opts: {
  name: string;
  world: unknown;
  factions: readonly Faction[];
  seed?: number;
  strictness?: Strictness;
}): Promise<CreateResult> {
  const json = JSON.stringify(opts);
  const body = await gzipped(json);

  return request<CreateResult>('/api/campaigns', {
    method: 'POST',
    body,
    ...(typeof body === 'string' ? {} : { headers: { 'content-encoding': 'gzip' } }),
  });
}

/** Mint a join link for one commander's seat. Referee only; returned once. */
export function issueSeatToken(
  session: Session,
  commanderId: string,
): Promise<{ commanderId: string; token: string }> {
  return request<{ commanderId: string; token: string }>(
    `/api/campaigns/${encodeURIComponent(session.campaignId)}` +
      `/commanders/${encodeURIComponent(commanderId)}/token`,
    { method: 'POST' },
    session.token,
  );
}

export function fetchView(session: Session): Promise<ClientView> {
  return request<ClientView>(
    `/api/campaigns/${encodeURIComponent(session.campaignId)}/view`,
    {},
    session.token,
  );
}

export interface CommandResult {
  readonly ok: boolean;
  readonly events?: number;
  readonly violations?: readonly { code: string; message: string; severity: string }[];
}

/**
 * Issue a command.
 *
 * A refusal is a 409 carrying the violations, which is a normal answer rather than a
 * failure — "you may not cross that river" is the game working. It comes back as a value
 * so the console can show the reason instead of a thrown error nobody reads.
 */
export async function sendCommand(
  session: Session,
  command: Command,
  opts: { force?: boolean; strictness?: Strictness } = {},
): Promise<CommandResult> {
  try {
    return await request<CommandResult>(
      `/api/campaigns/${encodeURIComponent(session.campaignId)}/commands`,
      { method: 'POST', body: JSON.stringify({ command, ...opts }) },
      session.token,
    );
  } catch (err) {
    if (err instanceof ApiError && err.status === 409) {
      return { ok: false, violations: [{ code: 'refused', message: err.message, severity: 'hard' }] };
    }
    throw err;
  }
}

/**
 * Write a despatch.
 *
 * For a commander, `from` is filled in by the server from the token rather than taken from
 * here — a forged *report* would let anyone feed a commander false intelligence signed by
 * their own subordinate — so a value sent by a commander is ignored rather than trusted.
 *
 * A referee's is honoured, because writing in a commander's name is their ordinary work: they
 * run most of the commanders on the map and take dictation from the players who hold the
 * rest. The despatch is from that commander; the event recording it is from the referee, so the
 * log says who actually put pen to paper.
 */
export function sendDespatch(
  session: Session,
  despatch: {
    to: string;
    body: DespatchBody;
    via?: readonly Hex[];
    forwardedFrom?: string;
    from?: string;
  },
): Promise<CommandResult> {
  return sendCommand(session, {
    kind: 'send_despatch',
    // A deliberately useless value is safer than a plausible one where the server is
    // going to overwrite it anyway.
    from: '',
    ...despatch,
  });
}

/**
 * A note to the referee, out of the game: no rider, no delay, and nobody else reads it.
 *
 * `from` is the server's to fill in, from the token, as for a despatch.
 */
export const writeToReferee = (session: Session, text: string): Promise<CommandResult> =>
  sendCommand(session, { kind: 'write_to_referee', from: '', text });

/**
 * Set a formation marching. Referee only, and deliberately so.
 *
 * A commander writes prose; turning prose into a march is the adjudication this whole
 * design exists to keep in human hands. A destination rather than a path: the referee says
 * where the corps is to be, and the engine works out how it gets there.
 *
 * `via` is the one qualification. A referee who says "to Ligny, by way of the bridge at
 * Genappe" is still naming places rather than fields, and the engine still routes between
 * them.
 */
export function setTask(
  session: Session,
  unitId: string,
  destination: Hex,
  opts: { fromDespatchId?: string; via?: readonly Hex[] } = {},
): Promise<CommandResult> {
  return sendCommand(session, { kind: 'set_task', unitId, destination, ...opts });
}

/**
 * Send a patrol out from a formation.
 *
 * Twenty troopers and a parent. What the patrol sees is what the parent's commander comes
 * to know, which is the whole reason a division bothers to detach one.
 */
export function detachPatrol(
  session: Session,
  unitId: string,
  opts: { at?: Hex; patrolId?: string; name?: string } = {},
): Promise<CommandResult> {
  return sendCommand(session, { kind: 'detach_patrol', unitId, ...opts });
}

/**
 * Put a formation on the map.
 *
 * The referee's, and the order of battle is their whole preparation for a game: they build
 * both sides, places them, and only then issues the links that let anyone see any of it.
 */
export function addUnit(session: Session, unit: Unit): Promise<CommandResult> {
  return sendCommand(session, { kind: 'add_unit', unit });
}

export function removeUnit(session: Session, unitId: string): Promise<CommandResult> {
  return sendCommand(session, { kind: 'remove_unit', unitId });
}

/**
 * Move a formation without marching it.
 *
 * Not a march and not pretending to be one: it breaks every movement rule at once, which
 * is why it is the referee's alone and why it is logged as what it is. Setting up a
 * scenario, correcting a mistake, and adjudicating something the rules do not cover are
 * all the same act.
 */
export function teleportUnit(
  session: Session,
  unitId: string,
  column: readonly Hex[],
): Promise<CommandResult> {
  return sendCommand(session, { kind: 'teleport_unit', unitId, column });
}

/** Change what a formation is. Every field optional; what is absent is left alone. */
export function setUnitStats(
  session: Session,
  unitId: string,
  changes: UnitStatChanges,
): Promise<CommandResult> {
  return sendCommand(session, { kind: 'set_unit_stats', unitId, changes });
}

export function addCommander(session: Session, commander: Commander): Promise<CommandResult> {
  return sendCommand(session, { kind: 'add_commander', commander });
}

export function removeCommander(
  session: Session,
  commanderId: string,
): Promise<CommandResult> {
  return sendCommand(session, { kind: 'remove_commander', commanderId });
}

/** Move a commander to another formation, or give them a new superior. */
export function reassignCommander(
  session: Session,
  commanderId: string,
  changes: { unitId?: string; superiorId?: string | null },
): Promise<CommandResult> {
  return sendCommand(session, { kind: 'reassign_commander', commanderId, ...changes });
}

export function clearTask(session: Session, unitId: string): Promise<CommandResult> {
  return sendCommand(session, { kind: 'clear_task', unitId });
}

/**
 * Order a change of formation.
 *
 * Making camp at the end of a day and breaking it to march again both happen without
 * asking. This is everything else — forming for battle, camping early, occupying a town —
 * and it is the referee's, like every other order.
 */
export function setFormation(
  session: Session,
  unitId: string,
  formation: Formation,
): Promise<CommandResult> {
  return sendCommand(session, { kind: 'set_formation', unitId, formation });
}

/** Mark a decision dealt with. The note is the referee's own record of why. */
export function resolveDecision(
  session: Session,
  decisionId: string,
  note?: string,
  favouring?: string,
): Promise<CommandResult> {
  return sendCommand(session, {
    kind: 'resolve_decision',
    decisionId,
    ...(note === undefined || note === '' ? {} : { note }),
    // Only a contested hex has anything to rule between. Everywhere else this is absent
    // and the decision is simply acknowledged.
    ...(favouring === undefined ? {} : { favouring }),
  });
}

export interface AdvanceResult {
  readonly clockHours: number;
  /** What stopped the clock, when the referee asked it to stop for something. */
  readonly halted: PendingDecision | null;
}

/**
 * Run the clock.
 *
 * `untilDecision` is the control a referee actually uses: run forward and stop the moment
 * something needs a human, rather than guessing at an interval and finding out afterwards
 * that two corps met each other ninety minutes in.
 */
export function advanceClock(
  session: Session,
  hours: number,
  untilDecision = false,
): Promise<AdvanceResult> {
  return request<AdvanceResult>(
    `/api/campaigns/${encodeURIComponent(session.campaignId)}/advance`,
    { method: 'POST', body: JSON.stringify({ hours, untilDecision }) },
    session.token,
  );
}

/**
 * Subscribe to this session's view.
 *
 * The server rebuilds the payload per socket, so two browsers on the same campaign
 * receive genuinely different pictures from the same event. Reconnects on drop with a
 * backoff, because a referee advancing the clock while somebody's laptop sleeps should
 * not mean that player is quietly looking at a stale map for the rest of the evening.
 */
export function subscribe(
  session: Session,
  handlers: {
    onView: (view: ClientView) => void;
    onStatus?: (status: 'open' | 'closed' | 'error') => void;
  },
): () => void {
  let socket: WebSocket | null = null;
  let retry = 0;
  let timer: number | undefined;
  let closed = false;

  const open = (): void => {
    if (closed) return;
    const scheme = location.protocol === 'https:' ? 'wss:' : 'ws:';
    // The socket cannot carry a header, so this is the one place the token is a query
    // parameter. It is the same secret either way; only its exposure in logs differs.
    const url =
      `${scheme}//${location.host}/api/campaigns/${encodeURIComponent(session.campaignId)}` +
      `/stream?token=${encodeURIComponent(session.token)}`;

    socket = new WebSocket(url);

    socket.onopen = () => {
      retry = 0;
      handlers.onStatus?.('open');
    };
    socket.onmessage = (ev: MessageEvent<string>) => {
      const msg = JSON.parse(ev.data) as { type: string; view?: ClientView };
      if (msg.type === 'view' && msg.view !== undefined) handlers.onView(msg.view);
      if (msg.type === 'error') handlers.onStatus?.('error');
    };
    socket.onclose = () => {
      handlers.onStatus?.('closed');
      if (closed) return;
      retry = Math.min(retry + 1, 6);
      timer = window.setTimeout(open, 250 * 2 ** retry);
    };
  };

  open();

  return () => {
    closed = true;
    if (timer !== undefined) window.clearTimeout(timer);
    socket?.close();
  };
}

/**
 * Referee: declare ground as being fought over.
 *
 * A set of hexes, not a battle object. The campaign layer records where the fighting is so
 * that the rules written for open country stop applying there; what happens inside is
 * below the resolution of a 1 km map.
 */
export const declareBattle = (s: Session, coords: readonly Hex[]): Promise<CommandResult> =>
  sendCommand(s, { kind: 'declare_battle', coords });

/** Referee: the fighting here is over, and the ground goes back to being ground. */
export const endBattle = (s: Session, coords: readonly Hex[]): Promise<CommandResult> =>
  sendCommand(s, { kind: 'end_battle', coords });
