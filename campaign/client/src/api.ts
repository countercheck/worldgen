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
  DespatchKind,
  Faction,
  Hex,
  PendingDecision,
  Strictness,
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
   * the map and men appointed to them. The referee sets up the order of battle and then
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

export function createCampaign(opts: {
  name: string;
  world: unknown;
  factions: readonly Faction[];
  seed?: number;
  strictness?: Strictness;
}): Promise<CreateResult> {
  return request<CreateResult>('/api/campaigns', {
    method: 'POST',
    body: JSON.stringify(opts),
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
 * The one command a commander issues. `from` is filled in by the server from the token
 * rather than taken from here — a forged *report* would let anyone feed a commander false
 * intelligence signed by his own subordinate — so the client does not send it at all,
 * and a value here would be ignored rather than trusted.
 */
export function sendDespatch(
  session: Session,
  despatch: {
    to: string;
    despatchKind: DespatchKind;
    body: DespatchBody;
    via?: readonly Hex[];
    inReplyTo?: string;
    forwardedFrom?: string;
  },
): Promise<CommandResult> {
  return sendCommand(session, {
    kind: 'send_despatch',
    // Overwritten server-side. Sent only because the command type wants it, and a
    // deliberately useless value is safer than a plausible one.
    from: '',
    ...despatch,
  });
}

/**
 * Set a formation marching. Referee only, and deliberately so.
 *
 * A commander writes prose; turning prose into a march is the adjudication this whole
 * design exists to keep in human hands. A destination rather than a path: the referee says
 * where the corps is to be, and the engine works out how it gets there.
 */
export function setTask(
  session: Session,
  unitId: string,
  destination: Hex,
  opts: { fromDespatchId?: string } = {},
): Promise<CommandResult> {
  return sendCommand(session, { kind: 'set_task', unitId, destination, ...opts });
}

export function clearTask(session: Session, unitId: string): Promise<CommandResult> {
  return sendCommand(session, { kind: 'clear_task', unitId });
}

/** Mark a decision dealt with. The note is the referee's own record of why. */
export function resolveDecision(
  session: Session,
  decisionId: string,
  note?: string,
): Promise<CommandResult> {
  return sendCommand(session, {
    kind: 'resolve_decision',
    decisionId,
    ...(note === undefined || note === '' ? {} : { note }),
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
