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

import type { ClientView, Command, Faction, Strictness } from '@campaign/shared';

/** A campaign and the token that says who you are in it. */
export interface Session {
  readonly campaignId: string;
  readonly token: string;
}

export interface CreateResult {
  readonly id: string;
  readonly refereeToken: string;
  /** One per faction. The referee keeps these to hand out; nobody else ever sees them. */
  readonly factionTokens: Record<string, string>;
  readonly joinLinks: Record<string, string>;
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
  const headers: Record<string, string> = { 'content-type': 'application/json' };
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

export function advanceClock(session: Session, hours: number): Promise<{ clockHours: number }> {
  return request<{ clockHours: number }>(
    `/api/campaigns/${encodeURIComponent(session.campaignId)}/advance`,
    { method: 'POST', body: JSON.stringify({ hours }) },
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
