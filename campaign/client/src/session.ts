/**
 * Who this browser is, and how it found out.
 *
 * A join link is the whole of identity here — no accounts, no password — so this is
 * mostly about getting a token out of a URL and into storage without leaving it lying
 * around in the address bar, where it would end up in history, in a bookmark and in the
 * first screenshot anybody takes of the game.
 *
 * The link is a fragment (`#/j/<campaign>/<token>`) rather than a path. A fragment is
 * never sent to the server, so the token stays out of access logs and out of the
 * `Referer` of anything the page later loads. It is read once on load and then stripped.
 */

import type { Session } from './api.js';

const SESSION_KEY = 'campaign.session';
const TOKENS_KEY = 'campaign.tokens';
const SEATS_KEY = 'campaign.seats';

/** Tokens a referee holds for the other sides, so they can look through those eyes. */
export type HeldTokens = Record<string, string>;

interface Stored {
  readonly session: Session;
  /** Present only for the referee who created the campaign. */
  readonly held: HeldTokens;
  /** The token this browser was issued, whatever it is currently viewing as. */
  readonly ownToken: string;
  /**
   * Who each held seat belongs to.
   *
   * Kept beside the tokens so the switcher can name a man before the first view has
   * arrived, and so it still can while sitting in a seat whose own view — correctly —
   * does not list the enemy's commanders.
   */
  readonly seats?: Record<string, { name: string; faction: string }>;
}

const read = <T>(key: string): T | null => {
  try {
    const raw = localStorage.getItem(key);
    return raw === null ? null : (JSON.parse(raw) as T);
  } catch {
    // Private browsing, a disabled store, or something that is not ours and not JSON.
    // None of those is worth a broken page: the user can paste their link again.
    return null;
  }
};

const write = (key: string, value: unknown): void => {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* Storage is a convenience; the session still works for this tab. */
  }
};

/** `#/j/<campaign>/<token>` if the page was opened from a join link. */
export function sessionFromHash(hash = location.hash): Session | null {
  const m = /^#\/j\/([^/]+)\/([^/]+)$/.exec(hash);
  if (m === null) return null;
  return { campaignId: decodeURIComponent(m[1]!), token: decodeURIComponent(m[2]!) };
}

/** The join link to hand to somebody, absolute so it can be pasted anywhere. */
export const joinLink = (session: Session): string =>
  `${location.origin}/#/j/${encodeURIComponent(session.campaignId)}/${encodeURIComponent(session.token)}`;

/**
 * The session this browser should use, preferring a link just followed over a stored one.
 *
 * Following a link always wins: somebody who has just been sent a commander's link and
 * clicks it expects to become that commander, not to be told they are already the referee
 * from last week's game.
 */
export function loadSession(): Stored | null {
  const fromLink = sessionFromHash();
  if (fromLink !== null) {
    history.replaceState(null, '', location.pathname + location.search);
    const held = read<Record<string, HeldTokens>>(TOKENS_KEY)?.[fromLink.campaignId] ?? {};
    const stored: Stored = {
      session: fromLink,
      held,
      ownToken: fromLink.token,
      seats: read<Record<string, Stored['seats']>>(SEATS_KEY)?.[fromLink.campaignId] ?? {},
    };
    write(SESSION_KEY, stored);
    return stored;
  }
  return read<Stored>(SESSION_KEY);
}

export function saveSession(stored: Stored): void {
  write(SESSION_KEY, stored);
  if (Object.keys(stored.held).length > 0) {
    const all = read<Record<string, HeldTokens>>(TOKENS_KEY) ?? {};
    all[stored.session.campaignId] = stored.held;
    write(TOKENS_KEY, all);

    const seats = read<Record<string, Stored['seats']>>(SEATS_KEY) ?? {};
    seats[stored.session.campaignId] = stored.seats ?? {};
    write(SEATS_KEY, seats);
  }
}

export function clearSession(): void {
  try {
    localStorage.removeItem(SESSION_KEY);
  } catch {
    /* Nothing to clear. */
  }
}

export type { Stored as StoredSession };
