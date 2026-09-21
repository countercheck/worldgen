/**
 * Who this browser is, and how it found out.
 *
 * A join link is the whole of identity here — no accounts, no password — so this is
 * mostly about getting a token out of a URL and into storage without leaving it lying
 * around in the address bar, where it would end up in history, in a bookmark and in the
 * first screenshot anybody takes of the game. `route.ts` owns the addresses; this owns
 * what is kept behind them.
 *
 * ## Why storage is keyed by campaign
 *
 * It used to hold one session: whatever this browser last opened. That made the campaign
 * a property of the browser rather than of the address, which is why there was nowhere to
 * go back to — the front page and the game were the same URL, and the last one to be
 * stored won. Keyed by campaign id, `#/c/<id>` can find its own token, the front page can
 * list every game this browser can still get into, and a referee can hold two campaigns
 * at once without one evicting the other.
 */

import { campaignHash, absolute } from './route.js';

import type { Session } from './api.js';
import type { WashMode } from './map/draw.js';

/** Every campaign this browser holds a token for, keyed by campaign id. */
const STORE_KEY = 'campaign.campaigns';
/** The single-session key this replaced. Read once, to migrate, then left alone. */
const LEGACY_SESSION_KEY = 'campaign.session';
const LEGACY_TOKENS_KEY = 'campaign.tokens';
const LEGACY_SEATS_KEY = 'campaign.seats';
const WASH_KEY = 'campaign.wash';

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
   * Kept beside the tokens so the switcher can name a commander before the first view has
   * arrived, and so it still can while sitting in a seat whose own view — correctly —
   * does not list the enemy's commanders.
   */
  readonly seats?: Record<string, { name: string; faction: string }>;
  /**
   * The campaign's name, as of the last view this browser saw.
   *
   * Only so the front page can list a game by its name rather than its id. It is a cache
   * of something the server owns, and it is allowed to be stale: a campaign renamed since
   * the last visit is listed under its old name until the next one, which is a better
   * failure than listing it as `a3f9c1`.
   */
  readonly name?: string;
  /**
   * Which seat this browser's own token opens, as the server reported it.
   *
   * Not inferred from `held`. A referee who arrives on their own join link holds no seat
   * tokens — those are minted once, when the campaign is created — so counting them
   * would file the referee's own link under "a commander". The server is the only thing
   * that actually knows, and it says so in every view.
   */
  readonly role?: 'referee' | 'commander';
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

/**
 * Everything this browser holds, migrating the old single session if it finds one.
 *
 * The app is deployed and people are holding links to games in progress, so the old key
 * is read rather than abandoned. It is read every time rather than migrated once and
 * deleted: a browser with two tabs open on the old build would otherwise lose whichever
 * session it wrote second, and leaving the legacy key alone costs nothing.
 */
function readAll(): Record<string, Stored> {
  const store = read<Record<string, Stored>>(STORE_KEY) ?? {};

  const legacy = read<Stored>(LEGACY_SESSION_KEY);
  if (legacy !== null && !Object.hasOwn(store, legacy.session.campaignId)) {
    const id = legacy.session.campaignId;
    store[id] = {
      ...legacy,
      held: legacy.held ?? read<Record<string, HeldTokens>>(LEGACY_TOKENS_KEY)?.[id] ?? {},
      seats:
        legacy.seats ?? read<Record<string, Stored['seats']>>(LEGACY_SEATS_KEY)?.[id] ?? {},
    };
  }

  return store;
}

/** What this browser holds for one campaign, or nothing if it was never given a link. */
export const loadCampaign = (campaignId: string): Stored | null =>
  readAll()[campaignId] ?? null;

export function saveCampaign(stored: Stored): void {
  const all = readAll();
  all[stored.session.campaignId] = stored;
  write(STORE_KEY, all);
}

/** Drop one campaign's tokens. The game is untouched; this browser just forgets the way in. */
export function forgetCampaign(campaignId: string): void {
  const all = readAll();
  delete all[campaignId];
  write(STORE_KEY, all);

  // The legacy key would otherwise resurrect it on the next read.
  try {
    const legacy = read<Stored>(LEGACY_SESSION_KEY);
    if (legacy?.session.campaignId === campaignId) localStorage.removeItem(LEGACY_SESSION_KEY);
  } catch {
    /* Nothing to clear. */
  }
}

/** One line on the front page: a game this browser can still get back into. */
export interface CampaignSummary {
  readonly campaignId: string;
  readonly name: string | null;
  /** Whether this browser's link opens the referee's seat. */
  readonly isReferee: boolean;
}

/**
 * Every campaign this browser can still open, for the front page to list.
 *
 * Without this the front page would be a dead end for anybody who followed a link: the
 * token is stripped from the address on arrival, so going home would strand them with no
 * way back that did not involve finding the original message again.
 */
export const listCampaigns = (): readonly CampaignSummary[] =>
  Object.values(readAll())
    .map((s) => ({
      campaignId: s.session.campaignId,
      name: s.name ?? null,
      // The stored role where the campaign has been opened before; otherwise the seat
      // tokens, which only a referee who created the campaign ever holds.
      isReferee: s.role === undefined ? Object.keys(s.held).length > 0 : s.role === 'referee',
    }))
    .sort((a, b) => (a.name ?? a.campaignId).localeCompare(b.name ?? b.campaignId));

/**
 * Note what the server says this campaign is called and which seat we hold in it.
 *
 * Called from the console once a view lands — the only place either is known. Returns
 * whether anything was written, so the caller can avoid a render it does not need.
 */
export function rememberCampaign(
  campaignId: string,
  facts: { name: string; role: 'referee' | 'commander' },
): boolean {
  const all = readAll();
  const stored = all[campaignId];
  if (stored === undefined) return false;
  if (stored.name === facts.name && stored.role === facts.role) return false;
  all[campaignId] = { ...stored, ...facts };
  write(STORE_KEY, all);
  return true;
}

/** The join link to hand to somebody, absolute so it can be pasted anywhere. */
export const joinLink = (session: Session): string =>
  absolute(joinHashFor(session));

const joinHashFor = (session: Session): string =>
  `#/j/${encodeURIComponent(session.campaignId)}/${encodeURIComponent(session.token)}`;

/** A campaign's own address, absolute — what a player bookmarks to come back to. */
export const campaignLink = (campaignId: string): string => absolute(campaignHash(campaignId));

const WASH_MODES: readonly WashMode[] = ['three', 'two', 'off'];

/**
 * How much wash this browser last asked for.
 *
 * Kept apart from the campaigns rather than inside one, because it belongs to the person
 * at the keyboard rather than to the seat: a referee switching between four commanders'
 * views should not have to re-press the key four times, and should still have their choice
 * after they reloads.
 *
 * Validated on the way out. The value is in `localStorage`, which anybody can edit, and
 * an unknown mode would otherwise reach the renderer as a band nothing matches.
 */
export function loadWash(): WashMode | null {
  const stored = read<string>(WASH_KEY);
  return WASH_MODES.includes(stored as WashMode) ? (stored as WashMode) : null;
}

export const saveWash = (mode: WashMode): void => write(WASH_KEY, mode);

/** The next mode in the cycle, so the key and any button agree on the order. */
export const nextWash = (mode: WashMode): WashMode =>
  WASH_MODES[(WASH_MODES.indexOf(mode) + 1) % WASH_MODES.length]!;

export type { Stored as StoredSession };
