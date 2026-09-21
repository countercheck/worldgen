/**
 * Where in the app this browser is, as a URL.
 *
 * Until now there was one address for everything. The token came in on a fragment, was
 * read once and stripped, and the browser then sat at `/` with the campaign held only in
 * `localStorage` — so a player who wanted the front page could not get to it, and a
 * player who wanted their game back had to find the original link again. One address
 * cannot be both.
 *
 * Three routes, and the split between them is about what is secret:
 *
 * - `#/` is the front page.
 * - `#/c/<campaign>` is a campaign, and carries **no token**. A campaign id is not a
 *   secret — it is in every join link and every log line — so this is safe to bookmark,
 *   paste into a chat, or leave in browser history. It only opens for somebody whose
 *   browser already holds a token for that campaign, which is the property that matters.
 * - `#/j/<campaign>/<token>` is a join link, and is the one address that carries a
 *   token. It is consumed on arrival: the token goes to storage and the address is
 *   replaced with the campaign's own. So the secret is in the URL for one paint and is
 *   never what the player is left looking at.
 *
 * All three are fragments rather than paths. A fragment is never sent to the server, so
 * the token in a join link stays out of access logs and out of the `Referer` of anything
 * the page later loads — the reason the original scheme used one, and a reason that has
 * not changed. It also means no server-side rewrite rule is needed for any of this.
 */

export type Route =
  | { readonly kind: 'home' }
  | { readonly kind: 'campaign'; readonly campaignId: string }
  | { readonly kind: 'join'; readonly campaignId: string; readonly token: string };

export const HOME_HASH = '#/';

/** A campaign's own address. Stable, tokenless, and safe to bookmark. */
export const campaignHash = (campaignId: string): string =>
  `#/c/${encodeURIComponent(campaignId)}`;

/** A join link's fragment. The only address that carries a token. */
export const joinHash = (campaignId: string, token: string): string =>
  `#/j/${encodeURIComponent(campaignId)}/${encodeURIComponent(token)}`;

/**
 * The route a fragment names.
 *
 * Anything unrecognised is the front page rather than an error. A mistyped or truncated
 * link should land somebody somewhere they can act from, and the front page now lists the
 * campaigns their browser holds — so the recovery from a bad address is visible rather
 * than something they have to know to do.
 */
export function parseRoute(hash: string = location.hash): Route {
  const join = /^#\/j\/([^/]+)\/([^/]+)\/?$/.exec(hash);
  if (join !== null) {
    return {
      kind: 'join',
      campaignId: decodeURIComponent(join[1]!),
      token: decodeURIComponent(join[2]!),
    };
  }

  const campaign = /^#\/c\/([^/]+)\/?$/.exec(hash);
  if (campaign !== null) {
    return { kind: 'campaign', campaignId: decodeURIComponent(campaign[1]!) };
  }

  return { kind: 'home' };
}

/** An absolute URL for a fragment, for a link somebody is going to paste elsewhere. */
export const absolute = (hash: string): string => `${location.origin}/${hash}`;

/** Go somewhere, leaving a history entry, so the back button means what it looks like. */
export const navigate = (hash: string): void => {
  location.hash = hash;
};

/**
 * Go somewhere without leaving a history entry.
 *
 * For swapping a consumed join link for the campaign's own address. A history entry there
 * would put the token back in the address bar on the first press of Back, which is the
 * one thing this whole scheme is arranged to avoid.
 */
export const replace = (hash: string): void => {
  history.replaceState(null, '', `${location.pathname}${location.search}${hash}`);
};
