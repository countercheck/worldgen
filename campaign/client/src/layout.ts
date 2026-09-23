/**
 * The console on a phone.
 *
 * A wide screen shows everything at once: the map, and a sidebar holding the post, the
 * command and whatever is under the cursor. A phone has room for the map or for the
 * sidebar, not both, so below `NARROW` the sidebar is split into panes and a tab bar
 * picks one. The map pane keeps the map and slides what is selected up over it; the others
 * take the whole screen.
 *
 * Nothing about what is shown changes — every section is still rendered, and CSS decides
 * which pane it is in. That keeps a single tree of panels rather than a phone copy that
 * drifts from the desktop one.
 */

import { useEffect, useState } from 'react';

/** The widest a screen can be and still get the phone layout. Matches `style.css`. */
export const NARROW = '(max-width: 700px)';

export type Pane = 'map' | 'post' | 'command' | 'orbat';

/**
 * The tabs a role gets, in order.
 *
 * A commander has their own formation to watch, so they get Command. A referee has no
 * formation of their own — what they watch is the queue, which is on the map pane because
 * every decision in it is about a place.
 */
export function panesFor(role: 'referee' | 'commander'): readonly Pane[] {
  return role === 'referee' ? ['map', 'post', 'orbat'] : ['map', 'post', 'command', 'orbat'];
}

/**
 * Whether the reader is holding the screen rather than pointing at it.
 *
 * Asked of the device, not of the width: a tablet is wide and has no hover, and a narrow
 * desktop window has one. It decides which words an instruction uses — "tap" or "click",
 * with or without Escape — and nothing about layout.
 */
export function coarsePointer(): boolean {
  return typeof window !== 'undefined' && window.matchMedia?.('(pointer: coarse)').matches === true;
}

/** Whether the screen is narrow enough for the phone layout, kept current as it changes. */
export function useNarrow(): boolean {
  const query = typeof window === 'undefined' ? undefined : window.matchMedia?.(NARROW);
  const [narrow, setNarrow] = useState(query?.matches === true);

  useEffect(() => {
    if (query === undefined) return;
    const onChange = (): void => setNarrow(query.matches);
    query.addEventListener('change', onChange);
    onChange();
    return () => query.removeEventListener('change', onChange);
  }, [query?.media]);

  return narrow;
}
