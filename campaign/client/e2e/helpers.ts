/**
 * What every browser test needs: the campaign the setup step made, and a hand.
 *
 * Touches go through the DevTools protocol rather than Playwright's `tap`, because `tap`
 * is one finger that never moves, and the things worth testing about the map are a finger
 * that does move and two fingers at once. Chromium turns them into the same pointer events
 * a phone would send.
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { expect, type Locator, type Page } from '@playwright/test';

export const STATE_DIR = fileURLToPath(new URL('./.state/', import.meta.url));
/** The referee's browser storage: the campaign's tokens, as the front page left them. */
export const STORAGE = join(STATE_DIR, 'referee.json');
const CAMPAIGN = join(STATE_DIR, 'campaign.json');

export type Point = { x: number; y: number };

export function campaignPath(): string {
  return (JSON.parse(readFileSync(CAMPAIGN, 'utf8')) as { path: string }).path;
}

export function saveCampaignPath(path: string): string {
  return JSON.stringify({ path });
}

export const CAMPAIGN_FILE = CAMPAIGN;

/** Open the demonstration as its referee, and wait for the map to be drawn. */
export async function openConsole(page: Page): Promise<void> {
  await page.goto(campaignPath());
  await page.locator('.map canvas').first().waitFor();
}

/** The middle of the map, and a point off to one side of it. */
export async function mapPoints(page: Page): Promise<{ centre: Point; aside: Point }> {
  const box = await page.locator('.map').boundingBox();
  if (box === null) throw new Error('the map is not on screen');
  const centre = { x: box.x + box.width / 2, y: box.y + box.height / 3 };
  return { centre, aside: { x: centre.x + box.width / 5, y: centre.y } };
}

type TouchType = 'touchStart' | 'touchMove' | 'touchEnd';

/**
 * One gesture's worth of touches, over one DevTools session. Chromium tracks a touch per
 * session, so a finger lifted on a different session from the one it went down on is a
 * finger it never saw go down.
 */
async function gesture(page: Page, steps: readonly [TouchType, readonly Point[]][]): Promise<void> {
  const cdp = await page.context().newCDPSession(page);
  for (const [type, points] of steps) {
    await cdp.send('Input.dispatchTouchEvent', {
      type,
      touchPoints: points.map((p, id) => ({ x: p.x, y: p.y, id })),
    });
  }
  await cdp.detach();
}

export async function fingerTap(page: Page, at: Point): Promise<void> {
  await gesture(page, [
    ['touchStart', [at]],
    ['touchEnd', []],
  ]);
}

/** One finger, pressed at `from` and dragged by `by` in ten steps. */
export async function fingerDrag(page: Page, from: Point, by: Point): Promise<void> {
  const moves = Array.from({ length: 10 }, (_, i): [TouchType, Point[]] => [
    'touchMove',
    [{ x: from.x + (by.x * (i + 1)) / 10, y: from.y + (by.y * (i + 1)) / 10 }],
  ]);
  await gesture(page, [['touchStart', [from]], ...moves, ['touchEnd', []]]);
}

/** Two fingers either side of `about`, spread from `from` to `to` pixels apart. */
export async function pinch(page: Page, about: Point, from: number, to: number): Promise<void> {
  const pair = (gap: number): Point[] => [
    { x: about.x - gap / 2, y: about.y },
    { x: about.x + gap / 2, y: about.y },
  ];
  const moves = Array.from({ length: 10 }, (_, i): [TouchType, Point[]] => [
    'touchMove',
    pair(from + ((to - from) * (i + 1)) / 10),
  ]);
  await gesture(page, [['touchStart', pair(from)], ...moves, ['touchEnd', []]]);
}

/** The hex the sidebar is reading, as its heading says it — `Hex 15, 8` — or null. */
export async function hexInSidebar(page: Page): Promise<string | null> {
  const heading = page.locator('.sidebar h3').filter({ hasText: /^hex -?\d+, -?\d+$/i });
  return (await heading.count()) === 0 ? null : (await heading.first().textContent())?.trim() ?? null;
}

export function moreButton(page: Page): Locator {
  return page.getByRole('button', { name: 'More', exact: true });
}

export function tab(page: Page, name: string): Locator {
  return page.locator('.tabbar').getByRole('button', { name, exact: true });
}

export async function tabNames(page: Page): Promise<string[]> {
  return (await page.locator('.tabbar button > span:not(.badge)').allTextContents()).map((t) =>
    t.trim(),
  );
}

/**
 * Take a commander's seat through the More sheet, as a referee holding every link does.
 * `nth` counts the seats after the referee's own.
 */
export async function takeSeat(page: Page, nth = 1): Promise<void> {
  await moreButton(page).click();
  const seats = page.getByRole('group', { name: 'Seat' }).getByRole('radio');
  // A click, not `check()`: taking a seat replaces the whole console, sheet and all, so
  // there is no radio left to confirm was checked.
  await seats.nth(nth).click();
  await expect(tab(page, 'Command')).toBeVisible();
}
