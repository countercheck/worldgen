/**
 * The console at 390 × 844 with a touch screen.
 *
 * Each test states one thing the phone layout promises and checks it the way a person
 * would find out: by what is on screen, where it is, and what a finger does to it. Nothing
 * here reads the console's state; the layout is only right if it looks right.
 */

import { expect, test } from '@playwright/test';

import {
  fingerDrag,
  fingerTap,
  hexInSidebar,
  mapPoints,
  moreButton,
  openConsole,
  pinch,
  tab,
  tabNames,
  takeSeat,
} from './helpers.js';

test.beforeEach(async ({ page }) => {
  await openConsole(page);
});

test.describe('the header', () => {
  test('keeps the name, the hour and a More button, and nothing else', async ({ page }) => {
    await expect(page.locator('header h1')).toBeVisible();
    await expect(page.locator('.clock')).toBeVisible();
    await expect(moreButton(page)).toBeVisible();
    await expect(page.locator('.header-tools')).toBeHidden();
    await expect(page.locator('button.home')).toBeHidden();
  });

  test('gives the whole header to the width of the screen', async ({ page }) => {
    const width = await page.evaluate(() => document.documentElement.scrollWidth);
    expect(width).toBeLessThanOrEqual(390);
  });
});

test.describe('the tabs', () => {
  test('are the map, the despatches and the order of battle for a referee', async ({ page }) => {
    expect(await tabNames(page)).toEqual(['Map', 'Despatches', 'Order of battle']);
    await expect(tab(page, 'Map')).toHaveAttribute('aria-current', 'page');
  });

  test('are the map, the post, the command and the order of battle for a commander', async ({
    page,
  }) => {
    await takeSeat(page);
    expect(await tabNames(page)).toEqual(['Map', 'Post', 'Command', 'Order of battle']);
  });

  test('show one pane at a time, and every pane but the map hides it', async ({ page }) => {
    const map = page.locator('.map');

    await tab(page, 'Despatches').click();
    await expect(map).toBeHidden();
    await expect(page.getByRole('button', { name: 'Write on a commander’s behalf' })).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Wants a decision' })).toBeHidden();

    await tab(page, 'Order of battle').click();
    await expect(map).toBeHidden();
    const roster = page.locator('.roster');
    await expect(roster).toBeVisible();
    expect((await roster.boundingBox())?.width).toBe(390);
    // The tab bar is the way out; the drawer's own Close would be a second one.
    await expect(roster.getByRole('button', { name: 'Close' })).toBeHidden();

    await tab(page, 'Map').click();
    await expect(map).toBeVisible();
    await expect(roster).toBeHidden();
    await expect(page.getByRole('heading', { name: 'Wants a decision' })).toBeVisible();
  });

  test('leave a formation opened in the order of battle open, and selected on the map', async ({
    page,
  }) => {
    await tab(page, 'Order of battle').click();
    const formation = page.locator('.roster button.cmd-unit').first();
    // The name is the button's own text; the swatch, the echelon mark and the rest are in
    // elements of their own.
    const name = await formation.evaluate(
      (el) =>
        [...el.childNodes]
          .find((n) => n.nodeType === Node.TEXT_NODE && n.textContent?.trim())
          ?.textContent?.trim() ?? '',
    );
    expect(name).not.toBe('');
    await formation.click();
    await expect(formation).toHaveAttribute('aria-expanded', 'true');
    await expect(page.locator('.roster')).toBeVisible();

    await tab(page, 'Map').click();
    await expect(page.locator('.sidebar')).toContainText(name);
  });
});

test.describe('the sheet over the map', () => {
  test('sits over the bottom of the map, above the tab bar', async ({ page }) => {
    const sheet = await page.locator('.sidebar').boundingBox();
    const tabs = await page.locator('.tabbar').boundingBox();
    const map = await page.locator('.map').boundingBox();
    if (sheet === null || tabs === null || map === null) throw new Error('not on screen');

    expect(Math.abs(sheet.y + sheet.height - tabs.y)).toBeLessThan(2);
    expect(sheet.y).toBeGreaterThan(map.y);
    expect(sheet.height).toBeLessThan(map.height * 0.45);
  });

  test('pulls up over most of the map, and back down', async ({ page }) => {
    const handle = page.getByRole('button', { name: 'Show more of this panel' });
    const sheet = page.locator('.sidebar');
    const before = (await sheet.boundingBox())?.height ?? 0;

    await handle.click();
    await expect(page.getByRole('button', { name: 'Show more of the map' })).toBeVisible();
    await expect
      .poll(async () => (await sheet.boundingBox())?.height ?? 0)
      .toBeGreaterThan(before * 1.8);

    await page.getByRole('button', { name: 'Show more of the map' }).click();
    await expect.poll(async () => (await sheet.boundingBox())?.height ?? 0).toBeCloseTo(before, 0);
  });
});

test.describe('a finger on the map', () => {
  test('taps to read the ground', async ({ page }) => {
    const { centre } = await mapPoints(page);
    expect(await hexInSidebar(page)).toBeNull();
    await fingerTap(page, centre);
    await expect.poll(() => hexInSidebar(page)).not.toBeNull();
  });

  test('drags to pan without reading, selecting or deselecting anything', async ({ page }) => {
    // Something selected first, so a drag that deselected on the press would show.
    await tab(page, 'Order of battle').click();
    await page.locator('.roster button.cmd-unit').first().click();
    await tab(page, 'Map').click();
    await expect(page.locator('.sidebar .unit-head')).not.toHaveCount(0);

    const { centre } = await mapPoints(page);
    const before = await page.locator('.sidebar').textContent();
    await fingerDrag(page, centre, { x: 90, y: 60 });
    await page.waitForTimeout(200);
    expect(await page.locator('.sidebar').textContent()).toBe(before);
  });

  test('pinches to zoom about the point between the fingers', async ({ page }) => {
    const { centre, aside } = await mapPoints(page);

    await fingerTap(page, centre);
    await expect.poll(() => hexInSidebar(page)).not.toBeNull();
    const middleBefore = await hexInSidebar(page);
    await fingerTap(page, aside);
    const asideBefore = await hexInSidebar(page);

    await pinch(page, centre, 40, 200);

    // The ground under the fingers stays under them; the ground beside it moves away.
    await fingerTap(page, centre);
    expect(await hexInSidebar(page)).toBe(middleBefore);
    await fingerTap(page, aside);
    await expect.poll(() => hexInSidebar(page)).not.toBe(asideBefore);
  });

  test('is told what to do in words for a finger', async ({ page }) => {
    await expect(page.getByText('Tap the ground to read it', { exact: false })).toBeVisible();
  });
});

test.describe('pointing at the map', () => {
  test('carries its own way out, and never asks for a key a phone does not have', async ({
    page,
  }) => {
    await page.locator('.clock-controls').getByRole('button', { name: 'Battle' }).click();
    const banner = page.locator('.notice.picking');
    await expect(banner).toBeVisible();
    await expect(banner).not.toContainText('Escape');

    await banner.getByRole('button', { name: 'Done' }).click();
    await expect(banner).toBeHidden();
  });
});

test.describe('the More sheet', () => {
  test('holds the seats and the way back to the front page', async ({ page }) => {
    await moreButton(page).click();
    const sheet = page.getByRole('dialog', { name: 'More' });
    await expect(sheet).toBeVisible();
    await expect(sheet.getByRole('group', { name: 'Seat' }).getByRole('radio')).not.toHaveCount(0);
    await expect(sheet.getByRole('radio', { name: 'Referee' })).toBeChecked();
    await expect(sheet.getByRole('button', { name: /^Campaigns/ })).toBeVisible();

    await sheet.getByRole('button', { name: 'Close' }).click();
    await expect(sheet).toBeHidden();
  });

  test('offers a commander the shading, and remembers which', async ({ page }) => {
    await takeSeat(page);
    await moreButton(page).click();
    const shading = page.getByRole('group', { name: 'Shading' });
    await shading.getByRole('radio', { name: /^Watched only/ }).check();
    await page.getByRole('dialog', { name: 'More' }).getByRole('button', { name: 'Close' }).click();

    await moreButton(page).click();
    await expect(shading.getByRole('radio', { name: /^Watched only/ })).toBeChecked();
  });

  test('switching seats starts a fresh console, not the last seat’s', async ({ page }) => {
    await tab(page, 'Despatches').click();
    await page.getByRole('button', { name: 'Write on a commander’s behalf' }).click();
    await page.locator('.composer textarea').fill('A draft in the referee’s hand');

    await takeSeat(page);
    await tab(page, 'Post').click();
    await expect(page.locator('.composer')).toHaveCount(0);
    await expect(page.getByText('A draft in the referee’s hand')).toHaveCount(0);
  });
});

test.describe('writing a despatch', () => {
  test('addresses it by row, each naming the formation, not by a dropdown', async ({ page }) => {
    await takeSeat(page);
    await tab(page, 'Post').click();
    await page.getByRole('button', { name: 'Write a despatch' }).click();

    const composer = page.locator('.composer');
    await expect(composer.locator('select')).toHaveCount(0);
    const to = composer.getByRole('group', { name: 'To' });
    const rows = to.getByRole('radio');
    expect(await rows.count()).toBeGreaterThan(1);
    await expect(to.getByRole('radio', { name: /The referee/ })).toBeVisible();
    await expect(to.locator('.choice-label .muted').first()).toContainText(' · ');

    await rows.nth(1).check();
    await expect(rows.nth(1)).toBeChecked();
    await expect(rows.nth(0)).not.toBeChecked();
  });

  test('opens where it can be seen', async ({ page }) => {
    await takeSeat(page);
    await tab(page, 'Post').click();
    await page.getByRole('button', { name: 'Write a despatch' }).click();
    await expect(page.getByRole('heading', { name: 'Write a despatch' })).toBeInViewport();
  });
});

test.describe('the order of battle on a touch screen', () => {
  test('shows the controls that wait for a hover on a desktop', async ({ page }) => {
    await tab(page, 'Order of battle').click();
    const action = page.locator('.roster .cmd-action').first();
    await expect(action).toBeVisible();
    expect(Number(await action.evaluate((el) => getComputedStyle(el).opacity))).toBe(1);
  });
});
