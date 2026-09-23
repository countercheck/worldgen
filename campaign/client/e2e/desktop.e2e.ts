/**
 * The console on a wide screen, where the phone layout must not show at all.
 *
 * The phone's panes are groups the desktop is told to ignore (`display: contents`), so a
 * mistake there shows up as a desktop that has quietly changed shape. These pin the
 * shape it had.
 */

import { expect, test } from '@playwright/test';

import { hexInSidebar, mapPoints, openConsole, type Point } from './helpers.js';

test.beforeEach(async ({ page }) => {
  await openConsole(page);
});

test('has no tab bar, no More button and no sheet handle', async ({ page }) => {
  await expect(page.locator('.tabbar')).toBeHidden();
  await expect(page.getByRole('button', { name: 'More', exact: true })).toBeHidden();
  await expect(page.locator('.sheet-handle')).toBeHidden();
});

test('keeps the seats, the order of battle and the clock in the header', async ({ page }) => {
  const header = page.locator('header');
  await expect(header.getByRole('button', { name: 'Referee' })).toBeVisible();
  await expect(header.getByRole('button', { name: /^Order of battle/ })).toBeVisible();
  await expect(header.getByRole('button', { name: 'Run' })).toBeVisible();
  await expect(header.getByRole('button', { name: 'Campaigns' })).toBeVisible();
});

test('puts the sidebar beside the map, with every pane in it at once', async ({ page }) => {
  const map = await page.locator('.map').boundingBox();
  const sidebar = await page.locator('.sidebar').boundingBox();
  if (map === null || sidebar === null) throw new Error('not on screen');
  expect(sidebar.x).toBeGreaterThanOrEqual(map.x + map.width - 1);
  expect(sidebar.height).toBeCloseTo(map.height, 0);

  await expect(page.getByRole('heading', { name: 'Wants a decision' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Despatches' })).toBeVisible();
});

test('still says Escape in a pointing banner, beside the new button', async ({ page }) => {
  await page.locator('header').getByRole('button', { name: 'Battle', exact: true }).click();
  const banner = page.locator('.notice.picking');
  await expect(banner).toContainText('Escape when the field is drawn.');
  await page.keyboard.press('Escape');
  await expect(banner).toBeHidden();
});

test('addresses a despatch with a dropdown', async ({ page }) => {
  await page.getByRole('button', { name: 'Write on a commander’s behalf' }).click();
  await expect(page.locator('.composer select')).toHaveCount(2);
  await expect(page.locator('.composer').getByRole('radio')).toHaveCount(0);
});

test('opens the order of battle as a drawer over the map, not instead of it', async ({ page }) => {
  await page.locator('header').getByRole('button', { name: /^Order of battle/ }).click();
  await expect(page.locator('.roster')).toBeVisible();
  await expect(page.locator('.map')).toBeVisible();
});

test('drags the map with the ground held under the cursor', async ({ page }) => {
  // A drag that only starts once the press has left the tap slop must still pan from where
  // the press went down, or the map trails the cursor by the slop for the rest of it.
  const { centre } = await mapPoints(page);
  const by = { x: 60, y: 40 };
  const probes = [-40, -20, 0, 20, 40].map((dx) => ({ x: centre.x + dx, y: centre.y + dx / 2 }));

  const hexUnder = async (p: Point): Promise<string | null> => {
    await page.mouse.move(p.x, p.y);
    return hexInSidebar(page);
  };
  const before: (string | null)[] = [];
  for (const p of probes) before.push(await hexUnder(p));

  await page.mouse.move(centre.x, centre.y);
  await page.mouse.down();
  await page.mouse.move(centre.x + by.x, centre.y + by.y, { steps: 30 });
  await page.mouse.up();

  const after: (string | null)[] = [];
  for (const p of probes) after.push(await hexUnder({ x: p.x + by.x, y: p.y + by.y }));
  expect(after).toEqual(before);
});
