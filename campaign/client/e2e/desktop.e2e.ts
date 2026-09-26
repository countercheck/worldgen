/**
 * The console on a wide screen, where the phone layout must not show at all.
 *
 * The phone's panes are groups the desktop is told to ignore (`display: contents`), so a
 * mistake there shows up as a desktop that has quietly changed shape. These pin the
 * shape it had.
 */

import { fileURLToPath } from 'node:url';

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

test('shows a sun or a moon by the clock, and lets the referee move the sun', async ({ page }) => {
  await expect(page.locator('.clock .day-night')).toHaveAttribute('aria-label', /^(Daylight|Night)\./);

  const sidebar = page.locator('.sidebar');
  await sidebar.getByLabel('Sunrise').selectOption('4');
  await sidebar.getByLabel('Sunset').selectOption('21');
  await sidebar.getByRole('button', { name: 'Set the sun' }).click();
  await expect(sidebar.getByText('Sunrise 04:00, sunset 21:00.')).toBeVisible();

  // Put it back, so the campaign every other test opens has the rules' own sun.
  await sidebar.getByLabel('Sunrise').selectOption('6');
  await sidebar.getByLabel('Sunset').selectOption('18');
  await sidebar.getByRole('button', { name: 'Set the sun' }).click();
  await expect(sidebar.getByText('Sunrise 06:00, sunset 18:00.')).toBeVisible();
});

test('gives a formation standing orders, and lifts them', async ({ page }) => {
  const sidebar = page.locator('.sidebar');
  await sidebar.locator('.unit-list button').first().click();
  await expect(sidebar.getByRole('heading', { name: 'Standing orders' })).toBeVisible();

  await sidebar.getByLabel('Step off at').selectOption('5');
  await sidebar.getByLabel('Off the road by').selectOption('19');
  await sidebar.getByLabel('Hours on the road').fill('10');
  await sidebar.getByRole('button', { name: 'Give these orders' }).click();
  await expect(
    sidebar.getByText('Steps off at 05:00 · off the road by 19:00 · 10 h on the road.'),
  ).toBeVisible();

  await sidebar.getByRole('button', { name: 'Lift them' }).click();
  await expect(sidebar.getByText(/^None given\./)).toBeVisible();
});

test('sets a formation’s values by hand, and refuses one that will not do', async ({ page }) => {
  const sidebar = page.locator('.sidebar');
  await sidebar.locator('.unit-list button').first().click();
  const form = sidebar.locator('.unit-edit');
  await form.getByText('Set by hand').click();
  const save = form.getByRole('button', { name: 'Set these' });

  await form.getByLabel('Fatigue').fill('101');
  await expect(form.getByText('fatigue runs from 0 to 100')).toBeVisible();
  await expect(save).toBeDisabled();
  await form.getByRole('button', { name: 'Undo changes' }).click();

  const morale = form.getByLabel('Morale');
  const road = form.getByLabel('On the road, last 24 h');
  const was = { morale: await morale.inputValue(), road: await road.inputValue() };
  await morale.fill('7');
  await road.fill('12');
  await save.click();
  // Taken: the form now reads what the unit is, with nothing left to send.
  await expect(save).toBeDisabled();
  await expect(morale).toHaveValue('7');
  await expect(road).toHaveValue('12');

  // A change under way, set by hand: it holds, and is cleared the same way.
  const changeTo = form.getByLabel('Changing to');
  await changeTo.selectOption('rest');
  await form.getByLabel('Finishes in, h').fill('3');
  await save.click();
  await expect(save).toBeDisabled();
  await expect(changeTo).toHaveValue('rest');
  await expect(form.getByLabel('Finishes in, h')).toHaveValue('3');
  await changeTo.selectOption('');
  await save.click();
  await expect(changeTo).toHaveValue('');

  // Put it back, so the campaign every other test opens is as it was.
  await morale.fill(was.morale);
  await road.fill(was.road);
  await save.click();
  await expect(morale).toHaveValue(was.morale);
});

test('hands the referee a commander’s link, and the same one when asked again', async ({ page }) => {
  await page.locator('header').getByRole('button', { name: /^Order of battle/ }).click();
  const roster = page.locator('.roster');
  const first = roster.getByRole('button', { name: 'Copy link' }).first();
  await first.focus();
  await first.click();

  const link = roster.locator('.cmd-link code').first();
  await expect(link).toContainText('#/j/');
  const sent = await link.textContent();

  // Asked again — a lost link resent — it is the same link, so the player's copy still works.
  await first.click();
  await expect(link).toHaveText(sent ?? '');
});

test('starts an uploaded world with no sides, and takes one the referee adds', async ({ page }) => {
  await page.goto('/');
  await page
    .locator('input[type=file]')
    .setInputFiles(fileURLToPath(new URL('../../shared/test/fixtures/world-32x32.json', import.meta.url)));
  await page.getByRole('button', { name: 'Enter as referee' }).click();
  await page.locator('.map canvas').first().waitFor();

  await page.locator('header').getByRole('button', { name: /^Order of battle/ }).click();
  const roster = page.locator('.roster');
  await expect(roster).toContainText('No sides yet.');

  await roster.getByPlaceholder('Armée du Nord').fill('Grande Armée');
  await roster.locator('input[type=color]').fill('#1f4e9c');
  await roster.getByRole('button', { name: 'Add side' }).click();

  const tab = roster.getByRole('tab', { name: /Grande Armée/ });
  await expect(tab).toBeVisible();
  await expect(tab.locator('.swatch')).toHaveCSS('background-color', 'rgb(31, 78, 156)');
  await expect(roster).not.toContainText('No sides yet.');
});
