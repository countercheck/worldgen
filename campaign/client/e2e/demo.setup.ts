/**
 * One demonstration campaign for the run, made through the front page.
 *
 * The referee's link lands in browser storage exactly as it would for a person, and that
 * storage is what every later test opens the console with.
 */

import { mkdirSync, writeFileSync } from 'node:fs';

import { expect, test as setup } from '@playwright/test';

import { CAMPAIGN_FILE, saveCampaignPath, STATE_DIR, STORAGE } from './helpers.js';

setup('create the demonstration campaign', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: 'Or run the demonstration' }).click();
  const enter = page.getByRole('button', { name: 'Enter as referee' });
  await enter.waitFor({ timeout: 60_000 });

  // Every link copies in one press — five commanders and the referee — and so do all the
  // commanders' links at once. A link selected by hand gets sent short a character.
  const links = page.locator('.links li');
  await expect(links).toHaveCount(6);
  await expect(page.locator('.links .copy-button')).toHaveCount(6);
  await expect(page.locator('.links li').last()).toContainText('Referee');
  await expect(page.locator('.links')).toContainText('The Earl of Uxbridge');
  await page.context().grantPermissions(['clipboard-read', 'clipboard-write']);
  await page.getByRole('button', { name: 'Copy all as one message' }).click();
  await expect(page.getByRole('button', { name: 'Copied' })).toBeVisible();
  const pasted = await page.evaluate(() => navigator.clipboard.readText());
  expect(pasted).toContain('Marshal Ney: http');
  expect(pasted).not.toContain(await page.locator('.links li').last().locator('code').innerText());

  await enter.click();
  await expect(page).toHaveURL(/#\/c\//);
  await page.locator('.map canvas').first().waitFor();

  mkdirSync(STATE_DIR, { recursive: true });
  await page.context().storageState({ path: STORAGE });
  writeFileSync(CAMPAIGN_FILE, saveCampaignPath(`/${new URL(page.url()).hash}`));
});
