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
  await page.getByRole('button', { name: 'Enter as referee' }).click({ timeout: 60_000 });
  await expect(page).toHaveURL(/#\/c\//);
  await page.locator('.map canvas').first().waitFor();

  mkdirSync(STATE_DIR, { recursive: true });
  await page.context().storageState({ path: STORAGE });
  writeFileSync(CAMPAIGN_FILE, saveCampaignPath(`/${new URL(page.url()).hash}`));
});
