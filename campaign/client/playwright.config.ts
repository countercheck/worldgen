/**
 * The console in a real browser.
 *
 * The vitest suites cover everything that can be worked out without a page: the view, the
 * board, the gestures, which tabs a role gets. What they cannot see is layout — which
 * pane is on screen at 390 pixels wide, whether a sheet is over the map, whether a banner
 * has a button in it — because the phone layout is CSS, and CSS needs a browser to mean
 * anything. These tests are for that.
 *
 * They run against the built server serving the built client, one process and a SQLite
 * file thrown away afterwards, which is how the thing is deployed. Build first:
 * `npm run e2e` from `campaign/` does both.
 */

import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { defineConfig } from '@playwright/test';

import { STORAGE } from './e2e/helpers.js';

const here = fileURLToPath(new URL('.', import.meta.url));
const PORT = 4173;

export default defineConfig({
  testDir: './e2e',
  // `.e2e.ts`, never `.test.ts`: vitest picks up anything named that way and would try to
  // run these without a browser.
  testMatch: /.*\.(e2e|setup)\.ts$/,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? 'github' : 'list',
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    browserName: 'chromium',
    trace: 'retain-on-failure',
  },
  projects: [
    // One demonstration campaign for the whole run, created the way a referee creates
    // one. Every test after it opens that campaign with the referee's stored link.
    { name: 'setup', testMatch: /demo\.setup\.ts$/ },
    {
      name: 'phone',
      testMatch: /phone\.e2e\.ts$/,
      dependencies: ['setup'],
      use: {
        viewport: { width: 390, height: 844 },
        deviceScaleFactor: 2,
        isMobile: true,
        hasTouch: true,
        storageState: STORAGE,
      },
    },
    {
      name: 'desktop',
      testMatch: /desktop\.e2e\.ts$/,
      dependencies: ['setup'],
      use: { viewport: { width: 1400, height: 860 }, storageState: STORAGE },
    },
  ],
  webServer: {
    command: 'node ../server/dist/index.js',
    url: `http://127.0.0.1:${PORT}/`,
    reuseExistingServer: !process.env.CI,
    env: {
      PORT: String(PORT),
      HOST: '127.0.0.1',
      CAMPAIGN_DB: join(tmpdir(), `campaign-e2e-${Date.now()}.db`),
      CAMPAIGN_CLIENT: join(here, 'dist'),
      CAMPAIGN_RATE_LIMIT: 'off',
    },
  },
});
