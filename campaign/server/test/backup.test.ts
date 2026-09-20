/**
 * That a backup is the campaign, and not most of it.
 *
 * The failure this guards against is silent: a WAL database copied file-by-file yields
 * something that opens, queries, and is missing the last thing that happened. A test that
 * only checked the file existed would pass against exactly that bug.
 */

import { existsSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

import { backup } from '../src/backup.js';
import { DatabaseSync, openDb } from '../src/db.js';

const dir = () => mkdtempSync(join(tmpdir(), 'campaign-backup-'));

describe('backing a campaign up', () => {
  it('carries writes the running server has not checkpointed', async () => {
    const d = dir();
    const source = join(d, 'campaign.db');
    const db = openDb(source);
    db.exec(`INSERT INTO campaigns (id, name, world_hash, world_blob, created_at)
             VALUES ('c1', 'Austerlitz', 'h', '{}', '1805-12-02')`);

    // Still open, still in WAL mode, nothing checkpointed — which is the state a server
    // mid-evening is actually in, and the state a plain file copy gets wrong.
    expect(existsSync(`${source}-wal`)).toBe(true);

    const snapshot = join(d, 'snapshot.db');
    backup(source, snapshot);

    const restored = new DatabaseSync(snapshot);
    const rows = restored.prepare('SELECT id, name FROM campaigns').all();
    expect(rows).toEqual([{ id: 'c1', name: 'Austerlitz' }]);
    restored.close();
    db.close();
  });

  it('writes one file, with no sidecars to forget', async () => {
    const d = dir();
    const source = join(d, 'campaign.db');
    const db = openDb(source);
    db.exec(`INSERT INTO campaigns (id, name, world_hash, world_blob, created_at)
             VALUES ('c1', 'n', 'h', '{}', '1805-12-02')`);

    const snapshot = join(d, 'snapshot.db');
    backup(source, snapshot);

    expect(existsSync(snapshot)).toBe(true);
    expect(existsSync(`${snapshot}-wal`)).toBe(false);
    expect(existsSync(`${snapshot}-shm`)).toBe(false);
    db.close();
  });

  it('refuses a source that is not there, rather than inventing one', async () => {
    const d = dir();
    expect(() => backup(join(d, 'nothing.db'), join(d, 'out.db'))).toThrow(/no database/);
    expect(existsSync(join(d, 'nothing.db'))).toBe(false);
  });

  it('refuses to overwrite, so a typo costs nothing', async () => {
    const d = dir();
    const source = join(d, 'campaign.db');
    const db = openDb(source);

    const snapshot = join(d, 'snapshot.db');
    backup(source, snapshot);
    expect(() => backup(source, snapshot)).toThrow();
    db.close();
  });

  it('handles a path with a quote in it', async () => {
    const d = dir();
    const source = join(d, 'campaign.db');
    const db = openDb(source);
    db.exec(`INSERT INTO campaigns (id, name, world_hash, world_blob, created_at)
             VALUES ('c1', 'n', 'h', '{}', '1805-12-02')`);

    const snapshot = join(d, "the referee's copy.db");
    backup(source, snapshot);
    expect(existsSync(snapshot)).toBe(true);
    db.close();
  });
});
