# Deploying the campaign server

One service, one volume, one replica, on Railway. The shape is not a simplification: the
campaign is a SQLite file only one process may write, and the WebSocket listeners that
push each player their view live in that process's memory. A second replica is not more
capacity — it is a second campaign that half the players are connected to.

## What lives where

| File | What it decides |
|---|---|
| `.railway/railway.ts` | The service, its volume, its domain, how the image is built, how a deploy behaves, and the environment the process reads |
| `campaign/Dockerfile` | The image. Nothing here depends on Railway |

Railway does not read `.railway/railway.ts` on deploy. A push to `master` builds and
deploys the code; a change to the file reaches the service only when it is applied:

```bash
npm install --prefix .railway   # once: the SDK the file imports
railway config plan             # what would change, touching nothing
railway config apply            # make it so, after a confirmation
```

So a merged change to the file is not live until someone applies it, and a setting
changed in the Railway console is reverted by the next apply. Run `plan` before either.

Two project settings the file cannot state, which were set when the project was made
and must stay that way: the project is **private** (the join links are the whole security
model), and **pull-request environments are off** (each would want its own volume, and a
throwaway environment holding a copy of somebody's campaign is not a thing to create by
accident).

## The settings that are not obvious

- **`overlapSeconds: 0`.** Railway would otherwise start the new container before stopping
  the old one, which is right for a stateless service and wrong for this one: two processes
  holding the same SQLite file is the one thing the design does not survive. The cost is a
  few seconds of downtime per deploy, during an evening nobody is playing.
- **`sleepApplication: false`.** A sleeping service drops WebSockets, and a player whose
  map stops updating has no way to tell that from a quiet turn.
- **`requiredMountPath: /data`.** Refuses to start without the volume rather than starting
  and writing a fresh campaign into the container's own filesystem, where it would look
  fine until the next deploy.
- **`TRUST_PROXY=true`.** Railway terminates TLS and forwards; without this every player
  shares one rate-limit bucket and the first one spends it.

## Backups

The volume is the campaign. `docs/CAMPAIGN.md` says a referee backs one up by copying a
file, which is true of the campaign and not quite true of the file: in WAL mode the
committed state is spread across `campaign.db` and `campaign.db-wal`, and copying the
first alone gets whatever was last checkpointed — a campaign missing its most recent
evening, which opens cleanly and so is not noticed until it is restored.

```bash
railway run node server/dist/backup.js /data/campaign.db /data/backup-$(date +%F).db
```

`VACUUM INTO` writes one file holding the committed state at one instant, with no sidecars
and no pause in play. Copy it off the volume afterwards — a backup on the same disk as the
original is a copy, not a backup.

## Changing a limit

The upload ceiling and the rate budgets are the `env` block in `.railway/railway.ts`,
with why each is what it is. Edit, `railway config plan`, `railway config apply`; do not
set them in the Railway console, where the next apply would silently revert them.

## Moving off Railway

Nothing in the image is Railway's. What a new host has to provide is the list below, and
`.railway/railway.ts` states each item in one place:

- **One container, one replica, never two at once.** No rolling deploy that starts the
  new container before the old one stops (`overlapSeconds: 0`), and no sleeping.
- **A persistent volume at `/data`**, and a refusal to start without it.
- **The image built from the repository root** with `campaign/Dockerfile`, because the
  client imports a world fixture from `shared/test/fixtures`.
- **A health check on `/health`**, and a restart when the process fails.
- **The four variables** in the `env` block, with `TRUST_PROXY` only if something in
  front terminates TLS and forwards.

Until 2026-09-23 this was described in OpenTofu, against the community Railway provider.
For a host with a Terraform provider, that is a starting point for the same shape:

```bash
git show 9db0e2a:deploy/main.tf
git show 9db0e2a:deploy/variables.tf
```
