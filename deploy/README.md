# Deploying the campaign server

One service, one volume, one replica, on Railway. The shape is not a simplification: the
campaign is a SQLite file only one process may write, and the WebSocket listeners that
push each player their view live in that process's memory. A second replica is not more
capacity — it is a second campaign that half the players are connected to.

## What lives where

| File | What it decides |
|---|---|
| `deploy/*.tf` | The project, the service, the volume, the domain, and the environment the process reads |
| `railway.json` (repo root) | How the image is built and how a deploy behaves. Railway reads this itself |
| `campaign/Dockerfile` | The image. Unchanged by any of this |

Two files rather than one because they answer to different readers: OpenTofu creates the
service and can be reviewed in a pull request, while `railway.json` is read by Railway at
build time and has to sit where Railway looks for it.

## First deploy

```bash
cd deploy
export RAILWAY_TOKEN=...        # an account token, not a project token
tofu init
tofu apply
```

`tofu apply` prints the URL. Everything after that is a push to the deployed branch.

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

The budgets and the upload ceiling are variables in `variables.tf`, applied as environment
variables. Edit and `tofu apply`; do not set them in the Railway console, where the next
apply would silently revert them.
