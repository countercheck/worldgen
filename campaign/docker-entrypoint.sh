#!/bin/sh
# Hand the mounted volume to the user that will write it, then stop being root.
#
# The image creates `/data` and gives it to `node` at build time, but a platform volume is
# mounted *over* that path at start, arriving owned by root with the build-time ownership
# underneath it and invisible. The server's first act is to open `/data/campaign.db`, so an
# unprivileged process meets `SQLITE_CANTOPEN` before it has logged anything, and the
# container restarts forever while the platform reports a healthy deploy and a 502.
#
# So: root long enough to chown the mount, `node` for everything after. The server itself
# never runs privileged, which is the property that mattered about `USER node` in the first
# place. This is the pattern the official postgres and redis images use, for this reason.
set -e

DB_DIR=$(dirname "${CAMPAIGN_DB:-/data/campaign.db}")

if [ "$(id -u)" = "0" ]; then
  mkdir -p "$DB_DIR"
  # Not `-R`: the campaign may be large and this runs on every start. The directory and
  # the database files are what need owning, and SQLite's sidecars are created beside them.
  chown node:node "$DB_DIR"
  for f in "$DB_DIR"/*; do
    [ -e "$f" ] && chown node:node "$f"
  done
  exec su-exec node "$@"
fi

# Already unprivileged — a platform that sets its own user, or a `docker run --user`. The
# mount is either writable or it is not, and pretending otherwise would only hide it.
exec "$@"
