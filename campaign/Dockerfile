# The campaign server, and the client it serves.
#
# One image, one process, one port, one file on disk. A referee running a game for five
# people should be able to start this and hand out links, and back the whole campaign up by
# copying a file — anything more elaborate would be infrastructure nobody asked for.
#
#     docker build -t campaign -f campaign/Dockerfile .
#     docker run -p 3000:3000 -v campaign-data:/data campaign
#
# The build context is the repository root rather than `campaign/`, because the client
# imports a world fixture from `shared/test/fixtures` for its demonstration scenario. See
# `.dockerignore` for what is kept out.

# ---- build ------------------------------------------------------------------
#
# Node 22 is a floor rather than a preference: storage is `node:sqlite`, which does not
# exist before it. That choice is itself deliberate — see `server/src/db.ts` — because the
# alternative compiles on install and a referee self-hosting this should never meet
# node-gyp.
FROM node:22-alpine AS build

WORKDIR /app

# The manifests first, so a change to the source does not invalidate the install layer.
# `npm ci` needs every workspace's manifest present or it refuses the lock file.
COPY campaign/package.json campaign/package-lock.json ./
COPY campaign/shared/package.json ./shared/
COPY campaign/server/package.json ./server/
COPY campaign/client/package.json ./client/

# `npm ci` and not `npm install`: it installs exactly what the lock file says and fails if
# the lock and the manifests disagree. A gate that moves on its own is not a gate.
RUN npm ci

COPY campaign/ ./

# Builds all three workspaces: the shared engine to `dist/` — which the server imports by
# package name — and the client to a bundle of static files.
RUN npm run build

# The tests are not run here. CI runs them on every push, and a container build that also
# ran them would be slower for no extra confidence: the image is built from a commit whose
# suite has already been green.

# Reinstall with dev dependencies dropped, so vite, vitest, eslint and typescript do not
# travel to the runtime image. `--ignore-scripts` because nothing here has an install step
# and running arbitrary ones in a build is a habit worth not having.
RUN npm ci --omit=dev --ignore-scripts

# ---- runtime ----------------------------------------------------------------

FROM node:22-alpine AS runtime

# Unprivileged, and the node image already ships a `node` user, so this is one line rather
# than a useradd. The data directory is created and handed over before the drop, because
# afterwards there is no permission to create it.
WORKDIR /app

# `su-exec` is how the entrypoint stops being root. Three lines of C in the base image's
# own repository, and the alternative is `gosu`, which is a Go binary an order of magnitude
# larger for the same one job.
RUN apk add --no-cache su-exec && mkdir -p /data && chown node:node /data

COPY --from=build --chown=node:node /app/node_modules ./node_modules
COPY --from=build --chown=node:node /app/shared/dist ./shared/dist
COPY --from=build --chown=node:node /app/shared/package.json ./shared/
COPY --from=build --chown=node:node /app/server/dist ./server/dist
COPY --from=build --chown=node:node /app/server/package.json ./server/
COPY --from=build --chown=node:node /app/client/dist ./client/dist
COPY --from=build --chown=node:node /app/package.json ./

# Deliberately *not* `USER node`, though the server runs as `node` all the same.
#
# A platform volume is mounted over `/data` at start, owned by root, hiding the ownership
# set above. A container that had already dropped privileges could not fix that and would
# die on its first write. So the entrypoint starts as root, chowns the mount, and execs the
# server as `node` — see `docker-entrypoint.sh`. Nothing that serves a request is
# privileged, which is what dropping to `node` was for.
COPY --chown=root:root campaign/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod 0755 /usr/local/bin/docker-entrypoint.sh

ENV NODE_ENV=production \
    PORT=3000 \
    # 0.0.0.0 rather than the loopback default. Inside a container the loopback is
    # unreachable from outside it, so a published port would answer nothing at all — the
    # one place where binding every interface is the conservative choice rather than the
    # careless one.
    HOST=0.0.0.0 \
    CAMPAIGN_DB=/data/campaign.db \
    CAMPAIGN_CLIENT=/app/client/dist

# No `VOLUME ["/data"]`, though `CAMPAIGN_DB` points inside it and the campaign lives
# there. Railway refuses to build an image that declares one — "docker VOLUME at Line 83
# is not supported, use Railway Volumes" — because it attaches its own, and a Dockerfile
# that asks for a second one is ambiguous about which the data is in.
#
# So the mount is the caller's job, and both callers do it: `docker run -v campaign-data:/data`
# as `docs/CAMPAIGN.md` gives it, and `requiredMountPath` in `.railway/railway.ts`, which refuses
# to start the container at all without the volume. What is lost is the anonymous volume
# Docker would have created for someone who forgot the flag — a safety net that mostly
# produced volumes nobody could name afterwards.

EXPOSE 3000

# Answers before any campaign exists and needs no token, which is what makes it usable as a
# health check rather than as a second thing to configure.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
  CMD node -e "fetch('http://127.0.0.1:'+(process.env.PORT??3000)+'/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["node", "server/dist/index.js"]
