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
RUN mkdir -p /data && chown -R node:node /data

COPY --from=build --chown=node:node /app/node_modules ./node_modules
COPY --from=build --chown=node:node /app/shared/dist ./shared/dist
COPY --from=build --chown=node:node /app/shared/package.json ./shared/
COPY --from=build --chown=node:node /app/server/dist ./server/dist
COPY --from=build --chown=node:node /app/server/package.json ./server/
COPY --from=build --chown=node:node /app/client/dist ./client/dist
COPY --from=build --chown=node:node /app/package.json ./

USER node

ENV NODE_ENV=production \
    PORT=3000 \
    # 0.0.0.0 rather than the loopback default. Inside a container the loopback is
    # unreachable from outside it, so a published port would answer nothing at all — the
    # one place where binding every interface is the conservative choice rather than the
    # careless one.
    HOST=0.0.0.0 \
    CAMPAIGN_DB=/data/campaign.db \
    CAMPAIGN_CLIENT=/app/client/dist

# The campaign lives here. Without a volume it is inside the container's writable layer and
# vanishes with the container, which for an append-only log of somebody's evening would be
# a poor surprise.
VOLUME ["/data"]

EXPOSE 3000

# Answers before any campaign exists and needs no token, which is what makes it usable as a
# health check rather than as a second thing to configure.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
  CMD node -e "fetch('http://127.0.0.1:'+(process.env.PORT??3000)+'/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"

CMD ["node", "server/dist/index.js"]
