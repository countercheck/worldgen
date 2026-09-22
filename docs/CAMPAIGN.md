# The campaign layer

A refereed Napoleonic campaign played on a generated world. One referee runs the clock;
each commander is sent a link and sees only what their own troops have told them.

The ruleset is the **Napoleonic Campaign Rules (1 km hexes) v4**, a simplification of
*Vol de l'Aigle*. `worldgen` generates at 1 hex = 1 km and the rules are written at 1 hex =
1 km, so there is no conversion factor anywhere: the speed table is in km/h, which on this
grid is also hexes per hour.

- **Where it lives:** `campaign/`, a TypeScript workspace of three packages.
- **What it is not:** a battle game. Combat is deferred; this is the layer above it —
  marching, finding the enemy, and getting orders to people.

---

## What the game is about

Almost everything here exists to make one thing true: **a commander does not know where
anything is.**

| They can see | How |
|---|---|
| The formation they ride with | It is in front of them |
| The ground | Everybody has a map — terrain fog is off by default |
| Their own corps | Where they *last heard* it was, by rider, hours ago |
| The enemy | Where somebody saw a column, labelled by their own staff |
| Whether their orders arrived | Only if somebody writes back |

That last row is the mechanic rather than a missing feature. A despatch takes real time,
can be intercepted, and its sender is never told what became of it.

### Orders are prose

The engine executes nothing on its own. A commander writes *"Move on Quatre Bras with all
speed"*; the referee reads it, decides what that commander's subordinate makes of it, and sets a
march. Two consequences worth knowing before playing:

- **A captured despatch is intelligence**, not a coordinate dump.
- **The referee's evening is reading post and pointing at ground.** The console is built
  around making that one motion — the arriving despatch and the control that sets the
  addressee's task are on the same card.

---

## Running it

### From source, for development

Two processes, because that is what gives hot reloading. Vite serves the client and
proxies `/api` to the server, so everything is same-origin and there is no CORS story.

```bash
cd campaign
npm ci
npm run dev            # client on :5173, server on :3000
```

Open <http://localhost:5173>. The front page can build a demonstration campaign with a
world it carries, so there is something to look at before you have generated anything.

### In a container, for actually playing

One image, one process, one port, one file on disk.

```bash
docker build -t campaign -f campaign/Dockerfile .
docker run -p 3000:3000 -v campaign-data:/data campaign
```

Open <http://localhost:3000>. The volume is the campaign: an append-only event log in
SQLite.

Back it up with `VACUUM INTO` rather than by copying the file. The database runs in WAL
mode, so the committed state is spread across `campaign.db` and `campaign.db-wal`, and
copying the first alone gets whatever was last checkpointed — a campaign missing its most
recent evening, which opens cleanly and so is not noticed until it is restored.

```bash
node server/dist/backup.js /data/campaign.db /data/backup-2026-09-20.db
```

For hosting it somewhere rather than running it at home, see `deploy/README.md`.

| Variable | Default | What it does |
|---|---|---|
| `PORT` | `3000` | |
| `HOST` | `127.0.0.1` | The image sets `0.0.0.0`, because inside a container the loopback answers nothing |
| `CAMPAIGN_DB` | `./campaign.db` | The image sets `/data/campaign.db` |
| `CAMPAIGN_CLIENT` | unset | Where the built client is. Unset means "do not serve it" |
| `CAMPAIGN_BODY_LIMIT` | `16777216` | The world upload ceiling, in bytes. A 32x32 world is 641 KB and a 64x64 about 2.5 MB |
| `TRUST_PROXY` | unset | `true` behind a reverse proxy, or a comma-separated list of addresses to believe. Without it every client behind the proxy shares one rate-limit bucket |
| `CAMPAIGN_RATE_LIMIT` | on | `off` disables rate limiting entirely |
| `CAMPAIGN_RATE_GLOBAL` | `600` | Requests per minute per address |
| `CAMPAIGN_RATE_CREATE` | `5` | Campaign creations per minute per address |

---

## Setting a campaign up

1. **Generate a world.** Use `--model organic`: the crossing stage only runs in that
   model, so a `classic` world has no bridge or ford tags and every major river becomes
   impassable.

   ```bash
   python3 -m worldgen.cli generate --seed 42 --width 64 --height 64 \
       --model organic --output-dir ./output
   ```

   The world lands at `./output/world.json`, alongside the rendered plates. At 64x64 that
   file is about 2.5 MB, which is why `CAMPAIGN_BODY_LIMIT` is what it is.

2. **Create the campaign** in the browser, uploading `./output/world.json`. You are handed the
   referee's link, and it is shown once — the token is stored only as a hash, so a lost
   link is reissued rather than recovered.

3. **Put formations on the map, and appoint troops to them.** A join link names a *seat*, and
   there are no seats until there are commanders, which is also the order a real game is
   prepared in.

4. **Issue a link per commander** somebody is going to play, and send it to them.

Join links are URL fragments — `#/j/<campaign>/<token>` — so the token never reaches a
server log, a proxy log, or anybody's browser history in a form that gets sent anywhere.

---

## Running an evening

The referee's controls, in the order they get used:

- **Run** — advance until something needs a decision, rather than guessing at an interval
  and finding out afterwards that two corps met each other ninety minutes in.
- **The queue** — what stopped the clock, and who it belongs to. Each row can march its
  formation somewhere or be marked dealt with. A player's note to the referee lands here
  too, at once and out of the game.
- **The post** — every despatch, with routes and fates. The only place either is visible.
- **Seat switching** — a referee holds every link, so they can read the game through any
  commander's eyes and adjudicate from *that commander's* information rather than from the map.

A commander's screen is their post, their formations with an hour against each, and the ground.
They write prose, forward what somebody else should see, and can write to the referee directly
when they need a ruling.

---

## The rules, as implemented

Everything below is config, in `shared/src/config.ts`. The ruleset is still moving, so a
revision should be an edit rather than a build.

**Movement** is a flat km/h lookup over four terrain grades — Highway, Road, Off-road, Bad
Going — with no gradient term and no movement points. A road edge decides the grade;
`PRIMARY` is Highway and everything else metalled is Road.

**A division is a column, not a counter.** At its own spacing it is between about 2.6 and
18 km of road, which on this grid is that many hexes. It is exposed along all of them, its
recon zone is measured from all of them, and its tail takes `columnLength / speed` hours to
close up. This is where most of the interesting behaviour comes from.

**Rivers are two-tier.** Major means navigable, using the same discharge threshold the
generator used to decide what to ship grain down; crossing needs a bridge, or pontooneers.
Minor is fordable at an hour per division. Delay is charged per division, so a corps queues
at a bridge.

**Reconnaissance** is one hex from the column, two with Scout. No line of sight, no
occlusion — a ridge hides nothing. The rules ask for terrain to mute the sound of guns and
that one clause is not modelled, being the only place the ruleset wants occlusion.

**Couriers** ride at 10 km/h on roads, 24 hours a day, taking the least-*time* path. Passing
an enemy column they throw a die, plus one for cavalry, one for a scouting formation, one
for a division: a single `1` stops the rider, two lose the paper to the enemy. The dice are
recorded in the event, so a referee can show them.

**Formations whose columns touch** hand paper over instantly and cannot be intercepted,
which is the mechanical reward for concentrating.

**Who may be written to.** A commander's direct superior, their direct subordinates, and
anyone on their own side whose column is inside their recon zone. Nobody else: a corps out
of sight is reached through the common superior, one link at a time, which is slow on
purpose. There is one kind of despatch, so an order is a despatch with orders in it. Nothing
passes itself down the chain: a despatch arriving at a commander nobody is playing goes to
the referee's queue like any other. This rule is hard, so not even the referee can force a
despatch past it.

---

## How it is built

```
campaign/
  shared/   the engine: types, hex math, rules, the reducer. No I/O, no DOM.
  server/   Fastify + SQLite. The fog authority.
  client/   Vite + React + Canvas. Draws what it is sent and nothing else.
```

### The fog is server-side, and that shapes everything

If the client received the world and filtered it for display, anyone with developer tools
would have the whole map. So there is exactly one function that turns state into something
a client may see — `viewFor` — and no route may reach for state directly.

The tests that matter most are `server/test/leakage.test.ts`. They assert against the
**serialised response bytes** rather than the object graph, because a getter or an
accidentally-enumerable field sails straight past a structural assertion, and they assert
*absence*: not that Ney can see their own division, but that Wellington's is nowhere in the
payload.

Three redactions are load-bearing enough to name:

- **A despatch's route** is the path to where its addressee actually is, so showing a
  sender their own rider's route would tell them exactly where their detached corps stands. It
  is referee-only, and so is anything derived from it — a delivery estimate is a distance,
  and a distance is a position.
- **A despatch's fate** is never shown to its sender. They learn it arrived only if the
  addressee writes back, and that is another despatch, which can also be lost.
- **A contact never names the formation it is a sighting of.** It carries the observing
  commander's own label, so two sightings hours apart cannot be correlated for free — that
  correlation is what these rules make you buy with a patrol.

### Event sourcing

A campaign is an append-only log; state is the fold of it. Snapshots are cache and deleting
them is a valid recovery.

```
check (cmd, state, world) -> Violation[]      pure, mutates nothing
decide(cmd, state, world, rng) -> Event[]     pure given rng
reduce(state, event) -> state                 pure, total, no rng
```

**Dice are recorded, not re-rolled.** An interception event carries the dice that decided
it, so replay reproduces a campaign exactly regardless of engine or platform, and the log
reads as an account of what happened rather than a seed nobody can interpret.

That buys the referee rewind, after-action review, and an audit of every rule that was
bent — which is the other half of the design: **hard and soft violations are not the same
axis as strict and open.** A soft violation is a rule of the game and a referee may bend
any of them; a hard one is a rule of arithmetic — a hex that is not on the map, a unit that
does not exist — and is refused at every setting, including `force`. That is what makes
permissive bookkeeping a safe mode rather than an unvalidated one.

### Testing

```bash
cd campaign
npm run typecheck      # tsc --build. The strict flags catch real fog bugs.
npm run lint
npm run build          # required before the tests: the suites import @campaign/shared
npm test               # by its built dist/
```

CI runs all four on every push, and the gate that branch protection points at waits for
both languages.

---

## What is deferred

Combat, fatigue accumulation, provisions and equipment consumption, depots and convoys, the
terrain muting of gunfire, and sub-commander personalities. Every stat field is on `Unit`
already and the midnight tick is scheduled, so these are additions rather than migrations.

The largest deferred piece is the **issued map**: a per-faction survey, generated once, that
is accurate about coastlines, elevation, major rivers and highways and *wrong* about minor
roads, fords, river courses and the exact placement of villages. It turns terrain fog from
darkness into doubt, which is the version worth having — a column reaches where the bridge
should be and finds a river. The masking machinery it needs is built and still tested;
`terrainFog` is the switch.
