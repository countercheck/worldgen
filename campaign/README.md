# Campaign

A refereed Napoleonic campaign layer over a generated `world.json`. Divisions are columns
several kilometres long rather than counters on a hex, time is a continuous clock of hours,
and each commander is sent only the ground their own troops have observed.

The rules are the Napoleonic Campaign Rules (1 km hexes) v4 — a simplification of
*Vol de l'Aigle* whose subject is fog of war. Every number from that document lives in
`shared/src/config.ts`, so a rules revision is a config edit.

## Running it

```bash
npm install
npm run dev          # server on :3000, client on :5173
```

Open <http://localhost:5173>. Upload a `world.json`, or click through to the
demonstration, which creates a campaign and posts the scenario's commands through the
same API a referee uses. The referee is then handed **one join link per seat** — a link
names a person, not a side — and sends each commander their own.

```bash
npm run build
CAMPAIGN_DB=./campaign.db node server/dist/index.js
```

One process and one SQLite file. A database that is a file is one a referee can back up
by copying it.

Generate a world with `--model organic`: `CrossingStage` runs only in that model, so a
classic world carries no ford or bridge tags and every major river is impassable.

## Layout

```
shared/    The engine: types, hex math, rules, the event reducer, masking. No I/O.
server/    Node: HTTP and WebSocket, the SQLite event log, and the fog authority.
client/    Browser: a canvas map and a sidebar, and nothing the server did not send.
```

`shared` runs on both sides. The server runs it over ground truth; the client runs the
same code over the masked world it was given, so a commander's reach preview is computed
from the country they actually know about.

## The three things worth knowing

**Every formation has a commander, and they see through the one they ride with.** A role is
a seat, not a side: two commanders on the same side see different wars. Their own formation is
live; everything else beneath them is a dated report, and the enemy is wherever somebody
last said they saw them. `superiorId` carries the whole chain of command, and writing past
a subordinate leaves that commander with a confidently wrong picture of their own corps — which
nothing had to be built to achieve.

**The fog is enforced, not drawn.** Exactly one function — `viewFor` in
`shared/src/view.ts` — turns campaign state into something a client may see, and no route
serialises state any other way. The WebSocket rebuilds a payload per socket rather than
broadcasting one, `assertMasked` re-checks at the boundary, and the leakage tests assert
against the serialised response body rather than the object graph, because a `toJSON` or
a field added later can put data on the wire that a structural assertion never looks at.
The seat switcher in the console is not a filter: it swaps the token and refetches, so
looking at a commander's map means genuinely asking the server as that commander.

Terrain fog is **off** by default (`terrainFog` in config). The ground is public and
accurate, because the tension the period turns on is where the enemy is and where one's
own III Corps is, not what the country looks like — and with sight limited to one
formation, a blacked-out map is unplayable. The masking machinery is unchanged and still
tested in both positions; it returns with the issued map. See `PLAN.md`.

**Formations are drawn as NATO symbols, and the standard does the fog for you.** A frame
with nothing in it means something is there and you cannot say what — which is exactly a
plain sighting at intel 2. The arm appears inside once a patrol has closed enough to earn
it, the size marks appear once you know roughly how big it is, and a dashed frame means a
position reported rather than observed. So a symbol degrades precisely as the intelligence
does, and a reader who knows the standard can tell how good a report is without reading a
word. Behind each symbol runs the ground its column occupies, because a division is two to
eighteen kilometres of road rather than a counter on a hex.

**The log is the campaign; everything else is cache.** State is the fold of an append-only
event log. Snapshots exist only so folding does not get slower forever, and a test deletes
them all and asserts the state rebuilds byte-identically. Rewind is replaying a prefix.
Dice outcomes are written into events rather than replayed from a seed, so a replay is
exact regardless of engine or floating-point behaviour, and the log reads as an account of
what happened.

## Known

**Reports are instantaneous.** Every formation reports to its superior the moment it sees
anything, because there are no despatch riders yet. That is a stated interim rule rather
than a leak — the reports are real, dated and attributed; they simply travel at infinite
speed. Riders and delay are the next step, and nothing else about the model changes.

**The whole world is sent to everybody**, which is now correct rather than wasteful, since
the ground is public. It becomes a real cost again if terrain fog is ever turned on, and
the fix is the same one: send the world once and patch it from the survey events.

**A contact carries the observed unit's real id.** Nothing else goes with it — not its
name, strength, arm or corps below the intelligence grade that earns them — but the id is
stable, so two sightings hours apart can be correlated as the same formation, which the
rules grade as top-level intelligence. Moving contacts from computed to reported, in the
next step, is where that gets fixed: an id becomes the observer's label rather than the
observed unit's.
