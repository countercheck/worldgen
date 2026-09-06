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
same API a referee uses. The referee is handed one join link per side; send each
commander theirs.

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

## The two things worth knowing

**The fog is enforced, not drawn.** Exactly one function — `viewFor` in
`shared/src/view.ts` — turns campaign state into something a client may see, and no route
serialises state any other way. The WebSocket rebuilds a payload per socket rather than
broadcasting one, `assertMasked` re-checks at the boundary, and the leakage tests assert
against the serialised response body rather than the object graph, because a `toJSON` or
a field added later can put data on the wire that a structural assertion never looks at.
The role switcher in the console is not a filter: it swaps the token and refetches, so
looking at a commander's map means genuinely asking the server as that commander.

**The log is the campaign; everything else is cache.** State is the fold of an append-only
event log. Snapshots exist only so folding does not get slower forever, and a test deletes
them all and asserts the state rebuilds byte-identically. Rewind is replaying a prefix.
Dice outcomes are written into events rather than replayed from a seed, so a replay is
exact regardless of engine or floating-point behaviour, and the log reads as an account of
what happened.

## Known

A commander who has seen seven hexes still receives a 395 KB payload, because a masked
world keeps every hex — that is what holds the canvas the same size for every player, so
the maps overlay. The fix is to send the world once and patch it from the observation
events, which the log already makes straightforward.

A contact carries the observed unit's real id. Nothing else about the unit goes with it —
not its name, strength, arm or corps below the intelligence grade that earns them — but
the id is stable, so two sightings hours apart can be correlated as the same formation,
which the rules grade as top-level intelligence. Deciding what identity a sighting should
carry is a rules question rather than a plumbing one.
