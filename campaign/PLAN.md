# Campaign phase 2: commanders and despatches

Phase 1 is built: the engine, the event log, the server, server-enforced fog, and a
console wired to it. What follows turns it from a map with units on it into the game the
ruleset is actually about — a commander who knows where almost nothing is, and learns by
despatch rider.

## Decisions taken

| Question | Decision |
|---|---|
| Who has a commander | **Every formation.** Most are run by the referee; some are players |
| What a commander sees | Through the formation he rides with |
| Chain of command | A tree of commanders, not a list of units |
| Terrain fog | **Off.** The ground is fully and accurately visible to everyone |
| What is hidden | Enemy positions, your own detached formations, despatches in transit |
| A despatch's fate | **Never known to the sender.** Only an acknowledgement tells him |
| A despatch's route | **Referee only.** See below — this one is load-bearing |
| A commander's inbox | **Delivered only.** Nothing in transit toward him is visible |
| Rider routing | Least-time path to the target's **actual** position |
| Rider routing, overridden | The sender may insist on waypoints — around pickets, say |
| Couriers available | **Unlimited.** Time and interception are the whole cost |
| An NPC commander's orders | **Cascade to subordinates immediately**, referee may override |
| A unit with no new orders | Continues its last one, until it discovers something |
| On discovery | Stops and asks. The referee decides |
| Reports on contact | Automatic, and free between formations whose columns touch |

### Why terrain fog is off

It is the least interesting fog in the design. The tension the period turns on is *where
is the enemy* and *where is my own III Corps*, not *what does the ground look like* — a
commander in 1815 had a map. And with sight limited to the formation a commander rides
with, terrain fog would leave him a two-hex bubble of ground and black everywhere else,
unable to plan a march at all.

**The machinery stays, switched off.** `terrainFog` is campaign config defaulting to
false; `maskWorld`, the fog tags, and the Python renderers' fog handling are unchanged and
still tested in both positions. It returns with the issued map (deferred, below), as a
switch rather than a rebuild.

**Observation keeps being recorded even though nothing masks on it.** The `hexes_revealed`
events keep flowing into the log. They cost almost nothing, and when the issued map
arrives it will want the history of who surveyed what — which an event-sourced system gets
for free if the events were written all along, and cannot reconstruct if they were not.

---

## Commanders

**Every formation has a commander.** Most are run by the referee; some are held by players
via a join link. There is no structural difference — a player taking over a division
mid-campaign is handed a token, and nothing else changes.

```ts
interface Commander {
  readonly id: string
  readonly name: string             // "Marshal Ney"
  readonly faction: string
  /** The formation he rides with: his position, his eyes, and where riders find him. */
  readonly unitId: string
  /** Who he answers to. The chain of command is this relation and no other. */
  readonly superiorId: string | null
  /** Passes an arriving order straight down. True for referee-run, false for players. */
  readonly autoCascade: boolean
}
```

Two fields carry the whole hierarchy. A commander may order anyone beneath him in the
tree. "A human who also commands the corps" needs no special case: he is the commander of
1re Division — `unitId` points at it — and the other divisional commanders have him as
their `superiorId`. One man, two appointments, one record.

`Unit.corps` becomes a label rather than a structure; the chain is derived from the tree.

### Units observe; commanders know

What is seen is a property of the formation — its column, its scouts, the ground it
covers. What is *known* is a property of the man. So observation is computed from the unit
and recorded against its commander:

```ts
interface CommanderKnowledge {
  readonly commanderId: string
  /** Enemies his own formation has seen, plus every one reported to him. */
  readonly contacts: ReadonlyMap<ContactId, Contact>
  /** Where he last heard each formation under him was. */
  readonly reports: ReadonlyMap<string, UnitReport>
  readonly surveyed: ReadonlySet<HexKey>   // recorded, not yet consumed
}
```

This replaces `CampaignState.knowledge` keyed by faction, which is the assumption
everything else is blocked on. Unpicking it is contained: `recon.ts`, `state.ts`,
`observe.ts`, `view.ts`.

### Your own corps is where you last heard it was

Only the formation a commander rides with is live to him. Everything else beneath him is a
dated snapshot:

```ts
interface UnitReport {
  readonly unitId: string
  readonly atHours: number
  readonly head: Hex
  readonly effectives: number
  readonly fatigue: number
  readonly formation: Formation
  readonly provisions: number
}
```

With terrain fog gone this is *the* fog of the game rather than one layer of three.

### What the tree gives away for free

A commander may write past a subordinate — Napoleon did it constantly — and when he does,
**the skipped commander does not know.** A corps commander whose division has been ordered
away by the army commander has a stale, confidently wrong picture of his own corps, and
nothing had to be built to make that happen.

The referee gains something too. Every NPC formation has a commander with a real, limited
view, so adjudicating what III Corps does on contact is done **from III Corps' own
information** rather than from omniscience exercised with restraint.

---

## Step 1 — Despatches

**One entity, four ledgers.** A despatch is an order, a report or an acknowledgement.
Every one runs commander to commander; the unit is only the address.

```ts
interface Despatch {
  readonly id: string
  readonly kind: 'order' | 'report' | 'acknowledgement'
  readonly from: string                  // commander id
  readonly to: string                    // commander id
  readonly sentAtHours: number
  readonly body: Order | Report
  /** Waypoints the sender insists on — around a wood he thinks holds enemy pickets. */
  readonly via?: readonly Hex[]
  /**
   * The rider's path, least-time through `via` to the addressee's ACTUAL position.
   *
   * REFEREE ONLY. See below: this field is the one place in the design where a leak
   * would be invisible and total.
   */
  readonly route: readonly Hex[]
  readonly inReplyTo?: string
}
```

| Ledger | Derived from | Visible to |
|---|---|---|
| Orders I have sent | despatches from me, kind `order` | me — never their fate or route |
| Orders I have received | despatches to me, kind `order`, **delivered** | me |
| Reports I have sent | despatches from me, kind `report` | me — never their fate or route |
| Reports I have received | despatches to me, kind `report`, **delivered** | me |

A commander sees his own outbox in full, because he wrote it. What he never learns is
whether any of it arrived.

### The route is the leak

Riders take the least-time path to the target's **actual** position, so the route is
computed from ground truth. Which means **the route betrays the target's location**. Show
a commander where his rider went and you have told him exactly where his detached corps
is, destroying the one thing this whole design exists to model.

So `route` is referee-only, absolutely. A commander sees *sent 0400, to Ney, via the
Charleroi road* — his own `via`, which he chose — and never a hex of what the rider
actually did. The route exists server-side to resolve interception and arrival, and for no
other purpose.

This is the same class of mistake as fabricating a `Unit` for an enemy contact so the map
had something to draw: a field that has to exist for the mechanic to work, sitting one
careless `send` away from ending the game. It gets a leakage test of its own.

### The rule that makes it a fog-of-war mechanic

**A sender never learns the fate of a despatch.** `courier_intercepted` is an event, and
the moment a commander's log renders events he knows his order was taken. So fate is
masked in `viewFor` exactly as a hex used to be: the sender sees *sent 0400, no
acknowledgement*, and nothing more, ever, unless an acknowledgement arrives.

An acknowledgement is itself a despatch and can itself be lost. That recursion is free
once despatches are one type, and it is most of the tension in the period.

```ts
type Fate =
  | { kind: 'in_transit' }
  | { kind: 'delivered';    atHours: number }
  | { kind: 'intercepted';  by: string; dice: readonly number[] }
  | { kind: 'lost' }
  | { kind: 'captured';     by: string; atHours: number }
```

Masking, per role: the **sender** gets everything but `fate` and `route`. The **addressee**
sees it only once delivered. A **captor** sees the body and that he took it. The
**referee** sees all of it, which is what a referee is for.

With terrain fog off this is the primary thing the fog architecture protects, and the
leakage tests move with it: assert against the serialised body that a commander's payload
carries no undelivered despatch's fate, no despatch route, no enemy unit record, and no
other commander's traffic.

### Interception now carries the whole weight

Riders are unlimited, they always find their man, and NPC orders cascade without delay.
Travel time and interception are therefore the *only* things that make command imperfect —
so if the game turns out to feel too reliable, the interception rules are the dial, not
the courier economy. Worth knowing before it is tuned in the wrong place.

### Contacts become reported, not computed

`spotted()` recomputes contacts live from current positions, which under this model is
wrong twice: it hands a commander what his divisions can see this instant, with no rider
involved. Contacts are recorded when a **formation** observes them, and reach a superior
only by report.

That also closes the leak found after the last commit. A contact id becomes the observer's
own label — *the column reported by 1re Division at 0400* — rather than the observed
unit's engine id, so two sightings can no longer be correlated for free.

### Free traffic between neighbours

Orders and reports between two commanders whose **formations touch** are instantaneous and
cannot be intercepted. Touching means any hex of one `occupied()` set is adjacent to any
hex of the other — generous, and right, because a rider covers intermingled baggage in
minutes.

It gives concentration a mechanical reward, which is the tension the period turns on:
concentrate and command well, disperse and forage well.

### Cascading

An NPC commander passes an arriving order straight down to his subordinates, immediately,
as fresh despatches with fresh riders — so the copies can still be intercepted
individually even though the decision cost nothing. The order passes down verbatim; a
referee who wants his divisions doing different things writes them himself.

`autoCascade` is false for player-held commanders, who write their own.

### Decision points

A unit with no new orders continues its last one. But on **discovering something** it
stops and asks, and the referee decides:

```ts
interface PendingDecision {
  readonly id: string
  readonly commanderId: string
  readonly atHours: number
  readonly trigger:
    | 'enemy_contact' | 'crossing_impassable' | 'gunfire_heard'
    | 'objective_reached' | 'order_arrived' | 'out_of_provisions' | 'attacked'
  readonly context: unknown
}
```

Raised against the **commander**, not the unit, because deciding is something a man does
and the referee should decide from that man's information. Triggers are config-listed.

The scheduler's primary control becomes **advance until something needs a human** — the
clock runs forward and halts at the first decision, which is how a refereed game runs.

This is also the hook for sub-commander personalities later: a personality is a policy
that resolves some decisions without asking, and it drops in exactly here.

---

## Step 2 — The despatch book

The client's centre of gravity moves. With sight limited to the formation a commander
rides with, the map shows accurate ground and almost nothing on it, and the **despatch
book is the primary interface**.

The current console has that backwards — map centre stage, sidebar for detail — right for
a referee looking at ground truth and wrong for a commander. The referee console keeps the
current shape, plus the decision queue.

What a commander's page must answer at a glance:

- Every formation beneath me, where I last heard it was, and **how long ago that was**
- Every order I have sent, to whom, at what hour, and whether it has been acknowledged
- Every report I have received, from whom, and **what hour it describes** rather than the
  hour it arrived — a report that took six hours to reach me is telling me about 0400
- What my own formation can see right now

Age is the thing to draw loudly. Every fact on a commander's screen carries an hour, and
the gap between that hour and the clock is now the whole of the fog.

---

## Deferred: the issued map

Kept because the thinking is done and only the timing changed. Terrain fog returns with
it, as `terrainFog: true` plus the machinery below.

**Fog becomes doubt rather than darkness.** Three states replace two: **known** (ground
truth, surveyed by a formation whose report has reached you), **mapped** (your issued
survey — drawn, and possibly wrong), **unmapped** (blank).

**Stored, not computed.** A document per faction, generated once at campaign creation. A
stored artifact can be diffed against truth in a test and can never accidentally come out
*more* accurate than intended — a function that runs on the server with ground truth in
hand every time is a function whose every bug is a leak.

**What lies, and what does not.** Accurate: coastlines, terrain class, elevation, major
rivers, highways. Falsified: minor river courses, tracks and secondary roads, fords, and
settlement markers displaced a hex or two — the marker, not the hex's settlement data, so
ground truth keeps the village where it is for quartering.

**Missing and invented are not the same.** A missing feature is frustrating and safe; an
invented one is dangerous. Both, weighted toward missing.

**Quality is a referee dial**, per faction: a good staff survey against a captured sketch.

**The mechanic this exists for** is that a march can *fail on discovery* — the column
reaches where the bridge should be and finds a river. That needs the scheduler and the
decision queue from step 1.

---

## Order of work

1. **Commanders as entities and roles; knowledge per commander; terrain fog off.**
   `recon.ts`, `state.ts`, `observe.ts`, `view.ts`, the `roles` table, join links.
2. **Despatches, orders, reports, decision points, the scheduler.**
3. **The despatch book**, and the referee's decision queue.
4. *Deferred:* issued maps, their falsification, and correction by recce.

Also deferred: combat, fatigue accumulation, provisions and equipment consumption, depots
and convoys, terrain muting of gunfire, sub-commander personalities.
