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
| What an order is | **Prose.** The referee reads it and sets the march |
| A despatch's fate | **Never known to the sender.** Only an acknowledgement tells him |
| A despatch's route | **Referee only.** See below — this one is load-bearing |
| A despatch's delivery estimate | **Never shown.** An ETA is a distance is a position |
| A commander's inbox | **Delivered only.** Nothing in transit toward him is visible |
| Rider routing | Least-time path to the target's **actual** position |
| Rider routing, overridden | The sender may insist on waypoints — around pickets, say |
| Couriers available | **Unlimited.** Time and interception are the whole cost |
| Contradictory orders | Despatches are dated; a later one already held wins |
| An NPC commander's orders | **Cascade to subordinates immediately**, referee may override |
| A unit with no new orders | Continues its current task, until it discovers something |
| On discovery | Stops and asks. The referee decides |
| Reports on contact | Automatic, and free between formations whose columns touch |
| Who may be written to | Anyone in your own faction. **Orders only downward** |

### Why terrain fog is off

It is the least interesting fog here. The tension the period turns on is *where is the
enemy* and *where is my own III Corps*, not *what does the ground look like* — a commander
in 1815 had a map. And with sight limited to the formation he rides with, terrain fog
would leave him a two-hex bubble of ground and black everywhere else, unable to plan a
march at all.

**The machinery stays, switched off.** `terrainFog` is campaign config defaulting to
false; `maskWorld`, the fog tags and the Python renderers' fog handling are unchanged and
still tested in both positions. It returns with the issued map (deferred, below), as a
switch rather than a rebuild.

**Observation keeps being recorded even though nothing masks on it.** The `hexes_revealed`
events keep flowing into the log. They cost almost nothing, and when the issued map
arrives it will want the history of who surveyed what — which an event-sourced system gets
for free if the events were written all along, and cannot reconstruct if they were not.

---

## The separation that everything else rests on

**An order is a message. A march is a command. The referee is the bridge.**

Orders are prose, so the engine executes nothing on its own. That makes the information
layer and the physics layer two different objects, and keeping them apart is what stops
either one distorting the other:

| | **Despatch** | **Task** |
|---|---|---|
| What it is | Prose a commander wrote | A march the engine is running |
| Who makes it | A commander | The referee, having read a despatch |
| Travels by | Rider, at courier speeds | Nothing — it is not a thing in the world |
| Can be | Intercepted, captured, lost, ignored | Interrupted by a decision point |
| Validated by | Nobody. It is a piece of paper | `check`, like every other command |

A player writes *"Move on Quatre Bras with all speed; the Emperor expects you astride the
crossroads by noon."* The referee reads it, decides what III Corps' commander makes of it,
and issues a march. The clock then runs that march hour by hour and halts when something
needs a human.

This is what a refereed game actually is, and it resolves several things at once. Order
validation moves to where movement already lives, rather than existing twice. Interception
becomes maximally interesting, because a captured despatch is *intelligence* rather than a
coordinate dump. And "a unit with no new orders continues its last one" needs no special
case: the task continues, and the prose is only the paper trail.

The cost is real and lands on the referee, who must read every NPC commander's post and
translate it. The console has to make that one motion rather than two — see §Step 3.

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
  /** Passes an arriving despatch straight down. True for referee-run, false for players. */
  readonly autoCascade: boolean
}
```

Two fields carry the whole hierarchy. "A human who also commands the corps" needs no
special case: he is the commander of 1re Division — `unitId` points at it — and the other
divisional commanders have him as their `superiorId`. One man, two appointments, one
record. `Unit.corps` becomes a label; the chain is derived from the tree.

He may send a message to anyone in his own faction — lateral coordination between corps
commanders was real, and mattered enormously — but an **order** only travels downward.

### Units observe; commanders know

What is seen is a property of the formation: its column, its scouts, the ground it covers.
What is *known* is a property of the man. So observation is computed from the unit and
recorded against its commander:

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
**the skipped commander does not know.** He holds a stale and confidently wrong picture of
his own corps, and nothing had to be built to make that happen.

The referee gains something too. Every NPC formation has a commander with a real, limited
view, so adjudicating what III Corps does on contact is done **from III Corps' own
information** rather than from omniscience exercised with restraint.

---

## Step 1 — Despatches

**One entity, four ledgers.** Every despatch runs commander to commander; the unit is only
the address.

```ts
interface Despatch {
  readonly id: string
  readonly kind: 'order' | 'report' | 'acknowledgement'
  readonly from: string                  // commander id
  readonly to: string                    // commander id
  readonly sentAtHours: number
  readonly body: DespatchBody
  /** Waypoints the sender insists on — around a wood he thinks holds enemy pickets. */
  readonly via?: readonly Hex[]
  /** A report being passed on. Two lags stack, which is very much the period. */
  readonly forwardedFrom?: string
  readonly inReplyTo?: string
  /**
   * The rider's path, least-time through `via` to the addressee's ACTUAL position.
   *
   * REFEREE ONLY. See below: this field is the one place in the design where a leak
   * would be invisible and total.
   */
  readonly route: readonly Hex[]
}

interface DespatchBody {
  /** What the commander wrote. Orders are prose and nothing but. */
  readonly text?: string
  /** Attached by an automatic report; a hand-written one may carry them too. */
  readonly contacts?: readonly Contact[]
  readonly unitReport?: UnitReport
}
```

An order is text only. An automatic contact report is data only. A commander writing up a
sighting himself may send both.

| Ledger | Derived from | Visible to |
|---|---|---|
| Orders I have sent | from me, kind `order` | me — never their fate or route |
| Orders I have received | to me, kind `order`, **delivered** | me |
| Reports I have sent | from me, kind `report` | me — never their fate or route |
| Reports I have received | to me, kind `report`, **delivered** | me |

A commander sees his own outbox in full, because he wrote it. What he never learns is
whether any of it arrived.

### The route is the leak

Riders take the least-time path to the target's **actual** position, so the route is
computed from ground truth — which means **the route betrays where the target is.** Show a
commander where his rider went and you have told him exactly where his detached corps is,
destroying the one thing this design exists to model: he would never need a report again,
he would read his own outbox.

So `route` is referee-only, absolutely. He sees *sent 0400, to Ney, via the Charleroi
road* — his own `via`, which he chose — and never a hex of what the rider actually did.
The route exists server-side to resolve interception and arrival, and for nothing else.

**The same reasoning bans a delivery estimate.** "Expected 0700" is a distance, and a
distance is a position. No despatch ever carries an ETA.

What the *client* may do instead is compute the commander's own guess from his last report
of that formation and label it as such: *"if they stand where they did at 0400, this
reaches them about 0700."* Stale, possibly wrong, and honest about it. This is the first
time the shared-engine decision from the original plan earns its keep — the client running
the same routing code over data the commander actually holds.

This is the same class of mistake as fabricating a `Unit` for an enemy contact so the map
had something to draw: a field that has to exist for the mechanic to work, one careless
`send` away from ending the game. Route and ETA each get a leakage test.

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
carries no undelivered despatch's fate, no despatch route, no ETA, no enemy unit record,
and no other commander's traffic.

### Dated orders, and arriving out of turn

Every despatch carries the hour it was written. A commander who already holds a later
order **disregards an earlier one that turns up afterwards** — correct staff practice, and
it means a second order reliably supersedes a first even when the riders overtake each
other.

The comparison is on the date alone, which is all the engine can do with prose, and all it
needs to do. The interesting case is not a stale order but the one that arrives with
nothing to compare it against.

### Interception now carries the whole weight

Riders are unlimited, they always find their man, and NPC orders cascade without delay.
Travel time and interception are therefore the *only* things that make command imperfect —
so if the game feels too reliable, the interception rules are the dial, not the courier
economy. Worth knowing before it is tuned in the wrong place.

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
minutes. It gives concentration a mechanical reward, which is the tension the period turns
on: concentrate and command well, disperse and forage well.

### Cascading

An NPC commander passes an arriving despatch straight down to his subordinates,
immediately, as fresh despatches with fresh riders — so the copies can still be
intercepted individually even though the decision cost nothing. The text passes down
verbatim; a referee who wants his divisions doing different things writes them himself.
`autoCascade` is false for player-held commanders, who write their own.

### Tasks and decision points

A task is what the engine is actually doing with a formation: a march to a destination,
with optional waypoints. **A destination, not a path** — the referee sets it knowing where
the formation really is, but the same rule protects a commander's own planning, where a
path drawn from a wrongly-believed start is nonsense. It is also what you actually wrote
in 1815: you told a corps where to go, not which fields to cross.

A formation continues its task until it **discovers something**, and then stops and asks:

```ts
interface PendingDecision {
  readonly id: string
  readonly commanderId: string
  readonly atHours: number
  readonly trigger:
    | 'enemy_contact' | 'crossing_impassable' | 'gunfire_heard'
    | 'objective_reached' | 'despatch_arrived' | 'out_of_provisions' | 'attacked'
  readonly context: unknown
}
```

Raised against the **commander**, not the unit, because deciding is something a man does
and the referee should decide from that man's information. Triggers are config-listed.

The scheduler's primary control is **advance until something needs a human** — the clock
runs forward and halts at the first decision. This is also the hook for sub-commander
personalities later: a personality is a policy that resolves some decisions without
asking, and it drops in exactly here.

---

## Step 2 — The commander's interface

Two jobs: write despatches, and read the post. With orders as prose, the composer is
small — a text box, an addressee, and one route override — and almost all the design
effort belongs in the reading.

### Writing

- **To**: any commander in my faction; the tree marks which of them I may *order* rather
  than merely write to.
- **Text**: prose. The whole content of an order.
- **Send my rider via…**: optional waypoints, chosen on the map from my own knowledge.
  Distinct in wording and interaction from anything about where a formation should march —
  these are two unrelated routes and a UI that blurs them will be misread.
- **My estimate**: computed client-side from my last report of that formation, labelled as
  the guess it is. Never a server figure.

### Reading

- **The hour it describes, not the hour it arrived**, shown first. A report that took six
  hours is telling me about 0400, and drawing the arrival time loudly would be a lie about
  what I know.
- **Age everywhere.** Every fact on the page carries an hour, and the gap between it and
  the clock is now the whole of the fog.
- **Acknowledge**, one click. It is the only feedback channel that exists — the sole way a
  sender ever learns anything arrived.
- **Forward**, about as cheap. Passing a sighting to a peer or superior is how Grouchy's
  despatches worked, and a forwarded report stacks two lags.
- **Superseded orders marked**, not hidden. An order that arrived after a later one is
  disregarded, and seeing that happen is half of understanding why the corps did what it
  did.

### The formations panel

Every formation beneath me, where I last heard it was, and how long ago. This is the piece
that makes the model legible: five rows, each with an hour, most of them old.

---

## Step 3 — The referee's console

The referee's cost under prose orders is reading everyone's post and turning it into
marches, so the console has to make that **one motion rather than two**: an arriving
despatch and the control that sets the addressee's task live on the same screen, with the
formation's own view beside them.

The current console — map centre stage, sidebar for detail — is right for this and stays.
It gains the decision queue, the despatch log, and the ability to sit in any commander's
seat, which the role switcher already does in shape.

---

## Deferred: the issued map

Kept because the thinking is done and only the timing changed. Terrain fog returns with
it, as `terrainFog: true`.

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

1. ~~**Commanders as entities and roles; knowledge per commander; terrain fog off.**~~
   Done. `recon.ts`, `state.ts`, `observe.ts`, `view.ts`, the `roles` table, join links.
2. ~~**Despatches and riders; tasks; decision points; the scheduler.**~~ Done.
   `despatch.ts`, `task.ts`, `scheduler.ts`; four new commands; `sent`/`received`/
   `captured` on the view, with `route` and `fate` pinned out of it by leakage tests.
3. ~~**The commander's interface**: composer, inbox, formations panel.~~ Done — and it
   turned the fog on properly on the way, see below.
4. ~~**The referee's console**: decision queue, despatch log, seat-switching.~~ Done.
5. *Deferred:* issued maps, their falsification, and correction by recce.

### What step 4 settled

- **One motion, not two.** A decision names the commander and the formation, and carries
  the button that sets that formation's task. Point at ground and the march is ordered.
  If translating prose into a march were two actions the game would be tiring to run, and
  a game that is tiring to run does not get run.
- **Run until something happens** is the clock control that matters, and the reply says
  *what* stopped it rather than making the referee ask again.
- **Riders are drawn, for the referee alone.** Not by filtering: a commander's payload has
  no route in it, so his board has nothing to build a rider from. Watching a courier cross
  the country between two armies, and seeing it about to pass a picket before the dice do,
  is the best thing on the screen and it costs nothing to draw.
- **Two modes on one map**, which is usually a mistake. The alternative was a coordinate
  box, and a referee reading a despatch that says *Quatre Bras* wants to point at Quatre
  Bras. Mitigated by making the mode loud — the cursor ring, the hint bar and a banner all
  change — and by Escape leaving it, since a map that has silently changed what a click
  does is a trap.
- **Resolved decisions are kept, with the referee's note.** The queue is a history as well
  as a workload, and why he decided something is the most interesting line in a review.

### What step 3 found

Building the formations panel exposed that **reports were still snapshotted live**. Every
subordinate read "now", beside a caption explaining that the hour was when he last heard —
a design about not knowing where your own corps is, displaying a list of exactly where it
was. The delay step 2 built never reached the one place it mattered most.

So reports became **held knowledge** rather than a computed view:

- `CommanderKnowledge.reports`, filed by a `report_filed` event and replaced only by a
  *later* one, because riders overtake each other.
- Every despatch carries `unitReport` — where its sender stood when he sealed it — so
  arriving paper refreshes the picture of the man who wrote it. Silence from a corps is
  not merely a missing order; it is a stale map.
- Three cases need no rider: the formation he is standing next to, one whose column is
  touching his own, and a one-off seed for a formation he has never had word of, because
  he wrote the order of battle. Everything else waits.

And **contacts were merged across every formation under him**, which handed him whatever a
division forty kilometres away was looking at, this instant. `viewFor` now uses
`spottedBy` on his own formation alone. What his subordinates see reaches him as sightings
attached to a report, hours late.

**Still open, and the same defect class:** contacts are computed rather than held, so they
neither persist nor age — an enemy his column loses sight of vanishes instead of going
stale. And a contact still carries the observed unit's engine id, which lets two sightings
be correlated for free. Both are the remaining half of *Contacts become reported, not
computed*, and both want the treatment reports just had.

### What step 2 settled that the design had left open

- **A rider re-plans when his man has moved.** The route is laid to where the addressee
  stood when the rider left; reaching the end and finding the corps gone, he routes again
  from where he is. That is what makes "riders always find their man" a rule rather than
  an approximation, and the cost is his time — which is the thing meant to hurt.
- **One event per rider per advance, not per hex.** A courier covers ten hexes an hour, so
  a day's advance would otherwise write two hundred events saying "still riding".
  `despatch_progressed` carries his position and the path he is on, and is written when
  the clock stops or when he re-plans.
- **The clock only moves when something happens.** Ticks are `cfg.tickHours`; a quiet tick
  emits nothing, so events are stamped at the hour they occurred rather than the hour the
  referee clicked, and a twelve-hour advance through empty country costs one event.
- **A halt truncates the clock only when the referee asked for one.** A decision raised
  during a plain `advance 12h` goes into the queue and the clock runs on. Conflating those
  was a real bug, caught by asserting that six one-hour advances land where one six-hour
  advance does.
- **The sender is whoever holds the token**, never the `from` in the payload. A forged
  *report* would let anyone feed a commander false intelligence signed by his own
  subordinate, which is worse than forging an order.
- **A courier crosses a major river in an hour** rather than being stopped by it. Not from
  the rules, which are silent; a communication system in which one river ends
  correspondence altogether is not the period.

Also deferred: combat, fatigue accumulation, provisions and equipment consumption, depots
and convoys, terrain muting of gunfire, sub-commander personalities.
