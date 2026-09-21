/**
 * Every word the console says.
 *
 * One file, for the same reason `config.ts` is one file: the wording of this game is a
 * thing that gets tuned, and a thing that gets tuned should be editable in one place by
 * somebody who is thinking about that thing and nothing else. A register that drifts —
 * half the panels addressing a commander, half addressing a user — is the failure this
 * prevents, and it is not a failure anybody catches while reading one component.
 *
 * ## Why this is TypeScript rather than a data file
 *
 * Most of the text here is not text. It is a sentence with a number in it, and the number
 * has to be formatted and placed. Held as YAML those become templates with named holes
 * and a helper to fill them, which buys editability at the price of every miswritten hole
 * becoming a blank space in front of a player rather than a build failure. Held as
 * functions, the compiler checks that the caller has the hour it is about to print.
 *
 * So: plain strings where a string is the whole of it, and a function where the sentence
 * needs something from the caller. Nothing else. No formatting, no markup, no decisions —
 * a component that wants to know whether to say "hex" or "hexes" asks for `hexes(n)` and
 * is not told how the choice was made.
 *
 * ## What is deliberately not here
 *
 * **Domain vocabulary that the engine owns.** Unit kinds, formations and traits are
 * rendered by unfolding the engine's own identifiers — `bad_going` becomes "Bad going" —
 * and that mapping lives with `prettify` below rather than as a hundred entries. A name
 * that wants to read differently from its identifier gets an entry in `TERMS`.
 *
 * **Server refusals.** A violation message comes back from the engine and is shown as it
 * arrived. Those are part of the rules rather than part of the console, and restating
 * them here would give the same refusal two wordings that could disagree.
 */

// ---------------------------------------------------------------------------
// Vocabulary
// ---------------------------------------------------------------------------

/**
 * Engine identifiers whose display name is not simply themselves unfolded.
 *
 * Short by design. Anything that reads correctly as "under_scores capitalised" is left to
 * `prettify`, so this list holds only the genuine exceptions and stays readable.
 */
export const TERMS: Readonly<Record<string, string>> = {
  off_road: 'Off-road',
  hq: 'HQ',
};

/** An engine identifier as a reader should see it: `bad_going` becomes `Bad going`. */
export const prettify = (s: string): string =>
  TERMS[s] ?? s.replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase());

/** The same, tolerating the absent value a nullable field may hold. */
export const prettyOrDash = (s: string | null): string => (s === null ? '—' : prettify(s));

/** `1 hex`, `4 hexes`. The one plural the console needs often enough to centralise. */
export const hexes = (n: number): string => `${n} ${n === 1 ? 'hex' : 'hexes'}`;

export const copy = {
  // -------------------------------------------------------------------------
  // Getting in
  // -------------------------------------------------------------------------
  join: {
    title: 'Campaign',
    blurb: 'A refereed Napoleonic campaign on a generated map. Each commander sees only what they see or what has been reported to them.',

    startHeading: 'Start a campaign',
    uploadLabel: 'Upload a world.json',
    /** Split around the `<code>` the command is shown in. */
    generateHintBefore: 'Generate one with',
    generateCommand: 'worldgen generate --model organic',
    generateHintAfter: '. A classic world loads, but carries no ford or bridge data, so every major river will be impassable.',
    demoButton: 'Or run the demonstration',

    /**
     * The campaigns this browser holds a link for.
     *
     * The front page is reachable now, which means it has to be worth arriving at: a
     * player who followed a link once has had the token stripped from their address bar,
     * and this list is the only way back in that does not involve finding the original
     * message again.
     */
    resumeHeading: 'Campaigns on this browser',
    resumeBlurb: 'Links you have followed on this browser. Nobody else can open these; the tokens never leave it.',
    resumeReferee: 'as referee',
    resumeCommander: 'as a commander',
    unnamedCampaign: 'Untitled campaign',
    forget: 'Forget',
    forgetHint: 'Drop this browser’s link. The campaign itself is untouched, and a referee can issue another.',
    noSuchCampaign: 'This browser does not hold a link for that campaign. Ask your referee to send you one, or start a campaign of your own.',

    joinHeading: 'Join one',
    joinBlurb: "Open the link your referee sent you. It carries your side's token in the URL fragment, which never reaches the server and never appears in its logs.",

    // The progress line, phase by phase. A referee watching a demo build reads these in
    // order, so they are worded as a sequence rather than as four unrelated captions.
    loadingDemoWorld: 'Loading the demonstration world…',
    readingWorld: 'Reading the world…',
    uploadingWorld: 'Uploading the world…',
    formingArmy: (done: number, total: number): string => `Forming the army… ${done} of ${total}`,
    issuingLinks: 'Issuing join links…',
    commandRefused: 'a command was refused',

    createdTitle: 'Campaign created',
    createdBlurb: 'One link per seat. Send each player their own and keep the referee’s — they are minted once and stored only as hashes, so a lost link is reissued rather than looked up.',
    refereeSeat: 'Referee',
    enterAsReferee: 'Enter as referee',

    demoCampaignName: 'Demonstration',
  },

  // -------------------------------------------------------------------------
  // The console frame: header, clock, notices
  // -------------------------------------------------------------------------
  console: {
    loadingTitle: 'Campaign',
    loading: 'Fetching your map…',
    unreadableTitle: 'Cannot read that campaign',
    useAnotherLink: 'Use a different link',

    refereeSeat: 'Referee',
    orderOfBattle: (count: number): string => `Order of battle · ${count}`,
    orderOfBattleHint: 'Every formation, and what it is doing',

    reachToggle: 'Reach of selected, 10 h',

    washHint: 'What is currently observed (v)',
    washThree: 'Watched · marched · unknown',
    washTwo: 'Watched only',
    washNone: 'No shading',

    advanceHour: '+1 h',
    advanceSix: '+6 h',
    run: 'Run',
    runHint: 'Advance until something needs a decision',
    battle: 'Battle',
    battleDone: 'Done',
    battleHint: 'Paint the ground being fought over',

    liveHint: (state: string): string => `Live updates ${state}`,
    liveOpen: 'live',
    liveReconnecting: 'reconnecting',

    /** The way back to the front page, and the reason there is one in the header. */
    home: 'Campaigns',
    homeHint: 'Back to the front page. This campaign keeps its own address, so you can return to it.',

    dismiss: 'Dismiss',
  },

  /**
   * The banners across the top of the map.
   *
   * Each one is a mode the referee is in and cannot see from the map itself, so each says
   * what a click will now do and how to get out. "Escape to think again" is the same
   * phrase every time on purpose: it is the one keystroke that always works.
   */
  notices: {
    clockStopped: (at: string, who: string, unit: string, trigger: string): string =>
      `The clock stopped at ${at}: ${who}${unit} ${trigger}. It is in the queue below.`,

    pickBattle: 'Point at the ground being fought over; point again to take it back out. Traffic rules stop applying there — formations in a battle are intermingled, and this map does not resolve what happens between them. Escape when the field is drawn.',

    pickForRaise: 'Point at the ground the new formation is to stand on. Escape to think again.',

    pickForPlace: (name: string): string =>
      `Point at the ground ${name} is to stand on. It goes there without marching, and the log records that you moved it. Escape to think again.`,

    pickForMarch: (name: string): string =>
      `Point at the ground ${name} is to march to. Point again to insist they go by way of somewhere first — the last place you name is where they are to end up. Places, not a route: between them they will find their own way, and discover what is in it when they get there. Escape to think again.`,

    /** What a commander is told once, on arriving. The whole design, in four sentences. */
    whoYouAre: (name: string, unit: string, visibleHexes: number): string =>
      `You are ${name}, riding with ${unit}. You can see ${visibleHexes} hexes from where you stand.`,
  },

  // -------------------------------------------------------------------------
  // The clock, and the age of a fact
  // -------------------------------------------------------------------------
  clock: {
    /** `Day 3, 14:00`. The reader's unit, as against the engine's hour count. */
    dayHour: (day: number, time: string): string => `Day ${day}, ${time}`,
    now: 'now',
    minutesAgo: (minutes: number): string => `${minutes} min ago`,
    hoursAgo: (hours: string): string => `${hours} h ago`,
  },

  // -------------------------------------------------------------------------
  // What a commander has: their own formation, and their memory of the rest
  // -------------------------------------------------------------------------
  command: {
    heading: 'Under my command',
    withMe: 'with me',
    nobodyElse: 'Nobody else answers to you.',
    drift: (km: number): string =>
      `Every hour above is when you last heard, not where they are. At infantry pace the oldest of them could be ${km} km from the hex you are looking at.`,

    halted: (q: number, r: number): string =>
      `Halted at ${q}, ${r} — the march is done.`,
    marchingOn: (q: number, r: number, ordered: string): string =>
      `Marching on ${q}, ${r}, ordered ${ordered}.`,
  },

  // -------------------------------------------------------------------------
  // The post
  // -------------------------------------------------------------------------
  post: {
    inboxHeading: 'In my hand',
    write: 'Write a despatch',
    emptyInbox: 'Nothing has reached you. Anything on the road toward you is invisible until a rider puts it in your hand.',

    written: (at: string, age: string): string => `Written ${at} · ${age}`,
    reached: (at: string, hours: string): string =>
      `Reached you ${at}, after ${hours} h on the road`,
    forwardedFrom: (who: string): string => ` · forwarded from ${who}`,
    superseded: 'Overtaken by a later order you already hold. Disregarded.',
    sightingsAttached: (n: number): string =>
      n === 1 ? '1 sighting attached' : `${n} sightings attached`,

    acknowledged: 'Acknowledged',
    acknowledge: 'Acknowledge',
    forward: 'Forward',
    /** The body of an acknowledgement, which is itself a despatch and can itself be lost. */
    acknowledgementText: (of: string): string => `Received your despatch of ${of}.`,

    sentHeading: 'Sent',
    emptyOutbox: 'You have written nothing yet.',
    to: (who: string): string => `To ${who}`,
    handed: 'Handed over on the spot — their column was touching yours.',
    arrived: 'Acknowledged. It arrived.',
    /** The mechanic, not a missing feature. It is worded to make that unambiguous. */
    unknownFate: 'Unacknowledged.',
    viaWaypoints: (n: number): string => ` · Rider sent via ${n} waypoint(s) of yours.`,
  },

  // -------------------------------------------------------------------------
  // Writing one
  // -------------------------------------------------------------------------
  composer: {
    heading: 'Write a despatch',
    from: 'From',
    to: 'To',
    messageOnly: ' — message only',

    orderPlaceholder: 'Move on Quatre Bras with all speed; I expect you astride the crossroads by noon.',
    reportPlaceholder: 'What you have seen, and when you saw it.',

    asReferee: 'Written in their name, and logged as yours. What the addressee makes of it is still a decision when it arrives.',
    isOrder: 'This is an order. The referee will read it and decide what your subordinate makes of it.',
    isMessage: 'This is a message. You may write to anyone on your own side; only those beneath you take orders.',

    /** A commander is told why there is no estimate, rather than shown an empty space. */
    noEstimate: 'How long the ride takes is not yours to know. Your rider will go until they find the addressee, and nobody will tell you when they did.',
    nothingToRideTo: 'Nothing on the map to ride to yet. Send the despatch anyway — your rider will find them.',
    rideLabel: 'The ride:',
    rideEstimate: (hours: string, arrival: string): string =>
      ` about ${hours} h over the ground as it stands, arriving around ${arrival}. Your rider may be stopped on the way, and the sender is never told whether they were.`,

    sending: 'Sealing…',
    send: 'Send by rider',
    cancel: 'Cancel',
  },

  // -------------------------------------------------------------------------
  // The referee's queue and log
  // -------------------------------------------------------------------------
  referee: {
    despatchesHeading: 'Despatches',
    writeOnBehalf: 'Write on a commander’s behalf',
    writeOnBehalfBlurb: 'For officers you run yourself, and for a player who hands you an order on paper. It goes by rider like any other and can be intercepted like any other.',

    queueHeading: 'Wants a decision',
    queueEmpty: 'Nothing is waiting on you. Run the clock until something is.',
    /** Traffic carries no commander: two columns meeting is a fact about the ground. */
    theGround: 'The ground',

    noStandingTask: ' · no standing task',
    taskHalted: ' · halted',
    taskMarching: (q: number, r: number): string => ` · marching on ${q}, ${r}`,

    giveItTo: (name: string): string => `Give it to ${name}`,
    march: 'March them somewhere',
    pointing: 'Pointing…',
    dealtWith: 'Dealt with',

    // The captions under a decision, one per trigger that has something to add.
    columnsSeen: (n: number, at: string): string =>
      `${n === 1 ? 'A column' : `${n} columns`} seen at ${at}`,
    /**
     * Where a patrol ran into something.
     *
     * The hex is optional and is printed as it arrives rather than defaulted: a position
     * invented to fill a gap is the one kind of wrong this game must never be.
     */
    patrolMet: (
      side: string,
      what: string,
      q: number | undefined,
      r: number | undefined,
    ): string => `${side} ${what} at ${q ?? ''}, ${r ?? ''}.`,
    patrolRoll: (dice: number): string =>
      ` Roll ${dice}d6: any 1 and the patrol is lost, otherwise it recoils 2 km.`,
    patrolHostile: 'Enemy',
    patrolFriendly: 'Friendly',
    patrolSomething: 'column',
    despatchArrived: (from: string, written: string): string =>
      `From ${from}, written ${written}. Read it in their seat, then tell their formation where to go.`,
    stoppedShort: (aq: number, ar: number, dq: number | undefined, dr: number | undefined): string =>
      `Stopped at ${aq}, ${ar}, short of ${dq}, ${dr}.`,
    somebody: 'somebody',

    logHeading: 'The post',
    logEmpty: 'Nobody has written to anybody.',
    riding: (done: number, total: number): string => `Riding — ${done} of ${total} hexes.`,
    ridingHanded: 'Handed over.',
    ridingBlind: 'The sender has no idea.',
    delivered: (at: string, hours: string): string => `Delivered ${at}, after ${hours} h.`,
    deliveredHanded: ' Handed over on the spot.',
    lost: (by: string, at: string, dice: string): string =>
      `Rider stopped by ${by} on ${at} — dice [${dice}]. The paper went with them.`,
    capturedLabel: 'Captured',
    captured: (by: string, at: string, dice: string): string =>
      ` by ${by} on ${at} — dice [${dice}]. They have read it; the sender has not been told.`,
  },

  /**
   * What each halt trigger means, in the referee's language rather than the engine's.
   *
   * Read through `triggerLabel` rather than indexed directly: the engine may add a
   * trigger this file has not been told about yet, and the referee should get a sentence
   * rather than a gap when it does.
   */
  triggers: {
    enemy_contact: 'has come into contact',
    crossing_impassable: 'cannot get across',
    gunfire_heard: 'hears guns',
    objective_reached: 'has arrived',
    despatch_arrived: 'has received a despatch',
    out_of_provisions: 'is out of provisions',
    attacked: 'is under attack',
    column_blocked: 'has run into a column in its way',
    column_contested: 'is contesting a hex, and neither is the faster',
    patrol_contact: 'has run into something',
    unknown: 'needs a decision',
  },

  // -------------------------------------------------------------------------
  // Orders, on the referee's side of the map
  // -------------------------------------------------------------------------
  orders: {
    heading: 'Orders',
    march: 'March them somewhere',
    pointing: 'Pointing…',
    halt: 'Halt',
    place: 'Place',
    placeHint: 'Put it there without marching it',
    sendPatrol: 'Send out a patrol',
    patrolCost: (troopers: number): string => ` · ${troopers} troopers`,

    nowhereNamed: 'Nowhere named yet.',
    destinationSuffix: ' — where they are to be',
    waypointSuffix: ' — by way of',
    confirmMarch: 'March',
    undoLast: 'Undo last',
    cancel: 'Cancel',
    refused: 'refused',

    noTask: 'No standing task.',
    arrived: (q: number, r: number): string =>
      `Arrived at ${q}, ${r}. Nothing further ordered.`,
    marchingOn: (q: number, r: number, ordered: string): string =>
      `Marching on ${q}, ${r}, ordered ${ordered}.`,
    routeAhead: (hexesLeft: number, hours: string): string =>
      `${hexesLeft} hexes still to march, about ${hours} h at the head.`,
    byWayOf: (places: string): string => ` By way of ${places}.`,

    /** A referee's shorthand for the roster's "doing" column. */
    taskArrived: (q: number, r: number): string => `arrived ${q}, ${r}`,
    taskMarching: (q: number, r: number): string => `marching on ${q}, ${r}`,
  },

  /** Changing formation. The hours are on the buttons because the cost is the rule. */
  formation: {
    changing: (from: string, to: string, ready: string, left: string): string =>
      `${from} → ${to}, ready at ${ready} (${left} h to go).`,
    alreadyIn: 'Already in this formation',
    changeHint: (hours: number): string => `${hours} h to change`,
    cost: (hours: number): string => ` · ${hours} h`,
  },

  // -------------------------------------------------------------------------
  // The order of battle
  // -------------------------------------------------------------------------
  roster: {
    label: 'Order of battle',
    heading: 'Order of battle',
    close: 'Close',
    empty: 'Nothing here.',

    columnFormation: 'Formation',
    columnWhere: 'Where',
    columnStrength: 'Strength',
    columnFatigue: 'Fatigue',
    columnDoing: 'Doing',

    patrol: ' · patrol',
    detachment: 'detachment',
    notTracked: '—',
    underArms: (n: string): string => `${n} under arms`,
    morale: (has: number, max: number): string => `morale ${has}/${max}`,
    withYou: 'with you',

    groupWithYou: 'With you',
    groupUnderCommand: 'Under your command',
    groupUnderCommandNote: 'As you last heard. Every hour below is when word reached you, not where they are now.',
  },

  /** Raising and appointing: the referee's preparation for a game. */
  orbat: {
    raiseFormation: 'Raise a formation',
    appointCommander: 'Appoint a commander',

    name: 'Name',
    namePlaceholder: '1re Division',
    side: 'Side',
    arm: 'Arm',
    paperStrength: 'Paper strength',
    experience: 'Experience',
    corps: 'Corps',
    corpsPlaceholder: 'I Corps',
    traits: 'Traits',

    pointAtGround: 'Point at the ground',
    standingAt: (q: number, r: number): string => `Standing at ${q}, ${r} — move`,
    raise: 'Raise',
    /** Split around the `<code>` the generated id is shown in. */
    willBeRaisedBefore: 'It will be raised as',
    willBeRaisedAfter: ', fresh and fully supplied.',

    commanderNamePlaceholder: 'Marshal Ney',
    ridesWith: 'Rides with',
    answersTo: 'Answers to',
    noSuperior: 'nobody — army command',
    none: '—',
    appoint: 'Appoint',
    appointBlurb: 'Appointing a commander does not give anybody a seat. Issue a link when you want somebody to play them.',

    // What is wrong with a draft, in the order a reader would find it.
    needsName: 'It needs a name.',
    idTaken: (id: string): string => `There is already a ${id}.`,
    strengthNegative: 'Paper strength cannot be negative.',
    needsGround: 'Point at the ground it stands on.',
    fallbackId: 'formation',
  },

  // -------------------------------------------------------------------------
  // The sidebar panels
  // -------------------------------------------------------------------------
  unit: {
    broken: 'Broken — must rout.',
    starving: 'Out of provisions — cannot march or fight.',

    strengthHeading: 'Strength',
    paperStrength: 'PaperStrength',
    presentUnderArms: 'Present under arms',
    presentUnderArmsHint: 'PaperStrength reduced by fatigue',
    guns: 'Guns',
    fatigue: 'Fatigue',
    morale: 'Morale',
    provisions: 'Provisions',
    equipment: 'Equipment',

    /** A patrol carries none of its own, and what is shown is its parent's. */
    patrolOrphaned: 'A detachment. Its parent is out of sight.',
    patrolOf: (parent: string): string =>
      `A detachment of ${parent}, and immune to all of it. These are the parent’s.`,

    columnHeading: 'Column',
    length: 'Length',
    lengthHint: 'PaperStrength × spacing × multiplier',
    occupies: 'Occupies',
    occupiesHint: (marchHexes: number, fold: string): string =>
      `${marchHexes} strung out on the march, ${fold}`,
    catchup: 'Catch-up',
    catchupHint: 'For the rear to reach the head, at road speed',
    spacing: 'Spacing',
    spacingValue: (m: number, multiplier: number): string => `${m} m × ${multiplier}`,

    marchHeading: 'March',
    formation: 'Formation',
    marchedToday: 'Marched today',
    marchedTodayValue: (done: string, cap: number): string => `${done} h of ${cap} h`,
    remaining: 'Remaining',
    speedByGoing: 'Speed by going',

    reconHeading: 'Reconnaissance',
    sees: 'Sees',
    seesValue: (radius: number): string => `${hexes(radius)} from the column`,
    patrols: 'Patrols',
    patrolsFree: (free: number): string => `up to ${free} without cost`,
    patrolsOut: (out: number, free: number): string => `${out} out of ${free} free`,
    patrolsPaid: (paid: number): string => ` · ${paid} paid for`,
    detachedFrom: 'Detached from',

    traitsHeading: 'Traits',

    /** Why the ground a formation stands on is not the length of its column. */
    fold: {
      battle: 'deployed at a kilometre of frontage per ten thousand troops',
      rest: 'gathered into camp',
      occupation: 'gone into quarters',
    } as Readonly<Record<string, string>>,
  },

  /**
   * One of your own, as you last heard.
   *
   * Stated in the past tense throughout, and with the hour before the numbers. A reader
   * who cannot tell this from a live panel at a glance will plan on the wrong one.
   */
  report: {
    lastReport: 'Last report',
    reportingHour: 'Reporting hour',
    reportingHourHint: 'The hour this describes, which is not necessarily the hour it reached you.',
    stoodAt: 'Stood at',
    paperStrength: 'PaperStrength',
    fatigue: 'Fatigue',
    fatigueValue: (n: number): string => `${n} of 100`,
    provisions: 'Provisions',
    formation: 'Formation',
    drift: (km: number): string =>
      `Where they are now is not something you know. At infantry pace they could be anywhere within ${km} km of that hex by now.`,
  },

  /** An enemy, as far as anybody knows — and no more than that. */
  contact: {
    title: (id: string): string => `Contact ${id}`,
    unidentified: 'Unidentified',
    seenAt: 'Seen at',
    reported: 'Reported',
    justNow: 'Just now',
    ago: (hours: string): string => `${hours} h ago`,
    arm: 'Arm',
    armUnknown: 'Unknown',
    armHint: 'Only a close patrol reports whether a formation is horse, foot or guns.',
    grade: 'Report grade',
    gradeValue: (level: number): string => `${level} of 6`,
    drift: (km: number): string =>
      `This is where it was, not where it is. At infantry pace it could be anywhere within ${km} km of that hex by now.`,
    yourOwnNumber: 'Your staff’s own number for this sighting. Whether it is the same body of troops as any other contact on your map is your judgement, not a fact you have been given.',

    /** What a report at each grade actually told you. Straight from the patrol table. */
    intel: {
      1: 'Something is there. Nothing more.',
      2: 'Presence and position.',
      3: 'Presence, position and the direction of march.',
      4: 'Rough strength.',
      5: 'The arm — horse, foot or guns.',
      6: 'The formation identified by name.',
    } as Readonly<Record<number, string>>,
  },

  /** The ground under the cursor. Derived answers beside the values that decided them. */
  hex: {
    title: (q: number, r: number): string => `Hex ${q}, ${r}`,
    neverObserved: 'Never observed. Nothing is known about this ground — what is stored here are defaults standing in for the unknown, not measurements.',

    terrain: 'Terrain',
    biome: 'Biome',
    cover: 'Cover',
    settlement: 'Settlement',
    status: 'Status',
    remembered: 'Remembered — not currently observed',

    reliefHeading: 'Relief',
    elevation: 'Elevation',
    slope: 'Slope',
    relief: 'Relief',
    metres: (n: number): string => `${n} m`,
    metresPerKm: (n: number): string => `${n} m / km`,

    goingHeading: 'Going',
    water: 'Water. Impassable to every unit in this ruleset.',
    grade: 'Grade',
    roads: 'Roads',
    hoursToEnter: 'Hours to enter, by arm',
    offRoadColumn: 'Off road',
    onRoadColumn: 'On road',
    hours: (n: string): string => `${n} h`,

    watercourseHeading: 'Watercourse',
    riverClass: 'Class',
    riverMajor: 'Major — navigable',
    riverMinor: 'Minor — fordable',
    catchment: 'Catchment',
    catchmentValue: (km2: number): string => `${km2} km²`,
    discharge: 'Discharge',
    dischargeValue: (has: string, navigable: string): string => `${has} of ${navigable}`,
    crossing: 'Crossing',
    bridge: 'Bridge',
    ford: 'Ford',

    crossingFor: (unitId: string): string => `For ${unitId}`,
    noCrossingNeeded: 'No crossing needed',
    crossingCost: (how: string, hours: number): string => `${how} — ${hours} h`,
    cannotCross: (why: string): string => `Cannot cross. ${why}`,

    tagsHeading: 'Tags',
  },

  /** With nothing under the cursor, the panel says what the map is for. */
  idle: {
    heading: 'Nothing',
    blurb: 'Move over the map to read the ground, or over a column to read the unit standing on it. Click a unit to keep it in view.',
    formations: 'Formations',
    contacts: 'Contacts',
    contactLabel: (id: string): string => `Contact ${id}`,
    campaignHeading: 'This campaign',
    /**
     * The campaign's own address, which is the thing worth bookmarking.
     *
     * Not the join link: that one carries a token and should be sent to one person, once.
     * This one carries nothing secret and opens only on a browser already holding a link,
     * so it is the address to keep.
     */
    bookmark: 'This campaign’s address:',
    bookmarkHint: 'Bookmark this. It carries no token, and opens only on a browser that already holds a link.',
    yourLink: 'Your link:',
    leave: 'Leave',
  },
};

/** A trigger in the referee's language, falling back to a sentence for an unknown one. */
export const triggerLabel = (trigger: string): string =>
  (copy.triggers as Record<string, string | undefined>)[trigger] ?? copy.triggers.unknown;

export type Copy = typeof copy;
