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
   *
   * The Escape sentence is kept apart from the rest and added only where there is a key to
   * press. On a touch screen it would name a key that is not there, and the banner's own
   * Cancel or Done button is the way out.
   */
  notices: {
    clockStopped: (at: string, who: string, unit: string, trigger: string): string =>
      `The clock stopped at ${at}: ${who}${unit} ${trigger}. It is in the queue below.`,

    pickBattle: 'Point at the ground being fought over; point again to take it back out. Traffic rules stop applying there — formations in a battle are intermingled, and this map does not resolve what happens between them.',
    battleEscape: 'Escape when the field is drawn.',
    escape: 'Escape to think again.',

    pickForRaise: 'Point at the ground the new formation is to stand on.',

    pickForPlace: (name: string): string =>
      `Point at the ground ${name} is to stand on. It goes there without marching, and the log records that you moved it.`,

    pickForMarch: (name: string): string =>
      `Point at the ground ${name} is to march to. Point again to insist they go by way of somewhere first — the last place you name is where they are to end up. Places, not a route: between them they will find their own way, and discover what is in it when they get there.`,

    /** What a commander is told once, on arriving. The whole design, in four sentences. */
    whoYouAre: (name: string, unit: string, visibleHexes: number): string =>
      `You are ${name}, riding with ${unit}. You can see ${visibleHexes} hexes from where you stand.`,
  },

  /**
   * The phone layout: a tab bar along the bottom and a sheet over the map.
   *
   * Each tab names a pane of what is a sidebar on a wide screen. A referee's post is
   * everybody's, so it is called what it holds rather than "Post", which reads as their own.
   */
  tabs: {
    label: 'Console',
    map: 'Map',
    post: 'Post',
    despatches: 'Despatches',
    command: 'Command',
    orbat: 'Order of battle',
    decisions: (n: number): string => (n === 1 ? '1 wants a decision' : `${n} want a decision`),
    sheetOpen: 'Show more of this panel',
    sheetClose: 'Show more of the map',
  },

  /** The header's controls, gathered behind one button on a screen too narrow for them. */
  more: {
    open: 'More',
    heading: 'More',
    close: 'Close',
    seat: 'Seat',
    seatBlurb: 'Switching asks the server as that commander, so you see only what they see.',
    shading: 'Shading',
    washThreeBlurb: 'Seen now, seen before, never seen',
    washTwoBlurb: 'Everything not in sight now is dark',
    washNoneBlurb: 'The map as the survey drew it',
    reachNeedsSelection: 'Select a formation first',
  },

  /**
   * The line in the corner of the map.
   *
   * Two sets, for a mouse and for a finger, chosen by what the device can do rather than
   * how wide it is. The touch version has no Escape in it: there is no key to press, and
   * the banner above the map carries a Cancel button instead.
   */
  map: {
    hint: 'scroll to zoom · drag to pan · click a unit to select',
    picking: 'click the ground you want them to march to · Esc to think again',
    hintTouch: 'pinch to zoom · drag to pan · tap a unit to select',
    pickingTouch: 'tap the ground you want them to march to',
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
  // Day and night
  // -------------------------------------------------------------------------
  daylight: {
    /** On the sun and moon beside the clock. */
    day: (sunset: string): string => `Daylight. The sun sets at ${sunset}.`,
    night: (sunrise: string): string => `Night. The sun rises at ${sunrise}.`,
    heading: 'Daylight',
    hours: (sunrise: string, sunset: string): string => `Sunrise ${sunrise}, sunset ${sunset}.`,
    blurb: 'The season moves, and the sun with it. Night fatigue is charged by these hours from now on, and every console shows the same sun.',
    sunrise: 'Sunrise',
    sunset: 'Sunset',
    set: 'Set the sun',
  },

  // -------------------------------------------------------------------------
  // Standing orders: when the head of a column is on the road
  // -------------------------------------------------------------------------
  standing: {
    heading: 'Standing orders',
    blurbOwn: 'When the head of your column is on the road. It halts at whichever limit it reaches first, and they hold every day until you change them. Anyone else’s march day you give by despatch.',
    blurbReferee: 'When the head of this column is on the road. It halts at whichever limit it reaches first. For a commander you run, or on reading a despatch that gives them.',
    startHour: 'Step off at',
    latestHour: 'Off the road by',
    maxHoursOnRoad: 'Hours on the road',
    any: '—',
    none: 'None given. The column marches, night or day, until it has had the rules’ hours on the road in the last 24.',
    dawn: (time: string): string => `Dawn (${time})`,
    startAtDawn: 'steps off at dawn',
    start: (time: string): string => `steps off at ${time}`,
    latest: (time: string): string => `off the road by ${time}`,
    max: (hours: number): string => `${hours} h on the road`,
    /** The three limits as a sentence: `Steps off at 05:00 · off the road by 19:00`. */
    summary: (parts: readonly string[]): string =>
      parts.join(' · ').replace(/^./, (c) => c.toUpperCase()) + '.',
    save: 'Give these orders',
    lift: 'Lift them',
  },

  unitEdit: {
    heading: 'Set by hand',
    blurb: 'Write any value onto the formation outright, for what happened off the board. Only what you change is sent. A formation set here is set at once, with no hours to change it; a change under way is set beside it, and finishes when you say.',
    fields: {
      name: 'Name',
      kind: 'Arm',
      echelon: 'Echelon',
      experience: 'Experience',
      formation: 'Formation',
      changeTo: 'Changing to',
      changeIn: 'Finishes in, h',
      parent: 'Reports to',
      corps: 'Corps',
      traits: 'Traits',
      paperStrength: 'Paper strength',
      fatigue: 'Fatigue',
      morale: 'Morale',
      provisions: 'Provisions',
      maxProvisions: 'Most provisions',
      equipment: 'Equipment',
      maxEquipment: 'Most equipment',
      guns: 'Guns',
      marchSpeedKmh: 'March speed, km/h',
      spacingM: 'Spacing, m a soldier',
      spacingMultiplier: 'Spacing multiplier',
      hoursMarchedToday: 'Hours since last rest',
      roadHoursLast24: 'On the road, last 24 h',
    },
    hoursHint: 'Hours on the road in the last 24 are what the march cap reads; hours since the last rest are what the fatigue table reads. Setting the first lays them down as the hours just gone.',
    save: 'Set these',
    reset: 'Undo changes',
    noChange: 'No change',
    reassign: 'Give it to them',
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
    sightingsAttached: (n: number): string =>
      n === 1 ? '1 sighting attached' : `${n} sightings attached`,

    forward: 'Forward',

    sentHeading: 'Sent',
    emptyOutbox: 'You have written nothing yet.',
    to: (who: string): string => `To ${who}`,
    /**
     * The formation an officer rides with, and whose side they are on.
     *
     * Under the name rather than beside it: the officer is who the despatch is from, and
     * the formation is which body of troops that makes it about. Both are wanted — "Ney"
     * alone is a name a new player has not learned yet, and "1re Division" alone loses who
     * wrote it — but they are not equals, so they are not on one line.
     */
    commands: (unit: string, faction: string): string => `${unit} · ${faction}`,
    handed: 'Handed over on the spot — their column was touching yours.',
    /** The mechanic, not a missing feature. It is worded to make that unambiguous. */
    unknownFate: 'Whether it arrived, you will know only if they write back.',
    viaWaypoints: (n: number): string => ` · Rider sent via ${n} waypoint(s) of yours.`,
  },

  // -------------------------------------------------------------------------
  // Writing one
  // -------------------------------------------------------------------------
  composer: {
    heading: 'Write a despatch',
    from: 'From',
    to: 'To',
    /**
     * Why an officer is on the list at all, after their name.
     *
     * Nothing for a subordinate — writing down the chain is the common case and needs no
     * caption — and a word for the other two, so the reader can see the rule working.
     */
    relation: {
      superior: ' — your superior',
      subordinate: '',
      in_sight: ' — in sight',
    },
    /**
     * The same three reasons, as a tag at the end of a row on a phone, where each addressee
     * is a row of its own rather than a line in a dropdown and needs no dash to set it off.
     */
    relationTag: {
      superior: 'your superior',
      subordinate: '',
      in_sight: 'in sight',
    },
    /** The second line of an addressee's row on a phone: what they command, and for whom. */
    correspondentLine: (unit: string, faction: string): string => `${unit} · ${faction}`,
    theReferee: 'The referee — out of the game',
    /**
     * One line of an addressee list: who, what they command, and which side.
     *
     * A dropdown has no room for two lines, and an addressee chosen by name alone is the
     * easiest mistake to make in this interface — two officers of the same rank, and the
     * order goes to the wrong corps. The formation disambiguates them.
     */
    correspondent: (name: string, unit: string, faction: string): string =>
      `${name} — ${unit}, ${faction}`,

    despatchPlaceholder: 'Move on Quatre Bras with all speed; I expect you astride the crossroads by noon.',
    notePlaceholder: 'A question about the rules, or something you think the referee should know.',

    asReferee: 'Written in their name, and logged as yours. What the addressee makes of it is still a decision when it arrives.',
    whoMayBeWritten: 'A rider can be sent to your superior, to those directly beneath you, and to anyone on your side you can see. Anyone else, write through one of them.',
    isNote: 'Out of the game: no rider, no delay, and nobody but the referee reads it.',

    /** A commander is told why there is no estimate, rather than shown an empty space. */
    noEstimate: 'How long the ride takes is not yours to know. Your rider will go until they find the addressee, and nobody will tell you when they did.',
    nothingToRideTo: 'Nothing on the map to ride to yet. Send the despatch anyway — your rider will find them.',
    rideLabel: 'The ride:',
    rideEstimate: (hours: string, arrival: string): string =>
      ` about ${hours} h over the ground as it stands, arriving around ${arrival}. Your rider may be stopped on the way, and the sender is never told whether they were.`,

    sending: 'Sealing…',
    send: 'Send by rider',
    sendNote: 'Send to the referee',
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
    /** Between the sender's formation and the addressee's, in the log's second line. */
    towards: ' → ',

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
    referee_note: '— its commander has written to you, out of the game',
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
    empty: 'Nobody on this side yet.',

    // The tree.
    armyCommand: 'army command',
    fold: 'Hide who answers to them',
    unfold: 'Show who answers to them',
    foldedCount: (n: number): string => (n === 1 ? '1 beneath' : `${n} beneath`),
    patrol: ' · patrol',
    withYou: 'with you',
    /** After a formation's name, when it has no report: a name in the order of battle and nothing more. */
    noWord: ' · no word',
    /** A second officer at the same column, so neither row looks like the only one. */
    alsoRiding: (names: string): string => ` · also ${names}`,
    asLastHeard: 'As you last heard. The hour against a formation is when word of it was written, not where it is now.',
    uncommanded: 'Nobody commands',
    uncommandedNote: 'A hole in the chain of command: nobody can order these and nobody reports for them.',

    // A formation, opened.
    asOf: 'As of',
    columnWhere: 'Where',
    corps: 'Corps',
    columnStrength: 'Strength',
    columnFatigue: 'Fatigue',
    supply: 'Supply',
    supplyLine: (prov: number, maxProv: number, equip: number, maxEquip: number): string =>
      `provisions ${prov}/${maxProv} · equipment ${equip}/${maxEquip}`,
    columnDoing: 'Doing',
    underArms: (n: string): string => `${n} under arms`,
    morale: (has: number, max: number): string => `morale ${has}/${max}`,
    noWordDetail: 'Nobody has sent you word of this formation. You know its name, and who rides with it.',

    // The referee's hands on it.
    addSubordinate: '+ subordinate',
    addOfficer: '+ officer',
    addArmyCommand: '+ army command',
  },

  /** Raising and appointing: the referee's preparation for a game. */
  orbat: {
    asArmyCommand: 'A new army command: answering to nobody.',
    under: (superior: string): string => `A new command answering to ${superior}.`,
    toRideWith: (unit: string): string => `An officer to ride with ${unit}.`,

    name: 'Name',
    ridingWith: 'Riding with',
    namePlaceholder: '1re Division',
    arm: 'Arm',
    echelon: 'Echelon',
    echelonGuessed: 'from strength',
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

    commander: 'Commander',
    commanderNamePlaceholder: 'Marshal Ney',
    answersTo: 'Answers to',
    noSuperior: 'nobody — army command',
    appoint: 'Appoint',
    cancel: 'Cancel',
    appointBlurb: 'Appointing a commander does not give anybody a seat. Issue a link when you want somebody to play them.',

    // What is wrong with a draft, in the order a reader would find it.
    needsName: 'It needs a name.',
    needsCommander: 'It needs a commander. Every formation has one — a formation nobody commands can neither be ordered nor report.',
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
    marchedToday: 'On the road, last 24 h',
    marchedTodayValue: (done: string, cap: number): string => `${done} h of ${cap} h`,
    sinceRest: 'Since last rest',
    sinceRestHint: (restHours: number): string =>
      `Hours on the road since the column last had ${restHours} off it. Fatigue reads this.`,
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
    echelon: 'Echelon',
    echelonGuessed: 'from strength',
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
    blurbTouch: 'Tap the ground to read it, or a column to read the unit standing on it.',
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

  /**
   * How to use the console, behind the Help button.
   *
   * Written for somebody who has just been sent a link and has never seen the game, so it
   * names every control by the words on it. When a control's label changes above, its
   * mention here changes with it — the two are in one file so that is one search.
   *
   * Four topics, and a reader lands on the one for their seat: a commander has no use for
   * the clock, and a referee reading how to write a despatch is reading the wrong half.
   * Both halves stay one tab away, because a referee explaining the game to a player wants
   * to read the player's half.
   */
  help: {
    open: 'Help',
    openHint: 'How to use this console (?)',
    heading: 'How to use the console',
    close: 'Close',
    topicsLabel: 'Help topics',
    topics: {
      start: 'Getting started',
      map: 'The map',
      command: 'Commanding',
      referee: 'Refereeing',
    },

    guide: {
      start: [
        {
          heading: 'What this is',
          body: [
            'A refereed Napoleonic campaign on a generated map, where one hex is one kilometre.',
            'The referee sees everything and runs the clock. Each commander sees only what the formation they ride with can see, and what has been reported to them by despatch. Everything else is a guess.',
            'There are no accounts. Every seat is a link.',
          ],
        },
        {
          heading: 'Starting a campaign',
          body: [
            'On the front page, upload a world.json made with worldgen generate --model organic, or press Or run the demonstration for a ready-made scenario.',
            'You are handed one link per seat and one for yourself as referee. Send each player their own and keep yours. Links are stored only as hashes, so a lost one is reissued rather than looked up. Copy link in the order of battle does that for any commander.',
          ],
        },
        {
          heading: 'Joining one',
          body: [
            'Open the link your referee sent you. Your seat’s token is in the part of the address after the #, which never reaches the server.',
            'Once it has opened, this browser remembers the link and the address bar changes to the campaign’s own address. That address carries no token and is the one to bookmark.',
          ],
        },
        {
          heading: 'Coming back',
          body: [
            'The front page lists every campaign this browser holds a link for, and which seat it holds. Campaigns, at the top left of the console, takes you there.',
            'Forget on the front page, or Leave in the sidebar, drops this browser’s link. The campaign itself is untouched.',
          ],
        },
        {
          heading: 'On a phone',
          body: [
            'The sidebar splits into tabs along the bottom: Map, Post (Despatches for a referee), Command for a commander, and Order of battle.',
            'Whatever you tap on the map slides up in a sheet. Tap its handle to pull it up over the map, and again to push it back down.',
            'The header’s controls sit behind the ⋯ button: switching seats, shading, reach, the way back to Campaigns, and this help.',
          ],
        },
      ],

      map: [
        {
          heading: 'Moving around',
          body: [
            'Scroll or pinch to zoom, and drag to pan.',
            'Hover over a hex, or tap it, to read the ground in the sidebar: terrain, relief, how many hours each arm takes to enter it on and off road, and whether a river there can be crossed.',
            'Click or tap a formation to select it, and it stays in the sidebar while you read the ground around it.',
          ],
        },
        {
          heading: 'Reading a symbol',
          body: [
            'Formations are NATO symbols in their side’s colour, and a symbol shows exactly as much as is known about it.',
          ],
          terms: [
            ['Solid frame', 'Seen now. What is drawn is true.'],
            ['Dashed frame', 'Reported, not seen. Either one of your own formations as its last despatch described it, or a sighting of the enemy.'],
            ['Empty frame', 'Something is there, and nobody can say what.'],
            ['Icon inside', 'The arm is known: horse, foot or guns.'],
            ['Marks above', 'The rough size is known.'],
            ['Ribbon behind', 'The road the column takes up. A division is kilometres of road, not a counter on one hex.'],
          ],
        },
        {
          heading: 'Shading',
          body: [
            'A commander’s map is washed by how much is known. Press v, or the shading button in the header, to cycle through three settings.',
          ],
          terms: [
            ['Watched · marched · unknown', 'Clear where your troops can see now, dimmer where they have been, darkest where nobody of yours has been.'],
            ['Watched only', 'Everything not in sight right now is dark.'],
            ['No shading', 'The map as the survey drew it, for reading the terrain.'],
          ],
        },
        {
          heading: 'Reach',
          body: [
            'Select a formation and tick Reach of selected, 10 h to shade every hex it could get to in ten hours of marching.',
            'It is worked out over the map you hold. For a commander, that makes it only as good as what you know of the country and of who is on the roads.',
          ],
        },
        {
          heading: 'Keys',
          terms: [
            ['v', 'Cycle the shading.'],
            ['Esc', 'Stop pointing at the map, or close whatever is open.'],
            ['?', 'Open this help.'],
          ],
          body: [],
        },
      ],

      command: [
        {
          heading: 'Who you are',
          body: [
            'You are one officer, riding with one formation. The line across the top of the map says who, and how many hexes you can see from where you stand.',
            'Your own formation is live. Everything else is dated: the formations under you as they last reported, and the enemy wherever somebody last said they saw them.',
          ],
        },
        {
          heading: 'Under my command',
          body: [
            'The Command tab (below the post on a wide screen) lists your own formation and everybody who answers to you.',
            'The hour against each is when word of it was written, not where it is now. A formation you last heard from six hours ago could be a long way from its marker.',
            'Order of battle, in the header, shows your whole side’s chain of command as you know it. A formation marked no word is one nobody has reported on to you.',
          ],
        },
        {
          heading: 'Giving orders',
          body: [
            'Orders are prose. Nothing on the map moves when you click it: you write what you want, and the referee reads it and sets the formation marching.',
            'Standing orders, under your command, are the exception. They say when the head of your column is on the road each day: when it steps off, when it must be off the road, and how many hours it may march. The column halts at whichever limit comes first, every day, until you change them. Give them with Give these orders; clear them with Lift them.',
          ],
        },
        {
          heading: 'Writing a despatch',
          body: [
            'Press Write a despatch in the post. A rider can be sent to your superior, to anyone directly beneath you, and to anyone on your side you can see. To reach anybody else, write through one of them.',
            'How long the ride takes is not yours to know. Your rider goes until they find the addressee, and may be stopped on the way without your being told. Whether it arrived, you learn only if they write back.',
            'The referee is on the list too, as The referee — out of the game. A note to them goes at once, with no rider, and nobody else reads it. Use it for rules questions, or for anything the referee should know.',
          ],
        },
        {
          heading: 'Reading the post',
          body: [
            'In my hand holds what has reached you: when it was written, when it arrived, and how long it was on the road. Anything still riding toward you is invisible until it arrives.',
            'Forward opens a new despatch with the text already in it, so you can pass it on to whoever needs it.',
            'Sent lists what you have written.',
          ],
        },
        {
          heading: 'Contacts',
          body: [
            'A contact is a sighting of the enemy, graded by what the patrol that saw it could make out. Select one to read its grade and how long ago it was seen.',
            'The number on a contact is your staff’s own. Whether two contacts are the same body of troops is your judgement, not a fact you have been given.',
          ],
          terms: [
            ['1', 'Something is there. Nothing more.'],
            ['2', 'Presence and position.'],
            ['3', 'Presence, position and the direction of march.'],
            ['4', 'Rough strength.'],
            ['5', 'The arm: horse, foot or guns.'],
            ['6', 'The formation identified by name.'],
          ],
        },
      ],

      referee: [
        {
          heading: 'The clock',
          body: [
            'Nothing happens until you move the clock. +1 h and +6 h advance it by that much.',
            'Run is the one you will use most. It advances until something needs a decision and stops there, so you are never told afterwards that two corps met ninety minutes into a six-hour jump.',
            'The sun or moon beside the hour shows day or night. With nothing selected, the sidebar’s Daylight section sets sunrise and sunset as the season moves. Night marching is charged in fatigue by those hours.',
          ],
        },
        {
          heading: 'Wants a decision',
          body: [
            'Each time the clock stops, the reason joins the queue at the top of the sidebar: a column has come into contact, cannot get across a river, has arrived, is out of provisions, has received a despatch, and so on.',
            'Each card carries the controls to answer it. March them somewhere points the formation somewhere new. Dealt with clears the card. When two columns contest a hex and neither is faster, Give it to … settles who has it. Dealt with settles nothing, and they will ask again.',
            'A despatch arriving is a decision too: read it, then march the addressee’s formation as the orders say.',
            'A patrol running into something tells you how many dice to roll: any 1 and the patrol is lost, otherwise it falls back 2 km.',
          ],
        },
        {
          heading: 'Marching a formation',
          body: [
            'Select a formation and press March them somewhere, then point at the ground on the map. Point again to send them by way of somewhere first. The last place you point at is where they end up. Undo last takes back one point, and March sends them.',
            'They find their own road between the places you name, and their planned route is drawn on your map in their side’s colour. Halt stops a march.',
            'Place puts a formation on the ground without marching it. Use it to set up a scenario or correct a mistake. It is logged as what it is.',
          ],
        },
        {
          heading: 'Formation, patrols and standing orders',
          body: [
            'The formation buttons (March, Battle, Rest, Occupation) show how many hours each change takes, and the change finishes on the clock.',
            'A formation with the Scout trait can Send out a patrol. The first few are free. After that the button shows the troopers each one costs, and that cost is permanent.',
            'Standing orders work as they do for a commander. Set them yourself for an officer you run, or when a despatch gives them.',
          ],
        },
        {
          heading: 'Set by hand',
          body: [
            'Under a selected formation, Set by hand writes any value onto it outright: strength, fatigue, morale, supply, hours marched, even who it reports to. Use it for whatever happened off the board. Only the fields you change are sent.',
          ],
        },
        {
          heading: 'Battles',
          body: [
            'Press Battle beside the clock and point at the ground being fought over. Point at it again to take it back out. Press Done when the field is drawn.',
            'Inside a battle the traffic rules stop applying, since the formations there are intermingled. The map does not resolve the fighting. That is for you and the players.',
          ],
        },
        {
          heading: 'The post',
          body: [
            'The Despatches tab (below the queue on a wide screen) logs every despatch: who wrote it, where the rider has got to, and whether it arrived, was lost or was captured. Riders are drawn on your map and on nobody else’s.',
            'Write on a commander’s behalf sends a despatch in the name of an officer you run, or for a player who handed you an order on paper. It rides and can be intercepted like any other.',
          ],
        },
        {
          heading: 'The order of battle',
          body: [
            'Order of battle, in the header, is where you set up the game. Start with + side for each army, with a name and colour. Then + army command for the top of each chain, + subordinate under a formation, and + officer for a second officer riding with a formation. Each new formation asks you to point at the ground it stands on.',
            'Appointing a commander does not give anybody a seat. Copy link on a commander gives you the link to send to whoever plays them, and asking again gives the same link.',
          ],
        },
        {
          heading: 'Seeing as a commander',
          body: [
            'The seat buttons in the header (behind ⋯ on a phone) switch this console to any commander whose link you hold. It genuinely asks the server as that commander, so you see only what they see. Switch back with Referee.',
          ],
        },
      ],
    } as Readonly<Record<HelpTopic, readonly HelpSection[]>>,
  },
};

/** One entry in the help: a heading, some paragraphs, and optionally a list of terms. */
export interface HelpSection {
  readonly heading: string;
  readonly body: readonly string[];
  readonly terms?: readonly (readonly [string, string])[];
}

/** The help's tabs, in the order they are shown. */
export type HelpTopic = 'start' | 'map' | 'command' | 'referee';

/** A trigger in the referee's language, falling back to a sentence for an unknown one. */
export const triggerLabel = (trigger: string): string =>
  (copy.triggers as Record<string, string | undefined>)[trigger] ?? copy.triggers.unknown;

export type Copy = typeof copy;
