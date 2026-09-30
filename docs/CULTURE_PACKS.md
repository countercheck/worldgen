# Writing a culture pack

A culture pack tells the generator how one people names places. It is a YAML file. The
generator reads the land around each settlement, turns what it finds into *meanings* such
as "ford", "marsh" or "north of the town", and the pack turns those meanings into that
people's words.

Seventeen packs ship with worldgen, in `worldgen/naming/packs/`. The quickest way to write a
new one is to copy the built-in pack closest to what you want and change its words.

## Using a pack

```yaml
# worldgen.yaml
naming_cultures: 3
naming_packs: [mercian, norse]       # region 1 speaks mercian, region 2 norse, region 3 an invented language
naming_substrate_pack: welsh         # an older people named the rivers
naming_pack_dirs: [my_packs]         # folders of your own pack files, read after the built-in ones
```

- Every `*.yaml` file in a folder listed under `naming_pack_dirs` is loaded.
- A pack is referred to by its `key`, not by its file name.
- If your pack has the same key as a built-in pack, yours replaces it. This is how you
  change a built-in pack without editing the repository.
- A key that no loaded pack has is refused when the world is generated, and the error
  lists every pack that is available.

A generated world records each pack it used in `metadata.cultures`:

- `pack`: the pack's key.
- `pack_source`: `builtin` or `user`.
- `pack_hash`: a fingerprint of the pack's contents.

If you edit a pack and regenerate, the hash changes, so you can tell a world named with
the old version from one named with the new. Editing comments or layout does not change
the hash.

## Renaming a saved world

`worldgen rename` names a world.json again, keeping its ground, settlements and roads:

```bash
worldgen rename --input world.json --output renamed.json --packs english,norse
worldgen rename --input world.json --output renamed.json --config naming.yaml
```

| Option | What it does |
|---|---|
| `--packs a,b` | `naming_packs`. Raises `naming_cultures` to fit if the list is longer. |
| `--cultures N` | `naming_cultures`, the number of regions. |
| `--substrate-pack k` | `naming_substrate_pack`; `none` for no older people, so each river is named by whoever holds its mouth. |
| `--config file` | A config file whose `naming_*` settings are used; anything else in it is ignored, and a naming setting it leaves out takes its default. |
| `--seed N` | The naming seed. Defaults to the world's own. |

Everything else — the world's size, terrain, the settings the stage reads the ground with —
comes from the config the world recorded. Renaming with the world's own seed and naming
settings gives back exactly the names it has, so a new seed or new packs is the only thing
that makes a difference. The new world records its naming settings in `metadata.config`
and the seed it was named with in `metadata.naming_seed`.

## How a name is built

For each settlement the generator chooses:

- a **head**: what kind of place it is (ford, harbour, farm, stronghold);
- one of:
  - a **qualifier** (marsh, oak, black, north);
  - a **founder**;
  - the **river** it stands on;
  - nothing.

The pack picks a word for each, then drops them into one of its **templates**. Template
parts are:

| Placeholder | What goes there |
|---|---|
| `{h}` | a word from `heads` |
| `{q}` | a word from `quals`, as written (any `~` removed) |
| `{a}` | the same qualifier word, with its `~` replaced by the ending that agrees with the head (see [Agreement](#agreement)) |
| `{p}` | a founder's name, from `proper_names` |
| `{r}` | the name of the river the place stands on |

Every word is lower-case in the template. Each part of the finished name is then
capitalised, with two exceptions:

- the little words you list as `particles`, such as `de` or `on`;
- if `hyphen_caps` is false, anything after a hyphen.

So `"{h}-on-{r}"` with the head `ford` and the river `Avon` gives `Ford-on-Avon`.

## The file, field by field

```yaml
key: mercian            # required. Lower-case letters, digits, underscores. How configs refer to the pack.
name: Mercian           # required. What the people are called; shown as the culture on the map.
description: |          # optional. Free text about where the words come from.
  Old English of the Midlands.

heads:                  # required. One entry for every head meaning (listed below).
  ford: [ford]
  bridge: [bridge, brigg]      # several words: one is picked at random for each name
  ...

quals:                  # required. One entry for every qualifier meaning (listed below).
  marsh: [mars, more]
  ...

templates:
  compound: ["{q}{h}"]           # required. A head and a qualifier: Ashford.
  adjective: ["{h} {a}"]         # optional. Used instead of compound when the qualifier word has a "~".
  founder:  ["{p}ing{h}"]        # required. A head and a founder: Edricington.
  river_on: ["{r}{h}", "{h}-on-{r}"]   # required. A head and its river: Avonmouth.
  bare:     ["{h}"]              # optional, default ["{h}"]. A head alone.

ranked:                 # optional. Templates that change with the settlement's rank.
  compound:             #   exactly three lists: rank 0 (city), rank 1 (town), rank 2 (village)
    - ["{h}-of-the-{q}"]
    - ["{h}-by-{q}"]
    - ["{q}-{h}"]

agreement:              # optional. For languages whose adjectives agree with the noun.
  genders: {fuente: f, vado: m, casas: fp}
  endings: {m: o, f: a, fp: as}

proper_names:           # required. How founders and rivers are named. See below.
  style: lists
  persons: {first: [Ead, Wulf], second: [ric, wine]}
  rivers:  {roots: [Tame, Leam], patterns: ["{root}", "little {root}"]}

spelling:               # optional
  particles: ["on", upon]        # stay lower-case inside a name
  hyphen_caps: true              # false: Khazad-dûm rather than Khazad-Dûm
  fixes:                         # [pattern, replacement] regexes applied to the finished name
    - ["ius#", "ii"]

founder_gloss: "{p}'s {h}"   # optional. How a founder reads in the settlement's stored meaning (its etymology).

notes:                  # optional. A note on any meaning's word, such as that you invented it.
  swine: "a hound; no word for swine is attested"
```

A template is picked at random from its list each time. Repeating a template in the list
makes it more likely to be picked.

### Head meanings (`heads`)

Every pack needs a word, or a list of words, for all 24 of these:

`ford` · `bridge` · `confluence` (watersmeet) · `mouth` (river mouth) · `falls` · `pass` ·
`harbour` · `shore` · `market` · `mine` · `clearing` · `spring` · `hill` · `steading` (farm) ·
`hamlet` · `enclosure` · `cottages` · `field` · `town` · `wick` (trading place) ·
`stow` (meeting place) · `burgh` (stronghold) · `hall` · `wall` (walled town)

Villages draw from `steading`, `hamlet`, `enclosure`, `cottages` and `field`; towns from
`town`, `wick` and `stow`; cities from `burgh`, `hall` and `wall`. The others come from the
ground or from what the settlement does: a ford, a harbour, a mine.

### Qualifier meanings (`quals`)

Every pack needs all 41 of these:

`marsh` · `fen` · `deepwood` · `wood` · `thorn` · `meadow` · `cold` · `sand` · `stone` ·
`salt` · `rich` · `high` · `oak` · `ash` · `elm` · `birch` · `pine` · `palm` · `broom` ·
`reed` · `ox` · `deer` · `swine` · `horse` · `hare` · `elk` · `wolf` · `bear` · `goat` ·
`crane` · `eagle` · `white` · `black` · `red` · `green` · `north` · `south` · `east` ·
`west` · `great` · `little`

A qualifier word can include its own little words, such as `del bosque` or `de la selva`.
That is how Spanish, French and Italian put a noun after the head.

## Agreement

Write an adjective as a stem followed by `~`, such as `blanc~`, and give each head word a
gender:

```yaml
agreement:
  genders: {fuente: f, puerto: m, casas: fp}
  endings: {m: o, f: a, fp: as}
quals:
  white: ["blanc~"]
templates:
  compound:  ["{h} {q}"]             # nouns:      Fuente del Roble
  adjective: ["{h} {a}", "{h}{a}"]   # adjectives: Fuente Blanca, Fuenteblanca
```

- `{a}` becomes `blanca` after `fuente` and `blanco` after `puerto`.
- `{q}` drops the `~` and adds no ending. Slavic uses this to make one-word names:
  `dubov~` in `"{q}ec"` gives Dubovec.
- If any qualifier uses `~`, every head word needs a gender, and every gender needs an
  ending.

## Founders and rivers: `proper_names`

**`style: lists`.**

- A founder's name is a word from `persons.first` joined to a word from `persons.second`.
  Give `second: [""]` if your names are single words.
- A river's name is a word from `rivers.roots` put into one of `rivers.patterns` (the
  pattern must contain `{root}`).
- If the map needs more names than the lists hold, names are joined in pairs, so you never
  run out.

**`style: syllables`.** Names are built from sounds instead of taken from lists. This is
how the hive pack works:

```yaml
proper_names:
  style: syllables
  registers:
    buzz:  {onsets: [zz, vr, sh], vowels: [aa, u, i], codas: [rr, mm, ""]}
    click: {onsets: ["k'", "tz'"], vowels: [i, a],    codas: [k, t, ""]}
  by_rank:
    - [buzz]          # rank 0: cities, great rivers
    - [buzz, click]   # rank 1: either register
    - [click]         # rank 2: villages, lesser rivers
```

Each syllable is one onset, one vowel and one coda. An empty `""` in a list means "none".

## Rank

The generator tells the pack how important each place is:

- rank 0 is a city, 1 a town, 2 a village;
- for rivers, 0 is a great river and 2 a lesser one.

Most packs ignore rank. Use `ranked` templates, or `by_rank` registers, when a people
names its great places differently from its small ones.

## When a pack is wrong

A pack is checked when it is loaded, and every error names the file and the field. For
example:

```
mercian.yaml: heads is missing 'ford'
mercian.yaml: templates.founder[0] uses {q}; a founder template may use only {h}, {p}
mercian.yaml: agreement.genders has no gender for head word 'brycg'
mercian.yaml: spelling.particles[0] is true, not a word: YAML reads a bare on, off, yes or no as true or false — put the word in quotes
```

The last one is the easiest mistake to make. YAML reads a bare `on`, `off`, `yes` or `no`
as true or false, so write `"on"` in quotes. If in doubt, quote every word.

## A complete small pack

This is valid as it stands. Save it as `my_packs/mercian.yaml`, then set
`naming_packs: [mercian]` and `naming_pack_dirs: [my_packs]`.

```yaml
key: mercian
name: Mercian
heads:
  ford: [ford]
  bridge: [brycg]
  confluence: [gemot]
  mouth: [mutha]
  falls: [hlynn]
  pass: [gap]
  harbour: [hyth]
  shore: [strand]
  market: [ceap]
  mine: [pytt]
  clearing: [leah]
  spring: [wella]
  hill: [dun]
  steading: [tun, ham]
  hamlet: [stede]
  enclosure: [worth]
  cottages: [cot]
  field: [feld]
  town: [tun]
  wick: [wic]
  stow: [stow]
  burgh: [burh]
  hall: [heall]
  wall: [weal]
quals:
  marsh: [mersc]
  fen: [fen]
  deepwood: [holt]
  wood: [wudu]
  thorn: [thorn]
  meadow: [mæd]
  cold: [ceald]
  sand: [sand]
  stone: [stan]
  salt: [sealt]
  rich: [eadig]
  high: [heah]
  oak: [ac]
  ash: [æsc]
  elm: [elm]
  birch: [beorc]
  pine: [furh]
  palm: [palm]
  broom: [brom]
  reed: [hreod]
  ox: [oxna]
  deer: [heorot]
  swine: [swin]
  horse: [hors]
  hare: [hara]
  elk: [eolh]
  wolf: [wulf]
  bear: [bera]
  goat: [gat]
  crane: [cran]
  eagle: [earn]
  white: [hwit]
  black: [blæc]
  red: [read]
  green: [grene]
  north: [north]
  south: [suth]
  east: [east]
  west: [west]
  great: [micel]
  little: [lytel]
templates:
  compound: ["{q}{h}"]
  founder: ["{p}ing{h}", "{p}s{h}"]
  river_on: ["{r}{h}"]
proper_names:
  style: lists
  persons: {first: [Ead, Wulf, Os, Beorn], second: [ric, wine, mund]}
  rivers: {roots: [Tame, Leam, Sowe, Arrow], patterns: ["{root}", "{root}brook"]}
```
