/**
 * The order of battle, in a drawer, drawn as the chain of command.
 *
 * The sidebar answers "what is this one formation". This answers the other question a
 * referee asks constantly — "who commands what, and what is everything doing" — without
 * giving up the map to a table.
 *
 * ## Two relationships, drawn two ways
 *
 * A commander has a place in the chain of command and a formation they ride with, and the
 * two are different things. The tree is built on the first: each officer, and beneath them
 * the officers who answer to them. The second hangs off each officer as a formation row —
 * where they are, and the troops they are standing among. A marshal rides with one division
 * and commands a corps, and a drawing that nested the corps under the division would have
 * the army upside down.
 *
 * ## Two very different lists
 *
 * A referee gets every side, in tabs, and every formation as it is. A commander gets their
 * own side only, their own formation as it is, and every other one *as they last heard*: a
 * `UnitReport` carries the hour it describes, not the hour it arrived. A formation they
 * have no word of at all — their superior's, a peer's — is a name and nothing more. See
 * `commandTrees`, which builds from what the role holds rather than filtering one list.
 */

import { useState, type ReactNode } from 'react';

import {
  ECHELON_MARKS,
  isPatrol,
  maxMorale,
  presentUnderArms,
  type CampaignConfig,
  type Commander,
  type Faction,
  type Hex,
  type PublicCommander,
  type PublicFaction,
  type Unit,
  type UnitReport,
} from '@campaign/shared';

import { ageLabel, dayHour } from '../board.js';
import { copy, prettify } from '../copy.js';
import {
  beneath,
  commandTrees,
  type CommandNode,
  type FormationNode,
  type RosterLine,
} from '../roster.js';
import { AppointForm, FactionForm, RaiseForm } from './Orbat.jsx';

/** The referee's order-of-battle controls. Absent for a commander, who raises nothing. */
export interface OrbatEditing {
  readonly cfg: CampaignConfig;
  readonly placing: Hex | null;
  readonly onPlace: () => void;
  readonly onRaise: (unit: Unit, commander: Commander) => void;
  readonly onAppoint: (commander: Commander) => void;
  /** Add a side. A campaign starts with none. */
  readonly onAddFaction: (faction: Faction) => void;
  /**
   * The join link for a commander's seat: the one this browser already holds, or a new one.
   * Rejects when the server will not issue it.
   */
  readonly linkFor: (commanderId: string) => Promise<string>;
  /** A form was closed without raising anything: forget the hex it was pointed at. */
  readonly onDiscard: () => void;
  readonly busy: boolean;
  readonly error: string | null;
}

/** Where a form is open: under a commander, at the top of a side, or on a formation. */
type Editing =
  | { readonly kind: 'raise'; readonly superiorId: string | null }
  | { readonly kind: 'appoint'; readonly unitId: string }
  | { readonly kind: 'side' }
  | null;

/** A commander's link, as far as the referee has got in asking for it. */
type LinkState =
  | { readonly kind: 'fetching' }
  | { readonly kind: 'shown'; readonly link: string; readonly copied: boolean }
  | { readonly kind: 'failed'; readonly why: string };

export function Roster({
  open,
  onClose,
  role,
  factions,
  commanders,
  units,
  reports,
  ownUnitId,
  clockHours,
  cfg,
  colorOf,
  selectedId,
  onSelect,
  taskOf,
  editing,
}: {
  open: boolean;
  onClose: () => void;
  role: 'referee' | 'commander';
  /** The sides to show, one tab each: every side for a referee, their own for a commander. */
  factions: readonly PublicFaction[];
  commanders: readonly PublicCommander[];
  /** Live formations: all of them for a referee, their own for a commander. */
  units: readonly Unit[];
  /** Dated reports. Empty for a referee, who has no need of them. */
  reports: readonly UnitReport[];
  ownUnitId: string | null;
  clockHours: number;
  cfg: CampaignConfig;
  colorOf: (faction: string) => string;
  selectedId: string | null;
  onSelect: (unitId: string) => void;
  taskOf: (unitId: string) => string | null;
  editing?: OrbatEditing;
}) {
  const [tab, setTab] = useState<string | null>(null);
  const [folded, setFolded] = useState<ReadonlySet<string>>(new Set());
  const [shown, setShown] = useState<ReadonlySet<string>>(new Set());
  const [form, setForm] = useState<Editing>(null);
  const [links, setLinks] = useState<Readonly<Record<string, LinkState>>>({});

  const trees = commandTrees({
    factions: factions.map((f) => f.id),
    commanders,
    units,
    reports,
  });
  const current = trees.find((t) => t.faction === tab) ?? trees[0];
  const faction = factions.find((f) => f.id === current?.faction);

  const toggle = (set: ReadonlySet<string>, id: string): Set<string> => {
    const next = new Set(set);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    return next;
  };

  const closeForm = (): void => {
    setForm(null);
    editing?.onDiscard();
  };

  /**
   * Put a commander's link in front of the referee, and on the clipboard where allowed.
   *
   * Shown as well as copied. The clipboard needs a secure page and, on some phones, a
   * gesture it will not count after a network round trip, and a copy that silently fails
   * leaves the referee pasting whatever they copied last.
   */
  const copyLink = (commanderId: string): void => {
    if (editing === undefined) return;
    setLinks((l) => ({ ...l, [commanderId]: { kind: 'fetching' } }));
    editing
      .linkFor(commanderId)
      .then(async (link) => {
        let copied = false;
        try {
          await navigator.clipboard.writeText(link);
          copied = true;
        } catch {
          /* Shown below instead, to be copied by hand. */
        }
        setLinks((l) => ({ ...l, [commanderId]: { kind: 'shown', link, copied } }));
      })
      .catch((err: unknown) => {
        const why = String((err as Error).message ?? err);
        setLinks((l) => ({ ...l, [commanderId]: { kind: 'failed', why } }));
      });
  };

  const linkLine = (commanderId: string): ReactNode => {
    const state = links[commanderId];
    if (state === undefined || state.kind === 'fetching') return null;
    if (state.kind === 'failed') {
      return <p className="error small">{copy.roster.linkFailed(state.why)}</p>;
    }
    return (
      <div className="cmd-link">
        <p className="muted small">{state.copied ? copy.roster.linkCopied : copy.roster.linkToCopy}</p>
        <code
          className="link"
          onClick={(e) => window.getSelection()?.selectAllChildren(e.currentTarget)}
        >
          {state.link}
        </code>
      </div>
    );
  };

  const officers = commanders.map((c) => ({ id: c.id, name: c.name, faction: c.faction }));

  const raiseForm = (superior: CommandNode | null): ReactNode =>
    editing !== undefined &&
    faction !== undefined && (
      <RaiseForm
        faction={faction.id}
        superior={superior === null ? null : { ...superior, faction: faction.id }}
        units={units}
        commanders={officers}
        cfg={editing.cfg}
        placing={editing.placing}
        onPlace={editing.onPlace}
        onRaise={(unit, commander) => {
          editing.onRaise(unit, commander);
          setForm(null);
        }}
        onCancel={closeForm}
        busy={editing.busy}
      />
    );

  const formationRow = (f: FormationNode): ReactNode => {
    const expanded = shown.has(f.unitId);
    const line = f.line;
    const patrol = line?.unit !== null && line?.unit !== undefined && isPatrol(line.unit);
    const appointing = form?.kind === 'appoint' && form.unitId === f.unitId;

    return (
      <li key={f.unitId} className="cmd-formation">
        <div className={`cmd-row${f.unitId === selectedId ? ' selected' : ''}`}>
          <button
            className="cmd-unit"
            aria-expanded={expanded}
            onClick={() => {
              setShown(toggle(shown, f.unitId));
              if (line !== null) onSelect(f.unitId);
            }}
          >
            <span
              className={`swatch small${line?.asOfHours == null ? '' : ' ghost'}`}
              style={{ background: colorOf(line?.faction ?? faction?.id ?? '') }}
            />
            {f.name}
            {line !== null && ECHELON_MARKS[line.echelon] !== '' && (
              <span className="cmd-echelon" title={prettify(line.echelon)}>
                {ECHELON_MARKS[line.echelon]}
              </span>
            )}
            {patrol && <span className="muted">{copy.roster.patrol}</span>}
            {line === null && <span className="muted">{copy.roster.noWord}</span>}
            {line !== null && line.asOfHours !== null && (
              <span className="muted" title={dayHour(line.asOfHours)}>
                {' · '}
                {ageLabel(line.asOfHours, clockHours)}
              </span>
            )}
            {f.unitId === ownUnitId && <span className="muted"> · {copy.roster.withYou}</span>}
            {f.alsoRiding.length > 0 && (
              <span className="muted">{copy.roster.alsoRiding(f.alsoRiding.join(', '))}</span>
            )}
          </button>
          {editing !== undefined && !patrol && (
            <button
              className="cmd-action"
              onClick={() => setForm(appointing ? null : { kind: 'appoint', unitId: f.unitId })}
            >
              {copy.roster.addOfficer}
            </button>
          )}
        </div>

        {expanded && <Details line={line} task={taskOf(f.unitId)} cfg={cfg} clockHours={clockHours} />}

        {appointing && editing !== undefined && (
          <AppointForm
            unit={{ id: f.unitId, name: f.name, faction: line?.faction ?? faction?.id ?? '' }}
            commanders={officers}
            onAppoint={(c) => {
              editing.onAppoint(c);
              setForm(null);
            }}
            onCancel={() => setForm(null)}
            busy={editing.busy}
          />
        )}

        {f.patrols.length > 0 && <ul className="cmd-patrols">{f.patrols.map(formationRow)}</ul>}
      </li>
    );
  };

  const commanderNode = (c: CommandNode, root: boolean): ReactNode => {
    const isFolded = folded.has(c.id);
    const raising = form?.kind === 'raise' && form.superiorId === c.id;

    return (
      <li key={c.id} className="cmd-node">
        <div className="cmd-row">
          <button
            className="cmd-fold"
            aria-label={isFolded ? copy.roster.unfold : copy.roster.fold}
            aria-expanded={!isFolded}
            disabled={c.subordinates.length === 0}
            onClick={() => setFolded(toggle(folded, c.id))}
          >
            {c.subordinates.length === 0 ? '·' : isFolded ? '▸' : '▾'}
          </button>
          <span className="cmd-name">{c.name}</span>
          {root && <span className="muted small">{copy.roster.armyCommand}</span>}
          {isFolded && (
            <span className="muted small">{copy.roster.foldedCount(beneath(c))}</span>
          )}
          {editing !== undefined && (
            <button
              className="cmd-action"
              onClick={() => setForm(raising ? null : { kind: 'raise', superiorId: c.id })}
            >
              {copy.roster.addSubordinate}
            </button>
          )}
          {editing !== undefined && (
            <button
              className="cmd-action"
              title={copy.roster.copyLinkHint}
              disabled={links[c.id]?.kind === 'fetching'}
              onClick={() => copyLink(c.id)}
            >
              {links[c.id]?.kind === 'fetching' ? copy.roster.fetchingLink : copy.roster.copyLink}
            </button>
          )}
        </div>

        {linkLine(c.id)}

        <ul className="cmd-rides">{formationRow(c.formation)}</ul>

        {raising && raiseForm(c)}

        {!isFolded && c.subordinates.length > 0 && (
          <ul className="cmd-tree">{c.subordinates.map((s) => commanderNode(s, false))}</ul>
        )}
      </li>
    );
  };

  // Hidden rather than unmounted: pointing at the map for a new formation closes the
  // drawer, and a half-filled form has to be there when it opens again.
  return (
    <aside className="roster" aria-label={copy.roster.label} style={open ? undefined : { display: 'none' }}>
      <header className="roster-head">
        <h2>{copy.roster.heading}</h2>
        <button className="dismiss" onClick={onClose}>
          {copy.roster.close}
        </button>
      </header>

      {(factions.length > 1 || (editing !== undefined && factions.length > 0)) && (
        <div className="tabs" role="tablist">
          {factions.map((f) => (
            <button
              key={f.id}
              role="tab"
              aria-selected={f.id === current?.faction}
              className={`tab${f.id === current?.faction ? ' on' : ''}`}
              onClick={() => {
                setTab(f.id);
                closeForm();
              }}
            >
              <span className="swatch small" style={{ background: f.color }} />
              {f.name}
              <span className="muted"> · {units.filter((u) => u.faction === f.id).length}</span>
            </button>
          ))}
          {editing !== undefined && (
            <button
              className={`tab${form?.kind === 'side' ? ' on' : ''}`}
              onClick={() => setForm(form?.kind === 'side' ? null : { kind: 'side' })}
            >
              {copy.roster.addSide}
            </button>
          )}
        </div>
      )}

      <div className="roster-body">
        {editing?.error != null && <p className="error">{editing.error}</p>}

        {editing !== undefined && factions.length === 0 && (
          <p className="muted small">{copy.roster.noSides}</p>
        )}

        {editing !== undefined && (form?.kind === 'side' || factions.length === 0) && (
          <FactionForm
            // Remounted per count, so the picker starts afresh away from the side just added.
            key={factions.length}
            factions={factions}
            onAdd={(faction) => {
              editing.onAddFaction(faction);
              setTab(faction.id);
              setForm(null);
            }}
            onCancel={() => setForm(null)}
            busy={editing.busy}
          />
        )}

        {role === 'commander' && reports.length > 0 && (
          <p className="muted small">{copy.roster.asLastHeard}</p>
        )}

        {factions.length === 0 ? null : current === undefined ||
          (current.roots.length === 0 && current.uncommanded.length === 0) ? (
          <p className="muted small">{copy.roster.empty}</p>
        ) : (
          <ul className="cmd-tree cmd-top">{current.roots.map((c) => commanderNode(c, true))}</ul>
        )}

        {editing !== undefined && faction !== undefined && (
          <div className="despatch-actions">
            <button
              onClick={() =>
                setForm(form?.kind === 'raise' && form.superiorId === null ? null : { kind: 'raise', superiorId: null })
              }
            >
              {copy.roster.addArmyCommand}
            </button>
          </div>
        )}
        {form?.kind === 'raise' && form.superiorId === null && raiseForm(null)}

        {current !== undefined && current.uncommanded.length > 0 && (
          <section className="roster-group">
            <h3>{copy.roster.uncommanded}</h3>
            <p className="muted small">{copy.roster.uncommandedNote}</p>
            <ul className="cmd-rides">{current.uncommanded.map(formationRow)}</ul>
          </section>
        )}
      </div>
    </aside>
  );
}

/**
 * What is known of one formation, opened in place under it.
 *
 * A report carries less than a unit does, and deliberately: a rider knows where a formation
 * was, roughly how strong and how tired, and not its ammunition. So a dated card shows what
 * the report holds and says when it was true, and never pads the rest with a blank that
 * could be read as a zero.
 */
function Details({
  line,
  task,
  cfg,
  clockHours,
}: {
  line: RosterLine | null;
  task: string | null;
  cfg: CampaignConfig;
  clockHours: number;
}) {
  if (line === null) return <p className="cmd-details muted small">{copy.roster.noWordDetail}</p>;

  const unit = line.unit;
  const patrol = unit !== null && isPatrol(unit);

  return (
    <dl className="cmd-details">
      {line.asOfHours !== null && (
        <>
          <dt>{copy.roster.asOf}</dt>
          <dd>
            {dayHour(line.asOfHours)} · {ageLabel(line.asOfHours, clockHours)}
          </dd>
        </>
      )}
      <dt>{copy.roster.columnWhere}</dt>
      <dd>
        {line.at.q}, {line.at.r}
      </dd>
      {line.corps !== null && (
        <>
          <dt>{copy.roster.corps}</dt>
          <dd>{line.corps}</dd>
        </>
      )}
      {!patrol && (
        <>
          <dt>{copy.roster.columnStrength}</dt>
          <dd>
            {line.paperStrength.toLocaleString()}
            {unit !== null &&
              ` · ${copy.roster.underArms(presentUnderArms(unit).toLocaleString())}`}
          </dd>
          <dt>{copy.roster.columnFatigue}</dt>
          <dd>
            {line.fatigue}
            {unit !== null && ` · ${copy.roster.morale(unit.morale, maxMorale(unit, cfg.maxMorale))}`}
          </dd>
        </>
      )}
      {unit !== null && !patrol && (
        <>
          <dt>{copy.roster.supply}</dt>
          <dd>{copy.roster.supplyLine(unit.provisions, unit.maxProvisions, unit.equipment, unit.maxEquipment)}</dd>
        </>
      )}
      <dt>{copy.roster.columnDoing}</dt>
      <dd>
        {line.formation}
        {task !== null && ` · ${task}`}
      </dd>
    </dl>
  );
}
