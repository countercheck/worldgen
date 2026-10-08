/**
 * Getting into a campaign.
 *
 * Two ways in, because there are two kinds of person here. A referee starts a campaign
 * from a `world.json` and is handed one link per side to send out; everybody else follows
 * a link they were sent. There is no third way and no account to make.
 *
 * The demo goes in through exactly the same door: the browser creates a campaign over the
 * API and posts the scenario's commands one at a time, as a referee would. That is slower
 * than assembling the state in the page, and it is the point — a demo built in the
 * browser would be the only campaign in the system whose fog was never enforced, which is
 * the last thing to put in front of somebody on their first look.
 */

import { useState } from 'react';

import {
  DEFAULT_CONFIG,
  DEMO_FACTIONS,
  demoCommands,
  parseWorld,
  type Command,
  type Faction,
} from '@campaign/shared';

import { createCampaign, fetchView, issueSeatToken, sendAll, type Session } from './api.js';
import { copy } from './copy.js';
import { parseOob, rehearse } from './oob.js';
import { SAMPLE_OOB_HREF, SAMPLE_OOB_NAME } from './sample.js';
import { Help } from './panels/Help.jsx';
import { campaignHash, navigate } from './route.js';
import { forgetCampaign, joinLink, listCampaigns, type CampaignSummary, type HeldTokens } from './session.js';

export interface Joined {
  readonly session: Session;
  readonly held: HeldTokens;
  readonly ownToken: string;
  /** Who each seat belongs to, so a link can be labelled with a name rather than an id. */
  readonly seats: Record<string, { name: string; faction: string }>;
}

/**
 * Build a campaign from a world document and put an order of battle on it.
 *
 * Three phases, in the order a real game is prepared: upload the world, put the order of
 * battle on the map, then issue a link per seat. The seats come last because a commander
 * needs a formation to ride with before anybody can be appointed to one.
 */
async function startCampaign(
  name: string,
  worldDoc: unknown,
  factions: readonly Faction[],
  commands: readonly Command[],
  onProgress: (message: string) => void,
): Promise<Joined> {
  onProgress(copy.join.uploadingWorld);
  const created = await createCampaign({ name, world: worldDoc, factions, seed: 20260906 });
  const session: Session = { campaignId: created.id, token: created.refereeToken };

  const result = await sendAll(session, commands, (done, total) =>
    onProgress(copy.join.formingArmy(done, total)),
  );
  if (!result.ok) {
    // A refusal is worth showing rather than swallowing: it means the engine
    // disagrees with the scenario, which is a real answer about the world.
    throw new Error(
      result.violations?.map((v) => v.message).join('; ') ?? copy.join.commandRefused,
    );
  }

  // One link per seat, so the referee can hand any of them to a player.
  onProgress(copy.join.issuingLinks);
  const view = await fetchView(session);
  const held: HeldTokens = {};
  const seats: Record<string, { name: string; faction: string }> = {};
  for (const commander of view.commanders) {
    const { token } = await issueSeatToken(session, commander.id);
    held[commander.id] = token;
    seats[commander.id] = { name: commander.name, faction: commander.faction };
  }

  return { session, held, ownToken: created.refereeToken, seats };
}

export function Join({
  onJoined,
  notice,
}: {
  onJoined: (joined: Joined) => void;
  /** Why they are looking at this page rather than the one they asked for, if they are. */
  notice?: string;
}) {
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [holding, setHolding] = useState<readonly CampaignSummary[]>(() => listCampaigns());
  const [links, setLinks] = useState<{
    /** What the referee called it, for the message that carries everybody's links. */
    name: string;
    referee: string;
    seats: { id: string; label: string; link: string }[];
  } | null>(null);
  const [pending, setPending] = useState<Joined | null>(null);
  const [help, setHelp] = useState(false);
  /** An order of battle chosen ahead of the world, which is what starts the campaign. */
  const [oob, setOob] = useState<{ name: string; text: string } | null>(null);

  const run = async (name: string, make: () => Promise<Joined>): Promise<void> => {
    setError(null);
    try {
      const joined = await make();
      // The links are shown before entering the campaign, because this is the only moment
      // they exist. They are minted here and stored only as hashes — a lost link is
      // reissued, never looked up — so a referee who clicks past this screen without
      // copying them has to start again.
      setLinks({
        name,
        referee: joinLink(joined.session),
        seats: Object.entries(joined.held).map(([id, token]) => ({
          id,
          label: joined.seats[id]?.name ?? id,
          link: joinLink({ campaignId: joined.session.campaignId, token }),
        })),
      });
      setPending(joined);
    } catch (err) {
      setError(String((err as Error).message ?? err));
    } finally {
      setBusy(null);
    }
  };

  const startDemo = (): void => {
    setBusy(copy.join.loadingDemoWorld);
    void run(copy.join.demoCampaignName, async () => {
      const { default: worldDoc } = await import(
        '../../shared/test/fixtures/world-32x32.json'
      );
      return startCampaign(
        copy.join.demoCampaignName,
        worldDoc,
        DEMO_FACTIONS,
        demoCommands(parseWorld(worldDoc)),
        setBusy,
      );
    });
  };

  const startFromFile = (file: File): void => {
    setBusy(copy.join.readingWorld);
    const name = file.name.replace(/\.json$/, '');
    void run(name, async () => {
      const worldDoc: unknown = JSON.parse(await file.text());
      if (oob === null) {
        // No sides: the referee names and colours each one once the map is up. Only the
        // demonstration and an uploaded order of battle arrive with theirs.
        return startCampaign(name, worldDoc, [], [], setBusy);
      }
      // Checked in full before the campaign exists, so a mistake in the file costs a
      // second try rather than a half-raised campaign to throw away.
      setBusy(copy.oob.readingFile);
      const world = parseWorld(worldDoc);
      const read = parseOob(oob.text, world, DEFAULT_CONFIG);
      if (!read.ok) throw new Error(read.problems.join('\n'));
      const refused = rehearse(world, read);
      if (refused.length > 0) throw new Error(refused.join('\n'));
      return startCampaign(name, worldDoc, read.newFactions, read.commands, setBusy);
    });
  };

  if (links !== null && pending !== null) {
    return (
      <div className="join">
        <h1>{copy.join.createdTitle}</h1>
        <p className="muted">{copy.join.createdBlurb}</p>
        <div className="links-tools">
          <CopyButton
            label={copy.join.copyAll}
            text={copy.join.allLinks(
              links.name,
              links.seats,
            )}
          />
        </div>
        <ul className="links">
          {links.seats.map((seat) => (
            <li key={seat.id}>
              <strong>{seat.label}</strong>
              <code>{seat.link}</code>
              <CopyButton label={copy.join.copyLink} text={seat.link} />
            </li>
          ))}
          {/* Last and apart: the one link not to send anybody. */}
          <li className="referee-link">
            <strong>{copy.join.refereeSeat}</strong>
            <code>{links.referee}</code>
            <CopyButton label={copy.join.copyLink} text={links.referee} />
          </li>
        </ul>
        <button className="primary" onClick={() => onJoined(pending)}>
          {copy.join.enterAsReferee}
        </button>
      </div>
    );
  }

  return (
    <>
    {help && <Help role={null} onClose={() => setHelp(false)} />}
    <div className="join">
      <h1>{copy.join.title}</h1>
      <p className="muted">{copy.join.blurb}</p>
      <button className="help-button" title={copy.help.openHint} onClick={() => setHelp(true)}>
        {copy.help.open}
      </button>

      {notice !== undefined && <p className="error">{notice}</p>}
      {busy !== null && <p className="busy">{busy}</p>}
      {error !== null && <p className="error">{error}</p>}

      {holding.length > 0 && (
        <section>
          <h3>{copy.join.resumeHeading}</h3>
          <p className="muted">{copy.join.resumeBlurb}</p>
          <ul className="held">
            {holding.map((c) => (
              <li key={c.campaignId}>
                <button className="resume" onClick={() => navigate(campaignHash(c.campaignId))}>
                  <span className="resume-name">{c.name ?? copy.join.unnamedCampaign}</span>
                  <span className="muted">
                    {c.isReferee ? copy.join.resumeReferee : copy.join.resumeCommander}
                  </span>
                </button>
                <button
                  className="dismiss"
                  title={copy.join.forgetHint}
                  onClick={() => {
                    forgetCampaign(c.campaignId);
                    setHolding(listCampaigns());
                  }}
                >
                  {copy.join.forget}
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}

      <section>
        <h3>{copy.join.startHeading}</h3>
        <label className="file">
          <input
            type="file"
            accept=".yaml,.yml,application/yaml,text/yaml"
            disabled={busy !== null}
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file === undefined) return setOob(null);
              void file.text().then((text) => setOob({ name: file.name, text }));
            }}
          />
          <span>{oob === null ? copy.oob.uploadLabel : copy.oob.chosen(oob.name)}</span>
        </label>
        <p className="muted">
          {copy.oob.uploadHint}{' '}
          <a href={SAMPLE_OOB_HREF} download={SAMPLE_OOB_NAME} title={copy.oob.sampleHint}>
            {copy.oob.sampleLink}
          </a>
        </p>
        <label className="file">
          <input
            type="file"
            accept="application/json,.json"
            disabled={busy !== null}
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file !== undefined) startFromFile(file);
            }}
          />
          <span>{copy.join.uploadLabel}</span>
        </label>
        <p className="muted">
          {copy.join.generateHintBefore} <code>{copy.join.generateCommand}</code>
          {copy.join.generateHintAfter}
        </p>
        <button onClick={startDemo} disabled={busy !== null}>
          {copy.join.demoButton}
        </button>
      </section>

      <section>
        <h3>{copy.join.joinHeading}</h3>
        <p className="muted">{copy.join.joinBlurb}</p>
      </section>
    </div>
    </>
  );
}

/**
 * A button that puts `text` on the clipboard and says so.
 *
 * The clipboard needs a secure page, and a copy that fails without a word leaves the
 * referee pasting whatever they copied last. So a failure says to copy by hand, and the
 * link beside it is selectable in one click for exactly that.
 */
function CopyButton({ label, text }: { label: string; text: string }) {
  const [state, setState] = useState<'idle' | 'copied' | 'failed'>('idle');
  return (
    <button
      className={`copy-button ${state}`}
      onClick={() => {
        navigator.clipboard
          .writeText(text)
          .then(() => setState('copied'))
          .catch(() => setState('failed'));
      }}
    >
      {state === 'copied' ? copy.join.copied : state === 'failed' ? copy.join.copyFailed : label}
    </button>
  );
}
