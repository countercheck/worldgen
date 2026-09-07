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

import { DEMO_FACTIONS, demoCommands, parseWorld, type Faction } from '@campaign/shared';

import { createCampaign, fetchView, issueSeatToken, sendCommand, type Session } from './api.js';
import { joinLink, type HeldTokens } from './session.js';

export interface Joined {
  readonly session: Session;
  readonly held: HeldTokens;
  readonly ownToken: string;
  /** Who each seat belongs to, so a link can be labelled with a name rather than an id. */
  readonly seats: Record<string, { name: string; faction: string }>;
}

/**
 * Build a campaign from a world document and populate it with the demo scenario.
 *
 * Three phases, in the order a real game is prepared: upload the world, put the order of
 * battle on the map, then issue a link per seat. The seats come last because a commander
 * needs a formation to ride with before anybody can be appointed to one.
 */
async function startCampaign(
  name: string,
  worldDoc: unknown,
  factions: readonly Faction[],
  populate: boolean,
  onProgress: (message: string) => void,
): Promise<Joined> {
  onProgress('Uploading the world…');
  const created = await createCampaign({ name, world: worldDoc, factions, seed: 20260906 });
  const session: Session = { campaignId: created.id, token: created.refereeToken };

  if (populate) {
    const commands = demoCommands(parseWorld(worldDoc));
    for (const [i, command] of commands.entries()) {
      onProgress(`Forming the army… ${i + 1} of ${commands.length}`);
      const result = await sendCommand(session, command);
      if (!result.ok) {
        // A refusal is worth showing rather than swallowing: it means the engine
        // disagrees with the scenario, which is a real answer about the world.
        throw new Error(
          result.violations?.map((v) => v.message).join('; ') ?? 'a command was refused',
        );
      }
    }
  }

  // One link per seat, so the referee can hand any of them to a player.
  onProgress('Issuing join links…');
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

export function Join({ onJoined }: { onJoined: (joined: Joined) => void }) {
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [links, setLinks] = useState<{
    referee: string;
    seats: { id: string; label: string; link: string }[];
  } | null>(null);
  const [pending, setPending] = useState<Joined | null>(null);

  const run = async (make: () => Promise<Joined>): Promise<void> => {
    setError(null);
    try {
      const joined = await make();
      // The links are shown before entering the campaign, because this is the only moment
      // they exist. They are minted here and stored only as hashes — a lost link is
      // reissued, never looked up — so a referee who clicks past this screen without
      // copying them has to start again.
      setLinks({
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
    setBusy('Loading the demonstration world…');
    void run(async () => {
      const { default: worldDoc } = await import(
        '../../shared/test/fixtures/world-32x32.json'
      );
      return startCampaign('Demonstration', worldDoc, DEMO_FACTIONS, true, setBusy);
    });
  };

  const startFromFile = (file: File): void => {
    setBusy('Reading the world…');
    void run(async () => {
      const worldDoc: unknown = JSON.parse(await file.text());
      return startCampaign(
        file.name.replace(/\.json$/, ''),
        worldDoc,
        DEMO_FACTIONS,
        false,
        setBusy,
      );
    });
  };

  if (links !== null && pending !== null) {
    return (
      <div className="join">
        <h1>Campaign created</h1>
        <p className="muted">
          One link per seat. Send each man his own and keep the referee's — they are minted
          once and stored only as hashes, so a lost link is reissued rather than looked up.
          Two commanders on the same side see different wars, which is the point.
        </p>
        <ul className="links">
          <li>
            <strong>Referee</strong>
            <code>{links.referee}</code>
          </li>
          {links.seats.map((seat) => (
            <li key={seat.id}>
              <strong>{seat.label}</strong>
              <code>{seat.link}</code>
            </li>
          ))}
        </ul>
        <button className="primary" onClick={() => onJoined(pending)}>
          Enter as referee
        </button>
      </div>
    );
  }

  return (
    <div className="join">
      <h1>Campaign</h1>
      <p className="muted">
        A refereed Napoleonic campaign on a generated map. Each commander sees only what
        their own troops have seen, and the map they are sent is masked on the server
        before it leaves it.
      </p>

      {busy !== null && <p className="busy">{busy}</p>}
      {error !== null && <p className="error">{error}</p>}

      <section>
        <h3>Start a campaign</h3>
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
          <span>Upload a world.json</span>
        </label>
        <p className="muted">
          Generate one with <code>worldgen generate --model organic</code>. A classic
          world loads, but carries no ford or bridge data, so every major river will be
          impassable.
        </p>
        <button onClick={startDemo} disabled={busy !== null}>
          Or run the demonstration
        </button>
      </section>

      <section>
        <h3>Join one</h3>
        <p className="muted">
          Open the link your referee sent you. It carries your side's token in the URL
          fragment, which never reaches the server and never appears in its logs.
        </p>
      </section>
    </div>
  );
}
