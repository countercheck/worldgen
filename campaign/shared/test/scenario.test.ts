/**
 * The demonstration scenario must be a legal campaign.
 *
 * It is the first thing anyone sees, and a demo that shows units standing in the sea or
 * columns teleporting across a bay teaches the reader the wrong thing about the engine
 * before they have read a line of it.
 *
 * These assertions run the commands the client would post rather than inspecting a state
 * built for the test. That is the only way to find out what the demo actually does: the
 * scenario is a list of commands now, and a command that `check` refuses produces nothing
 * at all, so a demo whose units never arrive would otherwise pass every structural test
 * by being empty.
 */

import { describe, expect, it } from 'vitest';

import worldDoc from './fixtures/world-32x32.json';

import { applyAll } from '../src/engine.js';
import { columnIsConnected, occupied } from '../src/column.js';
import { distance, key } from '../src/hex.js';
import { DEMO_FACTIONS, demoCommands } from '../src/scenario.js';
import { EMPTY_STATE } from '../src/state.js';
import { presentUnderArms } from '../src/unit.js';
import { isWater, parseWorld } from '../src/world.js';

const world = parseWorld(worldDoc);
const commands = demoCommands(world);

const out = applyAll(
  [
    {
      kind: 'create_campaign',
      name: 'Demonstration',
      world: {
        seed: world.seed,
        width: world.width,
        height: world.height,
        layout: world.layout,
        schemaVersion: world.schemaVersion,
        hash: 'sha256:test',
      },
      seed: 20260906,
    },
    ...DEMO_FACTIONS.map((faction) => ({ kind: 'add_faction' as const, faction })),
    ...commands,
  ],
  EMPTY_STATE,
  world,
  'strict',
);

describe('the demo scenario', () => {
  it('is accepted in full by the engine at its strictest', () => {
    // Not "does not throw" — a refusal is a value here, and the messages are the point.
    expect(out.violations.map((v) => v.message)).toEqual([]);
    expect(out.ok).toBe(true);
  });

  it('has both factions and every unit', () => {
    expect([...out.state.factions.keys()].sort()).toEqual(['blue', 'red']);
    expect(out.state.units.size).toBe(5);
  });

  it('names every unit it places', () => {
    for (const unit of out.state.units.values()) {
      expect(unit.name, `${unit.id} has no name`).toBeTruthy();
    }
  });

  it('stands every unit on dry land, along its whole column', () => {
    // The bug this catches looks like a rendering glitch and is not one: a column drawn
    // across a bay means the router returned a path through water.
    for (const unit of out.state.units.values()) {
      for (const c of unit.column) {
        const hex = world.hexes.get(key(c));
        expect(hex, `${unit.id} stands off the map at ${key(c)}`).toBeDefined();
        expect(isWater(hex!), `${unit.id} stands in water at ${key(c)}`).toBe(false);
      }
    }
  });

  it('gives every unit a connected column', () => {
    // Consecutive hexes must be adjacent, or the map draws a straight line between two
    // points the unit never marched between.
    for (const unit of out.state.units.values()) {
      expect(columnIsConnected(unit), `${unit.id} has a gap in its column`).toBe(true);
      for (let i = 1; i < unit.column.length; i++) {
        expect(distance(unit.column[i - 1]!, unit.column[i]!)).toBe(1);
      }
    }
  });

  it('gives the cavalry a long column and the infantry a short one', () => {
    // The thing the demo exists to show. If these ever converge, the scenario has stopped
    // demonstrating why column length matters.
    const cavalry = out.state.units.get('red-2')!;
    const infantry = out.state.units.get('red-1')!;
    expect(occupied(cavalry).length).toBeGreaterThan(occupied(infantry).length * 2);
  });

  it('marches its units far enough to have a visible tail', () => {
    for (const unit of out.state.units.values()) {
      expect(unit.column.length, `${unit.id} has not moved`).toBeGreaterThan(1);
    }
  });

  it('starts every unit fit', () => {
    for (const unit of out.state.units.values()) {
      expect(presentUnderArms(unit)).toBe(unit.paperStrength);
      expect(unit.morale).toBeGreaterThan(0);
      expect(unit.provisions).toBeGreaterThan(0);
    }
  });

  it('does not stack every unit on the same ground', () => {
    // Five units all in one place would look like one unit and prove nothing.
    const heads = new Set([...out.state.units.values()].map((u) => key(u.column[0]!)));
    expect(heads.size).toBe(out.state.units.size);
  });

  it('is the same scenario every time it is built', () => {
    // The client posts these commands one at a time over HTTP; if the second build of the
    // list disagreed with the first, a reload would produce a different campaign and the
    // demo would be unreproducible in exactly the way the generator is not.
    expect(demoCommands(world)).toEqual(commands);
  });
});
