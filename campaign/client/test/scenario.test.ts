/**
 * The demonstration scenario must be a legal campaign.
 *
 * It is the first thing anyone sees, and a demo that shows units standing in the sea or
 * columns teleporting across a bay teaches the reader the wrong thing about the engine
 * before they have read a line of it. It is built through the real commands and the real
 * router, so these assertions are really asking whether those did what they claim.
 */

import { describe, expect, it } from 'vitest';

import {
  columnIsConnected,
  distance,
  isWater,
  key,
  occupied,
  presentUnderArms,
} from '@campaign/shared';

import { buildDemo } from '../src/scenario.js';

const demo = buildDemo();

describe('the demo campaign', () => {
  it('has both factions and every unit', () => {
    expect([...demo.state.factions.keys()].sort()).toEqual(['blue', 'red']);
    expect(demo.state.units.size).toBe(5);
  });

  it('names every unit it places', () => {
    for (const id of demo.state.units.keys()) {
      expect(demo.names[id], `${id} has no name`).toBeTruthy();
    }
  });

  it('stands every unit on dry land, along its whole column', () => {
    // The bug this catches looks like a rendering glitch and is not one: a column drawn
    // across a bay means the router returned a path through water.
    for (const unit of demo.state.units.values()) {
      for (const c of unit.column) {
        const hex = demo.world.hexes.get(key(c));
        expect(hex, `${unit.id} stands off the map at ${key(c)}`).toBeDefined();
        expect(isWater(hex!), `${unit.id} stands in water at ${key(c)}`).toBe(false);
      }
    }
  });

  it('gives every unit a connected column', () => {
    // Consecutive hexes must be adjacent, or the map draws a straight line between two
    // points the unit never marched between.
    for (const unit of demo.state.units.values()) {
      expect(columnIsConnected(unit), `${unit.id} has a gap in its column`).toBe(true);
      for (let i = 1; i < unit.column.length; i++) {
        expect(distance(unit.column[i - 1]!, unit.column[i]!)).toBe(1);
      }
    }
  });

  it('gives the cavalry a long column and the infantry a short one', () => {
    // The thing the demo exists to show. If these ever converge, the scenario has stopped
    // demonstrating why column length matters.
    const cavalry = demo.state.units.get('red-2')!;
    const infantry = demo.state.units.get('red-1')!;
    expect(occupied(cavalry).length).toBeGreaterThan(occupied(infantry).length * 2);
  });

  it('marches its units far enough to have a visible tail', () => {
    for (const unit of demo.state.units.values()) {
      expect(unit.column.length, `${unit.id} has not moved`).toBeGreaterThan(1);
    }
  });

  it('starts every unit fit', () => {
    for (const unit of demo.state.units.values()) {
      expect(presentUnderArms(unit)).toBe(unit.effectives);
      expect(unit.morale).toBeGreaterThan(0);
      expect(unit.provisions).toBeGreaterThan(0);
    }
  });

  it('does not stack every unit on the same ground', () => {
    // Five units all in one place would look like one unit and prove nothing.
    const heads = new Set([...demo.state.units.values()].map((u) => key(u.column[0]!)));
    expect(heads.size).toBe(demo.state.units.size);
  });
});
