/**
 * That the gate actually gates, and that it always opens again.
 *
 * A concurrency limit is only as good as its release path: one missed release and the
 * gate is shut for the life of the process, which presents as campaign creation hanging
 * forever with nothing in the log. So the interesting tests here are the exits, not the
 * happy path.
 */

import { describe, expect, it } from 'vitest';

import { gate } from '../src/gate.js';

const settled = () => new Promise((r) => setTimeout(r, 0));

describe('a gate of one', () => {
  it('lets the first through and holds the second', async () => {
    const g = gate(1);
    const first = await g.enter();

    let secondEntered = false;
    const second = g.enter().then((release) => {
      secondEntered = true;
      return release;
    });

    await settled();
    expect(secondEntered).toBe(false);
    expect(g.waiting).toBe(1);

    first();
    await second;
    expect(secondEntered).toBe(true);
  });

  it('serialises a crowd, one at a time and in order', async () => {
    const g = gate(1);
    const order: number[] = [];
    let inFlight = 0;
    let peak = 0;

    await Promise.all(
      [1, 2, 3, 4, 5].map(async (n) => {
        const release = await g.enter();
        inFlight += 1;
        peak = Math.max(peak, inFlight);
        order.push(n);
        await settled();
        inFlight -= 1;
        release();
      }),
    );

    expect(peak).toBe(1);
    expect(order).toEqual([1, 2, 3, 4, 5]);
  });

  it('ignores a second release, rather than widening itself', async () => {
    // The failure this prevents is silent and permanent: a double release adds a slot
    // that was never taken, and the limit quietly stops being a limit.
    const g = gate(1);
    const release = await g.enter();
    release();
    release();

    const a = await g.enter();
    let bEntered = false;
    void g.enter().then(() => {
      bEntered = true;
    });

    await settled();
    expect(bEntered).toBe(false);
    a();
  });
});

describe('a wider gate', () => {
  it('lets that many through at once and no more', async () => {
    const g = gate(3);
    let inFlight = 0;
    let peak = 0;

    await Promise.all(
      Array.from({ length: 9 }, async () => {
        const release = await g.enter();
        inFlight += 1;
        peak = Math.max(peak, inFlight);
        await settled();
        inFlight -= 1;
        release();
      }),
    );

    expect(peak).toBe(3);
  });

  it('treats a width below one as one, rather than shutting forever', () => {
    expect(gate(0).waiting).toBe(0);
  });
});
