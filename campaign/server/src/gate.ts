/**
 * One heavy request at a time.
 *
 * Creating a campaign means holding a generated world in memory twice over: once as the
 * uploaded bytes and again as the object graph `JSON.parse` builds from them. Measured on
 * a 200x200 map that is 93 MB of resident memory becoming 348 MB, and three at once
 * reaching 677 MB — on a container with a fixed memory limit, that is not a slow request,
 * it is the process being killed and every commander's socket dropping with it.
 *
 * The gate is taken in `onRequest`, which is the one hook that runs *before* Fastify reads
 * the body. A queue around the route handler would be too late: by the time a handler runs
 * the bytes are already buffered and already parsed, and the spike has happened. Holding
 * the request here costs a socket and a place in line, which is cheap, and means the
 * second create waits for the first rather than joining it.
 *
 * Deliberately not a rate limit. A referee setting up three campaigns in a minute is doing
 * something reasonable and should not be refused — only made to do it in order.
 */

/** A gate that lets `width` holders through at once and queues the rest. */
export interface Gate {
  /** Resolves when the caller may proceed. The returned function gives the slot back. */
  enter(): Promise<() => void>;
  /** How many are waiting, for tests and for a log line worth having. */
  readonly waiting: number;
}

export function gate(width: number): Gate {
  let free = Math.max(1, width);
  const queue: (() => void)[] = [];

  const release = (): void => {
    // Guarded by the closure below, so a double release cannot hand out a slot that was
    // never taken — which would widen the gate silently and permanently.
    const next = queue.shift();
    if (next === undefined) {
      free += 1;
      return;
    }
    next();
  };

  return {
    get waiting() {
      return queue.length;
    },
    async enter() {
      if (free > 0) {
        free -= 1;
      } else {
        await new Promise<void>((resolve) => queue.push(resolve));
      }

      let released = false;
      return () => {
        if (released) return;
        released = true;
        release();
      };
    },
  };
}
