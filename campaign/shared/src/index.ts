/**
 * The shared campaign engine.
 *
 * Pure TypeScript: no I/O, no DOM, no Node APIs. The server runs this on ground truth,
 * authoritatively; the client runs the same code over the masked world it has been given,
 * so a commander's plans are computed from what they actually know.
 */

export * from './hex.js';
export * from './palette.js';
export * from './world.js';
