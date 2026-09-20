/**
 * The shared campaign engine.
 *
 * Pure TypeScript: no I/O, no DOM, no Node APIs. The server runs this on ground truth,
 * authoritatively; the client runs the same code over the masked world it has been given,
 * so a commander's plans are computed from what they actually know.
 *
 * `palette.ts` is deliberately not re-exported — it is generated, and everything should
 * go through `theme.ts` so a campaign can override how it is drawn.
 */

export * from './column.js';
export * from './commander.js';
export * from './config.js';
export * from './crossing.js';
export * from './despatch.js';
export * from './engine.js';
export * from './events.js';
export * from './fatigue.js';
export * from './hex.js';
export * from './mask.js';
export * from './movement.js';
export * from './observe.js';
export * from './recon.js';
export * from './rng.js';
export * from './ruling.js';
export * from './scenario.js';
export * from './scheduler.js';
export * from './state.js';
export * from './task.js';
export * from './terrain.js';
export * from './theme.js';
export * from './unit.js';
export * from './view.js';
export * from './world.js';
