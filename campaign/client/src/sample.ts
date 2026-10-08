/**
 * The sample order of battle, to download and edit.
 *
 * Bundled from `examples/order-of-battle.yaml` itself rather than copied, so the file a
 * referee starts from is the one the tests prove the engine accepts, and its header's list
 * of traits, arms and echelons is the one a test keeps in step with the rules.
 */

import text from '../../examples/order-of-battle.yaml?raw';

export const SAMPLE_OOB = text;
export const SAMPLE_OOB_NAME = 'order-of-battle.yaml';
/** A `data:` link: no object URL to revoke, and nothing for the server to serve. */
export const SAMPLE_OOB_HREF = `data:application/yaml;charset=utf-8,${encodeURIComponent(text)}`;
