/**
 * A column that has already been on the road, for tests that start partway through a march.
 *
 * The cap reads hours on the road by the hour over the last twenty-four, so a test that
 * wants a tired column has to say when it marched as well as how long.
 */

import { roadHoursEnding } from '../../src/movement.js';
import type { Unit } from '../../src/unit.js';

export const marched = (
  hours: number,
  untilHour: number,
): Pick<Unit, 'hoursMarchedToday' | 'roadHours'> => ({
  hoursMarchedToday: hours,
  roadHours: roadHoursEnding(hours, untilHour),
});
