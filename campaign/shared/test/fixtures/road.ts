/**
 * A column that has already been on the road, for tests that start partway through a march.
 *
 * The cap reads hours on the road by the hour over the last twenty-four, so a test that
 * wants a tired column has to say when it marched as well as how long. The hours run back
 * from the hour before `untilHour`, whole ones first and any fraction in the earliest.
 */

import type { RoadHour, Unit } from '../../src/unit.js';

export function marched(
  hours: number,
  untilHour: number,
): Pick<Unit, 'hoursMarchedToday' | 'roadHours'> {
  const roadHours: RoadHour[] = [];
  let left = hours;
  for (let hour = untilHour - 1; left > 1e-9; hour--) {
    roadHours.unshift({ hour, hours: Math.min(1, left) });
    left -= 1;
  }
  return { hoursMarchedToday: hours, roadHours };
}
