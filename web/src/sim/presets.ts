// Demo presets (presets/presets.json, shared with Python). Defined landscape; portrait = transpose.
import data from '../../../presets/presets.json';
import type { MazeMap } from './sim';

export interface Preset {
	name: string;
	width: number;
	height: number;
	walls: number[][];
	start: number[];
	goal: number[];
}

export const PRESETS: Preset[] = data.presets as Preset[];

/** Port of mapgen.preset_map: portrait swaps x/y of every point and maps heading th to pi/2 - th. */
export function presetMap(p: Preset, portrait = false): MazeMap {
	const [sx, sy, sth] = p.start;
	const [gx, gy] = p.goal;
	if (!portrait) {
		return { name: p.name, width: p.width, height: p.height, walls: p.walls.map((w) => [...w]), start: [sx, sy, sth], goal: [gx, gy] };
	}
	return {
		name: p.name,
		width: p.height,
		height: p.width,
		walls: p.walls.map((w) => [w[1], w[0], w[3], w[2], w[4]]),
		start: [sy, sx, Math.PI / 2 - sth],
		goal: [gy, gx],
	};
}
