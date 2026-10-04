// The Wheelys a visitor can drive (artifacts/release/versions/versions.json; "live": true). Weights load on
// demand, so only the chosen version is downloaded.
import type { WeightsJson } from '../policy/mlp';

export interface Version {
	id: string;
	name: string;
	blurb: string;
	load?: () => Promise<{ default: unknown }>; // none: the hand-written rule
}

export const VERSIONS: Version[] = [
	{
		id: 'heuristic',
		name: 'By-the-Book',
		blurb: 'No learning, just rules I wrote by hand. Every other Wheely has to beat this.',
	},
	{
		id: 'v3-rookie',
		name: 'Rookie',
		blurb: 'The first Wheely that learned to drive. Fine in the open, lost in mazes.',
		load: () => import('../../../artifacts/release/versions/v3-rookie/weights.json'),
	},
	{
		id: 'v4-owl-eyes',
		name: 'Owl Eyes',
		blurb: 'Rays all the way around, so it notices when it has driven into a pocket.',
		load: () => import('../../../artifacts/release/versions/v4-owl-eyes/weights.json'),
	},
	{
		id: 'final',
		name: 'Wheely',
		blurb: 'Remembers its last move, so it stops dithering. Beats the rules in 6 of 7 tests.',
		load: () => import('../../../artifacts/release/versions/final/weights.json'),
	},
];

export const DEFAULT_VERSION = 'final';

export async function loadWeights(v: Version): Promise<WeightsJson | null> {
	if (!v.load) return null;
	return (await v.load()).default as WeightsJson;
}
