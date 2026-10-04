// The Wheelys a visitor can drive (artifacts/release/versions/versions.json; "live": true). Weights load on
// demand, so only the chosen version is downloaded.
import type { WeightsJson } from '../policy/mlp';

export interface Version {
	id: string;
	name: string;
	blurb: string;
	load?: () => Promise<{ default: unknown }>; // none: the hand-written rule
	hidden?: boolean; // not offered as a chip (dev page only: failure versions for the write-up clips)
}

export const VERSIONS: Version[] = [
	{
		id: 'heuristic',
		name: 'By-the-Book',
		blurb: 'A heuristic policy with no learning. The baseline the trained versions are compared with.',
	},
	{
		id: 'v3-rookie',
		name: 'Rookie',
		blurb: 'The first version that learned to drive. Good in open layouts, weak in mazes.',
		load: () => import('../../../artifacts/release/versions/v3-rookie/weights.json'),
	},
	{
		id: 'v4-owl-eyes',
		name: 'Owl Eyes',
		blurb: 'Seven rays all around, including behind. Better at getting out of traps.',
		load: () => import('../../../artifacts/release/versions/v4-owl-eyes/weights.json'),
	},
	{
		id: 'final',
		name: 'Wheely',
		blurb: 'Gets its previous wheel command as input. Matches or beats the heuristic in 6 of 7 categories.',
		load: () => import('../../../artifacts/release/versions/final/weights.json'),
	},
];

export const DEFAULT_VERSION = 'final';

export async function loadWeights(v: Version): Promise<WeightsJson | null> {
	if (!v.load) return null;
	return (await v.load()).default as WeightsJson;
}
