// Sim parameters (port of mazebot/sim.py:SimParams). Every policy file carries the sim it was trained in
// (`sim_params` in weights.json); missing keys take the Python dataclass defaults below.

export type GoalInputs = 'sincos_dist' | 'sincos' | 'bearing';

export interface SimParams {
	radius: number; // robot collider radius
	wheelBase: number;
	vMax: number; // wheel surface speed at u = 1 (units/s)
	reverseMax: number; // body speed floor = -reverseMax * vMax (0: no reversing)
	dt: number; // physics step
	frameSkip: number; // physics steps per policy step
	rayAnglesDeg: number[];
	rayAngles: number[]; // radians
	rayRange: number;
	goalDistScale: number;
	goalRadius: number; // success when the centre is this close to B
	goalInputs: GoalInputs;
	prevActionInputs: boolean; // observation ends with the previous (uL, uR)
}

/** Python `sim_params` dict (snake_case), as stored in weights.json. */
export interface SimParamsJson {
	radius?: number;
	wheel_base?: number;
	v_max?: number;
	reverse_max?: number;
	dt?: number;
	frame_skip?: number;
	ray_angles_deg?: number[];
	ray_range?: number;
	goal_dist_scale?: number;
	goal_radius?: number;
	goal_inputs?: GoalInputs;
	prev_action_inputs?: boolean;
}

const DEG = Math.PI / 180; // same constant as CPython's math.radians

export function simParams(d: SimParamsJson = {}): SimParams {
	const rayAnglesDeg = (d.ray_angles_deg ?? [-60, -30, 0, 30, 60]).map(Number);
	return {
		radius: d.radius ?? 0.3,
		wheelBase: d.wheel_base ?? 0.6,
		vMax: d.v_max ?? 1.2,
		reverseMax: d.reverse_max ?? 0.0,
		dt: d.dt ?? 1.0 / 30.0,
		frameSkip: d.frame_skip ?? 2,
		rayAnglesDeg,
		rayAngles: rayAnglesDeg.map((a) => a * DEG),
		rayRange: d.ray_range ?? 3.0,
		goalDistScale: d.goal_dist_scale ?? 10.0,
		goalRadius: d.goal_radius ?? 0.35,
		goalInputs: d.goal_inputs ?? 'sincos_dist',
		prevActionInputs: d.prev_action_inputs ?? false,
	};
}

export function goalDim(p: SimParams): number {
	return { sincos_dist: 3, sincos: 2, bearing: 1 }[p.goalInputs];
}

export function obsDim(p: SimParams): number {
	return p.rayAngles.length + goalDim(p) + (p.prevActionInputs ? 2 : 0);
}

/** Human-readable input names, in order (mazebot/policy.py:obs_spec, shortened for the network panel). */
export function obsNames(p: SimParams): string[] {
	const goal = { sincos_dist: ['goal sin', 'goal cos', 'goal dist'], sincos: ['goal sin', 'goal cos'], bearing: ['goal angle'] }[
		p.goalInputs
	];
	return [...p.rayAngles.map((_, i) => `ray ${i + 1}`), ...goal, ...(p.prevActionInputs ? ['last L', 'last R'] : [])];
}
