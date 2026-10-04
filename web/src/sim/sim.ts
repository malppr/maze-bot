// Robot kinematics, observations and the episode loop (port of mazebot/sim.py and the parts of
// mazebot/env.py the browser needs). Conventions: x right, y down, heading th = 0 along +x, positive th
// turns clockwise on screen. Wheel commands uL, uR in [-1, 1].

import { border, castRays, concatCaps, capsules, pushOut, type Caps } from './geometry';
import { buildGrid, geodesicField, lookup, type Grid } from './grid';
import { goalDim, obsDim, type SimParams } from './params';

const TWO_PI = 2.0 * Math.PI;
export const BORDER_RAD = 0.05;

/** Wrap to (-pi, pi]. JS % on doubles is C fmod, like Python's math.fmod. */
export function wrapAngle(a: number): number {
	a = (a + Math.PI) % TWO_PI;
	if (a <= 0.0) a += TWO_PI;
	return a - Math.PI;
}

export interface Pose {
	x: number;
	y: number;
	th: number;
}

/**
 * Advance one physics step in place; returns whether the robot touched a wall. Differential drive with the
 * body speed floored at -reverseMax * vMax; midpoint integration, sub-stepped so the centre never moves more
 * than r/2 per sub-step.
 */
export function physicsStep(s: Pose, ul: number, ur: number, caps: Caps, p: SimParams): boolean {
	const v = Math.max(p.vMax * 0.5 * (ul + ur), -p.reverseMax * p.vMax);
	const w = (p.vMax * (ul - ur)) / p.wheelBase;
	const n = Math.max(1, Math.ceil((Math.abs(v) * p.dt) / (0.5 * p.radius)));
	const h = p.dt / n;
	let { x, y, th } = s;
	let contact = false;
	for (let k = 0; k < n; k++) {
		const tm = th + 0.5 * w * h;
		x = x + v * Math.cos(tm) * h;
		y = y + v * Math.sin(tm) * h;
		th = wrapAngle(th + w * h);
		const r = pushOut(x, y, p.radius, caps);
		x = r.x;
		y = r.y;
		contact = contact || r.contact;
	}
	s.x = x;
	s.y = y;
	s.th = th;
	return contact;
}

export function rayDirs(th: number, p: SimParams): Float64Array {
	const d = new Float64Array(2 * p.rayAngles.length);
	p.rayAngles.forEach((a, i) => {
		d[2 * i] = Math.cos(th + a);
		d[2 * i + 1] = Math.sin(th + a);
	});
	return d;
}

/**
 * Sensor part of the observation (rays / range, then goal inputs; the episode appends the memory inputs).
 * Bearing via dot/cross products: sin > 0 means the goal is clockwise (to the right on screen).
 */
export function observe(x: number, y: number, th: number, gx: number, gy: number, caps: Caps, p: SimParams) {
	const rays = castRays(x, y, rayDirs(th, p), caps, p.rayRange);
	const hx = Math.cos(th);
	const hy = Math.sin(th);
	const ex = gx - x;
	const ey = gy - y;
	const dist = Math.sqrt(ex * ex + ey * ey);
	let cosB = 1.0;
	let sinB = 0.0;
	if (dist > 1e-9) {
		const ux = ex / dist;
		const uy = ey / dist;
		cosB = hx * ux + hy * uy;
		sinB = hx * uy - hy * ux;
	}
	const k = p.rayAngles.length;
	const obs = new Float64Array(k + goalDim(p));
	for (let i = 0; i < k; i++) obs[i] = rays[i] / p.rayRange;
	if (p.goalInputs === 'bearing') {
		obs[k] = Math.atan2(sinB, cosB) / Math.PI;
	} else {
		obs[k] = sinB;
		obs[k + 1] = cosB;
		if (p.goalInputs === 'sincos_dist') obs[k + 2] = Math.min(dist / p.goalDistScale, 1.0);
	}
	return { obs, rays };
}

export function goalDistance(x: number, y: number, gx: number, gy: number): number {
	const ex = gx - x;
	const ey = gy - y;
	return Math.sqrt(ex * ex + ey * ey);
}

/** A layout: arena size, interior walls ([ax, ay, bx, by, rad] rows), start [x, y, th], goal [x, y]. */
export interface MazeMap {
	name?: string;
	width: number;
	height: number;
	walls: number[][];
	start: [number, number, number];
	goal: [number, number];
}

/** All capsules of a map: the arena border first, then the walls (same order as mapgen.Map.caps). */
export function mapCaps(m: MazeMap): Caps {
	return concatCaps(border(m.width, m.height, BORDER_RAD), capsules(m.walls));
}

export interface Timeout {
	factor: number; // time limit = factor * path length / vMax + slack
	slack: number; // seconds
}
export const TIMEOUT: Timeout = { factor: 3.0, slack: 10.0 };

export interface StepResult {
	success: boolean;
	lost: boolean; // ran out of time (env "truncated")
	contact: boolean;
}

/**
 * One drive from A to B (mirrors MazeEnv.reset/step without rewards). `observation()` is exactly what the
 * Python policy saw: float32-rounded, with the previous clipped action appended when the sim uses memory.
 */
export class Episode {
	readonly p: SimParams;
	map: MazeMap;
	caps!: Caps;
	grid!: Grid;
	field!: Float64Array;
	pose: Pose = { x: 0, y: 0, th: 0 };
	rays: Float64Array = new Float64Array(0);
	sensors: Float64Array = new Float64Array(0); // observe() output, float64
	prevAction: [number, number] = [0, 0];
	steps = 0;
	maxSteps = 0;
	geo0 = Infinity; // path length A -> B at reset (Infinity: no path)
	done: 'success' | 'lost' | null = null;

	constructor(map: MazeMap, p: SimParams, private timeout: Timeout = TIMEOUT) {
		this.p = p;
		this.map = map;
		this.setMap(map);
	}

	/** Swap in a new layout (walls / A / B) and restart. */
	setMap(map: MazeMap): void {
		this.map = map;
		this.caps = mapCaps(map);
		this.grid = buildGrid(map.width, map.height, this.caps, this.p.radius);
		this.field = geodesicField(this.grid, map.goal[0], map.goal[1]);
		this.reset();
	}

	reset(): void {
		const [x, y, th] = this.map.start;
		this.pose = { x, y, th };
		this.geo0 = lookup(this.grid, this.field, x, y);
		const policyDt = this.p.dt * this.p.frameSkip;
		const limitS = (this.timeout.factor * this.geo0) / this.p.vMax + this.timeout.slack;
		this.maxSteps = Math.ceil(limitS / policyDt); // Infinity when there is no path: never "lost" by time
		this.steps = 0;
		this.prevAction = [0, 0];
		this.done = null;
		this.sense();
	}

	get reachable(): boolean {
		return this.geo0 < Infinity;
	}

	private sense(): void {
		const { x, y, th } = this.pose;
		const r = observe(x, y, th, this.map.goal[0], this.map.goal[1], this.caps, this.p);
		this.sensors = r.obs;
		this.rays = r.rays;
	}

	/** Policy input: sensors (+ previous action), rounded to float32 like the Python env. */
	observation(): Float64Array {
		const n = obsDim(this.p);
		const o = new Float64Array(n);
		for (let i = 0; i < this.sensors.length; i++) o[i] = Math.fround(this.sensors[i]);
		if (this.p.prevActionInputs) {
			o[n - 2] = Math.fround(this.prevAction[0]);
			o[n - 1] = Math.fround(this.prevAction[1]);
		}
		return o;
	}

	/** One policy step: frameSkip physics steps, stopping early on reaching B. */
	step(action: ArrayLike<number>): StepResult {
		const clip = (a: number) => (a < -1.0 ? -1.0 : a > 1.0 ? 1.0 : a);
		const ul = clip(action[0]);
		const ur = clip(action[1]);
		const p = this.p;
		let contact = false;
		let success = false;
		for (let k = 0; k < p.frameSkip; k++) {
			contact = physicsStep(this.pose, ul, ur, this.caps, p) || contact;
			if (goalDistance(this.pose.x, this.pose.y, this.map.goal[0], this.map.goal[1]) < p.goalRadius) {
				success = true;
				break;
			}
		}
		this.steps += 1;
		this.prevAction = [ul, ur];
		this.sense();
		const lost = !success && this.steps >= this.maxSteps;
		if (success) this.done = 'success';
		else if (lost) this.done = 'lost';
		return { success, lost, contact };
	}
}
