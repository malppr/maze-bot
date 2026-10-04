// Parity with the Python reference. Fixtures: `uv run python scripts/make_parity_fixtures.py`.
// Tolerances (HANDOFF/PLAN M3): per physics step / ray / grid <= 1e-9, policy forward <= 1e-6,
// closed-loop rollouts <= 1e-6 (300 policy steps; The trap runs to the "lost" timeout, 486 steps).
import { readFileSync } from 'node:fs';
import { afterAll, describe, expect, test } from 'vitest';
import {
	Episode,
	HeuristicPolicy,
	MLPPolicy,
	buildGrid,
	clearance,
	geodesicField,
	lookup,
	mapCaps,
	observe,
	physicsStep,
	pushOut,
	rdp,
	reachable,
	simParams,
	type MazeMap,
	type SimParams,
	type WeightsJson,
} from '../src';

const root = new URL('../../', import.meta.url);
const read = (p: string) => JSON.parse(readFileSync(new URL(p, root), 'utf-8'));
const fx = (name: string) => read(`web/test/fixtures/${name}`);

const maps: MazeMap[] = fx('maps.json');
const caps = maps.map(mapCaps);
const weights = (id: string): WeightsJson => read(`artifacts/release/versions/${id}/weights.json`);

type Policy = { sim: SimParams; act(o: Float64Array): number[] };
function policy(id: string): Policy {
	if (id === 'heuristic') {
		const h = new HeuristicPolicy();
		return { sim: h.sim, act: (o) => h.act(o).action };
	}
	const m = new MLPPolicy(weights(id));
	return { sim: m.sim, act: (o) => m.forward(o).action };
}

// Worst error per check, printed after the run.
const worst: Record<string, number> = {};
function near(check: string, got: number, want: number | null, tol: number, ctx = '') {
	const w = want === null ? Infinity : want;
	const err = got === w ? 0 : Math.abs(got - w);
	worst[check] = Math.max(worst[check] ?? 0, Number.isNaN(err) ? Infinity : err);
	if (!(err <= tol)) expect.fail(`${check}${ctx}: got ${got}, want ${w} (|err| ${err} > ${tol})`);
}
const angleErr = (a: number, b: number) => Math.abs(Math.atan2(Math.sin(a - b), Math.cos(a - b)));

afterAll(() => {
	const rows = Object.entries(worst).map(([k, v]) => `  ${k.padEnd(26)} ${v.toExponential(2)}`);
	console.log(`max |error| per check:\n${rows.join('\n')}`);
});

describe('geometry', () => {
	const g = fx('geometry.json');
	const sims: Record<string, SimParams> = {
		fwd5: simParams(),
		rev: simParams({ reverse_max: 1.0 }),
		owl: simParams(weights('v4-owl-eyes').sim_params),
	};

	test('physics step (incl. reversing)', () => {
		for (const c of g.steps) {
			const [x, y, th, ul, ur] = c.in;
			const s = { x, y, th };
			const contact = physicsStep(s, ul, ur, caps[c.map], sims[c.sim]);
			near('physics x/y', s.x, c.out[0], 1e-9);
			near('physics x/y', s.y, c.out[1], 1e-9);
			worst['physics th'] = Math.max(worst['physics th'] ?? 0, angleErr(s.th, c.out[2]));
			expect(angleErr(s.th, c.out[2])).toBeLessThanOrEqual(1e-9);
			expect(contact).toBe(c.out[3]);
		}
	});

	test('rays and observation (5 forward, 7 all-round)', () => {
		for (const c of g.rays) {
			const m = maps[c.map];
			const [x, y, th] = c.in;
			const r = observe(x, y, th, m.goal[0], m.goal[1], caps[c.map], sims[c.sim]);
			c.rays.forEach((v: number, i: number) => near('rays', r.rays[i], v, 1e-9));
			c.obs.forEach((v: number, i: number) => near('observation', r.obs[i], v, 1e-9));
		}
	});

	test('goal input modes sincos / bearing', () => {
		for (const c of g.goal_modes) {
			const m = maps[c.map];
			const [x, y, th] = c.in;
			const r = observe(x, y, th, m.goal[0], m.goal[1], caps[c.map], simParams({ goal_inputs: c.goal_inputs }));
			c.obs.forEach((v: number, i: number) => near('observation', r.obs[i], v, 1e-9));
		}
	});

	test('push-out and clearance', () => {
		for (const c of g.pushes) {
			const r = pushOut(c.in[0], c.in[1], 0.3, caps[c.map]);
			near('push-out', r.x, c.out[0], 1e-9);
			near('push-out', r.y, c.out[1], 1e-9);
			expect(r.contact).toBe(c.out[2]);
			near('clearance', clearance(c.in[0], c.in[1], caps[c.map]), c.clear, 1e-12);
		}
	});

	test('RDP keeps the same points', () => {
		for (const c of g.rdp) expect(rdp(c.points, c.eps)).toEqual(c.kept);
	});
});

describe('grid', () => {
	test('occupancy, geodesic field, reachability', () => {
		for (const c of fx('grid.json')) {
			const m = maps[c.map];
			const grid = buildGrid(m.width, m.height, caps[c.map], 0.3);
			expect([grid.nx, grid.ny]).toEqual([c.nx, c.ny]);
			expect(Array.from(grid.free)).toEqual(c.free);
			const field = geodesicField(grid, m.goal[0], m.goal[1]);
			c.field.forEach((v: number | null, i: number) => near('geodesic field', field[i], v, 1e-9, ` map ${c.map} cell ${i}`));
			near('geodesic A', lookup(grid, field, m.start[0], m.start[1]), c.geo_start, 1e-9);
			expect(reachable(grid, m.start[0], m.start[1], m.goal[0], m.goal[1])).toBe(c.reachable);
		}
	});
});

describe('policies', () => {
	for (const v of fx('policy.json')) {
		test(`forward: ${v.version}`, () => {
			if (v.version === 'heuristic') {
				const h = new HeuristicPolicy();
				for (const c of v.cases) c.action.forEach((a: number, i: number) => near('heuristic action', h.act(c.obs).action[i], a, 1e-6));
				return;
			}
			const m = new MLPPolicy(weights(v.version));
			for (const c of v.cases) {
				const out = m.forward(c.obs);
				c.acts.forEach((layer: number[], k: number) =>
					layer.forEach((a: number, i: number) => near('MLP activations', out.layers[k + 1][i], a, 1e-6)),
				);
				c.action.forEach((a: number, i: number) => near('MLP action', out.action[i], a, 1e-6));
			}
		});
	}
});

describe('closed-loop rollouts', () => {
	for (const r of fx('rollouts.json')) {
		test(`${r.version} on ${maps[r.map].name ?? r.map} (map ${r.map})`, () => {
			const pol = policy(r.version);
			const ep = new Episode(maps[r.map], pol.sim);
			expect(ep.maxSteps).toBe(r.max_steps);
			let res = { success: false, lost: false };
			for (let k = 1; k < r.traj.length; k++) {
				res = ep.step(pol.act(ep.observation()));
				const [x, y, th] = r.traj[k];
				const ctx = ` step ${k}`;
				near('rollout x/y', ep.pose.x, x, 1e-6, ctx);
				near('rollout x/y', ep.pose.y, y, 1e-6, ctx);
				worst['rollout th'] = Math.max(worst['rollout th'] ?? 0, angleErr(ep.pose.th, th));
				expect(angleErr(ep.pose.th, th), `th${ctx}`).toBeLessThanOrEqual(1e-6);
			}
			expect(res.success).toBe(r.success);
			expect(res.lost).toBe(r.truncated);
		});
	}
});
