// "By-the-Book": the hand-written baseline (port of mazebot/baselines.py:ReactivePolicy). Sees the same
// observation as the learned policies (5 forward rays, goal sin/cos/distance) and has no memory.
import { simParams, type SimParams } from '../sim/params';

export interface RuleTrace {
	left: number; // closest ray on the left (rays 1-2), world units
	right: number; // closest ray on the right (rays 4-5)
	front: number; // closest of rays 2-4
	bearing: number; // goal angle, + = to the right
	rule: 'seek' | 'keep-wall-right' | 'keep-wall-left' | 'avoid';
	turn: number;
	speed: number;
}

const clip = (a: number, lo: number, hi: number) => (a < lo ? lo : a > hi ? hi : a);

export class HeuristicPolicy {
	readonly sim: SimParams = simParams();
	readonly arch = null;

	// The Python version receives the float32 observation, and NumPy 2 keeps float32 when a Python float meets a
	// float32 (the Python constant is cast to float32 first). `f` replays those float32 operations exactly; `t32`
	// tracks whether `turn` became a float32 (it depends on the branch taken).
	act(obs: ArrayLike<number>): { action: [number, number]; trace: RuleTrace } {
		const f = Math.fround;
		const p = this.sim;
		const rays = [0, 1, 2, 3, 4].map((i) => f(obs[i] * f(p.rayRange)));
		const bearing = Math.atan2(obs[5], obs[6]); // > 0: goal is to the right (clockwise)
		const left = Math.min(rays[0], rays[1]);
		const right = Math.min(rays[3], rays[4]);
		const front = Math.min(rays[1], rays[2], rays[3]);

		let rule: RuleTrace['rule'] = 'seek';
		let turn = clip(1.5 * bearing, -1.0, 1.0);
		let t32 = false;
		// don't steer into a close wall on the goal side
		if (turn > 0 && right < f(0.7)) {
			const t = f(f(right - 0.5) * 2.0);
			if (t < turn) {
				turn = t;
				t32 = true;
				rule = 'keep-wall-right';
			}
		}
		if (turn < 0 && left < f(0.7)) {
			const t = -f(f(left - 0.5) * 2.0);
			if (t > turn) {
				turn = t;
				t32 = true;
				rule = 'keep-wall-left';
			}
		}
		if (front < 1.0) {
			const side = right > left ? 1.0 : -1.0;
			const m = f(f(0.4) + f(1.0 - front));
			turn = side * Math.min(1.0, m);
			t32 = m < 1.0;
			rule = 'avoid';
		}
		const go = clip(f(f(front - f(0.4)) / f(0.8)), 0.0, 1.0);
		let speed: number;
		let ul: number;
		let ur: number;
		if (t32) {
			speed = f(go * f(1.0 - f(0.5 * Math.abs(turn))));
			const k = f(f(0.6) * turn);
			ul = clip(f(speed + k), -1.0, 1.0);
			ur = clip(f(speed - k), -1.0, 1.0);
		} else {
			speed = go * (1.0 - 0.5 * Math.abs(turn));
			ul = clip(speed + 0.6 * turn, -1.0, 1.0);
			ur = clip(speed - 0.6 * turn, -1.0, 1.0);
		}
		return { action: [ul, ur], trace: { left, right, front, bearing, rule, turn, speed } };
	}

	reset(): void {}
}
