// Exported actor (port of mazebot/policy.py:MLPPolicy): tanh hidden layers, linear output clipped to [-1, 1].
import { simParams, type SimParams, type SimParamsJson } from '../sim/params';

export interface WeightsJson {
	version: number;
	arch: number[];
	hidden_activation: 'tanh';
	output: 'clip';
	obs_spec?: string[];
	act_spec?: string[];
	sim_params?: SimParamsJson;
	W: number[][][]; // W[k]: (out, in)
	b: number[][];
	meta?: Record<string, unknown>;
}

export interface PolicyOutput {
	action: [number, number];
	/** Every layer for the network panel: [input, hidden..., output before clipping]. */
	layers: Float64Array[];
}

export class MLPPolicy {
	readonly arch: number[];
	readonly sim: SimParams;
	readonly W: Float64Array[]; // row-major (out, in)
	readonly b: Float64Array[];

	constructor(w: WeightsJson) {
		this.W = w.W.map((M) => Float64Array.from(M.flat()));
		this.b = w.b.map((v) => Float64Array.from(v));
		this.arch = [w.W[0][0].length, ...w.W.map((M) => M.length)];
		this.sim = simParams(w.sim_params);
	}

	/** Weight from neuron i of layer k to neuron j of layer k + 1. */
	weight(k: number, j: number, i: number): number {
		return this.W[k][j * this.arch[k] + i];
	}

	forward(obs: ArrayLike<number>): PolicyOutput {
		let h = Float64Array.from(obs);
		const layers = [h];
		const last = this.W.length - 1;
		for (let k = 0; k <= last; k++) {
			const nIn = this.arch[k];
			const nOut = this.arch[k + 1];
			const W = this.W[k];
			const out = new Float64Array(nOut);
			for (let j = 0; j < nOut; j++) {
				let s = 0.0;
				for (let i = 0; i < nIn; i++) s += W[j * nIn + i] * h[i];
				s += this.b[k][j];
				out[j] = k < last ? Math.tanh(s) : s;
			}
			h = out;
			layers.push(h);
		}
		const clip = (a: number) => (a < -1.0 ? -1.0 : a > 1.0 ? 1.0 : a);
		return { action: [clip(h[0]), clip(h[1])], layers };
	}

	reset(): void {}
}
