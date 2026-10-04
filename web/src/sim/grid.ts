// Occupancy grid inflated by the robot radius and geodesic distance fields on it (port of
// mazebot/distance_field.py). Answers "is B reachable from A?" (the "no path" warning) and gives the path
// length used for the "lost" timeout.

import { capCount, type Caps } from './geometry';

export interface Grid {
	width: number;
	height: number;
	cell: number;
	nx: number;
	ny: number;
	free: Uint8Array; // (ny, nx) row-major: 1 = the robot centre fits here
	clear: Float64Array; // clearance at cell centres
}

const SQRT2 = Math.sqrt(2.0);
const NEIGHBOURS: [number, number][] = [
	[-1, 0],
	[1, 0],
	[0, -1],
	[0, 1],
];
const DIAGONALS: [number, number][] = [
	[-1, -1],
	[1, -1],
	[-1, 1],
	[1, 1],
];

export function buildGrid(width: number, height: number, caps: Caps, robotRadius: number, cell = robotRadius / 2.0): Grid {
	const nx = Math.max(1, Math.ceil(width / cell));
	const ny = Math.max(1, Math.ceil(height / cell));
	const free = new Uint8Array(nx * ny);
	const clear = new Float64Array(nx * ny);
	const n = capCount(caps);
	for (let j = 0; j < ny; j++) {
		const py = (j + 0.5) * cell;
		for (let i = 0; i < nx; i++) {
			const px = (i + 0.5) * cell;
			let best = Infinity;
			for (let k = 0; k < n; k++) {
				// same expression order as geometry.clearance_grid
				const o = k * 5;
				const ax = caps[o];
				const ay = caps[o + 1];
				const dx = caps[o + 2] - ax;
				const dy = caps[o + 3] - ay;
				const l2 = dx * dx + dy * dy;
				let t = l2 > 0.0 ? ((px - ax) * dx + (py - ay) * dy) / l2 : 0.0;
				t = t < 0.0 ? 0.0 : t > 1.0 ? 1.0 : t;
				const ex = px - (ax + t * dx);
				const ey = py - (ay + t * dy);
				const c = Math.sqrt(ex * ex + ey * ey) - caps[o + 4];
				if (c < best) best = c;
			}
			clear[j * nx + i] = best;
			free[j * nx + i] = best >= robotRadius && px < width && py < height ? 1 : 0;
		}
	}
	return { width, height, cell, nx, ny, free, clear };
}

export function cellIndex(g: Grid, x: number, y: number): [number, number] {
	const i = Math.min(Math.max(Math.trunc(x / g.cell), 0), g.nx - 1);
	const j = Math.min(Math.max(Math.trunc(y / g.cell), 0), g.ny - 1);
	return [i, j];
}

/** Binary min-heap of (d, i, j), ordered like Python tuples (heapq). */
class Heap {
	private d: number[] = [];
	private i: number[] = [];
	private j: number[] = [];
	get size() {
		return this.d.length;
	}
	private less(a: number, b: number) {
		const { d, i, j } = this;
		return d[a] < d[b] || (d[a] === d[b] && (i[a] < i[b] || (i[a] === i[b] && j[a] < j[b])));
	}
	private swap(a: number, b: number) {
		for (const arr of [this.d, this.i, this.j]) [arr[a], arr[b]] = [arr[b], arr[a]];
	}
	push(d: number, i: number, j: number) {
		this.d.push(d);
		this.i.push(i);
		this.j.push(j);
		let k = this.d.length - 1;
		while (k > 0) {
			const p = (k - 1) >> 1;
			if (!this.less(k, p)) break;
			this.swap(k, p);
			k = p;
		}
	}
	pop(): [number, number, number] {
		const top: [number, number, number] = [this.d[0], this.i[0], this.j[0]];
		const last = this.d.length - 1;
		this.swap(0, last);
		this.d.pop();
		this.i.pop();
		this.j.pop();
		let k = 0;
		for (;;) {
			const l = 2 * k + 1;
			const r = l + 1;
			let m = k;
			if (l < last && this.less(l, m)) m = l;
			if (r < last && this.less(r, m)) m = r;
			if (m === k) break;
			this.swap(k, m);
			k = m;
		}
		return top;
	}
}

/**
 * Geodesic distance (world units) from (gx, gy) to every free cell; Infinity where unreachable. 8-connected,
 * diagonal moves only when both orthogonal neighbours are free. Seeds every free cell within `seedRadius`
 * (default 1.5 cells) of the point, so the field is ~0 at the goal even if the goal's own cell is blocked.
 */
export function geodesicField(g: Grid, gx: number, gy: number, seedRadius = 1.5 * g.cell): Float64Array {
	const { nx, ny, cell: c, free } = g;
	const dist = new Float64Array(nx * ny).fill(Infinity);
	const heap = new Heap();
	const [i0, j0] = cellIndex(g, gx, gy);
	const k = Math.ceil(seedRadius / c) + 1;
	for (let j = Math.max(0, j0 - k); j < Math.min(ny, j0 + k + 1); j++) {
		for (let i = Math.max(0, i0 - k); i < Math.min(nx, i0 + k + 1); i++) {
			if (!free[j * nx + i]) continue;
			const ex = (i + 0.5) * c - gx;
			const ey = (j + 0.5) * c - gy;
			const d = Math.sqrt(ex * ex + ey * ey);
			if (d <= seedRadius && d < dist[j * nx + i]) {
				dist[j * nx + i] = d;
				heap.push(d, i, j);
			}
		}
	}
	const diag = SQRT2 * c;
	while (heap.size) {
		const [d, i, j] = heap.pop();
		if (d > dist[j * nx + i]) continue;
		for (const [di, dj] of NEIGHBOURS) {
			const a = i + di;
			const b = j + dj;
			if (a >= 0 && a < nx && b >= 0 && b < ny && free[b * nx + a]) {
				const nd = d + 1.0 * c;
				if (nd < dist[b * nx + a]) {
					dist[b * nx + a] = nd;
					heap.push(nd, a, b);
				}
			}
		}
		for (const [di, dj] of DIAGONALS) {
			const a = i + di;
			const b = j + dj;
			if (a >= 0 && a < nx && b >= 0 && b < ny && free[b * nx + a] && free[j * nx + a] && free[b * nx + i]) {
				const nd = d + diag;
				if (nd < dist[b * nx + a]) {
					dist[b * nx + a] = nd;
					heap.push(nd, a, b);
				}
			}
		}
	}
	return dist;
}

/** Continuous geodesic distance at (x, y): min over nearby reached cells of field + straight line. */
export function lookup(g: Grid, dist: Float64Array, x: number, y: number, k = 2): number {
	const [i0, j0] = cellIndex(g, x, y);
	const c = g.cell;
	let best = Infinity;
	for (let j = Math.max(0, j0 - k); j < Math.min(g.ny, j0 + k + 1); j++) {
		for (let i = Math.max(0, i0 - k); i < Math.min(g.nx, i0 + k + 1); i++) {
			const d = dist[j * g.nx + i];
			if (d < Infinity) {
				const ex = (i + 0.5) * c - x;
				const ey = (j + 0.5) * c - y;
				const v = d + Math.sqrt(ex * ex + ey * ey);
				if (v < best) best = v;
			}
		}
	}
	return best;
}

export function reachable(g: Grid, ax: number, ay: number, bx: number, by: number): boolean {
	return lookup(g, geodesicField(g, bx, by), ax, ay) < Infinity;
}
