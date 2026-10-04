// Capsule geometry (port of mazebot/geometry.py, line by line).
//
// A wall is a capsule: segment (ax, ay)-(bx, by) swept by radius `rad`. Capsules are a flat Float64Array of
// rows [ax, ay, bx, by, rad]. A zero-length segment is a circle (post). Only + - * / and sqrt are used, in the
// same order as the NumPy code, so results match Python bit for bit (sin/cos aside, which callers supply).

export type Caps = Float64Array;

export const capCount = (caps: Caps): number => caps.length / 5;

export function capsules(rows: ArrayLike<number>[]): Caps {
	const out = new Float64Array(rows.length * 5);
	rows.forEach((r, i) => {
		for (let k = 0; k < 5; k++) out[i * 5 + k] = r[k];
	});
	return out;
}

export function concatCaps(...parts: Caps[]): Caps {
	const out = new Float64Array(parts.reduce((n, p) => n + p.length, 0));
	let o = 0;
	for (const p of parts) {
		out.set(p, o);
		o += p.length;
	}
	return out;
}

/** Four capsules on the arena edges (their inner surface is `rad` inside the arena). */
export function border(w: number, h: number, rad = 0.05): Caps {
	return capsules([
		[0, 0, w, 0, rad],
		[w, 0, w, h, rad],
		[w, h, 0, h, rad],
		[0, h, 0, 0, rad],
	]);
}

/** Closest point on capsule i's core segment to (px, py) and the distance to it. Writes into `out`. */
export function closestPoint(px: number, py: number, caps: Caps, i: number, out: Float64Array): void {
	const o = i * 5;
	const ax = caps[o];
	const ay = caps[o + 1];
	const dx = caps[o + 2] - ax;
	const dy = caps[o + 3] - ay;
	const l2 = dx * dx + dy * dy;
	let t = l2 > 0.0 ? ((px - ax) * dx + (py - ay) * dy) / l2 : 0.0;
	t = t < 0.0 ? 0.0 : t > 1.0 ? 1.0 : t;
	const cx = ax + t * dx;
	const cy = ay + t * dy;
	const ex = px - cx;
	const ey = py - cy;
	out[0] = cx;
	out[1] = cy;
	out[2] = Math.sqrt(ex * ex + ey * ey);
}

const tmp = new Float64Array(3);

/** Distance from a point to the nearest capsule surface (negative inside a wall). */
export function clearance(px: number, py: number, caps: Caps): number {
	let best = Infinity;
	for (let i = 0, n = capCount(caps); i < n; i++) {
		closestPoint(px, py, caps, i, tmp);
		const c = tmp[2] - caps[i * 5 + 4];
		if (c < best) best = c;
	}
	return best;
}

function rayCircle(ox: number, oy: number, dx: number, dy: number, cx: number, cy: number, r: number): number {
	const qx = ox - cx;
	const qy = oy - cy;
	const b = dx * qx + dy * qy;
	const c = qx * qx + qy * qy - r * r;
	const h = b * b - c;
	const t = -b - Math.sqrt(h > 0.0 ? h : 0.0);
	return h >= 0.0 && t > 0.0 ? t : Infinity;
}

/** First hit distance of one ray (unit direction dx, dy) on capsule i, Infinity on a miss (iq's capsule test). */
export function rayCapsule(ox: number, oy: number, dx: number, dy: number, caps: Caps, i: number): number {
	const o = i * 5;
	const ax = caps[o];
	const ay = caps[o + 1];
	const bx = caps[o + 2];
	const by = caps[o + 3];
	const rad = caps[o + 4];
	// End circles.
	let t = Math.min(rayCircle(ox, oy, dx, dy, ax, ay, rad), rayCircle(ox, oy, dx, dy, bx, by, rad));
	// Body: infinite cylinder around the segment line, clipped to the segment's extent.
	const sx = bx - ax;
	const sy = by - ay;
	const oax = ox - ax;
	const oay = oy - ay;
	const baba = sx * sx + sy * sy;
	const bard = sx * dx + sy * dy;
	const baoa = sx * oax + sy * oay;
	const rdoa = dx * oax + dy * oay;
	const oaoa = oax * oax + oay * oay;
	const a = baba - bard * bard;
	const b = baba * rdoa - baoa * bard;
	const c = baba * oaoa - baoa * baoa - rad * rad * baba;
	const h = b * b - a * c;
	let ok = a > 1e-12 && h >= 0.0;
	const tb = (-b - Math.sqrt(h > 0.0 ? h : 0.0)) / (ok ? a : 1.0);
	const y = baoa + tb * bard;
	ok = ok && tb > 0.0 && y > 0.0 && y < baba;
	if (ok && tb < t) t = tb;
	return t;
}

/** Distance along each unit direction (dirs = [dx0, dy0, dx1, dy1, ...]) to the first hit, capped at maxRange. */
export function castRays(ox: number, oy: number, dirs: ArrayLike<number>, caps: Caps, maxRange: number): Float64Array {
	const k = dirs.length / 2;
	const out = new Float64Array(k);
	const n = capCount(caps);
	for (let r = 0; r < k; r++) {
		const dx = dirs[2 * r];
		const dy = dirs[2 * r + 1];
		let best = Infinity;
		for (let i = 0; i < n; i++) {
			const t = rayCapsule(ox, oy, dx, dy, caps, i);
			if (t < best) best = t;
		}
		out[r] = Math.min(best, maxRange);
	}
	return out;
}

/**
 * Move a circle of radius r at (px, py) out of any capsule it overlaps. Each iteration resolves the deepest
 * overlap (first index on ties) along its contact normal, so the robot slides along walls.
 */
export function pushOut(px: number, py: number, r: number, caps: Caps, iters = 4): { x: number; y: number; contact: boolean } {
	let contact = false;
	const n = capCount(caps);
	if (n === 0) return { x: px, y: py, contact };
	for (let it = 0; it < iters; it++) {
		let bi = 0;
		let bp = -Infinity;
		let bcx = 0;
		let bcy = 0;
		let bd = 0;
		for (let i = 0; i < n; i++) {
			closestPoint(px, py, caps, i, tmp);
			const pen = r + caps[i * 5 + 4] - tmp[2];
			if (pen > bp) {
				bp = pen;
				bi = i;
				bcx = tmp[0];
				bcy = tmp[1];
				bd = tmp[2];
			}
		}
		if (bp <= 0.0) break;
		contact = true;
		let nx: number;
		let ny: number;
		if (bd > 1e-12) {
			nx = (px - bcx) / bd;
			ny = (py - bcy) / bd;
		} else {
			// centre exactly on the core segment: push perpendicular to it
			const o = bi * 5;
			const sx = caps[o + 2] - caps[o];
			const sy = caps[o + 3] - caps[o + 1];
			const ln = Math.sqrt(sx * sx + sy * sy);
			if (ln > 0.0) {
				nx = -sy / ln;
				ny = sx / ln;
			} else {
				nx = 1.0;
				ny = 0.0;
			}
		}
		px = px + nx * bp;
		py = py + ny * bp;
	}
	return { x: px, y: py, contact };
}

/** Ramer-Douglas-Peucker polyline simplification. points: [[x, y], ...]. */
export function rdp(points: number[][], eps: number): number[][] {
	const n = points.length;
	if (n < 3) return points.map((p) => [p[0], p[1]]);
	const keep = new Uint8Array(n);
	keep[0] = keep[n - 1] = 1;
	const seg = new Float64Array(5);
	const stack: [number, number][] = [[0, n - 1]];
	while (stack.length) {
		const [i, j] = stack.pop()!;
		if (j <= i + 1) continue;
		seg.set([points[i][0], points[i][1], points[j][0], points[j][1], 0]);
		let k = -1;
		let dk = -Infinity;
		for (let m = i + 1; m < j; m++) {
			closestPoint(points[m][0], points[m][1], seg, 0, tmp);
			if (tmp[2] > dk) {
				dk = tmp[2];
				k = m;
			}
		}
		if (dk > eps) {
			keep[k] = 1;
			stack.push([i, k], [k, j]);
		}
	}
	return points.filter((_, m) => keep[m]).map((p) => [p[0], p[1]]);
}

/** Chain of capsules along a polyline (consecutive points); a single point becomes a post. */
export function polylineCapsules(points: number[][], rad: number): Caps {
	if (points.length === 1) return capsules([[points[0][0], points[0][1], points[0][0], points[0][1], rad]]);
	const rows: number[][] = [];
	for (let i = 0; i + 1 < points.length; i++) rows.push([points[i][0], points[i][1], points[i + 1][0], points[i + 1][1], rad]);
	return capsules(rows);
}
