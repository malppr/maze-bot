// Live network panel. Nodes are coloured by activation (accent = positive, --ai = negative). Edges show the
// signal travelling right now (weight x sending activation): each neuron shows only its strongest incoming
// signal, and only if it is >= FLOW_MIN of the strongest in that layer. Hover / tap a neuron to see all of its
// connections and its value.
import type { MLPPolicy } from '../policy/mlp';
import { mix, rgb, type Theme } from './theme';

const FLOW_MIN = 0.35;
const MAX_DPR = 2;

export class BrainView {
	readonly canvas: HTMLCanvasElement;
	private ctx: CanvasRenderingContext2D;
	private net: MLPPolicy | null = null;
	private names: string[] = [];
	private layers: Float64Array[] = [];
	private xs: number[] = [];
	private ys: number[][] = [];
	private nodeR = 7;
	private cssW = 1;
	private cssH = 1;
	private focus: [number, number] | null = null;
	compact = false;
	theme!: Theme;

	constructor(canvas: HTMLCanvasElement) {
		this.canvas = canvas;
		this.ctx = canvas.getContext('2d')!;
		canvas.addEventListener('pointermove', (e) => e.pointerType === 'mouse' && this.setFocus(this.hit(e)));
		canvas.addEventListener('pointerleave', (e) => e.pointerType === 'mouse' && this.setFocus(null));
		canvas.addEventListener('pointerdown', (e) => {
			if (e.pointerType === 'mouse') return;
			const h = this.hit(e);
			this.setFocus(h && this.focus && h[0] === this.focus[0] && h[1] === this.focus[1] ? null : h);
		});
	}

	setNet(net: MLPPolicy, names: string[]): void {
		this.net = net;
		this.names = names;
		this.focus = null;
		this.layers = [];
		this.layout();
	}

	update(layers: Float64Array[]): void {
		this.layers = layers;
		this.draw();
	}

	layout(): void {
		if (!this.net) return;
		const arch = this.net.arch;
		const cssW = this.canvas.clientWidth || 1;
		const narrow = cssW < 460;
		const full = !this.compact;
		const maxN = Math.max(...arch);
		const step = full ? (narrow ? 21 : 25) : 9.5;
		const pad = full ? 18 : 10;
		const cssH = Math.round((maxN - 1) * step + 2 * pad + (full ? 18 : 0));
		const padL = full ? (narrow ? 86 : 104) : 12;
		const padR = full ? (narrow ? 96 : 124) : 12;
		this.nodeR = full ? (narrow ? 6 : 7.5) : 4.2;
		this.xs = arch.map((_, k) => padL + ((cssW - padL - padR) * k) / (arch.length - 1));
		const mid = (cssH - (full ? 18 : 0)) / 2;
		this.ys = arch.map((n) => Array.from({ length: n }, (_, i) => mid + (i - (n - 1) / 2) * step));
		this.cssW = cssW;
		this.cssH = cssH;
		this.canvas.style.height = `${cssH}px`;
		const dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR);
		this.canvas.width = Math.round(cssW * dpr);
		this.canvas.height = Math.round(cssH * dpr);
		this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
		this.draw();
	}

	private setFocus(f: [number, number] | null): void {
		const same = f === this.focus || (f && this.focus && f[0] === this.focus[0] && f[1] === this.focus[1]);
		if (same) return;
		this.focus = f;
		this.draw();
	}

	private hit(e: PointerEvent): [number, number] | null {
		const r = this.canvas.getBoundingClientRect();
		const x = e.clientX - r.left;
		const y = e.clientY - r.top;
		let best: [number, number] | null = null;
		let bd = (this.nodeR + 6) ** 2;
		this.ys.forEach((col, k) =>
			col.forEach((ny, i) => {
				const d = (this.xs[k] - x) ** 2 + (ny - y) ** 2;
				if (d < bd) {
					bd = d;
					best = [k, i];
				}
			}),
		);
		return best;
	}

	draw(): void {
		const { ctx, net, theme: t } = this;
		ctx.clearRect(0, 0, this.cssW, this.cssH);
		if (!net || !this.layers.length) return;
		const L = this.layers;
		const pos = rgb(t.accent);
		const neg = rgb(t.neg);
		const surf = rgb(t.surface);
		const fc = this.focus;
		const full = !this.compact;

		// edges
		ctx.lineCap = 'round';
		for (let k = 0; k < net.W.length; k++) {
			const nIn = net.arch[k];
			const nOut = net.arch[k + 1];
			const sig = (j: number, i: number) => net.weight(k, j, i) * L[k][i];
			let max = 1e-12;
			for (let j = 0; j < nOut; j++) for (let i = 0; i < nIn; i++) max = Math.max(max, Math.abs(sig(j, i)));
			for (let j = 0; j < nOut; j++) {
				if (fc) {
					// focused neuron: all of its incoming / outgoing connections, by signal strength
					for (let i = 0; i < nIn; i++) {
						const mine = (fc[0] === k && fc[1] === i) || (fc[0] === k + 1 && fc[1] === j);
						if (mine) this.edge(k, i, j, sig(j, i), max, pos, neg, surf);
					}
					continue;
				}
				let bi = 0;
				for (let i = 1; i < nIn; i++) if (Math.abs(sig(j, i)) > Math.abs(sig(j, bi))) bi = i;
				if (Math.abs(sig(j, bi)) >= FLOW_MIN * max) this.edge(k, bi, j, sig(j, bi), max, pos, neg, surf);
			}
		}

		// nodes
		const last = L.length - 1;
		L.forEach((layer, k) =>
			layer.forEach((v0, i) => {
				const v = k === last ? Math.max(-1, Math.min(1, v0)) : v0;
				const on = fc && fc[0] === k && fc[1] === i;
				const dim = fc && !on && !this.linked(k, i);
				ctx.globalAlpha = dim ? 0.35 : 1;
				ctx.beginPath();
				ctx.arc(this.xs[k], this.ys[k][i], on ? this.nodeR * 1.35 : this.nodeR, 0, 2 * Math.PI);
				ctx.fillStyle = mix(surf, v >= 0 ? pos : neg, 0.12 + 0.88 * Math.min(1, Math.abs(v)));
				ctx.fill();
				ctx.lineWidth = on ? 2 : 1;
				ctx.strokeStyle = on ? t.fg : t.border;
				ctx.stroke();
				ctx.globalAlpha = 1;
				if (on && full && k > 0 && k < last) this.text(v.toFixed(2), this.xs[k], this.ys[k][i] - this.nodeR - 9, 'center', t.fg, true);
			}),
		);
		if (!full) return;

		// labels
		const narrow = this.cssW < 460;
		const f = (v: number) => (v >= 0 ? ' ' : '') + v.toFixed(2);
		L[0].forEach((v, i) => {
			this.text(this.names[i] ?? '', this.xs[0] - 13, this.ys[0][i], 'right', t.fg);
			this.text(f(v), this.xs[0] - (narrow ? 58 : 70), this.ys[0][i], 'right', t.muted, true);
		});
		['left wheel', 'right wheel'].forEach((name, i) => {
			const v = Math.max(-1, Math.min(1, L[last][i]));
			this.text(narrow ? name.replace(' wheel', '') : name, this.xs[last] + 13, this.ys[last][i], 'left', t.fg);
			this.text(f(v), this.xs[last] + (narrow ? 50 : 84), this.ys[last][i], 'left', t.fg, true, true);
		});
		const cols = ['inputs', ...net.arch.slice(1, -1).map(String), 'wheels'];
		cols.forEach((c, k) => this.text(c, this.xs[k], this.cssH - 6, 'center', t.muted, true));
	}

	private linked(k: number, i: number): boolean {
		const fc = this.focus!;
		return Math.abs(fc[0] - k) === 1;
	}

	private edge(k: number, i: number, j: number, s: number, max: number, pos: number[], neg: number[], surf: number[]) {
		const { ctx } = this;
		const m = Math.min(1, Math.abs(s) / max);
		ctx.strokeStyle = mix(surf as [number, number, number], (s >= 0 ? pos : neg) as [number, number, number], 0.25 + 0.75 * m);
		ctx.lineWidth = (this.compact ? 0.5 : 0.7) + (this.compact ? 1.4 : 2.6) * m;
		ctx.beginPath();
		ctx.moveTo(this.xs[k], this.ys[k][i]);
		ctx.lineTo(this.xs[k + 1], this.ys[k + 1][j]);
		ctx.stroke();
	}

	private text(s: string, x: number, y: number, align: CanvasTextAlign, color: string, mono = false, bold = false) {
		const { ctx } = this;
		const narrow = this.cssW < 460;
		ctx.font = `${bold ? 600 : 400} ${mono ? (narrow ? 9.5 : 10.5) : narrow ? 10.5 : 12}px ${mono ? this.theme.mono : this.theme.sans}`;
		ctx.textAlign = align;
		ctx.textBaseline = 'middle';
		ctx.fillStyle = color;
		ctx.fillText(s, x, y);
	}
}
