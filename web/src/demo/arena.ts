// Arena canvas: walls, A/B, trace, rays, goal line and Wheely. World units in, CSS pixels out (y down).
import type { MazeMap, Pose } from '../sim/sim';
import { mix, rgb, type Theme } from './theme';

export interface ArenaFrame {
	map: MazeMap;
	pose: Pose;
	rays: ArrayLike<number>;
	rayAngles: number[];
	rayRange: number;
	radius: number;
	trace: number[]; // x0, y0, x1, y1, ...
	stroke?: number[][]; // pen stroke in progress (world points)
	brush: number;
	eraser?: { x: number; y: number; r: number };
	grab?: 'A' | 'B' | null; // handle being dragged / hovered
}

const MAX_DPR = 2;

export class ArenaView {
	readonly canvas: HTMLCanvasElement;
	private ctx: CanvasRenderingContext2D;
	private sprite: HTMLImageElement | null = null;
	private w = 10;
	private h = 6.25;
	scale = 1; // CSS px per world unit
	theme!: Theme;

	constructor(canvas: HTMLCanvasElement, spriteUrl?: string) {
		this.canvas = canvas;
		this.ctx = canvas.getContext('2d')!;
		if (spriteUrl) {
			const img = new Image();
			img.decoding = 'async';
			img.onload = () => (this.sprite = img);
			img.src = spriteUrl;
		}
	}

	setWorld(width: number, height: number): void {
		this.w = width;
		this.h = height;
		this.canvas.style.aspectRatio = `${width} / ${height}`;
		this.resize();
	}

	resize(): void {
		const cssW = this.canvas.clientWidth || 1;
		const dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR);
		this.scale = cssW / this.w;
		const pxW = Math.round(cssW * dpr);
		const pxH = Math.round((cssW * this.h * dpr) / this.w);
		if (this.canvas.width !== pxW || this.canvas.height !== pxH) {
			this.canvas.width = pxW;
			this.canvas.height = pxH;
		}
		this.ctx.setTransform(dpr * this.scale, 0, 0, dpr * this.scale, 0, 0);
	}

	toWorld(clientX: number, clientY: number): [number, number] {
		const r = this.canvas.getBoundingClientRect();
		return [((clientX - r.left) / r.width) * this.w, ((clientY - r.top) / r.height) * this.h];
	}

	draw(f: ArenaFrame): void {
		const { ctx, theme: t } = this;
		const px = 1 / this.scale; // one CSS pixel in world units
		const { map } = f;
		ctx.fillStyle = t.surface;
		ctx.fillRect(0, 0, this.w, this.h);

		// trace
		if (f.trace.length >= 4) {
			ctx.beginPath();
			ctx.moveTo(f.trace[0], f.trace[1]);
			for (let i = 2; i < f.trace.length; i += 2) ctx.lineTo(f.trace[i], f.trace[i + 1]);
			ctx.strokeStyle = t.accent;
			ctx.globalAlpha = 0.45;
			ctx.lineWidth = Math.max(0.05, 2 * px);
			ctx.lineJoin = 'round';
			ctx.stroke();
			ctx.globalAlpha = 1;
		}

		// walls
		ctx.strokeStyle = t.fg;
		ctx.fillStyle = t.fg;
		ctx.lineCap = 'round';
		ctx.globalAlpha = 0.85;
		for (const w of map.walls) this.capsule(w);
		if (f.stroke && f.stroke.length) {
			ctx.globalAlpha = 0.5;
			for (let i = 0; i < f.stroke.length; i++) {
				const a = f.stroke[Math.max(0, i - 1)];
				const b = f.stroke[i];
				this.capsule([a[0], a[1], b[0], b[1], f.brush]);
			}
		}
		ctx.globalAlpha = 1;

		// goal line and rays
		const { x, y, th } = f.pose;
		ctx.lineCap = 'butt';
		ctx.strokeStyle = t.muted;
		ctx.globalAlpha = 0.45;
		ctx.lineWidth = Math.max(0.02, px);
		ctx.beginPath();
		ctx.moveTo(x, y);
		ctx.lineTo(map.goal[0], map.goal[1]);
		ctx.stroke();
		ctx.globalAlpha = 1;
		const muted = rgb(t.muted);
		const accent = rgb(t.accent);
		ctx.setLineDash([0.09, 0.07]);
		ctx.lineWidth = Math.max(0.03, 1.5 * px);
		f.rayAngles.forEach((a, i) => {
			const d = f.rays[i];
			ctx.strokeStyle = mix(muted, accent, 0.25 + 0.75 * (1 - d / f.rayRange));
			ctx.beginPath();
			ctx.moveTo(x, y);
			ctx.lineTo(x + d * Math.cos(th + a), y + d * Math.sin(th + a));
			ctx.stroke();
		});
		ctx.setLineDash([]);

		// A and B
		this.marker(map.start[0], map.start[1], 'A', false, f.grab === 'A');
		this.marker(map.goal[0], map.goal[1], 'B', true, f.grab === 'B');

		// Wheely
		this.robot(x, y, th, f.radius);

		// eraser cursor
		if (f.eraser) {
			ctx.strokeStyle = t.fg;
			ctx.lineWidth = px;
			ctx.setLineDash([4 * px, 3 * px]);
			ctx.beginPath();
			ctx.arc(f.eraser.x, f.eraser.y, f.eraser.r, 0, 2 * Math.PI);
			ctx.stroke();
			ctx.setLineDash([]);
		}

		// arena edge
		ctx.strokeStyle = t.fg;
		ctx.lineWidth = 0.1;
		ctx.strokeRect(0, 0, this.w, this.h);
	}

	private capsule(w: ArrayLike<number>): void {
		const { ctx } = this;
		if (w[0] === w[2] && w[1] === w[3]) {
			ctx.beginPath();
			ctx.arc(w[0], w[1], w[4], 0, 2 * Math.PI);
			ctx.fill();
			return;
		}
		ctx.lineWidth = 2 * w[4];
		ctx.beginPath();
		ctx.moveTo(w[0], w[1]);
		ctx.lineTo(w[2], w[3]);
		ctx.stroke();
	}

	private marker(x: number, y: number, label: string, goal: boolean, active: boolean): void {
		const { ctx, theme: t } = this;
		const r = 0.26;
		if (goal) {
			ctx.fillStyle = t.accent;
			ctx.globalAlpha = 0.18;
			ctx.beginPath();
			ctx.arc(x, y, 0.35, 0, 2 * Math.PI);
			ctx.fill();
			ctx.globalAlpha = 1;
		}
		ctx.beginPath();
		ctx.arc(x, y, active ? r * 1.15 : r, 0, 2 * Math.PI);
		ctx.fillStyle = goal ? t.accent : t.surface;
		ctx.fill();
		if (!goal || active) {
			ctx.lineWidth = 0.05;
			ctx.strokeStyle = active ? t.fg : t.muted;
			ctx.stroke();
		}
		ctx.fillStyle = goal ? t.accentFg : t.muted;
		ctx.font = `600 0.28px ${t.sans}`;
		ctx.textAlign = 'center';
		ctx.textBaseline = 'middle';
		ctx.fillText(label, x, y + 0.01);
	}

	private robot(x: number, y: number, th: number, r: number): void {
		const { ctx, theme: t } = this;
		ctx.save();
		ctx.translate(x, y);
		ctx.rotate(th + Math.PI / 2); // sprite faces -y
		const w = 2.6 * r;
		if (this.sprite) {
			const h = (w * this.sprite.naturalHeight) / this.sprite.naturalWidth || (w * 180) / 260;
			ctx.drawImage(this.sprite, -w / 2, -h / 2, w, h);
		} else {
			// simple stand-in: body + two wheels + a nose
			ctx.fillStyle = t.fg;
			ctx.fillRect(-w / 2, -r * 0.8, r * 0.35, r * 1.6);
			ctx.fillRect(w / 2 - r * 0.35, -r * 0.8, r * 0.35, r * 1.6);
			ctx.fillStyle = t.accent;
			ctx.beginPath();
			ctx.arc(0, 0, r * 0.85, 0, 2 * Math.PI);
			ctx.fill();
			ctx.fillStyle = t.accentFg;
			ctx.beginPath();
			ctx.arc(0, -r * 0.5, r * 0.18, 0, 2 * Math.PI);
			ctx.fill();
		}
		ctx.restore();
	}
}
