// mountMazeDemo: the Wheely's maze demo (contract: PLAN.md §7). Layout agreed with Bryan (site PLAN §14):
// version chips -> arena -> toolbar -> brain strip. The sim runs at a fixed 15 Hz policy rate; the arena is
// drawn every animation frame with Wheely interpolated between policy steps.
import { HeuristicPolicy } from '../policy/heuristic';
import { MLPPolicy } from '../policy/mlp';
import { closestPoint, polylineCapsules, rdp } from '../sim/geometry';
import { obsNames, simParams, type SimParams } from '../sim/params';
import { PRESETS, presetMap, type Preset } from '../sim/presets';
import { Episode, wrapAngle, type MazeMap, type Pose } from '../sim/sim';
import { ArenaView } from './arena';
import { BrainView } from './brain';
import { RuleView } from './rule';
import { injectStyles } from './styles';
import { readTheme, watchTheme } from './theme';
import { DEFAULT_VERSION, VERSIONS, loadWeights } from './versions';

export interface MazeDemoOptions {
	mode?: 'obstacles' | 'draw'; // "obstacles" = presets; "draw" = empty arena with the pen selected
	showNetwork?: boolean; // brain panel (default true)
	version?: string; // which Wheely to start with (default: the final one)
	sprite?: string; // URL of the top-down mascot sprite (faces -y); a simple robot is drawn without it
}

export interface MazeDemoHandle {
	destroy(): void;
}

type Tool = 'move' | 'draw' | 'erase';

const BRUSH = 0.1; // drawn wall radius (training strokes: 0.04-0.2)
const ERASER = 0.25;
const RDP_EPS = 0.03;
const MAX_WALLS = 400; // capsules, keeps phones fast
const PORTRAIT_BELOW = 600; // container px
const SPEEDS = [0.5, 1, 4];
const MAX_TRACE = 4000;
const HINT_AFTER = 4; // seconds of driving before the drawing tip appears

const ICON = {
	reset: '<path d="M4 12a8 8 0 1 0 2.4-5.7M4 4v4h4"/>',
	play: '<path d="M7 5l12 7-12 7z"/>',
	pause: '<path d="M8 5v14M16 5v14"/>',
	step: '<path d="M6 5l9 7-9 7zM18 5v14"/>',
};
const svg = (p: string) => `<svg viewBox="0 0 24 24" aria-hidden="true">${p}</svg>`;

/** Mini map of a preset for its layout chip (walls, A and B). */
function thumb(p: Preset): string {
	const svg = (m: MazeMap, cls: string) => {
		const walls = m.walls
			.map((w) => `<line x1="${w[0]}" y1="${w[1]}" x2="${w[2]}" y2="${w[3]}" stroke-width="${Math.max(0.3, 2 * w[4])}"/>`)
			.join('');
		return `<svg class="${cls}" viewBox="-0.2 -0.2 ${m.width + 0.4} ${m.height + 0.4}" aria-hidden="true"><rect x="0" y="0" width="${m.width}" height="${m.height}" rx="0.3"/>${walls}<circle class="wm-a" cx="${m.start[0]}" cy="${m.start[1]}" r="0.45"/><circle class="wm-b" cx="${m.goal[0]}" cy="${m.goal[1]}" r="0.45"/></svg>`;
	};
	return svg(presetMap(p), 'wm-land') + svg(presetMap(p, true), 'wm-port');
}

function transpose(m: MazeMap): MazeMap {
	return {
		name: m.name,
		width: m.height,
		height: m.width,
		walls: m.walls.map((w) => [w[1], w[0], w[3], w[2], w[4]]),
		start: [m.start[1], m.start[0], Math.PI / 2 - m.start[2]],
		goal: [m.goal[1], m.goal[0]],
	};
}

export function mountMazeDemo(el: HTMLElement, opts: MazeDemoOptions = {}): MazeDemoHandle {
	injectStyles();
	const showNetwork = opts.showNetwork ?? true;
	const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

	// ------------------------------------------------------------------ DOM
	const root = document.createElement('div');
	root.className = 'wm';
	const startLayout = opts.mode === 'draw' ? 'Open field' : 'Rooms';
	root.innerHTML = `
		<div class="wm-versions">
			<div class="wm-chips" role="radiogroup" aria-label="Which Wheely">
				${VERSIONS.map((v) => `<button type="button" class="wm-chip" role="radio" aria-checked="false" data-v="${v.id}">${v.name}</button>`).join('')}
			</div>
			<p class="wm-blurb"></p>
		</div>
		<div class="wm-stage">
			<canvas class="wm-arena" tabindex="0" role="img" aria-label="Arena: Wheely drives from A to B. Drag A or B; draw or erase walls with the tools below."></canvas>
			<span class="wm-status"></span>
			<span class="wm-warn" hidden>No path to B — Wheely will get lost</span>
			<div class="wm-result" hidden><span class="wm-result-text"></span>
				<button type="button" class="wm-primary" data-act="again">Again</button><button type="button" data-act="shuffle">New A/B</button></div>
			<div class="wm-hint" hidden role="status"><span class="wm-hint-text">Try drawing a wall in Wheely’s way</span>
				<button type="button" class="wm-primary" data-act="hint-draw">✏️ Draw</button><button type="button" class="wm-x" data-act="hint-close" aria-label="Dismiss tip">✕</button></div>
		</div>
		<p class="wm-note" hidden></p>
		<div class="wm-layouts" role="radiogroup" aria-label="Layout">
			${PRESETS.map((p) => `<button type="button" class="wm-lay" role="radio" aria-checked="${p.name === startLayout}" data-layout="${p.name}">${thumb(p)}<span>${p.name}</span></button>`).join('')}
		</div>
		<div class="wm-toolbar">
			<div class="wm-group">
				<div class="wm-seg wm-tools" role="radiogroup" aria-label="Tool">
					<button type="button" role="radio" data-tool="move" title="Drag A and B">Move</button>
					<button type="button" role="radio" data-tool="draw" title="Draw walls">Draw</button>
					<button type="button" role="radio" data-tool="erase" title="Erase walls">Erase</button>
				</div>
				<div class="wm-seg"><button type="button" data-act="clear" title="Remove all walls">Clear</button></div>
			</div>
			<div class="wm-group wm-play">
				<button type="button" class="wm-icon" data-act="reset" aria-label="Restart from A" title="Restart from A">${svg(ICON.reset)}</button>
				<button type="button" class="wm-icon wm-main" data-act="play" aria-label="Pause">${svg(ICON.pause)}</button>
				<button type="button" class="wm-icon" data-act="step" aria-label="Step once" title="Step once">${svg(ICON.step)}</button>
				<div class="wm-seg wm-speed" role="radiogroup" aria-label="Speed">
					${SPEEDS.map((s) => `<button type="button" role="radio" data-speed="${s}">${s === 0.5 ? '½' : s}×</button>`).join('')}
				</div>
			</div>
		</div>
		<div class="wm-panel"${showNetwork ? '' : ' hidden'}>
			<div class="wm-panel-head"><p class="wm-title"></p>
				<div class="wm-legend"><span><i style="background:var(--wm-accent)"></i>pushes up</span><span><i style="background:var(--wm-neg)"></i>pushes down</span></div></div>
			<canvas class="wm-brain" aria-label="Wheely's neural network, live. Hover or tap a neuron to see its connections."></canvas>
			<div class="wm-panel-rule"></div>
			<div class="wm-foot"><span>left <b class="wm-ul">0.00</b></span><span>right <b class="wm-ur">0.00</b></span><button type="button" data-act="enlarge">tap to enlarge</button></div>
		</div>
		<p class="wm-sr" aria-live="polite"></p>`;
	el.appendChild(root);
	const $ = <T extends Element = HTMLElement>(s: string) => root.querySelector(s) as unknown as T;
	const stage = $('.wm-stage');
	const statusEl = $('.wm-status');
	const warnEl = $('.wm-warn');
	const resultEl = $('.wm-result');
	const resultText = $('.wm-result-text');
	const hintEl = $('.wm-hint');
	const noteEl = $('.wm-note');
	const playBtn = $<HTMLButtonElement>('[data-act=play]');
	const panel = $('.wm-panel');
	const ruleBox = $('.wm-panel-rule');
	const brainCanvas = $<HTMLCanvasElement>('.wm-brain');
	const live = $('[aria-live]');

	const arena = new ArenaView($<HTMLCanvasElement>('.wm-arena'), opts.sprite);
	const brain = new BrainView(brainCanvas);
	const rule = new RuleView();
	ruleBox.appendChild(rule.el);
	let theme = readTheme(root);
	arena.theme = brain.theme = theme;

	// ------------------------------------------------------------------ state
	let versionId = VERSIONS.some((v) => v.id === opts.version) ? opts.version! : DEFAULT_VERSION;
	let net: MLPPolicy | null = null;
	let heuristic: HeuristicPolicy | null = null;
	let p: SimParams = simParams();
	let portrait = root.clientWidth > 0 && root.clientWidth < PORTRAIT_BELOW;
	let layout = startLayout;
	let map: MazeMap = presetMap(PRESETS.find((x) => x.name === layout)!, portrait);
	let ep = new Episode(map, p);
	let running = !reduced;
	let started = !reduced; // reduced motion: wait for an explicit Play
	let speed = 1;
	let tool: Tool = opts.mode === 'draw' ? 'draw' : 'move';
	let ready = false;
	let trace: number[] = [];
	let prev: Pose = { ...ep.pose };
	let acc = 0;
	let lastAction: [number, number] = [0, 0];
	let visible = true;
	let pageVisible = !document.hidden;
	let raf = 0;
	let lastT = 0;
	let dirty = true;
	let expanded = false;
	let loadToken = 0;
	// pointer interaction
	let drag: 'A' | 'B' | null = null;
	let hoverHandle: 'A' | 'B' | null = null;
	let stroke: number[][] | null = null;
	let strokeBase: number[][] = [];
	let erasing = false;
	let eraser: { x: number; y: number; r: number } | undefined;
	let pointerId = -1;
	// drawing tip: pops up after HINT_AFTER s of driving, gone after the first stroke / dismiss / own tool choice
	let hint: 'waiting' | 'shown' | 'drawing' | 'done' = 'waiting';
	let driven = 0;

	const policyDt = () => p.dt * p.frameSkip;
	const simTime = () => (ep.steps * policyDt()).toFixed(1);

	// ------------------------------------------------------------------ episode / UI sync
	function restart(): void {
		ep.reset();
		prev = { ...ep.pose };
		trace = [ep.pose.x, ep.pose.y];
		acc = 0;
		lastAction = [0, 0];
		resultEl.hidden = true;
		sense();
		hintEl.hidden = hint !== 'shown' && hint !== 'drawing';
		syncWarn();
		syncStatus();
		dirty = true;
		kick();
	}

	function setMap(m: MazeMap): void {
		map = m;
		arena.setWorld(m.width, m.height);
		ep.setMap(m);
		restart();
	}

	function sense(): void {
		// network / rule panel for the current observation (before the next step)
		const obs = ep.observation();
		if (net) brain.update(net.forward(obs).layers);
		else if (heuristic) {
			const r = heuristic.act(obs);
			rule.update(r.trace, r.action);
		}
	}

	function syncWarn(): void {
		warnEl.hidden = ep.reachable;
	}

	function syncStatus(): void {
		const t = `${simTime()} s`;
		statusEl.textContent = !ready
			? 'Loading…'
			: ep.done === 'success'
				? `Reached B · ${t}`
				: ep.done === 'lost'
					? `Lost · ${t}`
					: !started
						? 'Press play'
						: running
							? `Driving · ${t}`
							: `Paused · ${t}`;
		playBtn.setAttribute('aria-label', running ? 'Pause' : 'Play');
		playBtn.title = running ? 'Pause' : 'Play';
		playBtn.innerHTML = svg(running ? ICON.pause : ICON.play);
	}

	function announce(s: string): void {
		live.textContent = s;
	}

	function stepOnce(): void {
		if (ep.done) return;
		prev = { ...ep.pose };
		const obs = ep.observation();
		let action: [number, number];
		if (net) {
			action = net.forward(obs).action;
		} else {
			action = heuristic!.act(obs).action;
		}
		lastAction = action;
		const res = ep.step(action);
		driven += policyDt();
		if (hint === 'waiting' && driven >= HINT_AFTER) setHint('shown');
		trace.push(ep.pose.x, ep.pose.y);
		if (trace.length > 2 * MAX_TRACE) trace = trace.slice(-2 * MAX_TRACE);
		sense();
		$('.wm-ul').textContent = lastAction[0].toFixed(2);
		$('.wm-ur').textContent = lastAction[1].toFixed(2);
		if (res.success || res.lost) {
			acc = 0;
			prev = { ...ep.pose };
			resultText.textContent = res.success ? `Reached B in ${simTime()} s` : `Lost after ${simTime()} s`;
			resultEl.hidden = false;
			hintEl.hidden = true;
			announce(res.success ? `Wheely reached B in ${simTime()} seconds.` : 'Wheely got lost.');
		}
		dirty = true;
	}

	// ------------------------------------------------------------------ versions
	async function setVersion(id: string): Promise<void> {
		versionId = id;
		const v = VERSIONS.find((x) => x.id === id)!;
		root.querySelectorAll<HTMLElement>('.wm-chip').forEach((c) => c.setAttribute('aria-checked', String(c.dataset.v === id)));
		$('.wm-blurb').textContent = v.blurb;
		const token = ++loadToken;
		const w = await loadWeights(v);
		if (token !== loadToken) return; // a newer choice won
		if (w) {
			net = new MLPPolicy(w);
			heuristic = null;
			p = net.sim;
		} else {
			heuristic = new HeuristicPolicy();
			net = null;
			p = heuristic.sim;
		}
		ep = new Episode(map, p);
		ready = true;
		const owner = v.name.endsWith('s') ? `${v.name}'` : `${v.name}'s`;
		$('.wm-title').textContent = `${owner} ${net ? 'brain' : 'rule'} · live`;
		$('.wm-legend').hidden = !net;
		brainCanvas.hidden = !net;
		ruleBox.hidden = !!net;
		root.querySelector<HTMLElement>('.wm-foot')!.style.visibility = net ? '' : 'hidden';
		if (net) brain.setNet(net, obsNames(p));
		restart();
	}

	// ------------------------------------------------------------------ loop
	function kick(): void {
		if (!raf && visible && pageVisible) raf = requestAnimationFrame(frame);
	}

	function frame(t: number): void {
		raf = 0;
		const dt = lastT ? Math.min(0.1, (t - lastT) / 1000) : 0;
		lastT = t;
		const stepping = ready && running && !ep.done && !drag && !stroke;
		if (stepping) {
			acc += dt * speed;
			let n = 0;
			while (acc >= policyDt() && n < 8 && !ep.done) {
				stepOnce();
				acc -= policyDt();
				n++;
			}
			if (n) syncStatus();
		}
		if (stepping || dirty) {
			const a = stepping && !ep.done ? Math.min(1, acc / policyDt()) : 1;
			const pose = {
				x: prev.x + (ep.pose.x - prev.x) * a,
				y: prev.y + (ep.pose.y - prev.y) * a,
				th: prev.th + wrapAngle(ep.pose.th - prev.th) * a,
			};
			arena.draw({
				map: ep.map,
				pose,
				rays: ep.rays,
				rayAngles: p.rayAngles,
				rayRange: p.rayRange,
				radius: p.radius,
				trace: a < 1 ? [...trace.slice(0, -2), pose.x, pose.y] : trace,
				stroke: stroke ?? undefined,
				brush: BRUSH,
				eraser,
				grab: drag ?? hoverHandle,
			});
			dirty = false;
		}
		if (stepping) kick();
		else lastT = 0;
	}

	// ------------------------------------------------------------------ controls
	function setHint(h: typeof hint): void {
		hint = h;
		hintEl.hidden = !(h === 'shown' || h === 'drawing') || !resultEl.hidden;
		if (h === 'drawing') {
			$('.wm-hint-text').textContent = 'Drag across the arena to draw a wall';
			$('[data-act=hint-draw]').hidden = true;
		}
	}

	function setTool(t: Tool, byUser = false): void {
		if (byUser) {
			if (hint === 'waiting') setHint('done');
			else if (hint === 'shown') setHint(t === 'draw' ? 'drawing' : 'done');
			else if (hint === 'drawing' && t !== 'draw') setHint('done');
		}
		tool = t;
		stage.dataset.tool = t;
		root.querySelectorAll<HTMLElement>('[data-tool]').forEach((b) => b.setAttribute('aria-checked', String(b.dataset.tool === t)));
		eraser = undefined;
		dirty = true;
		kick();
	}

	function setSpeed(s: number): void {
		speed = s;
		root.querySelectorAll<HTMLElement>('[data-speed]').forEach((b) => b.setAttribute('aria-checked', String(Number(b.dataset.speed) === s)));
	}

	function togglePlay(force?: boolean): void {
		running = force ?? !running;
		started = started || running;
		if (running && ep.done) restart();
		syncStatus();
		kick();
	}

	function shuffle(): void {
		// random A and B on free cells with some clearance, B reachable, path at least 40% of the arena's long side
		const g = ep.grid;
		const free: number[] = [];
		for (let k = 0; k < g.free.length; k++) if (g.free[k] && g.clear[k] >= p.radius + 0.15) free.push(k);
		if (free.length < 2) return;
		const pick = () => {
			const k = free[Math.floor(Math.random() * free.length)];
			return [((k % g.nx) + 0.5) * g.cell, (Math.floor(k / g.nx) + 0.5) * g.cell];
		};
		const minD = 0.4 * Math.max(map.width, map.height);
		for (let tries = 0; tries < 60; tries++) {
			const [bx, by] = pick();
			const [ax, ay] = pick();
			if (Math.hypot(bx - ax, by - ay) < minD) continue;
			const m: MazeMap = { ...map, start: [ax, ay, Math.atan2(by - ay, bx - ax)], goal: [bx, by] };
			const test = new Episode(m, p);
			if (!test.reachable && tries < 59) continue;
			setMap(m);
			return;
		}
	}

	function setWalls(walls: number[][], replan: boolean): void {
		map = { ...map, walls };
		ep.updateWalls(walls, replan);
		if (replan) {
			// mid-run: from where Wheely is; before / after a run: from A (the next run)
			const midRun = started && !ep.done && ep.steps > 0;
			const ok = midRun ? ep.reachableFrom(ep.pose.x, ep.pose.y) : ep.reachableFrom(map.start[0], map.start[1]);
			warnEl.hidden = ok;
			if (!ok) announce('No path to B. Wheely will get lost.');
		}
		dirty = true;
		kick();
	}

	root.addEventListener('click', (e) => {
		const b = (e.target as HTMLElement).closest<HTMLElement>('button');
		if (!b || !root.contains(b)) return;
		if (b.dataset.v) void setVersion(b.dataset.v);
		else if (b.dataset.tool) setTool(b.dataset.tool as Tool, true);
		else if (b.dataset.speed) setSpeed(Number(b.dataset.speed));
		else if (b.dataset.layout) setLayout(b.dataset.layout);
		else
			switch (b.dataset.act) {
				case 'play':
					togglePlay();
					break;
				case 'reset':
					restart();
					break;
				case 'step':
					if (running) togglePlay(false);
					if (ep.done) restart();
					else if (ready) {
						started = true;
						stepOnce();
						prev = { ...ep.pose };
						syncStatus();
						kick();
					}
					break;
				case 'again':
					restart();
					togglePlay(true);
					break;
				case 'shuffle':
					shuffle();
					togglePlay(true);
					break;
				case 'clear':
					setWalls([], true);
					break;
				case 'hint-draw':
					setTool('draw', true);
					break;
				case 'hint-close':
					setHint('done');
					break;
				case 'enlarge':
					expanded = !expanded;
					syncPanel();
					break;
			}
	});

	function setLayout(name: string): void {
		layout = name;
		root.querySelectorAll<HTMLElement>('[data-layout]').forEach((b) => b.setAttribute('aria-checked', String(b.dataset.layout === name)));
		setMap(presetMap(PRESETS.find((x) => x.name === name)!, portrait));
	}

	root.addEventListener('keydown', (e) => {
		if (e.key === ' ' && e.target === arena.canvas) {
			e.preventDefault();
			togglePlay();
		}
	});

	// ------------------------------------------------------------------ pointer: drag A/B, draw, erase
	const canvas = arena.canvas;
	const handleAt = (x: number, y: number): 'A' | 'B' | null => {
		const r = Math.max(0.45, 22 / arena.scale);
		const dA = Math.hypot(x - map.start[0], y - map.start[1]);
		const dB = Math.hypot(x - map.goal[0], y - map.goal[1]);
		if (dB <= r && dB <= dA) return 'B';
		return dA <= r ? 'A' : null;
	};
	const clampIn = (x: number, y: number): [number, number] => {
		const m = p.radius + 0.06;
		return [Math.min(map.width - m, Math.max(m, x)), Math.min(map.height - m, Math.max(m, y))];
	};

	// Touch: let the page scroll in Move mode unless the finger lands on A or B.
	const onTouchStart = (e: TouchEvent) => {
		const t = e.touches[0];
		if (tool === 'move' && t && handleAt(...arena.toWorld(t.clientX, t.clientY))) e.preventDefault();
	};
	canvas.addEventListener('touchstart', onTouchStart, { passive: false });

	function eraseAt(x: number, y: number): void {
		const tmp = new Float64Array(3);
		const kept = map.walls.filter((w) => {
			closestPoint(x, y, Float64Array.from(w), 0, tmp);
			return tmp[2] > ERASER + w[4];
		});
		if (kept.length !== map.walls.length) setWalls(kept, false);
	}

	canvas.addEventListener('pointerdown', (e) => {
		if (!ready || (e.pointerType === 'mouse' && e.button !== 0)) return;
		const [x, y] = arena.toWorld(e.clientX, e.clientY);
		const h = handleAt(x, y);
		if (h) drag = h;
		else if (tool === 'draw') {
			if (map.walls.length >= MAX_WALLS) {
				noteEl.textContent = 'Wall limit reached: erase or clear some walls to draw more.';
				noteEl.hidden = false;
				return;
			}
			stroke = [[x, y]];
			strokeBase = map.walls;
		} else if (tool === 'erase') {
			erasing = true;
			eraser = { x, y, r: ERASER };
			eraseAt(x, y);
		} else return;
		pointerId = e.pointerId;
		canvas.setPointerCapture(e.pointerId);
		e.preventDefault();
		dirty = true;
		kick();
	});

	canvas.addEventListener('pointermove', (e) => {
		const [x, y] = arena.toWorld(e.clientX, e.clientY);
		if (e.pointerId !== pointerId) {
			// hover feedback (mouse only)
			if (e.pointerType !== 'mouse') return;
			const h = handleAt(x, y);
			canvas.classList.toggle('wm-grab', !!h);
			if (tool === 'erase') eraser = { x, y, r: ERASER };
			if (h !== hoverHandle || tool === 'erase') {
				hoverHandle = h;
				dirty = true;
				kick();
			}
			return;
		}
		if (drag) {
			const [cx, cy] = clampIn(x, y);
			map = drag === 'A' ? { ...map, start: [cx, cy, map.start[2]] } : { ...map, goal: [cx, cy] };
			ep.map = map;
			if (drag === 'A') {
				ep.pose = { x: cx, y: cy, th: map.start[2] };
				prev = { ...ep.pose };
				trace = [cx, cy];
			}
		} else if (stroke) {
			const last = stroke[stroke.length - 1];
			if (Math.hypot(x - last[0], y - last[1]) < 0.06) return;
			stroke.push([x, y]);
			// walls exist while drawing (Wheely can bump into them); simplified on release
			const seg = polylineCapsules(stroke.slice(-2), BRUSH);
			const rows = [...map.walls, Array.from(seg)];
			setWalls(rows, false);
		} else if (erasing) {
			eraser = { x, y, r: ERASER };
			eraseAt(x, y);
		}
		dirty = true;
		kick();
	});

	const endPointer = (e: PointerEvent) => {
		if (e.pointerId !== pointerId) return;
		pointerId = -1;
		if (drag) {
			const d = drag;
			drag = null;
			if (d === 'A') {
				const [ax, ay] = map.start;
				map = { ...map, start: [ax, ay, Math.atan2(map.goal[1] - ay, map.goal[0] - ax)] }; // face B
			}
			setMap(map);
			if (started) togglePlay(true);
		} else if (stroke) {
			const pts = rdp(stroke, RDP_EPS);
			stroke = null;
			if (hint !== 'done') setHint('done');
			const caps = polylineCapsules(pts, BRUSH);
			const rows: number[][] = [...strokeBase];
			for (let k = 0; k < caps.length; k += 5) rows.push(Array.from(caps.subarray(k, k + 5)));
			if (rows.length > MAX_WALLS) {
				noteEl.textContent = 'Wall limit reached: erase or clear some walls to draw more.';
				noteEl.hidden = false;
				setWalls(strokeBase, true);
			} else {
				noteEl.hidden = true;
				setWalls(rows, true);
			}
		} else if (erasing) {
			erasing = false;
			if (e.pointerType !== 'mouse') eraser = undefined;
			noteEl.hidden = true;
			setWalls(map.walls, true);
		}
		dirty = true;
		kick();
	};
	canvas.addEventListener('pointerup', endPointer);
	canvas.addEventListener('pointercancel', endPointer);
	canvas.addEventListener('pointerleave', () => {
		if (pointerId !== -1) return;
		hoverHandle = null;
		if (tool === 'erase') eraser = undefined;
		dirty = true;
		kick();
	});

	// ------------------------------------------------------------------ layout, theme, visibility
	function syncPanel(): void {
		const narrow = root.clientWidth < PORTRAIT_BELOW;
		brain.compact = narrow && !expanded;
		panel.dataset.compact = String(brain.compact);
		$('[data-act=enlarge]').textContent = expanded ? 'show less' : 'tap to enlarge';
		brain.layout();
	}
	brainCanvas.addEventListener('click', () => {
		if (brain.compact) {
			expanded = true;
			syncPanel();
		}
	});

	const ro = new ResizeObserver(() => {
		const w = root.clientWidth;
		if (!w) return;
		const wantPortrait = w < PORTRAIT_BELOW;
		if (wantPortrait !== portrait) {
			portrait = wantPortrait;
			setMap(transpose(map));
		} else arena.resize();
		syncPanel();
		dirty = true;
		kick();
	});
	ro.observe(root);

	const io = new IntersectionObserver(([entry]) => {
		visible = entry.isIntersecting;
		lastT = 0;
		kick();
	});
	io.observe(root);

	const onVisibility = () => {
		pageVisible = !document.hidden;
		lastT = 0;
		kick();
	};
	document.addEventListener('visibilitychange', onVisibility);

	const stopTheme = watchTheme(() =>
		requestAnimationFrame(() => {
			theme = readTheme(root);
			arena.theme = brain.theme = theme;
			brain.draw();
			dirty = true;
			kick();
		}),
	);

	// ------------------------------------------------------------------ start
	setTool(tool);
	setSpeed(1);
	arena.setWorld(map.width, map.height);
	syncPanel();
	syncStatus();
	void setVersion(versionId);

	return {
		destroy() {
			cancelAnimationFrame(raf);
			raf = 0;
			loadToken++;
			ro.disconnect();
			io.disconnect();
			stopTheme();
			document.removeEventListener('visibilitychange', onVisibility);
			canvas.removeEventListener('touchstart', onTouchStart);
			root.remove();
		},
	};
}
