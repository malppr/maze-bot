// By-the-Book has no network: show its rule instead, with live inputs, the rule that fired and the wheels.
import type { RuleTrace } from '../policy/heuristic';

const RULES: [RuleTrace['rule'], string][] = [
	['seek', 'Steer towards B'],
	['keep-wall-right', 'Wall close on the right: don’t turn into it'],
	['keep-wall-left', 'Wall close on the left: don’t turn into it'],
	['avoid', 'Wall ahead: slow down, turn to the open side'],
];

export class RuleView {
	readonly el: HTMLElement;
	private bars: Record<string, [HTMLElement, HTMLElement]> = {};
	private rules = new Map<string, HTMLElement>();
	private goal!: HTMLElement;
	private goalText!: HTMLElement;

	constructor() {
		const el = document.createElement('div');
		el.className = 'wm-rule';
		const bar = (key: string, label: string, signed = false) =>
			`<div class="wm-gauge${signed ? ' wm-signed' : ''}" data-k="${key}"><span>${label}</span><i><b></b></i><em></em></div>`;
		el.innerHTML = `
			<div class="wm-rule-col">
				<p class="wm-sub">Sees</p>
				${bar('left', 'left')}${bar('front', 'ahead')}${bar('right', 'right')}
				<div class="wm-goal"><svg viewBox="-12 -12 24 24" aria-hidden="true"><circle r="10"/><path d="M0 -8 L3 -2 L-3 -2 Z"/></svg><span></span></div>
			</div>
			<div class="wm-rule-col wm-rules">
				<p class="wm-sub">Rule in charge</p>
				<ol>${RULES.map(([k, s]) => `<li data-r="${k}">${s}</li>`).join('')}</ol>
			</div>
			<div class="wm-rule-col">
				<p class="wm-sub">Wheels</p>
				${bar('ul', 'left', true)}${bar('ur', 'right', true)}
			</div>`;
		el.querySelectorAll<HTMLElement>('.wm-gauge').forEach((g) => (this.bars[g.dataset.k!] = [g.querySelector('b')!, g.querySelector('em')!]));
		el.querySelectorAll<HTMLElement>('li').forEach((li) => this.rules.set(li.dataset.r!, li));
		this.goal = el.querySelector('.wm-goal path')!;
		this.goalText = el.querySelector('.wm-goal span')!;
		this.el = el;
	}

	update(t: RuleTrace, action: [number, number]): void {
		const dist = (k: string, v: number) => {
			const [b, em] = this.bars[k];
			b.style.width = `${(Math.min(v, 3) / 3) * 100}%`;
			em.textContent = v >= 3 ? 'clear' : v.toFixed(1);
		};
		dist('left', t.left);
		dist('front', t.front);
		dist('right', t.right);
		const wheel = (k: string, v: number) => {
			const [b, em] = this.bars[k];
			b.style.width = `${Math.abs(v) * 50}%`;
			b.style.marginLeft = v >= 0 ? '50%' : `${50 - Math.abs(v) * 50}%`;
			em.textContent = (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(2);
		};
		wheel('ul', action[0]);
		wheel('ur', action[1]);
		const deg = (t.bearing * 180) / Math.PI;
		this.goal.setAttribute('transform', `rotate(${deg.toFixed(1)})`);
		const a = Math.abs(deg);
		this.goalText.textContent = a < 5 ? 'B straight ahead' : `B ${a.toFixed(0)}° ${deg > 0 ? 'right' : 'left'}`;
		for (const [k, li] of this.rules) li.classList.toggle('wm-on', k === t.rule);
	}
}
