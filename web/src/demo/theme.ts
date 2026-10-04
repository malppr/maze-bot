// Colours come from the host page's CSS custom properties on the container (re-read on theme change).
export interface Theme {
	fg: string;
	bg: string;
	surface: string;
	border: string;
	muted: string;
	accent: string;
	accentFg: string;
	neg: string; // negative activations / signals
	sans: string; // font families for canvas text
	mono: string;
}

const FALLBACK: Theme = {
	fg: '#1d1c1a',
	bg: '#fbfaf7',
	surface: '#ffffff',
	border: '#e7e3db',
	muted: '#67635c',
	accent: '#c94436',
	accentFg: '#ffffff',
	neg: '#2862cf',
	sans: 'system-ui, sans-serif',
	mono: 'ui-monospace, monospace',
};

export function readTheme(el: HTMLElement): Theme {
	const cs = getComputedStyle(el);
	const v = (name: string, fb: string) => cs.getPropertyValue(name).trim() || fb;
	return {
		fg: v('--fg', FALLBACK.fg),
		bg: v('--bg', FALLBACK.bg),
		surface: v('--surface', FALLBACK.surface),
		border: v('--border', FALLBACK.border),
		muted: v('--muted', FALLBACK.muted),
		accent: v('--accent', FALLBACK.accent),
		accentFg: v('--accent-fg', FALLBACK.accentFg),
		neg: v('--ai', FALLBACK.neg),
		sans: cs.fontFamily || FALLBACK.sans,
		mono: v('--font-mono', FALLBACK.mono),
	};
}

/** Call `cb` when the page theme may have changed (class / data-theme / style on <html>, or the OS scheme). */
export function watchTheme(cb: () => void): () => void {
	const mo = new MutationObserver(cb);
	mo.observe(document.documentElement, { attributes: true, attributeFilter: ['class', 'data-theme', 'style'] });
	const mq = matchMedia('(prefers-color-scheme: dark)');
	mq.addEventListener('change', cb);
	return () => {
		mo.disconnect();
		mq.removeEventListener('change', cb);
	};
}

/** Parse a CSS colour to [r, g, b] (any format the browser understands). */
const probe = typeof document !== 'undefined' ? document.createElement('canvas').getContext('2d') : null;
export function rgb(color: string): [number, number, number] {
	if (!probe) return [0, 0, 0];
	probe.fillStyle = '#000';
	probe.fillStyle = color;
	const s = probe.fillStyle as string;
	if (s.startsWith('#')) return [parseInt(s.slice(1, 3), 16), parseInt(s.slice(3, 5), 16), parseInt(s.slice(5, 7), 16)];
	const m = s.match(/[\d.]+/g) ?? ['0', '0', '0'];
	return [Number(m[0]), Number(m[1]), Number(m[2])];
}

/** Mix two colours: t = 0 gives a, t = 1 gives b. */
export function mix(a: [number, number, number], b: [number, number, number], t: number): string {
	const c = a.map((x, i) => Math.round(x + (b[i] - x) * t));
	return `rgb(${c[0]},${c[1]},${c[2]})`;
}
