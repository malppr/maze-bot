// Demo CSS, injected once per document. Everything is scoped under .wm and themed by the host's CSS vars.
const CSS = `
.wm{--wm-fg:var(--fg,#1d1c1a);--wm-bg:var(--bg,#fbfaf7);--wm-surface:var(--surface,#fff);--wm-border:var(--border,#e7e3db);
--wm-muted:var(--muted,#67635c);--wm-accent:var(--accent,#c94436);--wm-accent-fg:var(--accent-fg,#fff);--wm-neg:var(--ai,#2862cf);
container-type:inline-size;display:grid;gap:12px;color:var(--wm-fg);font:inherit}
.wm *{box-sizing:border-box}
.wm [hidden]{display:none!important}
.wm button,.wm select{font:inherit;color:inherit}
.wm button{cursor:pointer}
.wm :focus-visible{outline:2px solid var(--wm-accent);outline-offset:2px}
.wm-versions{display:grid;gap:6px}
.wm-chips{display:flex;flex-wrap:wrap;gap:8px}
.wm-chip{padding:6px 14px;border:1px solid var(--wm-border);border-radius:999px;background:var(--wm-surface);font-size:14px;font-weight:500}
.wm-chip[aria-checked=true]{border-color:var(--wm-fg);background:var(--wm-fg);color:var(--wm-bg)}
.wm-blurb{margin:0;font-size:14px;color:var(--wm-muted)}
.wm-stage{position:relative;overflow:hidden;border:1px solid var(--wm-border);border-radius:var(--radius,12px);background:var(--wm-surface)}
.wm-arena{display:block;width:100%;touch-action:pan-y;user-select:none;-webkit-user-select:none}
.wm-stage[data-tool=draw] .wm-arena,.wm-stage[data-tool=erase] .wm-arena{touch-action:none;cursor:crosshair}
.wm-stage[data-tool=erase] .wm-arena{cursor:cell}
.wm-arena.wm-grab{cursor:grab}
.wm-status{position:absolute;top:10px;left:10px;padding:2px 10px;border:1px solid var(--wm-border);border-radius:999px;
background:color-mix(in srgb,var(--wm-surface) 88%,transparent);font:500 12px var(--font-mono,ui-monospace,monospace);color:var(--wm-muted);pointer-events:none}
.wm-warn{position:absolute;top:10px;right:10px;left:auto;max-width:calc(100% - 20px);padding:4px 12px;border-radius:999px;
background:var(--wm-accent);color:var(--wm-accent-fg);font-size:13px;font-weight:600;pointer-events:none}
.wm-result{position:absolute;left:50%;bottom:14px;transform:translateX(-50%);display:flex;align-items:center;gap:10px;flex-wrap:wrap;justify-content:center;
padding:8px 10px 8px 16px;border:1px solid var(--wm-border);border-radius:999px;background:var(--wm-surface);box-shadow:var(--shadow,0 10px 30px rgb(0 0 0/.1));
font-weight:600;white-space:nowrap}
.wm-result button{padding:5px 12px;border:1px solid var(--wm-border);border-radius:999px;background:var(--wm-bg);font-size:13px;font-weight:500}
.wm-result button.wm-primary{border-color:var(--wm-fg);background:var(--wm-fg);color:var(--wm-bg)}
.wm-note{margin:0;font-size:13px;color:var(--wm-muted)}
.wm-toolbar{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:8px 12px}
.wm-group{display:flex;align-items:center;gap:6px}
.wm select{padding:6px 10px;border:1px solid var(--wm-border);border-radius:8px;background:var(--wm-surface);font-size:14px;font-weight:500}
.wm-seg{display:inline-flex;padding:2px;border:1px solid var(--wm-border);border-radius:9px;background:var(--wm-surface)}
.wm-seg button{padding:5px 11px;border:0;border-radius:7px;background:none;color:var(--wm-muted);font-size:13px;font-weight:500}
.wm-seg button[aria-checked=true]{background:var(--wm-bg);color:var(--wm-fg);box-shadow:inset 0 0 0 1px var(--wm-border)}
.wm-icon{display:grid;place-items:center;width:34px;height:34px;padding:0;border:1px solid var(--wm-border);border-radius:50%;background:var(--wm-surface)}
.wm-icon.wm-main{width:40px;height:40px;border-color:var(--wm-fg);background:var(--wm-fg);color:var(--wm-bg)}
.wm-icon svg{width:18px;height:18px;fill:none;stroke:currentColor;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
.wm-panel{padding:14px 18px;border:1px solid var(--wm-border);border-radius:var(--radius,12px);background:var(--wm-surface)}
.wm-panel-head{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:6px}
.wm-title{margin:0;font:500 12px var(--font-mono,ui-monospace,monospace);text-transform:uppercase;letter-spacing:.06em;color:var(--wm-muted)}
.wm-legend{display:flex;gap:14px;font-size:12px;color:var(--wm-muted)}
.wm-legend i{display:inline-block;width:9px;height:9px;margin-right:5px;border-radius:50%}
.wm-brain{display:block;width:100%;max-width:760px;margin-inline:auto}
.wm-foot{display:none;gap:14px;align-items:center;font-size:13px;color:var(--wm-muted)}
.wm-foot b{font-family:var(--font-mono,ui-monospace,monospace);color:var(--wm-fg)}
.wm-foot button{margin-left:auto;padding:0;border:0;background:none;color:var(--wm-muted);font-size:13px;text-decoration:underline;text-underline-offset:3px}
.wm-rule{display:grid;grid-template-columns:1fr 1.3fr 1fr;gap:20px;padding:4px 0}
.wm-sub{margin:0 0 6px;font-size:12px;color:var(--wm-muted)}
.wm-gauge{display:grid;grid-template-columns:44px 1fr 44px;align-items:center;gap:8px;margin:5px 0;font-size:13px}
.wm-gauge i{position:relative;height:8px;border-radius:4px;background:var(--wm-bg);box-shadow:inset 0 0 0 1px var(--wm-border);overflow:hidden}
.wm-gauge b{display:block;height:100%;border-radius:4px;background:var(--wm-muted)}
.wm-gauge.wm-signed b{background:var(--wm-accent)}
.wm-gauge em{font:normal 12px var(--font-mono,ui-monospace,monospace);text-align:right;color:var(--wm-muted)}
.wm-goal{display:flex;align-items:center;gap:8px;margin-top:8px;font-size:13px}
.wm-goal svg{width:24px;height:24px}
.wm-goal circle{fill:none;stroke:var(--wm-border);stroke-width:1.5}
.wm-goal path{fill:var(--wm-accent)}
.wm-rules ol{margin:0;padding-left:20px;font-size:13px;color:var(--wm-muted)}
.wm-rules li{padding:3px 6px;border-radius:6px}
.wm-rules li.wm-on{background:color-mix(in srgb,var(--wm-accent) 14%,transparent);color:var(--wm-fg);font-weight:600}
.wm-sr{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%);white-space:nowrap}
@container (max-width: 600px){
.wm-toolbar{justify-content:flex-start}
.wm-play{width:100%;justify-content:space-between}
.wm-panel{padding:10px 12px}
.wm-legend{display:none}
.wm-foot{display:flex}
.wm-rule{grid-template-columns:1fr;gap:10px}
.wm-result{bottom:10px;width:max-content;max-width:calc(100% - 20px);padding:8px 12px;border-radius:16px;font-size:14px;white-space:normal}
.wm-result-text{flex-basis:100%;text-align:center}
.wm-warn{top:40px;left:10px;right:auto}
}
`;

export function injectStyles(doc: Document = document): void {
	if (doc.getElementById('wm-styles')) return;
	const s = doc.createElement('style');
	s.id = 'wm-styles';
	s.textContent = CSS;
	doc.head.appendChild(s);
}
