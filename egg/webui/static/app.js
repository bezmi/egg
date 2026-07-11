// Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
//
// PolyForm Noncommercial License 2.0.0-pre.2
// https://github.com/bezmi/egg/blob/main/LICENSE.md
// Free to use and redistribute for personal and noncommercial purposes.
// See the license for details.
// For commercial licensing, contact s.imran@tuta.io


// Reload without a flash: if there is a saved script to restore, hide the
// server-rendered default grid before first paint. Cleared once the restored
// script re-renders (applyView) or, failing that, a fallback timeout in the
// DOMContentLoaded restore below. Runs in <head>, before the body parses.
try {
  if (localStorage.getItem('egg-webui-code'))
    document.documentElement.classList.add('restoring');
} catch (e) {}

// Catppuccin flavor: applied before first paint (this classic script runs
// in <head>), defaulting from the OS color-scheme preference.
const THEMES = ['mocha', 'macchiato', 'frappe', 'latte'];
let eggTheme = localStorage.getItem('egg-webui-theme');
if (!THEMES.includes(eggTheme))
  eggTheme = window.matchMedia('(prefers-color-scheme: dark)').matches
    ? 'mocha' : 'latte';
document.documentElement.dataset.theme = eggTheme;
// The favicon is the same egg mark as the header logo, recolored to the active
// flavor's yellow. A favicon SVG can't read the page's data-theme, so we bake
// the color in and regenerate the data-URI whenever the flavor changes.
function eggFaviconSvg(color) {
  const d = 'M50 7C65 7 83 35 83 59 83 81 68 93 50 93 32 93 17 81 17 59 17 35 35 7 50 7Z';
  return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" fill="none" stroke="' +
    color + '" stroke-width="7" stroke-linecap="round" stroke-linejoin="round">' +
    '<clipPath id="e"><path d="' + d + '"/></clipPath><path d="' + d + '"/>' +
    '<g clip-path="url(#e)" stroke-width="6"><path d="M40 15V95M60 15V95M10 48H90M10 66H90"/></g></svg>';
}
function updateFavicon() {
  const c = getComputedStyle(document.documentElement)
    .getPropertyValue('--ctp-yellow').trim() || '#f9e2af';
  let link = document.getElementById('favicon');
  if (!link) {
    link = document.createElement('link');
    link.id = 'favicon'; link.rel = 'icon'; link.type = 'image/svg+xml';
    document.head.appendChild(link);
  }
  link.href = 'data:image/svg+xml,' + encodeURIComponent(eggFaviconSvg(c));
}
function setTheme(name) {
  eggTheme = name;
  document.documentElement.dataset.theme = name;
  localStorage.setItem('egg-webui-theme', name);
  // The editor and header logo re-theme automatically off the live --ctp-*
  // variables; only the baked-in favicon needs a manual refresh.
  updateFavicon();
  window.dispatchEvent(new Event('egg-theme'));
}
updateFavicon();  // tint the initial favicon to the active flavor
window.addEventListener('DOMContentLoaded', () => {
  const sel = document.getElementById('theme-select');
  if (!sel) return;
  sel.value = eggTheme;
  sel.addEventListener('change', () => setTheme(sel.value));
});

document.addEventListener('keydown', (e) => {
  // Tab-to-indent only for the plain-textarea fallback; when prism-code-editor
  // is active its editorCommands extension owns Tab (and its own textarea).
  if (e.key === 'Tab' && !window.eggEditor && e.target.matches('.editor textarea')) {
    e.preventDefault();
    const t = e.target, s = t.selectionStart;
    t.setRangeText('    ', s, t.selectionEnd, 'end');
    t.dispatchEvent(new Event('input', {bubbles: true}));
  }
});

let savedVB = null;
// The grid SVG, scoped to the canvas: #view also holds chrome icons (e.g. the
// reset button), so an unscoped '#view svg' would grab whichever comes first.
const svgEl = () => document.querySelector('#view .canvas svg');
// Point markers (node/corner circles, fixed-point squares) are authored in
// the fixed 1200x900 render space, so viewBox zoom balloons them like any
// filled shape. Counter-scale each by the current zoom (viewBox width / fit
// width), about its own centre, so it holds a constant on-screen size — the
// dot equivalent of the lines' non-scaling-stroke. Base sizes are captured
// once per element (attributes are authored fresh after every render) and
// reused, so repeated calls never compound.
function scaleMarkers(el) {
  el = el || svgEl();
  if (!el || !el.dataset.fit) return;
  const vb = el.getAttribute('viewBox');
  if (!vb) return;
  const k = 1.5 * (+vb.split(' ')[2]) / (+el.dataset.fit.split(' ')[2]);
  // the selection band is a viewBox-space rect, not a constant-size marker —
  // rescaling it makes it drift off the cursor
  el.querySelectorAll('circle, rect:not(.ed-band)').forEach((m) => {
    const d = m.dataset;
    if (m.tagName === 'circle') {
      if (d.br === undefined) d.br = m.getAttribute('r');
      m.setAttribute('r', (+d.br * k).toFixed(2));
    } else {
      if (d.bw === undefined) {
        d.bx = m.getAttribute('x'); d.by = m.getAttribute('y');
        d.bw = m.getAttribute('width'); d.bh = m.getAttribute('height');
      }
      const w = +d.bw * k, h = +d.bh * k;
      const cx = +d.bx + +d.bw / 2, cy = +d.by + +d.bh / 2;
      m.setAttribute('width', w.toFixed(2));
      m.setAttribute('height', h.toFixed(2));
      m.setAttribute('x', (cx - w / 2).toFixed(2));
      m.setAttribute('y', (cy - h / 2).toFixed(2));
    }
  });
}
function setVB(el, x, y, w, h) {
  const prevW = savedVB ? +savedVB.split(' ')[2] : null;
  el.setAttribute('viewBox', `${x} ${y} ${w} ${h}`);
  savedVB = el.getAttribute('viewBox');
  // pan leaves width unchanged; only a zoom needs the markers rescaled
  if (prevW === null || Math.abs(w - prevW) > 1e-6) scaleMarkers(el);
}
document.addEventListener('wheel', (e) => {
  const el = svgEl();
  if (!el || !e.target.closest('.canvas')) return;
  e.preventDefault();
  const [x, y, w, h] = el.getAttribute('viewBox').split(' ').map(Number);
  const r = el.getBoundingClientRect();
  const px = x + (e.clientX - r.left) / r.width * w;
  const py = y + (e.clientY - r.top) / r.height * h;
  const k = e.deltaY > 0 ? 1.2 : 1 / 1.2;
  setVB(el, px - (px - x) * k, py - (py - y) * k, w * k, h * k);
}, {passive: false});
// Pointer gestures on the canvas: one pointer pans, two pinch-zoom.
const ptrs = new Map();
let panStart = null, panMoved = false;
document.addEventListener('pointerdown', (e) => {
  if (!e.target.closest('.canvas')) return;
  e.target.setPointerCapture(e.pointerId);
  ptrs.set(e.pointerId, {x: e.clientX, y: e.clientY});
  panStart = {x: e.clientX, y: e.clientY};
  panMoved = false;
  if (eggEd) eggEdPointerDown(e);  // may grab a node/edge/band -> suppresses pan
});
document.addEventListener('pointermove', (e) => {
  if (eggEdDrag) { eggEdPointerMove(e); return; }  // node move / rubber-band
  const el = svgEl();
  if (!el || !ptrs.has(e.pointerId)) return;
  const prev = new Map(ptrs);
  ptrs.set(e.pointerId, {x: e.clientX, y: e.clientY});
  if (panStart && Math.hypot(e.clientX - panStart.x, e.clientY - panStart.y) > 5)
    panMoved = true;
  const ids = [...ptrs.keys()];
  const [x, y, w, h] = el.getAttribute('viewBox').split(' ').map(Number);
  const r = el.getBoundingClientRect();
  if (ids.length === 1) {
    const p0 = prev.get(e.pointerId), p1 = ptrs.get(e.pointerId);
    setVB(el, x - (p1.x - p0.x) / r.width * w,
              y - (p1.y - p0.y) / r.height * h, w, h);
  } else if (ids.length === 2) {
    const a0 = prev.get(ids[0]), b0 = prev.get(ids[1]);
    const a1 = ptrs.get(ids[0]), b1 = ptrs.get(ids[1]);
    const d1 = Math.hypot(b1.x - a1.x, b1.y - a1.y);
    if (d1 < 1) return;
    const k = Math.max(0.2, Math.min(5, Math.hypot(b0.x - a0.x, b0.y - a0.y) / d1));
    // pan by the midpoint delta, then zoom about the new midpoint
    const m0x = (a0.x + b0.x) / 2, m0y = (a0.y + b0.y) / 2;
    const m1x = (a1.x + b1.x) / 2, m1y = (a1.y + b1.y) / 2;
    const nx = x - (m1x - m0x) / r.width * w;
    const ny = y - (m1y - m0y) / r.height * h;
    const px = nx + (m1x - r.left) / r.width * w;
    const py = ny + (m1y - r.top) / r.height * h;
    setVB(el, px - (px - nx) * k, py - (py - ny) * k, w * k, h * k);
  }
});
const endPtr = (e) => { ptrs.delete(e.pointerId); if (eggEdDrag) eggEdPointerUp(); };
document.addEventListener('pointerup', endPtr);
document.addEventListener('pointercancel', endPtr);

// World-coordinate readout: pointer position in the geometry frame, via
// the inverse-transform data attributes the renderer emits. Touch has no
// hover, so pointerdown (tap / drag start) updates it too, and it stays
// visible after the finger lifts.
function updateCoords(e) {
  const el = svgEl(), out = document.getElementById('coords');
  if (!out) return;
  const d = el ? el.dataset : {};
  const inCanvas = e.target instanceof Element && e.target.closest('.canvas');
  if (!el || !d.sx || !inCanvas) {
    // hide only for hover-out; a tap elsewhere keeps the last value
    if (e.type === 'pointermove') out.style.display = 'none';
    return;
  }
  const [x, y, w, h] = el.getAttribute('viewBox').split(' ').map(Number);
  const r = el.getBoundingClientRect();
  const vx = x + (e.clientX - r.left) / r.width * w;
  const vy = y + (e.clientY - r.top) / r.height * h;
  const wx = +d.lox + (vx - +d.ox) / +d.sx;
  const wy = +d.hiy - (vy - +d.oy) / +d.sx;
  out.textContent = wx.toPrecision(5) + ', ' + wy.toPrecision(5);
  positionCoords();
  out.style.display = 'block';
}
document.addEventListener('pointermove', updateCoords);
document.addEventListener('pointerdown', updateCoords);

// Editor zoom: Ctrl+wheel (also trackpad pinch), touch pinch, and
// Ctrl +/-/0. Adjusts a persisted font size, not the page zoom.
let edFont = parseFloat(localStorage.getItem('egg-webui-edfont')) || 13;
function setEdFont(px) {
  edFont = Math.min(28, Math.max(8, px));
  document.documentElement.style.setProperty('--egg-edfont', edFont.toFixed(1) + 'px');
  localStorage.setItem('egg-webui-edfont', edFont.toFixed(1));
  // prism-code-editor reflows from CSS (font-size on .prism-code-editor), so
  // a font-zoom needs no explicit remeasure.
}
window.addEventListener('DOMContentLoaded', () => setEdFont(edFont));
document.addEventListener('wheel', (e) => {
  if (!e.ctrlKey || !e.target.closest('.editor')) return;
  e.preventDefault();
  setEdFont(edFont * (e.deltaY > 0 ? 1 / 1.08 : 1.08));
}, {passive: false});
document.addEventListener('keydown', (e) => {
  if (!(e.ctrlKey || e.metaKey) || !e.target.closest('.editor')) return;
  if (e.key === '=' || e.key === '+') { e.preventDefault(); setEdFont(edFont + 1); }
  else if (e.key === '-') { e.preventDefault(); setEdFont(edFont - 1); }
  else if (e.key === '0') { e.preventDefault(); setEdFont(13); }
});
const edPtrs = new Map();
let edPinch = null;
document.addEventListener('pointerdown', (e) => {
  if (!e.target.closest('.editor')) return;
  edPtrs.set(e.pointerId, {x: e.clientX, y: e.clientY});
  if (edPtrs.size === 2) {
    const [a, b] = [...edPtrs.values()];
    edPinch = {d: Math.hypot(b.x - a.x, b.y - a.y), font: edFont};
  }
});
document.addEventListener('pointermove', (e) => {
  if (!edPtrs.has(e.pointerId)) return;
  edPtrs.set(e.pointerId, {x: e.clientX, y: e.clientY});
  if (edPinch && edPtrs.size === 2) {
    const [a, b] = [...edPtrs.values()];
    const d = Math.hypot(b.x - a.x, b.y - a.y);
    if (d > 1) setEdFont(edPinch.font * d / edPinch.d);
  }
});
const edEnd = (e) => {
  edPtrs.delete(e.pointerId);
  if (edPtrs.size < 2) edPinch = null;
};
document.addEventListener('pointerup', edEnd);
document.addEventListener('pointercancel', edEnd);

// Distance^2 from point (px,py) to segment a-b, all in viewBox coords.
function segDist2(px, py, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const L2 = dx * dx + dy * dy;
  let t = L2 ? ((px - a[0]) * dx + (py - a[1]) * dy) / L2 : 0;
  t = Math.max(0, Math.min(1, t));
  const qx = a[0] + t * dx - px, qy = a[1] + t * dy - py;
  return qx * qx + qy * qy;
}

// Nearest selectable element within maxD (viewBox units) of the click.
// Block polygons are excluded — their filled area is a direct-hit target,
// and letting them win here would shadow the thin things ON their edges.
function pickNear(svg, vx, vy, maxD) {
  let best = null, bestD = maxD * maxD;
  svg.querySelectorAll('.topo-sel:not(polygon), .curve, .pt, .ctrl-pt')
     .forEach((el) => {
    if (!el.getClientRects().length) return;  // in a hidden layer
    let d2 = Infinity;
    if (el.tagName === 'circle') {
      const dx = vx - +el.getAttribute('cx'), dy = vy - +el.getAttribute('cy');
      d2 = dx * dx + dy * dy;
    } else if (el.tagName === 'rect') {
      const cx = +el.getAttribute('x') + el.getAttribute('width') / 2;
      const cy = +el.getAttribute('y') + el.getAttribute('height') / 2;
      d2 = (vx - cx) ** 2 + (vy - cy) ** 2;
    } else {
      const pts = (el.getAttribute('points') || '')
        .split(' ').map((p) => p.split(',').map(Number));
      for (let i = 0; i + 1 < pts.length; i++)
        d2 = Math.min(d2, segDist2(vx, vy, pts[i], pts[i + 1]));
    }
    if (d2 < bestD) { bestD = d2; best = el; }
  });
  return best;
}

// Click-to-select in both views: topology elements highlight themselves
// plus the geometry (data-eid curves) that will constrain them; curves
// and points are selectable too. The selected element's name shows in
// the bottom-center readout. Taps snap to the nearest target within a
// finger-sized radius, so thin lines and dots don't demand precision.
document.addEventListener('click', (e) => {
  const svg = svgEl(), sel = document.getElementById('selinfo');
  if (!svg || !e.target.closest('.canvas') || panMoved) return;
  // edit view: draw mode draws on click; select mode handled on pointerdown
  if (eggEd) { eggEditClick(e); return; }
  svg.querySelectorAll('.hl').forEach((x) => x.classList.remove('hl'));
  if (sel) sel.style.display = 'none';
  const [x, y, w, h] = svg.getAttribute('viewBox').split(' ').map(Number);
  const r = svg.getBoundingClientRect();
  const vx = x + (e.clientX - r.left) / r.width * w;
  const vy = y + (e.clientY - r.top) / r.height * h;
  const coarse = window.matchMedia('(pointer: coarse)').matches;
  const maxD = (coarse ? 24 : 12) / r.width * w;
  const t = pickNear(svg, vx, vy, maxD)
            || e.target.closest('.topo-sel, .curve, .pt, .ctrl-pt');
  if (!t) {
    // grid view: a block's fill is a direct-hit target for its cell
    // counts; anywhere else reports the whole grid's total.
    const blk = e.target.closest('.grid-block');
    const text = blk ? blk.dataset.label : svg.dataset.cells;
    if (!sel || !text) return;
    if (blk) blk.classList.add('hl');
    sel.textContent = text;
    positionCoords();
    sel.style.display = 'block';
    return;
  }
  t.classList.add('hl');
  (t.dataset.eids || '').split(',').filter(Boolean).forEach((eid) => {
    svg.querySelectorAll('[data-eid="' + eid + '"]')
       .forEach((x) => x.classList.add('hl'));
  });
  const title = t.querySelector('title');
  if (sel && title) {
    sel.textContent = title.textContent;
    positionCoords();
    sel.style.display = 'block';
  }
});
document.addEventListener('click', (e) => {
  const el = svgEl();
  if (el && e.target.closest('#fit')) {
    el.setAttribute('viewBox', el.dataset.fit);
    savedVB = null;
    scaleMarkers(el);
  }
});

// Keep the coordinate readout and axis gizmo pinned to the CANVAS
// bottom corners, so the chart / stdout / error panels below push
// them up rather than under them.
function positionCoords() {
  const el = svgEl(), out = document.getElementById('coords');
  if (!el || !out) return;
  const vr = out.parentElement.getBoundingClientRect();
  const cr = el.closest('.canvas').getBoundingClientRect();
  const b = (vr.bottom - cr.bottom + 8) + 'px';
  out.style.bottom = b;
  for (const id of ['axes', 'selinfo']) {
    const el2 = document.getElementById(id);
    if (el2) el2.style.bottom = b;
  }
}

// run parameters are only actionable in grid view (where the run happens);
// topology/edit views hide them to keep the canvas uncluttered
function eggSyncParamsVis() {
  const par = document.getElementById('params');
  const vm = document.getElementById('viewmode');
  if (par) par.style.display = vm && vm.value !== 'grid' ? 'none' : '';
}
// Browsers restore a <select>'s value across a reload / bfcache, so the view
// dropdown can show a stale view while the server actually rendered grid. Snap
// it back to the option the server marked selected (grid on a fresh load) and
// re-sync anything keyed off it. pageshow fires after that restoration, and on
// bfcache restore, so it is the right hook.
function eggResetViewMode() {
  const s = document.getElementById('viewmode');
  if (!s) return;
  const def = [...s.options].find((o) => o.defaultSelected) || s.options[0];
  if (def && s.value !== def.value) s.value = def.value;
  eggSyncParamsVis();
}
window.addEventListener('pageshow', eggResetViewMode);
function applyView() {
  // The restore render has landed — reveal the canvas (hidden on reload).
  document.documentElement.classList.remove('restoring');
  // keep the live run log (egg.webui_print) pinned to its newest line: each
  // OOB frame replaces #runlog, which would otherwise reset its scroll to top
  const rlog = document.querySelector('#runlog pre');
  if (rlog) rlog.scrollTop = rlog.scrollHeight;
  const el = svgEl();
  if (!el) return;
  positionCoords();
  // re-renders drop the selection; hide its stale name too
  const sel = document.getElementById('selinfo');
  if (sel && !el.querySelector('.hl')) sel.style.display = 'none';
  if (savedVB) el.setAttribute('viewBox', savedVB);
  scaleMarkers(el);  // fresh markers are authored-size; match the current zoom
  document.querySelectorAll('.layer-toggle').forEach((cb) => {
    el.querySelectorAll('.layer-' + cb.dataset.layer).forEach((g) => {
      g.style.display = cb.checked ? '' : 'none';
    });
  });
  const vd = document.getElementById('view');
  document.querySelectorAll('.scene-toggle').forEach((cb) => {
    if (vd) vd.classList.toggle('no-' + cb.dataset.toggle, !cb.checked);
  });
  const par = document.getElementById('params');
  if (par && par.tagName === 'DETAILS')
    par.open = localStorage.getItem('egg-params-open') === '1';
  eggSyncParamsVis();
  // a watched file is never modified by the UI — that includes the panel
  const watching = document.getElementById('watch-toggle');
  document.querySelectorAll('.param-input').forEach((i) => {
    i.disabled = !!(watching && watching.checked);
  });
  eggEditInit();  // edit view: (re)build the wireframe overlay + draw tools
}
// Single entry point for programmatic code changes (examples, restore,
// param-panel rewrites). Routes through the editor when it took over (which
// mirrors back into the textarea and fires 'input'), else the textarea
// directly; both paths end in an 'input' event that drives the HTMX render
// trigger and localStorage persistence. The cursor survives the replace — a
// param edit must not fling the editor to the top (eggEditorApi.setValue does
// a minimal-diff edit).
window.eggSetCode = (code) => {
  if (window.eggEditorApi) {
    window.eggEditorApi.setValue(code);
  } else {
    const t = document.querySelector('.editor textarea');
    const cur = Math.min(t.selectionStart || 0, code.length);
    const top = t.scrollTop;
    t.value = code;
    t.setSelectionRange(cur, cur);
    t.scrollTop = top;
    t.dispatchEvent(new Event('input', {bubbles: true}));
  }
};

document.addEventListener('change', (e) => {
  if (e.target.matches('.layer-toggle, .scene-toggle')) applyView();
});

// Parameter panel: an input change rewrites that value's source span in
// the code buffer server-side; the returned script goes back through
// eggSetCode, so the editor, persistence, and the 500ms re-render all
// see one consistent code state.
document.addEventListener('change', async (e) => {
  const inp = e.target;
  if (!inp.matches('.param-input')) return;
  const t = document.querySelector('.editor textarea');
  if (!t) return;
  const val = inp.type === 'checkbox' ? (inp.checked ? 'True' : 'False')
                                      : inp.value;
  const body = new URLSearchParams(
      {code: t.value, name: inp.dataset.param, value: val});
  try {
    const r = await fetch('/api/param', {method: 'POST', body});
    if (!r.ok) { inp.classList.add('param-bad'); return; }
    inp.classList.remove('param-bad');
    window.eggSetCode(await r.text());
  } catch { inp.classList.add('param-bad'); }
});
// Collapsed/open state survives the per-edit re-renders.
document.addEventListener('toggle', (e) => {
  if (e.target.id === 'params')
    localStorage.setItem('egg-params-open', e.target.open ? '1' : '0');
}, true);
// NB: document, not document.body — this classic script executes in
// <head>, where document.body is still null and the property access
// would throw, killing every handler registered below this line.
// htmx events bubble, so listening on document is equivalent.
document.addEventListener('htmx:afterSwap', applyView);
document.addEventListener('htmx:oobAfterSwap', applyView);

// Editor persistence (only when no file was passed on the CLI).
const reveal = () => document.documentElement.classList.remove('restoring');
window.addEventListener('DOMContentLoaded', () => {
  const t = document.querySelector('.editor textarea');
  const saved = t && t.dataset.persist === '1'
      ? localStorage.getItem('egg-webui-code') : null;
  if (saved && saved !== t.value) {
    // Keep the canvas hidden (set in <head>) until the restore render lands;
    // applyView reveals it on the swap, and this is a safety net in case that
    // render never arrives (worker error, etc.).
    window.eggSetCode(saved);
    setTimeout(reveal, 4000);
  } else {
    reveal();  // nothing to restore — show the server-rendered grid now
  }
});
document.addEventListener('input', (e) => {
  if (e.target.matches('.editor textarea') && e.target.dataset.persist === '1')
    localStorage.setItem('egg-webui-code', e.target.value);
});

// Jump the editor to the error line.
document.addEventListener('click', (e) => {
  const chip = e.target.closest('.errline');
  if (!chip) return;
  const line = +chip.dataset.line;
  if (window.eggEditorApi) {
    window.eggEditorApi.gotoLine(line);
    return;
  }
  const t = document.querySelector('.editor textarea');
  const lines = t.value.split('\n');
  const pos = lines.slice(0, line - 1).join('\n').length + (line > 1 ? 1 : 0);
  t.focus();
  t.setSelectionRange(pos, pos + (lines[line - 1] || '').length);
  t.scrollTop = Math.max(0, (line - 4) * 19);
});

// Resizable editor/viewer split via Split.js (sizes persisted). Narrow
// windows stack the panes vertically and the split re-initializes in the
// other direction. When split.js is unavailable the static CSS layout
// (side-by-side, or stacked under the media query) stays in place.
let splitInst = null;
const stackedMQ = window.matchMedia('(max-width: 900px)');
function initSplit() {
  if (!window.Split) return;
  const panes = document.querySelector('.panes');
  if (splitInst) { splitInst.destroy(); splitInst = null; }
  const stacked = stackedMQ.matches;
  panes.classList.toggle('stacked', stacked);
  let sizes = [42, 58];
  try {
    sizes = JSON.parse(localStorage.getItem('egg-webui-split-sizes')) || sizes;
  } catch (err) { /* stale value */ }
  // finger-sized gutter on touch devices, slim one for mouse
  const coarse = window.matchMedia('(pointer: coarse)').matches;
  splitInst = Split(['.editor', '.viewer'], {
    sizes, minSize: 120, gutterSize: coarse ? 20 : 8, snapOffset: 0,
    direction: stacked ? 'vertical' : 'horizontal',
    onDragEnd: (s) => localStorage.setItem('egg-webui-split-sizes', JSON.stringify(s)),
  });
  panes.classList.add('split-active');
}
window.addEventListener('DOMContentLoaded', () => {
  initSplit();
  stackedMQ.addEventListener('change', initSplit);
});

// Downloads: current view as SVG (client-side), current mesh as SU2.
function dl(name, blob) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}
// header dropdown menus (file / view). Checkbox items keep the menu
// open; action buttons and outside clicks close it.
document.addEventListener('click', (e) => {
  const btn = e.target.closest('.menu-btn');
  document.querySelectorAll('.menu').forEach((m) => {
    if (btn && m.contains(btn)) m.classList.toggle('open');
    else if (!m.contains(e.target)
             || e.target.closest('.menu-items button, .menu-items a'))
      m.classList.remove('open');
  });
});

document.addEventListener('click', async (e) => {
  if (e.target.closest('#dl-svg, #file-dl-svg')) {
    const el = svgEl();
    if (el) {
      // styling lives in the page stylesheet — embed it so the export
      // is self-contained; in the standalone file the svg element is
      // :root, so it carries the flavor itself
      const clone = el.cloneNode(true);
      clone.setAttribute('data-theme', eggTheme);
      const st = document.createElementNS('http://www.w3.org/2000/svg', 'style');
      st.textContent = document.querySelector('style').textContent;
      clone.insertBefore(st, clone.firstChild);
      dl('egg-scene.svg',
         new Blob([new XMLSerializer().serializeToString(clone)],
                  {type: 'image/svg+xml'}));
    }
  }
  if (e.target.closest('#dl-su2, #file-dl-su2')) {
    const code = document.querySelector('.editor textarea').value;
    const path = document.getElementById('scriptpath').value;
    const r = await fetch('/export/su2', {method: 'POST', body: new URLSearchParams({code, path})});
    const t = await r.text();
    if (!r.ok) { alert(t); return; }
    dl('mesh.su2', new Blob([t], {type: 'text/plain'}));
  }
});

// --- open/save: normal file workflow against the server's filesystem
// (local single-user tool). State: the open file's path + its last
// saved content, for the dirty dot on the filename chip.
let curFile = null, lastSaved = null, fsMode = 'open', fsDir = null;
const baseOf = (p) => p.slice(p.lastIndexOf('/') + 1);
const dirOf = (p) => p.slice(0, Math.max(1, p.lastIndexOf('/')));
const joinP = (d, n) => d + (d.endsWith('/') ? '' : '/') + n;
const currentCode = () => {
  return window.eggEditorApi
    ? window.eggEditorApi.getValue()
    : document.querySelector('.editor textarea').value;
};
function setFile(path, savedCode) {
  curFile = path; lastSaved = savedCode;
  localStorage.setItem('egg-webui-file', path || '');
  if (path) setScriptPath(path);
  updateChip();
}
// the script's on-disk location: sent with renders/runs so sibling
// imports (from driver import ...) resolve; independent of curFile so
// pasted edits keep resolving against the last opened script's dir
function setScriptPath(path) {
  document.getElementById('scriptpath').value = path || '';
  localStorage.setItem('egg-webui-path', path || '');
}
function updateChip(flash) {
  const c = document.getElementById('filechip');
  if (!c) return;
  if (!curFile) { c.textContent = ''; c.title = ''; return; }
  const dirty = lastSaved === null || currentCode() !== lastSaved;
  c.textContent = baseOf(curFile) + (flash ? ' ✓' : dirty ? ' •' : '');
  c.title = curFile;
}
function fsHide() { document.getElementById('fsmodal').style.display = 'none'; }
async function fsList(dir) {
  const r = await fetch('/api/files?dir=' + encodeURIComponent(dir || ''));
  const j = await r.json();
  if (j.error) { alert(j.error); return; }
  fsDir = j.dir;
  document.getElementById('fs-dir').textContent = j.dir;
  const list = document.getElementById('fs-list');
  list.innerHTML = '';
  const add = (label, cls, fn) => {
    const b = document.createElement('button');
    b.textContent = label; b.className = 'fs-entry ' + cls; b.onclick = fn;
    list.appendChild(b);
  };
  if (j.parent && j.parent !== j.dir) add('..', 'fs-dirent', () => fsList(j.parent));
  j.dirs.forEach((d) => add(d + '/', 'fs-dirent', () => fsList(joinP(j.dir, d))));
  j.files.forEach((f) => add(f, 'fs-fileent', () => fsPick(joinP(j.dir, f))));
}
async function fsShow(mode) {
  fsMode = mode;
  document.getElementById('fsmodal').style.display = 'flex';
  document.querySelector('.fs-saverow').style.display = mode === 'save' ? 'flex' : 'none';
  document.getElementById('fs-title').textContent =
    mode === 'save' ? 'save as' : 'open';
  if (mode === 'save')
    document.getElementById('fs-name').value = curFile ? baseOf(curFile) : 'geometry.py';
  await fsList(fsDir || (curFile ? dirOf(curFile) : ''));
}
async function fsPick(path) {
  if (fsMode === 'save') {  // clicking a file in save mode = take its name
    document.getElementById('fs-name').value = baseOf(path);
    return;
  }
  const r = await fetch('/api/file?path=' + encodeURIComponent(path));
  const j = await r.json();
  if (j.error) { alert(j.error); return; }
  // library-style script (defines a builder, no __egg_webui__ block):
  // watching -> show the block to paste (never touch a watched file);
  // otherwise offer to append it to the file, on confirmation only
  if (j.suggest) {
    if (watchTimer) {
      window.eggSetCode(j.code);
      setFile(j.path, j.code);
      fsHide();
      showSuggestion(j.suggest);
      return;
    }
    if (confirm(
        'this script draws nothing and registers no run in the web UI.\n\n'
        + 'append an __egg_webui__ block (build + egg_webui.run) and save '
        + 'the file?')) {
      window.eggSetCode(j.code.trimEnd() + j.suggest);
      setFile(j.path, null);
      await doSave(j.path);
      fsHide();
      return;
    }
  }
  window.eggSetCode(j.code);
  setFile(j.path, j.code);
  fsHide();
}
function showSuggestion(text) {
  document.getElementById('sug-text').textContent = text.trim() + '\n';
  document.getElementById('sugmodal').style.display = 'flex';
}
async function doSave(path) {
  const code = currentCode();
  const r = await fetch('/api/file/save', {
    method: 'POST', body: new URLSearchParams({path, code}),
  });
  const j = await r.json();
  if (j.error) { alert('save failed: ' + j.error); return; }
  setFile(j.path, code);
  updateChip(true);
  setTimeout(() => updateChip(), 1200);
}
document.addEventListener('click', (e) => {
  if (e.target.closest('#file-open')) fsShow('open');
  const ex = e.target.closest('#file-examples');
  if (ex) { fsDir = ex.dataset.dir; fsShow('open'); }
  if (e.target.closest('#file-saveas')) fsShow('save');
  if (e.target.closest('#file-save')) curFile ? doSave(curFile) : fsShow('save');
  if (e.target.closest('#fs-cancel') || e.target.id === 'fsmodal') fsHide();
  if (e.target.closest('#sug-close') || e.target.id === 'sugmodal')
    document.getElementById('sugmodal').style.display = 'none';
  if (e.target.closest('#sug-copy'))
    navigator.clipboard.writeText(document.getElementById('sug-text').textContent);
  if (e.target.closest('#save-close') || e.target.id === 'savemodal')
    document.getElementById('savemodal').style.display = 'none';
  if (e.target.closest('#save-copy'))
    navigator.clipboard.writeText(document.getElementById('save-text').textContent);
  if (e.target.closest('#save-write')) {  // write to file / apply, maybe remember
    const rem = document.getElementById('save-remember');
    if (rem && rem.checked) eggWriteThrough = true;
    if (eggPendingCommit != null) eggApplyCommit(eggPendingCommit);
    eggPendingCommit = null;
    document.getElementById('savemodal').style.display = 'none';
  }
  if (e.target.closest('#fs-do-save')) {
    const name = document.getElementById('fs-name').value.trim();
    if (name) { doSave(joinP(fsDir, name)); fsHide(); }
  }
});
document.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 's') {
    e.preventDefault();
    curFile ? doSave(curFile) : fsShow('save');
  }
  if (e.key === 'Escape') {
    fsHide();
    document.getElementById('sugmodal').style.display = 'none';
    document.getElementById('savemodal').style.display = 'none';
  }
});
// dirty dot tracks every edit (CodeMirror mirrors into the textarea)
document.addEventListener('input', (e) => {
  if (e.target.matches('.editor textarea')) updateChip();
});
document.addEventListener('keydown', (e) => {
  if (e.target.id === 'fs-name' && e.key === 'Enter')
    document.getElementById('fs-do-save').click();
});
// Ctrl+Enter (Cmd+Enter): run — or stop, when a run is streaming. Capture
// phase, so CodeMirror's Mod-Enter (insert blank line) never sees it.
document.addEventListener('keydown', (e) => {
  if (!(e.ctrlKey || e.metaKey) || e.key !== 'Enter' || e.repeat) return;
  const btn = document.querySelector('#viewbar .btns button.primary:not(:disabled)')
           || document.querySelector('#viewbar .btns button.danger:not(:disabled)');
  if (!btn) return;
  e.preventDefault();
  e.stopPropagation();
  btn.click();
}, true);
// --- watch mode: the user edits the file in their own editor; the UI
// hides the editor pane, polls the opened file, and re-renders on change
// (the run button keeps working against the synced buffer).
let watchTimer = null, watchLast = null, watchSuggested = false;
// once per watched file: if it registers no run, show the block to paste
async function maybeSuggest() {
  if (watchSuggested || !curFile) return;
  watchSuggested = true;  // one nag per file, even on failure
  try {
    const r = await fetch('/api/file?path=' + encodeURIComponent(curFile));
    const j = await r.json();
    if (j.suggest) showSuggestion(j.suggest);
  } catch (err) { /* ignore */ }
}
async function watchTick() {
  if (!curFile) return;
  try {
    const r = await fetch('/api/file?path=' + encodeURIComponent(curFile) + '&check=0');
    const j = await r.json();
    if (j.error || j.code === watchLast) return;
    const changed = watchLast !== null;
    watchLast = j.code;
    if (j.code !== currentCode()) {
      window.eggSetCode(j.code);
      lastSaved = j.code;
      updateChip();
    }
    if (changed) { watchSuggested = false; maybeSuggest(); }
  } catch (err) { /* transient; next tick retries */ }
}
function setWatch(on) {
  const cb = document.getElementById('watch-toggle');
  const panes = document.querySelector('.panes');
  if (on && !curFile) {
    alert('open a file first — watch mode follows the opened file');
    cb.checked = false;
    return;
  }
  if (on && lastSaved !== null && currentCode() !== lastSaved
      && !confirm('discard unsaved editor changes and follow the file on disk?')) {
    cb.checked = false;
    return;
  }
  cb.checked = on;
  panes.classList.toggle('watching', on);
  localStorage.setItem('egg-webui-watch', on ? '1' : '');
  if (on) {
    if (splitInst) { splitInst.destroy(); splitInst = null; }
    panes.classList.remove('split-active');
    watchLast = null;
    watchSuggested = false;
    watchTick();
    maybeSuggest();
    watchTimer = setInterval(watchTick, 1000);
  } else {
    if (watchTimer) { clearInterval(watchTimer); watchTimer = null; }
    initSplit();
  }
}
document.addEventListener('change', (e) => {
  if (e.target.id === 'watch-toggle') setWatch(e.target.checked);
});

// restore the file association: CLI-passed script wins, else localStorage
window.addEventListener('DOMContentLoaded', () => {
  const t = document.querySelector('.editor textarea');
  if (t && t.dataset.file) {
    setFile(t.dataset.file, t.value);
    if (t.dataset.watch === '1') { setWatch(true); return; }
  } else {
    const sp = localStorage.getItem('egg-webui-path');
    if (sp) setScriptPath(sp);
    const saved = localStorage.getItem('egg-webui-file');
    if (saved) {
      curFile = saved;
      updateChip();
      fetch('/api/file?path=' + encodeURIComponent(saved) + '&check=0')
        .then((r) => r.json())
        .then((j) => {
          if (j.error) setFile(null, null);
          else { lastSaved = j.code; updateChip(); }
        })
        .catch(() => {});
    }
  }
  if (localStorage.getItem('egg-webui-watch') && curFile) setWatch(true);
});

// ============ editable topology: the edit view's draw tools ============
// The working blocking lives in eggEd (survives htmx swaps as a JS global,
// localStorage survives reloads). The source stays canonical — writing the
// drawing back into it is a separate commit step.
let eggEd = null;
let eggEdDrag = null;    // in-flight grab (move) / selection gesture
const eggEK = (a, b) => (a < b ? a + '|' + b : b + '|' + a);  // stable edge key

function eggEditData() {
  const el = document.getElementById('egg-edit-data');
  if (!el) return null;
  try { return JSON.parse(el.textContent); } catch (e) { return null; }
}
// world <-> viewBox coords via the render's inverse-transform data-attrs
function eggV2W(el, vx, vy) {
  const d = el.dataset;
  return [+d.lox + (vx - +d.ox) / +d.sx, +d.hiy - (vy - +d.oy) / +d.sx];
}
function eggW2V(el, x, y) {
  const d = el.dataset;
  return [+d.ox + +d.sx * (x - +d.lox), +d.oy + +d.sx * (+d.hiy - y)];
}
// client (screen) px -> SVG user (viewBox) coords via the live CTM, so it
// respects the viewBox letterbox and any pan/zoom (a plain rect ratio does not)
function eggClientVB(svg, cx, cy) {
  const m = svg.getScreenCTM();
  if (!m) return null;
  const pt = svg.createSVGPoint();
  pt.x = cx; pt.y = cy;
  const p = pt.matrixTransform(m.inverse());
  return [p.x, p.y];
}
// a node's world position; split children ride their parent edge at t
function eggNodePos(g, id, seen) {
  const n = g.nodes.get(id);
  if (!n || (!n.xy && !n.split)) {  // no own position -> the base corner's
    const bn = g.baseGraph && g.baseGraph.nodes.get(id);
    return bn ? bn.xy : null;
  }
  if (n.xy) return n.xy;
  if (n.split) {
    seen = seen || new Set();
    if (seen.has(id)) return null;
    seen.add(id);
    const a = eggNodePos(g, n.split[0], seen), b = eggNodePos(g, n.split[1], seen);
    if (!a || !b) return null;
    const t = (n.t == null) ? 0.5 : n.t;
    return [a[0] * (1 - t) + b[0] * t, a[1] * (1 - t) + b[1] * t];
  }
  return null;
}
function eggBuildGraph(data) {
  const f = data.blocking || {};
  const nodes = new Map();
  for (const [id, s] of Object.entries(f.nodes || {}))
    nodes.set(id, {xy: s.xy || null, split: s.split || null, t: s.t,
                   from: s.from || null,
                   fixed: ('fixed' in s ? !!s.fixed : undefined),
                   on: (s.on || []).slice()});
  const edges = (f.edges || []).map(
      (e) => ({a: e.a, b: e.b, bind: e.bind || null, base: e.base || null,
               res: e.res || null}));
  let seq = 0;
  for (const id of nodes.keys()) {
    const m = /^u(\d+)$/.exec(id);
    if (m) seq = Math.max(seq, +m[1] + 1);
  }
  return {geometry: data.geometry || [], res: f.res == null ? 10 : f.res,
          nodes, edges, drawing: null, cursor: null, seq, dirty: false,
          hist: {undo: [], redo: []}, last: null,
          selN: new Set(), selE: new Set()};
}
function eggToBlocking(g) {
  const nodes = {};
  for (const [id, n] of g.nodes) {
    const s = {};
    if (n.xy) s.xy = [+n.xy[0].toFixed(4), +n.xy[1].toFixed(4)];
    if (n.split) { s.split = n.split; s.t = (n.t == null) ? 0.5 : n.t; }
    if (n.from) s.from = n.from;  // split provenance (endpoints of the edge it split)
    if (n.on && n.on.length) s.on = n.on;
    if (typeof n.fixed === 'boolean') s.fixed = n.fixed;
    nodes[id] = s;
  }
  const edges = g.edges.map((e) => {
    const o = {a: e.a, b: e.b};
    if (e.bind) o.bind = e.bind;
    if (e.base) o.base = e.base;  // which base edge this sub-edge subdivides
    if (e.res) o.res = e.res;     // explicit resolution -> drives its loop
    return o;
  });
  return {nodes, edges, res: g.res};
}
function eggSnapshot() { return JSON.stringify(eggToBlocking(eggEd)); }
function eggRestore(json) {  // replace the graph, keep history + geometry
  const g = eggBuildGraph({blocking: JSON.parse(json), geometry: eggEd.geometry});
  eggEd.nodes = g.nodes; eggEd.edges = g.edges; eggEd.res = g.res; eggEd.seq = g.seq;
  eggEd.drawing = null; eggEd.cursor = null;
}
const EGG_ED_KEY = 'egg-edit-blocking';

function eggScriptPath() {
  const el = document.getElementById('scriptpath');
  return el ? el.value : '';
}
function eggReadBuffer() {
  try { return JSON.parse(localStorage.getItem(EGG_ED_KEY) || 'null'); }
  catch (e) { return null; }
}
function eggWriteBuffer() {  // {path, base, work}: only re-adopted for the same
  try {                     // script + the same source blocking it diverged from
    localStorage.setItem(EGG_ED_KEY, JSON.stringify(
      {path: eggEd.path, base: eggEd.base, work: eggEd.last}));
  } catch (e) {}
}
function eggClearBuffer() { try { localStorage.removeItem(EGG_ED_KEY); } catch (e) {} }

function eggEditInit() {
  const svg = svgEl();
  const data = eggEditData();
  const on = !!(svg && data && data.editable);
  document.body.classList.toggle('editing', on);
  if (!on) { eggEd = null; return; }
  const srcKey = JSON.stringify(data.blocking || {});
  const path = eggScriptPath();
  if (!eggEd || !eggEd.dirty) {
    // adopt the working buffer only if it belongs to THIS script and started
    // from the SAME source blocking; a new/edited/empty topology has a different
    // source, so a stale buffer is dropped instead of shadowing it
    let blocking = data.blocking || {}, dirty = false;
    const buf = eggReadBuffer();
    if (buf && buf.path === path && buf.base === srcKey) {
      try { blocking = JSON.parse(buf.work); dirty = true; } catch (e) {}
    } else if (buf) {
      eggClearBuffer();
    }
    eggEd = eggBuildGraph({blocking: blocking, geometry: data.geometry});
    eggEd.dirty = dirty;
    eggEd.base = srcKey;
    eggEd.path = path;
  }
  if (eggEd.last == null) eggEd.last = eggSnapshot();  // history baseline
  // base topology (read-only context to snap to / move / split); refreshed each
  // render since it comes from the source, which is canonical
  const bg = data.base || {nodes: {}, edges: []};
  eggEd.baseGraph = {nodes: new Map(Object.entries(bg.nodes || {})),
                     edges: bg.edges || []};
  eggEnsureControls();
  eggEditRender();
  if (eggEd.dirty) eggValidate();
}
function eggEditReset() {  // discard the in-progress drawing, reload the source
  eggClearBuffer();
  eggEd = null;
  eggEditInit();
}
function eggEnsureControls() {
  const bar = document.querySelector('#viewbar .btns');
  if (!bar) return;
  if (!document.getElementById('ed-reset')) {
    const b = document.createElement('button');
    b.id = 'ed-reset';
    b.textContent = 'reset drawing';
    b.title = "discard the in-progress drawing and reload the script's topology";
    b.addEventListener('click', eggEditReset);
    bar.insertBefore(b, bar.firstChild);
  }
  if (!document.getElementById('ed-auto')) {
    const ab = document.createElement('button');
    ab.id = 'ed-auto';
    ab.textContent = eggAuto ? 'auto: on' : 'auto: off';
    ab.title = 'auto-commit each valid edit to the source (writes to the file)';
    ab.addEventListener('click', () => {
      if (!eggAuto) {  // turning ON: auto-commit writes the file every edit
        if (curFile && eggIsWatching() && !confirm(
            'Auto-save will write the drawing to the WATCHED file on every valid '
            + 'edit. Continue?')) return;  // declined -> stay off
        eggAuto = true; eggWriteThrough = true; eggMaybeAuto();
      } else { eggAuto = false; }
      ab.textContent = eggAuto ? 'auto: on' : 'auto: off';
    });
    bar.insertBefore(ab, bar.firstChild);
  }
  if (!document.getElementById('ed-commit')) {
    const cb = document.createElement('button');
    cb.id = 'ed-commit';
    cb.className = 'primary';
    cb.textContent = 'save edits';
    cb.title = 'write the drawing into the editable({...}) source (only when valid)';
    cb.addEventListener('click', eggCommit);
    bar.insertBefore(cb, bar.firstChild);
    eggUpdateCommitBtn();
  }
  if (!document.getElementById('ed-bind')) {
    const sel = document.createElement('select');
    sel.id = 'ed-bind';
    sel.title = 'assign the selected edge / nodes to a geometry curve (F pins a node)';
    sel.addEventListener('change', () => {
      const v = sel.value;
      if (v) eggBindSelection(v === '__none__' ? null : v);
      sel.value = '';
    });
    bar.insertBefore(sel, bar.firstChild);
  }
  eggRefreshBindOptions();
}
function eggRefreshBindOptions() {
  const sel = document.getElementById('ed-bind');
  if (!sel || !eggEd) return;
  const labels = (eggEd.geometry || []).map((g) => g.label);
  sel.innerHTML = '<option value="">bind to…</option>'
    + labels.map((l) => `<option value="${l}">${l}</option>`).join('')
    + '<option value="__none__">— unbind —</option>';
}
let eggAuto = false;
// "save edits writes straight to the file" — off by default (so a watched file
// is never modified without consent), remembered for the session, reset on load
let eggWriteThrough = false;
let eggPendingCommit = null;
function eggIsWatching() {
  const w = document.getElementById('watch-toggle');
  return !!(w && w.checked);
}
function eggUpdateCommitBtn() {
  const cb = document.getElementById('ed-commit');
  if (cb) cb.disabled = !(eggEd && eggEd.valid);
}
// land a committed source: update the editor buffer, and persist to the file
// when one is open (the write-through / popup path has already consented)
function eggApplyCommit(code) {
  eggClearBuffer();
  eggEd.dirty = false;
  window.eggSetCode(code);
  if (curFile) doSave(curFile);
}
async function eggCommit() {
  if (!eggEd || !eggEd.valid) return;
  const t = document.querySelector('.editor textarea');
  const sp = document.getElementById('scriptpath');
  if (!t) return;
  const body = new URLSearchParams({code: t.value, path: sp ? sp.value : '',
                                    blocking: JSON.stringify(eggToBlocking(eggEd))});
  let j;
  try {
    const res = await fetch('/api/topo/commit', {method: 'POST', body});
    j = await res.json();
    if (!res.ok) { alert(j.error || 'commit failed'); return; }
  } catch (e) { alert('commit failed: ' + e); return; }
  // Not watching -> clicking save is explicit permission to write the file.
  // Watching -> show the block to paste (never silently touch a watched file),
  // unless the user opted into write-through this session.
  if (!eggIsWatching() || eggWriteThrough) { eggApplyCommit(j.code); return; }
  eggPendingCommit = j.code;
  const m = document.getElementById('savemodal');
  document.getElementById('save-text').textContent = j.block || j.code;
  const w = document.getElementById('save-write');
  if (w) w.textContent = curFile ? 'write to file' : 'apply to editor';
  const row = document.getElementById('save-remember-row');
  if (row) row.style.display = curFile ? 'flex' : 'none';
  const rem = document.getElementById('save-remember');
  if (rem) rem.checked = false;
  if (m) m.style.display = 'flex';
}
function eggMaybeAuto() {  // auto-commit a valid, dirty drawing (write-through)
  if (eggAuto && eggEd && eggEd.valid && eggEd.dirty) eggCommit();
}
function eggEditRender() {
  const svg = svgEl();
  if (!svg || !eggEd) return;
  let g = svg.querySelector('.edit-overlay');
  if (!g) {
    g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.setAttribute('class', 'edit-overlay');
    svg.appendChild(g);
  }
  const out = [];
  // which base edges are bifurcated (a split node sits on them)?
  const splitOnBase = new Map();
  for (const [id, n] of eggEd.nodes)
    if (n.split) splitOnBase.set(eggEK(n.split[0], n.split[1]), id);
  const cutBase = eggCutSet();  // edges now drawn as their blocking sub-edges
  // base block edges (read-only context) so a moved corner drags them live; a
  // bifurcated edge draws as its two halves, which follow the split node
  if (eggEd.baseGraph) for (const [ba, bb] of eggEd.baseGraph.edges) {
    const key = eggEK(ba, bb);
    if (cutBase.has(key)) continue;  // subdivided -> the blocking edges draw it
    const m = splitOnBase.get(key);
    const segs = m ? [[ba, m], [m, bb]] : [[ba, bb]];
    const cls = 'ed-edge ed-baseedge' + (eggEd.selE.has(key) ? ' ed-sel' : '')
              + (eggEd.hover && eggEd.hover.ek === key ? ' ed-hover' : '');
    for (const [x, y] of segs) {
      const a = eggNodePos(eggEd, x), b = eggNodePos(eggEd, y);
      if (!a || !b) continue;
      const p = eggW2V(svg, a[0], a[1]), q = eggW2V(svg, b[0], b[1]);
      out.push(`<line x1="${p[0].toFixed(1)}" y1="${p[1].toFixed(1)}" x2="${q[0].toFixed(1)}" y2="${q[1].toFixed(1)}" class="${cls}"/>`);
    }
  }
  for (const e of eggEd.edges) {
    const a = eggNodePos(eggEd, e.a), b = eggNodePos(eggEd, e.b);
    if (!a || !b) continue;
    const p = eggW2V(svg, a[0], a[1]), q = eggW2V(svg, b[0], b[1]);
    const sel = eggEd.selE.has(eggEK(e.a, e.b)) ? ' ed-sel' : '';
    const hov = eggEd.hover && eggEd.hover.ek === eggEK(e.a, e.b) ? ' ed-hover' : '';
    // a binding is shown on hover, not by colour — bound edges stay the default
    // colour so they match the (also-bound) base topology
    out.push(`<line x1="${p[0].toFixed(1)}" y1="${p[1].toFixed(1)}" x2="${q[0].toFixed(1)}" y2="${q[1].toFixed(1)}" class="ed-edge${sel}${hov}"/>`);
  }
  if (eggEd.drawing && eggEd.cursor) {
    const a = eggNodePos(eggEd, eggEd.drawing);
    if (a) {
      const p = eggW2V(svg, a[0], a[1]);
      out.push(`<line x1="${p[0].toFixed(1)}" y1="${p[1].toFixed(1)}" x2="${eggEd.cursor[0].toFixed(1)}" y2="${eggEd.cursor[1].toFixed(1)}" class="ed-rubber"/>`);
    }
  }
  if (eggEdDrag && eggEdDrag.type === 'select' && eggEdDrag.moved) {
    const a = eggEdDrag.v0, b = eggEdDrag.v1;
    out.push(`<rect x="${Math.min(a[0],b[0]).toFixed(1)}" y="${Math.min(a[1],b[1]).toFixed(1)}" width="${Math.abs(b[0]-a[0]).toFixed(1)}" height="${Math.abs(b[1]-a[1]).toFixed(1)}" class="ed-band"/>`);
  }
  for (const id of eggAllNodeIds()) {
    const wp = eggNodePos(eggEd, id);
    if (!wp) continue;
    const v = eggW2V(svg, wp[0], wp[1]);
    const fn = eggEd.nodes.get(id);
    const bn = eggEd.baseGraph && eggEd.baseGraph.nodes.get(id);
    const cls = 'ed-node' + (id === eggEd.drawing ? ' ed-active' : '')
              + (eggEd.selN.has(id) ? ' ed-sel' : '')
              + (bn ? ' ed-basecorner' : '')
              + (eggIsFixedNode(id) ? ' ed-fixed' : '')
              + (eggEd.hover && eggEd.hover.node === id ? ' ed-hover' : '');
    out.push(`<circle cx="${v[0].toFixed(1)}" cy="${v[1].toFixed(1)}" r="5" class="${cls}"/>`);
  }
  g.innerHTML = out.join('');
  scaleMarkers(svg);  // constant on-screen marker size, like the server dots
  const bindSel = document.getElementById('ed-bind');
  if (bindSel) bindSel.disabled = !(eggEd.selN.size || eggEd.selE.size);
  eggSyncToolbar();
}
// all snap-able node ids: the editable blocking + base corners
function eggAllNodeIds() {
  const ids = new Set(eggEd.nodes.keys());
  if (eggEd.baseGraph) for (const k of eggEd.baseGraph.nodes.keys()) ids.add(k);
  return ids;
}
function eggIsFixed(id) {
  const bn = eggEd.baseGraph && eggEd.baseGraph.nodes.get(id);
  return !!(bn && bn.fixed);
}
// promote an unmoved base corner into the blocking so a move can override its
// position (a fresh Corner is made server-side; the base object is untouched)
function eggPromoteBase(id) {
  if (eggEd.nodes.has(id)) return;
  const bn = eggEd.baseGraph && eggEd.baseGraph.nodes.get(id);
  if (bn && !bn.fixed) eggEd.nodes.set(id, {xy: bn.xy.slice(), split: null, on: []});
}
// promote a base corner to a blocking override that keeps its base position (no
// xy) so it can carry a fixed / on change without freezing where it sits
function eggPromoteAny(id) {
  if (eggEd.nodes.has(id)) return;
  if (eggEd.baseGraph && eggEd.baseGraph.nodes.has(id))
    eggEd.nodes.set(id, {split: null, on: []});
}
function eggIsFixedNode(id) {
  const fn = eggEd.nodes.get(id);
  if (fn && typeof fn.fixed === 'boolean') return fn.fixed;  // explicit override
  const bn = eggEd.baseGraph && eggEd.baseGraph.nodes.get(id);
  return !!(bn && bn.fixed);
}
function eggToggleFixed() {  // F: pin / unpin the selected nodes
  if (!eggEd || !eggEd.selN.size) return;
  const target = ![...eggEd.selN].every(eggIsFixedNode);  // all fixed -> unfix
  for (const id of eggEd.selN) {
    eggPromoteAny(id);
    const n = eggEd.nodes.get(id);
    if (n) n.fixed = target;  // explicit true/false (a base corner can be unfixed)
  }
  eggEditChanged();
}
function eggSetOn(id, label) {
  eggPromoteAny(id);
  const n = eggEd.nodes.get(id);
  if (n) n.on = label ? [label] : [];
}
// assign the selection to a geometry curve (declarative associate): a selected
// edge becomes a block face on the curve; nodes get bound to it
function eggBindSelection(label) {
  if (!eggEd || (!eggEd.selN.size && !eggEd.selE.size)) return;
  for (const ek of eggEd.selE) {
    const e = eggEd.edges.find((x) => eggEK(x.a, x.b) === ek);
    if (e) { e.bind = label || null; eggSetOn(e.a, label); eggSetOn(e.b, label); }
  }
  for (const id of eggEd.selN) eggSetOn(id, label);
  eggEditChanged();
}
// a blocking (editable) node, i.e. not an untouched base corner
function eggIsBlockingNode(id) {
  return eggEd.nodes.has(id) && !(eggEd.baseGraph && eggEd.baseGraph.nodes.has(id));
}
// split-at-node: un-weld a shared node so each incident edge gets its own
// coincident copy (two edges through one node -> two separate edges)
function eggSplitAtNode() {
  if (!eggEd) return;
  let changed = false;
  for (const id of [...eggEd.selN]) {
    if (!eggIsBlockingNode(id)) continue;
    const inc = eggEd.edges.filter((e) => e.a === id || e.b === id);
    if (inc.length < 2) continue;
    const at = eggNodePos(eggEd, id), on = (eggEd.nodes.get(id).on || []).slice();
    inc.slice(1).forEach((e) => {  // the first edge keeps the original node
      const nid = 'u' + (eggEd.seq++);
      eggEd.nodes.set(nid, {xy: at.slice(), split: null, on: on.slice()});
      if (e.a === id) e.a = nid; else e.b = nid;
    });
    changed = true;
  }
  if (changed) { eggEd.selN.clear(); eggEd.selE.clear(); eggEditChanged(); }
}
// join: weld the selected blocking nodes into the first, reconnecting edges and
// dropping self-loops / duplicates (base corners are left untouched)
function eggJoinNodes() {
  if (!eggEd) return;
  const ids = [...eggEd.selN].filter(eggIsBlockingNode);
  if (ids.length < 2) return;
  const keep = ids[0], gone = new Set(ids.slice(1));
  for (const e of eggEd.edges) {
    if (gone.has(e.a)) e.a = keep;
    if (gone.has(e.b)) e.b = keep;
  }
  const seen = new Set(), kept = [];
  for (const e of eggEd.edges) {
    if (e.a === e.b) continue;                       // self-loop from the merge
    const k = eggEK(e.a, e.b);
    if (seen.has(k)) continue;                       // duplicate
    seen.add(k); kept.push(e);
  }
  eggEd.edges = kept;
  for (const id of gone) eggEd.nodes.delete(id);
  eggEd.selN = new Set([keep]); eggEd.selE = new Set();
  eggEditChanged();
}
// coincident: snap the one selected node onto the one selected edge — it rides
// that edge at the projected parameter
// coincident: insert the one selected node into the one selected edge, snapping
// it onto the line and splitting the edge into two through it (a base edge cuts,
// carrying its geometry; a blocking edge splits, inheriting bind/base)
function eggMakeCoincident() {
  if (!eggEd || eggEd.selN.size !== 1 || eggEd.selE.size !== 1) return;
  const nid = [...eggEd.selN][0], [a, b] = [...eggEd.selE][0].split('|');
  if (!eggIsBlockingNode(nid) || nid === a || nid === b) return;
  const pa = eggNodePos(eggEd, a), pb = eggNodePos(eggEd, b), pn = eggNodePos(eggEd, nid);
  const ex = pb[0] - pa[0], ey = pb[1] - pa[1], L2 = ex * ex + ey * ey;
  const t = Math.max(0, Math.min(1, L2 ? ((pn[0] - pa[0]) * ex + (pn[1] - pa[1]) * ey) / L2 : 0.5));
  const n = eggEd.nodes.get(nid);
  n.xy = [pa[0] + t * ex, pa[1] + t * ey];  // onto the line
  delete n.split;
  const key = eggEK(a, b);
  const fe = eggEd.edges.find((e) => eggEK(e.a, e.b) === key);
  if (fe) {  // blocking edge -> split it through nid
    eggEd.edges = eggEd.edges.filter((e) => e !== fe);
    eggEd.edges.push({a: a, b: nid, bind: fe.bind || null, base: fe.base || null});
    eggEd.edges.push({a: nid, b: b, bind: fe.bind || null, base: fe.base || null});
  } else if (eggEd.baseGraph) {  // base edge -> cut it through nid
    let curve = null;
    for (const [x, y, c] of eggEd.baseGraph.edges)
      if (eggEK(x, y) === key) { curve = c || null; break; }
    if (curve) n.on = [curve];
    eggEd.edges.push({a: a, b: nid, bind: curve, base: [a, b]});
    eggEd.edges.push({a: nid, b: b, bind: curve, base: [a, b]});
  }
  eggEd.selE = new Set();
  eggEditChanged();
}
// set resolution: cell count along the selected edge(s). Because opposite faces
// of a block share a resolution (and shared faces link blocks), the flatten
// propagates one setting around the whole loop — so the user sets one edge and
// every edge that must stay consistent follows. A blocking edge stores it inline;
// a base edge gets a res-only blocking edge (no cut, no re-block). Opens a themed
// input (not a browser prompt), applied on 'set'/Enter, dismissed on cancel/Esc.
let eggResPending = null;  // selE snapshot awaiting the res modal
function eggSetEdgeRes() {
  if (!eggEd || !eggEd.selE.size) return;
  let cur = null;
  for (const ek of eggEd.selE) {
    const fe = eggEd.edges.find((e) => eggEK(e.a, e.b) === ek);
    if (fe && fe.res) { cur = fe.res; break; }
  }
  eggResPending = new Set(eggEd.selE);
  const inp = document.getElementById('res-input');
  inp.value = cur || eggEd.res || 10;
  inp.classList.remove('invalid');
  document.getElementById('resmodal').style.display = 'flex';
  inp.focus();
  inp.select();
}
function eggResApply() {
  const inp = document.getElementById('res-input');
  const n = Math.round(+inp.value);
  if (!isFinite(n) || n < 1) { inp.classList.add('invalid'); inp.focus(); inp.select(); return; }
  if (eggEd && eggResPending) {
    for (const ek of eggResPending) {
      const fe = eggEd.edges.find((e) => eggEK(e.a, e.b) === ek);
      if (fe) { fe.res = n; continue; }
      const [a, b] = ek.split('|');  // an uncut base edge -> res-only override
      eggEd.edges.push({a, b, bind: null, base: null, res: n});
    }
    eggEditChanged();
  }
  eggResClose();
}
function eggResClose() {
  eggResPending = null;
  const m = document.getElementById('resmodal');
  if (m) m.style.display = 'none';
}
document.addEventListener('click', (e) => {
  if (e.target.closest('#res-ok')) eggResApply();
  else if (e.target.closest('#res-cancel') || e.target.closest('#res-close')
           || e.target.id === 'resmodal') eggResClose();
});
document.addEventListener('keydown', (e) => {
  const m = document.getElementById('resmodal');
  if (!m || m.style.display !== 'flex') return;
  if (e.key === 'Enter') { e.preventDefault(); e.stopPropagation(); eggResApply(); }
  else if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); eggResClose(); }
}, true);
function eggSyncToolbar() {
  const nF = [...eggEd.selN].filter(eggIsBlockingNode).length;
  const set = (id, ok) => { const b = document.getElementById(id); if (b) b.disabled = !ok; };
  set('tool-split', [...eggEd.selN].some((id) => eggIsBlockingNode(id)
        && eggEd.edges.filter((e) => e.a === id || e.b === id).length >= 2));
  set('tool-join', nF >= 2);
  set('tool-coincident', eggEd.selN.size === 1 && eggEd.selE.size === 1);
  set('tool-res', eggEd.selE.size >= 1);
}
document.addEventListener('click', (e) => {
  if (!eggEd) return;
  if (e.target.closest('#tool-split')) eggSplitAtNode();
  else if (e.target.closest('#tool-join')) eggJoinNodes();
  else if (e.target.closest('#tool-coincident')) eggMakeCoincident();
  else if (e.target.closest('#tool-res')) eggSetEdgeRes();
});
// nearest snap-able node within maxD viewBox units of a world point
function eggPickNode(svg, wx, wy, maxD) {
  let best = null, bestD = maxD * maxD;
  const c = eggW2V(svg, wx, wy);
  for (const id of eggAllNodeIds()) {
    const wp = eggNodePos(eggEd, id);
    if (!wp) continue;
    const v = eggW2V(svg, wp[0], wp[1]);
    const d2 = (v[0] - c[0]) ** 2 + (v[1] - c[1]) ** 2;
    if (d2 < bestD) { bestD = d2; best = id; }
  }
  return best;
}
// nearest edge (by key) within maxPx of a viewBox point, or null
function eggPickEdge(svg, vb, maxPx) {
  const maxD = maxPx / svg.getScreenCTM().a;
  let best = null, bestD = maxD * maxD;
  const consider = (na, nb) => {
    const a = eggNodePos(eggEd, na), b = eggNodePos(eggEd, nb);
    if (!a || !b) return;
    const va = eggW2V(svg, a[0], a[1]), vb2 = eggW2V(svg, b[0], b[1]);
    const d2 = segDist2(vb[0], vb[1], va, vb2);
    if (d2 < bestD) { bestD = d2; best = eggEK(na, nb); }
  };
  for (const e of eggEd.edges) consider(e.a, e.b);         // editable edges
  if (eggEd.baseGraph) {                                   // uncut base edges too
    const cut = eggCutSet();
    for (const [ba, bb] of eggEd.baseGraph.edges)
      if (!cut.has(eggEK(ba, bb))) consider(ba, bb);
  }
  return best;
}
// nearest bindable geometry curve to a click: its label + the projected
// world point on it, within maxPx, or null
function eggPickCurve(svg, vb, maxPx) {
  const maxD = maxPx / svg.getScreenCTM().a;
  let best = null, bestD = maxD * maxD;
  for (const gc of eggEd.geometry || []) {
    const pts = gc.points || [];
    for (let i = 0; i + 1 < pts.length; i++) {
      const a = eggW2V(svg, pts[i][0], pts[i][1]);
      const b = eggW2V(svg, pts[i + 1][0], pts[i + 1][1]);
      const dx = b[0] - a[0], dy = b[1] - a[1], L2 = dx * dx + dy * dy;
      let t = L2 ? ((vb[0] - a[0]) * dx + (vb[1] - a[1]) * dy) / L2 : 0;
      t = Math.max(0, Math.min(1, t));
      const qx = a[0] + t * dx, qy = a[1] + t * dy;
      const d2 = (vb[0] - qx) ** 2 + (vb[1] - qy) ** 2;
      if (d2 < bestD) {
        bestD = d2;
        best = {label: gc.label,
                xy: [pts[i][0] * (1 - t) + pts[i + 1][0] * t,
                     pts[i][1] * (1 - t) + pts[i + 1][1] * t]};
      }
    }
  }
  return best;
}
// nearest editable edge to a click: the edge object + projected world point
function eggEdgeHit(svg, vb, maxPx) {
  const maxD = maxPx / svg.getScreenCTM().a;
  let best = null, bestD = maxD * maxD;
  for (const e of eggEd.edges) {
    const a = eggNodePos(eggEd, e.a), b = eggNodePos(eggEd, e.b);
    if (!a || !b) continue;
    const va = eggW2V(svg, a[0], a[1]), vb2 = eggW2V(svg, b[0], b[1]);
    const dx = vb2[0] - va[0], dy = vb2[1] - va[1], L2 = dx * dx + dy * dy;
    let t = L2 ? ((vb[0] - va[0]) * dx + (vb[1] - va[1]) * dy) / L2 : 0;
    t = Math.max(0, Math.min(1, t));
    const d2 = (vb[0] - (va[0] + t * dx)) ** 2 + (vb[1] - (va[1] + t * dy)) ** 2;
    if (d2 < bestD) {
      bestD = d2;
      best = {edge: e, xy: [a[0] * (1 - t) + b[0] * t, a[1] * (1 - t) + b[1] * t]};
    }
  }
  return best;
}
// nearest base block edge to a click: its endpoints (names), param t, and the
// projected world point — for splitting/bifurcating a programmatic edge
// base edges already subdivided into blocking sub-edges (by the base tag) — those
// are now editable edges, not base ones
function eggCutSet() {
  const s = new Set();
  for (const e of eggEd.edges) if (e.base) s.add(eggEK(e.base[0], e.base[1]));
  return s;
}
function eggBaseEdgeHit(svg, vb, maxPx) {
  if (!eggEd.baseGraph) return null;
  const cut = eggCutSet();
  const maxD = maxPx / svg.getScreenCTM().a;
  let best = null, bestD = maxD * maxD;
  for (const [na, nb, curve] of eggEd.baseGraph.edges) {
    if (cut.has(eggEK(na, nb))) continue;  // already subdivided into blocking edges
    const a = eggNodePos(eggEd, na), b = eggNodePos(eggEd, nb);
    if (!a || !b) continue;
    const va = eggW2V(svg, a[0], a[1]), vb2 = eggW2V(svg, b[0], b[1]);
    const dx = vb2[0] - va[0], dy = vb2[1] - va[1], L2 = dx * dx + dy * dy;
    let t = L2 ? ((vb[0] - va[0]) * dx + (vb[1] - va[1]) * dy) / L2 : 0;
    t = Math.max(0, Math.min(1, t));
    const d2 = (vb[0] - (va[0] + t * dx)) ** 2 + (vb[1] - (va[1] + t * dy)) ** 2;
    if (d2 < bestD) {
      bestD = d2;
      best = {a: na, b: nb, t, curve: curve || null,
              xy: [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]};
    }
  }
  return best;
}
// split an editable edge at a new node: replace it with two halves, both still
// attached to their old endpoints; the node inherits the edge's curves and each
// half inherits the base-edge tag, so a cut base edge stays cut recursively
function eggSplitEdge(eh, id) {
  const e = eh.edge;
  const oa = eggEd.nodes.get(e.a) && eggEd.nodes.get(e.a).on || [];
  const ob = eggEd.nodes.get(e.b) && eggEd.nodes.get(e.b).on || [];
  eggEd.nodes.set(id, {xy: eh.xy, split: null, from: [e.a, e.b],
                       on: oa.filter((l) => ob.includes(l))});
  eggEd.edges = eggEd.edges.filter((x) => x !== e);
  eggEd.edges.push({a: e.a, b: id, bind: e.bind || null, base: e.base || null});
  eggEd.edges.push({a: id, b: e.b, bind: e.bind || null, base: e.base || null});
}
// cut a base block edge into two editable sub-edges — each tagged with the base
// edge it came from and bound to that edge's geometry, so the programmatic
// association survives and the sub-edges are fully editable (and re-cuttable)
function eggCutBaseEdge(a, b, curve, xy) {
  const M = eggSplitName(a, b);
  eggEd.nodes.set(M, {xy: xy.slice(), split: null, from: [a, b],
                      on: curve ? [curve] : []});
  const base = [a, b];
  eggEd.edges.push({a: a, b: M, bind: curve || null, base});
  eggEd.edges.push({a: M, b: b, bind: curve || null, base});
  return M;
}
// nodes a move drags: the selection (nodes + selected-edge endpoints)
function eggMoveNodes() {
  const s = new Set(eggEd.selN);
  for (const ek of eggEd.selE) { const p = ek.split('|'); s.add(p[0]); s.add(p[1]); }
  return s;
}
// Modeless pointer model, disambiguated by gesture — no draw/select mode:
//   plain click        -> edit (draw / continue / bifurcate / place)
//   plain drag on item  -> move that node/edge; on empty -> pan
//   shift|ctrl click    -> toggle-select the item under the cursor
//   shift|ctrl drag     -> rubber-band box select
//   middle button       -> always pan (not grabbed here)
function eggEdPointerDown(e) {
  const svg = svgEl();
  if (!svg || !eggEd || !svg.getScreenCTM()) return;
  if (e.pointerType === 'mouse' && e.button === 1) return;  // middle -> pan
  const vb = eggClientVB(svg, e.clientX, e.clientY);
  if (!vb) return;
  const [wx, wy] = eggV2W(svg, vb[0], vb[1]);
  const node = eggPickNode(svg, wx, wy, 16 / svg.getScreenCTM().a);
  const ek = node ? null : eggPickEdge(svg, vb, 12);
  if (e.shiftKey || e.ctrlKey) {  // selection: click toggles, drag boxes
    eggEdDrag = {type: 'select', node, ek, v0: vb, v1: vb,
                 px: e.clientX, py: e.clientY, moved: false};
    eggEditRender();
    return;
  }
  if (node || ek) {  // on an item: click edits, drag moves
    eggEdDrag = {type: 'grab', node, ek, world: [wx, wy],
                 px: e.clientX, py: e.clientY, moved: false};
    eggEditRender();
    return;
  }
  // empty + no modifier: don't grab, so the pan handler pans on a drag and a
  // click (panMoved stays false) falls through to eggEditClick
}
function eggEdPointerMove(e) {
  const svg = svgEl();
  if (!svg || !eggEdDrag) return;
  const vb = eggClientVB(svg, e.clientX, e.clientY);
  if (!vb) return;
  const far = Math.hypot(e.clientX - eggEdDrag.px, e.clientY - eggEdDrag.py) >= 4;
  if (eggEdDrag.type === 'select') {
    if (far) { eggEdDrag.moved = true; panMoved = true; }
    eggEdDrag.v1 = vb;
    eggEditRender();
    return;
  }
  // grab: a few px of slop before a click becomes a move (so a click never nudges)
  if (!eggEdDrag.moved && !far) return;
  if (!eggEdDrag.moved) {  // first real motion -> resolve what moves
    eggEdDrag.moved = true;
    panMoved = true;
    let set;
    if (eggEdDrag.node)
      set = eggEd.selN.has(eggEdDrag.node) ? eggMoveNodes() : new Set([eggEdDrag.node]);
    else { const p = eggEdDrag.ek.split('|'); set = new Set([p[0], p[1]]); }  // move an edge = both ends
    eggEdDrag.nodes = new Set([...set].filter((id) => !eggIsFixed(id)));
    eggEdDrag.nodes.forEach(eggPromoteBase);
  }
  const [wx, wy] = eggV2W(svg, vb[0], vb[1]);
  const dx = wx - eggEdDrag.world[0], dy = wy - eggEdDrag.world[1];
  panMoved = true;
  for (const id of eggEdDrag.nodes) {
    const n = eggEd.nodes.get(id);
    if (!n) continue;
    if (!n.xy && n.split) n.xy = eggNodePos(eggEd, id);  // detach from its edge
    if (n.xy) n.xy = [n.xy[0] + dx, n.xy[1] + dy];  // move freely; a bifurcated
    // edge stays split and its two halves follow this node
  }
  eggEdDrag.world = [wx, wy];
  eggEditRender();
}
function eggEdPointerUp() {
  if (!eggEdDrag) return;
  const g = eggEdDrag;
  eggEdDrag = null;
  if (g.type === 'select') {
    if (g.moved) eggBandSelect(g.v0, g.v1);
    else eggSelectToggle(g.node, g.ek);
  } else if (g.type === 'grab' && g.moved) {
    eggEditChanged();  // a committed move
  }  // a non-moved grab is a plain click -> eggEditClick (the click event) draws
  eggEditRender();
}
function eggSelectToggle(node, ek) {
  if (node) {
    // shift-picking a node that connects to an already-selected one also grabs
    // the edge between them (blocking or base)
    if (!eggEd.selN.has(node)) {
      const sel = (x, y) => (x === node && eggEd.selN.has(y))
                         || (y === node && eggEd.selN.has(x));
      for (const e of eggEd.edges)
        if (sel(e.a, e.b)) eggEd.selE.add(eggEK(e.a, e.b));
      if (eggEd.baseGraph)
        for (const [x, y] of eggEd.baseGraph.edges)
          if (sel(x, y)) eggEd.selE.add(eggEK(x, y));
    }
    eggEd.selN.has(node) ? eggEd.selN.delete(node) : eggEd.selN.add(node);
  } else if (ek) {
    eggEd.selE.has(ek) ? eggEd.selE.delete(ek) : eggEd.selE.add(ek);
  } else {
    eggEd.selN.clear(); eggEd.selE.clear();  // shift-click empty clears
  }
  eggEditRender();
}
function eggBandSelect(v0, v1) {
  const svg = svgEl();
  const x0 = Math.min(v0[0], v1[0]), x1 = Math.max(v0[0], v1[0]);
  const y0 = Math.min(v0[1], v1[1]), y1 = Math.max(v0[1], v1[1]);
  if (x1 - x0 < 3 && y1 - y0 < 3) return;  // a click, not a box
  for (const [id] of eggEd.nodes) {
    const wp = eggNodePos(eggEd, id);
    if (!wp) continue;
    const v = eggW2V(svg, wp[0], wp[1]);
    if (v[0] >= x0 && v[0] <= x1 && v[1] >= y0 && v[1] <= y1) eggEd.selN.add(id);
  }
  for (const e of eggEd.edges)  // an edge is in the box iff both ends are
    if (eggEd.selN.has(e.a) && eggEd.selN.has(e.b)) eggEd.selE.add(eggEK(e.a, e.b));
}
function eggDeleteSel() {
  if (!eggEd.selN.size && !eggEd.selE.size) return;
  const dn = eggEd.selN;
  eggEd.edges = eggEd.edges.filter((e) =>
    !dn.has(e.a) && !dn.has(e.b) && !eggEd.selE.has(eggEK(e.a, e.b)));
  for (const id of dn) eggEd.nodes.delete(id);
  eggEd.selN = new Set(); eggEd.selE = new Set(); eggEd.drawing = null;
  eggEditChanged();
}
function eggIdTaken(id) {
  return eggEd.nodes.has(id) || (eggEd.baseGraph && eggEd.baseGraph.nodes.has(id));
}
function eggUniqueId(base) {  // base, base_2, base_3, … — first free id
  if (!eggIdTaken(base)) return base;
  let k = 2;
  while (eggIdTaken(base + '_' + k)) k++;
  return base + '_' + k;
}
// a split node is named after the two nodes it lands between (retaining the
// programmatic names it came from, with a uniqueness suffix)
// compact sequential id for a split/cut node; its provenance (the two endpoints
// of the edge it split) is kept in the node's `from` field, not baked into the
// name — so names stay short instead of concatenating (u16_u17_u17_...).
function eggSplitName(a, b) { return 'u' + (eggEd.seq++); }
// rename a blocking node's connectivity-dict key (base corners are programmatic
// and keep their names)
function eggRenameNode(id) {
  if (!eggIsBlockingNode(id)) return;
  const nn = (prompt('rename node "' + id + '" to:', id) || '').trim();
  if (!nn || nn === id) return;
  if (eggIdTaken(nn)) { alert('name "' + nn + '" is already in use'); return; }
  eggEd.nodes.set(nn, eggEd.nodes.get(id));
  eggEd.nodes.delete(id);
  for (const e of eggEd.edges) { if (e.a === id) e.a = nn; if (e.b === id) e.b = nn; }
  for (const [, n] of eggEd.nodes)  // split parents that referenced it
    if (n.split) n.split = n.split.map((x) => (x === id ? nn : x));
  if (eggEd.selN.delete(id)) eggEd.selN.add(nn);
  eggEditChanged();
}
document.addEventListener('contextmenu', (e) => {
  if (!eggEd) return;
  const svg = svgEl();
  if (!svg || !svg.getScreenCTM() || !e.target.closest('.canvas')) return;
  const vb = eggClientVB(svg, e.clientX, e.clientY);
  if (!vb) return;
  const [wx, wy] = eggV2W(svg, vb[0], vb[1]);
  const node = eggPickNode(svg, wx, wy, 16 / svg.getScreenCTM().a);
  if (node && eggIsBlockingNode(node)) { e.preventDefault(); eggRenameNode(node); }
});
// place / snap a node at a click (existing node, edge split, curve, or free)
function eggPlaceNode(svg, vb, wx, wy, coarse, maxD) {
  const snap = eggPickNode(svg, wx, wy, maxD);
  if (snap) return snap;  // connect to an existing node
  const eh = eggEdgeHit(svg, vb, coarse ? 24 : 12);
  const be = eh ? null : eggBaseEdgeHit(svg, vb, coarse ? 24 : 12);
  if (eh) {  // editable edge -> bifurcate it (named after the edge it splits)
    const target = eggSplitName(eh.edge.a, eh.edge.b);
    eggSplitEdge(eh, target);
    return target;
  }
  if (be)  // base block edge -> cut it into two editable sub-edges
    return eggCutBaseEdge(be.a, be.b, be.curve, be.xy);
  const target = 'u' + (eggEd.seq++);
  const hit = eggPickCurve(svg, vb, coarse ? 24 : 12);  // snap+bind to a curve
  if (hit) eggEd.nodes.set(target, {xy: hit.xy, split: null, on: [hit.label]});
  else eggEd.nodes.set(target, {xy: [wx, wy], split: null, on: []});
  return target;
}
function eggPlaceAndContinue(svg, vb, wx, wy, coarse, maxD) {
  const target = eggPlaceNode(svg, vb, wx, wy, coarse, maxD);
  if (eggEd.drawing && eggEd.drawing !== target)
    eggEd.edges.push({a: eggEd.drawing, b: target, bind: null});
  eggEd.drawing = target;
  eggEditChanged();
}
// plain single click while idle: replace the selection (or clear it on empty)
function eggSelectAt(svg, vb, wx, wy, maxD) {
  const node = eggPickNode(svg, wx, wy, maxD);
  const ek = node ? null : eggPickEdge(svg, vb, 12);
  eggEd.selN.clear();
  eggEd.selE.clear();
  if (node) eggEd.selN.add(node);
  else if (ek) eggEd.selE.add(ek);
  eggEditRender();
}
// Click model: double-click starts a chain (and ends one); while a chain is
// open, single click places a node and continues; while idle, single click is
// a plain (replace) select, empty-click deselects. Shift/Ctrl stay additive.
function eggEditClick(e) {
  const svg = svgEl();
  if (!svg || !eggEd || !e.target.closest('.canvas') || panMoved) return;
  if (e.shiftKey || e.ctrlKey) return;  // additive select — handled on pointerup
  const vb = eggClientVB(svg, e.clientX, e.clientY);
  if (!vb) return;
  const [wx, wy] = eggV2W(svg, vb[0], vb[1]);
  const coarse = window.matchMedia('(pointer: coarse)').matches;
  const maxD = (coarse ? 24 : 14) / svg.getScreenCTM().a;  // px radius -> viewBox units
  const drawing = eggEd.drawing != null;
  if (e.detail >= 2) {  // double-click: end an open chain, or start a new one
    if (drawing) { eggEd.drawing = null; eggEd.cursor = null; eggEditRender(); }
    else {
      eggEd.selN.clear(); eggEd.selE.clear();
      eggPlaceAndContinue(svg, vb, wx, wy, coarse, maxD);
    }
    return;
  }
  if (drawing) eggPlaceAndContinue(svg, vb, wx, wy, coarse, maxD);
  else eggSelectAt(svg, vb, wx, wy, maxD);
}
function eggEditChanged() {
  eggEd.dirty = true;
  const cur = eggSnapshot();
  if (cur !== eggEd.last) {  // push the prior state for undo
    eggEd.hist.undo.push(eggEd.last);
    if (eggEd.hist.undo.length > 200) eggEd.hist.undo.shift();
    eggEd.hist.redo = [];
    eggEd.last = cur;
  }
  eggWriteBuffer();
  eggEditRender();
  eggValidate();
}
function eggAfterHistory() {
  eggEd.dirty = true;
  eggWriteBuffer();
  eggEditRender();
  eggValidate();
}
function eggUndo() {
  const h = eggEd.hist;
  if (!h.undo.length) return;
  h.redo.push(eggEd.last);
  eggEd.last = h.undo.pop();
  eggRestore(eggEd.last);
  eggAfterHistory();
}
function eggRedo() {
  const h = eggEd.hist;
  if (!h.redo.length) return;
  h.undo.push(eggEd.last);
  eggEd.last = h.redo.pop();
  eggRestore(eggEd.last);
  eggAfterHistory();
}
let eggValTimer = null;
function eggValidate() {
  clearTimeout(eggValTimer);
  eggValTimer = setTimeout(async () => {
    if (!eggEd) return;
    const t = document.querySelector('.editor textarea');
    const sp = document.getElementById('scriptpath');
    if (!t) return;
    const body = new URLSearchParams({code: t.value, path: sp ? sp.value : '',
                                      blocking: JSON.stringify(eggToBlocking(eggEd))});
    try {
      const res = await fetch('/api/topo/validate', {method: 'POST', body});
      const j = await res.json();
      eggShowValidity(j.diagnostics || []);
    } catch (e) {}
  }, 350);
}
function eggShowValidity(diags) {
  const bar = document.getElementById('viewbar');
  let chip = document.getElementById('ed-validity');
  if (!chip && bar) {
    chip = document.createElement('span');
    chip.id = 'ed-validity';
    bar.insertBefore(chip, bar.querySelector('.btns') || null);
  }
  // warn_* diagnostics are advisory (dropped markers) — they are shown but do
  // not block a commit; only real errors keep the topology red
  const errs = diags.filter((d) => !(d.kind || '').startsWith('warn'));
  const warns = diags.filter((d) => (d.kind || '').startsWith('warn'));
  if (eggEd) eggEd.valid = errs.length === 0;
  eggUpdateCommitBtn();
  eggMaybeAuto();
  if (!chip) return;
  if (errs.length) {
    chip.className = 'bad';
    chip.textContent = errs.length + ' issue' + (errs.length > 1 ? 's' : '');
  } else if (warns.length) {
    chip.className = 'warn';
    chip.textContent = warns.length + ' warning' + (warns.length > 1 ? 's' : '');
  } else {
    chip.className = 'q-tag';
    chip.textContent = 'valid';
  }
  chip.title = diags.map((d) => d.msg).join('\n');
}
// what the cursor is over, for the properties readout: node, edge, and any
// geometry curve — all of them, so an overlap (edge on a boundary) shows both
function eggNodeDesc(id) {
  const bn = eggEd.baseGraph && eggEd.baseGraph.nodes.get(id);
  const fn = eggEd.nodes.get(id);  // blocking binds + base associations both count
  const on = [...new Set([...((fn && fn.on) || []), ...((bn && bn.on) || [])])];
  let s = (bn ? 'corner ' : 'node ') + id;
  if (eggIsFixedNode(id)) s += ' · fixed';
  else if (on.length >= 2) s += ' · pinned on ' + on.join(', ');
  else if (on.length === 1) s += ' · slides on ' + on[0];
  else s += ' · free';
  return s;
}
function eggEdgeDesc(ek) {
  const p = ek.split('|');
  const fe = eggEd.edges.find((e) => eggEK(e.a, e.b) === ek);
  let isBase = false, curve = null;
  if (eggEd.baseGraph)
    for (const [x, y, c] of eggEd.baseGraph.edges)
      if (eggEK(x, y) === ek) { isBase = true; curve = c; break; }
  let s = (isBase ? 'base edge ' : 'edge ') + p[0] + '–' + p[1];
  const bind = (fe && fe.bind) || curve;
  if (bind) s += ' · projected onto ' + bind;
  if (fe && fe.res) s += ' · ' + fe.res + ' cells';
  return s;
}
function eggHoverInfo(svg, vb, wx, wy) {
  const out = [];
  const node = eggPickNode(svg, wx, wy, 16 / svg.getScreenCTM().a);
  const ek = eggPickEdge(svg, vb, 12);
  const cv = eggPickCurve(svg, vb, 12);
  if (node) out.push(eggNodeDesc(node));
  if (ek) out.push(eggEdgeDesc(ek));
  if (cv) out.push('curve ' + cv.label);
  return out;
}
// hover: light up the snap target under the cursor, show properties in the
// readout, and trail the rubber band while a chain is open
document.addEventListener('pointermove', (e) => {
  if (!eggEd || eggEdDrag) return;
  const svg = svgEl();
  const info = document.getElementById('selinfo');
  if (!svg || !svg.getScreenCTM() || !e.target.closest('.canvas')) {
    if (eggEd.hover) { eggEd.hover = null; eggEditRender(); }
    if (info) info.style.display = 'none';
    return;
  }
  const vb = eggClientVB(svg, e.clientX, e.clientY);
  if (!vb) return;
  const [wx, wy] = eggV2W(svg, vb[0], vb[1]);
  const node = eggPickNode(svg, wx, wy, 16 / svg.getScreenCTM().a);
  const hover = {node, ek: node ? null : eggPickEdge(svg, vb, 12)};
  const changed = !eggEd.hover || eggEd.hover.node !== hover.node
                  || eggEd.hover.ek !== hover.ek;
  eggEd.hover = hover;
  if (eggEd.drawing) eggEd.cursor = vb;
  if (info) {
    const lines = eggHoverInfo(svg, vb, wx, wy);
    if (lines.length) {
      info.textContent = lines.join('   |   ');
      positionCoords();
      info.style.display = 'block';
    } else { info.style.display = 'none'; }
  }
  if (changed || eggEd.drawing) eggEditRender();
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && eggEd && eggEd.drawing) {
    eggEd.drawing = null; eggEd.cursor = null; eggEditRender();
  }
});
// delete the selection; F pins/unpins (not while typing in the editor)
document.addEventListener('keydown', (e) => {
  if (!eggEd || (e.target && e.target.closest && e.target.closest('.editor'))) return;
  if ((e.key === 'Delete' || e.key === 'Backspace')
      && (eggEd.selN.size || eggEd.selE.size)) {
    e.preventDefault(); eggDeleteSel();
  } else if (!e.ctrlKey && !e.metaKey && (e.key === 'f' || e.key === 'F')
             && eggEd.selN.size) {
    e.preventDefault(); eggToggleFixed();  // pin / unpin selected nodes
  }
});
// undo / redo the drawing (only when not typing in the code editor, which
// keeps its own history)
document.addEventListener('keydown', (e) => {
  if (!eggEd) return;
  const el = e.target;
  if (el && el.closest && el.closest('.editor')) return;  // editor keeps its own undo
  if (!(e.ctrlKey || e.metaKey)) return;
  const k = e.key.toLowerCase();
  if (k === 'z' && !e.shiftKey) { e.preventDefault(); eggUndo(); }
  else if (k === 'y' || (k === 'z' && e.shiftKey)) { e.preventDefault(); eggRedo(); }
});
// initial load renders the edit view server-side (no swap fires applyView),
// so build the overlay once the DOM is ready too
window.addEventListener('DOMContentLoaded', eggEditInit);
window.addEventListener('DOMContentLoaded', eggSyncParamsVis);
