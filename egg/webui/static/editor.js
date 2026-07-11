// Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
//
// PolyForm Noncommercial License 2.0.0-pre.2
// https://github.com/bezmi/egg/blob/main/LICENSE.md
// Free to use and redistribute for personal and noncommercial purposes.
// See the license for details.
// For commercial licensing, contact s.imran@tuta.io


// Progressive-enhancement upgrade of the plain <textarea> into a
// prism-code-editor. All modules are served locally from the /vendor mount
// (produced by tools/vendor_webui.py) — there is NO CDN fallback by design,
// and the app refuses to start without these assets, so an import failure
// here is a genuine bug, not an offline condition.
try {
  const man = await (await fetch('/vendor/manifest.json')).json();
  const load = (k) => import('/vendor/' + man[k]);
  const [core, cmds, mb, hb, cur, srch, ac, _tips, utils] = await Promise.all(
    ['core', 'commands', 'match-brackets', 'highlight-brackets', 'cursor',
     'search', 'autocomplete', 'tooltips', 'utils'].map(load));
  // Side-effect imports: the token grammar registers 'python' into prism's
  // tokenizer registry, and the language module registers its comment/indent
  // behavior into languageMap.
  await load('prism-python');
  await load('lang-python');

  const ta = document.querySelector('.editor textarea');
  const host = ta.parentElement;  // the .editor div (keeps the named textarea)

  // Compose the editor in the light DOM (not the shadow-root setups) so the
  // page's catppuccin CSS variables reach the tokens. Extensions are exactly
  // the set the UI enables: editorCommands (editing/indent/auto-close),
  // editHistory (undo/redo), matchBrackets + highlightBracketPairs,
  // cursorPosition (also required by autocomplete) + customCursor,
  // searchWidget + highlightSelectionMatches, and autoComplete.
  const editor = core.createEditor(
    host,
    {language: 'python', value: ta.value, tabSize: 4, insertSpaces: true,
     lineNumbers: true, wordWrap: localStorage.getItem('egg-webui-wrap') === '1'},
    cmds.editorCommands(cmds.defaultKeymap),
    cmds.editHistory(),
    mb.matchBrackets(),
    hb.highlightBracketPairs(),
    cur.cursorPosition(),
    cur.customCursor(),
    srch.searchWidget(),
    srch.highlightSelectionMatches(),
    ac.autoComplete({filter: ac.fuzzyFilter}),
  );

  // Completions: the introspected /api/completions list (egg geometry API,
  // TopologyBuilder methods, PipelineConfig fields). Replaced/augmented by the
  // based-pyright LSP source in lsp.js. prism fuzzy-filters `options` against
  // the identifier under the cursor, so we hand back the full list anchored at
  // the start of that identifier's final segment.
  const eggItems = await fetch('/api/completions').then((r) => r.json()).catch(() => []);
  const items = eggItems.map((it) => ({label: it.label, icon: it.type, detail: it.info}));
  const listSource = (ctx) => {
    const m = /[A-Za-z_][\w]*$/.exec(ctx.lineBefore);
    if (!m && !ctx.explicit) return;
    return {from: ctx.pos - (m ? m[0].length : 0), options: items};
  };
  ac.registerCompletions(['python'], {sources: [listSource]});

  // Mirror edits into the server-visible <textarea name=code> and fire the
  // 'input' event that drives the 500ms HTMX re-render + localStorage.
  editor.on('update', (value) => {
    ta.value = value;
    ta.dispatchEvent(new Event('input', {bubbles: true}));
  });
  // Persist the cursor (head offset), restored on the next editor (re)load.
  editor.on('selectionChange', (sel) => {
    localStorage.setItem('egg-webui-cursor', String(sel[1]));
  });

  host.classList.add('pce-active');  // hides the now-mirrored textarea (app.css)

  // Reopen where the user left off (clamped in case the buffer shrank).
  const savedCur = Math.min(
    +(localStorage.getItem('egg-webui-cursor') || 0), editor.value.length);
  if (savedCur > 0) utils.setSelection(editor, savedCur);

  window.eggEditor = editor;
  // Small implementation-agnostic surface app.js drives (see its call sites):
  // getValue/setValue (minimal-diff replace) and gotoLine (error-chip jump).
  window.eggEditorApi = {
    getValue: () => editor.value,
    setValue: (code) => {
      const old = editor.value;
      if (old === code) return;
      // Replace only the differing middle so a param-panel edit maps the
      // cursor through a tiny change instead of snapping to the top.
      let a = 0;
      const n = Math.min(old.length, code.length);
      while (a < n && old.charCodeAt(a) === code.charCodeAt(a)) a++;
      let b = 0;
      while (b < n - a &&
             old.charCodeAt(old.length - 1 - b) === code.charCodeAt(code.length - 1 - b))
        b++;
      utils.insertText(editor, code.slice(a, code.length - b), a, old.length - b);
    },
    gotoLine: (line) => {
      const lines = editor.value.split('\n');
      const i = Math.max(1, Math.min(line, lines.length)) - 1;
      const from = lines.slice(0, i).reduce((s, l) => s + l.length + 1, 0);
      utils.setSelection(editor, from, from + (lines[i] || '').length);
      editor.textarea.focus();
    },
  };

  // ---- based-pyright language features over the /lsp websocket ----
  // completions (async LSP bridged into prism's synchronous source via a cache
  // + startQuery, with snippet support) and diagnostics (errors-only squiggle
  // overlay). Degrades silently: if the server has no language server,
  // egg/unavailable arrives and the static /api/completions source stays.
  (() => {
    let sock;
    try {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws';
      sock = new WebSocket(`${proto}://${location.host}/lsp`);
    } catch (e) {
      return;
    }
    let ready = false, docUri = null, version = 1, nextId = 0;
    const pending = new Map();
    const rpc = (method, params) => new Promise((resolve) => {
      const id = ++nextId;
      pending.set(id, resolve);
      sock.send(JSON.stringify({jsonrpc: '2.0', id, method, params}));
    });
    const notify = (method, params) =>
      sock.send(JSON.stringify({jsonrpc: '2.0', method, params}));

    // Push the current buffer to the server. Called debounced from edits, and
    // flushed synchronously right before a completion query so the server
    // never answers against a stale document.
    let lastSynced = null;
    const syncDoc = () => {
      if (!ready) return;
      const text = editor.value;
      if (text === lastSynced) return;
      lastSynced = text;
      version++;
      notify('textDocument/didChange',
        {textDocument: {uri: docUri, version}, contentChanges: [{text}]});
    };

    // LSP CompletionItemKind -> prism icon name.
    const KIND = {1: 'text', 2: 'function', 3: 'function', 4: 'function',
      5: 'property', 6: 'variable', 7: 'class', 8: 'interface', 9: 'namespace',
      10: 'property', 11: 'unit', 12: 'constant', 13: 'enum', 14: 'keyword',
      15: 'snippet', 16: 'constant', 17: 'text', 18: 'text', 19: 'namespace',
      20: 'enum', 21: 'constant', 22: 'class', 23: 'event', 24: 'keyword',
      25: 'parameter'};

    // Callable kinds that take a `(…)` call: method, function, constructor,
    // class. basedpyright/pyright never emit `f(…)` snippets themselves
    // (`completeFunctionParens` is a closed-source Pylance feature — absent from
    // the open-source server), so we synthesize the parens client-side, exactly
    // as Pylance does in-editor. See toPrism + the parensOk guards below.
    const CALLABLE_KINDS = new Set([2, 3, 4, 7]);
    // Contexts where appending `()` is wrong: an import/decorator reference or a
    // type annotation names the callable without calling it.
    const forbidsParens = (lineBefore, nextChar) =>
      nextChar === '(' ||                                   // already a call
      /^\s*(from\b.*\bimport\b|import\b)/.test(lineBefore) ||  // import target
      /^\s*@[\w.]*$/.test(lineBefore) ||                    // bare decorator ref
      /->\s*[\w.\[\], ]*$/.test(lineBefore) ||              // return annotation
      /^\s*[A-Za-z_]\w*\s*:\s*[\w.\[\], ]*$/.test(lineBefore) ||  // var annotation
      /\bdef\b.*\(.*:\s*[\w.\[\], ]*$/.test(lineBefore);    // def param annotation

    const offsetToLC = (value, off) => {
      let line = 0, last = 0;
      for (let i = 0; i < off; i++)
        if (value.charCodeAt(i) === 10) { line++; last = i + 1; }
      return {line, character: off - last};
    };

    // LSP snippet ($1, ${1:name}, $0) -> prism {insert, tabStops}. Tab stops
    // are byte offsets into the inserted text (even = start, odd = end).
    const fromSnippet = (text) => {
      const re = /\$(\d+)|\$\{(\d+):((?:[^\\}]|\\.)*)\}|\$\{(\d+)\}/g;
      let out = '', stops = [], last = 0, m;
      while ((m = re.exec(text))) {
        out += text.slice(last, m.index);
        const ph = (m[3] || '').replace(/\\(.)/g, '$1');
        stops.push(out.length, out.length + ph.length);
        out += ph;
        last = re.lastIndex;
      }
      out += text.slice(last).replace(/\\(.)/g, '$1');
      return {insert: out, tabStops: stops.length ? stops : undefined};
    };
    // Parse the parameter list out of pyright's resolved signature doc (a
    // ```python\ndef name(\n  p1: T,\n  kw: T = default\n) -> R\n``` block) into
    // a snippet `base(${1:p1}, kw=${2:default})$0` — required params become
    // placeholders, defaulted params become keyword args at their default.
    // Bracket-aware so nested types/defaults (dict[str, int], (1, 2)) don't split
    // wrong. Returns {insert, tabStops} (via fromSnippet), or null if no signature.
    const matchDelim = (s, open, chars) => {  // index of the delimiter closing s[open]
      let depth = 0;
      for (let i = open; i < s.length; i++) {
        if (chars.open.includes(s[i])) depth++;
        else if (chars.close.includes(s[i])) { if (--depth === 0) return i; }
      }
      return -1;
    };
    const BR = {open: '([{', close: ')]}'};
    const paramSnippet = (base, resolved) => {
      const doc = resolved && resolved.documentation;
      const md = doc && (doc.value != null ? doc.value : doc);
      if (typeof md !== 'string') return null;
      const cb = /```(?:python)?\n([\s\S]*?)```/.exec(md);
      const sig = cb ? cb[1] : md;
      const open = sig.indexOf('(');
      if (open < 0) return null;
      const close = matchDelim(sig, open, BR);
      if (close < 0) return null;
      const inner = sig.slice(open + 1, close);
      const toks = [];
      let depth = 0, start = 0;
      for (let i = 0; i <= inner.length; i++) {
        const c = inner[i];
        if (i === inner.length || (c === ',' && depth === 0)) {
          const t = inner.slice(start, i).trim();
          if (t) toks.push(t);
          start = i + 1;
        } else if (BR.open.includes(c)) depth++;
        else if (BR.close.includes(c)) depth--;
      }
      const params = [];
      for (let tok of toks) {
        tok = tok.replace(/\s+/g, ' ').trim();
        if (tok === '/' || tok === '*' || tok.startsWith('*')) continue;  // markers / *args
        const nm = /^([A-Za-z_]\w*)/.exec(tok);
        if (!nm || nm[1] === 'self' || nm[1] === 'cls') continue;
        let eq = -1, d = 0;
        for (let i = nm[1].length; i < tok.length; i++) {
          const c = tok[i];
          if (BR.open.includes(c)) d++;
          else if (BR.close.includes(c)) d--;
          else if (c === '=' && d === 0 && tok[i + 1] !== '=' &&
                   !'=!<>'.includes(tok[i - 1])) { eq = i; break; }
        }
        params.push(eq >= 0 ? {name: nm[1], def: tok.slice(eq + 1).trim()} : {name: nm[1]});
      }
      if (!params.length) return {insert: base + '()', tabStops: [base.length + 2]};
      const esc = (s) => s.replace(/\\/g, '\\\\').replace(/\}/g, '\\}');
      const body = params.map((p, i) => p.def != null
        ? `${p.name}=\${${i + 1}:${esc(p.def)}}` : `\${${i + 1}:${p.name}}`).join(', ');
      return fromSnippet(base + '(' + body + ')$0');
    };

    // LSP CompletionItem -> prism Completion (with snippet support). Callables
    // get a client-side `f(<cursor>)` (see CALLABLE_KINDS note) unless the server
    // already supplied a snippet; upgradeParams later fills the arg placeholders.
    const toPrism = (it, parensOk) => {
      const o = {label: it.label, icon: KIND[it.kind] || 'variable',
        detail: (it.detail || (it.labelDetails && it.labelDetails.description) || '').slice(0, 64)};
      const raw = (it.textEdit && it.textEdit.newText) || it.insertText;
      let server = false;
      if (raw) {
        if (it.insertTextFormat === 2) { Object.assign(o, fromSnippet(raw)); server = true; }
        else o.insert = raw;
      }
      if (parensOk && !server && CALLABLE_KINDS.has(it.kind)) {
        const base = o.insert != null ? o.insert : o.label;
        if (!base.includes('(')) {
          o.insert = base + '()';
          o.tabStops = [base.length + 1];  // cursor between the parens
          o._raw = it;                     // upgradeParams resolves this for params
        }
      }
      return o;
    };
    // Resolve callable completions and rewrite each option's insert to a full
    // arg/kwarg placeholder snippet, in place — accepting after the (local,
    // sub-100ms) resolve yields `f(${1:arg}, kw=${2:default})`; an instant accept
    // falls back to the `f()` from toPrism. Bounded so a huge list can't fan out.
    const resolveCache = new Map();
    const resolveItem = (raw) => {
      const key = raw.label + '\0' + (raw.kind || 0) + '\0' +
        (raw.data ? JSON.stringify(raw.data) : '');
      let p = resolveCache.get(key);
      if (!p) { p = rpc('completionItem/resolve', raw).then((r) => r && r.result); resolveCache.set(key, p); }
      return p;
    };
    const upgradeParams = (options) => {
      let budget = 60;
      for (const opt of options) {
        if (!opt._raw || budget <= 0) continue;
        // Classes: pyright renders a class completion's signature as `class
        // Name()` — it never expands the (dataclass-)synthesized __init__ fields,
        // so there is nothing to fill. Leave toPrism's `Name(<cursor>)` as-is
        // (cursor between the parens) rather than parsing zero params and jumping
        // the cursor past `)`. Constructor fields surface via signatureHelp.
        if (opt._raw.kind === 7) continue;
        budget--;
        const base = opt.label;
        resolveItem(opt._raw).then((res) => {
          const snip = paramSnippet(base, res);
          if (snip) { opt.insert = snip.insert; opt.tabStops = snip.tabStops; }
        });
      }
    };

    // Completions: a synchronous prism source backed by a cache. On a miss it
    // fires an async LSP query; when that resolves it fills the cache and
    // reopens the completion window (startQuery), so the same source now hits.
    let cache = null, reqPos = -1;
    const source = (ctx) => {
      if (cache && cache.pos === ctx.pos) return {from: cache.from, options: cache.options};
      if (ready) fetchCompletion(ctx);
      return undefined;
    };
    async function fetchCompletion(ctx) {
      if (ctx.pos === reqPos) return;
      reqPos = ctx.pos;
      syncDoc();  // flush pending edits so pyright sees the current buffer
      const r = await rpc('textDocument/completion', {
        textDocument: {uri: docUri},
        position: offsetToLC(editor.value, ctx.pos),
        context: {triggerKind: ctx.explicit ? 1 : 2},
      });
      reqPos = -1;
      const res = r && r.result;
      if (!res) return;
      const list = Array.isArray(res) ? res : (res.items || []);
      const m = /[A-Za-z_][\w]*$/.exec(ctx.lineBefore);
      const from = ctx.pos - (m ? m[0].length : 0);
      const parensOk = !forbidsParens(ctx.lineBefore, editor.value[ctx.pos] || '');
      cache = {pos: ctx.pos, from,
        options: list.slice(0, 200).map((it) => toPrism(it, parensOk))};
      if (parensOk) upgradeParams(cache.options);
      const q = editor.extensions.autoComplete;
      if (q && cache.options.length) q.startQuery(ctx.explicit);
    }

    // Diagnostics: a squiggle overlay in the editor's scrolled coordinate space
    // (monospace metrics -> per-line underline boxes).
    let diagnostics = [];
    const overlay = document.createElement('div');
    overlay.className = 'pce-diagnostics';
    overlay.style.cssText = 'position:absolute;top:0;left:0;pointer-events:none;z-index:1;';
    (utils.addOverlay || ((_e, el) => editor.wrapper.appendChild(el)))(editor, overlay);
    const metrics = () => {
      const line = editor.lines[1] || editor.wrapper;
      const cs = getComputedStyle(line);
      const probe = document.createElement('span');
      probe.textContent = '0'.repeat(10);
      probe.style.cssText = 'position:absolute;visibility:hidden;white-space:pre;font:' + cs.font;
      line.appendChild(probe);
      const cw = probe.getBoundingClientRect().width / 10;
      probe.remove();
      return {cw: cw || 7.2, padL: parseFloat(cs.paddingLeft) || 0};
    };
    function renderDiagnostics() {
      overlay.textContent = '';
      if (!diagnostics.length) return;
      const {cw, padL} = metrics();
      const lines = editor.value.split('\n');
      for (const d of diagnostics) {
        for (let ln = d.range.start.line; ln <= d.range.end.line; ln++) {
          const el = editor.lines[ln + 1];
          if (!el) continue;
          const text = lines[ln] || '';
          const sc = ln === d.range.start.line ? d.range.start.character : 0;
          const ec = ln === d.range.end.line ? d.range.end.character : text.length;
          const u = document.createElement('div');
          u.style.cssText = `position:absolute;left:${padL + sc * cw}px;top:${el.offsetTop}px;` +
            `height:${el.offsetHeight - 1}px;width:${Math.max(cw, (ec - sc) * cw)}px;` +
            `border-bottom:2px solid var(--ctp-red);box-sizing:border-box;`;
          overlay.appendChild(u);
        }
      }
    }
    // Reposition squiggles on any reflow that isn't an edit — font zoom
    // (--egg-edfont), pane resize (split.js), window resize — since those
    // change line offsets and the (monospace) character width.
    if (window.ResizeObserver) {
      let raf;
      new ResizeObserver(() => {
        cancelAnimationFrame(raf);
        raf = requestAnimationFrame(renderDiagnostics);
      }).observe(editor.wrapper);
    }

    // Diagnostic hover: show the error message(s) when the pointer is over a
    // squiggle. Purely local (no LSP round-trip), so it never gets stuck.
    const inRange = (r, p) =>
      (p.line > r.start.line || (p.line === r.start.line && p.character >= r.start.character)) &&
      (p.line < r.end.line || (p.line === r.end.line && p.character <= r.end.character));
    const tip = document.createElement('div');
    tip.className = 'pce-diag-tip';
    tip.style.cssText = 'position:fixed;z-index:30;display:none;max-width:520px;' +
      'padding:5px 8px;border-radius:6px;white-space:pre-wrap;pointer-events:none;' +
      'font:12px/1.5 ui-monospace,monospace;';
    document.body.appendChild(tip);
    const hideTip = () => { tip.style.display = 'none'; };
    // Map a viewport point to a line/col. prism overlays a transparent
    // <textarea> that owns pointer events, so find the line by its rect and the
    // column by (monospace) metrics rather than caretFromPoint.
    const locate = (x, y) => {
      const lineText = editor.value.split('\n');
      for (let i = 1; i < editor.lines.length; i++) {
        const r = editor.lines[i].getBoundingClientRect();
        if (y >= r.top && y < r.bottom) {
          const {cw, padL} = metrics();
          const col = Math.round((x - (r.left + padL)) / cw);
          return col < 0 ? null
            : {line: i - 1, character: Math.min(col, (lineText[i - 1] || '').length)};
        }
      }
      return null;
    };
    let hoverTimer;
    editor.container.addEventListener('mousemove', (e) => {
      clearTimeout(hoverTimer);
      const x = e.clientX, y = e.clientY;
      hoverTimer = setTimeout(() => {
        const lc = locate(x, y);
        const msgs = lc
          ? diagnostics.filter((d) => inRange(d.range, lc)).map((d) => d.message) : [];
        if (!msgs.length) { hideTip(); return; }
        tip.textContent = msgs.join('\n\n');
        tip.style.left = Math.min(x + 12, innerWidth - 530) + 'px';
        tip.style.top = (y + 16) + 'px';
        tip.style.display = 'block';
      }, 120);
    });
    editor.container.addEventListener('mouseleave', hideTip);

    // signatureHelp: a parameter-hint popup shown while the cursor sits inside a
    // call's parens. This is how constructor/dataclass fields surface — pyright
    // renders a class completion as `class Name()` (no params), but signatureHelp
    // inside `Name(…)` returns the full field list. Works for functions too.
    // Keyed off cursor position, so it needs no completion-accept hook.
    const sigTip = document.createElement('div');
    sigTip.className = 'pce-sig-tip';
    sigTip.style.cssText = 'position:fixed;z-index:29;display:none;max-width:640px;' +
      'padding:5px 9px;border-radius:6px;white-space:pre-wrap;' +
      'font:12px/1.5 ui-monospace,monospace;';
    document.body.appendChild(sigTip);
    const hideSig = () => { sigTip.style.display = 'none'; };
    // Cheap client-side gate: only ask the server when there's an unclosed '('
    // before the cursor (bounded look-back), so ordinary cursor moves don't spam
    // signatureHelp requests. False positives (grouping parens) just yield an
    // empty result and hide.
    const insideCall = (value, pos) => {
      let depth = 0;
      for (let i = pos - 1, n = Math.max(0, pos - 5000); i >= n; i--) {
        const c = value[i];
        if (c === ')' || c === ']' || c === '}') depth++;
        else if (c === '[' || c === '{') depth = Math.max(0, depth - 1);
        else if (c === '(') { if (depth) depth--; else return true; }
      }
      return false;
    };
    const renderSig = (sig, active) => {
      sigTip.textContent = '';
      const label = sig.label || '';
      const p = (sig.parameters || [])[active];
      let a = 0, b = 0;
      if (p) {
        if (Array.isArray(p.label)) [a, b] = p.label;
        else { const i = label.indexOf(p.label); if (i >= 0) { a = i; b = i + p.label.length; } }
      }
      const seg = (text, cls) => {
        const s = document.createElement('span');
        s.textContent = text;
        if (cls) s.className = cls;
        sigTip.appendChild(s);
      };
      if (b > a) { seg(label.slice(0, a)); seg(label.slice(a, b), 'pce-sig-active'); seg(label.slice(b)); }
      else seg(label);
    };
    const placeSig = (pos) => {
      const {line, character} = offsetToLC(editor.value, pos);
      const el = editor.lines[line + 1];
      if (!el) { hideSig(); return; }
      const r = el.getBoundingClientRect();
      const {cw, padL} = metrics();
      const x = r.left + padL + character * cw;
      sigTip.style.display = 'block';
      sigTip.style.left = Math.max(4, Math.min(x, innerWidth - sigTip.offsetWidth - 8)) + 'px';
      const above = r.top - sigTip.offsetHeight - 6;
      sigTip.style.top = (above >= 4 ? above : r.bottom + 6) + 'px';
    };
    let sigTimer, sigReq = -1;
    const updateSig = () => {
      if (!ready) { hideSig(); return; }
      const pos = editor.getSelection()[1];
      if (!insideCall(editor.value, pos)) { hideSig(); return; }
      sigReq = pos;
      syncDoc();  // pyright must see the current buffer
      rpc('textDocument/signatureHelp', {
        textDocument: {uri: docUri},
        position: offsetToLC(editor.value, pos),
        context: {triggerKind: 1, isRetrigger: false},
      }).then((r) => {
        if (editor.getSelection()[1] !== sigReq) return;  // cursor moved; stale
        const res = r && r.result;
        const sigs = res && res.signatures;
        if (!sigs || !sigs.length) { hideSig(); return; }
        const si = res.activeSignature || 0;
        const sig = sigs[si] || sigs[0];
        const active = sig.activeParameter != null ? sig.activeParameter
          : (res.activeParameter || 0);
        renderSig(sig, active);
        placeSig(pos);
      });
    };
    editor.on('selectionChange', () => {
      clearTimeout(sigTimer);
      sigTimer = setTimeout(updateSig, 120);
    });
    editor.textarea.addEventListener('blur', hideSig);
    editor.textarea.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') hideSig();
    });

    sock.onmessage = (ev) => {
      let m;
      try { m = JSON.parse(ev.data); } catch (e) { return; }
      if (m.method === 'egg/ready') { initLsp(m.params); return; }
      if (m.method === 'egg/unavailable' || m.method === 'egg/closed') { ready = false; return; }
      if (m.id != null && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); return; }
      if (m.method === 'textDocument/publishDiagnostics' && m.params.uri === docUri) {
        // Only surface errors (severity 1); warnings/info/hints are suppressed.
        diagnostics = (m.params.diagnostics || []).filter((d) => d.severity === 1);
        renderDiagnostics();
      }
    };
    sock.onclose = () => { ready = false; };

    // The LSP document is the open file at its on-disk path (#scriptpath, set by
    // app.js), so pyright resolves sibling imports (`from driver import ...`)
    // against the file's own directory. Falls back to the server's scratch
    // workspace for an unsaved/pathless buffer. NB: do NOT advertise the
    // workspaceFolders capability — with it, pyright waits on per-folder config
    // and never publishes diagnostics.
    let scratch = null;
    const pathToUri = (p) => 'file://' + encodeURI(p).replace(/#/g, '%23');
    const resolveUri = () => {
      const el = document.getElementById('scriptpath');
      const p = el && el.value;
      return p ? pathToUri(p) : scratch.docUri;
    };

    function openDoc() {
      lastSynced = editor.value;
      version++;
      notify('textDocument/didOpen', {textDocument:
        {uri: docUri, languageId: 'python', version, text: lastSynced}});
    }

    async function initLsp(p) {
      scratch = {docUri: p.docUri, rootUri: p.rootUri};
      docUri = resolveUri();
      await rpc('initialize', {
        processId: null,
        rootUri: p.rootUri,
        // configuration: true lets pyright pull our settings (interpreter path,
        // analysis mode). NOT workspaceFolders — that one makes pyright wait on
        // per-folder config and never publish diagnostics.
        capabilities: {
          workspace: {configuration: true},
          textDocument: {
            synchronization: {},
            publishDiagnostics: {},
            completion: {completionItem: {snippetSupport: true, labelDetailsSupport: true,
              documentationFormat: ['markdown', 'plaintext']}, contextSupport: true},
            signatureHelp: {signatureInformation: {
              parameterInformation: {labelOffsetSupport: true}, activeParameterSupport: true}},
          },
        },
      });
      notify('initialized', {});
      openDoc();
      ready = true;
      // LSP source first, static introspected list as fallback.
      ac.registerCompletions(['python'], {sources: [source, listSource]});
      let changeTimer;
      editor.on('update', () => {
        renderDiagnostics();  // keep existing squiggles aligned as text reflows
        clearTimeout(changeTimer);
        changeTimer = setTimeout(syncDoc, 250);  // push edits for fresh diagnostics
      });
      // Re-home the LSP on the new file when the open file changes (app.js
      // dispatches this from setScriptPath), so its directory resolves imports.
      window.addEventListener('egg-scriptpath', () => {
        if (!ready) return;
        const uri = resolveUri();
        if (uri === docUri) return;
        notify('textDocument/didClose', {textDocument: {uri: docUri}});
        docUri = uri;
        openDoc();
        cache = null;
        diagnostics = [];
        renderDiagnostics();
      });
    }
  })();
} catch (err) {
  console.warn('prism-code-editor unavailable, plain textarea fallback:', err);
}
