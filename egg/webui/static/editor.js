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
     lineNumbers: true},
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
  // + startQuery), diagnostics (squiggle overlay), and hover (type tooltip).
  // Degrades silently: if the server has no language server, egg/unavailable
  // arrives and the static /api/completions source above stays in effect.
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
    // flushed synchronously right before a completion/hover query so the server
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

    const offsetToLC = (value, off) => {
      let line = 0, last = 0;
      for (let i = 0; i < off; i++)
        if (value.charCodeAt(i) === 10) { line++; last = i + 1; }
      return {line, character: off - last};
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
      cache = {pos: ctx.pos, from, options: list.slice(0, 200).map((it) => ({
        label: it.label,
        icon: KIND[it.kind] || 'variable',
        detail: (it.detail || (it.labelDetails && it.labelDetails.description) || '').slice(0, 64),
      }))};
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
    const inRange = (r, p) =>
      (p.line > r.start.line || (p.line === r.start.line && p.character >= r.start.character)) &&
      (p.line < r.end.line || (p.line === r.end.line && p.character <= r.end.character));

    // Hover: pointer -> line/col -> LSP hover, plus any diagnostic message here.
    const tip = document.createElement('div');
    tip.className = 'pce-hover-tip';
    tip.style.cssText = 'position:fixed;z-index:30;display:none;max-width:540px;max-height:340px;' +
      'overflow:auto;padding:6px 9px;border-radius:6px;white-space:pre-wrap;pointer-events:none;' +
      'font:12px/1.5 ui-monospace,monospace;';
    document.body.appendChild(tip);
    const hideTip = () => { tip.style.display = 'none'; };
    // Map a viewport point to an LSP position. prism lays a transparent
    // <textarea> over the content (it owns pointer events), so caretFromPoint
    // can't reach the tokens — instead locate the line by its rect and the
    // column by monospace metrics.
    const locate = (x, y) => {
      const lines = editor.lines;
      const lineText = editor.value.split('\n');
      for (let i = 1; i < lines.length; i++) {
        const r = lines[i].getBoundingClientRect();
        if (y >= r.top && y < r.bottom) {
          const {cw, padL} = metrics();
          const col = Math.round((x - (r.left + padL)) / cw);
          if (col < 0) return null;
          return {line: i - 1, character: Math.min(col, (lineText[i - 1] || '').length)};
        }
      }
      return null;
    };
    const hoverText = (c) => {
      const one = (x) => (typeof x === 'string' ? x : (x && x.value) || '');
      const t = Array.isArray(c) ? c.map(one).join('\n\n') : one(c);
      return t.replace(/```[\w-]*\n?/g, '').trim();
    };
    let hoverTimer;
    editor.container.addEventListener('mousemove', (e) => {
      clearTimeout(hoverTimer);
      if (!ready) { hideTip(); return; }
      const x = e.clientX, y = e.clientY;
      hoverTimer = setTimeout(async () => {
        const lc = locate(x, y);
        if (!lc) { hideTip(); return; }
        syncDoc();
        const dmsgs = diagnostics.filter((d) => inRange(d.range, lc)).map((d) => d.message);
        const r = await rpc('textDocument/hover', {textDocument: {uri: docUri}, position: lc});
        const hv = r && r.result;
        const body = [...dmsgs, hv && hv.contents ? hoverText(hv.contents) : '']
          .filter(Boolean).join('\n\n');
        if (!body) { hideTip(); return; }
        tip.textContent = body;
        tip.style.left = Math.min(x + 12, innerWidth - 550) + 'px';
        tip.style.top = (y + 16) + 'px';
        tip.style.display = 'block';
      }, 180);
    });
    editor.container.addEventListener('mouseleave', hideTip);

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

    async function initLsp(p) {
      docUri = p.docUri;
      await rpc('initialize', {
        processId: null,
        rootUri: p.rootUri,
        workspaceFolders: [{uri: p.rootUri, name: 'egg'}],
        capabilities: {textDocument: {
          synchronization: {},
          publishDiagnostics: {},
          completion: {completionItem: {labelDetailsSupport: true,
            documentationFormat: ['markdown', 'plaintext']}, contextSupport: true},
          hover: {contentFormat: ['markdown', 'plaintext']},
        }},
      });
      notify('initialized', {});
      lastSynced = editor.value;
      notify('textDocument/didOpen', {textDocument:
        {uri: docUri, languageId: 'python', version, text: lastSynced}});
      ready = true;
      // LSP source first, static introspected list as fallback.
      ac.registerCompletions(['python'], {sources: [source, listSource]});
      let changeTimer;
      editor.on('update', () => {
        renderDiagnostics();  // keep existing squiggles aligned as text reflows
        clearTimeout(changeTimer);
        changeTimer = setTimeout(syncDoc, 250);  // push edits for fresh diagnostics
      });
    }
  })();
} catch (err) {
  console.warn('prism-code-editor unavailable, plain textarea fallback:', err);
}
