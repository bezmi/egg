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
  // Importing the grammar registers 'python' into prism's language registry.
  await load('prism-python');

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
} catch (err) {
  console.warn('prism-code-editor unavailable, plain textarea fallback:', err);
}
