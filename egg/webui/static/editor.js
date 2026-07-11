// Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
//
// PolyForm Noncommercial License 2.0.0-pre.2
// https://github.com/bezmi/egg/blob/main/LICENSE.md
// Free to use and redistribute for personal and noncommercial purposes.
// See the license for details.
// For commercial licensing, contact s.imran@tuta.io


try {
  // Prefer the locally vendored modules (webui/vendor.py); fall back to
  // the CDN with identical pins. Keep the pins in lockstep with vendor.py:
  // plain codemirror@6 resolves to a legacy 6.65.x (CM5-era) npm publish.
  const KEYS = ['codemirror', 'view', 'lang-python', 'autocomplete', 'commands',
                'state', 'language', 'lezer-highlight'];
  const CDN = [
    'https://esm.sh/codemirror@6.0.2',
    'https://esm.sh/@codemirror/view@6.43.4',
    'https://esm.sh/@codemirror/lang-python@6.2.1',
    'https://esm.sh/@codemirror/autocomplete@6.20.3',
    'https://esm.sh/@codemirror/commands@6.10.4',
    'https://esm.sh/@codemirror/state@6.7.0',
    'https://esm.sh/@codemirror/language@6.12.4',
    'https://esm.sh/@lezer/highlight@1.2.3',
  ];
  let mods = null;
  try {
    const man = await (await fetch('/vendor/manifest.json')).json();
    mods = await Promise.all(KEYS.map((k) => import('/vendor/' + man[k])));
  } catch (err) {
    console.warn('vendored editor modules unavailable, using CDN:', err);
    mods = await Promise.all(CDN.map((u) => import(u)));
  }
  const [cm, vw, lp, ac, cmds, st, lang, lz] = mods;
  const ta = document.querySelector('.editor textarea');
  const eggItems = await fetch('/api/completions').then(r => r.json()).catch(() => []);
  // Editor theme built from the live --ctp-* custom properties, so the
  // catppuccin flavor picked in the view menu styles the code too; a
  // Compartment lets a flavor switch reconfigure without losing state.
  const themeComp = new st.Compartment();
  function cmTheme() {
    const cs = getComputedStyle(document.documentElement);
    const v = (n) => cs.getPropertyValue('--ctp-' + n).trim();
    const t = lz.tags;
    return [
      vw.EditorView.theme({
        '&': {backgroundColor: v('base'), color: v('text')},
        '.cm-content': {caretColor: v('rosewater')},
        '.cm-cursor, .cm-dropCursor': {borderLeftColor: v('rosewater')},
        '.cm-activeLine': {backgroundColor: v('surface0') + '55'},
        '.cm-activeLineGutter': {backgroundColor: v('surface0')},
        '&.cm-focused > .cm-scroller > .cm-selectionLayer .cm-selectionBackground, .cm-selectionBackground, ::selection':
          {backgroundColor: v('overlay2') + '4d'},
        '.cm-selectionMatch': {backgroundColor: v('surface1')},
        '&.cm-focused .cm-matchingBracket': {backgroundColor: v('surface1')},
        '.cm-gutters': {backgroundColor: v('mantle'), color: v('overlay0'),
                        borderRight: '1px solid ' + v('surface0')},
        '.cm-tooltip': {backgroundColor: v('mantle'), color: v('text'),
                        border: '1px solid ' + v('surface1')},
        '.cm-tooltip-autocomplete ul li[aria-selected]':
          {backgroundColor: v('surface1'), color: v('text')},
        '.cm-panels': {backgroundColor: v('mantle'), color: v('text')},
        '.cm-searchMatch': {backgroundColor: v('yellow') + '44'},
      }, {dark: document.documentElement.dataset.theme !== 'latte'}),
      lang.syntaxHighlighting(lang.HighlightStyle.define([
        {tag: t.keyword, color: v('mauve')},
        {tag: [t.string, t.special(t.string)], color: v('green')},
        {tag: [t.number, t.bool, t.atom], color: v('peach')},
        {tag: [t.function(t.variableName), t.function(t.propertyName)],
         color: v('blue')},
        {tag: [t.className, t.typeName], color: v('yellow')},
        {tag: [t.operator, t.operatorKeyword], color: v('sky')},
        {tag: [t.comment, t.meta], color: v('overlay0'), fontStyle: 'italic'},
        {tag: [t.propertyName, t.attributeName], color: v('lavender')},
        {tag: t.self, color: v('red')},
        {tag: t.invalid, color: v('red')},
      ])),
    ];
  }
  // Reopen where the user left off (persisted by the updateListener below;
  // clamped in case the buffer shrank since).
  const savedCur = Math.min(
      +(localStorage.getItem('egg-webui-cursor') || 0), ta.value.length);
  const view = new cm.EditorView({
    doc: ta.value,
    selection: {anchor: savedCur},
    extensions: [
      cm.basicSetup,
      vw.keymap.of([cmds.indentWithTab]),
      lp.python(),
      lp.pythonLanguage.data.of({autocomplete: ac.completeFromList(eggItems)}),
      themeComp.of(cmTheme()),
      cm.EditorView.updateListener.of((u) => {
        if (u.docChanged) {
          ta.value = u.state.doc.toString();
          ta.dispatchEvent(new Event('input', {bubbles: true}));
        }
        // Last cursor position, restored on the next editor (re)load.
        if (u.docChanged || u.selectionSet)
          localStorage.setItem('egg-webui-cursor',
                               String(u.state.selection.main.head));
      }),
    ],
  });
  window.addEventListener('egg-theme', () => {
    view.dispatch({effects: themeComp.reconfigure(cmTheme())});
  });
  ta.parentElement.appendChild(view.dom);
  ta.parentElement.classList.add('cm-active');
  if (savedCur > 0)
    view.dispatch({effects: cm.EditorView.scrollIntoView(savedCur, {y: 'center'})});
  window.eggEditor = view;
} catch (err) {
  console.warn('CodeMirror unavailable, plain textarea fallback:', err);
}
