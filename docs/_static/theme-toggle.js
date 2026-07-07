// Catppuccin flavor selector for the Pallets "flask" theme.
// Sets data-theme on <html>; theme.css re-skins under
// html[data-theme="<flavor>"]. The choice shares the egg web UI's
// localStorage key, so docs opened from the UI (same origin, /docs/)
// follow the flavor picked in its view menu — live, via the storage
// event. Standalone, it falls back to the OS color-scheme preference.
(function () {
  const KEY = "egg-webui-theme";
  const LEGACY = "egg-color-scheme"; // the pre-flavor light/dark toggle
  const FLAVORS = ["mocha", "macchiato", "frappe", "latte"];
  const root = document.documentElement;
  const media = window.matchMedia("(prefers-color-scheme: dark)");

  function preferred() {
    const saved = localStorage.getItem(KEY);
    if (FLAVORS.includes(saved)) return saved;
    const legacy = localStorage.getItem(LEGACY);
    if (legacy === "dark") return "mocha";
    if (legacy === "light") return "latte";
    return media.matches ? "mocha" : "latte";
  }

  function apply(flavor) {
    root.setAttribute("data-theme", flavor);
    const sel = document.getElementById("theme-select");
    if (sel) sel.value = flavor;
  }

  // Runs from <head>, before first paint — no flash of the wrong flavor.
  apply(preferred());

  media.addEventListener("change", function () {
    if (!localStorage.getItem(KEY)) apply(preferred());
  });

  // Flavor changed in another tab (e.g. the web UI's view menu).
  window.addEventListener("storage", function (e) {
    if (e.key === KEY) apply(preferred());
  });

  document.addEventListener("DOMContentLoaded", function () {
    const sel = document.createElement("select");
    sel.id = "theme-select";
    sel.setAttribute("aria-label", "color theme");
    for (const f of FLAVORS) {
      const o = document.createElement("option");
      o.value = f;
      o.textContent = f === "frappe" ? "frappé" : f;
      sel.appendChild(o);
    }
    sel.addEventListener("change", function () {
      localStorage.setItem(KEY, sel.value);
      apply(sel.value);
    });
    const side = document.querySelector("div.sphinxsidebarwrapper");
    const home = side && side.querySelector("h3.home-link");
    if (home) home.after(sel);
    else if (side) side.insertBefore(sel, side.firstChild);
    else document.body.appendChild(sel);
    apply(root.getAttribute("data-theme") || preferred());

    // Generated C++ names in the TOC: drop the egg:: prefix and
    // doxygen's spaced template brackets — the sidebar is cramped, the
    // page body keeps the full names.
    document.querySelectorAll("div.sphinxsidebar li > a").forEach(function (a) {
      const code = a.querySelector("code") || a;
      const t = code.textContent;
      const s = t.replace(/^egg::/, "").replace(/\s*([<>])\s*/g, "$1");
      if (s !== t) code.textContent = s;
    });
  });
})();
