// Light/dark toggle for the Pallets "flask" theme.
// Sets data-theme on <html>; dark.css re-skins under html[data-theme="dark"].
// Choice persists in localStorage; without one, follows the OS preference.
(function () {
  const KEY = "egg-color-scheme";
  const root = document.documentElement;
  const media = window.matchMedia("(prefers-color-scheme: dark)");

  function preferred() {
    const saved = localStorage.getItem(KEY);
    if (saved === "light" || saved === "dark") return saved;
    return media.matches ? "dark" : "light";
  }

  function apply(mode) {
    root.setAttribute("data-theme", mode);
    const btn = document.getElementById("theme-toggle");
    if (btn) {
      btn.textContent = (mode === "dark" ? "☾" : "☀") + " theme";
      btn.setAttribute("aria-label", "Switch to " +
        (mode === "dark" ? "light" : "dark") + " mode");
    }
  }

  // Runs from <head>, before first paint — no flash of the wrong scheme.
  apply(preferred());

  media.addEventListener("change", function () {
    if (!localStorage.getItem(KEY)) apply(preferred());
  });

  document.addEventListener("DOMContentLoaded", function () {
    const btn = document.createElement("button");
    btn.id = "theme-toggle";
    btn.type = "button";
    btn.addEventListener("click", function () {
      const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
      localStorage.setItem(KEY, next);
      apply(next);
    });
    const side = document.querySelector("div.sphinxsidebarwrapper");
    if (side) side.insertBefore(btn, side.firstChild);
    else document.body.appendChild(btn);
    apply(root.getAttribute("data-theme") || preferred());
  });
})();
