// Light / dark toggle shared by every page. The OS setting is the default; a click stores the choice for this browser.
(function () {
  const KEY = "pf-theme";
  const root = document.documentElement;
  const read = () => { try { return localStorage.getItem(KEY); } catch (e) { return null; } };
  const saved = read();
  if (saved === "light" || saved === "dark") root.dataset.theme = saved;
  const current = () => root.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  document.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("theme");
    if (!btn) return;
    const label = () => { btn.textContent = current() === "dark" ? "◐ light" : "◑ dark"; };
    label();
    btn.addEventListener("click", () => {
      root.dataset.theme = current() === "dark" ? "light" : "dark";
      try { localStorage.setItem(KEY, root.dataset.theme); } catch (e) { /* storage blocked: the toggle still works for this page */ }
      label();
    });
  });
})();
