/* Shared navigation, injected into every page.
 *
 * One file so a new page gets the menu by adding one script tag, and so the Field / HQ split
 * stays consistent everywhere. Must run BEFORE a page's own inline script, because it creates
 * the #status pill that index.html writes into: load it as the first element inside <body>.
 *
 * Shared navigation and an explicit theme preference, defaulting to light.
 */
(function () {
  var theme = "light";
  try { if (localStorage.getItem("msf-theme") === "dark") theme = "dark"; } catch (e) {}
  document.documentElement.dataset.theme = theme;
  var I = {
    search: '<circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/>',
    sms: '<rect x="6" y="2" width="12" height="20" rx="2"/><path d="M11 18h2"/>',
    map: '<circle cx="7" cy="8" r="1.6"/><circle cx="16" cy="6" r="1.6"/><circle cx="12" cy="14" r="1.6"/><circle cx="18" cy="16" r="1.6"/><circle cx="6" cy="17" r="1.6"/>',
    hq: '<path d="M4 20V9l8-5 8 5v11"/><path d="M10 20v-6h4v6"/>',
    gaps: '<path d="M12 3v10M12 17v2"/><circle cx="12" cy="12" r="9"/>',
    radar: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4"/><path d="M12 3v18M3 12h18"/>',
    trail: '<path d="M6 4h9l4 4v12H6z"/><path d="M9 12h7M9 16h5"/>',
    cam: '<path d="M3 8h3l2-3h8l2 3h3v11H3z"/><circle cx="12" cy="13" r="3.4"/>'
  };
  var GROUPS = [
    {name: "Field", title: "Runs offline on the device in the clinic", items: [
      {href: "/", label: "Search", icon: I.search},
      {href: "/sms", label: "SMS line", icon: I.sms},
      {href: "/scan", label: "Scan a label", icon: I.cam},
      {href: "/galaxy", label: "Library map", icon: I.map}
    ]},
    {name: "HQ", title: "What the field sends back, and what HQ ships next", items: [
      {href: "/hq", label: "Dashboard", icon: I.hq},
      {href: "/gaps", label: "Gaps", icon: I.gaps},
      {href: "/radar", label: "Signal radar", icon: I.radar},
      {href: "/trail", label: "Evidence trail", icon: I.trail}
    ]}
  ];

  var path = location.pathname.replace(/\/+$/, "") || "/";
  var css = document.createElement("style");
  document.head.appendChild(css);

  function paint() {
  var c = {bg: "var(--card)", line: "var(--line)", ink: "var(--ink)", dim: "var(--mute)", red: "var(--red)", pillbg: "var(--surface)"};
  css.textContent = [
    "#msfnav{background:" + c.bg + ";border-bottom:1px solid " + c.line + ";position:sticky;top:0;z-index:50;font:15px/1.5 system-ui,-apple-system,'Segoe UI',Roboto,sans-serif}",
    "#msfnav *{box-sizing:border-box}",
    "#msfnav .mn-bar{max-width:980px;margin:0 auto;padding:9px 16px;display:flex;align-items:center;gap:10px;min-width:0}",
    "#msfnav .mn-logo{width:28px;height:28px;border-radius:7px;background:" + c.red + ";display:grid;place-items:center;color:#fff;font-weight:800;font-size:18px;flex:none}",
    "#msfnav .mn-t{font-size:16px;font-weight:650;color:" + c.ink + ";letter-spacing:-.01em;margin:0;min-width:0}",
    "#msfnav .pill{margin-left:auto;font-size:12.5px;color:" + c.dim + ";border:1px solid " + c.line + ";border-radius:999px;padding:5px 12px;white-space:nowrap;background:" + c.pillbg + "}",
    "#msfnav .pill b{color:#1f7a3f}",
    "#msfnav .pill.off{border-color:#1f7a3f;color:#1f7a3f;font-weight:600}",
    "#msfnav .mn-row{border-top:1px solid " + c.line + "}",
    "#msfnav .mn-in{max-width:980px;margin:0 auto;padding:0 10px;display:flex;align-items:stretch;gap:8px;flex-wrap:wrap;min-width:0}",
    "#msfnav .mn-g{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));align-items:stretch;gap:2px;flex:1 1 400px;min-width:0}",
    "#msfnav .mn-lab{display:none;font-size:10.5px;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:" + c.dim + ";padding:0 9px 0 4px;align-self:center;white-space:nowrap;opacity:.8}",
    "#msfnav .mn-sep{display:none;width:1px;background:" + c.line + ";margin:8px 10px}",
    "#msfnav a{min-width:0;display:flex;align-items:center;justify-content:center;gap:7px;padding:10px 6px;min-height:44px;text-decoration:none;color:" + c.dim + ";font-size:14px;font-weight:550;border-bottom:2.5px solid transparent;white-space:normal;text-align:center;line-height:1.25}",
    "#msfnav a:hover{color:" + c.ink + "}",
    "#msfnav a[aria-current='page']{color:" + c.red + ";border-bottom-color:" + c.red + "}",
    "#msfnav svg{width:17px;height:17px;flex:none}",
    "#msfnav .mn-theme{margin-left:auto;flex:none;display:flex;align-items:center;gap:8px;min-height:40px;padding:6px 9px;border:1px solid var(--line);border-radius:999px;background:var(--surface);color:var(--ink);cursor:pointer}",
    "#msfnav .mn-theme:focus-visible{outline:3px solid var(--accent);outline-offset:3px}",
    "#msfnav .mn-track{display:block;position:relative;width:34px;height:20px;border-radius:99px;background:var(--line);transition:background .2s}",
    "#msfnav .mn-knob{position:absolute;left:3px;top:3px;width:14px;height:14px;background:var(--card);border-radius:50%;box-shadow:0 1px 3px #0003;transition:transform .2s}",
    "#msfnav .mn-theme[aria-checked=true] .mn-track{background:var(--accent)}",
    "#msfnav .mn-theme[aria-checked=true] .mn-knob{transform:translateX(14px)}",
    "@media (max-width:560px){#msfnav .pill{display:none}#msfnav .mn-t{font-size:14px;flex:1}#msfnav .mn-bar{padding:8px 12px;gap:8px}#msfnav .mn-in{padding:0 6px;gap:0}#msfnav a{font-size:12px;gap:4px;padding:8px 4px}#msfnav .mn-g+.mn-g{border-top:1px solid var(--line)}}",
    "@media (prefers-reduced-motion:reduce){#msfnav .mn-track,#msfnav .mn-knob{transition:none}}"
  ].join("\n");
  }

  function link(it) {
    return '<a href="' + it.href + '"' + (it.href === path ? ' aria-current="page"' : '') + '>' +
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" ' +
      'stroke-linecap="round" stroke-linejoin="round">' + it.icon + '</svg>' + it.label + '</a>';
  }

  var html = '<div class="mn-bar"><div class="mn-logo" aria-hidden="true">+</div>' +
    '<p class="mn-t">MSF Field Library</p>' +
    '<span class="pill" id="status">local library</span>' +
    '<button type="button" class="mn-theme" role="switch" aria-label="Dark mode" aria-checked="' + (theme === "dark") + '" title="Switch light / dark theme">' +
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5"/></svg>' +
    '<span class="mn-track" aria-hidden="true"><span class="mn-knob"></span></span></button></div>' +
    '<div class="mn-row"><div class="mn-in">' +
    GROUPS.map(function (g, i) {
      return (i ? '<div class="mn-sep" aria-hidden="true"></div>' : '') +
        '<div class="mn-g" title="' + g.title + '"><span class="mn-lab">' + g.name + '</span>' +
        g.items.map(link).join("") + '</div>';
    }).join("") +
    '</div></div>';

  /* Pages without an explicit <body> tag keep a <script> in <head>, where document.body is
     still null. Insert as soon as the body exists, whichever is true now. */
  function insert() {
    var el = document.createElement("header");
    el.id = "msfnav";
    el.setAttribute("aria-label", "Sections");
    el.innerHTML = html;
    document.body.insertBefore(el, document.body.firstChild);
    paint();
    var toggle = el.querySelector(".mn-theme");
    toggle.addEventListener("click", function () {
      theme = theme === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = theme;
      toggle.setAttribute("aria-checked", String(theme === "dark"));
      try { localStorage.setItem("msf-theme", theme); } catch (e) {}
    });
  }
  if (document.body) insert();
  else document.addEventListener("DOMContentLoaded", insert);

  /* The search page owns #status and keeps it live. Elsewhere, fill it once. */
  if (path !== "/") {
    fetch("/api/info").then(function (r) { return r.json(); }).then(function (info) {
      var p = document.getElementById("status");
      if (!p || !info || !info.n_docs) return;
      var lib = info.n_docs + " docs, " + info.index_mb + " MB";
      p.classList.toggle("off", !navigator.onLine);
      p.innerHTML = "<b>●</b> " + (navigator.onLine ? "Online · local library (" : "Offline: local library (") + lib + ")";
    }).catch(function () {});
  }
})();
