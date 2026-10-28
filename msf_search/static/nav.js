/* Shared navigation, injected into every page.
 *
 * One file so a new page gets the menu by adding one script tag, and so the Field / HQ split
 * stays consistent everywhere. Must run BEFORE a page's own inline script, because it creates
 * the #status pill that index.html writes into: load it as the first element inside <body>.
 *
 * Theming: the search page follows the system light/dark setting, while the feature pages are
 * always dark. Rather than force one on the other, the bar measures the page's own background
 * luminance and matches it.
 */
(function () {
  var I = {
    search: '<circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/>',
    sms: '<rect x="6" y="2" width="12" height="20" rx="2"/><path d="M11 18h2"/>',
    map: '<circle cx="7" cy="8" r="1.6"/><circle cx="16" cy="6" r="1.6"/><circle cx="12" cy="14" r="1.6"/><circle cx="18" cy="16" r="1.6"/><circle cx="6" cy="17" r="1.6"/>',
    hq: '<path d="M4 20V9l8-5 8 5v11"/><path d="M10 20v-6h4v6"/>',
    gaps: '<path d="M12 3v10M12 17v2"/><circle cx="12" cy="12" r="9"/>',
    radar: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4"/><path d="M12 3v18M3 12h18"/>',
    trail: '<path d="M6 4h9l4 4v12H6z"/><path d="M9 12h7M9 16h5"/>'
  };
  var GROUPS = [
    {name: "Field", title: "Runs offline on the device in the clinic", items: [
      {href: "/", label: "Search", icon: I.search},
      {href: "/sms", label: "SMS line", icon: I.sms},
      {href: "/galaxy", label: "Library map", icon: I.map}
    ]},
    {name: "HQ", title: "What the field sends back, and what HQ ships next", items: [
      {href: "/hq", label: "Dashboard", icon: I.hq},
      {href: "/gaps", label: "Gaps", icon: I.gaps},
      {href: "/radar", label: "Signal radar", icon: I.radar},
      {href: "/trail", label: "Evidence trail", icon: I.trail}
    ]}
  ];

  function isDark() {
    try {
      if (!document.body) return matchMedia("(prefers-color-scheme: dark)").matches;
      var bg = getComputedStyle(document.body).backgroundColor || "";
      var m = bg.match(/\d+(\.\d+)?/g);
      if (!m || m.length < 3) return matchMedia("(prefers-color-scheme: dark)").matches;
      var l = (0.2126 * m[0] + 0.7152 * m[1] + 0.0722 * m[2]) / 255;
      return l < 0.5;
    } catch (e) { return false; }
  }

  var path = location.pathname.replace(/\/+$/, "") || "/";
  var dark = isDark();
  var c = dark
    ? {bg: "#12161d", line: "#2a3340", ink: "#e6edf3", dim: "#8b949e", red: "#ff4d5e", pillbg: "#0d1117"}
    : {bg: "#ffffff", line: "#e3e1dd", ink: "#1d1d1f", dim: "#6b6b70", red: "#e2001a", pillbg: "#f6f5f3"};

  var css = document.createElement("style");
  css.textContent = [
    "#msfnav{background:" + c.bg + ";border-bottom:1px solid " + c.line + ";position:sticky;top:0;z-index:50;font:15px/1.5 system-ui,-apple-system,'Segoe UI',Roboto,sans-serif}",
    "#msfnav *{box-sizing:border-box}",
    "#msfnav .mn-bar{max-width:980px;margin:0 auto;padding:9px 16px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}",
    "#msfnav .mn-logo{width:28px;height:28px;border-radius:7px;background:" + c.red + ";display:grid;place-items:center;color:#fff;font-weight:800;font-size:18px;flex:none}",
    "#msfnav .mn-t{font-size:16px;font-weight:650;color:" + c.ink + ";letter-spacing:-.01em;margin:0}",
    "#msfnav .pill{margin-left:auto;font-size:12.5px;color:" + c.dim + ";border:1px solid " + c.line + ";border-radius:999px;padding:5px 12px;white-space:nowrap;background:" + c.pillbg + "}",
    "#msfnav .pill b{color:#1f7a3f}",
    "#msfnav .pill.off{border-color:#1f7a3f;color:#1f7a3f;font-weight:600}",
    "#msfnav .mn-row{border-top:1px solid " + c.line + "}",
    "#msfnav .mn-in{max-width:980px;margin:0 auto;padding:0 10px;display:flex;align-items:stretch;gap:2px;overflow-x:auto}",
    "#msfnav .mn-g{display:flex;align-items:center;gap:2px}",
    "#msfnav .mn-lab{font-size:10.5px;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:" + c.dim + ";padding:0 9px 0 4px;align-self:center;white-space:nowrap;opacity:.8}",
    "#msfnav .mn-sep{width:1px;background:" + c.line + ";margin:8px 10px}",
    "#msfnav a{flex:none;display:flex;align-items:center;gap:7px;padding:10px 12px;min-height:44px;text-decoration:none;color:" + c.dim + ";font-size:14px;font-weight:550;border-bottom:2.5px solid transparent;white-space:nowrap}",
    "#msfnav a:hover{color:" + c.ink + "}",
    "#msfnav a[aria-current='page']{color:" + c.red + ";border-bottom-color:" + c.red + "}",
    "#msfnav svg{width:17px;height:17px;flex:none}",
    "@media (max-width:560px){#msfnav .mn-lab{display:none}#msfnav .mn-sep{margin:8px 6px}}"
  ].join("\n");
  document.head.appendChild(css);

  function link(it) {
    return '<a href="' + it.href + '"' + (it.href === path ? ' aria-current="page"' : '') + '>' +
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" ' +
      'stroke-linecap="round" stroke-linejoin="round">' + it.icon + '</svg>' + it.label + '</a>';
  }

  var html = '<div class="mn-bar"><div class="mn-logo" aria-hidden="true">+</div>' +
    '<p class="mn-t">MSF Field Library</p>' +
    '<span class="pill" id="status">local library</span></div>' +
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
