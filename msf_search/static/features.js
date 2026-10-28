/* MSF Field Library: UI rendering helpers (no framework, no CDN).
 *
 * PURE FUNCTIONS (take API JSON, return an HTML string). The GUI can call or restyle these:
 *   renderQuickAnswer(r)          r = /api/search response
 *   renderProvenance(x)           x = one result; labels ALL machine-made / OCR text
 *   renderImportance(x)           importance badge + "Actionable" chip
 *   renderSummary(x)              labelled AI summary
 *   renderEvidence(x)             the ORIGINAL passage behind a machine-made hit
 *   renderTopicChips(topics)      topics = /api/topics response .topics
 *   renderProfilePicker(profiles) profiles = /api/profiles response .profiles
 *
 * SAFETY RULE: machine-made text is never shown without its label, and always with its source passage.
 * The glue at the bottom wires these into the current index.html; a new GUI can drop the glue.
 */
(function () {
  const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
  const LABEL = {
    summary: "AI summary (made at HQ) · check the source",
    qa: "AI-prepared answer (made at HQ from this document) · check the source",
    qvec: "matched via AI-generated question",
  };
  const ENGINE = {ocr_tesseract: "tesseract OCR", ocr_vlm: "vision AI"};
  const IMPORTANCE = {1: "Essential field reference", 2: "Useful", 3: "Background", 4: "Archive"};

  function originalHref(x, page) {
    if (!x || !x.original_name) return "";
    const pdf = /\.pdf$/i.test(x.original_name);
    return `/original/${encodeURIComponent(x.original_name)}${pdf && page ? "#page=" + page : ""}`;
  }
  function viewLink(x, page) {
    const h = originalHref(x, page);
    if (!h) return "";
    const pdf = /\.pdf$/i.test(x.original_name);
    return `<a class="fx-link" href="${h}" target="_blank">${pdf ? "View PDF" + (page ? " p." + page : "") : "View original"}</a>`;
  }

  function renderProvenance(x) {
    const p = x.provenance || {}, out = [];
    if (x.ocr_status === "photo_only_queued") {
      out.push(`<span class="fx-b fx-ai">Photo only: handwriting unreadable by OCR · queued for HQ transcription ${viewLink(x, x.page)}</span>`);
      return out.join("");
    }
    if (p.text_source && p.text_source !== "parsed") {
      const conf = p.ocr_confidence != null ? `, confidence ${Number(p.ocr_confidence).toFixed(2)}` : "";
      out.push(`<span class="fx-b fx-ocr">Transcribed from scan/handwriting (${esc(ENGINE[p.text_source] || p.text_source)}${conf}) · check the original ${viewLink(x, x.page)}</span>`);
      if (p.ocr_confidence != null && p.ocr_confidence < 0.6)
        out.push(`<span class="fx-b fx-ai">Low-confidence transcription: check the original ${viewLink(x, x.page)}</span>`);
    }
    if (p.machine_generated && LABEL[p.generator]) out.push(`<span class="fx-b fx-ai">${esc(LABEL[p.generator])}</span>`);
    if (x.matched_via === "qvec") out.push(`<span class="fx-b fx-ai">${LABEL.qvec}</span>`);
    return out.join("");
  }

  let SHOW_IMPORTANCE = true;  // off while every enriched doc has the same importance (badge would say nothing)
  function renderImportance(x) {
    const out = [];
    if (x.importance && SHOW_IMPORTANCE) out.push(`<span class="fx-b fx-imp${x.importance}" title="${esc(x.importance_reason || "")}">${IMPORTANCE[x.importance] || ""}</span>`);
    if (x.actionable) out.push(`<span class="fx-b fx-act">Actionable</span>`);
    return out.join("");
  }

  function renderSummary(x) {
    if (!x.summary) return "";
    return `<div class="fx-sum"><div class="fx-lab">${LABEL.summary}</div><div dir="auto">${esc(x.summary)}</div></div>`;
  }

  function renderEvidence(x) {
    const e = x.evidence;
    if (!e) return "";
    return `<div class="fx-ev"><div class="fx-lab">Source passage${e.heading ? " · " + esc(e.heading) : ""} ${viewLink(x, e.page)}</div>
      <div dir="auto">${e.snippet}</div></div>`;
  }

  function renderQuickAnswer(r) {
    const a = r && r.quick_answer;
    if (!a) return "";
    const ev = a.evidence || {};
    return `<div class="fx-qa">
      <div class="fx-lab">${esc(a.label || LABEL.qa)}</div>
      <div class="fx-q" dir="auto">${esc(a.question)}</div>
      <div class="fx-a" dir="auto">${esc(a.answer)}</div>
      <blockquote dir="auto">“${esc(a.evidence_quote)}”</blockquote>
      <div class="fx-src">From <b dir="auto">${esc(a.title || a.doc_id)}</b> (${esc(a.doc_id)}) ${viewLink(a, ev.page)}</div>
    </div>`;
  }

  function renderTopicChips(topics, active) {
    if (!topics || !topics.length) return "";
    return `<div class="fx-topics">${topics.map(t =>
      `<button type="button" class="fx-chip${t.topic === active ? " on" : ""}" data-topic="${esc(t.topic)}">${esc(t.topic)} <span>${t.count}</span></button>`).join("")}</div>`;
  }

  function renderProfilePicker(profiles, active) {
    if (!profiles || !profiles.length) return "";
    return `<label class="fx-prof">Field profile <select id="fx-profile"><option value="">None</option>${profiles.map(p =>
      `<option value="${esc(p.id)}"${p.id === active ? " selected" : ""}>${esc(p.label)}</option>`).join("")}</select></label>`;
  }

  _fetchInfo();
  function _fetchInfo() {
    window.fetch("/api/info").then(r => r.json()).then(i => {
      SHOW_IMPORTANCE = Object.keys(i.importance_distribution || {}).length > 1;
    }).catch(() => {});
  }
  window.MSFFeatures = {renderQuickAnswer, renderProvenance, renderImportance, renderSummary, renderEvidence,
                        renderTopicChips, renderProfilePicker, originalHref};
  Object.assign(window, {renderQuickAnswer, renderProvenance, renderImportance, renderTopicChips, renderProfilePicker});

  /* ---------- styles (theme tokens from index.html) ---------- */
  const css = `.fx-b{font-size:11.5px;padding:2px 8px;border-radius:6px;background:var(--low-soft);color:var(--mute)}
  .fx-ocr{background:var(--mid-soft);color:var(--mid);font-weight:600}.fx-ai{background:var(--red-soft);color:var(--red);font-weight:600}
  .fx-imp1{background:var(--ok-soft);color:var(--ok);font-weight:700}.fx-imp2{color:var(--ink);font-weight:600}.fx-act{background:var(--ink);color:var(--bg)}
  .fx-link{color:inherit;text-decoration:underline;margin-left:4px}
  .fx-lab{font-size:12px;font-weight:600;color:var(--red);margin-bottom:3px}
  .fx-sum,.fx-ev{border-left:3px solid var(--line);padding:4px 0 4px 10px;margin:8px 0;font-size:14px}
  .fx-ev .fx-lab{color:var(--mute)}
  .fx-qa{background:var(--card);border:2px solid var(--red);border-radius:12px;padding:14px 16px;margin:10px 0}
  .fx-q{font-weight:700;margin:4px 0}.fx-a{font-size:16px;margin-bottom:8px}
  .fx-qa blockquote{margin:6px 0;padding:6px 10px;border-left:3px solid var(--line);color:var(--mute);font-size:14px}
  .fx-src{font-size:13px;color:var(--mute)}
  .fx-topics{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}.fx-chip{padding:3px 10px;border-radius:999px;font-size:13px}
  .fx-chip span{color:var(--mute);font-size:12px}.fx-chip.on{background:var(--ink);color:var(--bg)}
  .fx-prof{font-size:13px;color:var(--mute);display:flex;gap:6px;align-items:center}
  .fx-boost{font-size:13px;color:var(--mute);margin:6px 2px}.fx-boost a{color:inherit}
  .pill.off{background:var(--low-soft)!important;color:var(--mute)!important;border-color:var(--line)!important}
  .pill.off b{color:var(--mute)!important}`;  /* offline = grey: still fully working, just no network */
  const st = document.createElement("style"); st.textContent = css; document.head.appendChild(st);

  /* ---------- glue for the current index.html (a new GUI can ignore everything below) ---------- */
  let LAST = null, LAST_RESULTS = {};
  const _fetch = window.fetch.bind(window);
  window.fetch = async function (url, opts) {
    const res = await _fetch(url, opts);
    if (typeof url === "string" && url.startsWith("/api/search")) {
      try {  // read before the page renders, so the quick answer can be placed with the results
        const r = await res.clone().json();
        LAST = r;
        LAST_RESULTS = {};
        [...(r.results || []), ...(r.weaker_matches || [])].forEach(x => { LAST_RESULTS[x.doc_id] = x; });
      } catch (e) {}
    }
    return res;
  };

  if (typeof window.card === "function") {
    const baseCard = window.card;
    window.card = function (x, weak) {
      const t = document.createElement("template");
      t.innerHTML = baseCard(x, weak).trim();
      const el = t.content.firstElementChild;
      if (!el) return baseCard(x, weak);
      el.querySelectorAll(".b.scan").forEach(b => b.remove());  // replaced by the fuller provenance label
      const badges = el.querySelector(".badges");
      if (badges) badges.insertAdjacentHTML("beforeend", renderImportance(x) + renderProvenance(x));
      const h3 = el.querySelector("h3");
      if (h3) h3.insertAdjacentHTML("afterend", renderSummary(x));
      const snip = el.querySelector(".snip");
      if (snip && x.ocr_status === "photo_only_queued") snip.textContent = "(no reliable text: open the photo)";
      if (snip && x.matched_via && x.matched_via !== "text") {
        // label sits right above the machine-made text, so the duplicate badge goes
        el.querySelectorAll(".badges .fx-ai").forEach(b => { if (b.textContent === LABEL[x.matched_via]) b.remove(); });
        snip.insertAdjacentHTML("beforebegin", `<div class="fx-lab">${esc(LABEL[x.matched_via] || "")}</div>`);
        snip.insertAdjacentHTML("afterend", renderEvidence(x));
      }
      return el.outerHTML;
    };
  }

  const out = document.getElementById("out");
  if (out) {
    new MutationObserver(() => {
      if (!LAST || out.querySelector(".fx-qa")) return;
      const html = renderQuickAnswer(LAST);
      if (html) out.insertAdjacentHTML("afterbegin", html);
    }).observe(out, {childList: true});
  }
  window.MSFFeatures.lastSearch = () => LAST;

  /* topic chips: browse a topic (empty search box) or narrow the current query */
  let TOPIC = "", PROFILE = "";
  try { PROFILE = localStorage.getItem("msf-profile") || ""; } catch (e) {}
  const _fetch2 = window.fetch;
  window.fetch = function (url, opts) {
    if (typeof url === "string" && url.startsWith("/api/search")) {
      if (TOPIC && !url.includes("topic=")) url += "&topic=" + encodeURIComponent(TOPIC);
      if (PROFILE && !url.includes("profile=")) url += "&profile=" + encodeURIComponent(PROFILE);
    }
    return _fetch2(url, opts);
  };
  /* field profile: boosts, never filters; one click turns it off */
  const filters = document.querySelectorAll(".filters")[1];
  if (filters) {
    _fetch("/api/profiles").then(r => r.json()).then(({profiles}) => {
      const box = document.createElement("span");
      box.innerHTML = renderProfilePicker(profiles, PROFILE);
      filters.insertBefore(box, filters.lastElementChild);
      box.addEventListener("change", e => {
        PROFILE = e.target.value;
        try { localStorage.setItem("msf-profile", PROFILE); } catch (err) {}
        const q = document.getElementById("q");
        if (q && q.value.trim()) document.getElementById("f").requestSubmit();
      });
    }).catch(() => {});
    new MutationObserver(() => {
      const r = LAST;
      if (!out || out.querySelector(".fx-boost") || !r || !r.profile || !r.profile.boosted) return;
      out.insertAdjacentHTML("afterbegin", `<div class="fx-boost">Boosted for your profile: ${esc(r.profile.label)} (${r.profile.boosted} results) · <a href="#" id="fx-boost-off">turn off</a></div>`);
      document.getElementById("fx-boost-off").onclick = ev => {
        ev.preventDefault(); PROFILE = "";
        try { localStorage.removeItem("msf-profile"); } catch (err) {}
        const sel = document.getElementById("fx-profile"); if (sel) sel.value = "";
        document.getElementById("f").requestSubmit();
      };
    }).observe(out, {childList: true});
  }
  async function browseTopic() {
    const r = await (await window.fetch("/api/search?q=&k=20")).json();
    out.innerHTML = `<div class="meta">${r.results.length} documents · topic <b>${esc(TOPIC)}</b> · ordered by importance, evidence tier, aboutness · ${r.ms} ms</div>`
      + r.results.map(x => window.card(x)).join("");
  }
  const langs = document.getElementById("langs");
  if (langs) {
    _fetch("/api/topics").then(r => r.json()).then(({topics}) => {
      const box = document.createElement("div");
      box.id = "fx-topic-box";
      const draw = () => { box.innerHTML = renderTopicChips(topics.slice(0, 24), TOPIC); };
      const fromUrl = new URLSearchParams(location.search).get("topic");
      if (fromUrl) { TOPIC = fromUrl; if (!new URLSearchParams(location.search).get("q")) browseTopic(); }
      draw();
      langs.insertAdjacentElement("afterend", box);
      box.addEventListener("click", e => {
        const b = e.target.closest("[data-topic]");
        if (!b) return;
        TOPIC = TOPIC === b.dataset.topic ? "" : b.dataset.topic;
        draw();
        const q = document.getElementById("q");
        if (q && q.value.trim()) document.getElementById("f").requestSubmit();
        else if (TOPIC) browseTopic();
        else out.innerHTML = "";
      });
    }).catch(() => {});
  }

  /* ---------- on-device chat (POST /api/chat, SSE stream) ---------- */
  function renderSourceChips(sources) {
    return (sources || []).map(s => {
      const pdf = originalHref(s, s.page);
      return `<button type="button" class="fx-chip" onclick="openDoc('${esc(s.doc_id)}')" title="${esc(s.title)}">[${s.n}] ${esc(s.title).slice(0, 38)}</button>`
        + (pdf && /\.pdf/i.test(s.original_name || "") ? `<a class="fx-link" href="${pdf}" target="_blank">View PDF${s.page ? " p." + s.page : ""}</a>` : "");
    }).join(" ");
  }
  window.MSFFeatures.renderSourceChips = renderSourceChips;
  const LANGN = {en: "English", fr: "Français", es: "Español", ar: "العربية"};
  function renderSourceCards(sources, mid) {
    return `<ol class="fx-cards">${(sources || []).map(s => `<li id="fx-src-${mid}-${s.n}" class="fx-card" data-doc="${esc(s.doc_id)}"
      data-chunk="${s.chunk_id}" data-href="${esc(/\.pdf$/i.test(s.original_name || "") ? originalHref(s, s.page) : "")}" tabindex="0">
      <div><b>[${s.n}]</b> <span dir="auto">${esc(s.display_title || s.title)}</span></div>
      <div class="fx-cardmeta"><span class="fx-b">${esc(s.type || s.genre || "")}</span> ${esc(LANGN[s.language] || s.language || "")}${s.year ? " · " + s.year : ""}${s.location ? " · " + esc(s.location) : ""}
        <span class="fx-id">${esc(s.doc_id)}</span></div>
      <div class="fx-quote" dir="auto">“${esc(s.quote || "")}”</div></li>`).join("")}</ol>`;
  }
  function linkCitations(text, sources, mid) {
    const by = Object.fromEntries((sources || []).map(s => [String(s.n), s]));
    return esc(text).replace(/\*\*([^*\n]+)\*\*/g, "<b>$1</b>").replace(/\[(\d+)\]/g, (m, n) => by[n]
      ? `<a class="fx-cite" href="#fx-src-${mid}-${n}" title="${esc(by[n].display_title || by[n].title)}">[${n}]</a>` : m);
  }
  async function openPassage(docId, chunkId, quote) {
    const d = await (await _fetch("/api/doc/" + encodeURIComponent(docId))).json();
    const dlg = document.getElementById("dlg"), body = document.getElementById("dbody");
    document.getElementById("dt").textContent = d.title + " (" + docId + ")";
    const text = d.text || "", key = (quote || "").replace(/^“|”$/g, "").replace(/ \.\.\.$/, "").slice(0, 60);
    const i = key ? text.indexOf(key) : -1;
    body.dir = "auto";
    body.innerHTML = i < 0 ? esc(text) : esc(text.slice(0, i)) + "<mark id='fx-hit'>" + esc(text.slice(i, i + key.length)) + "</mark>" + esc(text.slice(i + key.length));
    dlg.showModal();
    const hit = document.getElementById("fx-hit"); if (hit) hit.scrollIntoView({block: "center"});
  }
  document.addEventListener("click", e => {
    const card = e.target.closest(".fx-card");
    if (card) {
      if (card.dataset.href) window.open(card.dataset.href, "_blank");
      else openPassage(card.dataset.doc, card.dataset.chunk, card.querySelector(".fx-quote").textContent);
    }
  });
  Object.assign(window.MSFFeatures, {renderSourceCards, linkCitations});
  const chatCss = `.fx-chatbtn{position:fixed;right:16px;bottom:16px;z-index:5;background:var(--red);color:#fff;border:0;border-radius:999px;padding:12px 18px;font-weight:700;box-shadow:0 2px 8px #0003}
  .fx-chat{position:fixed;right:16px;bottom:70px;width:min(440px,calc(100vw - 32px));max-height:72vh;display:none;flex-direction:column;z-index:6;background:var(--card);border:1px solid var(--line);border-radius:14px;box-shadow:0 6px 24px #0004}
  .fx-chat.on{display:flex}.fx-chat header{padding:10px 14px;border-bottom:1px solid var(--line);display:flex;gap:8px;align-items:center}
  .fx-msgs{overflow:auto;padding:10px 12px;flex:1}.fx-u,.fx-bot{margin:8px 0;padding:9px 12px;border-radius:12px;font-size:14.5px;white-space:pre-wrap}
  .fx-u{background:var(--ink);color:var(--bg);margin-left:40px}.fx-bot{background:var(--bg);border:1px solid var(--line);margin-right:20px}
  .fx-meta{font-size:11.5px;color:var(--mute);margin-top:6px;white-space:normal}.fx-warn{color:var(--mid)}
  .fx-cards{margin:8px 0 0;padding:0;list-style:none;white-space:normal}.fx-card{border:1px solid var(--line);border-radius:10px;padding:7px 9px;margin:6px 0;cursor:pointer;background:var(--card);font-size:13.5px}
  .fx-card:hover,.fx-card:target{border-color:var(--red)}.fx-cardmeta{font-size:12px;color:var(--mute);margin:3px 0}
  .fx-id{float:right;font-size:11px;color:var(--mute);opacity:.7}.fx-quote{font-size:12.5px;color:var(--mute);font-style:italic}
  .fx-cite{color:var(--red);font-weight:700;text-decoration:none}.fx-top{font-size:11.5px;font-weight:600;color:var(--red);margin-bottom:6px;white-space:normal}
  .fx-chat form{display:flex;gap:6px;padding:10px;border-top:1px solid var(--line)}.fx-chat input{flex:1;min-width:0;padding:9px 11px;border-radius:10px;border:1px solid var(--line);background:var(--card);color:var(--ink)}`;
  const st2 = document.createElement("style"); st2.textContent = chatCss; document.head.appendChild(st2);
  const btn = document.createElement("button"); btn.className = "fx-chatbtn"; btn.textContent = "Ask (offline AI)";
  const panel = document.createElement("div"); panel.className = "fx-chat";
  panel.innerHTML = `<header><b style="flex:1">Ask the library</b><button type="button" id="fx-chat-x">Close</button></header>
    <div class="fx-msgs" id="fx-msgs"><div class="fx-meta">Answers come only from MSF documents on this device. If the library has nothing reliable, the AI is not asked.</div></div>
    <form id="fx-chat-f"><input id="fx-chat-q" placeholder="How do I treat post-partum haemorrhage?" autocomplete="off" dir="auto"><button class="primary">Send</button></form>`;
  document.body.append(btn, panel);
  btn.onclick = () => { panel.classList.toggle("on"); document.getElementById("fx-chat-q").focus(); };
  panel.querySelector("#fx-chat-x").onclick = () => panel.classList.remove("on");
  const HISTORY = [];
  const askUrl = new URLSearchParams(location.search).get("ask");  // deep link: /?ask=question
  if (askUrl) setTimeout(() => { panel.classList.add("on"); document.getElementById("fx-chat-q").value = askUrl;
                                 document.getElementById("fx-chat-f").requestSubmit(); }, 300);
  panel.querySelector("#fx-chat-f").onsubmit = async e => {
    e.preventDefault();
    const qi = document.getElementById("fx-chat-q"), q = qi.value.trim();
    if (!q) return;
    qi.value = "";
    const msgs = document.getElementById("fx-msgs");
    msgs.insertAdjacentHTML("beforeend", `<div class="fx-u" dir="auto">${esc(q)}</div>`);
    const bot = document.createElement("div"); bot.className = "fx-bot"; bot.dir = "auto";
    const mid = "m" + Date.now();
    bot.innerHTML = `<div class="fx-top"></div><span class="fx-ans">…</span>`; msgs.appendChild(bot); msgs.scrollTop = msgs.scrollHeight;
    const ans = bot.querySelector(".fx-ans");
    let text = "", meta = {}, done = {};
    try {
      const res = await _fetch("/api/chat", {method: "POST", headers: {"Content-Type": "application/json"},
                                             body: JSON.stringify({question: q, history: HISTORY.slice(-4)})});
      const reader = res.body.getReader(), dec = new TextDecoder();
      let buf = "", ev = "";
      for (;;) {
        const {value, done: end} = await reader.read();
        if (end) break;
        buf += dec.decode(value, {stream: true});
        let i;
        while ((i = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1);
          if (line.startsWith("event:")) ev = line.slice(6).trim();
          else if (line.startsWith("data:")) {
            const d = JSON.parse(line.slice(5));
            if (ev === "meta") { meta = d; if (d.llm_called) bot.querySelector(".fx-top").textContent =
              `AI answer generated on this device by a small offline model (${d.model_label || d.model}) from MSF documents · check the sources`; }
            else if (ev === "token") { text += d.text; ans.textContent = text; msgs.scrollTop = msgs.scrollHeight; }
            else if (ev === "done") done = d;
            else if (ev === "error") ans.textContent = text + "\n" + d.error;
          }
        }
      }
    } catch (err) { ans.textContent = "Chat is unavailable (search still works)."; return; }
    HISTORY.push({role: "user", content: q}, {role: "assistant", content: text});
    const warn = (done.unsupported_terms || []).length
      ? `<div class="fx-meta fx-warn">Words not found in the sources: ${done.unsupported_terms.map(esc).join(", ")}: check before acting.</div>` : "";
    if (done.llm_called) {
      ans.innerHTML = linkCitations(text, meta.sources, mid);
      bot.querySelector(".fx-top").textContent += ` · ${(done.total_ms / 1000).toFixed(1)} s`;
    }
    bot.insertAdjacentHTML("beforeend", done.llm_called
      ? `${warn}<div class="fx-meta">Sources</div>${renderSourceCards(meta.sources, mid)}
         <div class="fx-meta">${esc(done.model)} · first token ${done.ttft_ms} ms · total ${(done.total_ms / 1000).toFixed(1)} s · retrieval ${meta.retrieval_ms} ms</div>`
      : `<div class="fx-meta">The library has no reliable match, so the AI was not asked (${done.total_ms} ms).</div>`);
  };
})();
