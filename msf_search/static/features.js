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
    if (p.text_source && p.text_source !== "parsed") {
      const conf = p.ocr_confidence != null ? `, confidence ${Number(p.ocr_confidence).toFixed(2)}` : "";
      out.push(`<span class="fx-b fx-ocr">Transcribed from scan/handwriting (${esc(ENGINE[p.text_source] || p.text_source)}${conf}) · check the original ${viewLink(x, x.page)}</span>`);
    }
    if (p.machine_generated && LABEL[p.generator]) out.push(`<span class="fx-b fx-ai">${esc(LABEL[p.generator])}</span>`);
    if (x.matched_via === "qvec") out.push(`<span class="fx-b fx-ai">${LABEL.qvec}</span>`);
    return out.join("");
  }

  function renderImportance(x) {
    const out = [];
    if (x.importance) out.push(`<span class="fx-b fx-imp${x.importance}" title="${esc(x.importance_reason || "")}">${IMPORTANCE[x.importance] || ""}</span>`);
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
  .fx-boost{font-size:13px;color:var(--mute);margin:6px 2px}.fx-boost a{color:inherit}`;
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
})();
