/* Page de recherche : onglets texte / image / audio, appel /api/search, rendu des résultats.
   Le texte venant du serveur est toujours inséré avec textContent (jamais innerHTML). */
(function () {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const caps = JSON.parse(document.body.dataset.caps || "{}");
  let kind = "text";

  const tabs = document.querySelectorAll(".tab");
  tabs.forEach((t) => {
    if (caps[t.dataset.kind] === false) { t.disabled = true; t.title = "Modalité indisponible"; }
    t.addEventListener("click", () => setKind(t.dataset.kind));
  });

  function setKind(k) {
    kind = k;
    tabs.forEach((t) => t.classList.toggle("active", t.dataset.kind === k));
    $("#field-text").classList.toggle("hidden", k !== "text");
    $("#field-file").classList.toggle("hidden", k === "text");
    $("#classify").classList.toggle("hidden", k !== "audio");
    $("#file").accept = k === "image" ? "image/*,.jpg,.jpeg,.png,.webp" : ".wav,.mp3,.flac,.ogg,audio/*";
    $("#file").value = ""; $("#preview").replaceChildren(); $("#classes").classList.add("hidden");
  }

  $("#file").addEventListener("change", () => {
    const f = $("#file").files[0]; $("#preview").replaceChildren();
    if (!f) return;
    const el = document.createElement(kind === "image" ? "img" : "audio");
    el.src = URL.createObjectURL(f);
    if (kind === "audio") el.controls = true;
    $("#preview").appendChild(el);
  });

  function status(msg, err) { const s = $("#status"); s.textContent = msg || ""; s.className = err ? "err" : ""; }

  function buildForm() {
    const fd = new FormData();
    fd.append("kind", kind);
    fd.append("k", $("#k").value);
    fd.append("mode", $("#mode").value);
    const filters = {};
    if ($("#f-modality").value) filters.modality = $("#f-modality").value;
    if ($("#f-dataset").value) filters.dataset = $("#f-dataset").value;
    if ($("#f-split").value) filters.split = $("#f-split").value;
    if ($("#f-min").value !== "") filters.min_score = parseFloat($("#f-min").value);
    fd.append("filters", JSON.stringify(filters));
    if (kind === "text") fd.append("q", $("#q").value);
    else { const f = $("#file").files[0]; if (f) fd.append("file", f); }
    return fd;
  }

  async function call(url, fd) {
    const r = await fetch(url, { method: "POST", body: fd });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error((j.error && j.error.message) || ("Erreur HTTP " + r.status));
    return j;
  }

  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }

  function card(h, maxScore) {
    const c = el("article", "card");
    const media = el("div", "media");
    if (h.modality === "image") { const i = el("img"); i.src = h.media_url; i.alt = h.caption || h.item_id; i.loading = "lazy"; media.appendChild(i); }
    else { const a = el("audio"); a.controls = true; a.preload = "none"; a.src = h.media_url; media.appendChild(a); }
    c.appendChild(media);
    const b = el("div", "body");
    const head = el("div"); head.appendChild(el("span", "rank", "#" + h.rank + " ")); head.appendChild(el("span", "", h.item_id));
    b.appendChild(head);
    const badges = el("div", "badges");
    [h.modality, h.dataset, h.label].filter(Boolean).forEach((t) => badges.appendChild(el("span", "", t)));
    b.appendChild(badges);
    b.appendChild(el("div", "score", "score (cosinus) : " + h.score.toFixed(4)));
    const bar = el("div", "bar"); const fill = el("i");
    fill.style.width = Math.max(2, Math.min(100, (h.score / (maxScore || 1)) * 100)) + "%";
    bar.appendChild(fill); b.appendChild(bar);
    const mods = el("div", "mods");
    Object.entries(h.scores_by_modality || {}).forEach(([k, v]) => { mods.appendChild(el("span", "", k)); const bb = el("div", "bar"); const ff = el("i"); ff.style.width = Math.max(2, Math.min(100, v * 100 / (maxScore || 1))) + "%"; bb.appendChild(ff); mods.appendChild(bb); mods.appendChild(el("span", "", v.toFixed(3))); });
    b.appendChild(mods);
    if (h.caption) b.appendChild(el("div", "caption", "« " + h.caption + " »"));
    const d = el("details"); d.appendChild(el("summary", "", "Légendes"));
    d.addEventListener("toggle", async () => {
      if (!d.open || d.dataset.loaded) return; d.dataset.loaded = "1";
      try { const r = await fetch("/api/item/" + encodeURIComponent(h.item_id)); const j = await r.json(); const ol = el("ol"); (j.captions || []).forEach((x) => ol.appendChild(el("li", "", x.text))); d.appendChild(ol); }
      catch (e) { d.appendChild(el("div", "", "indisponible")); }
    });
    b.appendChild(d); c.appendChild(b);
    return c;
  }

  $("#search-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    if (kind !== "text" && !$("#file").files[0]) return status("Choisissez un fichier.", true);
    if (kind === "text" && !$("#q").value.trim()) return status("Saisissez une requête.", true);
    $("#go").disabled = true; status(kind === "audio" ? "Encodage audio en cours (quelques secondes sur CPU)…" : "Recherche…");
    try {
      const j = await call("/api/search", buildForm());
      const box = $("#results"); box.replaceChildren();
      const max = j.hits.length ? Math.max(...j.hits.map((h) => h.score), 1e-6) : 1;
      j.hits.forEach((h) => box.appendChild(card(h, max)));
      status(j.query.n_hits + " résultat(s) en " + j.query.latency_ms + " ms · mode " + j.query.mode);
    } catch (e) { status(e.message, true); }
    finally { $("#go").disabled = false; }
  });

  $("#classify").addEventListener("click", async () => {
    const f = $("#file").files[0];
    if (!f) return status("Choisissez un fichier audio.", true);
    const fd = new FormData(); fd.append("file", f);
    $("#classify").disabled = true; status("Classification…");
    try {
      const j = await call("/api/classify", fd);
      const box = $("#classes"); box.replaceChildren(); box.classList.remove("hidden");
      box.appendChild(el("h3", "", "Classes ESC-50 les plus probables"));
      j.classes.forEach((c) => { const r = el("div", "cls"); r.appendChild(el("span", "", c.label.replace(/_/g, " "))); const bar = el("div", "bar"); const i = el("i"); i.style.width = (c.prob * 100) + "%"; bar.appendChild(i); r.appendChild(bar); r.appendChild(el("span", "", (c.prob * 100).toFixed(1) + " %")); box.appendChild(r); });
      status("Classification en " + j.latency_ms + " ms");
    } catch (e) { status(e.message, true); }
    finally { $("#classify").disabled = false; }
  });

  if (caps.text === false) setKind(caps.image !== false ? "image" : "audio");
})();
