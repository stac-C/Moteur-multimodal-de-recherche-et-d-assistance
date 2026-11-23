/* Nuage de points 2D de l'espace commun : canvas natif, sans bibliothèque externe (démo hors-ligne). */
(function () {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const canvas = $("#plot"), ctx = canvas.getContext("2d"), tip = $("#tip");
  const MOD_COLORS = { image: "#2f6fdb", caption: "#2e9d57", audio: "#e08a1e" };
  let pts = [], view = [], W = 0, H = 0;

  function hashColor(s) {
    let h = 0; for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
    return "hsl(" + (h % 360) + ",65%,50%)";
  }
  function colorOf(p) {
    if ($("#color").value === "label") return p.label ? hashColor(p.label) : "#9aa3b866";
    return MOD_COLORS[p.modality] || "#888";
  }
  function resize() {
    const r = canvas.getBoundingClientRect(), d = window.devicePixelRatio || 1;
    W = r.width; H = r.height; canvas.width = W * d; canvas.height = H * d; ctx.setTransform(d, 0, 0, d, 0, 0);
  }
  function layout() {
    if (!pts.length) { view = []; return; }
    const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys), pad = 24;
    view = pts.map((p) => ({ p, sx: pad + ((p.x - x0) / (x1 - x0 || 1)) * (W - 2 * pad), sy: H - pad - ((p.y - y0) / (y1 - y0 || 1)) * (H - 2 * pad) }));
  }
  function draw() {
    ctx.clearRect(0, 0, W, H);
    view.forEach((v) => { ctx.fillStyle = colorOf(v.p); ctx.beginPath(); ctx.arc(v.sx, v.sy, 3.2, 0, 6.283); ctx.fill(); });
    const lg = $("#legend"); lg.replaceChildren();
    const keys = $("#color").value === "label" ? [...new Set(pts.filter((p) => p.label).map((p) => p.label))].sort() : Object.keys(MOD_COLORS).filter((m) => pts.some((p) => p.modality === m));
    keys.slice(0, 60).forEach((k) => { const s = document.createElement("span"), i = document.createElement("i"); i.style.background = $("#color").value === "label" ? hashColor(k) : MOD_COLORS[k]; s.appendChild(i); s.appendChild(document.createTextNode(k.replace(/_/g, " "))); lg.appendChild(s); });
  }
  async function load() {
    const q = new URLSearchParams({ method: $("#method").value, limit: $("#limit").value });
    if ($("#modality").value) q.set("modality", $("#modality").value);
    if ($("#dataset").value) q.set("dataset", $("#dataset").value);
    $("#info").textContent = "Chargement…";
    try {
      const r = await fetch("/api/embeddings?" + q), j = await r.json();
      if (!r.ok) throw new Error((j.error && j.error.message) || "Erreur " + r.status);
      pts = j.points; resize(); layout(); draw();
      const ev = j.explained_variance ? " · variance expliquée : " + j.explained_variance.map((v) => (v * 100).toFixed(1) + " %").join(" + ") : "";
      $("#info").textContent = j.n_points + " points · " + j.method.toUpperCase() + ev;
    } catch (e) { pts = []; view = []; draw(); $("#info").textContent = e.message; }
  }
  canvas.addEventListener("mousemove", (ev) => {
    const r = canvas.getBoundingClientRect(), mx = ev.clientX - r.left, my = ev.clientY - r.top;
    let best = null, bd = 64;
    for (const v of view) { const d = (v.sx - mx) ** 2 + (v.sy - my) ** 2; if (d < bd) { bd = d; best = v; } }
    if (!best) { tip.classList.add("hidden"); return; }
    tip.textContent = best.p.modality + " · " + best.p.dataset + (best.p.label ? " · " + best.p.label : "") + " · " + best.p.id;
    tip.style.left = Math.min(best.sx + 10, W - 220) + "px"; tip.style.top = (best.sy + 10) + "px"; tip.classList.remove("hidden");
  });
  canvas.addEventListener("mouseleave", () => tip.classList.add("hidden"));
  ["method", "modality", "dataset", "limit"].forEach((id) => $("#" + id).addEventListener("change", load));
  $("#color").addEventListener("change", draw);
  window.addEventListener("resize", () => { resize(); layout(); draw(); });
  load();
})();
