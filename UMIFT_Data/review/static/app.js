// Episode review client. All scrubbing is local: the video element seeks (at most one
// pending seek, latest wins), and one requestAnimationFrame loop updates overlays.
import { createPlots } from "./plots.js";
import { createPath3d, lowerBound } from "./path3d.js";

const $ = id => document.getElementById(id);
const S = {
  episodes: [], meta: null, filter: "all", cur: null, ep: null, token: 0, loading: null, flagging: false,
  frame: 0, t: 0, playing: false, dirty: true,
  seeking: false, pendingSeek: null,
  showHist: true, histMode: "training", showUnsampled: true, colorBy: "time", showVt: false,
  range3d: [0, 1], sprite: null, plots: null, path: null, moreOpen: false,
};

export function decodePayload(buf) {
  const dv = new DataView(buf);
  const hlen = dv.getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hlen)));
  let off = 4 + hlen;
  off += (4 - (off % 4)) % 4;
  const arrays = {};
  for (const a of header.arrays) { arrays[a.name] = new Float32Array(buf, off, a.length); off += 4 * a.length; }
  return { header, arrays };
}

// ---------- episode list ----------
function visible(e) {
  switch (S.filter) {
    case "base": return e.source === "original";
    case "dagger": return e.source === "dagger";
    case "none": return e.flag === "none";
    case "keep": case "reject": return e.flag === S.filter;
    case "stale": return e.stale;
    default: return true;
  }
}
const filtered = () => S.episodes.filter(visible);
const dotClass = e => (e.stale ? "stale" : e.flag === "none" ? "" : e.flag);

function renderList() {
  const ul = $("eplist");
  ul.innerHTML = "";
  for (const e of filtered()) {
    const li = document.createElement("li");
    li.dataset.key = e.key;
    if (S.cur && e.key === S.cur.key) li.className = "cur";
    li.innerHTML = `<img loading="lazy" src="/api/episode/${e.key}/thumb.jpg">
      <div><div class="name">${e.dataset}/ep${e.index}</div>
      <div class="meta"><span class="src ${e.source}">${e.source}</span> ${e.duration_s.toFixed(1)} s${e.note ? " · ✎" : ""}</div></div>
      <span class="dot ${dotClass(e)}" title="${e.stale ? "stale (zarr changed)" : e.flag}"></span>`;
    li.onclick = () => openEpisode(e.key);
    ul.appendChild(li);
  }
  renderCounts();
}

function updateListItem(e) {
  const li = $("eplist").querySelector(`li[data-key="${CSS.escape(e.key)}"]`);
  if (!li) return;
  li.querySelector(".dot").className = `dot ${dotClass(e)}`;
  li.querySelector(".meta").innerHTML = `<span class="src ${e.source}">${e.source}</span> ${e.duration_s.toFixed(1)} s${e.note ? " · ✎" : ""}`;
}

function renderCounts() {
  const by = {};
  for (const e of S.episodes) {
    const c = (by[e.dataset] ??= { keep: 0, reject: 0, none: 0, stale: 0, n: 0 });
    c.n++; c[e.flag]++; if (e.stale) c.stale++;
  }
  $("counts").textContent = Object.entries(by)
    .map(([d, c]) => `${d.padEnd(9)} ${String(c.n).padStart(3)}  keep ${c.keep}  rej ${c.reject}  open ${c.none}${c.stale ? `  stale ${c.stale}` : ""}`)
    .join("\n") + (S.meta.orphans.length ? `\norphan flags: ${S.meta.orphans.join(", ")}` : "");
}

// ---------- open episode ----------
async function openEpisode(key) {
  const tok = ++S.token;
  const e = S.episodes.find(x => x.key === key);
  if (!e) return;
  S.loading = key;  // flag keys are ignored until this episode is on screen
  let ep;
  try {
    const r = await fetch(`/api/episode/${key}/data`);
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || `HTTP ${r.status}`);
    const buf = await r.arrayBuffer();
    if (tok !== S.token) return;
    ep = decodePayload(buf);
  } catch (err) {
    if (tok !== S.token) return;
    S.loading = null;
    alert(`Could not load ${key}: ${err.message}\nStill showing ${S.cur ? S.cur.key : "nothing"}; flags apply to that.`);
    return;
  }
  S.loading = null;
  S.cur = e; S.ep = ep;
  history.replaceState(null, "", "#" + key);
  pause();
  const v = $("video");
  S.seeking = false; S.pendingSeek = null;
  v.src = `/api/episode/${key}/video.mp4`;
  v.playbackRate = parseFloat($("speed").value);

  if (S.plots) { S.moreOpen = S.plots.moreOpen; S.plots.destroy(); }
  if (S.path) S.path.destroy();
  S.plots = createPlots($("charts"), ep.arrays, ep.header, { syncKey: "ep", onSeekTime: seekTime, moreOpen: S.moreOpen, showVt: S.showVt });
  S.plots.setHistory(S.showHist && S.meta.has_ckpt, S.histMode);
  S.plots.setUnsampled(S.showUnsampled);
  try {
    S.path = createPath3d($("path3d"), ep.arrays, ep.header);
    S.path.setColorBy(S.colorBy);
    S.path.setShowVt(S.showVt);
  } catch (err) {  // no WebGL (e.g. X forwarding): keep everything else working
    console.warn("3D view disabled:", err);
    $("path3d").innerHTML = `<p class="muted" style="padding:10px">3D view unavailable (WebGL: ${err.message})</p>`;
    const noop = () => {};
    S.path = { setRange: noop, setCursor: noop, setColorBy: noop, setShowVt: noop, destroy: () => { $("path3d").innerHTML = ""; } };
  }

  const n = ep.arrays.rgb_t.length;
  $("timeSlider").max = n - 1;
  S.range3d = [ep.arrays.rgb_t[0], ep.arrays.rgb_t[n - 1]];
  $("r3dStart").value = 0; $("r3dEnd").value = 1000;
  applyRange3d();

  S.sprite = null;
  if (S.meta.has_ckpt) {
    const img = new Image();
    img.onload = () => { if (tok === S.token) { S.sprite = img; S.dirty = true; } };
    img.src = `/api/episode/${key}/sprite.jpg`;
  }
  $("note").value = e.note || "";
  renderInfo();
  for (const li of $("eplist").children) li.classList.toggle("cur", li.dataset.key === key);
  $("eplist").querySelector("li.cur")?.scrollIntoView({ block: "nearest" });
  setFrame(0);
}

function renderInfo() {
  const e = S.cur;
  $("epTitle").textContent = `${e.dataset}/ep${e.index}`;
  const b = $("flagBadge");
  b.className = `badge ${e.stale ? "stale" : e.flag}`;
  b.textContent = e.stale ? `${e.flag} (stale)` : e.flag === "none" ? "unflagged" : e.flag.toUpperCase();
}

// ---------- time / video ----------
const fps = () => S.ep.header.video_fps || 15;
const nFrames = () => S.ep.arrays.rgb_t.length;

function requestSeek(vt) {
  const v = $("video");
  if (S.seeking) { S.pendingSeek = vt; return; }
  S.seeking = true;
  v.currentTime = vt;
}

function setFrame(f, seek = true) {
  f = Math.max(0, Math.min(nFrames() - 1, f | 0));
  S.frame = f;
  S.t = S.ep.arrays.rgb_t[f];
  if (seek) requestSeek((f + 0.5) / fps());
  S.dirty = true;
}

function seekTime(t) {
  const rgb = S.ep.arrays.rgb_t;
  let i = lowerBound(rgb, t);
  if (i > 0 && (i >= rgb.length || t - rgb[i - 1] < rgb[i] - t)) i--;
  pause();
  setFrame(i);
}

function play() {
  if (!S.ep) return;
  const v = $("video");
  if (S.frame >= nFrames() - 1) setFrame(0);
  S.pendingSeek = null;
  v.play(); S.playing = true; $("playBtn").textContent = "❚❚";
}
function pause() {
  const v = $("video");
  if (!v.paused) v.pause();
  if (S.playing && S.ep) setFrame(S.frame, true);  // snap to the shown frame
  S.playing = false; $("playBtn").textContent = "▶";
}

function histStart() {
  const h = S.ep.header.window;
  if (!S.showHist || !h) return null;
  const a = S.ep.arrays.win_pose, k = h.pose, f = S.frame;
  const first = a[f * k];
  const tq = S.ep.arrays.rgb_t[f];
  return S.histMode === "rollout" ? tq - S.ep.header.rollout_slowdown * (tq - first) : first;
}

function frameTicks(name) {
  const h = S.ep.header.window;
  if (!h || !S.ep.arrays["win_" + name]) return null;
  const k = h[name];
  return S.ep.arrays["win_" + name].subarray(S.frame * k, S.frame * k + k);
}

function drawStrip() {
  const cv = $("rgbStrip"), ctx = cv.getContext("2d");
  const W = cv.clientWidth, H = 18, dpr = window.devicePixelRatio || 1;
  if (cv.width !== Math.round(W * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const rgb = S.ep.arrays.rgb_t, n = rgb.length, t0 = rgb[0], t1 = rgb[n - 1];
  const X = t => ((t - t0) / (t1 - t0 || 1)) * W;
  ctx.fillStyle = "#eee"; ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = "rgba(0,0,0,0.08)";
  ctx.fillRect(X(S.range3d[0]), 0, X(S.range3d[1]) - X(S.range3d[0]), H);
  const sm = S.ep.arrays.sampled;
  if (sm && S.showUnsampled) {
    ctx.fillStyle = "rgba(120,120,120,0.45)";
    for (let i = 0; i < n; i++) if (!sm[i]) ctx.fillRect(X(rgb[i]), 0, Math.max(1, X(rgb[Math.min(n - 1, i + 1)]) - X(rgb[i])), 5);
  }
  if (S.showHist) {
    const s = S.histMode === "rollout" ? S.ep.header.rollout_slowdown : 1, tq = rgb[S.frame];
    const kr = frameTicks("rgb"), ka = frameTicks("action");
    if (ka) { ctx.fillStyle = "rgba(255,160,0,0.6)"; const a = tq + s * (ka[0] - tq), b = tq + s * (ka[ka.length - 1] - tq); ctx.fillRect(X(a), 6, Math.max(1, X(b) - X(a)), 12); }
    if (kr) { ctx.fillStyle = "#1f77b4"; for (const k of kr) ctx.fillRect(X(tq - s * (tq - k)) - 0.5, 6, 1.5, 12); }
  }
  ctx.fillStyle = "#000"; ctx.fillRect(X(S.t) - 1, 0, 2, H);
}

function drawFilmstrip() {
  const cv = $("filmstrip"), ctx = cv.getContext("2d");
  const W = cv.clientWidth, H = 44, dpr = window.devicePixelRatio || 1;
  if (cv.width !== Math.round(W * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const kr = frameTicks("rgb");
  if (!kr || !S.sprite || !S.showHist) return;
  const g = S.ep.header.sprite, rgb = S.ep.arrays.rgb_t, tile = 38;
  const s = S.histMode === "rollout" ? S.ep.header.rollout_slowdown : 1, tq = rgb[S.frame];
  for (let i = 0; i < kr.length; i++) {
    let f = lowerBound(rgb, tq - s * (tq - kr[i]) - 1e-6);
    f = Math.min(g.n - 1, Math.max(0, f));
    ctx.drawImage(S.sprite, (f % g.cols) * g.tile, Math.floor(f / g.cols) * g.tile, g.tile, g.tile, i * (tile + 2), 3, tile, tile);
  }
}

function fmtT(t) { return t.toFixed(2).padStart(6); }

function tick() {
  if (S.ep) {
    const v = $("video");
    if (S.playing) {
      const vt = v.currentTime * fps() - 0.5;
      const f = Math.max(0, Math.min(nFrames() - 1, Math.floor(vt + 0.5)));
      const rgb = S.ep.arrays.rgb_t, i = Math.max(0, Math.min(nFrames() - 2, Math.floor(vt)));
      const fr = Math.max(0, Math.min(1, vt - i));
      S.frame = f;
      S.t = rgb[i] + (rgb[i + 1] - rgb[i]) * fr;
      S.dirty = true;
      if (v.ended) { S.playing = false; $("playBtn").textContent = "▶"; }
    }
    if (S.dirty) {
      S.dirty = false;
      S.plots.setCursor(S.t, S.frame);
      S.path.setCursor(S.t, histStart());
      drawStrip();
      drawFilmstrip();
      $("timeSlider").value = S.frame;
      const sm = S.ep.arrays.sampled;
      $("timeLbl").textContent = `t=${fmtT(S.t)} s  frame ${S.frame + 1}/${nFrames()}`;
      $("sampledLbl").textContent = S.ep.header.sampled_error ? `sampler rejects this episode (${S.ep.header.sampled_error})`
        : sm ? (sm[S.frame] ? "" : "not a training query") : "";
    }
  }
  requestAnimationFrame(tick);
}

// ---------- flags ----------
async function postFlag(body) {
  const e = S.cur;
  const r = await fetch("/api/flag", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ key: e.key, ...body }) });
  const d = await r.json();
  if (!r.ok) { alert(d.error); return; }
  e.flag = d.flag; e.note = d.note || ""; e.stale = d.stale;
  updateListItem(e); renderCounts(); renderInfo();
}

async function flag(f) {
  // One flag at a time, and none while the next episode loads: a key pressed during
  // either would otherwise land on the episode that is still on screen.
  if (!S.cur || S.loading || S.flagging) return;
  S.flagging = true;
  try {
    await postFlag({ flag: f });
    if ($("autoadv").checked && f !== "none") step(1);
  } finally {
    S.flagging = false;
  }
}

function step(d) {
  const list = filtered();
  if (!list.length) return;
  let i = list.findIndex(e => S.cur && e.key === S.cur.key);
  i = i < 0 ? 0 : Math.max(0, Math.min(list.length - 1, i + d));
  openEpisode(list[i].key);
}

// ---------- 3D range ----------
function applyRange3d() {
  const rgb = S.ep.arrays.rgb_t, t0 = rgb[0], t1 = rgb[rgb.length - 1];
  let a = +$("r3dStart").value, b = +$("r3dEnd").value;
  if (a > b - 1) { if (document.activeElement === $("r3dStart")) a = b - 1; else b = a + 1; $("r3dStart").value = a; $("r3dEnd").value = b; }
  S.range3d = [t0 + (a / 1000) * (t1 - t0), t0 + (b / 1000) * (t1 - t0)];
  $("r3dStartLbl").textContent = `${S.range3d[0].toFixed(1)} s`;
  $("r3dEndLbl").textContent = `${S.range3d[1].toFixed(1)} s`;
  S.path.setRange(...S.range3d);
  if ($("lockRange").checked) S.plots.setXRange(...S.range3d);
  S.dirty = true;
}
function setRangeEdge(which) {
  const rgb = S.ep.arrays.rgb_t, t0 = rgb[0], t1 = rgb[rgb.length - 1];
  const v = Math.round(((S.t - t0) / (t1 - t0)) * 1000);
  $(which === 0 ? "r3dStart" : "r3dEnd").value = v;
  applyRange3d();
}

// ---------- wiring ----------
function wire() {
  const v = $("video");
  v.addEventListener("seeked", () => {
    S.seeking = false;
    if (S.pendingSeek !== null) { const x = S.pendingSeek; S.pendingSeek = null; requestSeek(x); }
  });
  v.addEventListener("loadedmetadata", () => { S.seeking = false; requestSeek((S.frame + 0.5) / fps()); });
  $("playBtn").onclick = () => (S.playing ? pause() : play());
  $("speed").onchange = () => { v.playbackRate = parseFloat($("speed").value); };
  $("timeSlider").addEventListener("input", e => { if (S.playing) pause(); setFrame(+e.target.value); });
  for (const id of ["r3dStart", "r3dEnd"]) $(id).addEventListener("input", applyRange3d);
  $("r3dReset").onclick = () => { $("r3dStart").value = 0; $("r3dEnd").value = 1000; applyRange3d(); };
  $("lockRange").onchange = () => { if ($("lockRange").checked) S.plots.setXRange(...S.range3d); else S.plots.resetX(); };
  $("showVt").onchange = e => { S.showVt = e.target.checked; S.path.setShowVt(S.showVt); S.plots.setVt(S.showVt); };
  $("colorBy").onchange = e => { S.colorBy = e.target.value; S.path.setColorBy(S.colorBy); };
  $("showHist").onchange = e => { S.showHist = e.target.checked; S.plots.setHistory(S.showHist, S.histMode); S.dirty = true; };
  $("histMode").onchange = e => { S.histMode = e.target.value; S.plots.setHistory(S.showHist, S.histMode); S.dirty = true; };
  $("showUnsampled").onchange = e => { S.showUnsampled = e.target.checked; S.plots.setUnsampled(S.showUnsampled); S.dirty = true; };
  for (const b of $("filters").children) b.onclick = () => {
    S.filter = b.dataset.f;
    for (const x of $("filters").children) x.classList.toggle("on", x === b);
    renderList();
  };
  const note = $("note");
  note.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === "Escape") { note.blur(); } e.stopPropagation(); });
  note.addEventListener("change", () => { if (S.cur && note.value !== (S.cur.note || "")) postFlag({ note: note.value }); });

  document.addEventListener("keydown", e => {
    if (!S.ep || $("exportDlg").open) return;
    if (e.target.tagName === "INPUT" && e.target.type !== "range" && e.target.type !== "checkbox") return;
    const k = e.key;
    if (k === " ") { e.preventDefault(); S.playing ? pause() : play(); }
    else if (k === "ArrowLeft" || k === "ArrowRight") {
      e.preventDefault();
      const d = k === "ArrowLeft" ? -1 : 1;
      if (S.playing) pause();
      setFrame(S.frame + (e.shiftKey ? d * Math.round(fps()) : d));
    }
    else if (k === "ArrowUp" || k === "ArrowDown") { e.preventDefault(); step(k === "ArrowUp" ? -1 : 1); }
    else if (k === "p" || k === "P") flag("keep");
    else if (k === "x" || k === "X") flag("reject");
    else if (k === "u" || k === "U") flag("none");
    else if (k === "n" || k === "N") { e.preventDefault(); note.focus(); }
    else if (k === "[") setRangeEdge(0);
    else if (k === "]") setRangeEdge(1);
  });

  const dlg = $("exportDlg");
  $("exportBtn").onclick = () => { $("exResult").textContent = ""; dlg.showModal(); };
  const doExport = async dry => {
    const inc = dlg.querySelector("input[name=inc]:checked").value;
    $("exResult").textContent = dry ? "checking…" : "exporting… (can take minutes)";
    const r = await fetch("/api/export", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ output: $("exOut").value, include: inc, overwrite: $("exOverwrite").checked, dry_run: dry }) });
    const d = await r.json();
    $("exResult").textContent = r.ok ? JSON.stringify(d, null, 1) : "ERROR: " + d.error;
  };
  $("exDry").onclick = () => doExport(true);
  $("exGo").onclick = () => doExport(false);
}

async function init() {
  S.meta = await (await fetch("/api/episodes")).json();
  S.episodes = S.meta.episodes;
  $("flagsFile").textContent = `flags: ${S.meta.flags_file}`;
  $("exOut").value = "/mnt/data/umidata/ft/datasets/stringer_v3.5_reviewed/acp_replay_buffer.zarr";
  if (!S.meta.has_ckpt) {
    $("ckptToggles").style.display = "none";
    $("filmstripRow").style.display = "none";
    S.showHist = false;
  }
  $("legendKey").innerHTML = S.meta.has_ckpt
    ? `<i style="background:rgba(31,119,180,.3)"></i>pose obs window <i style="background:rgba(214,39,40,.3)"></i>wrench obs window
       <i style="background:rgba(255,160,0,.45)"></i>action horizon <i style="background:rgba(120,120,120,.35)"></i>never sampled
       <i style="background:rgba(0,0,0,.08)"></i>start pad <i style="background:rgba(148,103,189,.3)"></i>scan gap <i style="background:#f0f"></i>pose jump
       <span class="muted">(ckpt: ${S.meta.ckpt.split("/").slice(-3).join("/")})</span>`
    : `<i style="background:rgba(0,0,0,.08)"></i>start pad <i style="background:rgba(148,103,189,.3)"></i>scan gap <i style="background:#f0f"></i>pose jump
       <span class="muted">(run with --ckpt for history windows)</span>`;
  wire();
  renderList();
  const want = decodeURIComponent(location.hash.slice(1));
  await openEpisode(S.episodes.some(e => e.key === want) ? want : S.episodes[0].key);
  requestAnimationFrame(tick);
}

init();
