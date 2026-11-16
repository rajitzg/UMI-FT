// uPlot charts for one episode. Series are drawn once; the cursor, policy history
// ticks and action-ahead span live on a separate overlay canvas per chart, so
// scrubbing never redraws the series. Static markers (start pad, scan gaps, pose
// jumps, never-sampled band) are drawn in uPlot's draw hook (redrawn on zoom only).
import uPlot from "./vendor/uPlot.esm.js";

const C = { x: "#d62728", y: "#2ca02c", z: "#1f77b4", mag: "#333", a: "#9467bd", b: "#8c564b", c: "#e377c2" };
const HIST = { pose: "rgba(31,119,180,0.13)", wrench: "rgba(214,39,40,0.12)", action: "rgba(255,160,0,0.22)" };
const TICK = { pose: "rgba(31,119,180,0.9)", wrench: "rgba(214,39,40,0.9)", action: "rgba(230,130,0,0.95)" };

// Series entry: [array name, label, colour, y scale?, "vt" = dashed virtual-target series hidden by default].
const CHARTS = [
  { id: "force", title: "force (N, TCP frame)", t: "wrench_t", s: [["fx", "fx", C.x], ["fy", "fy", C.y], ["fz", "fz", C.z], ["fmag", "|F|", C.mag]], hist: "wrench" },
  { id: "pos", title: "position (m, fixture frame) · dashed = virtual setpoint", t: "pose_t",
    s: [["px", "x", C.x], ["py", "y", C.y], ["pz", "z", C.z], ["vtx", "vt x", C.x, null, "vt"], ["vty", "vt y", C.y, null, "vt"], ["vtz", "vt z", C.z, null, "vt"]], hist: "pose" },
  { id: "rot", title: "orientation (deg, euler xyz relative to reference)", t: "pose_t", s: [["rx", "rx", C.x], ["ry", "ry", C.y], ["rz", "rz", C.z]], hist: "pose" },
  { id: "torque", title: "torque (Nm, TCP frame)", t: "wrench_t", s: [["tx", "tx", C.x], ["ty", "ty", C.y], ["tz", "tz", C.z]], hist: "wrench", more: true },
  { id: "misc", title: "speed (m/s) · VT offset (m) · stiffness (right axis)", t: "pose_t",
    s: [["speed", "speed", C.a], ["vt_off", "vt offset", C.b], ["stiffness", "stiffness", C.c, "stiff"]], hist: "pose", more: true },
];

function intervalsFromMask(rgbT, sampled) {
  const out = [];
  let start = -1;
  for (let i = 0; i < sampled.length; i++) {
    if (!sampled[i] && start < 0) start = i;
    if ((sampled[i] || i === sampled.length - 1) && start >= 0) {
      const end = sampled[i] ? i : i + 1;
      out.push([rgbT[start], end < rgbT.length ? rgbT[end] : rgbT[rgbT.length - 1]]);
      start = -1;
    }
  }
  return out;
}

export function createPlots(container, arrays, header, opts) {
  container.innerHTML = "";
  const charts = [];
  const unsampled = arrays.sampled ? intervalsFromMask(arrays.rgb_t, arrays.sampled) : [];
  const state = { t: 0, frame: 0, showHist: true, mode: "training", showUnsampled: true, slow: header.rollout_slowdown || 2, showVt: !!opts.showVt };
  let syncing = false;
  const t0 = arrays.rgb_t[0], t1 = arrays.rgb_t[arrays.rgb_t.length - 1];
  let moreBox = null;

  function drawStatic(u) {
    const ctx = u.ctx, { top, height } = u.bbox;
    const X = v => u.valToPos(v, "x", true);
    ctx.save();
    const band = (a, b, fill) => {
      const xa = Math.max(u.bbox.left, X(a)), xb = Math.min(u.bbox.left + u.bbox.width, X(b));
      if (xb > xa) { ctx.fillStyle = fill; ctx.fillRect(xa, top, xb - xa, height); }
    };
    if (header.start_pad_s > 0) band(t0, t0 + header.start_pad_s, "rgba(0,0,0,0.05)");
    if (state.showUnsampled) for (const [a, b] of unsampled) band(a, b, "rgba(120,120,120,0.22)");
    for (const [a, b] of header.markers.scan_gaps) band(a, b, "rgba(148,103,189,0.18)");
    ctx.strokeStyle = "rgba(255,0,255,0.8)";
    ctx.lineWidth = 1;
    for (const tj of header.markers.pose_jumps) {
      const x = X(tj);
      ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + height); ctx.stroke();
    }
    ctx.restore();
  }

  function mk(def, parent) {
    const div = document.createElement("div");
    div.className = "chart";
    const title = document.createElement("div");
    title.className = "title";
    title.textContent = def.title;
    div.appendChild(title);
    parent.appendChild(div);
    const tArr = arrays[def.t];
    const defs = def.s.filter(s => arrays[s[0]]);
    const data = [tArr, ...defs.map(s => arrays[s[0]])];
    const series = [{}, ...defs.map(s => ({
      label: s[1], stroke: s[2], width: s[4] === "vt" ? 1.5 : 1, scale: s[3] || "y", points: { show: false },
      ...(s[4] === "vt" ? { dash: [5, 4], show: state.showVt } : {}),
    }))];
    const axes = [{ size: 30 }, { size: 55 }];
    const scales = { x: { time: false } };
    if (def.s.some(s => s[3] === "stiff")) { axes.push({ scale: "stiff", side: 1, grid: { show: false }, size: 50 }); scales.stiff = {}; }
    const width = Math.max(300, container.clientWidth - 2);
    const u = new uPlot({
      width, height: 150, scales, series, axes,
      legend: { show: true, live: true },
      cursor: { drag: { x: true, y: false, setScale: true }, sync: { key: opts.syncKey } },
      hooks: {
        draw: [u => { drawStatic(u); if (u._rc) requestAnimationFrame(() => { sizeOverlay(u._rc); drawOverlay(u._rc); }); }],
        setScale: [(u, key) => {
          if (key !== "x" || syncing) return;
          syncing = true;
          const { min, max } = u.scales.x;
          for (const c of charts) if (c.u !== u) c.u.setScale("x", { min, max });
          syncing = false;
        }],
      },
    }, data, div);
    const ovl = document.createElement("canvas");
    ovl.className = "ovl";
    u.over.appendChild(ovl);
    const c = { def, u, ovl, ctx: ovl.getContext("2d"), dirty: true, div, vtIdx: defs.flatMap((s, i) => (s[4] === "vt" ? [i + 1] : [])) };
    u._rc = c;
    sizeOverlay(c);
    let downX = null;
    u.over.addEventListener("mousedown", e => { downX = e.clientX; });
    u.over.addEventListener("mouseup", e => {
      if (downX !== null && Math.abs(e.clientX - downX) < 3) opts.onSeekTime(u.posToVal(e.offsetX, "x"));
      downX = null;
    });
    charts.push(c);
    return c;
  }

  function sizeOverlay(c) {
    const dpr = window.devicePixelRatio || 1;
    const w = c.u.over.clientWidth, h = c.u.over.clientHeight;
    c.ovl.width = Math.round(w * dpr); c.ovl.height = Math.round(h * dpr);
    c.ovl.style.width = w + "px"; c.ovl.style.height = h + "px";
    c.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.w = w; c.h = h;
  }

  for (const def of CHARTS.filter(d => !d.more)) mk(def, container);
  moreBox = document.createElement("details");
  moreBox.className = "more";
  moreBox.innerHTML = "<summary>torque · speed · VT offset · stiffness</summary>";
  container.appendChild(moreBox);
  const moreDefs = CHARTS.filter(d => d.more);
  let moreBuilt = false;
  moreBox.addEventListener("toggle", () => {
    if (moreBox.open && !moreBuilt) {
      moreBuilt = true;
      const { min, max } = charts[0].u.scales.x;
      for (const d of moreDefs) { const c = mk(d, moreBox); c.u.setScale("x", { min, max }); }
    }
  });
  if (opts.moreOpen) moreBox.open = true;

  function ticks(name, frame) {
    const h = header.window && header.window[name];
    const a = arrays["win_" + name];
    if (!h || !a) return null;
    return a.subarray(frame * h, frame * h + h);
  }

  function drawOverlay(c) {
    const { ctx, u, w, h } = c;
    ctx.clearRect(0, 0, w, h);
    const X = v => u.valToPos(v, "x");
    const tq = state.t;
    if (state.showHist && header.window) {
      const f = state.frame, s = state.mode === "rollout" ? state.slow : 1;
      const tqf = arrays.rgb_t[f];
      const back = v => tqf - s * (tqf - v);
      const fwd = v => tqf + s * (v - tqf);
      const k = ticks(c.def.hist, f);
      if (k) {
        const xa = X(back(k[0])), xb = X(back(k[k.length - 1]));
        ctx.fillStyle = HIST[c.def.hist];
        ctx.fillRect(xa, 0, Math.max(1, xb - xa), h);
        ctx.strokeStyle = TICK[c.def.hist];
        ctx.beginPath();
        for (let i = 0; i < k.length; i++) { const x = Math.round(X(back(k[i]))) + 0.5; ctx.moveTo(x, h - 9); ctx.lineTo(x, h); }
        ctx.stroke();
      }
      const ka = ticks("action", f);
      if (ka) {
        const xa = X(fwd(ka[0])), xb = X(fwd(ka[ka.length - 1]));
        ctx.fillStyle = HIST.action;
        ctx.fillRect(xa, 0, Math.max(1, xb - xa), h);
        ctx.strokeStyle = TICK.action;
        ctx.beginPath();
        for (let i = 0; i < ka.length; i++) { const x = Math.round(X(fwd(ka[i]))) + 0.5; ctx.moveTo(x, 0); ctx.lineTo(x, 7); }
        ctx.stroke();
      }
    }
    const x = Math.round(X(tq)) + 0.5;
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    c.dirty = false;
  }

  const ro = new ResizeObserver(() => {
    const width = Math.max(300, container.clientWidth - 2);
    for (const c of charts) { c.u.setSize({ width, height: 150 }); sizeOverlay(c); c.dirty = true; }
  });
  ro.observe(container);

  return {
    setCursor(t, frame) { state.t = t; state.frame = frame; for (const c of charts) drawOverlay(c); },
    redrawOverlays() { for (const c of charts) drawOverlay(c); },
    setXRange(a, b) { charts[0].u.setScale("x", { min: a, max: b }); },
    resetX() { charts[0].u.setScale("x", { min: t0, max: t1 }); },
    setHistory(show, mode) { state.showHist = show; state.mode = mode; for (const c of charts) drawOverlay(c); },
    setVt(show) { state.showVt = show; for (const c of charts) for (const i of c.vtIdx) c.u.setSeries(i, { show }); },
    setUnsampled(show) { state.showUnsampled = show; for (const c of charts) c.u.redraw(false, false); },
    get moreOpen() { return moreBox.open; },
    destroy() { ro.disconnect(); for (const c of charts) c.u.destroy(); container.innerHTML = ""; },
  };
}
