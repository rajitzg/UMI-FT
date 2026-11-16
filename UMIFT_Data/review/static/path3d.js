// three.js tool path: thick Line2 path coloured by time or |F|, optional dashed
// virtual-target path, solid axis triad at the cursor. Renders on demand only.
import * as THREE from "./vendor/three.module.min.js";
import { OrbitControls } from "./vendor/OrbitControls.js";
import { Line2 } from "./vendor/Line2.js";
import { LineGeometry } from "./vendor/LineGeometry.js";
import { LineMaterial } from "./vendor/LineMaterial.js";

const PATH_WIDTH_PX = 3;
const VT_WIDTH_PX = 2;
const MAX_LINE_POINTS = 2500;

// RGB axis triad from solid cylinders (WebGL ignores line widths > 1).
function makeTriad(len, rad) {
  const g = new THREE.Group();
  const geo = new THREE.CylinderGeometry(rad, rad, len, 8);
  geo.translate(0, len / 2, 0); // cylinder runs along +Y from the origin
  const axes = [[0xd62728, [0, 0, -Math.PI / 2]], [0x2ca02c, [0, 0, 0]], [0x1f77b4, [Math.PI / 2, 0, 0]]];
  for (const [color, [rx, ry, rz]] of axes) {
    const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color }));
    m.rotation.set(rx, ry, rz);
    g.add(m);
  }
  return g;
}

const STOPS = [[0.267, 0.005, 0.329], [0.229, 0.322, 0.546], [0.128, 0.567, 0.551], [0.369, 0.789, 0.383], [0.993, 0.906, 0.144]];
function viridis(u) {
  u = Math.min(1, Math.max(0, u)) * (STOPS.length - 1);
  const i = Math.min(STOPS.length - 2, Math.floor(u)), f = u - i, a = STOPS[i], b = STOPS[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

export function lowerBound(arr, v) {
  let lo = 0, hi = arr.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] < v) lo = m + 1; else hi = m; }
  return lo;
}

function interp(tSrc, ySrc, tDst) {
  const out = new Float32Array(tDst.length);
  let j = 0;
  for (let i = 0; i < tDst.length; i++) {
    const t = tDst[i];
    while (j < tSrc.length - 2 && tSrc[j + 1] < t) j++;
    const t0 = tSrc[j], t1 = tSrc[j + 1] ?? t0;
    const f = t1 > t0 ? Math.min(1, Math.max(0, (t - t0) / (t1 - t0))) : 0;
    out[i] = ySrc[j] + ((ySrc[j + 1] ?? ySrc[j]) - ySrc[j]) * f;
  }
  return out;
}

export function createPath3d(container, arrays, header) {
  container.innerHTML = "";
  const n = arrays.pose_t.length;
  const tArr = arrays.pose_t;
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setClearColor(0xffffff);
  container.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1, 0.001, 100);
  camera.up.set(0, -1, 0); // umift_std_v2: +Y down
  const controls = new OrbitControls(camera, renderer.domElement);

  const pos = new Float32Array(n * 3);
  const c = new THREE.Vector3();
  for (let i = 0; i < n; i++) { pos[3 * i] = arrays.px[i]; pos[3 * i + 1] = arrays.py[i]; pos[3 * i + 2] = arrays.pz[i]; }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geom.computeBoundingSphere();
  const bs = geom.boundingSphere;
  c.copy(bs.center);
  const colors = new Float32Array(n * 3);
  geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  // Thick path + dashed virtual-target path as Line2. Line2 has no draw range, so the
  // visible [start, end] slice is rebuilt (lazily, once per animation frame).
  const vtPos = new Float32Array(n * 3);
  const hasVt = !!arrays.vtx;
  if (hasVt) for (let i = 0; i < n; i++) { vtPos[3 * i] = arrays.vtx[i]; vtPos[3 * i + 1] = arrays.vty[i]; vtPos[3 * i + 2] = arrays.vtz[i]; }
  const pathMat = new LineMaterial({ vertexColors: true, linewidth: PATH_WIDTH_PX });
  const vtMat = new LineMaterial({ color: 0x444444, linewidth: VT_WIDTH_PX, dashed: true });
  const pathLine = new Line2(new LineGeometry(), pathMat);
  const vtLine = new Line2(new LineGeometry(), vtMat);
  vtLine.visible = false;
  scene.add(pathLine, vtLine);
  // Thick lines cost one quad per segment; cap the drawn vertices (pose is ~200 Hz,
  // so this is still several points per mm of tool travel).
  function decimate(arr, i0, i1) {
    const k = Math.max(1, Math.ceil((i1 - i0) / MAX_LINE_POINTS));
    if (k === 1) return arr.subarray(3 * i0, 3 * i1);
    const m = Math.ceil((i1 - i0) / k) + 1, out = new Float32Array(3 * m);
    let j = 0;
    for (let i = i0; i < i1; i += k, j++) { out[3 * j] = arr[3 * i]; out[3 * j + 1] = arr[3 * i + 1]; out[3 * j + 2] = arr[3 * i + 2]; }
    const l = i1 - 1; out[3 * j] = arr[3 * l]; out[3 * j + 1] = arr[3 * l + 1]; out[3 * j + 2] = arr[3 * l + 2];
    return out.subarray(0, 3 * (j + 1));
  }
  let range = [0, n];
  let linesDirty = true;
  function rebuildLines() {
    const [i0, i1] = range;
    if (i1 - i0 < 2) { pathLine.visible = false; return; }
    pathLine.visible = true;
    const g = new LineGeometry();
    g.setPositions(decimate(pos, i0, i1));
    g.setColors(decimate(colors, i0, i1));
    pathLine.geometry.dispose(); pathLine.geometry = g;
    if (hasVt && vtLine.visible) {
      const gv = new LineGeometry();
      gv.setPositions(decimate(vtPos, i0, i1));
      vtLine.geometry.dispose(); vtLine.geometry = gv;
      vtLine.computeLineDistances();
    }
  }

  const histGeom = new THREE.BufferGeometry(); // shares the position buffer, own draw range
  histGeom.setAttribute("position", geom.attributes.position);
  histGeom.setDrawRange(0, 0);
  const hist = new THREE.Points(histGeom, new THREE.PointsMaterial({ color: 0x1f77b4, size: 7, sizeAttenuation: false }));
  scene.add(hist);

  const r = Math.max(bs.radius, 0.01);
  const marker = new THREE.Mesh(new THREE.SphereGeometry(r * 0.025, 12, 8), new THREE.MeshBasicMaterial({ color: 0x000000 }));
  scene.add(marker);
  const triad = makeTriad(r * 0.18, r * 0.008);
  scene.add(triad);
  vtMat.dashSize = r * 0.02; vtMat.gapSize = r * 0.015;
  const grid = new THREE.AxesHelper(r * 0.3);
  grid.position.copy(c);
  grid.material.transparent = true; grid.material.opacity = 0.25;
  scene.add(grid);

  controls.target.copy(c);
  camera.position.set(c.x, c.y - r * 0.8, c.z + r * 2.6); // operator side (+Z), slightly above
  controls.update();

  let fmagAtPose = null;
  let dirty = true;
  function setColorBy(mode) {
    let vals, lo, hi;
    if (mode === "force") {
      fmagAtPose ??= interp(arrays.wrench_t, arrays.fmag, tArr);
      vals = fmagAtPose;
      const sorted = Float32Array.from(vals).sort();
      lo = sorted[Math.floor(0.02 * (n - 1))]; hi = sorted[Math.floor(0.98 * (n - 1))];
    } else { vals = tArr; lo = tArr[0]; hi = tArr[n - 1]; }
    const span = hi > lo ? hi - lo : 1;
    for (let i = 0; i < n; i++) { const [rr, g, b] = viridis((vals[i] - lo) / span); colors[3 * i] = rr; colors[3 * i + 1] = g; colors[3 * i + 2] = b; }
    linesDirty = true;
    dirty = true;
  }
  setColorBy("time");

  function resize() {
    const w = container.clientWidth, h = container.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h);
    pathMat.resolution.set(w, h); vtMat.resolution.set(w, h);
    camera.aspect = w / h; camera.updateProjectionMatrix(); dirty = true;
  }
  const ro = new ResizeObserver(resize);
  ro.observe(container);
  resize();
  controls.addEventListener("change", () => { dirty = true; });

  let alive = true;
  (function loop() {
    if (!alive) return;
    if (linesDirty) { rebuildLines(); linesDirty = false; dirty = true; }
    if (dirty) { renderer.render(scene, camera); dirty = false; }
    requestAnimationFrame(loop);
  })();

  const q = new THREE.Quaternion();
  return {
    setRange(a, b) {
      const i0 = lowerBound(tArr, a), i1 = Math.max(i0 + 1, lowerBound(tArr, b));
      range = [i0, Math.min(n, i1)]; linesDirty = true;
    },
    setShowVt(show) { vtLine.visible = show && hasVt; linesDirty = true; },
    setCursor(t, histFrom) {
      const i = Math.min(n - 1, lowerBound(tArr, t));
      marker.position.set(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
      triad.position.copy(marker.position);
      q.set(arrays.qx[i], arrays.qy[i], arrays.qz[i], arrays.qw[i]);
      triad.quaternion.copy(q);
      if (histFrom != null) { const j = lowerBound(tArr, histFrom); hist.geometry.setDrawRange(j, Math.max(0, i - j + 1)); }
      else hist.geometry.setDrawRange(0, 0);
      dirty = true;
    },
    setColorBy,
    destroy() { alive = false; ro.disconnect(); controls.dispose(); renderer.dispose(); geom.dispose(); pathLine.geometry.dispose(); vtLine.geometry.dispose(); pathMat.dispose(); vtMat.dispose(); container.innerHTML = ""; },
  };
}
