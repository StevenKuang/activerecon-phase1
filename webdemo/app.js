// Static replay viewer: Spark (3DGS) + three.js overlays, no backend.
// The sim clock drives chunk/frustum
// visibility; the 3DGS layer and the point cloud are mutually exclusive;
// switching reconstructions never touches the clock. Compare = a single
// window split into two viewports sharing one (synced) camera, each pane
// showing its own run + reconstruction — no two-window hack.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
// Fat lines: LineBasicMaterial.linewidth is ignored on ~all WebGL (always
// 1px), so trail/frusta use LineSegments2, which draws pixel-width lines.
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";

const loadingOverlay = document.getElementById("loading");
const loadingTitle = document.getElementById("loading-title");
const loadingModels = document.getElementById("loading-models");
const loadingMin = document.getElementById("loading-min");
const loadChip = document.getElementById("load-chip");
const loadChipLabel = document.getElementById("load-chip-label");
const loadChipBar = document.getElementById("load-chip-bar");
const loadChipPct = document.getElementById("load-chip-pct");
let viewerLoads = [];

function renderViewerLoads() {
  loadingModels.replaceChildren(...viewerLoads.map((item) => {
    const row = document.createElement("div");
    row.className = "viewer-load-row";
    const tag = document.createElement("span");
    tag.className = "viewer-load-tag";
    tag.textContent = item.tag;
    const track = document.createElement("div");
    track.className = "viewer-load-track";
    const bar = document.createElement("div");
    bar.className = "viewer-load-bar" + (item.progress < 1 ? " active" : "");
    bar.style.width = `${Math.max(2, Math.round(100 * item.progress))}%`;
    track.appendChild(bar);
    const percent = document.createElement("span");
    percent.className = "viewer-load-percent";
    percent.textContent = `${Math.round(100 * item.progress)}%`;
    const message = document.createElement("span");
    message.className = "viewer-load-message";
    message.textContent = item.message;
    row.append(tag, track, percent, message);
    return row;
  }));
}

function setViewerLoad(index, progress, message) {
  const item = viewerLoads[index];
  if (!item) return;
  item.progress = Math.max(item.progress, Math.min(1, Math.max(0, progress)));
  item.message = message;
  renderViewerLoads();
}

function showSimpleLoading(message) {
  loadingTitle.textContent = message;
  viewerLoads = [];
  renderViewerLoads();
  loadingMin.hidden = true; // hard waits (boot, job polling) are not minimizable
  loadingOverlay.hidden = false;
}

const SPLAT_QUATERNION = new THREE.Quaternion(); // identity; flip here if needed
const STRATUM_COLORS = { level: 0xffffff, lookup: 0xff9628, lookdown: 0x3cdcff };
const SHARED_COLORS = { severe: 0xe63c3c, clean: 0x46c85a, mixed: 0xaaaaaa };
// Cube eval faces. The four horizontal faces share a hue so the ring reads as
// one thing; up and down are the views a level-only eval set cannot see, so
// they get the two high-contrast colours.
const FACE_COLORS = {
  front: 0x5aa9e6, right: 0x5aa9e6, back: 0x5aa9e6, left: 0x5aa9e6,
  up: 0xffc83c, down: 0xc86ee6,
};
const STANDING_POINT_COLOR = 0xffffff;
const TRAIL_COLOR = 0xeb503c;

loadingTitle.textContent = "Loading view manifest…";
const manifestResponse = await fetch("manifest.json");
if (!manifestResponse.ok) throw new Error(`Failed to load manifest: ${manifestResponse.status}`);
const manifest = await manifestResponse.json();
const live = manifest.live || null;
if (manifest.live && "serviceWorker" in navigator) {
  navigator.serviceWorker.register("/sw.js").catch(() => {});
}
const NAVIGATION_STATE_KEY = "activebench.spark.navigation.v1";

function consumeNavigationState() {
  if (!manifest.live) return null;
  try {
    const raw = sessionStorage.getItem(NAVIGATION_STATE_KEY);
    sessionStorage.removeItem(NAVIGATION_STATE_KEY);
    if (!raw) return null;
    const saved = JSON.parse(raw);
    const selection = manifest.live.selection;
    // A cold RAD build can take several minutes; sessionStorage is tab-scoped,
    // so a generous TTL preserves state without leaking it across sessions.
    const fresh = Date.now() - Number(saved.savedAt || 0) < 2 * 60 * 60 * 1000;
    return fresh && saved.target?.round === selection.round
      && saved.target?.scene === selection.scene ? saved : null;
  } catch (_error) {
    return null;
  }
}
const navigationState = consumeNavigationState();
viewerLoads = manifest.runs.map((_run, index) => ({
  tag: index === 0 ? "A" : "B",
  progress: 0.03,
  message: "Loading replay metadata",
}));
loadingTitle.textContent = `Loading ${viewerLoads.length} selected model${viewerLoads.length === 1 ? "" : "s"}…`;
renderViewerLoads();

const jsonRequests = new Map();
function fetchJson(url) {
  if (!jsonRequests.has(url)) {
    jsonRequests.set(url, fetch(url).then((response) => {
      if (!response.ok) throw new Error(`Failed to load ${url}: ${response.status}`);
      return response.json();
    }).catch((error) => {
      jsonRequests.delete(url); // a failed request must be retryable
      throw error;
    }));
  }
  return jsonRequests.get(url);
}
async function resolveRunManifest(spec) {
  if (!spec.replay) return spec; // self-contained static export
  const [replay, reconstruction] = await Promise.all([
    fetchJson(spec.replay),
    spec.reconstruction ? fetchJson(spec.reconstruction) : null,
  ]);
  const selectedReconstruction = reconstruction && spec.summary
    ? { ...reconstruction, summary: spec.summary }
    : reconstruction;
  const fractions = spec.distractor_pixel_fractions;
  const frames = Array.isArray(fractions) && fractions.length === replay.frames.length
    ? replay.frames.map((frame, index) => ({
      ...frame, distractor_pixel_fraction: fractions[index],
    }))
    : replay.frames;
  return {
    ...replay,
    frames,
    summary: spec.summary ?? replay.summary,
    reconstructions: selectedReconstruction ? [selectedReconstruction] : [],
  };
}
const binaryRequests = new Map();
function fetchArrayBufferWithProgress(url, onProgress) {
  let entry = binaryRequests.get(url);
  if (!entry) {
    entry = { progress: 0, listeners: new Set(), promise: null };
    binaryRequests.set(url, entry);
    const emit = (progress) => {
      entry.progress = progress;
      for (const listener of entry.listeners) listener(progress);
    };
    entry.promise = (async () => {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`Failed to load ${url}: ${response.status}`);
      const total = Number(response.headers.get("Content-Length")) || 0;
      if (!response.body) {
        const buffer = await response.arrayBuffer();
        emit(1);
        return buffer;
      }
      const reader = response.body.getReader();
      const chunks = [];
      let received = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        received += value.byteLength;
        if (total) emit(Math.min(0.99, received / total));
      }
      const joined = new Uint8Array(received);
      let offset = 0;
      for (const chunk of chunks) {
        joined.set(chunk, offset);
        offset += chunk.byteLength;
      }
      emit(1);
      return joined.buffer;
    })();
  }
  entry.listeners.add(onProgress);
  onProgress(entry.progress);
  return entry.promise.finally(() => {
    entry.listeners.delete(onProgress);
    // Deduplicate in-flight downloads only. Active Runs own the buffers;
    // retaining fulfilled promises here kept every visited scene in RAM.
    if (!entry.listeners.size && binaryRequests.get(url) === entry) {
      binaryRequests.delete(url);
    }
  });
}

const restoredPreferences = navigationState?.preferences || {};
const restoredView = navigationState?.view || {};
const state = {
  t: Number.isFinite(restoredView.t) ? restoredView.t : 0,
  playing: typeof restoredView.playing === "boolean" ? restoredView.playing : true,
  speed: Number.isFinite(restoredPreferences.speed) ? restoredPreferences.speed : 3,
  loop: typeof restoredPreferences.loop === "boolean" ? restoredPreferences.loop : true,
  detailScale: Number.isFinite(restoredPreferences.detailScale)
    ? restoredPreferences.detailScale : 1,
  layers: {
    splat: true, contamination: true, history: true,
    distractors: true, eval: false, shared: false,
    ...(restoredPreferences.layers || {}),
  },
};

// --- renderer / scene (shared across panes) --------------------------------

const viewport = document.getElementById("viewport");
const renderer = new THREE.WebGLRenderer({ antialias: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
viewport.appendChild(renderer.domElement);

const camera = new THREE.PerspectiveCamera(60, 1, 0.02, 200);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

function enableRotateAroundMe() {
  const dir = new THREE.Vector3();
  camera.getWorldDirection(dir);
  controls.target.copy(camera.position).addScaledVector(dir, 0.05);
  controls.update();
}

// Each run gets its OWN scene + SparkRenderer. Spark accumulates every
// visible SplatMesh in a scene into one globally-sorted buffer and re-sorts
// asynchronously (not per render() call), so a single shared scene can't
// show two different splat sets in two scissor passes of the same frame —
// both passes would draw the same buffer (and mid-rebuild it briefly holds
// both sets, which flickers). Separate scene+renderer per run keeps each
// pane's splats isolated. lodSplatCount is the per-frame budget (Spark
// desktop default 2.5M); the detail slider scales it live via lodSplatScale.
const LOD_SPLAT_COUNT = 1_500_000;
function makeSpark() {
  return new SparkRenderer({ renderer, lodSplatCount: LOD_SPLAT_COUNT });
}
function makeRunScene() {
  const s = new THREE.Scene();
  s.background = new THREE.Color(0x101014);
  s.add(new THREE.AmbientLight(0xffffff, 1.2));
  const sun = new THREE.DirectionalLight(0xffffff, 1.0);
  sun.position.set(3, 10, 4);
  s.add(sun);
  return s;
}

// Live SPA switches rebuild runs, but the SparkRenderers persist per pane
// slot and are re-parented into each new run scene. This is load-bearing for
// paged RAD models: Spark bakes `mesh.paged.pager` into the mesh's dyno
// shader graph at build time and never rebuilds it while the PagedSplats
// object lives, so a pooled mesh must only ever be rendered by the
// SparkRenderer whose pager it was first bound to. Keeping one renderer (and
// therefore one pager) per pane slot for the page's lifetime guarantees
// that, and keeps the pager's resident page pool warm across switches.
// Static bundles never rebuild runs, so each run owns a throwaway renderer.
const paneSparks = [];
function paneSpark(index) {
  if (!paneSparks[index]) paneSparks[index] = makeSpark();
  return paneSparks[index];
}

// Fat-line materials shared by (color, width); resolution must track canvas.
const lineMaterials = [];
const _lineMatCache = new Map();
function lineMaterial(color, widthPx) {
  const key = color + ":" + widthPx;
  let m = _lineMatCache.get(key);
  if (!m) {
    m = new LineMaterial({ color, linewidth: widthPx });
    m.resolution.set(viewport.clientWidth || 1, viewport.clientHeight || 1);
    _lineMatCache.set(key, m);
    lineMaterials.push(m);
  }
  return m;
}

function frustumLines(fov, aspect, scale, color, widthPx = 2.5) {
  const hh = Math.tan(fov / 2) * scale, hw = hh * aspect, z = -scale;
  const c = [[-hw, -hh, z], [hw, -hh, z], [hw, hh, z], [-hw, hh, z]];
  const pts = [];
  for (let i = 0; i < 4; i++) pts.push(0, 0, 0, ...c[i], ...c[i], ...c[(i + 1) % 4]);
  const geo = new LineSegmentsGeometry();
  geo.setPositions(pts);
  return new LineSegments2(geo, lineMaterial(color, widthPx));
}

function srgbToLinearAttr(rgbU8) {
  const out = new Float32Array(rgbU8.length);
  for (let i = 0; i < rgbU8.length; i++) out[i] = Math.pow(rgbU8[i] / 255, 2.2);
  return out;
}

// --- resource teardown helpers ------------------------------------------------

// Fat-line materials come from the module-level _lineMatCache (shared across
// runs) and must survive every run disposal.
function disposeRunMaterial(material) {
  if (!material || material.isLineMaterial) return;
  for (const value of Object.values(material)) {
    if (value?.isTexture) value.dispose();
  }
  // Own-enumerable sweep above misses getter-backed, Symbol-keyed, and
  // uniform-held textures; shader uniforms are the common hiding place.
  for (const uniform of Object.values(material.uniforms ?? {})) {
    if (uniform?.value?.isTexture) uniform.value.dispose();
  }
  material.dispose();
}

// Frees every geometry/material/texture reachable from root. Caller must have
// detached pooled SplatMeshes and the SparkRenderer first — their resources
// are owned elsewhere and are never touched here.
function disposeObjectTree(root) {
  root.traverse((node) => {
    node.geometry?.dispose?.();
    const materials = Array.isArray(node.material) ? node.material : [node.material];
    for (const material of materials) disposeRunMaterial(material);
  });
}

// --- one run (method): its own path, point cloud, frusta, splats -----------

class Run {
  // sharedSpark: a persistent pane-slot SparkRenderer (live mode). Omitted =
  // the run owns a throwaway renderer (static bundles). Shared renderers are
  // attached to the scene later, by applyManifest after the run swap, so the
  // outgoing runs keep theirs until the frame the new runs take over.
  constructor(rm, sharedSpark = null) {
    this.rm = rm;
    this._disposed = false;
    this.scene = makeRunScene();
    this.ownsSpark = !sharedSpark;
    this.spark = sharedSpark ?? makeSpark();
    if (this.ownsSpark) this.scene.add(this.spark);
    this.frames = rm.frames;
    this.times = this.frames.map((f) => f.time);
    this.tEnd = rm.t_end || 1;
    const intr = rm.intrinsics;
    this.fov = 2 * Math.atan(intr.height / (2 * intr.fy));
    this.aspect = intr.width / intr.height;

    this.framePos = [];
    this.frameQuat = [];
    for (const f of this.frames) {
      const m = new THREE.Matrix4().set(...f.c2w);
      const p = new THREE.Vector3(), q = new THREE.Quaternion(), s = new THREE.Vector3();
      m.decompose(p, q, s);
      this.framePos.push(p);
      this.frameQuat.push(q);
    }
    this.center = this.framePos.reduce((a, p) => a.add(p.clone()), new THREE.Vector3())
      .divideScalar(this.framePos.length || 1);
    this.extent = new THREE.Box3().setFromPoints(this.framePos)
      .getSize(new THREE.Vector3()).length() || 10;

    this.pointsBuf = null;
    this.contamBuf = null;
    const n = this.frames.length;
    this.chunkNodes = new Array(n).fill(null);
    this.contamNodes = new Array(n).fill(null);
    this.pointsMat = new THREE.PointsMaterial({ size: 0.02, vertexColors: true });
    this.contamMat = new THREE.PointsMaterial({ size: 0.045, color: 0xff0000 });
    this.splatMeshes = new Map();
    this._thumbTex = new Map();
  }

  async load(onProgress = () => {}) {
    const parts = [0, 0];
    const update = (index, value) => {
      parts[index] = value;
      onProgress((parts[0] + parts[1]) / 2, "Loading replay buffers");
    };
    [this.pointsBuf, this.contamBuf] = await Promise.all([
      fetchArrayBufferWithProgress(this.rm.points.bin, (value) => update(0, value)),
      fetchArrayBufferWithProgress(this.rm.contam.bin, (value) => update(1, value)),
    ]);
    onProgress(0.96, "Building replay overlays");
    this._build();
    onProgress(1, "Replay ready");
    return this;
  }

  _build() {
    this.historyNodes = this.frames.map((_f, i) => {
      const node = frustumLines(this.fov, this.aspect, 0.06, 0xa0a0a0, 2.5);
      node.position.copy(this.framePos[i]);
      node.quaternion.copy(this.frameQuat[i]);
      node.visible = false;
      this.scene.add(node);
      return node;
    });
    this.evalNodes = this._staticFrusta(this.rm.eval_frusta,
      (e) => STRATUM_COLORS[e.stratum] ?? 0xc8c8c8, this.rm.eval_intrinsics);
    this.sharedNodes = this._staticFrusta(this.rm.shared_frusta,
      (e) => FACE_COLORS[e.stratum] ?? SHARED_COLORS[e.cls] ?? 0xaaaaaa,
      this.rm.shared_intrinsics);
    this.sharedPointNodes = this._standingPoints(this.rm.shared_frusta);

    const seg = [];
    for (let i = 0; i < this.framePos.length - 1; i++) {
      seg.push(this.framePos[i].x, this.framePos[i].y, this.framePos[i].z,
        this.framePos[i + 1].x, this.framePos[i + 1].y, this.framePos[i + 1].z);
    }
    const trailGeo = new LineSegmentsGeometry();
    trailGeo.setPositions(seg.length ? seg : [0, 0, 0, 0, 0, 0]);
    this.trail = new LineSegments2(trailGeo, lineMaterial(TRAIL_COLOR, 4));
    trailGeo.instanceCount = 0;
    this.trail.visible = false;
    this.scene.add(this.trail);

    this.marker = new THREE.Group();
    this.marker.add(frustumLines(this.fov, this.aspect, 0.35, TRAIL_COLOR, 3.5));
    this.markerImg = new THREE.Mesh(
      new THREE.PlaneGeometry(2 * Math.tan(this.fov / 2) * 0.35 * this.aspect,
        2 * Math.tan(this.fov / 2) * 0.35),
      new THREE.MeshBasicMaterial({ side: THREE.DoubleSide }),
    );
    this.markerImg.position.z = -0.35;
    this.marker.add(this.markerImg);
    this.marker.visible = false;
    this.scene.add(this.marker);

    const gltf = new GLTFLoader();
    this.distractorGroups = (this.rm.distractors || []).map((track) => {
      const group = new THREE.Group();
      group.visible = false;
      this.scene.add(group);
      const fallback = () => {
        if (this._disposed) return;
        group.add(new THREE.Mesh(
          new THREE.IcosahedronGeometry(0.15, 2),
          new THREE.MeshStandardMaterial({ color: 0xff7800 })));
      };
      if (track.glb) {
        gltf.load(track.glb, (g) => {
          // A switch may retire this run while a GLB is still streaming in.
          if (this._disposed) {
            disposeObjectTree(g.scene);
            return;
          }
          group.add(g.scene);
        }, undefined, fallback);
      } else {
        fallback();
      }
      return group;
    });
  }

  _staticFrusta(list, colorOf, intr) {
    // An eval set may use a different camera from the run that is being
    // replayed (the cube set is a square 90 deg frustum). Draw each set with
    // its own shape; fall back to the run's for bundles that predate this.
    const fov = intr ? 2 * Math.atan(intr.height / (2 * intr.fy)) : this.fov;
    const aspect = intr ? intr.width / intr.height : this.aspect;
    return (list || []).map((e) => {
      const m = new THREE.Matrix4().set(...e.c2w);
      const node = frustumLines(fov, aspect, 0.07, colorOf(e), 2.5);
      m.decompose(node.position, node.quaternion, new THREE.Vector3());
      node.visible = false;
      this.scene.add(node);
      return node;
    });
  }

  // One marker per distinct camera origin: a cube set puts six views at each
  // standing point, and the set of points is itself worth seeing (coverage of
  // the navigable area, and whether any room went unsampled).
  _standingPoints(list) {
    const seen = new Map();
    for (const e of list || []) {
      const m = new THREE.Matrix4().set(...e.c2w);
      const p = new THREE.Vector3().setFromMatrixPosition(m);
      const key = [p.x, p.y, p.z].map((v) => v.toFixed(3)).join(",");
      if (!seen.has(key)) seen.set(key, p);
    }
    if (seen.size >= (list || []).length) return [];  // no shared origins
    const geo = new THREE.SphereGeometry(0.035, 12, 8);
    const mat = new THREE.MeshBasicMaterial({ color: STANDING_POINT_COLOR });
    return [...seen.values()].map((p) => {
      const node = new THREE.Mesh(geo, mat);
      node.position.copy(p);
      node.visible = false;
      this.scene.add(node);
      return node;
    });
  }

  captureIndexAt(t) {
    let lo = 0, hi = this.times.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (this.times[mid] <= t + 1e-9) lo = mid; else hi = mid - 1;
    }
    return lo;
  }

  _makeChunk(buf, entry, colored) {
    const n = entry.count;
    if (!n) return false;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position",
      new THREE.BufferAttribute(new Float32Array(buf, entry.offset, n * 3), 3));
    if (colored) {
      const rgb = new Uint8Array(buf, entry.offset + n * 12, n * 3);
      geo.setAttribute("color", new THREE.BufferAttribute(srgbToLinearAttr(rgb), 3));
    }
    const node = new THREE.Points(geo, colored ? this.pointsMat : this.contamMat);
    node.visible = false;
    this.scene.add(node);
    return node;
  }

  ensureChunk(i) {
    if (this.chunkNodes[i] === null)
      this.chunkNodes[i] = this._makeChunk(this.pointsBuf, this.rm.points.chunks[i], true);
    return this.chunkNodes[i];
  }
  ensureContam(i) {
    if (this.contamNodes[i] === null)
      this.contamNodes[i] = this._makeChunk(this.contamBuf, this.rm.contam.chunks[i], false);
    return this.contamNodes[i];
  }

  hasReconstruction(name) {
    return !!name && this.rm.reconstructions.some((r) => r.name === name);
  }

  ensureSplat(name, onProgress = () => {}) {
    if (this.splatMeshes.has(name)) return this.splatMeshes.get(name);
    const rec = this.rm.reconstructions.find((r) => r.name === name);
    if (!rec) return null;
    const mesh = acquireSplatMesh(rec, onProgress, this.spark);
    mesh.visible = false;
    // scene.add re-parents when the pool hands back a mesh another run held.
    this.scene.add(mesh);
    this.splatMeshes.set(name, mesh);
    return mesh;
  }

  async loadSplat(name, onProgress = () => {}) {
    const mesh = this.ensureSplat(name, onProgress);
    if (!mesh) return;
    const record = mesh.userData.splatPoolRecord;
    if (record) record.progressSink = onProgress;
    await mesh.initialized;
    onProgress(1, "Model ready");
  }

  _thumb(k) {
    if (!this._thumbTex.has(k)) {
      const tex = new THREE.TextureLoader().load(this.frames[k].thumb);
      tex.colorSpace = THREE.SRGBColorSpace;
      this._thumbTex.set(k, tex);
    }
    return this._thumbTex.get(k);
  }

  _poseDistractors(t) {
    (this.rm.distractors || []).forEach((track, d) => {
      const rows = track.track;
      if (!rows.length) return;
      const x = Math.min(t / track.dt, rows.length - 1);
      const i = Math.floor(x), a = x - i;
      const r0 = rows[i], r1 = rows[Math.min(i + 1, rows.length - 1)];
      const g = this.distractorGroups[d];
      g.position.set(r0[0] + a * (r1[0] - r0[0]), r0[1] + a * (r1[1] - r0[1]),
        r0[2] + a * (r1[2] - r0[2]));
      let dyaw = r1[3] - r0[3];
      dyaw -= 2 * Math.PI * Math.round(dyaw / (2 * Math.PI));
      g.rotation.set(0, r0[3] + a * dyaw, 0);
    });
  }

  // Apply sim time + layers with this run's own reconstruction; returns k.
  apply(t, layers, reconName) {
    const tt = Math.min(t, this.tEnd);
    const k = this.captureIndexAt(tt);
    // Readiness gate: even with the layer checked, the point cloud stays up
    // until the splat is parsed; the swap happens automatically on the first
    // frame after the background load flags the mesh ready.
    const showSplat = layers.splat && this.hasReconstruction(reconName)
      && Boolean(this.splatMeshes.get(reconName)?.userData.splatReady);
    const showPoints = !showSplat;

    for (let i = 0; i < this.frames.length; i++) {
      const on = showPoints && i <= k;
      if (on || this.chunkNodes[i]) {
        const nd = on ? this.ensureChunk(i) : this.chunkNodes[i];
        if (nd) nd.visible = on;
      }
      const cOn = showPoints && layers.contamination && i <= k;
      if (cOn || this.contamNodes[i]) {
        const nd = cOn ? this.ensureContam(i) : this.contamNodes[i];
        if (nd) nd.visible = cOn;
      }
      this.historyNodes[i].visible = layers.history && i <= k;
    }

    for (const [name, mesh] of this.splatMeshes) {
      mesh.visible = showSplat && name === reconName && Boolean(mesh.userData.splatReady);
    }

    const segs = Math.min(k + (tt > this.times[k] ? 1 : 0), this.frames.length - 1);
    this.trail.geometry.instanceCount = Math.max(0, segs);
    this.trail.visible = true;

    if (k < this.frames.length - 1) {
      const span = Math.max(this.times[k + 1] - this.times[k], 1e-9);
      const alpha = THREE.MathUtils.clamp((tt - this.times[k]) / span, 0, 1);
      this.marker.position.lerpVectors(this.framePos[k], this.framePos[k + 1], alpha);
      this.marker.quaternion.slerpQuaternions(this.frameQuat[k], this.frameQuat[k + 1], alpha);
    } else {
      this.marker.position.copy(this.framePos[this.frames.length - 1]);
      this.marker.quaternion.copy(this.frameQuat[this.frames.length - 1]);
    }
    this.markerImg.material.map = this._thumb(k);
    this.markerImg.material.needsUpdate = true;
    // A thumbnail plane at its own capture pose would fill the inspection
    // view and masquerade as the reconstructed scene. Keep it only at a
    // distance; the panel still displays the source image for comparison.
    this.markerImg.visible = camera.position.distanceToSquared(this.marker.position) > 0.25;
    this.marker.visible = true;

    this.evalNodes.forEach((n) => (n.visible = layers.eval));
    this.sharedNodes.forEach((n) => (n.visible = layers.shared));
    (this.sharedPointNodes || []).forEach((n) => (n.visible = layers.shared));
    this.distractorGroups.forEach((g) => (g.visible = layers.distractors));
    this._poseDistractors(tt);
    return k;
  }

  // Tear down every GPU/JS resource owned by this run. Idempotent: the guard
  // flag makes repeat calls and partially-constructed runs safe. Pooled splat
  // meshes are only detached (meshCache owns them); shared fat-line materials
  // are skipped inside disposeObjectTree.
  async dispose() {
    if (this._disposed) return;
    this._disposed = true;

    for (const mesh of this.splatMeshes.values()) {
      // A mesh handed off to a newer run is parented to that run's scene now;
      // only detach the ones this scene still owns.
      if (mesh.parent === this.scene) this.scene.remove(mesh);
    }
    this.splatMeshes.clear();

    const spark = this.spark;
    if (this.spark.parent === this.scene) this.scene.remove(spark);
    if (this.ownsSpark) {
      quiesceSparkRenderer(spark);
      await waitForSparkIdle(spark);
      releaseSparkRenderer(spark);
    }
    // Shared pane-slot renderers persist: their pager (and its resident RAD
    // pages) must outlive this run so pooled meshes stay valid and warm.

    disposeObjectTree(this.scene);
    this.scene.clear();
    this.pointsMat.dispose();
    this.contamMat.dispose();
    for (const texture of this._thumbTex.values()) texture.dispose();
    this._thumbTex.clear();

    this.chunkNodes = [];
    this.contamNodes = [];
    this.historyNodes = [];
    this.evalNodes = [];
    this.sharedNodes = [];
    this.sharedPointNodes = [];
    this.distractorGroups = [];
    this.framePos = [];
    this.frameQuat = [];
    this.pointsBuf = null;
    this.contamBuf = null;
  }
}

// --- splat mesh pool (shared across SPA view switches) ----------------------

// RAD pages and GPU buffers are the expensive part of a view switch, so splat
// meshes are pooled by reconstruction URL (never by name — both panes can
// offer a "gsplat" backed by different URLs) and re-parented into whichever
// run scene needs them. A pool can hold several records per URL because a
// mesh lives in exactly one scene and compare mode may show the same
// reconstruction in both panes.
const MAX_RESIDENT_MESHES = 4;
const meshCache = new Map(); // reconstruction URL -> pool records
let splatUsageClock = 0;

function createSplatPoolRecord(rec, url) {
  const record = { url, mesh: null, progressSink: null, lastUsed: 0 };
  const options = {
    url,
    onProgress: (event) => {
      if (!record.progressSink) return;
      const ratio = event?.total ? Math.min(0.99, event.loaded / event.total) : 0;
      record.progressSink(ratio, rec.rad ? "Streaming RAD model" : "Loading PLY model");
    },
  };
  if (rec.rad) options.paged = true;
  else options.lod = true;
  const mesh = new SplatMesh(options);
  mesh.quaternion.copy(SPLAT_QUATERNION);
  mesh.userData.splatPoolRecord = record;
  record.mesh = mesh;
  // Readiness gate for Run.apply: the point cloud keeps showing until the
  // splat's first frame is parsed, then the swap happens on the next frame.
  mesh.initialized.then(() => { mesh.userData.splatReady = true; }, () => {});
  return record;
}

function splatMeshInUse(mesh) {
  return runs.some((run) => {
    for (const held of run.splatMeshes.values()) if (held === mesh) return true;
    return false;
  });
}

// A paged mesh's shader graph permanently reads the pager it was first bound
// to (Spark bakes the closure at dyno-graph build and never rebuilds it), so
// a pooled RAD mesh is only reusable under the SparkRenderer owning that same
// pager. A never-displayed mesh (pager still unset) binds to whichever
// renderer shows it first; PLY meshes carry no pager and move freely.
function pagerCompatible(mesh, spark) {
  const pager = mesh.paged?.pager;
  return !pager || pager === spark.pager;
}

function acquireSplatMesh(rec, onProgress, spark) {
  const url = rec.rad || rec.ply;
  let pool = meshCache.get(url);
  if (!pool) {
    pool = [];
    meshCache.set(url, pool);
  }
  let record = pool.find((entry) =>
    !splatMeshInUse(entry.mesh) && pagerCompatible(entry.mesh, spark));
  if (!record) {
    record = createSplatPoolRecord(rec, url);
    pool.push(record);
  }
  record.progressSink = onProgress;
  record.lastUsed = ++splatUsageClock;
  return record.mesh;
}

// Keep pool-resident paged meshes registered with their pane renderer's LOD
// worker. Spark disposes a splats' LOD-tree record after ~3s out of view, but
// its pager keeps the pages resident — and a chunk that is already resident
// never re-emits the lodTree/rootPage registration on re-display, so the
// re-acquired mesh would be stuck skipped by the LOD traverse. Touching the
// records while the mesh is pooled prevents the cleanup; records of meshes
// evicted from the pool go stale and are cleaned up by Spark as designed.
let lastLodTouch = 0;
function touchPooledLodRecords(now) {
  if (now - lastLodTouch < 1000) return;
  lastLodTouch = now;
  for (const spark of paneSparks) {
    const lodIds = spark?.lodIds;
    if (!lodIds) continue;
    for (const pool of meshCache.values()) {
      for (const { mesh } of pool) {
        const record = mesh.paged && lodIds.get(mesh.paged);
        if (record) record.lastTouched = now;
      }
    }
  }
}

// Evict idle meshes beyond the resident cap, least recently used first.
function retireSplatMeshes() {
  const idle = [];
  let resident = 0;
  for (const pool of meshCache.values()) {
    resident += pool.length;
    for (const record of pool) {
      if (!splatMeshInUse(record.mesh)) idle.push(record);
    }
  }
  idle.sort((a, b) => a.lastUsed - b.lastUsed);
  for (const record of idle) {
    if (resident <= MAX_RESIDENT_MESHES) break;
    record.mesh.parent?.remove(record.mesh);
    record.mesh.dispose?.();
    const pool = meshCache.get(record.url);
    pool.splice(pool.indexOf(record), 1);
    if (!pool.length) meshCache.delete(record.url);
    releaseOrphanedPager(record.mesh);
    resident -= 1;
  }
}

// --- SparkRenderer teardown ---------------------------------------------------

// A pooled RAD mesh keeps streaming through the SplatPager of the first
// SparkRenderer that displayed it (spark never reassigns mesh.paged.pager), so
// a pager referenced by the pool must outlive its SparkRenderer. The pool
// releases it once the last referencing mesh is evicted.
function pooledMeshReferencesPager(pager) {
  for (const pool of meshCache.values()) {
    for (const record of pool) {
      if (record.mesh.paged?.pager === pager) return true;
    }
  }
  return false;
}

// Dispose a pager no live run and no pooled mesh can still reference.
//
// Pager ownership protocol (why this disposes exactly once, never twice):
//   (a) a pager is either attached to a live SparkRenderer or pool-owned,
//       never both — releaseSparkRenderer detaches it (spark.pager =
//       undefined) only when pooledMeshReferencesPager shows pool refs exist;
//   (b) callers invoke this only AFTER splicing the mesh's record out of the
//       pool (see retireSplatMeshes), so pooledMeshReferencesPager === false
//       here means the pool's last reference is already gone;
//   (c) therefore reaching pager.dispose() implies zero remaining owners, and
//       the detachment in (a) guarantees the SparkRenderer that created the
//       pager no longer holds it — exactly one dispose, no double-free.
function releaseOrphanedPager(mesh) {
  const pager = mesh.paged?.pager;
  if (!pager) return;
  if (runs.some((run) => run.spark?.pager === pager)) return;
  // Persistent pane-slot renderers keep their pager (and warm page pool) even
  // while no current run holds them — e.g. pane B's after compare turns off.
  if (paneSparks.some((spark) => spark?.pager === pager)) return;
  if (pooledMeshReferencesPager(pager)) return;
  pager.dispose();
}

// Stop every async surface of a SparkRenderer so disposing it cannot reject an
// in-flight worker call ("Worker terminate" unhandled rejections) or resurrect
// workers from a pending timer. Spark 2.1.0 fields; degrade gracefully if a
// future version renames them.
function quiesceSparkRenderer(spark) {
  spark.autoUpdate = false;
  if (spark.updateTimeoutId !== -1) {
    clearTimeout(spark.updateTimeoutId);
    spark.updateTimeoutId = -1;
  }
  if (spark.sortTimeoutId !== -1) {
    clearTimeout(spark.sortTimeoutId);
    spark.sortTimeoutId = -1;
  }
  spark.sortDirty = false;
  spark.lodDirty = false;
}

function sparkRendererBusy(spark) {
  if (spark.sorting) return true;
  const lodWorker = spark.lodWorker;
  return Boolean(lodWorker && lodWorker.queue != null);
}

const SPARK_IDLE_TIMEOUT_MS = 5000;
async function waitForSparkIdle(spark) {
  const deadline = performance.now() + SPARK_IDLE_TIMEOUT_MS;
  while (sparkRendererBusy(spark) && performance.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  if (sparkRendererBusy(spark)) {
    console.warn("retiring a spark renderer that is still busy");
  }
}

function releaseSparkRenderer(spark) {
  if (spark.pager && pooledMeshReferencesPager(spark.pager)) {
    spark.pager = undefined; // pool-owned now; see releaseOrphanedPager
  }
  // Capture before dispose: SparkRenderer.dispose() may clear or invalidate
  // its own geometry/material references.
  const geometry = spark.geometry;
  const material = spark.material;
  // A spark retired while still busy (waitForSparkIdle timed out) can throw
  // or reject in-flight worker work here; disposal must never break the
  // teardown chain, so log and continue.
  try {
    // SparkRenderer.dispose() (2.1.0) frees render targets, splat
    // accumulators, the ordering texture, LOD instances, workers, and the
    // pager — but not the per-instance mesh geometry/material.
    spark.dispose?.();
  } catch (error) {
    console.warn("spark renderer disposal failed; continuing teardown", error);
  }
  geometry?.dispose?.();
  material?.dispose?.();
}

// --- deferred splat loading (live mode): registry, chip, soft waits ----------

// Live manifests show the point cloud immediately and stream 3DGS in the
// background: a floating chip reports aggregate progress, and the blocking
// overlay is reserved for an explicit user wait (the 3DGS layer checkbox).
// Registry entries are keyed by Run; `splatLoadEpoch` is bumped by every
// applyManifest so callbacks from superseded manifests never write to the UI.
const splatLoadRegistry = new Map(); // Run -> { recon, promise, progress, message, epoch, failed }
let splatLoadEpoch = 0;
let activeSoftWait = null; // { cancelled, paneRows: Map<Run, rowIndex> }
let loadChipHideTimer = 0;

function currentSplatLoads() {
  return [...splatLoadRegistry.values()].filter((entry) => entry.epoch === splatLoadEpoch);
}

function hideLoadChip() {
  clearTimeout(loadChipHideTimer);
  loadChip.hidden = true;
}

function flashLoadChip(label, className, holdMs) {
  clearTimeout(loadChipHideTimer);
  loadChipLabel.textContent = label;
  loadChip.className = className;
  loadChipBar.classList.remove("active");
  loadChipBar.style.width = "100%";
  loadChipPct.textContent = "";
  loadChip.hidden = false;
  loadChipHideTimer = setTimeout(hideLoadChip, holdMs);
}

function updateLoadChip() {
  const entries = currentSplatLoads();
  const pending = entries.filter((entry) => entry.progress < 1 && !entry.failed);
  if (!pending.length) {
    if (entries.some((entry) => entry.failed)) {
      flashLoadChip("3DGS failed", "error", 3000);
    } else if (entries.length) {
      flashLoadChip("3DGS ready ✓", "ready", 1500);
    } else {
      hideLoadChip();
    }
    return;
  }
  clearTimeout(loadChipHideTimer);
  const mean = pending.reduce((sum, entry) => sum + entry.progress, 0) / pending.length;
  loadChipLabel.textContent = "3DGS";
  loadChip.className = "";
  loadChipBar.classList.add("active");
  loadChipBar.style.width = `${Math.max(2, Math.round(100 * mean))}%`;
  loadChipPct.textContent = `${Math.round(100 * mean)}%`;
  loadChip.hidden = false;
}

// Detach a failed mesh from its run and evict its pool record so a later
// attempt starts from a fresh fetch instead of awaiting a rejected mesh.
function evictFailedSplat(run, recon) {
  const mesh = run.splatMeshes.get(recon);
  if (!mesh) return; // run already disposed (stale epoch) — nothing to evict
  if (mesh.parent === run.scene) run.scene.remove(mesh);
  run.splatMeshes.delete(recon);
  if (splatMeshInUse(mesh)) return; // another live run re-acquired the same mesh
  const record = mesh.userData.splatPoolRecord;
  if (!record) return;
  const pool = meshCache.get(record.url);
  if (pool) {
    const index = pool.indexOf(record);
    if (index !== -1) pool.splice(index, 1);
    if (!pool.length) meshCache.delete(record.url);
  }
  record.mesh.dispose?.();
  releaseOrphanedPager(record.mesh); // after the pool splice: single-owner protocol
}

// Start (or adopt) background loads for the given panes. loadSplat acquires
// the pooled mesh synchronously — re-parenting it into the run's scene — and
// re-points the pool record's progress sink, so a mesh a previous run was
// still streaming is shared, never re-fetched.
function beginBackgroundSplatLoads(panes) {
  for (const pane of panes) {
    if (!pane.recon) continue;
    const run = runs[pane.runIndex];
    const mesh = run.splatMeshes.get(pane.recon);
    if (mesh?.userData.splatReady) continue; // already warm in the pool
    const entry = {
      recon: pane.recon,
      promise: null,
      progress: 0.02,
      message: "Streaming RAD model",
      epoch: splatLoadEpoch,
      failed: false,
    };
    const mirrorToSoftWait = (progress, message) => {
      const row = activeSoftWait?.paneRows.get(run);
      if (row !== undefined) setViewerLoad(row, progress, message);
    };
    entry.promise = run.loadSplat(pane.recon, (progress, message) => {
      if (entry.epoch !== splatLoadEpoch) return;
      entry.progress = Math.max(entry.progress, Math.min(1, progress));
      entry.message = message;
      updateLoadChip();
      mirrorToSoftWait(entry.progress, message);
    }).then(() => {
      if (entry.epoch !== splatLoadEpoch) return;
      entry.progress = 1;
      entry.message = "Model ready";
      updateLoadChip();
      mirrorToSoftWait(1, "Model ready");
    }, (error) => {
      console.error("background 3DGS load failed", error);
      try {
        evictFailedSplat(run, pane.recon);
      } catch (_e) { /* best-effort cleanup */ }
      if (entry.epoch !== splatLoadEpoch) return;
      entry.failed = true;
      updateLoadChip();
      throw error; // reject so an awaiting soft wait can react
    });
    entry.promise.catch(() => {}); // swallow when nobody is soft-waiting
    splatLoadRegistry.set(run, entry);
  }
}

function paneSplatReady(pane) {
  if (!pane.recon) return true; // replay-only pane: nothing to wait for
  const mesh = runs[pane.runIndex].splatMeshes.get(pane.recon);
  return Boolean(mesh?.userData.splatReady);
}

// Mark any active soft wait stale without touching the overlay: manifest
// switches own the overlay while they load; checkbox/minimize paths hide it.
function invalidateSoftWait() {
  if (activeSoftWait) activeSoftWait.cancelled = true;
  activeSoftWait = null;
  loadingMin.hidden = true;
}

function dismissSoftWaitOverlay() {
  invalidateSoftWait();
  loadingOverlay.hidden = true;
}

// Open the blocking overlay as a *soft* wait on the in-flight background
// loads: same fetches, minimizable into the chip, cancelled by unchecking.
function beginSoftWait() {
  const panes = [paneA, ...(paneState.compare ? [paneB] : [])];
  const waitingPanes = panes.filter((pane) => !paneSplatReady(pane));
  if (!waitingPanes.length) return;
  // Entries that failed (or were evicted) restart fresh; healthy in-flight
  // entries are awaited — never duplicated.
  const restartPanes = waitingPanes.filter((pane) => {
    const entry = splatLoadRegistry.get(runs[pane.runIndex]);
    return !entry || entry.failed || entry.epoch !== splatLoadEpoch;
  });
  if (restartPanes.length) beginBackgroundSplatLoads(restartPanes);
  const token = { cancelled: false, paneRows: new Map() };
  activeSoftWait = token;
  viewerLoads = waitingPanes.map((pane, row) => {
    const run = runs[pane.runIndex];
    token.paneRows.set(run, row);
    const entry = splatLoadRegistry.get(run);
    return {
      tag: pane === paneA ? "A" : "B",
      progress: Math.max(0.02, entry?.progress ?? 0),
      message: entry?.message ?? "Streaming RAD model",
    };
  });
  renderViewerLoads();
  loadingTitle.textContent = "Loading 3DGS. Minimize to keep exploring.";
  loadingMin.hidden = false;
  loadingOverlay.hidden = false;
  const waits = waitingPanes.map((pane) => splatLoadRegistry.get(runs[pane.runIndex])?.promise);
  Promise.all(waits).then(() => {
    if (token.cancelled || activeSoftWait !== token) return;
    activeSoftWait = null;
    loadingOverlay.hidden = true;
    loadingMin.hidden = true;
    hideLoadChip();
  }, () => {
    if (token.cancelled || activeSoftWait !== token) return;
    activeSoftWait = null;
    // Keep the overlay up; the minimize button is the dismiss path.
    loadingTitle.textContent = "3DGS failed to load";
    state.layers.splat = false;
    syncManifestUi();
  });
}

function onSplatToggleLive(on) {
  if (!on) {
    dismissSoftWaitOverlay(); // cancel the wait; the load keeps warming the pool
    return;
  }
  beginSoftWait();
}

// Chip click re-opens the soft-wait overlay while a load is active (without
// touching the checkbox); the minimize button collapses back to the chip.
loadChip.addEventListener("click", () => {
  if (activeSoftWait) return;
  const loading = currentSplatLoads().some((entry) => entry.progress < 1 && !entry.failed);
  if (loading) beginSoftWait();
});
loadingMin.addEventListener("click", dismissSoftWaitOverlay);

// --- per-manifest assembly (initial boot and SPA switches) -------------------

function preferredRecon(run) {
  return run.rm.reconstructions.length ? run.rm.reconstructions[0].name : null;
}

let runs = [];
const paneA = { runIndex: 0, recon: null };
const paneB = { runIndex: 0, recon: null };
const paneState = { compare: false };

// Build runs + panes for one manifest. One-time boot (renderer, camera,
// controls, UI listeners) lives outside; everything here is safe to re-run.
async function applyManifest(manifest) {
  // A new manifest supersedes every in-flight splat load: bump the epoch so
  // stale progress/completion callbacks can't touch the UI, drop their
  // registry entries, and cancel any soft wait bound to the previous runs.
  splatLoadEpoch += 1;
  for (const [run, entry] of splatLoadRegistry) {
    if (entry.epoch !== splatLoadEpoch) splatLoadRegistry.delete(run);
  }
  invalidateSoftWait();
  updateLoadChip();
  const runManifests = await Promise.all(manifest.runs.map(resolveRunManifest));
  runManifests.forEach((_run, index) => setViewerLoad(index, 0.08, "Loading replay buffers"));
  // Live runs borrow the persistent pane-slot SparkRenderer for their index
  // (run 0 = pane A, run 1 = pane B); static runs own theirs.
  const loadingRuns = runManifests.map((rm, index) => new Run(rm, live ? paneSpark(index) : null)
    .load((progress, message) => setViewerLoad(index, 0.08 + 0.27 * progress, message)));
  let nextRuns;
  try {
    nextRuns = await Promise.all(loadingRuns);
  } catch (error) {
    // A failed load rejects the whole batch; siblings that still finish would
    // otherwise be orphaned (never swapped in, never rendered). The failure
    // itself falls back to full navigation, so fire-and-forget is fine here.
    for (const loading of loadingRuns) {
      loading.then((run) => run.dispose()).catch(() => {});
    }
    throw error;
  }
  // Swap runs before acquiring splats so the previous runs' pooled meshes
  // become reusable. These assignments are synchronous, so the render loop
  // never observes a half-updated pane configuration.
  const previousRuns = runs;
  runs = nextRuns;
  // Shared renderers move to the new scenes only now, so the outgoing runs
  // rendered with theirs until this exact swap (no sparkless frame).
  for (const run of runs) {
    if (!run.ownsSpark) run.scene.add(run.spark);
  }
  paneA.runIndex = 0;
  paneA.recon = preferredRecon(runs[0]);
  paneB.runIndex = runs.length > 1 ? 1 : 0;
  paneB.recon = preferredRecon(runs[paneB.runIndex]);
  paneState.compare = manifest.mode === "compare";
  if (live) {
    // Live mode defers 3DGS: the point cloud + trajectory show immediately
    // (the overlay hides once replay buffers are ready) while splats stream
    // in the background and replace the point cloud when ready.
    const activePanes = [paneA, ...(paneState.compare ? [paneB] : [])];
    activePanes.forEach((_pane, index) => setViewerLoad(index, 1, "Replay ready"));
    // The 3DGS layer keeps the user's setting across switches: the readiness
    // gate in Run.apply shows the point cloud until each pane's splat is
    // parsed, then swaps automatically — no forced un-check, no hard wait.
    // Acquisition (re-parenting pooled meshes) is synchronous inside
    // beginBackgroundSplatLoads, so it precedes the previousRuns disposal —
    // the same handoff invariant as the swap comment above. Never awaited.
    beginBackgroundSplatLoads(activePanes);
  } else {
    try {
      await Promise.all([paneA, ...(paneState.compare ? [paneB] : [])].map(async (pane, index) => {
        if (!pane.recon) {
          setViewerLoad(index, 1, "Replay ready");
          return;
        }
        await runs[pane.runIndex].loadSplat(pane.recon, (progress, message) =>
          setViewerLoad(index, 0.35 + 0.65 * progress, message));
      }));
    } catch (error) {
      // The new runs were already swapped into `runs` above, so a splat-load
      // failure would orphan them (the error falls back to full navigation and
      // the previousRuns disposal below never runs). Tear down only the new
      // runs here — Run.dispose detaches pooled meshes rather than disposing
      // them — then rethrow so the navigation fallback still proceeds.
      await Promise.all(runs.map((run) => run.dispose?.()));
      throw error;
    }
  }
  // Retire the old runs only after the mesh handoff: pooled splats re-parented
  // into the new scenes must never be caught by the disposal traverse.
  await Promise.all(previousRuns.map((run) => run.dispose()));
  retireSplatMeshes();
}

await applyManifest(manifest);
loadingOverlay.hidden = true;

// An offset from the trajectory centroid can be inside a wall or outside an
// indoor room. A recorded capture is a useful, already-observed entry pose.
function goToCapture(index) {
  const run = runs[paneA.runIndex];
  const k = index ?? run.captureIndexAt(Math.min(state.t, run.tEnd));
  camera.position.copy(run.framePos[k]);
  camera.quaternion.copy(run.frameQuat[k]);
  enableRotateAroundMe();
}
goToCapture(0);
const restoredCamera = restoredView.camera;
const validVec3 = (value) => Array.isArray(value) && value.length === 3
  && value.every(Number.isFinite);
if (validVec3(restoredCamera?.position) && validVec3(restoredCamera?.target)) {
  camera.position.fromArray(restoredCamera.position);
  controls.target.fromArray(restoredCamera.target);
  controls.update();
} else {
  enableRotateAroundMe();
}
const FLY_BASE = runs[0].extent * 0.25;

function resize() {
  const w = viewport.clientWidth, h = viewport.clientHeight;
  renderer.setSize(w, h);
  for (const m of lineMaterials) m.resolution.set(w, h);
}
window.addEventListener("resize", resize);
resize();

function renderPane(pane, x, w, H) {
  const run = runs[pane.runIndex];
  run.apply(state.t, state.layers, pane.recon);
  renderer.setViewport(x, 0, w, H);
  renderer.setScissor(x, 0, w, H);
  renderer.setScissorTest(true);
  camera.aspect = w / H;
  camera.updateProjectionMatrix();
  renderer.render(run.scene, camera);  // each run's own scene + SparkRenderer
}

// --- UI ---------------------------------------------------------------------

const el = (id) => document.getElementById(id);
const PANE_DIMENSIONS = ["difficulty", "method", "reconstruction", "seed"];

function liveQuery(selection) {
  const query = new URLSearchParams({ round: selection.round, scene: selection.scene });
  for (const key of PANE_DIMENSIONS) query.set(`a_${key}`, selection.a[key]);
  if (selection.b) {
    query.set("compare", "1");
    for (const key of PANE_DIMENSIONS) query.set(`b_${key}`, selection.b[key]);
  }
  return query;
}

function saveNavigationState(selection) {
  if (!live) return;
  const sameScene = selection.round === live.selection.round
    && selection.scene === live.selection.scene;
  const saved = {
    savedAt: Date.now(),
    target: { round: selection.round, scene: selection.scene },
    preferences: {
      speed: state.speed,
      loop: state.loop,
      detailScale: state.detailScale,
      layers: { ...state.layers },
    },
    view: sameScene ? {
      t: state.t,
      playing: state.playing,
      camera: {
        position: camera.position.toArray(),
        target: controls.target.toArray(),
      },
    } : null,
  };
  try {
    sessionStorage.setItem(NAVIGATION_STATE_KEY, JSON.stringify(saved));
  } catch (_error) {
    // Navigation still works if storage is blocked by browser policy.
  }
}

function navigateLive(selection, message = "Preparing selected views…") {
  saveNavigationState(selection);
  showSimpleLoading(message);
  window.location.assign(`/view?${liveQuery(selection)}`);
}

// Generous: cold RAD builds can take minutes; this only guards permanent
// stalls so the catch below can fall back to classic navigation.
const LIVE_JOB_TIMEOUT_MS = 10 * 60 * 1000;

async function waitForLiveJob(jobId) {
  const deadline = Date.now() + LIVE_JOB_TIMEOUT_MS;
  while (true) {
    if (Date.now() > deadline) {
      throw new Error("view preparation timed out after 10 minutes");
    }
    const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`export job polling failed: ${response.status}`);
    const job = await response.json();
    job.panes.forEach((pane, index) =>
      setViewerLoad(index, 0.02 + 0.48 * (pane.progress / 100), pane.message));
    if (job.status === "complete") return job;
    if (job.status === "error") throw new Error(job.error || "view preparation failed");
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
}

// Switch the A/B selection without a page reload: fetch the export job +
// manifest in place, rebuild runs around the pooled meshes, and fall back to
// the classic full navigation on any failure so the user is never stuck.
async function spaSwitch(selection, { push = true, message = "Preparing selected views…" } = {}) {
  if (!live) return;
  showSimpleLoading(message);
  viewerLoads = (selection.b ? ["A", "B"] : ["A"]).map((tag) => ({
    tag, progress: 0.02, message: "Requesting view",
  }));
  renderViewerLoads();
  try {
    const response = await fetch(`/api/view?${liveQuery(selection)}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`view request failed: ${response.status}`);
    const initial = await response.json();
    if (initial.status === "error") throw new Error(initial.error || "view preparation failed");
    if (initial.status !== "complete" && !initial.job) {
      throw new Error("view request returned no export job");
    }
    const job = initial.status === "complete" ? initial : await waitForLiveJob(initial.job);
    if (!job.target) throw new Error("export job completed without a target");
    const manifestResponse = await fetch(`${job.target}manifest.json`, { cache: "no-store" });
    if (!manifestResponse.ok) {
      throw new Error(`view manifest failed to load: ${manifestResponse.status}`);
    }
    await applyManifest(await manifestResponse.json());
  } catch (error) {
    console.error("spa switch failed; falling back to full navigation", error);
    navigateLive(selection, message);
    return;
  }
  // Mutate the shared selection so the one-time control bindings stay valid.
  live.selection.a = selection.a;
  live.selection.b = selection.b;
  fillLivePane("a", selection.a);
  fillLivePane("b", selection.b || selection.a);
  panelK = -1;
  viewerLoads = [];
  renderViewerLoads();
  loadingOverlay.hidden = true;
  syncManifestUi();
  if (push) history.pushState({}, "", `/view?${liveQuery(selection)}`);
}

// Latest request wins; a switch already in flight runs to completion first.
let requestedSwitch = null;
let switchInFlight = false;

function requestLiveSwitch(selection, options = {}) {
  requestedSwitch = { selection, options };
  if (switchInFlight) return;
  switchInFlight = true;
  (async () => {
    try {
      while (requestedSwitch) {
        const next = requestedSwitch;
        requestedSwitch = null;
        await spaSwitch(next.selection, next.options);
      }
    } finally {
      switchInFlight = false;
    }
  })();
}

function selectionFromParams(params) {
  const round = params.get("round");
  const scene = params.get("scene");
  if (!round || !scene) return null;
  const pane = (prefix) => {
    const values = {};
    for (const key of PANE_DIMENSIONS) {
      const value = params.get(`${prefix}_${key}`);
      if (value === null) return null;
      values[key] = value;
    }
    return values;
  };
  const a = pane("a");
  if (!a) return null;
  if (!params.get("compare")) return { round, scene, a, b: null };
  const b = pane("b");
  return b ? { round, scene, a, b } : null;
}

window.addEventListener("popstate", () => {
  if (!live) return;
  const parsed = selectionFromParams(new URLSearchParams(window.location.search));
  if (!parsed) {
    location.reload();
    return;
  }
  if (parsed.round !== live.selection.round || parsed.scene !== live.selection.scene) {
    location.reload();
    return;
  }
  requestLiveSwitch(parsed, { push: false });
});

function paneFromView(view) {
  return Object.fromEntries(PANE_DIMENSIONS.map((key) => [key, view[key]]));
}

function liveSceneViews(round, scene) {
  return live.catalog.views.filter((view) => view.round === round && view.scene === scene);
}

function unique(items) {
  return [...new Set(items)];
}

function fillValues(select, values, selected, label = (value) => value || "replay only") {
  select.innerHTML = "";
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label(value);
    select.appendChild(option);
  }
  select.value = selected;
}

function setupLiveSelection() {
  const selection = live.selection;
  fillValues(el("live-round"), live.catalog.rounds.map((round) => round.id), selection.round,
    (id) => live.catalog.rounds.find((round) => round.id === id).label);
  const scenes = unique(live.catalog.views
    .filter((view) => view.round === selection.round).map((view) => view.scene));
  fillValues(el("live-scene"), scenes, selection.scene);

  const switchDataset = (round, scene) => {
    const first = liveSceneViews(round, scene)[0];
    if (!first) return;
    const pane = paneFromView(first);
    navigateLive({ round, scene, a: pane, b: selection.b ? { ...pane } : null },
      `Preparing ${scene}…`);
  };
  el("live-round").addEventListener("change", (e) => {
    const round = e.target.value;
    const scene = live.catalog.views.find((view) => view.round === round).scene;
    switchDataset(round, scene);
  });
  el("live-scene").addEventListener("change", (e) =>
    switchDataset(selection.round, e.target.value));
  el("live-selection").hidden = false;
}

function liveDimensionId(key, pane) {
  return `${key === "reconstruction" ? "recon" : key}-${pane}`;
}

function compatibleLiveValues(pane, key) {
  return unique(liveSceneViews(live.selection.round, live.selection.scene)
    .filter((view) => PANE_DIMENSIONS
      .every((other) => other === key || view[other] === pane[other]))
    .map((view) => view[key]));
}

function fillLivePane(tag, pane) {
  for (const key of PANE_DIMENSIONS) {
    fillValues(el(liveDimensionId(key, tag)), compatibleLiveValues(pane, key), pane[key]);
  }
}

function fillMethods(select) {
  select.innerHTML = "";
  runs.forEach((run, i) => {
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = run.rm.method;
    select.appendChild(opt);
  });
}
function fillRecons(select, runIndex, selected) {
  select.innerHTML = "";
  const recs = runs[runIndex].rm.reconstructions;
  for (const rec of recs) {
    const opt = document.createElement("option");
    opt.value = rec.name;
    opt.textContent = `${rec.name} (${(rec.count / 1e6).toFixed(1)}M)`;
    select.appendChild(opt);
  }
  select.value = selected && recs.some((r) => r.name === selected) ? selected : (recs[0]?.name ?? "");
  select.disabled = recs.length === 0;
}

// Static bundles: apply() never acquires splats, so pane-control changes warm
// the newly selected reconstruction in the background (as the old lazy
// ensureSplat-in-apply path did); the readiness gate keeps the point cloud up
// until the mesh is parsed.
function warmPaneSplat(pane) {
  const run = runs[pane.runIndex];
  if (!pane.recon || !run?.hasReconstruction(pane.recon)) return;
  if (run.splatMeshes.get(pane.recon)?.userData.splatReady) return;
  run.loadSplat(pane.recon).catch((error) => {
    console.error("3DGS load failed", error);
    try {
      evictFailedSplat(run, pane.recon);
    } catch (_e) { /* best-effort cleanup */ }
  });
}

function setupStaticPaneControls() {
  fillMethods(el("method-a"));
  fillMethods(el("method-b"));
  el("method-a").value = String(paneA.runIndex);
  el("method-b").value = String(paneB.runIndex);
  fillRecons(el("recon-a"), paneA.runIndex, paneA.recon);
  fillRecons(el("recon-b"), paneB.runIndex, paneB.recon);
  if (runs.length > 1) el("compare-row").hidden = false;

  el("compare").addEventListener("change", (e) => {
    paneState.compare = e.target.checked;
    if (paneState.compare) warmPaneSplat(paneB);
    applyCompareVisibility();
    updateSummary();
  });
  el("method-a").addEventListener("change", (e) => {
    paneA.runIndex = parseInt(e.target.value, 10);
    paneA.recon = preferredRecon(runs[paneA.runIndex]);
    fillRecons(el("recon-a"), paneA.runIndex, paneA.recon);
    warmPaneSplat(paneA);
    updateSummary();
  });
  el("method-b").addEventListener("change", (e) => {
    paneB.runIndex = parseInt(e.target.value, 10);
    paneB.recon = preferredRecon(runs[paneB.runIndex]);
    fillRecons(el("recon-b"), paneB.runIndex, paneB.recon);
    warmPaneSplat(paneB);
    updateSummary();
  });
  el("recon-a").addEventListener("change", (e) => {
    paneA.recon = e.target.value;
    warmPaneSplat(paneA);
  });
  el("recon-b").addEventListener("change", (e) => {
    paneB.recon = e.target.value;
    warmPaneSplat(paneB);
  });
}

function setupLivePaneControls() {
  const selection = live.selection;
  const paneBSelection = selection.b || selection.a;
  document.querySelectorAll(".live-only").forEach((node) => (node.hidden = false));
  fillLivePane("a", selection.a);
  fillLivePane("b", paneBSelection);
  el("compare-row").hidden = liveSceneViews(selection.round, selection.scene).length < 2;

  const applyBtn = el("live-apply");
  let dirty = false;

  function markDirty() {
    if (!dirty) { dirty = true; applyBtn.classList.add("dirty"); }
  }

  function readPaneSelection(tag) {
    const pane = {};
    for (const key of PANE_DIMENSIONS) {
      pane[key] = el(liveDimensionId(key, tag)).value;
    }
    return pane;
  }

  applyBtn.addEventListener("click", () => {
    const next = { ...selection, a: readPaneSelection("a") };
    if (paneState.compare) { next.b = readPaneSelection("b"); }
    dirty = false;
    applyBtn.classList.remove("dirty");
    requestLiveSwitch(next, { message: "Applying selection…" });
  });

  el("compare").addEventListener("change", (e) => {
    if (e.target.checked) {
      requestLiveSwitch({ ...selection, b: { ...selection.a } },
        { message: "Preparing comparison…" });
    } else {
      requestLiveSwitch({ ...selection, b: null },
        { message: "Returning to single view…" });
    }
  });

  for (const tag of ["a", "b"]) {
    for (const key of PANE_DIMENSIONS) {
      el(liveDimensionId(key, tag)).addEventListener("change", () => {
        const pane = readPaneSelection(tag);
        fillLivePane(tag, pane);
        markDirty();
      });
    }
  }
}

function updateSummary() {
  const selectedSummary = (pane) => {
    const run = runs[pane.runIndex];
    return run.rm.reconstructions.find((rec) => rec.name === pane.recon)?.summary
      || run.rm.summary || "";
  };
  const line = (tag, pane) => `${tag} ${runs[pane.runIndex].rm.method}: ${
    selectedSummary(pane).replace(/\*\*/g, "")}`;
  el("summary").innerHTML = paneState.compare
    ? line("A", paneA) + "<br>" + line("B", paneB)
    : selectedSummary(paneA).replace(/\*\*/g, "");
  const paneLabel = (tag, pane, run) => live
    ? `${tag} · ${pane.difficulty} · ${pane.method} · ${pane.reconstruction || "replay"} · ${pane.seed}`
    : `${tag} · ${run.rm.method}`;
  el("label-a").textContent = paneLabel("A", live?.selection.a, runs[paneA.runIndex]);
  el("label-b").textContent = paneLabel(
    "B", live?.selection.b || live?.selection.a, runs[paneB.runIndex]);
}

function applyCompareVisibility() {
  const on = paneState.compare;
  el("paneB-row").hidden = !on;
  el("label-b").hidden = !on;
  el("divider").hidden = !on;
  el("tag-a").textContent = on ? "A" : "•";
  el("label-a").hidden = false;
}

// Sync the chrome whose state depends on the current runs (boot + switches).
function syncManifestUi() {
  const hasAnySplats = runs.some((r) => r.rm.reconstructions.length > 0);
  el("layer-splat").disabled = !hasAnySplats;
  state.layers.splat = hasAnySplats && state.layers.splat;
  el("layer-splat").checked = state.layers.splat;
  el("compare").checked = paneState.compare;
  applyCompareVisibility();
  updateSummary();
}

if (!navigationState && !live) state.layers.splat = runs.some((r) => r.rm.reconstructions.length > 0);
for (const [id, key] of [
  ["layer-distractors", "distractors"], ["layer-contamination", "contamination"],
  ["layer-history", "history"], ["layer-eval", "eval"], ["layer-shared", "shared"],
]) el(id).checked = Boolean(state.layers[key]);
el("loop").checked = state.loop;
el("speed").value = String(state.speed);
el("speed-val").textContent = state.speed.toFixed(1);
el("detail").value = String(state.detailScale);
el("detail-val").textContent = state.detailScale.toFixed(2);

if (live) {
  setupLiveSelection();
  setupLivePaneControls();
} else {
  setupStaticPaneControls();
}
syncManifestUi();

el("layer-splat").addEventListener("change", (e) => {
  state.layers.splat = e.target.checked;
  if (live) onSplatToggleLive(e.target.checked);
});
el("detail").addEventListener("input", (e) => {
  state.detailScale = parseFloat(e.target.value);
  el("detail-val").textContent = state.detailScale.toFixed(2);
});
for (const [id, key] of [
  ["layer-distractors", "distractors"], ["layer-contamination", "contamination"],
  ["layer-history", "history"], ["layer-eval", "eval"], ["layer-shared", "shared"],
]) el(id).addEventListener("change", (e) => (state.layers[key] = e.target.checked));

el("play").addEventListener("click", () => (state.playing = !state.playing));
el("loop").addEventListener("change", (e) => (state.loop = e.target.checked));
el("speed").addEventListener("input", (e) => {
  state.speed = parseFloat(e.target.value);
  el("speed-val").textContent = state.speed.toFixed(1);
});

const tMax = () => Math.max(...runs.map((r) => r.tEnd));
state.t = THREE.MathUtils.clamp(state.t, 0, tMax());
function seekCapture(delta) {
  const run = runs[paneA.runIndex];
  const k = THREE.MathUtils.clamp(run.captureIndexAt(state.t) + delta, 0, run.frames.length - 1);
  state.t = run.times[k];
  state.playing = false;
}
el("prev").addEventListener("click", () => seekCapture(-1));
el("next").addEventListener("click", () => seekCapture(1));

let sliderGuard = false;
el("progress").addEventListener("input", (e) => {
  if (sliderGuard) return;
  state.t = parseFloat(e.target.value) * tMax();
  state.playing = false;
});

el("fpv").addEventListener("click", enableRotateAroundMe);
el("capture-view").addEventListener("click", () => goToCapture());

// --- collapsible control panel ------------------------------------------------
const PANEL_COLLAPSED_KEY = "activebench-panel-collapsed";
function setPanelCollapsed(collapsed) {
  document.body.classList.toggle("panel-collapsed", collapsed);
  el("panel-fab").hidden = !collapsed;
  try { localStorage.setItem(PANEL_COLLAPSED_KEY, collapsed ? "1" : "0"); } catch (_e) {}
  resize(); // immediate; and once more when the width transition settles
}
el("panel-collapse").addEventListener("click", () => setPanelCollapsed(true));
el("panel-fab").addEventListener("click", () => setPanelCollapsed(false));
el("panel").addEventListener("transitionend", (e) => { if (e.propertyName === "width") resize(); });
try { if (localStorage.getItem(PANEL_COLLAPSED_KEY) === "1") setPanelCollapsed(true); } catch (_e) {}

// --- WASD/QE fly controls ---------------------------------------------------
const MOVE_KEYS = new Set(["w", "a", "s", "d", "q", "e", "shift"]);
const held = new Set();
const editable = (t) => t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA");
window.addEventListener("keydown", (e) => {
  if (editable(document.activeElement)) return;
  const k = e.key.toLowerCase();
  if (MOVE_KEYS.has(k)) { held.add(k); if (k !== "shift") e.preventDefault(); }
});
window.addEventListener("keyup", (e) => held.delete(e.key.toLowerCase()));
window.addEventListener("blur", () => held.clear());

const _fwd = new THREE.Vector3(), _right = new THREE.Vector3(), _move = new THREE.Vector3();
function flyStep(dt) {
  if (held.size === 0) return;
  _move.set(0, 0, 0);
  camera.getWorldDirection(_fwd);
  _right.crossVectors(_fwd, camera.up).normalize();
  if (held.has("w")) _move.add(_fwd);
  if (held.has("s")) _move.sub(_fwd);
  if (held.has("d")) _move.add(_right);
  if (held.has("a")) _move.sub(_right);
  if (held.has("e")) _move.y += 1;
  if (held.has("q")) _move.y -= 1;
  if (_move.lengthSq() === 0) return;
  _move.normalize().multiplyScalar(FLY_BASE * (held.has("shift") ? 3 : 1) * dt);
  camera.position.add(_move);
  controls.target.add(_move);
}

// --- main loop --------------------------------------------------------------

const thumbEl = el("thumb");
let panelK = -1, panelRun = -1;
function updatePanel() {
  const run = runs[paneA.runIndex];
  const k = run.captureIndexAt(Math.min(state.t, run.tEnd));
  if (k === panelK && paneA.runIndex === panelRun) return;
  panelK = k; panelRun = paneA.runIndex;
  thumbEl.src = run.frames[k].thumb;
  const distractorFraction = Number(run.frames[k].distractor_pixel_fraction || 0);
  el("frame-info").textContent =
    `frame ${k + 1}/${run.frames.length} · sim t ${run.times[k].toFixed(1)}s` +
    ` · distractor pixels ${(100 * distractorFraction).toFixed(1)}%`;
}

// Read-only introspection for automated browser tests (no behavior change).
window.__abDebug = {
  get runs() { return runs; },
  paneA, paneB, paneState, state, camera, controls, meshCache, paneSparks, renderer,
  get pendingBinaryRequests() { return binaryRequests.size; },
  // The stale-pager invariant: every paged mesh a live run holds must read
  // from the pager of the SparkRenderer that renders it (or not be bound yet).
  pagerMismatches() {
    const bad = [];
    runs.forEach((run, i) => {
      for (const [name, mesh] of run.splatMeshes) {
        const pager = mesh.paged?.pager;
        if (pager && run.spark.pager && pager !== run.spark.pager) {
          bad.push({ run: i, recon: name });
        }
      }
    });
    return bad;
  },
};

let last = performance.now();
renderer.setAnimationLoop(() => {
  const now = performance.now();
  const dt = (now - last) / 1000;
  last = now;
  flyStep(dt);
  if (state.playing) {
    state.t += dt * state.speed;
    if (state.t > tMax()) {
      state.t = state.loop ? 0 : tMax();
      state.playing = state.loop;
    }
  }
  for (const r of runs) r.spark.lodSplatScale = state.detailScale;
  touchPooledLodRecords(now);

  const W = viewport.clientWidth, H = viewport.clientHeight;
  if (paneState.compare) {
    const half = Math.floor(W / 2);
    renderPane(paneA, 0, half, H);
    renderPane(paneB, half, W - half, H);
  } else {
    renderPane(paneA, 0, W, H);
  }

  updatePanel();
  sliderGuard = true;
  el("progress").value = String(THREE.MathUtils.clamp(state.t / tMax(), 0, 1));
  sliderGuard = false;
  controls.update();
});
