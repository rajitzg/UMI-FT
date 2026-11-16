#!/usr/bin/env python3
"""Episode review server: browse, scrub, flag and export UMIFT episodes.

  conda activate pyrite
  source /home/boeing/umi_ft_ws/worktrees/activate.sh UMI-FT-standard_bc
  python UMIFT_Data/review/review_server.py --ckpt <ckpt>   # then open http://127.0.0.1:8770

Source zarrs are opened read-only. Writes only the flags file, the cache dir and
export outputs. See UMIFT_Data/review/README.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import re
import sys
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional, Sequence

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import numpy as np  # noqa: E402

from review import video  # noqa: E402
from review.episodes import DatasetIndex, EpisodeRef, append_arrays, build_payload, encode_payload  # noqa: E402
from review.flags import FlagStore  # noqa: E402

DATA_ROOT = "/mnt/data/umidata/ft/datasets"
DEFAULT_BASE = [f"{DATA_ROOT}/stringer_v3.5_base/acp_replay_buffer.zarr"]
DEFAULT_DAGGER = [f"{DATA_ROOT}/stringer_v3.5_dagger{s}/acp_replay_buffer.zarr" for s in ("", "_2", "_3")]
STATIC = _HERE / "static"
LRU_SIZE = 6
_LAST_STATE = None  # the most recent ReviewState (tests reach it through this)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", nargs="*", default=DEFAULT_BASE, help="base (original) zarrs")
    p.add_argument("--dagger", nargs="*", default=DEFAULT_DAGGER, help="DAgger zarrs")
    p.add_argument("--ckpt", default=None, help="checkpoint: enables history window + never-sampled band")
    p.add_argument("--rollout-slowdown", type=float, default=2.0,
                   help="runner slow_down_factor for the rollout history view")
    p.add_argument("--session", default="stringer_v3.5")
    p.add_argument("--flags", default=None, help=f"default {DATA_ROOT}/review_flags/<session>.json")
    p.add_argument("--mold", default=None,
                   help="fixb_field.npz stringer mesh for the 3D overlay (fixture frame); "
                        "default: the stringer_v3.5_fixb field if present; 'none' disables")
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--cache-dir", default=str(Path.home() / ".cache" / "umift_review"))
    a = p.parse_args(argv)
    if a.mold is None:
        from review.mold import DEFAULT_FIELD
        a.mold = DEFAULT_FIELD if Path(DEFAULT_FIELD).exists() else "none"
    if a.flags is None:
        a.flags = f"{DATA_ROOT}/review_flags/{a.session}.json"
    return a


class ReviewState:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.index = DatasetIndex(args.base, args.dagger)
        if not self.index.episodes:
            raise SystemExit("no episodes found")
        self.flags = FlagStore(Path(args.flags))
        self.flags.load()
        self.cache_dir = Path(args.cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.r_ref = self.index.reference_rotation()
        self.spec = None
        if args.ckpt:
            from review.policy_window import load_window_spec
            self.spec = load_window_spec(args.ckpt, self.cache_dir)
        self._lru: "OrderedDict[str, bytes]" = OrderedDict()
        self._lru_lock = threading.Lock()
        self._build_locks: dict = {}
        self._pool = ThreadPoolExecutor(max_workers=2)
        self.export_lock = threading.Lock()
        self._summaries = {e.key: self.index.summary(e) for e in self.index.episodes}
        self.mold_bytes = None
        if args.mold and args.mold != "none":
            from review.mold import load_mold
            v, f = load_mold(args.mold, self.cache_dir)
            self.mold_bytes = encode_payload(
                {"arrays": [{"name": "vertices", "length": int(v.size)}, {"name": "faces", "length": int(f.size)}],
                 "n_faces": int(len(f)), "source": str(args.mold)},
                [v.ravel(), f.ravel().astype(np.float32)])  # indices < 2**24, exact in float32

    def cache_key(self, ref: EpisodeRef) -> str:
        fp = json.dumps(self._summaries[ref.key]["fingerprint"], sort_keys=True)
        return hashlib.sha1(f"{ref.zarr_path}|{ref.index}|{fp}".encode()).hexdigest()[:20]

    def episodes_json(self) -> dict:
        eps = []
        for ref in self.index.episodes:
            s = dict(self._summaries[ref.key])
            st = self.flags.status(ref, s["fingerprint"])
            s.update(flag=st["flag"], note=st.get("note", ""), stale=st["stale"])
            eps.append(s)
        return {"episodes": eps, "orphans": self.flags.orphans(self.index),
                "has_ckpt": self.spec is not None, "ckpt": self.args.ckpt,
                "rollout_slowdown": self.args.rollout_slowdown,
                "window": self.spec.header() if self.spec else None,
                "flags_file": str(self.flags.path), "has_mold": self.mold_bytes is not None,
                "mold": self.args.mold if self.mold_bytes is not None else None}

    def payload(self, ref: EpisodeRef) -> bytes:
        with self._lru_lock:
            if ref.key in self._lru:
                self._lru.move_to_end(ref.key)
                return self._lru[ref.key]
            lock = self._build_locks.setdefault(ref.key, threading.Lock())
        with lock:
            with self._lru_lock:
                if ref.key in self._lru:
                    return self._lru[ref.key]
            g = self.index.group(ref)
            extra = {"key": ref.key, "source": ref.source, "video_fps": video.VIDEO_FPS,
                     "sprite": video.sprite_geometry(g["rgb_0"].shape[0])}
            header, arrays = build_payload(g, extra=extra, root_attrs=self.index.root_attrs(ref), r_ref=self.r_ref)
            if self.spec is not None:
                from review.policy_window import episode_window_arrays
                win, info = episode_window_arrays(g, ref.index, self.spec)
                append_arrays(header, arrays, win)
                header.update(info)
                header["window"] = self.spec.header()
                header["rollout_slowdown"] = self.args.rollout_slowdown
            buf = encode_payload(header, arrays)
            with self._lru_lock:
                self._lru[ref.key] = buf
                while len(self._lru) > LRU_SIZE:
                    self._lru.popitem(last=False)
            return buf

    def video(self, ref: EpisodeRef) -> Path:
        return video.episode_video(self.index.group(ref), self.cache_dir, self.cache_key(ref))

    def thumb(self, ref: EpisodeRef) -> Path:
        return video.episode_thumb(self.index.group(ref), self.cache_dir, self.cache_key(ref))

    def sprite(self, ref: EpisodeRef) -> Path:
        return video.episode_sprite(self.index.group(ref), self.cache_dir, self.cache_key(ref))[0]

    def prefetch_neighbours(self, ref: EpisodeRef) -> None:
        eps = self.index.episodes
        i = eps.index(ref)
        for j in (i + 1, i - 1):
            if 0 <= j < len(eps):
                n = eps[j]
                self._pool.submit(self._warm, n)

    def _warm(self, ref: EpisodeRef) -> None:
        try:
            self.video(ref)
            self.payload(ref)
            if self.spec is not None:
                self.sprite(ref)
        except Exception as e:  # prefetch is best effort
            print(f"[review] prefetch {ref.key} failed: {e}")


_EP_RE = re.compile(r"^/api/episode/([^/]+)/(\d+)/(data|video\.mp4|thumb\.jpg|sprite\.jpg)$")
_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)$")


def make_handler(state: ReviewState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str, extra: Optional[dict] = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, code: int, obj):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _file(self, path: Path, ctype: str, ranged: bool = False):
            data = path.read_bytes()
            rng = self.headers.get("Range") if ranged else None
            m = _RANGE_RE.match(rng.strip()) if rng else None
            if m and (m.group(1) or m.group(2)):
                size = len(data)
                a, b = m.group(1), m.group(2)
                if a == "":
                    start, end = max(0, size - int(b)), size - 1
                else:
                    start, end = int(a), (int(b) if b else size - 1)
                end = min(end, size - 1)
                if start > end:
                    return self._send(416, b"", ctype, {"Content-Range": f"bytes */{size}"})
                return self._send(206, data[start:end + 1], ctype,
                                  {"Content-Range": f"bytes {start}-{end}/{size}", "Accept-Ranges": "bytes"})
            self._send(200, data, ctype, {"Accept-Ranges": "bytes"} if ranged else None)

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            try:
                if path == "/api/episodes":
                    return self._json(200, state.episodes_json())
                if path == "/api/mold":
                    if state.mold_bytes is None:
                        return self._json(404, {"error": "no mold mesh (run with --mold)"})
                    return self._send(200, state.mold_bytes, "application/octet-stream")
                m = _EP_RE.match(path)
                if m:
                    try:
                        ref = state.index.get(f"{m.group(1)}/{int(m.group(2))}")
                    except KeyError:
                        return self._json(404, {"error": "unknown episode"})
                    kind = m.group(3)
                    if kind == "data":
                        body = state.payload(ref)
                        state.prefetch_neighbours(ref)
                        return self._send(200, body, "application/octet-stream")
                    if kind == "video.mp4":
                        return self._file(state.video(ref), "video/mp4", ranged=True)
                    if kind == "thumb.jpg":
                        return self._file(state.thumb(ref), "image/jpeg")
                    return self._file(state.sprite(ref), "image/jpeg")
                rel = "index.html" if path in ("/", "/index.html") else path.removeprefix("/static/")
                f = (STATIC / rel).resolve()
                if path != "/" and not path.startswith("/static/") and path != "/index.html":
                    return self._json(404, {"error": "not found"})
                if STATIC.resolve() not in f.parents or not f.is_file():
                    return self._json(404, {"error": "not found"})
                ctype = "text/javascript" if f.suffix == ".js" else (mimetypes.guess_type(f.name)[0] or "application/octet-stream")
                return self._file(f, ctype)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                print(f"[review] GET {path} failed: {e!r}")
                try:
                    self._json(500, {"error": str(e)})
                except Exception:
                    pass

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            # Only same-origin JSON: blocks no-preflight cross-site form/text posts.
            if (self.headers.get("Content-Type") or "").split(";")[0].strip() != "application/json":
                return self._json(415, {"error": "Content-Type must be application/json"})
            origin = self.headers.get("Origin")
            port = self.server.server_address[1]
            if origin and origin not in (f"http://127.0.0.1:{port}", f"http://localhost:{port}"):
                return self._json(403, {"error": f"foreign origin {origin}"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                if path == "/api/flag":
                    try:
                        ref = state.index.get(body["key"])
                    except KeyError:
                        return self._json(404, {"error": "unknown episode"})
                    fp = state.index.summary(ref)["fingerprint"]
                    state.flags.set(ref, fp, flag=body.get("flag"), note=body.get("note"))
                    return self._json(200, state.flags.status(ref, fp))
                if path == "/api/export":
                    from review.review_export import export_selection
                    if not state.export_lock.acquire(blocking=False):
                        return self._json(409, {"error": "an export is already running"})
                    try:
                        s = export_selection(state.index, state.flags, Path(body["output"]).expanduser(),
                                         include=body.get("include", "keep"),
                                         overwrite=bool(body.get("overwrite")),
                                         dry_run=bool(body.get("dry_run")),
                                         command=["review_server /api/export", json.dumps(body)])
                    finally:
                        state.export_lock.release()
                    return self._json(200, s)
                return self._json(404, {"error": "not found"})
            except (ValueError, KeyError, FileExistsError, OSError) as e:
                return self._json(400, {"error": str(e)})
            except Exception as e:
                print(f"[review] POST {path} failed: {e!r}")
                return self._json(500, {"error": str(e)})

    return Handler


def make_server(args: argparse.Namespace) -> ThreadingHTTPServer:
    global _LAST_STATE
    state = ReviewState(args)
    _LAST_STATE = state
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(state))
    srv.daemon_threads = True
    srv.state = state
    return srv


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    srv = make_server(args)
    st = srv.state
    print(f"[review] {len(st.index.episodes)} episodes; flags -> {st.flags.path}; "
          f"ckpt -> {args.ckpt or 'none (no history overlays)'}")
    print(f"[review] open http://127.0.0.1:{srv.server_address[1]}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
