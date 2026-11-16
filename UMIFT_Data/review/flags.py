"""Keep/reject flags per episode, persisted as JSON (atomic replace on every change).

An entry is *stale* when the episode's current fingerprint differs from the one
stored with the flag (zarr rebuilt or trimmed in place), and an *orphan* when the
episode no longer exists. Export refuses stale selections.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

FLAGS = ("keep", "reject", "none")


class FlagStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.entries: Dict[str, dict] = {}
        self._lock = threading.Lock()

    def load(self) -> None:
        if self.path.exists():
            d = json.loads(self.path.read_text())
            self.entries = dict(d.get("episodes", {}))

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w") as f:
                json.dump({"version": 1, "episodes": self.entries}, f, indent=1, sort_keys=True)
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def set(self, ref, fp: dict, flag: Optional[str] = None, note: Optional[str] = None) -> dict:
        if flag is not None and flag not in FLAGS:
            raise ValueError(f"flag must be one of {FLAGS}, got {flag!r}")
        with self._lock:
            prev = self.entries.get(ref.key)
            e = dict(prev or {"flag": "none", "note": ""})
            e.update({"episode": ref.index, "source": ref.source,
                      "updated": datetime.now().isoformat(timespec="seconds")})
            # Only an explicit flag decision binds the entry to the current episode
            # content; a note-only edit must not turn a stale flag fresh.
            if flag is not None or prev is None:
                e["fingerprint"] = fp
                e["dataset_path"] = ref.zarr_path
            if flag is not None:
                e["flag"] = flag
            if note is not None:
                e["note"] = note
            self.entries[ref.key] = e
            self._save()
            return dict(e)

    def status(self, ref, fp: dict) -> dict:
        e = self.entries.get(ref.key)
        if e is None:
            return {"flag": "none", "note": "", "stale": False}
        out = dict(e)
        out["stale"] = (e.get("fingerprint") != fp or e.get("dataset_path") != ref.zarr_path)
        return out

    def orphans(self, index) -> List[str]:
        known = {e.key for e in index.episodes}
        return sorted(k for k in self.entries if k not in known)
