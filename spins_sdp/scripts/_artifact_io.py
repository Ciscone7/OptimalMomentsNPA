from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Tuple

import numpy as np


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(config: Mapping[str, Any], *, n_chars: int = 16) -> str:
    payload = stable_json_dumps(config).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:n_chars]


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_bytes(path, stable_json_dumps(obj).encode("utf-8"))


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    """Atomically write an .npz (write temp then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.savez(tmp, **arrays)
    # np.savez appends .npz if missing
    tmp_npz = Path(str(tmp) + ".npz")
    os.replace(tmp_npz, path)


def load_records_npz(
    npz_path: Path,
    *,
    key: str,
    fields: Mapping[str, np.dtype],
) -> Dict[int, Dict[str, Any]]:
    """Load an NPZ table into a dict keyed by int(key).

    fields: mapping field_name -> dtype
    """
    if not npz_path.exists():
        return {}

    data = np.load(npz_path, allow_pickle=False)

    if key not in data:
        raise KeyError(f"Missing key array {key!r} in {npz_path}")

    keys = data[key].astype(int)
    records: Dict[int, Dict[str, Any]] = {}

    for i, k in enumerate(keys):
        rec: Dict[str, Any] = {}
        for field, dtype in fields.items():
            if field not in data:
                raise KeyError(f"Missing field array {field!r} in {npz_path}")
            value = data[field][i]
            # Convert numpy scalar -> python scalar
            if np.issubdtype(dtype, np.integer):
                rec[field] = int(value)
            elif np.issubdtype(dtype, np.floating):
                rec[field] = float(value)
            else:
                rec[field] = value.item() if hasattr(value, "item") else value
        records[int(k)] = rec

    return records


def save_records_npz(
    npz_path: Path,
    *,
    key: str,
    fields: Mapping[str, np.dtype],
    records: Mapping[int, Mapping[str, Any]],
) -> None:
    """Save a dict-of-records as column arrays into NPZ (sorted by key)."""
    keys_sorted = np.array(sorted(int(k) for k in records.keys()), dtype=int)

    arrays: Dict[str, np.ndarray] = {key: keys_sorted}

    for field, dtype in fields.items():
        arrays[field] = np.array([records[int(k)][field] for k in keys_sorted], dtype=dtype)

    atomic_save_npz(npz_path, **arrays)


def upsert_meta_json(meta_path: Path, meta: MutableMapping[str, Any]) -> None:
    """Write meta.json, preserving created_at if it exists."""
    if meta_path.exists():
        try:
            old = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(old, dict) and "created_at" in old:
                meta["created_at"] = old["created_at"]
        except Exception:
            pass

    atomic_write_json(meta_path, meta)
