from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd


def anonymous_key(*parts: object, length: int = 16) -> str:
    payload = "|".join("" if pd.isna(p) else str(p) for p in parts)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()[:length]


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def relative_to_data(path: Path, data_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(data_root.resolve()))
    except ValueError:
        return str(path.resolve())


def absolute_from_data(value: object, data_root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else data_root / path
