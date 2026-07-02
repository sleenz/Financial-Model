"""
Named portfolio preset manager for PortfolioOptimizer.

Independent of data/user_settings.json (the single-portfolio last-session
autosave, handled by settings_manager.py) — that file and its behavior
are untouched by this module. Presets are an additive, separate feature:
named, renameable snapshots of a portfolio (tickers, weights, value) that
the user can save, list, load, update in place, rename, and delete.

Storage layout
--------------
data/portfolio_presets/<preset_id>.json   one file per preset
data/portfolio_presets/_index.json        preset_id -> {name, updated_at}

The filename is always the preset's UUID, never derived from its display
name, so renaming a preset never touches the filename and there is no
filename collision/sanitization concern. The index exists purely as a
fast-listing cache; it is rebuilt from the directory contents whenever it
is missing, corrupt, or out of sync with what's on disk, so index
corruption can never hide an existing preset.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .logger import get_logger

logger = get_logger(__name__)

PRESETS_DIR = Path("data/portfolio_presets")
INDEX_PATH = PRESETS_DIR / "_index.json"

SCHEMA_VERSION = 1


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _preset_path(preset_id: str) -> Path:
    return PRESETS_DIR / f"{preset_id}.json"


def _read_json(path: Path) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"preset_manager: failed to parse {path}: {e}")
        return None


def _write_json(path: Path, data: Any) -> None:
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def _rebuild_index() -> dict[str, dict[str, str]]:
    """Scan the presets directory and regenerate _index.json from the files found."""
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    index: dict[str, dict[str, str]] = {}
    for path in sorted(PRESETS_DIR.glob("*.json")):
        if path.name == "_index.json":
            continue
        data = _read_json(path)
        if data is None:
            continue  # corrupt preset file — already logged, skip it
        preset_id = data.get("preset_id", path.stem)
        index[preset_id] = {
            "name": data.get("name", "Untitled"),
            "updated_at": data.get("updated_at", _now_iso()),
        }
    _save_index(index)
    logger.info(f"preset_manager: rebuilt index with {len(index)} preset(s)")
    return index


def _save_index(index: dict[str, dict[str, str]]) -> None:
    _write_json(INDEX_PATH, index)


def _load_index() -> dict[str, dict[str, str]]:
    """Load the index, self-healing (rebuild from disk) on any corruption or drift."""
    if not INDEX_PATH.exists():
        return _rebuild_index()

    index = _read_json(INDEX_PATH)
    if not isinstance(index, dict):
        logger.warning("preset_manager: _index.json is corrupt or unreadable, rebuilding from disk")
        return _rebuild_index()

    disk_ids = {p.stem for p in PRESETS_DIR.glob("*.json") if p.name != "_index.json"}
    index_ids = set(index.keys())
    if disk_ids != index_ids:
        for pid in disk_ids - index_ids:
            logger.warning(
                f"preset_manager: preset {pid} exists on disk but is missing from "
                "the index (index drift) — rebuilding"
            )
        return _rebuild_index()

    return index


def list_presets() -> list[dict[str, str]]:
    """
    List all saved presets, sorted by updated_at descending.

    Returns
    -------
    list[dict]
        Each item: {"preset_id": str, "name": str, "updated_at": str}
    """
    index = _load_index()
    items = [
        {"preset_id": pid, "name": meta.get("name", "Untitled"), "updated_at": meta.get("updated_at", "")}
        for pid, meta in index.items()
    ]
    items.sort(key=lambda x: x["updated_at"], reverse=True)
    return items


def preset_name_exists(name: str, exclude_id: Optional[str] = None) -> Optional[str]:
    """Return the preset_id of an existing preset with this display name (case-insensitive), or None."""
    target = name.strip().lower()
    for item in list_presets():
        if item["preset_id"] == exclude_id:
            continue
        if item["name"].strip().lower() == target:
            return item["preset_id"]
    return None


def load_preset(preset_id: str) -> Optional[dict[str, Any]]:
    """Load a single preset's full data by id. Returns None (and logs ERROR) if missing/corrupt."""
    path = _preset_path(preset_id)
    if not path.exists():
        logger.error(f"preset_manager: load failed, no file for preset_id={preset_id}")
        return None
    return _read_json(path)


def save_preset(
    name: str,
    tickers: list[str],
    weights: list[float],
    portfolio_value: float,
    currency: str = "USD",
) -> str:
    """Create a brand-new preset with a freshly generated UUID. Returns the new preset_id."""
    preset_id = str(uuid.uuid4())
    now = _now_iso()
    data = {
        "schema_version": SCHEMA_VERSION,
        "preset_id": preset_id,
        "name": name,
        "created_at": now,
        "updated_at": now,
        "tickers": list(tickers),
        "weights": [float(w) for w in weights],
        "portfolio_value": float(portfolio_value),
        "currency": currency,
    }
    # Load the index before writing the new file — the file doesn't exist yet,
    # so this can never observe (and spuriously "self-heal") drift.
    index = _load_index()
    _write_json(_preset_path(preset_id), data)

    index[preset_id] = {"name": name, "updated_at": now}
    _save_index(index)

    logger.info(f"preset_manager: saved preset_id={preset_id} name='{name}' at {now}")
    return preset_id


def update_preset(
    preset_id: str,
    tickers: list[str],
    weights: list[float],
    portfolio_value: float,
    currency: str = "USD",
) -> bool:
    """Overwrite an existing preset's portfolio data in place (same file, same preset_id)."""
    existing = load_preset(preset_id)
    if existing is None:
        logger.error(f"preset_manager: update failed, preset_id={preset_id} not found")
        return False

    index = _load_index()

    now = _now_iso()
    existing["schema_version"] = SCHEMA_VERSION
    existing["tickers"] = list(tickers)
    existing["weights"] = [float(w) for w in weights]
    existing["portfolio_value"] = float(portfolio_value)
    existing["currency"] = currency
    existing["updated_at"] = now
    _write_json(_preset_path(preset_id), existing)

    index.setdefault(preset_id, {})["name"] = existing.get("name", "Untitled")
    index[preset_id]["updated_at"] = now
    _save_index(index)

    logger.info(f"preset_manager: updated preset_id={preset_id} name='{existing.get('name')}' at {now}")
    return True


def rename_preset(preset_id: str, new_name: str) -> bool:
    """Change a preset's display name only. The filename and preset_id never change."""
    existing = load_preset(preset_id)
    if existing is None:
        logger.error(f"preset_manager: rename failed, preset_id={preset_id} not found")
        return False

    index = _load_index()

    now = _now_iso()
    existing["name"] = new_name
    existing["updated_at"] = now
    _write_json(_preset_path(preset_id), existing)

    index.setdefault(preset_id, {})["name"] = new_name
    index[preset_id]["updated_at"] = now
    _save_index(index)

    logger.info(f"preset_manager: renamed preset_id={preset_id} to name='{new_name}' at {now}")
    return True


def apply_preset_to_state(preset: dict[str, Any], state: Any) -> None:
    """
    Populate a Streamlit-like session_state object with a preset's saved
    tickers/weights/value and track it as the currently loaded preset, so
    that any page offering a "load this preset" action (the dedicated
    Presets page, or a quick-load shortcut elsewhere) behaves identically.
    """
    tickers = list(preset.get("tickers", []))
    weights = list(preset.get("weights", []))
    value = float(preset.get("portfolio_value", 0.0))

    state["tickers"] = tickers
    state["weights"] = pd.Series(weights, index=tickers, dtype=float)
    state["current_portfolio_weights"] = state["weights"]
    state["portfolio_value"] = value

    if not state.get("settings"):
        state["settings"] = {}
    state["settings"]["total_capital"] = value

    state["loaded_preset_id"] = preset.get("preset_id")


def delete_preset(preset_id: str) -> bool:
    """Delete a preset's file and remove it from the index."""
    existing = load_preset(preset_id)
    name = existing.get("name") if existing else None
    index = _load_index()

    path = _preset_path(preset_id)
    try:
        if path.exists():
            path.unlink()
    except Exception as e:
        logger.error(f"preset_manager: delete failed for preset_id={preset_id}: {e}")
        return False

    if preset_id in index:
        del index[preset_id]
        _save_index(index)

    logger.info(f"preset_manager: deleted preset_id={preset_id} name='{name}' at {_now_iso()}")
    return True
