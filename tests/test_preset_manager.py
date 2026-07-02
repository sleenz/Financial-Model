"""
Smoke tests for src/utils/preset_manager.py.

Each test corresponds to one of the mandatory smoke tests for the portfolio
presets feature. All tests operate on a temporary directory so they never
touch the real data/portfolio_presets/.
"""
from __future__ import annotations

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from src.utils import preset_manager as pm


@pytest.fixture(autouse=True)
def _isolated_presets_dir(tmp_path, monkeypatch):
    presets_dir = tmp_path / "portfolio_presets"
    monkeypatch.setattr(pm, "PRESETS_DIR", presets_dir)
    monkeypatch.setattr(pm, "INDEX_PATH", presets_dir / "_index.json")
    yield presets_dir


TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
WEIGHTS = [0.2, 0.2, 0.2, 0.2, 0.2]


# ---------------------------------------------------------------------------
# 1. Save a new preset with 5 tickers
# ---------------------------------------------------------------------------

def test_save_creates_file_with_correct_schema(_isolated_presets_dir):
    preset_id = pm.save_preset("Portfolio 1", TICKERS, WEIGHTS, 100000.0)

    path = _isolated_presets_dir / f"{preset_id}.json"
    assert path.exists()

    with open(path) as f:
        data = json.load(f)

    assert data["schema_version"] == 1
    assert data["preset_id"] == preset_id
    assert data["name"] == "Portfolio 1"
    assert data["tickers"] == TICKERS
    assert data["weights"] == WEIGHTS
    assert data["portfolio_value"] == 100000.0
    assert data["currency"] == "USD"
    assert data["created_at"] == data["updated_at"]


# ---------------------------------------------------------------------------
# 2. Load that preset — values match exactly, no rounding drift
# ---------------------------------------------------------------------------

def test_load_round_trips_values_exactly():
    weights = [0.11111111, 0.22222222, 0.33333333, 0.15555555, 0.17777779]
    preset_id = pm.save_preset("Precise", TICKERS, weights, 123456.78)

    loaded = pm.load_preset(preset_id)

    assert loaded["tickers"] == TICKERS
    assert loaded["weights"] == weights
    assert loaded["portfolio_value"] == 123456.78
    assert len(loaded["tickers"]) == 5
    assert len(loaded["weights"]) == 5


# ---------------------------------------------------------------------------
# 3. Update Current overwrites the SAME file / preset_id, no duplicate created
# ---------------------------------------------------------------------------

def test_update_overwrites_same_file_no_duplicate(_isolated_presets_dir):
    preset_id = pm.save_preset("Portfolio 1", TICKERS, WEIGHTS, 100000.0)
    original = pm.load_preset(preset_id)

    new_weights = [0.4, 0.3, 0.1, 0.1, 0.1]
    ok = pm.update_preset(preset_id, TICKERS, new_weights, 105000.0)
    assert ok is True

    files = [p for p in _isolated_presets_dir.glob("*.json") if p.name != "_index.json"]
    assert len(files) == 1  # no new file created

    updated = pm.load_preset(preset_id)
    assert updated["preset_id"] == preset_id
    assert updated["weights"] == new_weights
    assert updated["portfolio_value"] == 105000.0
    assert updated["updated_at"] != original["updated_at"] or updated["updated_at"] >= original["created_at"]
    assert updated["created_at"] == original["created_at"]


# ---------------------------------------------------------------------------
# 4. Rename changes display name only — preset_id/filename untouched
# ---------------------------------------------------------------------------

def test_rename_changes_name_not_id_or_filename(_isolated_presets_dir):
    preset_id = pm.save_preset("Old Name", TICKERS, WEIGHTS, 100000.0)
    path_before = _isolated_presets_dir / f"{preset_id}.json"
    assert path_before.exists()

    ok = pm.rename_preset(preset_id, "New Name")
    assert ok is True

    # filename/preset_id unchanged
    assert path_before.exists()
    reloaded = pm.load_preset(preset_id)
    assert reloaded["preset_id"] == preset_id
    assert reloaded["name"] == "New Name"

    index = json.loads((_isolated_presets_dir / "_index.json").read_text())
    assert index[preset_id]["name"] == "New Name"


# ---------------------------------------------------------------------------
# 5. Duplicate name detection — no silent overwrite
# ---------------------------------------------------------------------------

def test_duplicate_name_is_detected_before_any_save():
    id1 = pm.save_preset("Portfolio 1", TICKERS, WEIGHTS, 100000.0)

    conflict = pm.preset_name_exists("Portfolio 1")
    assert conflict == id1

    # Case-insensitive match too
    assert pm.preset_name_exists("portfolio 1") == id1

    # A genuinely new name has no conflict
    assert pm.preset_name_exists("Portfolio 2") is None

    # Saving under the duplicate name (as the UI would only do after explicit
    # confirmation) creates an independent second file — save_preset itself
    # never silently overwrites.
    id2 = pm.save_preset("Portfolio 1", TICKERS, WEIGHTS, 50000.0)
    assert id2 != id1
    assert pm.load_preset(id1) is not None
    assert pm.load_preset(id2) is not None


# ---------------------------------------------------------------------------
# 6. Delete removes file, updates index, disappears from listing
# ---------------------------------------------------------------------------

def test_delete_removes_file_and_index_entry(_isolated_presets_dir):
    preset_id = pm.save_preset("To Delete", TICKERS, WEIGHTS, 100000.0)
    assert (_isolated_presets_dir / f"{preset_id}.json").exists()

    ok = pm.delete_preset(preset_id)
    assert ok is True

    assert not (_isolated_presets_dir / f"{preset_id}.json").exists()
    index = json.loads((_isolated_presets_dir / "_index.json").read_text())
    assert preset_id not in index
    assert preset_id not in [p["preset_id"] for p in pm.list_presets()]


# ---------------------------------------------------------------------------
# 7. Missing/corrupt _index.json self-heals instead of crashing or going empty
# ---------------------------------------------------------------------------

def test_missing_index_rebuilds_from_disk(_isolated_presets_dir):
    id1 = pm.save_preset("Portfolio 1", TICKERS, WEIGHTS, 100000.0)
    id2 = pm.save_preset("Portfolio 2", TICKERS, WEIGHTS, 50000.0)

    (_isolated_presets_dir / "_index.json").unlink()

    presets = pm.list_presets()
    ids = {p["preset_id"] for p in presets}
    assert ids == {id1, id2}
    assert (_isolated_presets_dir / "_index.json").exists()


def test_corrupt_index_rebuilds_from_disk(_isolated_presets_dir):
    id1 = pm.save_preset("Portfolio 1", TICKERS, WEIGHTS, 100000.0)

    (_isolated_presets_dir / "_index.json").write_text("{not valid json")

    presets = pm.list_presets()
    ids = {p["preset_id"] for p in presets}
    assert ids == {id1}


def test_stale_index_entry_for_deleted_file_self_heals(_isolated_presets_dir):
    id1 = pm.save_preset("Portfolio 1", TICKERS, WEIGHTS, 100000.0)
    (_isolated_presets_dir / f"{id1}.json").unlink()  # bypass delete_preset()

    presets = pm.list_presets()
    assert presets == []


def test_corrupt_single_preset_file_does_not_crash_listing(_isolated_presets_dir):
    id1 = pm.save_preset("Good Preset", TICKERS, WEIGHTS, 100000.0)
    bad_path = _isolated_presets_dir / "bad-preset-id.json"
    bad_path.write_text("{not valid json")

    presets = pm.list_presets()
    ids = {p["preset_id"] for p in presets}
    assert id1 in ids
    assert "bad-preset-id" not in ids
