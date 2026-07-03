"""
Persistent user settings for PortfolioOptimizer.
Reads and writes data/user_settings.json.
Settings survive page navigation, browser refresh, and session timeout.

Priority order (highest to lowest):
    1. st.session_state   — live UI values
    2. data/user_settings.json — last saved values
    3. DEFAULT_SETTINGS   — fallback if file missing or key absent
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from loguru import logger

SETTINGS_PATH = Path("data/user_settings.json")

DEFAULT_SETTINGS: dict[str, Any] = {
    "version": "1.0",
    "portfolio": {
        "total_capital": 100000,
        "start_date": "2020-01-01",
        "tickers": [],
        "holdings": {},
    },
    "optimization": {
        "method": "max_sharpe",
        "risk_free_rate": 0.05,
        "max_weight": 0.40,
        "min_weight": 0.02,
        "target_volatility": 0.15,
        "allow_fractional": False,
    },
    "constraints": {
        "turnover_enabled": False,
        "reduction_pct": 0.50,
        "increase_pct": 0.30,
        "allow_full_exit": True,
        "constraint_mode": "Both",
    },
}


def load_settings() -> dict[str, Any]:
    """
    Load settings from disk. Missing keys filled from DEFAULT_SETTINGS.

    Returns
    -------
    dict
        Merged settings: file values take priority, defaults fill gaps.
        Never raises — returns defaults on any read error.
    """
    if not SETTINGS_PATH.exists():
        logger.info(f"settings_manager: no file at {SETTINGS_PATH}, using defaults")
        return _deep_copy(DEFAULT_SETTINGS)

    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            saved = json.load(f)
        merged = _deep_merge(DEFAULT_SETTINGS, saved)
        logger.info(f"settings_manager: loaded from {SETTINGS_PATH}")
        return merged
    except Exception as e:
        logger.error(f"settings_manager: failed to load {SETTINGS_PATH}: {e}")
        return _deep_copy(DEFAULT_SETTINGS)


def save_settings(settings: dict[str, Any]) -> bool:
    """
    Write settings to disk.

    Parameters
    ----------
    settings : dict
        Must follow the same nested structure as DEFAULT_SETTINGS.
        Extra keys are preserved. Missing keys are not written.

    Returns
    -------
    bool
        True if saved successfully, False on error.
    """
    try:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        settings["version"] = DEFAULT_SETTINGS["version"]
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, default=str)
        logger.info(f"settings_manager: saved to {SETTINGS_PATH}")
        return True
    except Exception as e:
        logger.error(f"settings_manager: failed to save {SETTINGS_PATH}: {e}")
        return False


def get(key_path: str, fallback: Any = None) -> Any:
    """
    Get a single setting value by dot-path from the settings file.

    Parameters
    ----------
    key_path : str
        Dot-separated path e.g. "optimization.risk_free_rate"
    fallback : Any
        Returned if the key does not exist.

    Example
    -------
    get("portfolio.total_capital")   # returns 100000
    get("constraints.reduction_pct") # returns 0.50
    """
    settings = load_settings()
    keys = key_path.split(".")
    node = settings
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return fallback
        node = node[k]
    return node


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override values win."""
    result = _deep_copy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _deep_copy(d: dict) -> dict:
    import copy
    return copy.deepcopy(d)
