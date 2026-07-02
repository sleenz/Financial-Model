"""Portfolio Presets Page - Save, load, rename, update, and delete named portfolio snapshots."""

import streamlit as st
import pandas as pd
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.preset_manager import (
    list_presets,
    preset_name_exists,
    load_preset,
    save_preset,
    update_preset,
    rename_preset,
    delete_preset,
    apply_preset_to_state,
)

st.set_page_config(page_title="Portfolio Presets", page_icon=None, layout="wide")

st.title("Portfolio Presets")
st.markdown(
    "Save named snapshots of your portfolio (tickers, weights, value) so you can "
    "switch between them without re-entering data. This is separate from the "
    "automatic last-session save — presets are saved and loaded explicitly."
)

# --- Session state -----------------------------------------------------
if 'loaded_preset_id' not in st.session_state:
    st.session_state.loaded_preset_id = None
if 'preset_pending_confirm' not in st.session_state:
    st.session_state.preset_pending_confirm = None


def _fmt_ts(iso_str: str) -> str:
    if not iso_str:
        return "unknown"
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return iso_str


def _current_portfolio_state():
    """Read the live portfolio input (tickers, weights, value) from the same
    session_state keys the Portfolio Input / Optimization widgets already use."""
    weights_obj = st.session_state.get('weights')
    if weights_obj is not None:
        tickers = [str(t) for t in weights_obj.index]
        weights = [float(w) for w in weights_obj.values]
    else:
        tickers = [str(t) for t in st.session_state.get('tickers', [])]
        n = len(tickers)
        weights = [1.0 / n] * n if n else []
    portfolio_value = float(st.session_state.get('portfolio_value', 0.0) or 0.0)
    return tickers, weights, portfolio_value


def _apply_preset_to_session(preset: dict) -> None:
    """Populate session_state / widgets with a loaded preset's saved data."""
    apply_preset_to_state(preset, st.session_state)


st.markdown("---")

# --- Current portfolio summary + Save As New ---------------------------
st.subheader("Current Portfolio")

_cur_tickers, _cur_weights, _cur_value = _current_portfolio_state()

if _cur_tickers:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tickers", len(_cur_tickers))
    with col2:
        st.metric("Portfolio Value", f"${_cur_value:,.2f}")
    with col3:
        _loaded = st.session_state.loaded_preset_id
        _loaded_name = None
        if _loaded:
            for _p in list_presets():
                if _p["preset_id"] == _loaded:
                    _loaded_name = _p["name"]
                    break
        st.metric("Loaded Preset", _loaded_name or "None")

    with st.expander("View current tickers / weights", expanded=False):
        st.dataframe(
            pd.DataFrame({"Ticker": _cur_tickers, "Weight": [f"{w*100:.2f}%" for w in _cur_weights]}),
            width="stretch",
        )
else:
    st.info("No portfolio loaded yet. Enter tickers on the **Portfolio Input** page first.")

save_new_name = st.text_input("Preset name", key="preset_save_new_name", placeholder="e.g. Portfolio 1")

if st.button("Save As New Preset", type="primary", disabled=not _cur_tickers):
    name = save_new_name.strip()
    if not name:
        st.error("Please enter a name for the preset.")
    else:
        conflict_id = preset_name_exists(name)
        if conflict_id:
            st.session_state.preset_pending_confirm = {
                "action": "save_new_overwrite",
                "conflict_id": conflict_id,
                "name": name,
            }
        else:
            new_id = save_preset(name, _cur_tickers, _cur_weights, _cur_value)
            st.session_state.loaded_preset_id = new_id
            st.success(f"Saved new preset '{name}'.")
        st.rerun()

st.markdown("---")

# --- Saved presets list / actions ---------------------------------------
st.subheader("Saved Presets")

presets = list_presets()

if not presets:
    st.info("No presets saved yet. Use **Save As New Preset** above to create one.")
else:
    listing_df = pd.DataFrame(
        [{"Name": p["name"], "Last Updated": _fmt_ts(p["updated_at"])} for p in presets]
    )
    st.dataframe(listing_df, width="stretch", hide_index=True)

    options = [p["preset_id"] for p in presets]
    labels = {p["preset_id"]: f"{p['name']}  —  {_fmt_ts(p['updated_at'])}" for p in presets}

    selected_id = st.selectbox(
        "Select a preset",
        options,
        format_func=lambda pid: labels.get(pid, pid),
        key="preset_selected_id",
    )

    col_load, col_update, col_rename, col_delete = st.columns(4)

    with col_load:
        if st.button("Load", width="stretch"):
            preset = load_preset(selected_id)
            if preset is None:
                st.error("Failed to load this preset — the file may be corrupt. Check logs for details.")
            else:
                _apply_preset_to_session(preset)
                st.success(f"Loaded preset '{preset['name']}'.")
                st.rerun()

    with col_update:
        _can_update = st.session_state.loaded_preset_id is not None
        if st.button("Update Current", width="stretch", disabled=not _can_update):
            target_id = st.session_state.loaded_preset_id
            tickers, weights, value = _current_portfolio_state()
            if update_preset(target_id, tickers, weights, value):
                st.success("Updated the loaded preset in place.")
            else:
                st.error("Failed to update — the preset file may have been deleted. Check logs.")
            st.rerun()
        if not _can_update:
            st.caption("Load a preset first to enable Update.")

    with col_rename:
        if st.button("Rename", width="stretch"):
            st.session_state.preset_pending_confirm = {
                "action": "rename_form",
                "target_id": selected_id,
            }
            st.rerun()

    with col_delete:
        if st.button("Delete", width="stretch"):
            st.session_state.preset_pending_confirm = {
                "action": "delete_confirm",
                "target_id": selected_id,
            }
            st.rerun()

# --- Pending confirmation / follow-up UI ---------------------------------
pending = st.session_state.preset_pending_confirm

if pending is not None:
    st.markdown("---")

    if pending["action"] == "save_new_overwrite":
        conflict = load_preset(pending["conflict_id"])
        conflict_name = conflict["name"] if conflict else pending["name"]
        st.warning(
            f"A preset named **'{pending['name']}'** already exists. "
            "Saving with this name will overwrite that preset's data "
            "(its underlying file stays the same, only its contents change)."
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button(f"Overwrite '{conflict_name}'", type="primary"):
                tickers, weights, value = _current_portfolio_state()
                update_preset(pending["conflict_id"], tickers, weights, value)
                st.session_state.loaded_preset_id = pending["conflict_id"]
                st.session_state.preset_pending_confirm = None
                st.success(f"Overwrote preset '{conflict_name}'.")
                st.rerun()
        with c2:
            if st.button("Cancel"):
                st.session_state.preset_pending_confirm = None
                st.rerun()

    elif pending["action"] == "rename_form":
        target = load_preset(pending["target_id"])
        if target is None:
            st.error("This preset no longer exists.")
            st.session_state.preset_pending_confirm = None
        else:
            st.markdown(f"**Rename '{target['name']}'**")
            new_name = st.text_input("New name", value=target["name"], key="preset_rename_input")
            if st.button("Confirm Rename", type="primary"):
                stripped = new_name.strip()
                if not stripped:
                    st.error("Please enter a non-empty name.")
                elif stripped == target["name"]:
                    st.session_state.preset_pending_confirm = None
                    st.rerun()
                else:
                    conflict_id = preset_name_exists(stripped, exclude_id=pending["target_id"])
                    if conflict_id:
                        st.session_state.preset_pending_confirm = {
                            "action": "rename_overwrite_confirm",
                            "target_id": pending["target_id"],
                            "new_name": stripped,
                            "conflict_id": conflict_id,
                        }
                    else:
                        rename_preset(pending["target_id"], stripped)
                        st.session_state.preset_pending_confirm = None
                        st.success(f"Renamed to '{stripped}'.")
                    st.rerun()
            if st.button("Cancel", key="cancel_rename"):
                st.session_state.preset_pending_confirm = None
                st.rerun()

    elif pending["action"] == "rename_overwrite_confirm":
        conflict = load_preset(pending["conflict_id"])
        conflict_name = conflict["name"] if conflict else pending["new_name"]
        st.warning(
            f"Another preset is already named **'{pending['new_name']}'**. "
            "Two presets with the same display name would be ambiguous in the "
            "dropdown. Rename anyway?"
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Rename Anyway", type="primary"):
                rename_preset(pending["target_id"], pending["new_name"])
                st.session_state.preset_pending_confirm = None
                st.success(f"Renamed to '{pending['new_name']}'.")
                st.rerun()
        with c2:
            if st.button("Cancel", key="cancel_rename_overwrite"):
                st.session_state.preset_pending_confirm = None
                st.rerun()

    elif pending["action"] == "delete_confirm":
        target = load_preset(pending["target_id"])
        target_name = target["name"] if target else pending["target_id"]
        st.warning(f"Delete preset **'{target_name}'**? This cannot be undone.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Confirm Delete", type="primary"):
                delete_preset(pending["target_id"])
                if st.session_state.loaded_preset_id == pending["target_id"]:
                    st.session_state.loaded_preset_id = None
                st.session_state.preset_pending_confirm = None
                st.success(f"Deleted preset '{target_name}'.")
                st.rerun()
        with c2:
            if st.button("Cancel", key="cancel_delete"):
                st.session_state.preset_pending_confirm = None
                st.rerun()
