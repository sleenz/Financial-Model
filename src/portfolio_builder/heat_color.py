"""
Heat color for the Portfolio Builder's ranked list.

Replaces RankedStock.rank_tier's undefined "high" | "mid" | "low" bucket
scheme (see ranking.py — that field had no cutoff logic ever implemented
anywhere in this codebase) with a continuous RdYlGn diverging colormap over
a composite_score percentile band. A vetted, perceptually-calibrated named
colormap is used deliberately instead of hand-rolled red/yellow/green RGB
interpolation, which muddies badly at the transition points; the red-
yellow-green color SCHEME itself is an accepted, deliberate choice — this
module only fixes how positions within that scheme are computed.

DEVIATION FROM SPEC: the given pseudocode calls
`matplotlib.cm.get_cmap(colormap)`. That function was removed in the
matplotlib version this project pins (>=3.7.0; confirmed removed by 3.11.0,
installed here) — calling it raises AttributeError. The current,
non-deprecated equivalent is `matplotlib.colormaps[name]` (subscript
access), used below. Same "matplotlib does the interpolation, we don't"
guarantee either way — only the accessor API differs.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib
import matplotlib.colors as mcolors


@dataclass
class HeatColorConfig:
    colormap: str = "RdYlGn"
    # Vetted matplotlib diverging colormap — NOT hand-rolled RGB
    # interpolation. A linear blend between red/yellow/green muddies badly
    # at the transition points; the named colormap is calibrated to avoid
    # that. This is the actual fix — the color scheme itself (red-yellow-
    # green) is an accepted, deliberate choice, not what's being corrected.
    low_percentile: float = 0.05
    high_percentile: float = 0.95
    # Bounds computed from the FULL cached universe (nightly batch job
    # output), not the current user selection. Otherwise a mediocre stock
    # reads as "good" purely because everything else in a small selection
    # happens to be worse — a relative signal masquerading as absolute.


def composite_score_to_color(
    score: float, low_bound: float, high_bound: float, colormap: str = "RdYlGn"
) -> str:
    """
    Clip score to [low_bound, high_bound], normalize to [0, 1], map through
    matplotlib's named colormap. Do not hand-roll the interpolation.

    low_bound/high_bound are absolute composite_score values (e.g. the
    5th/95th percentile of the FULL cached universe — see HeatColorConfig),
    not [0, 1] fractions themselves; this function does the clip-and-
    normalize step so callers just pass the two raw score bounds.

    Raises ValueError if high_bound <= low_bound — a degenerate or inverted
    bound range can't be normalized into a meaningful [0, 1] position, and
    silently returning some arbitrary color would hide that upstream bug.
    """
    if high_bound <= low_bound:
        raise ValueError(
            f"composite_score_to_color: high_bound ({high_bound}) must be > "
            f"low_bound ({low_bound})"
        )
    clipped = max(low_bound, min(high_bound, score))
    normalized = (clipped - low_bound) / (high_bound - low_bound)
    cmap = matplotlib.colormaps[colormap]
    return mcolors.to_hex(cmap(normalized))


if __name__ == "__main__":
    def _smoke_test():
        from src.portfolio_builder.heat_color import (
            HeatColorConfig,
            composite_score_to_color,
        )

        # ── HeatColorConfig defaults ──────────────────────────────────────
        config = HeatColorConfig()
        assert config.colormap == "RdYlGn"
        assert abs(config.low_percentile - 0.05) < 1e-9
        assert abs(config.high_percentile - 0.95) < 1e-9
        print("✓ HeatColorConfig: defaults match spec (RdYlGn, 5th/95th percentile)")

        # ── composite_score_to_color: hand-computable at the exact bounds
        # and midpoint, cross-checked against matplotlib's OWN RdYlGn output
        # at those normalized positions — not a re-derivation of this
        # module's own formula, an independent call to the same colormap. ──
        import matplotlib
        import matplotlib.colors as mcolors

        cmap = matplotlib.colormaps["RdYlGn"]
        expected_low = mcolors.to_hex(cmap(0.0))
        expected_mid = mcolors.to_hex(cmap(0.5))
        expected_high = mcolors.to_hex(cmap(1.0))

        low_bound, high_bound = 10.0, 90.0
        mid_score = (low_bound + high_bound) / 2.0  # 50.0 -> normalized 0.5

        assert composite_score_to_color(low_bound, low_bound, high_bound) == expected_low
        assert composite_score_to_color(mid_score, low_bound, high_bound) == expected_mid
        assert composite_score_to_color(high_bound, low_bound, high_bound) == expected_high
        print(
            f"✓ composite_score_to_color: matches matplotlib's own RdYlGn output "
            f"at 0.0/0.5/1.0 ({expected_low}/{expected_mid}/{expected_high})"
        )

        # Clipping: a score below low_bound or above high_bound must clamp,
        # not extrapolate past the colormap's own endpoints.
        assert composite_score_to_color(-1000.0, low_bound, high_bound) == expected_low
        assert composite_score_to_color(1000.0, low_bound, high_bound) == expected_high
        print("✓ composite_score_to_color: out-of-range scores clip to the colormap's endpoints, don't extrapolate")

        # A percentile-derived bound pair, exactly as HeatColorConfig intends
        # (5th/95th percentile of a FULL universe of composite_scores) —
        # hand-computable with a simple synthetic universe.
        import pandas as pd

        universe_scores = pd.Series(range(0, 101))  # 0..100, so 5th pct=5.0, 95th pct=95.0
        low_p = float(universe_scores.quantile(config.low_percentile))
        high_p = float(universe_scores.quantile(config.high_percentile))
        assert abs(low_p - 5.0) < 1e-9 and abs(high_p - 95.0) < 1e-9, (low_p, high_p)
        # A mediocre score (50, dead center of the FULL universe) must map to
        # the neutral midpoint color regardless of what a SMALLER selection
        # containing it happens to look like — this is the whole point of
        # computing bounds from the full universe, not the current selection.
        assert composite_score_to_color(50.0, low_p, high_p) == expected_mid
        print("✓ composite_score_to_color: percentile bounds from a full-universe series behave as documented")

        # Degenerate bound range raises rather than returning a meaningless color.
        try:
            composite_score_to_color(5.0, 10.0, 10.0)
            raise AssertionError("expected ValueError for high_bound == low_bound")
        except ValueError:
            pass
        try:
            composite_score_to_color(5.0, 10.0, 5.0)
            raise AssertionError("expected ValueError for high_bound < low_bound")
        except ValueError:
            pass
        print("✓ composite_score_to_color: degenerate/inverted bound range raises ValueError")

        print("✓ heat_color.py smoke test passed")

    _smoke_test()
