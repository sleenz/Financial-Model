"""
Correlation network (MST) + semantic zoom for the Portfolio Builder.

Reuse (Phase 0 audit + Phase 1 answer #1): the ticker x ticker correlation
matrix is reconstructed from Phase 1's UniverseCache (correlation_row per
ticker, aligned to cache.get_correlation_index()) — plain-correlation
based, NOT DCC-GARCH (see fetch.py's module docstring for why: DCC-GARCH
is fit at sector count and has an unresolved convergence-misreport issue;
running it at ticker count would multiply that risk for a feature where a
wrong number is invisible in the UI).

Distance transform is the standard Mantegna (1999) MST metric,
d = sqrt(2*(1-rho)) — a fixed mathematical definition, not a tunable
business parameter, so it isn't a config field (same treatment as
Sharpe's formula being a fixed definition rather than a knob).

No 3D: this module only produces a networkx.Graph (2D-agnostic graph
structure with edge weights). Any (x, y) layout is a rendering concern
for Phase 5, not stored here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.portfolio_builder.cache import UniverseCache
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import networkx as nx
    _NETWORKX_AVAILABLE = True
except ImportError:
    nx = None
    _NETWORKX_AVAILABLE = False
    logger.warning(
        "networkx is not installed. MST network construction will be "
        "unavailable. Install with: pip install networkx>=3.0"
    )


def _require_networkx() -> None:
    if not _NETWORKX_AVAILABLE:
        raise ImportError(
            "networkx is required for this operation. Install with: pip install networkx>=3.0"
        )


@dataclass
class NetworkConfig:
    mst_algorithm: str = "kruskal"          # networkx minimum_spanning_tree algorithm: "kruskal" | "prim" | "boruvka"
    sector_aggregation: str = "average"     # "average" | "min" | "max" — cross-sector distance aggregation for the supernode graph
    require_full_correlation_row: bool = True  # exclude tickers whose cached correlation_row is empty/incomplete
                                                # (e.g. Phase 1's on-demand-fetch placeholder, deferred to next nightly refresh)
                                                # rather than silently treating missing entries as zero correlation


@dataclass
class TickerNetwork:
    """Full ticker-level MST — the "expand on zoom-in" detail view."""
    mst: object                # networkx.Graph — nodes=tickers, edge attr 'weight'=distance
    distance_matrix: pd.DataFrame
    excluded_tickers: list


@dataclass
class SectorNetwork:
    """Sector-supernode MST — the default-zoom overview."""
    mst: object                # networkx.Graph — nodes=sector names, edge attr 'weight'=aggregated distance
    sector_members: dict       # {sector: [tickers]}
    distance_matrix: pd.DataFrame  # full sector x sector aggregated distance — the MST only keeps
                                   # n-1 of these edges; a threshold-based edge filter (see
                                   # filter_edges_by_threshold) needs the full matrix to consider
                                   # re-adding a non-MST pair.


@dataclass
class SemanticZoomNetwork:
    """Two-level correlation network. Default zoom renders sector_network
    (one node per sector); zooming into a sector supernode swaps in the
    ticker-level subgraph for just that sector's members (see
    get_sector_subgraph()). No 3D — both levels are plain 2D-agnostic graphs."""
    sector_network: SectorNetwork
    ticker_network: TickerNetwork


def _row_is_complete(row: Optional[list], expected_length: int) -> bool:
    """A correlation_row only counts as usable if it's the expected length
    AND every entry is an actual number — not empty, not short, and not
    containing a None from an unresolvable pairwise correlation (e.g. two
    tickers with insufficient mutual return overlap; fetch.py stores those
    as None, not NaN, per its JSON-safety convention). A row with even one
    None can't feed a complete graph — every kept ticker needs a defined
    distance to every other kept ticker before an MST can be built — so
    "full" here means fully populated, not just correctly sized."""
    if not row or len(row) != expected_length:
        return False
    return all(v is not None for v in row)


def build_correlation_matrix(
    cache: UniverseCache,
    tickers: Optional[list] = None,
    config: NetworkConfig = NetworkConfig(),
):
    """
    Reconstruct the ticker x ticker correlation matrix from UniverseCache's
    cached correlation_row values, aligned via cache.get_correlation_index().

    Returns (correlation_df, excluded_tickers) — tickers with no cache
    entry, a stale entry, or (if config.require_full_correlation_row) a
    correlation_row that's empty, the wrong length, or contains a None
    for any pairwise correlation, are excluded and reported, not silently
    dropped from the network with no trace (and not left in to crash MST
    construction on a NaN edge weight later).
    """
    index = cache.get_correlation_index()
    if not index:
        raise ValueError(
            "build_correlation_matrix: cache has no correlation_index yet — "
            "run_nightly_refresh() must run at least once first"
        )

    universe = tickers if tickers is not None else index
    rows: dict = {}
    excluded: list = []
    for ticker in universe:
        entry = cache.get(ticker)
        if entry is None:
            excluded.append(ticker)
            continue
        if config.require_full_correlation_row and not _row_is_complete(
            entry.correlation_row, len(index)
        ):
            excluded.append(ticker)
            continue
        rows[ticker] = entry.correlation_row

    if excluded:
        logger.warning(
            f"build_correlation_matrix: excluded {excluded} (missing cache entry, "
            "or correlation_row empty/wrong-length/containing an unresolved "
            "pairwise correlation)"
        )
    if not rows:
        raise ValueError("build_correlation_matrix: no tickers had usable correlation data")

    kept = list(rows.keys())
    corr = pd.DataFrame.from_dict(rows, orient="index", columns=index)
    corr = corr.loc[kept, kept]
    for t in corr.index:
        corr.loc[t, t] = 1.0  # defensive — should already be 1.0 from source data
    return corr, excluded


def compute_distance_matrix(correlation: pd.DataFrame) -> pd.DataFrame:
    """Mantegna (1999) MST distance transform: d = sqrt(2*(1-rho)).
    rho=1 -> d=0 (identical); rho=-1 -> d=2 (maximally distant)."""
    clipped = correlation.clip(-1.0, 1.0)
    distance = np.sqrt(2.0 * (1.0 - clipped))
    values = distance.values.copy()  # np.fill_diagonal needs a writable array;
                                      # .values can return a read-only view here
    np.fill_diagonal(values, 0.0)
    return pd.DataFrame(values, index=distance.index, columns=distance.columns)


def _mst_from_distance(distance: pd.DataFrame, config: NetworkConfig):
    _require_networkx()
    graph = nx.Graph()
    nodes = list(distance.index)
    graph.add_nodes_from(nodes)
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            graph.add_edge(a, b, weight=float(distance.loc[a, b]))
    return nx.minimum_spanning_tree(graph, weight="weight", algorithm=config.mst_algorithm)


def build_ticker_mst(distance: pd.DataFrame, config: NetworkConfig = NetworkConfig()):
    """Full ticker-level MST from a precomputed distance matrix."""
    return _mst_from_distance(distance, config)


def build_sector_distance_matrix(
    distance: pd.DataFrame, sector_map: dict, config: NetworkConfig = NetworkConfig()
) -> pd.DataFrame:
    """Aggregate ticker-level distances into a sector x sector distance
    matrix, per config.sector_aggregation, for the default-zoom supernode graph."""
    agg_fns = {"average": np.mean, "min": np.min, "max": np.max}
    agg_fn = agg_fns.get(config.sector_aggregation)
    if agg_fn is None:
        raise ValueError(
            f"build_sector_distance_matrix: unknown sector_aggregation "
            f"'{config.sector_aggregation}', expected one of {list(agg_fns)}"
        )

    tickers_by_sector: dict = {}
    unmapped: list = []
    for t in distance.index:
        sector = sector_map.get(t)
        if sector is None:
            sector = "Unknown"
            unmapped.append(t)
        tickers_by_sector.setdefault(sector, []).append(t)

    if unmapped:
        logger.warning(
            f"build_sector_distance_matrix: {unmapped} missing from sector_map; "
            "bucketed under 'Unknown'"
        )

    sectors = sorted(tickers_by_sector.keys())
    sector_distance = pd.DataFrame(0.0, index=sectors, columns=sectors)
    for s1 in sectors:
        members1 = tickers_by_sector[s1]
        for s2 in sectors:
            if s1 == s2:
                continue
            members2 = tickers_by_sector[s2]
            pair_distances = [distance.loc[a, b] for a in members1 for b in members2]
            sector_distance.loc[s1, s2] = float(agg_fn(pair_distances))

    return sector_distance


def build_sector_mst(sector_distance: pd.DataFrame, config: NetworkConfig = NetworkConfig()):
    """Sector-supernode MST from a precomputed sector-level distance matrix."""
    return _mst_from_distance(sector_distance, config)


def get_sector_subgraph(ticker_network: TickerNetwork, sector_members: dict, sector: str):
    """Ticker-level MST edges restricted to one sector's members — the
    "expand on zoom-in" detail view for a single sector supernode."""
    members = sector_members.get(sector, [])
    return ticker_network.mst.subgraph(members).copy()


def correlation_from_distance(distance: float) -> float:
    """Exact inverse of the Mantegna transform above: d = sqrt(2*(1-rho))
    => rho = 1 - d^2/2. Recovers the underlying correlation for an MST edge
    from its stored distance weight, so callers that need the correlation
    (e.g. to color/shade an edge by correlation strength) don't have to
    carry a second parallel matrix through TickerNetwork/SectorNetwork.
    Same fixed-mathematical-definition treatment as compute_distance_matrix
    itself — not a tunable business parameter, so no config argument."""
    return 1.0 - (distance ** 2) / 2.0


@dataclass
class NetworkStyleConfig:
    """Visual-encoding constants for rendering the correlation network.
    Figure/layout construction itself is a rendering concern (stays in the
    page, per this module's docstring above) — this dataclass only owns
    the NUMBERS and color tokens that turn domain values (correlation,
    rank percentile) into visuals, so a fix for "hardcoded colors/opacity
    in the page" is an auditable config field, not a bare literal.

    Edge STRENGTH is encoded as a diverging COLOR GRADIENT, not opacity: a
    correlation of 0 renders edge_color_neutral, -1 renders edge_color_negative,
    +1 renders edge_color_positive, and anything in between is a linear RGB
    interpolation toward whichever end it's closer to (see edge_color_for_correlation).
    Opacity is a single fixed value (edge_opacity) applied to every edge —
    it no longer varies with correlation, so a strong hedge is exactly as
    visually prominent (in fully-saturated hedge color) as an equally strong
    positive correlation is in fully-saturated positive color, not a faded
    afterthought. Color tokens below are the app's existing palette
    (edge_color_positive/negative are hex equivalents of
    app/pages/2_Optimization.py's "coral"/"steelblue"; node tier colors
    reuse app/pages/4_Stress_Testing.py's low/warning/critical
    green/orange/red), not new colors invented for this feature.
    """
    edge_opacity: float = 1.0
    edge_color_positive: str = "#ff7f50"  # "coral", as a hex triplet so it can be gradient-interpolated
    edge_color_negative: str = "#4682b4"  # "steelblue", same reason
    edge_color_neutral: str = "#d3d3d3"   # "lightgray" — the gradient's zero-correlation midpoint
    node_color_top: str = "green"
    node_color_mid: str = "orange"
    node_color_bottom: str = "red"
    tier_top_percentile: float = 0.67    # matches the Ranked List's own 🟢 threshold
    tier_bottom_percentile: float = 0.33  # matches the Ranked List's own 🔴 threshold


def _hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb: tuple) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(max(0, min(255, round(c))) for c in rgb))


def _lerp_hex(color_a: str, color_b: str, t: float) -> str:
    """Linear interpolation between two '#rrggbb' colors at t in [0.0, 1.0]
    (t=0 -> color_a, t=1 -> color_b), channel by channel in RGB space —
    a plain, dependency-free gradient, not a perceptual color space."""
    t = max(0.0, min(1.0, t))
    a, b = _hex_to_rgb(color_a), _hex_to_rgb(color_b)
    return _rgb_to_hex(tuple(a[i] + (b[i] - a[i]) * t for i in range(3)))


def edge_color_for_correlation(
    correlation: float, config: NetworkStyleConfig = NetworkStyleConfig()
) -> str:
    """Diverging gradient color for one edge's correlation: edge_color_negative
    at rho=-1, edge_color_neutral at rho=0, edge_color_positive at rho=+1,
    linearly interpolated in between — so correlation STRENGTH reads as a
    color gradient rather than an opacity difference. rho is clamped to
    [-1, 1] defensively (it's always in that range by construction, but a
    gradient function shouldn't extrapolate past its own endpoints)."""
    rho = max(-1.0, min(1.0, correlation))
    if rho <= 0.0:
        return _lerp_hex(config.edge_color_negative, config.edge_color_neutral, rho + 1.0)
    return _lerp_hex(config.edge_color_neutral, config.edge_color_positive, rho)


def node_color_for_percentile(
    percentile: float, config: NetworkStyleConfig = NetworkStyleConfig()
) -> str:
    """Rank-tier node color, same tri-tier convention (top/middle/bottom
    third) as the Ranked List's heat emoji, so a ticker's node color and
    its ranked-list emoji always agree."""
    if percentile >= config.tier_top_percentile:
        return config.node_color_top
    if percentile >= config.tier_bottom_percentile:
        return config.node_color_mid
    return config.node_color_bottom


@dataclass
class CorrelationNetworkConfig:
    """Thresholds for which NON-MST edges get drawn alongside the always-
    present MST — a pure rendering filter over an already-computed distance
    matrix, not a parameter of the correlation/MST computation itself (see
    filter_edges_by_threshold). Split into two independent, same-signed
    thresholds rather than one |correlation| cutoff, because "strongly
    correlated" and "strongly anti-correlated" (hedge-like) are different
    things a user looks for on this chart, not two ends of the same slider:
    positive_threshold surfaces the former, hedge_threshold the latter, and
    a pair can only qualify via one or the other (never both, since a
    single correlation can't be both >= a positive number and <= a negative
    one). Both are deliberately named for a UI widget's initial value, not a
    fixed business constant: the page reads these defaults once to seed the
    widgets, then values come from each widget's own state on every rerun.
    """
    always_include_mst: bool = True   # MST edges are always drawn regardless of threshold — this is
                                       # what guarantees every node stays connected/visible even at
                                       # the strictest threshold setting.
    positive_threshold: float = 0.30  # in [0.0, 1.0] — an additional edge is drawn if
                                       # correlation >= this ("strongly correlated" edges).
    hedge_threshold: float = -0.30    # in [-1.0, 0.0] — an additional edge is drawn if
                                       # correlation <= this ("hedge-like", anti-correlated edges).


def filter_edges_by_threshold(
    distance: pd.DataFrame,
    mst,
    config: CorrelationNetworkConfig = CorrelationNetworkConfig(),
):
    """Build the graph to RENDER from an ALREADY-COMPUTED distance matrix and
    MST — every MST edge (if config.always_include_mst) plus any additional
    non-MST pair whose correlation clears EITHER threshold: >= positive_threshold
    (strongly correlated) or <= hedge_threshold (strongly anti-correlated /
    hedge-like). Pure filtering over data the caller already has: no
    correlation recomputation, no MST rebuild. This is what lets a UI
    threshold slider be a cheap client-side re-filter on every rerun rather
    than a re-fetch/re-fit.

    Returns a networkx.Graph with the same nodes as `distance`'s index and
    'weight'=distance on every edge (same attribute MST edges already use),
    so downstream rendering code (edge color via correlation_from_distance)
    doesn't need to know which edges came from the MST vs. a threshold.
    """
    _require_networkx()
    graph = nx.Graph()
    graph.add_nodes_from(distance.index)

    if config.always_include_mst:
        for u, v, data in mst.edges(data=True):
            graph.add_edge(u, v, weight=float(data["weight"]))

    nodes = list(distance.index)
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            d = float(distance.loc[a, b])
            corr = correlation_from_distance(d)
            if corr >= config.positive_threshold or corr <= config.hedge_threshold:
                graph.add_edge(a, b, weight=d)

    return graph


def build_semantic_zoom_network(
    cache: UniverseCache,
    sector_map: dict,
    tickers: Optional[list] = None,
    config: NetworkConfig = NetworkConfig(),
    correlation: Optional[pd.DataFrame] = None,
) -> SemanticZoomNetwork:
    """Full pipeline: correlation matrix -> distance transform -> ticker-level
    MST (detail) + sector-level MST (default-zoom overview).

    correlation: if given, used directly instead of reading UniverseCache's
    cached correlation_row values via build_correlation_matrix(). This
    matters because OnDemandFetcher.get_or_fetch() always leaves a fresh
    ticker's correlation_row empty (deferred to the next
    run_nightly_refresh() — see fetch.py) — with no nightly job actually
    scheduled in a given deployment, every first-time ticker would
    otherwise be excluded here, and a cold cache means build_correlation_matrix
    raises "no tickers had usable correlation data" for every new user.
    Passing an already-computed correlation matrix (e.g. from price data a
    caller already fetched for another purpose) sidesteps that entirely.
    Must be a ticker x ticker DataFrame; tickers requested but absent from
    its index/columns are reported via TickerNetwork.excluded_tickers,
    same as build_correlation_matrix's own exclusion contract.
    """
    if correlation is not None:
        universe = tickers if tickers is not None else list(correlation.index)
        kept = [t for t in universe if t in correlation.index and t in correlation.columns]
        excluded = [t for t in universe if t not in kept]
        if excluded:
            logger.warning(
                f"build_semantic_zoom_network: excluded {excluded} (not present "
                "in the supplied correlation matrix)"
            )
        correlation = correlation.loc[kept, kept]
    else:
        correlation, excluded = build_correlation_matrix(cache, tickers, config)

    distance = compute_distance_matrix(correlation)
    ticker_mst = build_ticker_mst(distance, config)

    sector_members: dict = {}
    unmapped: list = []
    for t in distance.index:
        sector = sector_map.get(t)
        if sector is None:
            sector = "Unknown"
            unmapped.append(t)
        sector_members.setdefault(sector, []).append(t)
    if unmapped:
        logger.warning(
            f"build_semantic_zoom_network: {unmapped} missing from sector_map; "
            "bucketed under 'Unknown'"
        )

    sector_distance = build_sector_distance_matrix(distance, sector_map, config)
    sector_mst = build_sector_mst(sector_distance, config)

    return SemanticZoomNetwork(
        sector_network=SectorNetwork(
            mst=sector_mst, sector_members=sector_members, distance_matrix=sector_distance,
        ),
        ticker_network=TickerNetwork(mst=ticker_mst, distance_matrix=distance, excluded_tickers=excluded),
    )


if __name__ == "__main__":
    def _smoke_test():
        from datetime import datetime, timezone

        from src.portfolio_builder.cache import CacheConfig, RankedUniverseEntry, UniverseCache
        from src.portfolio_builder.network import (
            NetworkConfig,
            SemanticZoomNetwork,
            TickerNetwork,
            build_correlation_matrix,
            build_semantic_zoom_network,
            build_sector_distance_matrix,
            build_sector_mst,
            build_ticker_mst,
            compute_distance_matrix,
            get_sector_subgraph,
        )

        # ── compute_distance_matrix: hand-computable values ──────────────
        # rho=1 -> d=0; rho=0 -> d=sqrt(2); rho=-1 -> d=2
        corr = pd.DataFrame(
            {"A": [1.0, 0.0, -1.0], "B": [0.0, 1.0, 0.5], "C": [-1.0, 0.5, 1.0]},
            index=["A", "B", "C"],
        )
        dist = compute_distance_matrix(corr)
        assert dist.loc["A", "A"] == 0.0
        assert abs(dist.loc["A", "B"] - np.sqrt(2.0)) < 1e-9
        assert abs(dist.loc["A", "C"] - 2.0) < 1e-9
        print("✓ compute_distance_matrix: Mantegna transform matches hand calc")

        # ── build_ticker_mst: spanning tree properties ───────────────────
        mst = build_ticker_mst(dist)
        assert set(mst.nodes()) == {"A", "B", "C"}
        assert mst.number_of_edges() == 2, "a 3-node MST must have exactly n-1=2 edges"
        assert nx.is_connected(mst)
        # A-C is the most distant pair (d=2.0); the MST must not include it
        # when a cheaper path exists (A-B=sqrt(2)=1.414, B-C=1.0, both < 2.0)
        assert not mst.has_edge("A", "C"), "MST should skip the most distant edge"
        print("✓ build_ticker_mst: correct edge count, connected, skips the most distant edge")

        # ── build_sector_distance_matrix + build_sector_mst ──────────────
        sector_map = {"A": "Tech", "B": "Tech", "C": "Energy"}
        sector_dist = build_sector_distance_matrix(dist, sector_map)
        # Tech-Energy = average(d(A,C), d(B,C)) = average(2.0, 1.0) = 1.5
        # (d(B,C) = sqrt(2*(1-0.5)) = sqrt(1.0) = 1.0)
        expected = (2.0 + 1.0) / 2.0
        assert abs(sector_dist.loc["Tech", "Energy"] - expected) < 1e-9
        sector_mst = build_sector_mst(sector_dist)
        assert set(sector_mst.nodes()) == {"Tech", "Energy"}
        assert sector_mst.number_of_edges() == 1
        print("✓ build_sector_distance_matrix / build_sector_mst: aggregation and MST correct")

        # ── get_sector_subgraph: "expand on zoom-in" ─────────────────────
        sector_members = {"Tech": ["A", "B"], "Energy": ["C"]}
        ticker_net = TickerNetwork(mst=mst, distance_matrix=dist, excluded_tickers=[])
        subgraph = get_sector_subgraph(ticker_net, sector_members, "Tech")
        assert set(subgraph.nodes()) == {"A", "B"}
        print("✓ get_sector_subgraph: zoom-in subgraph restricted to sector members")

        # ── build_correlation_matrix: reconstruct from a real UniverseCache ──
        cache = UniverseCache(CacheConfig(cache_path=":memory:"))
        now = datetime.now(timezone.utc).isoformat()
        index = ["A", "B", "C", "D"]
        cache.set_correlation_index(index)
        # A, B, C have full 4-length correlation_row; D has an empty row
        # (simulating Phase 1's on-demand-fetch placeholder, not yet refreshed)
        cache.upsert(RankedUniverseEntry("A", "Tech", "US", 10.0, {}, [1.0, 0.0, -1.0, 0.2], now))
        cache.upsert(RankedUniverseEntry("B", "Tech", "US", 20.0, {}, [0.0, 1.0, 0.5, 0.1], now))
        cache.upsert(RankedUniverseEntry("C", "Energy", "US", 30.0, {}, [-1.0, 0.5, 1.0, 0.3], now))
        cache.upsert(RankedUniverseEntry("D", "Energy", "US", 40.0, {}, [], now))

        rebuilt_corr, excluded = build_correlation_matrix(cache)
        assert excluded == ["D"], excluded
        assert set(rebuilt_corr.index) == {"A", "B", "C"}
        assert abs(rebuilt_corr.loc["A", "B"] - 0.0) < 1e-9
        print("✓ build_correlation_matrix: reconstructs from cache, excludes incomplete rows with a trace")

        # Regression: a None inside an otherwise full-length correlation_row
        # (independent review caught this — this is exactly what fetch.py
        # produces for an unresolvable pairwise correlation, e.g. two tickers
        # with insufficient mutual return overlap; the original check only
        # looked at row length, so a full-length row containing a None still
        # got treated as "complete" and later crashed nx.minimum_spanning_tree
        # with a NaN edge weight instead of being excluded up front).
        cache_none = UniverseCache(CacheConfig(cache_path=":memory:"))
        cache_none.set_correlation_index(index)
        cache_none.upsert(RankedUniverseEntry("A", "Tech", "US", 10.0, {}, [1.0, None, 0.3, 0.1], now))
        cache_none.upsert(RankedUniverseEntry("B", "Tech", "US", 20.0, {}, [None, 1.0, 0.2, 0.15], now))
        cache_none.upsert(RankedUniverseEntry("C", "Energy", "US", 30.0, {}, [0.3, 0.2, 1.0, 0.25], now))
        cache_none.upsert(RankedUniverseEntry("D", "Energy", "US", 40.0, {}, [0.1, 0.15, 0.25, 1.0], now))

        corr_none, excluded_none = build_correlation_matrix(cache_none)
        assert set(excluded_none) == {"A", "B"}, excluded_none
        assert set(corr_none.index) == {"C", "D"}
        # Must not crash — a None-containing row must never reach MST construction
        zoom_none = build_semantic_zoom_network(cache_none, sector_map)
        assert set(zoom_none.ticker_network.mst.nodes()) == {"C", "D"}
        print("✓ build_correlation_matrix: None inside a full-length row is excluded, not a crash")

        # ── build_semantic_zoom_network: end-to-end, two-level structure ──
        zoom = build_semantic_zoom_network(cache, sector_map)
        assert isinstance(zoom, SemanticZoomNetwork)
        assert set(zoom.sector_network.mst.nodes()) == {"Tech", "Energy"}
        assert set(zoom.ticker_network.mst.nodes()) == {"A", "B", "C"}
        assert zoom.ticker_network.excluded_tickers == ["D"]
        print("✓ build_semantic_zoom_network: end-to-end sector + ticker level MSTs built")

        # Regression: a COLD cache (every ticker just on-demand-fetched, so
        # every correlation_row is empty per fetch.py's OnDemandFetcher
        # contract) must not make the network entirely unusable. This is
        # exactly what happened in production: with no nightly refresh job
        # actually scheduled, build_correlation_matrix excluded every
        # first-time ticker and raised "no tickers had usable correlation
        # data" for every new user. Passing a live-computed correlation
        # matrix (e.g. from price data the caller already fetched for
        # another purpose) must produce a working network instead.
        cold_cache = UniverseCache(CacheConfig(cache_path=":memory:"))
        cold_cache.set_correlation_index(["A", "B", "C"])
        cold_cache.upsert(RankedUniverseEntry("A", "Tech", "US", 10.0, {}, [], now))
        cold_cache.upsert(RankedUniverseEntry("B", "Tech", "US", 20.0, {}, [], now))
        cold_cache.upsert(RankedUniverseEntry("C", "Energy", "US", 30.0, {}, [], now))

        # Cache-only path fails entirely on a cold cache:
        try:
            build_correlation_matrix(cold_cache)
            raise AssertionError("expected ValueError: cold cache has no usable correlation data")
        except ValueError:
            pass

        # Live-computed correlation bypasses the cache dependency entirely:
        live_corr = pd.DataFrame(
            [[1.0, 0.0, -1.0], [0.0, 1.0, 0.5], [-1.0, 0.5, 1.0]],
            index=["A", "B", "C"], columns=["A", "B", "C"],
        )
        cold_zoom = build_semantic_zoom_network(
            cold_cache, sector_map, tickers=["A", "B", "C"], correlation=live_corr
        )
        assert set(cold_zoom.ticker_network.mst.nodes()) == {"A", "B", "C"}
        assert cold_zoom.ticker_network.excluded_tickers == []
        print("✓ build_semantic_zoom_network: live-computed correlation works on a cold cache")

        # A ticker missing from the supplied live correlation matrix is
        # excluded and reported, not silently dropped or crashed on.
        partial_zoom = build_semantic_zoom_network(
            cold_cache, sector_map, tickers=["A", "B", "C", "E"], correlation=live_corr
        )
        assert set(partial_zoom.ticker_network.mst.nodes()) == {"A", "B", "C"}
        assert partial_zoom.ticker_network.excluded_tickers == ["E"]
        print("✓ build_semantic_zoom_network: ticker missing from supplied correlation is excluded, not dropped silently")

        # ── correlation_from_distance: exact inverse of the Mantegna transform ──
        from src.portfolio_builder.network import (
            NetworkStyleConfig,
            edge_color_for_correlation,
            node_color_for_percentile,
        )

        for rho in [1.0, 0.5, 0.0, -0.5, -1.0]:
            d = float(np.sqrt(2.0 * (1.0 - rho)))
            assert abs(correlation_from_distance(d) - rho) < 1e-9, (rho, d)
        print("✓ correlation_from_distance: exact inverse of compute_distance_matrix's transform")

        # ── edge_color_for_correlation: diverging gradient, not opacity ──
        style = NetworkStyleConfig()
        assert edge_color_for_correlation(1.0, style) == style.edge_color_positive
        assert edge_color_for_correlation(-1.0, style) == style.edge_color_negative
        assert edge_color_for_correlation(0.0, style) == style.edge_color_neutral

        # A halfway-positive correlation is exactly halfway (in RGB space)
        # between neutral and the positive token -- and independently
        # recomputed here, not just re-deriving the function's own formula.
        def _independent_lerp(a_hex, b_hex, t):
            a = tuple(int(a_hex.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
            b = tuple(int(b_hex.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
            return "#{:02x}{:02x}{:02x}".format(*(round(a[i] + (b[i] - a[i]) * t) for i in range(3)))

        expected_half_positive = _independent_lerp(style.edge_color_neutral, style.edge_color_positive, 0.5)
        assert edge_color_for_correlation(0.5, style) == expected_half_positive
        expected_half_negative = _independent_lerp(style.edge_color_negative, style.edge_color_neutral, 0.5)
        assert edge_color_for_correlation(-0.5, style) == expected_half_negative

        # Opacity is now a single fixed constant, not derived from correlation
        # at all -- a strong hedge and a strong positive correlation get the
        # exact same (default: fully opaque) edge_opacity, so strength is
        # legible purely from color, never from how faded a line looks.
        assert style.edge_opacity == 1.0
        print("✓ edge_color_for_correlation: diverging gradient (negative→neutral→positive), opacity fixed not correlation-derived")

        # ── node_color_for_percentile: matches Ranked List's tri-tier thresholds ──
        assert node_color_for_percentile(0.9, style) == style.node_color_top
        assert node_color_for_percentile(0.5, style) == style.node_color_mid
        assert node_color_for_percentile(0.1, style) == style.node_color_bottom
        # boundary values are inclusive on the upper tier, per >= comparisons above
        assert node_color_for_percentile(0.67, style) == style.node_color_top
        assert node_color_for_percentile(0.33, style) == style.node_color_mid
        print("✓ node_color_for_percentile: tri-tier thresholds match Ranked List's own 🟢🟡🔴 cutoffs")

        # ── filter_edges_by_threshold: pure re-filter over an already-built
        # distance matrix + MST, no recomputation, dual positive/hedge
        # thresholds instead of one |correlation| cutoff ───────────────────
        from src.portfolio_builder.network import (
            CorrelationNetworkConfig,
            filter_edges_by_threshold,
        )

        # Reuse the hand-computable A/B/C distance matrix + MST from above:
        # correlations A-B=0.0, A-C=-1.0 (hedge-like), B-C=0.5 (correlated)
        # MST (from build_ticker_mst above) = {A-B, B-C}, skips A-C (most distant)

        # Default (positive_threshold=0.30, hedge_threshold=-0.30),
        # always_include_mst=True: MST edges {A-B, B-C} plus every pair
        # clearing either threshold -> A-C (-1.0 <= -0.30) qualifies via the
        # hedge side, B-C (0.5 >= 0.30, already in the MST) qualifies via the
        # positive side; A-B (0.0) doesn't qualify on its own but is kept via
        # the MST -> complete triangle, 3 edges.
        default_cfg = CorrelationNetworkConfig()
        assert abs(default_cfg.positive_threshold - 0.30) < 1e-9
        assert abs(default_cfg.hedge_threshold - (-0.30)) < 1e-9
        g_default = filter_edges_by_threshold(dist, mst, default_cfg)
        assert set(g_default.nodes()) == {"A", "B", "C"}
        assert g_default.number_of_edges() == 3
        assert g_default.has_edge("A", "B") and g_default.has_edge("B", "C") and g_default.has_edge("A", "C")
        print("✓ filter_edges_by_threshold: default thresholds pull in both a positive (B-C) and a hedge (A-C) non-MST edge")

        # Isolate the HEDGE threshold: always_include_mst=False, positive
        # side disabled (2.0 is unreachable), hedge_threshold=-0.9 -> only
        # A-C (-1.0 <= -0.9) qualifies; B ends up with no edge at all.
        g_hedge_only = filter_edges_by_threshold(dist, mst, CorrelationNetworkConfig(
            always_include_mst=False, positive_threshold=2.0, hedge_threshold=-0.9,
        ))
        assert set(g_hedge_only.nodes()) == {"A", "B", "C"}  # nodes always present
        assert g_hedge_only.number_of_edges() == 1
        assert g_hedge_only.has_edge("A", "C")
        assert g_hedge_only.degree("B") == 0
        print("✓ filter_edges_by_threshold: hedge_threshold in isolation surfaces only the anti-correlated pair (A-C)")

        # Isolate the POSITIVE threshold: always_include_mst=False, hedge
        # side disabled (-2.0 is unreachable), positive_threshold=0.4 -> only
        # B-C (0.5 >= 0.4) qualifies; A ends up with no edge at all.
        g_positive_only = filter_edges_by_threshold(dist, mst, CorrelationNetworkConfig(
            always_include_mst=False, positive_threshold=0.4, hedge_threshold=-2.0,
        ))
        assert set(g_positive_only.nodes()) == {"A", "B", "C"}
        assert g_positive_only.number_of_edges() == 1
        assert g_positive_only.has_edge("B", "C")
        assert g_positive_only.degree("A") == 0
        print("✓ filter_edges_by_threshold: positive_threshold in isolation surfaces only the correlated pair (B-C)")

        # CHECK 1 (Rule 1 spec): both thresholds at their strictest (beyond
        # the max possible |correlation|=1.0) with always_include_mst=True ->
        # every node still has >= 1 edge (the MST guarantee), no extra edges.
        g_strict = filter_edges_by_threshold(dist, mst, CorrelationNetworkConfig(
            positive_threshold=1.5, hedge_threshold=-1.5,
        ))
        assert g_strict.number_of_edges() == mst.number_of_edges() == 2
        assert nx.is_connected(g_strict)
        assert all(deg >= 1 for _, deg in g_strict.degree())
        print("✓ filter_edges_by_threshold: max-strictness on both thresholds still satisfies the MST connectivity guarantee")

        # CHECK 2 (Rule 1 spec): both thresholds at their most permissive
        # (positive_threshold=0.0, hedge_threshold=0.0 -- every correlation is
        # either >= 0 or <= 0) -> every pair qualifies -> complete graph.
        # Reporting the actual edge count, not asserting it's bounded -- for
        # n=3 that's n*(n-1)/2 = 3.
        g_loose = filter_edges_by_threshold(dist, mst, CorrelationNetworkConfig(
            positive_threshold=0.0, hedge_threshold=0.0,
        ))
        assert g_loose.number_of_edges() == 3 == (3 * 2) // 2
        print("✓ filter_edges_by_threshold: both thresholds at 0.0 -> complete graph (3 edges for 3 nodes), reported not hidden")

        print("✓ network.py smoke test passed")

    _smoke_test()
