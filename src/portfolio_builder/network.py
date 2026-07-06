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


@dataclass
class SemanticZoomNetwork:
    """Two-level correlation network. Default zoom renders sector_network
    (one node per sector); zooming into a sector supernode swaps in the
    ticker-level subgraph for just that sector's members (see
    get_sector_subgraph()). No 3D — both levels are plain 2D-agnostic graphs."""
    sector_network: SectorNetwork
    ticker_network: TickerNetwork


def build_correlation_matrix(
    cache: UniverseCache,
    tickers: Optional[list] = None,
    config: NetworkConfig = NetworkConfig(),
):
    """
    Reconstruct the ticker x ticker correlation matrix from UniverseCache's
    cached correlation_row values, aligned via cache.get_correlation_index().

    Returns (correlation_df, excluded_tickers) — tickers with no cache
    entry, a stale entry, or (if config.require_full_correlation_row) an
    empty/incomplete correlation_row are excluded and reported, not
    silently dropped from the network with no trace.
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
        if config.require_full_correlation_row and (
            not entry.correlation_row or len(entry.correlation_row) != len(index)
        ):
            excluded.append(ticker)
            continue
        rows[ticker] = entry.correlation_row

    if excluded:
        logger.warning(
            f"build_correlation_matrix: excluded {excluded} (missing cache entry or "
            "incomplete/empty correlation_row)"
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
    for t in distance.index:
        sector = sector_map.get(t, "Unknown")
        tickers_by_sector.setdefault(sector, []).append(t)

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


def build_semantic_zoom_network(
    cache: UniverseCache,
    sector_map: dict,
    tickers: Optional[list] = None,
    config: NetworkConfig = NetworkConfig(),
) -> SemanticZoomNetwork:
    """Full pipeline: cached correlation_row values -> distance transform ->
    ticker-level MST (detail) + sector-level MST (default-zoom overview)."""
    correlation, excluded = build_correlation_matrix(cache, tickers, config)
    distance = compute_distance_matrix(correlation)
    ticker_mst = build_ticker_mst(distance, config)

    sector_members: dict = {}
    for t in distance.index:
        sector_members.setdefault(sector_map.get(t, "Unknown"), []).append(t)

    sector_distance = build_sector_distance_matrix(distance, sector_map, config)
    sector_mst = build_sector_mst(sector_distance, config)

    return SemanticZoomNetwork(
        sector_network=SectorNetwork(mst=sector_mst, sector_members=sector_members),
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

        # ── build_semantic_zoom_network: end-to-end, two-level structure ──
        zoom = build_semantic_zoom_network(cache, sector_map)
        assert isinstance(zoom, SemanticZoomNetwork)
        assert set(zoom.sector_network.mst.nodes()) == {"Tech", "Energy"}
        assert set(zoom.ticker_network.mst.nodes()) == {"A", "B", "C"}
        assert zoom.ticker_network.excluded_tickers == ["D"]
        print("✓ build_semantic_zoom_network: end-to-end sector + ticker level MSTs built")

        print("✓ network.py smoke test passed")

    _smoke_test()
