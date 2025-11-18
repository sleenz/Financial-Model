"""Hierarchical Risk Parity (HRP) implementation.

Based on: Marcos López de Prado (2016)
"Building Diversified Portfolios that Outperform Out of Sample"
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

from ..utils.logger import get_logger

logger = get_logger(__name__)


class HRPOptimizer:
    """
    Hierarchical Risk Parity optimizer.

    Uses hierarchical clustering to group similar assets and
    allocates risk through recursive bisection.

    Advantages over Mean-Variance:
    - More stable allocations
    - Reduces estimation error in covariance matrix
    - Handles singularity issues
    - Works well with high-dimensional data
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        use_shrinkage: bool = True,
        linkage_method: str = 'single',
    ):
        """
        Initialize HRP optimizer.

        Args:
            returns: DataFrame of asset returns
            use_shrinkage: Whether to use Ledoit-Wolf shrinkage for covariance
            linkage_method: Hierarchical clustering method
        """
        self.returns = returns
        self.tickers = list(returns.columns)
        self.n_assets = len(self.tickers)
        self.use_shrinkage = use_shrinkage
        self.linkage_method = linkage_method

        # Calculate covariance matrix
        if use_shrinkage:
            lw = LedoitWolf().fit(returns)
            self.cov_matrix = pd.DataFrame(
                lw.covariance_,
                index=self.tickers,
                columns=self.tickers
            )
        else:
            self.cov_matrix = returns.cov()

        # Calculate correlation matrix
        self.corr_matrix = returns.corr()

        logger.info(f"HRP Optimizer initialized with {self.n_assets} assets")

    def optimize(self) -> Dict:
        """
        Run HRP optimization.

        Returns:
            Dictionary with weights and clustering info
        """
        # Step 1: Tree Clustering
        dist_matrix, linkage_matrix = self._tree_clustering()

        # Step 2: Quasi-Diagonalization
        sorted_indices = self._quasi_diagonalize(linkage_matrix)
        sorted_tickers = [self.tickers[i] for i in sorted_indices]

        # Step 3: Recursive Bisection
        weights = self._recursive_bisection(sorted_tickers)

        # Create weights series in original order
        weight_series = pd.Series(
            [weights[t] for t in self.tickers],
            index=self.tickers
        )

        # Calculate portfolio metrics
        port_var = self._portfolio_variance(weight_series.values)
        port_vol = np.sqrt(port_var * 252)  # Annualized

        return {
            'weights': weight_series,
            'volatility': port_vol,
            'sorted_tickers': sorted_tickers,
            'linkage_matrix': linkage_matrix,
            'distance_matrix': dist_matrix,
        }

    def _tree_clustering(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform hierarchical tree clustering.

        Returns:
            Tuple of (distance_matrix, linkage_matrix)
        """
        # Calculate distance matrix from correlation
        # Distance = sqrt(0.5 * (1 - correlation))
        dist = np.sqrt(0.5 * (1 - self.corr_matrix))

        # Convert to condensed form for scipy
        dist_condensed = squareform(dist.values, checks=False)

        # Perform hierarchical clustering
        link = linkage(dist_condensed, method=self.linkage_method)

        return dist.values, link

    def _quasi_diagonalize(self, link: np.ndarray) -> List[int]:
        """
        Quasi-diagonalize the covariance matrix.

        Reorders assets so that similar assets are close together,
        producing a quasi-diagonal structure.

        Args:
            link: Linkage matrix from hierarchical clustering

        Returns:
            List of asset indices in quasi-diagonal order
        """
        return list(leaves_list(link))

    def _recursive_bisection(self, sorted_tickers: List[str]) -> Dict[str, float]:
        """
        Allocate weights through recursive bisection.

        Args:
            sorted_tickers: Tickers in quasi-diagonal order

        Returns:
            Dictionary of ticker weights
        """
        # Initialize all weights to 1
        weights = {ticker: 1.0 for ticker in sorted_tickers}

        # Stack of clusters to process
        clusters = [sorted_tickers]

        while clusters:
            cluster = clusters.pop(0)

            if len(cluster) == 1:
                continue

            # Bisect the cluster
            mid = len(cluster) // 2
            left_cluster = cluster[:mid]
            right_cluster = cluster[mid:]

            # Calculate cluster variances
            left_var = self._get_cluster_variance(left_cluster)
            right_var = self._get_cluster_variance(right_cluster)

            # Inverse variance allocation
            total_var = left_var + right_var
            if total_var > 0:
                left_weight = 1 - left_var / total_var
            else:
                left_weight = 0.5

            # Update weights
            for ticker in left_cluster:
                weights[ticker] *= left_weight
            for ticker in right_cluster:
                weights[ticker] *= (1 - left_weight)

            # Add sub-clusters for further processing
            if len(left_cluster) > 1:
                clusters.append(left_cluster)
            if len(right_cluster) > 1:
                clusters.append(right_cluster)

        return weights

    def _get_cluster_variance(self, tickers: List[str]) -> float:
        """
        Calculate variance of an inverse-variance weighted cluster.

        Args:
            tickers: List of tickers in the cluster

        Returns:
            Cluster variance
        """
        # Get cluster covariance matrix
        cov_slice = self.cov_matrix.loc[tickers, tickers]

        # Inverse variance portfolio within cluster
        inv_var = 1 / np.diag(cov_slice)
        inv_var_weights = inv_var / inv_var.sum()

        # Cluster variance
        cluster_var = np.dot(inv_var_weights, np.dot(cov_slice, inv_var_weights))

        return cluster_var

    def _portfolio_variance(self, weights: np.ndarray) -> float:
        """Calculate portfolio variance."""
        return np.dot(weights.T, np.dot(self.cov_matrix, weights))

    def get_dendrogram_data(self) -> Dict:
        """
        Get data for plotting dendrogram.

        Returns:
            Dictionary with dendrogram plotting data
        """
        dist = np.sqrt(0.5 * (1 - self.corr_matrix))
        dist_condensed = squareform(dist.values, checks=False)
        link = linkage(dist_condensed, method=self.linkage_method)

        return {
            'linkage': link,
            'labels': self.tickers,
        }

    def get_cluster_members(self, n_clusters: int = None) -> Dict[int, List[str]]:
        """
        Get cluster membership for each asset.

        Args:
            n_clusters: Number of clusters (default: sqrt(n_assets))

        Returns:
            Dictionary mapping cluster ID to list of tickers
        """
        from scipy.cluster.hierarchy import fcluster

        if n_clusters is None:
            n_clusters = max(2, int(np.sqrt(self.n_assets)))

        dist = np.sqrt(0.5 * (1 - self.corr_matrix))
        dist_condensed = squareform(dist.values, checks=False)
        link = linkage(dist_condensed, method=self.linkage_method)

        cluster_labels = fcluster(link, n_clusters, criterion='maxclust')

        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(self.tickers[i])

        return clusters


def hrp_allocation(returns: pd.DataFrame, **kwargs) -> pd.Series:
    """
    Convenience function for HRP allocation.

    Args:
        returns: DataFrame of asset returns
        **kwargs: Additional arguments for HRPOptimizer

    Returns:
        Series of portfolio weights
    """
    optimizer = HRPOptimizer(returns, **kwargs)
    result = optimizer.optimize()
    return result['weights']
