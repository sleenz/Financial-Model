"""
Chart Generation for Reports

Creates charts that can be embedded in PDF reports.
"""

import io
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ReportChartGenerator:
    """
    Generates charts for embedding in PDF reports.
    """

    def __init__(
        self,
        style: str = 'seaborn-v0_8-whitegrid',
        figsize: Tuple[int, int] = (10, 6),
        dpi: int = 150
    ):
        """
        Initialize chart generator.

        Args:
            style: Matplotlib style
            figsize: Default figure size
            dpi: Resolution for saved images
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn' in plt.style.available else 'ggplot')

        self.figsize = figsize
        self.dpi = dpi

        # Color palette
        self.colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]

        logger.info("ReportChartGenerator initialized")

    def _save_figure(self, fig: Figure, format: str = 'png') -> io.BytesIO:
        """Save figure to bytes buffer."""
        buf = io.BytesIO()
        fig.savefig(buf, format=format, dpi=self.dpi, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf

    def allocation_pie_chart(
        self,
        weights: pd.Series,
        title: str = "Portfolio Allocation"
    ) -> io.BytesIO:
        """
        Create portfolio allocation pie chart.

        Args:
            weights: Portfolio weights
            title: Chart title

        Returns:
            BytesIO buffer with chart image
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        # Sort and get top holdings
        sorted_weights = weights.sort_values(ascending=False)

        # Group small positions
        threshold = 0.03
        main_weights = sorted_weights[sorted_weights >= threshold]
        other_weight = sorted_weights[sorted_weights < threshold].sum()

        if other_weight > 0:
            main_weights = pd.concat([
                main_weights,
                pd.Series({'Other': other_weight})
            ])

        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            main_weights.values,
            labels=main_weights.index,
            autopct='%1.1f%%',
            colors=self.colors[:len(main_weights)],
            pctdistance=0.75,
            startangle=90
        )

        # Style
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_weight('bold')

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        return self._save_figure(fig)

    def performance_line_chart(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        title: str = "Cumulative Performance"
    ) -> io.BytesIO:
        """
        Create cumulative performance chart.

        Args:
            returns: Portfolio returns
            benchmark_returns: Optional benchmark returns
            title: Chart title

        Returns:
            BytesIO buffer with chart image
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Calculate cumulative returns
        cumulative = (1 + returns).cumprod()

        ax.plot(
            cumulative.index,
            cumulative.values,
            label='Portfolio',
            color=self.colors[0],
            linewidth=2
        )

        if benchmark_returns is not None:
            bench_cumulative = (1 + benchmark_returns).cumprod()
            ax.plot(
                bench_cumulative.index,
                bench_cumulative.values,
                label='Benchmark',
                color=self.colors[1],
                linewidth=2,
                linestyle='--'
            )

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)

        return self._save_figure(fig)

    def drawdown_chart(
        self,
        returns: pd.Series,
        title: str = "Drawdown"
    ) -> io.BytesIO:
        """
        Create drawdown chart.

        Args:
            returns: Portfolio returns
            title: Chart title

        Returns:
            BytesIO buffer with chart image
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Calculate drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max

        ax.fill_between(
            drawdown.index,
            drawdown.values,
            0,
            color=self.colors[3],
            alpha=0.5
        )
        ax.plot(
            drawdown.index,
            drawdown.values,
            color=self.colors[3],
            linewidth=1
        )

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown')
        ax.grid(True, alpha=0.3)

        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)

        return self._save_figure(fig)

    def correlation_heatmap(
        self,
        returns: pd.DataFrame,
        title: str = "Correlation Matrix"
    ) -> io.BytesIO:
        """
        Create correlation heatmap.

        Args:
            returns: Asset returns DataFrame
            title: Chart title

        Returns:
            BytesIO buffer with chart image
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        corr = returns.corr()

        # Create heatmap
        im = ax.imshow(corr, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation')

        # Set ticks
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr.columns)

        # Add correlation values
        for i in range(len(corr)):
            for j in range(len(corr)):
                text = ax.text(
                    j, i, f'{corr.iloc[i, j]:.2f}',
                    ha='center', va='center',
                    color='white' if abs(corr.iloc[i, j]) > 0.5 else 'black',
                    fontsize=8
                )

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        return self._save_figure(fig)

    def monthly_returns_heatmap(
        self,
        returns: pd.Series,
        title: str = "Monthly Returns"
    ) -> io.BytesIO:
        """
        Create monthly returns heatmap.

        Args:
            returns: Portfolio returns
            title: Chart title

        Returns:
            BytesIO buffer with chart image
        """
        # Resample to monthly
        monthly = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)

        # Create pivot table (year x month)
        monthly_df = pd.DataFrame({
            'Year': monthly.index.year,
            'Month': monthly.index.month,
            'Return': monthly.values
        })

        pivot = monthly_df.pivot(index='Year', columns='Month', values='Return')

        fig, ax = plt.subplots(figsize=(12, 6))

        # Create heatmap
        im = ax.imshow(
            pivot.values,
            cmap='RdYlGn',
            aspect='auto',
            vmin=-0.1,
            vmax=0.1
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Return')

        # Set ticks
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticks(range(12))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels(months)
        ax.set_yticklabels(pivot.index)

        # Add return values
        for i in range(len(pivot.index)):
            for j in range(12):
                if j < len(pivot.columns) and not pd.isna(pivot.iloc[i, j]):
                    val = pivot.iloc[i, j]
                    text = ax.text(
                        j, i, f'{val*100:.1f}%',
                        ha='center', va='center',
                        color='white' if abs(val) > 0.05 else 'black',
                        fontsize=7
                    )

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        return self._save_figure(fig)

    def risk_contribution_bar(
        self,
        risk_contrib: pd.Series,
        title: str = "Risk Contribution"
    ) -> io.BytesIO:
        """
        Create risk contribution bar chart.

        Args:
            risk_contrib: Risk contribution by asset
            title: Chart title

        Returns:
            BytesIO buffer with chart image
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        sorted_contrib = risk_contrib.sort_values(ascending=True)

        y_pos = range(len(sorted_contrib))

        bars = ax.barh(
            y_pos,
            sorted_contrib.values * 100,
            color=self.colors[0],
            alpha=0.8
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_contrib.index)
        ax.set_xlabel('Risk Contribution (%)')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for bar, val in zip(bars, sorted_contrib.values):
            ax.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height()/2,
                f'{val*100:.1f}%',
                va='center',
                fontsize=9
            )

        return self._save_figure(fig)

    def stress_test_bar(
        self,
        stress_results: pd.DataFrame,
        title: str = "Stress Test Results"
    ) -> io.BytesIO:
        """
        Create stress test results bar chart.

        Args:
            stress_results: Stress test results with 'Portfolio Impact' column
            title: Chart title

        Returns:
            BytesIO buffer with chart image
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        if 'Portfolio Impact' in stress_results.columns:
            impacts = stress_results['Portfolio Impact'] * 100
        else:
            impacts = stress_results.iloc[:, 0] * 100

        sorted_results = impacts.sort_values()

        y_pos = range(len(sorted_results))

        colors = [self.colors[3] if v < 0 else self.colors[2] for v in sorted_results.values]

        bars = ax.barh(
            y_pos,
            sorted_results.values,
            color=colors,
            alpha=0.8
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_results.index)
        ax.set_xlabel('Portfolio Impact (%)')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='x')

        return self._save_figure(fig)

    def distribution_histogram(
        self,
        returns: pd.Series,
        title: str = "Return Distribution"
    ) -> io.BytesIO:
        """
        Create return distribution histogram.

        Args:
            returns: Portfolio returns
            title: Chart title

        Returns:
            BytesIO buffer with chart image
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Histogram
        n, bins, patches = ax.hist(
            returns * 100,
            bins=50,
            color=self.colors[0],
            alpha=0.7,
            edgecolor='white'
        )

        # Add VaR lines
        var_95 = np.percentile(returns, 5) * 100
        var_99 = np.percentile(returns, 1) * 100

        ax.axvline(
            var_95, color=self.colors[1],
            linestyle='--', linewidth=2,
            label=f'VaR 95%: {var_95:.2f}%'
        )
        ax.axvline(
            var_99, color=self.colors[3],
            linestyle='--', linewidth=2,
            label=f'VaR 99%: {var_99:.2f}%'
        )

        ax.set_xlabel('Daily Return (%)')
        ax.set_ylabel('Frequency')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return self._save_figure(fig)

    def generate_all_charts(
        self,
        weights: pd.Series,
        returns: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, io.BytesIO]:
        """
        Generate all standard charts.

        Args:
            weights: Portfolio weights
            returns: Asset returns DataFrame
            benchmark_returns: Optional benchmark returns

        Returns:
            Dictionary of chart name -> BytesIO buffer
        """
        portfolio_returns = (returns[weights.index] * weights).sum(axis=1)

        charts = {
            'allocation': self.allocation_pie_chart(weights),
            'performance': self.performance_line_chart(portfolio_returns, benchmark_returns),
            'drawdown': self.drawdown_chart(portfolio_returns),
            'correlation': self.correlation_heatmap(returns[weights.index]),
            'monthly': self.monthly_returns_heatmap(portfolio_returns),
            'distribution': self.distribution_histogram(portfolio_returns)
        }

        logger.info(f"Generated {len(charts)} charts")

        return charts
