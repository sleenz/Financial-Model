"""
Tests for Automated Reporting Module (Phase 9)
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from datetime import datetime

from src.reports.generator import ReportGenerator, generate_portfolio_report
from src.reports.templates import (
    PortfolioSummaryTemplate,
    PerformanceReviewTemplate,
    RiskDashboardTemplate
)
from src.reports.charts import ReportChartGenerator


@pytest.fixture
def sample_returns():
    """Create sample returns data."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='B')
    returns = pd.DataFrame(
        np.random.randn(252, 4) * 0.01,
        index=dates,
        columns=['AAPL', 'MSFT', 'GOOGL', 'JPM']
    )
    return returns


@pytest.fixture
def sample_weights():
    """Create sample portfolio weights."""
    return pd.Series({
        'AAPL': 0.30,
        'MSFT': 0.30,
        'GOOGL': 0.25,
        'JPM': 0.15
    })


@pytest.fixture
def sample_metrics():
    """Create sample portfolio metrics."""
    return {
        'annual_return': 0.12,
        'annual_volatility': 0.15,
        'sharpe_ratio': 0.8,
        'max_drawdown': -0.10,
        'var_95': -0.02,
        'cvar_95': -0.03,
        'sortino_ratio': 1.2,
        'calmar_ratio': 1.0
    }


class TestReportGenerator:
    """Tests for the main report generator."""

    def test_initialization(self):
        """Test generator initialization."""
        generator = ReportGenerator(title="Test Report")
        assert generator is not None
        assert generator.title == "Test Report"

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        generator = ReportGenerator(
            title="Custom Report",
            author="Test Author",
            page_size="A4"
        )
        assert generator.author == "Test Author"

    def test_add_title_page(self):
        """Test adding title page."""
        generator = ReportGenerator()
        generator.add_title_page(subtitle="Test Subtitle")

        # Should have elements
        assert len(generator.elements) > 0

    def test_add_section_header(self):
        """Test adding section header."""
        generator = ReportGenerator()
        generator.add_section_header("Test Section")

        assert len(generator.elements) > 0

    def test_add_paragraph(self):
        """Test adding paragraph."""
        generator = ReportGenerator()
        generator.add_paragraph("Test paragraph text")

        assert len(generator.elements) > 0

    def test_add_metrics_row(self):
        """Test adding metrics row."""
        generator = ReportGenerator()
        generator.add_metrics_row([
            ("Metric 1", "Value 1"),
            ("Metric 2", "Value 2")
        ])

        assert len(generator.elements) > 0

    def test_add_table(self):
        """Test adding table."""
        generator = ReportGenerator()
        data = [
            ["Row 1", "Value 1"],
            ["Row 2", "Value 2"]
        ]
        generator.add_table(data, headers=["Column 1", "Column 2"])

        assert len(generator.elements) > 0

    def test_add_spacer(self):
        """Test adding spacer."""
        generator = ReportGenerator()
        initial_count = len(generator.elements)
        generator.add_spacer(0.5)

        assert len(generator.elements) == initial_count + 1

    def test_add_page_break(self):
        """Test adding page break."""
        generator = ReportGenerator()
        initial_count = len(generator.elements)
        generator.add_page_break()

        assert len(generator.elements) == initial_count + 1

    def test_generate_pdf(self):
        """Test generating PDF file."""
        generator = ReportGenerator(title="Test PDF")
        generator.add_title_page()
        generator.add_section_header("Test Section")
        generator.add_paragraph("Test content")

        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            output_path = f.name

        try:
            result = generator.generate(output_path)
            assert os.path.exists(result)
            assert os.path.getsize(result) > 0
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_generate_bytes(self):
        """Test generating PDF as bytes."""
        generator = ReportGenerator(title="Test PDF")
        generator.add_title_page()
        generator.add_paragraph("Test content")

        pdf_bytes = generator.generate_bytes()

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0
        # PDF should start with %PDF
        assert pdf_bytes[:4] == b'%PDF'


class TestPortfolioSummaryTemplate:
    """Tests for portfolio summary template."""

    def test_build(self, sample_returns, sample_weights, sample_metrics):
        """Test building portfolio summary."""
        generator = ReportGenerator()
        template = PortfolioSummaryTemplate(generator)

        template.build(
            weights=sample_weights,
            returns=sample_returns,
            metrics=sample_metrics,
            portfolio_value=100000
        )

        # Should have elements
        assert len(generator.elements) > 0

    def test_generate_pdf(self, sample_returns, sample_weights, sample_metrics):
        """Test generating PDF with template."""
        generator = ReportGenerator()
        template = PortfolioSummaryTemplate(generator)

        template.build(
            weights=sample_weights,
            returns=sample_returns,
            metrics=sample_metrics
        )

        pdf_bytes = generator.generate_bytes()
        assert len(pdf_bytes) > 0


class TestPerformanceReviewTemplate:
    """Tests for performance review template."""

    def test_build(self, sample_returns, sample_weights):
        """Test building performance review."""
        generator = ReportGenerator()
        template = PerformanceReviewTemplate(generator)

        template.build(
            weights=sample_weights,
            returns=sample_returns,
            period_start=datetime(2023, 1, 1),
            period_end=datetime(2023, 12, 31)
        )

        assert len(generator.elements) > 0

    def test_build_with_benchmark(self, sample_returns, sample_weights):
        """Test building with benchmark."""
        generator = ReportGenerator()
        template = PerformanceReviewTemplate(generator)

        # Create benchmark returns
        benchmark = sample_returns.mean(axis=1)

        template.build(
            weights=sample_weights,
            returns=sample_returns,
            period_start=datetime(2023, 1, 1),
            period_end=datetime(2023, 12, 31),
            benchmark_returns=benchmark
        )

        assert len(generator.elements) > 0


class TestRiskDashboardTemplate:
    """Tests for risk dashboard template."""

    def test_build(self, sample_returns, sample_weights):
        """Test building risk dashboard."""
        generator = ReportGenerator()
        template = RiskDashboardTemplate(generator)

        template.build(
            weights=sample_weights,
            returns=sample_returns
        )

        assert len(generator.elements) > 0

    def test_build_with_var_results(self, sample_returns, sample_weights):
        """Test building with VaR results."""
        generator = ReportGenerator()
        template = RiskDashboardTemplate(generator)

        var_results = {
            'var_95': -0.02,
            'cvar_95': -0.03
        }

        template.build(
            weights=sample_weights,
            returns=sample_returns,
            var_results=var_results
        )

        assert len(generator.elements) > 0

    def test_build_with_stress_results(self, sample_returns, sample_weights):
        """Test building with stress test results."""
        generator = ReportGenerator()
        template = RiskDashboardTemplate(generator)

        stress_results = pd.DataFrame({
            'Portfolio Impact': [-0.15, -0.10, -0.05]
        }, index=['Scenario 1', 'Scenario 2', 'Scenario 3'])

        template.build(
            weights=sample_weights,
            returns=sample_returns,
            stress_results=stress_results
        )

        assert len(generator.elements) > 0


class TestReportChartGenerator:
    """Tests for chart generation."""

    def test_initialization(self):
        """Test chart generator initialization."""
        chart_gen = ReportChartGenerator()
        assert chart_gen is not None

    def test_allocation_pie_chart(self, sample_weights):
        """Test allocation pie chart."""
        chart_gen = ReportChartGenerator()
        chart = chart_gen.allocation_pie_chart(sample_weights)

        # Should return BytesIO with PNG data
        assert hasattr(chart, 'read')
        data = chart.getvalue()
        assert len(data) > 0

    def test_performance_line_chart(self, sample_returns, sample_weights):
        """Test performance line chart."""
        chart_gen = ReportChartGenerator()
        portfolio_returns = (sample_returns * sample_weights).sum(axis=1)

        chart = chart_gen.performance_line_chart(portfolio_returns)

        data = chart.getvalue()
        assert len(data) > 0

    def test_performance_with_benchmark(self, sample_returns, sample_weights):
        """Test performance chart with benchmark."""
        chart_gen = ReportChartGenerator()
        portfolio_returns = (sample_returns * sample_weights).sum(axis=1)
        benchmark = sample_returns.mean(axis=1)

        chart = chart_gen.performance_line_chart(
            portfolio_returns,
            benchmark_returns=benchmark
        )

        data = chart.getvalue()
        assert len(data) > 0

    def test_drawdown_chart(self, sample_returns, sample_weights):
        """Test drawdown chart."""
        chart_gen = ReportChartGenerator()
        portfolio_returns = (sample_returns * sample_weights).sum(axis=1)

        chart = chart_gen.drawdown_chart(portfolio_returns)

        data = chart.getvalue()
        assert len(data) > 0

    def test_correlation_heatmap(self, sample_returns):
        """Test correlation heatmap."""
        chart_gen = ReportChartGenerator()
        chart = chart_gen.correlation_heatmap(sample_returns)

        data = chart.getvalue()
        assert len(data) > 0

    def test_monthly_returns_heatmap(self, sample_returns, sample_weights):
        """Test monthly returns heatmap."""
        chart_gen = ReportChartGenerator()
        portfolio_returns = (sample_returns * sample_weights).sum(axis=1)

        chart = chart_gen.monthly_returns_heatmap(portfolio_returns)

        data = chart.getvalue()
        assert len(data) > 0

    def test_risk_contribution_bar(self):
        """Test risk contribution bar chart."""
        chart_gen = ReportChartGenerator()
        risk_contrib = pd.Series({
            'AAPL': 0.30,
            'MSFT': 0.25,
            'GOOGL': 0.25,
            'JPM': 0.20
        })

        chart = chart_gen.risk_contribution_bar(risk_contrib)

        data = chart.getvalue()
        assert len(data) > 0

    def test_stress_test_bar(self):
        """Test stress test bar chart."""
        chart_gen = ReportChartGenerator()
        stress_results = pd.DataFrame({
            'Portfolio Impact': [-0.15, -0.10, 0.05]
        }, index=['Crash', 'Correction', 'Rally'])

        chart = chart_gen.stress_test_bar(stress_results)

        data = chart.getvalue()
        assert len(data) > 0

    def test_distribution_histogram(self, sample_returns, sample_weights):
        """Test distribution histogram."""
        chart_gen = ReportChartGenerator()
        portfolio_returns = (sample_returns * sample_weights).sum(axis=1)

        chart = chart_gen.distribution_histogram(portfolio_returns)

        data = chart.getvalue()
        assert len(data) > 0

    def test_generate_all_charts(self, sample_returns, sample_weights):
        """Test generating all charts."""
        chart_gen = ReportChartGenerator()
        charts = chart_gen.generate_all_charts(sample_weights, sample_returns)

        assert 'allocation' in charts
        assert 'performance' in charts
        assert 'drawdown' in charts
        assert 'correlation' in charts
        assert 'monthly' in charts
        assert 'distribution' in charts


class TestGeneratePortfolioReport:
    """Tests for convenience function."""

    def test_generate_portfolio_report(self, sample_returns, sample_weights, sample_metrics):
        """Test generate_portfolio_report function."""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            output_path = f.name

        try:
            result = generate_portfolio_report(
                weights=sample_weights,
                returns=sample_returns,
                metrics=sample_metrics,
                output_path=output_path
            )

            assert os.path.exists(result)
            assert os.path.getsize(result) > 0
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_report_with_charts(self, sample_returns, sample_weights, sample_metrics):
        """Test generating full report with embedded charts."""
        # Generate charts
        chart_gen = ReportChartGenerator()
        charts = chart_gen.generate_all_charts(sample_weights, sample_returns)

        # Build report
        generator = ReportGenerator(title="Full Integration Test")
        template = PortfolioSummaryTemplate(generator)

        template.build(
            weights=sample_weights,
            returns=sample_returns,
            metrics=sample_metrics,
            chart_images=charts
        )

        # Generate PDF
        pdf_bytes = generator.generate_bytes()

        assert len(pdf_bytes) > 0
        assert pdf_bytes[:4] == b'%PDF'

    def test_multiple_templates(self, sample_returns, sample_weights, sample_metrics):
        """Test combining multiple templates."""
        generator = ReportGenerator(title="Multi-Template Test")

        # Portfolio summary
        summary_template = PortfolioSummaryTemplate(generator)
        summary_template.build(
            weights=sample_weights,
            returns=sample_returns,
            metrics=sample_metrics
        )

        generator.add_page_break()

        # Risk dashboard
        risk_template = RiskDashboardTemplate(generator)
        risk_template.build(
            weights=sample_weights,
            returns=sample_returns
        )

        pdf_bytes = generator.generate_bytes()
        assert len(pdf_bytes) > 0
