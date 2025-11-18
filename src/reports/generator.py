"""
PDF Report Generator

Main class for generating portfolio analysis reports.
"""

import io
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image, PageBreak, HRFlowable
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """
    Generates professional PDF reports for portfolio analysis.
    """

    def __init__(
        self,
        title: str = "Portfolio Analysis Report",
        author: str = "Portfolio Optimization System",
        page_size: str = "letter"
    ):
        """
        Initialize report generator.

        Args:
            title: Report title
            author: Report author
            page_size: 'letter' or 'A4'
        """
        if not REPORTLAB_AVAILABLE:
            raise ImportError(
                "reportlab is required for PDF generation. "
                "Install with: pip install reportlab"
            )

        self.title = title
        self.author = author
        self.page_size = letter if page_size == "letter" else A4

        # Setup styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

        self.elements = []

        logger.info(f"ReportGenerator initialized: {title}")

    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1a1a2e')
        ))

        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='ReportSubtitle',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#666666')
        ))

        # Section header
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor('#16213e')
        ))

        # Metric label
        self.styles.add(ParagraphStyle(
            name='MetricLabel',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#666666')
        ))

        # Metric value
        self.styles.add(ParagraphStyle(
            name='MetricValue',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=colors.HexColor('#1a1a2e')
        ))

    def add_title_page(
        self,
        subtitle: Optional[str] = None,
        date: Optional[datetime] = None
    ):
        """Add a title page to the report."""
        # Title
        self.elements.append(Spacer(1, 2 * inch))
        self.elements.append(Paragraph(self.title, self.styles['ReportTitle']))

        # Subtitle
        if subtitle:
            self.elements.append(Paragraph(subtitle, self.styles['ReportSubtitle']))

        # Date
        report_date = date or datetime.now()
        date_str = report_date.strftime("%B %d, %Y")
        self.elements.append(Spacer(1, 0.5 * inch))
        self.elements.append(Paragraph(
            f"Generated: {date_str}",
            self.styles['ReportSubtitle']
        ))

        # Author
        self.elements.append(Paragraph(
            f"By: {self.author}",
            self.styles['ReportSubtitle']
        ))

        self.elements.append(PageBreak())

    def add_section_header(self, text: str):
        """Add a section header."""
        self.elements.append(Paragraph(text, self.styles['SectionHeader']))
        self.elements.append(HRFlowable(
            width="100%",
            thickness=1,
            color=colors.HexColor('#e0e0e0'),
            spaceAfter=10
        ))

    def add_paragraph(self, text: str, style: str = 'Normal'):
        """Add a paragraph of text."""
        self.elements.append(Paragraph(text, self.styles[style]))
        self.elements.append(Spacer(1, 0.1 * inch))

    def add_spacer(self, height: float = 0.25):
        """Add vertical space (in inches)."""
        self.elements.append(Spacer(1, height * inch))

    def add_metrics_row(self, metrics: List[Tuple[str, str]]):
        """
        Add a row of metrics.

        Args:
            metrics: List of (label, value) tuples
        """
        data = [[m[0] for m in metrics], [m[1] for m in metrics]]

        table = Table(data, colWidths=[1.5 * inch] * len(metrics))
        table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#666666')),
            ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 1), (-1, 1), 14),
            ('TEXTCOLOR', (0, 1), (-1, 1), colors.HexColor('#1a1a2e')),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 5),
            ('TOPPADDING', (0, 1), (-1, 1), 5),
        ]))

        self.elements.append(table)
        self.elements.append(Spacer(1, 0.2 * inch))

    def add_table(
        self,
        data: List[List],
        headers: Optional[List[str]] = None,
        col_widths: Optional[List[float]] = None
    ):
        """
        Add a data table.

        Args:
            data: Table data (list of rows)
            headers: Optional column headers
            col_widths: Optional column widths in inches
        """
        if headers:
            table_data = [headers] + data
        else:
            table_data = data

        if col_widths:
            widths = [w * inch for w in col_widths]
        else:
            widths = None

        table = Table(table_data, colWidths=widths)

        style = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16213e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e0e0e0')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        ]

        table.setStyle(TableStyle(style))
        self.elements.append(table)
        self.elements.append(Spacer(1, 0.2 * inch))

    def add_image(
        self,
        image_path: str,
        width: float = 6,
        height: Optional[float] = None
    ):
        """
        Add an image to the report.

        Args:
            image_path: Path to image file or BytesIO object
            width: Image width in inches
            height: Optional height in inches
        """
        if height:
            img = Image(image_path, width=width * inch, height=height * inch)
        else:
            img = Image(image_path, width=width * inch)

        self.elements.append(img)
        self.elements.append(Spacer(1, 0.2 * inch))

    def add_page_break(self):
        """Add a page break."""
        self.elements.append(PageBreak())

    def generate(self, output_path: str) -> str:
        """
        Generate the PDF report.

        Args:
            output_path: Output file path

        Returns:
            Path to generated PDF
        """
        doc = SimpleDocTemplate(
            output_path,
            pagesize=self.page_size,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch
        )

        doc.build(self.elements)
        logger.info(f"Report generated: {output_path}")

        return output_path

    def generate_bytes(self) -> bytes:
        """
        Generate PDF as bytes (for downloads).

        Returns:
            PDF content as bytes
        """
        buffer = io.BytesIO()

        doc = SimpleDocTemplate(
            buffer,
            pagesize=self.page_size,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch
        )

        doc.build(self.elements)

        pdf_bytes = buffer.getvalue()
        buffer.close()

        return pdf_bytes


def generate_portfolio_report(
    weights: pd.Series,
    returns: pd.DataFrame,
    metrics: Dict,
    output_path: str,
    title: str = "Portfolio Analysis Report"
) -> str:
    """
    Convenience function to generate a complete portfolio report.

    Args:
        weights: Portfolio weights
        returns: Asset returns DataFrame
        metrics: Portfolio metrics dictionary
        output_path: Output file path
        title: Report title

    Returns:
        Path to generated PDF
    """
    from .templates import PortfolioSummaryTemplate

    generator = ReportGenerator(title=title)
    template = PortfolioSummaryTemplate(generator)

    template.build(
        weights=weights,
        returns=returns,
        metrics=metrics
    )

    return generator.generate(output_path)
