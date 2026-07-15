import re
import os
from datetime import datetime
from typing import Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    HRFlowable,
    Table,
    TableStyle,
    PageBreak,
)

_DARK_BLUE   = colors.HexColor("#1a3a5c")
_MID_BLUE    = colors.HexColor("#2c5f8a")
_LIGHT_BLUE  = colors.HexColor("#e8f0f7")
_ACCENT      = colors.HexColor("#c8392b")
_LIGHT_GREY  = colors.HexColor("#f5f5f5")
_MID_GREY    = colors.HexColor("#888888")
_DARK_GREY   = colors.HexColor("#333333")



def _build_styles() -> dict:
    base = getSampleStyleSheet()

    styles = {
        "cover_title": ParagraphStyle(
            "cover_title",
            fontSize=26,
            fontName="Helvetica-Bold",
            textColor=_DARK_BLUE,
            spaceAfter=8,
            leading=32,
        ),
        "cover_subtitle": ParagraphStyle(
            "cover_subtitle",
            fontSize=13,
            fontName="Helvetica",
            textColor=_MID_GREY,
            spaceAfter=4,
        ),
        "cover_meta": ParagraphStyle(
            "cover_meta",
            fontSize=11,
            fontName="Helvetica",
            textColor=_DARK_GREY,
            spaceAfter=4,
        ),
        "section_heading": ParagraphStyle(
            "section_heading",
            fontSize=13,
            fontName="Helvetica-Bold",
            textColor=_DARK_BLUE,
            spaceBefore=18,
            spaceAfter=6,
            borderPad=4,
        ),
        "body": ParagraphStyle(
            "body",
            fontSize=10,
            fontName="Helvetica",
            textColor=_DARK_GREY,
            spaceAfter=6,
            leading=15,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            fontSize=10,
            fontName="Helvetica",
            textColor=_DARK_GREY,
            spaceAfter=4,
            leading=14,
            leftIndent=16,
            bulletIndent=4,
        ),
        "bold_body": ParagraphStyle(
            "bold_body",
            fontSize=10,
            fontName="Helvetica-Bold",
            textColor=_DARK_GREY,
            spaceAfter=4,
            leading=14,
        ),
        "warning": ParagraphStyle(
            "warning",
            fontSize=9,
            fontName="Helvetica-Oblique",
            textColor=_ACCENT,
            spaceAfter=4,
        ),
        "footer": ParagraphStyle(
            "footer",
            fontSize=8,
            fontName="Helvetica",
            textColor=_MID_GREY,
        ),
        "stat_label": ParagraphStyle(
            "stat_label",
            fontSize=9,
            fontName="Helvetica",
            textColor=_MID_GREY,
            spaceAfter=2,
        ),
        "stat_value": ParagraphStyle(
            "stat_value",
            fontSize=14,
            fontName="Helvetica-Bold",
            textColor=_DARK_BLUE,
            spaceAfter=2,
        ),
    }
    return styles



def _on_page(canvas, doc):
    """Draw header and footer on every page except the cover."""
    if doc.page == 1:
        return

    canvas.saveState()
    width, height = letter

    # Header bar
    canvas.setFillColor(_DARK_BLUE)
    canvas.rect(0, height - 36, width, 36, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(0.5 * inch, height - 22, "PI CASE STRESS-TESTER")
    canvas.setFont("Helvetica", 9)
    canvas.drawRightString(width - 0.5 * inch, height - 22, "Ontario Personal Injury Research")

    # Footer
    canvas.setFillColor(_MID_GREY)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(
        0.5 * inch, 0.4 * inch,
        "CONFIDENTIAL — For counsel use only. Not legal advice. Verify all citations independently."
    )
    canvas.drawRightString(
        width - 0.5 * inch, 0.4 * inch,
        f"Page {doc.page}"
    )

    canvas.restoreState()



def _build_cover(styles: dict, lawyer_query: str, stats: dict) -> list:
    story = []
    width, _ = letter

    story.append(Table(
        [[""]],
        colWidths=[6.5 * inch],
        rowHeights=[6],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), _DARK_BLUE),
            ("LINEBELOW", (0, 0), (-1, -1), 0, colors.white),
        ]),
    ))
    story.append(Spacer(1, 0.4 * inch))

    story.append(Paragraph("PI Case Stress-Test", styles["cover_title"]))
    story.append(Paragraph("Ontario Personal Injury Research Report", styles["cover_subtitle"]))
    story.append(Spacer(1, 0.15 * inch))

    story.append(HRFlowable(width=6.5 * inch, thickness=1, color=_LIGHT_BLUE))
    story.append(Spacer(1, 0.15 * inch))

    # Case description box
    query_display = lawyer_query[:800] + "..." if len(lawyer_query) > 800 else lawyer_query
    case_table = Table(
        [[Paragraph(f"<b>Case Description</b><br/>{query_display}", styles["body"])]],
        colWidths=[6.5 * inch],
        style=TableStyle([
            ("BACKGROUND",   (0, 0), (-1, -1), _LIGHT_BLUE),
            ("BOX",          (0, 0), (-1, -1), 1, _MID_BLUE),
            ("TOPPADDING",   (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 10),
            ("LEFTPADDING",  (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ]),
    )
    story.append(case_table)
    story.append(Spacer(1, 0.3 * inch))

    # Stats summary row
    damages = stats.get("damages", {})
    win_rate = stats.get("win_rate_pct", "N/A")
    total    = stats.get("total_cases", 0)
    median   = f"${damages.get('median', 0):,}" if damages.get("median") else "N/A"

    stat_data = [
        [
            _stat_cell("Win Rate", win_rate, styles),
            _stat_cell("Comparable Cases", str(total), styles),
            _stat_cell("Median Award", median, styles),
        ]
    ]
    stat_table = Table(
        stat_data,
        colWidths=[1.625 * inch] * 3,
        style=TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), _LIGHT_GREY),
            ("BOX",           (0, 0), (-1, -1), 1, _MID_BLUE),
            ("INNERGRID",     (0, 0), (-1, -1), 0.5, _LIGHT_BLUE),
            ("TOPPADDING",    (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ]),
    )
    story.append(stat_table)
    story.append(Spacer(1, 0.3 * inch))

    if stats.get("low_sample"):
        story.append(Paragraph(
            f"Low sample warning: only {total} comparable cases found. "
            "Statistics are directional only.",
            styles["warning"],
        ))
        story.append(Spacer(1, 0.1 * inch))

    story.append(HRFlowable(width=6.5 * inch, thickness=0.5, color=_LIGHT_BLUE))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}  |  "
        f"Dataset: Ontario PI decisions  |  "
        f"Model: voyage-law-2 + Gemma 4 31B",
        styles["cover_meta"],
    ))

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(
        "CONFIDENTIAL — For counsel use only. Not legal advice. "
        "Verify all citations independently before relying on them in court.",
        styles["warning"],
    ))

    story.append(PageBreak())
    return story


def _stat_cell(label: str, value: str, styles: dict) -> list:
    return [
        Paragraph(label, styles["stat_label"]),
        Paragraph(value, styles["stat_value"]),
    ]


def _parse_memo(memo: str, styles: dict) -> list:
    """
    Convert the plain-text memo into reportlab Platypus flowables.

    Handles:
      ## Section headings
      **Bold text** inline
      * Bullet points
      Regular paragraphs
    """
    story = []
    lines = memo.split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            story.append(Spacer(1, 4))
            continue

        # Section heading
        if line.startswith("## "):
            text = line[3:].strip()
            story.append(HRFlowable(
                width="100%", thickness=0.5,
                color=_LIGHT_BLUE, spaceAfter=2
            ))
            story.append(Paragraph(text, styles["section_heading"]))
            continue

        # Bullet point (*, -, or numbered)
        if re.match(r"^[\*\-\•]\s+", line) or re.match(r"^\d+\.\s+", line):
            text = re.sub(r"^[\*\-\•\d\.]+\s+", "", line)
            text = _md_to_rl(text)
            story.append(Paragraph(f"• {text}", styles["bullet"]))
            continue

        # Regular paragraph
        text = _md_to_rl(line)
        story.append(Paragraph(text, styles["body"]))

    return story


def _build_stress_test_section(stress_test: str, styles: dict) -> list:
    """
    Build the adverse stress test section for the PDF.

    Starts on a new page with a red accent header to visually
    distinguish it from the research memo.
    """
    story = []

    story.append(PageBreak())

    story.append(Table(
        [[""]],
        colWidths=[6.5 * inch],
        rowHeights=[6],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), _ACCENT),
        ]),
    ))
    story.append(Spacer(1, 0.3 * inch))

    story.append(Paragraph(
        "ADVERSE STRESS TEST",
        ParagraphStyle(
            "stress_title",
            fontSize=20,
            fontName="Helvetica-Bold",
            textColor=_ACCENT,
            spaceAfter=16,
            leading=28,
        )
    ))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        "Defense's strongest case against you",
        ParagraphStyle(
            "stress_subtitle",
            fontSize=11,
            fontName="Helvetica-Oblique",
            textColor=_MID_GREY,
            spaceAfter=16,
        )
    ))
    story.append(Spacer(1, 0.15 * inch))

    story.append(HRFlowable(
        width="100%", thickness=1,
        color=_ACCENT, spaceAfter=12
    ))

    warning_table = Table(
        [[Paragraph(
            "This section presents the defense's strongest arguments against "
            "this case. It is intended to help counsel prepare for trial, not "
            "to reflect the likely outcome. Review with your client accordingly.",
            ParagraphStyle(
                "stress_warning",
                fontSize=9,
                fontName="Helvetica-Oblique",
                textColor=_ACCENT,
            )
        )]],
        colWidths=[6.5 * inch],
        style=TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#fdf2f2")),
            ("BOX",           (0, 0), (-1, -1), 1, _ACCENT),
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING",   (0, 0), (-1, -1), 10),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ]),
    )
    story.append(warning_table)
    story.append(Spacer(1, 0.2 * inch))

    lines = stress_test.split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            story.append(Spacer(1, 4))
            continue

        if line.isupper() and len(line) > 3:
            story.append(Spacer(1, 8))
            story.append(Paragraph(
                line,
                ParagraphStyle(
                    "stress_heading",
                    fontSize=11,
                    fontName="Helvetica-Bold",
                    textColor=_ACCENT,
                    spaceBefore=10,
                    spaceAfter=4,
                )
            ))
            story.append(HRFlowable(
                width="100%", thickness=0.5,
                color=colors.HexColor("#f5c6c6"), spaceAfter=4
            ))
            continue

        if line.startswith("==="):
            continue

        if re.match(r"^[\*\-\•]\s+", line) or re.match(r"^\d+\.\s+", line):
            text = re.sub(r"^[\*\-\•\d\.]+\s+", "", line)
            text = _md_to_rl(text)
            story.append(Paragraph(
                f"• {text}",
                ParagraphStyle(
                    "stress_bullet",
                    fontSize=10,
                    fontName="Helvetica",
                    textColor=_DARK_GREY,
                    spaceAfter=4,
                    leading=14,
                    leftIndent=16,
                    bulletIndent=4,
                )
            ))
            continue

        text = _md_to_rl(line)
        story.append(Paragraph(text, styles["body"]))

    return story


def _md_to_rl(text: str) -> str:
    """Convert basic markdown to reportlab XML."""
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.*?)\*", r"<i>\1</i>", text)
    text = re.sub(r"&(?!amp;|lt;|gt;|quot;|#)", "&amp;", text)
    return text


def export_pdf(
    lawyer_query:   str,
    memo:           str,
    stats:          dict,
    stress_test:    Optional[str] = None,
    output_path:    Optional[str] = None,
) -> str:
    """
    Generate a PDF report from the research memo.

    Args:
        lawyer_query:  Original plain-English case description.
        memo:          Full memo text from memo.py.
        stats:         Aggregated stats from aggregator.py.
        output_path:   Override output path. Defaults to timestamped filename.

    Returns:
        Path to the generated PDF file.
    """
    if output_path is None:
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"exports/pi_memo_{timestamp}.pdf"

    styles = _build_styles()

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.65 * inch,
        title="PI Case Stress-Test Report",
        author="PI Case Stress-Tester",
        subject="Ontario Personal Injury Research",
    )

    story = []
    story += _build_cover(styles, lawyer_query, stats)
    story += _parse_memo(memo, styles)

    if stress_test:
        story += _build_stress_test_section(stress_test, styles)

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)

    return output_path