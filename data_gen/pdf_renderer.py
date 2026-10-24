"""Render hospital_bill.pdf and discharge_summary.pdf with reportlab."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from data_gen.models import ClaimSpec, Narrative

NAVY = colors.HexColor("#1f3a5f")
LIGHT = colors.HexColor("#eef2f7")
BLANK = "________________"

_base = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=_base["Normal"], fontName="Helvetica", fontSize=9, leading=12)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8, leading=10, textColor=colors.HexColor("#444444"))
HOSP = ParagraphStyle("hosp", parent=BODY, fontName="Helvetica-Bold", fontSize=16, leading=19, textColor=NAVY)
TITLE = ParagraphStyle("title", parent=BODY, fontName="Helvetica-Bold", fontSize=12, alignment=1, leading=15)
HEAD = ParagraphStyle("head", parent=BODY, fontName="Helvetica-Bold", fontSize=10, textColor=NAVY, spaceBefore=8)
CELL_R = ParagraphStyle("cell_r", parent=BODY, alignment=2)


def _fmt_date(d: date | None) -> str:
    return d.strftime("%d-%m-%Y") if d else BLANK


def _money(amount: int) -> str:
    return f"{amount:,}"


def _header(spec: ClaimSpec) -> list:
    h = spec.hospital
    return [
        Paragraph(h.name, HOSP),
        Paragraph(f"{h.address}", SMALL),
        Paragraph(f"Phone: {h.phone} &nbsp;|&nbsp; GSTIN: {h.gstin} &nbsp;|&nbsp; Reg. No: {h.registration_no}", SMALL),
        Spacer(1, 3),
        HRFlowable(width="100%", thickness=1.5, color=NAVY),
        Spacer(1, 4),
    ]


def _info_table(rows: list[tuple[str, str, str, str]]) -> Table:
    data = [[Paragraph(f"<b>{a}</b>", BODY), Paragraph(b, BODY), Paragraph(f"<b>{c}</b>", BODY), Paragraph(d, BODY)] for a, b, c, d in rows]
    table = Table(data, colWidths=[32 * mm, 58 * mm, 32 * mm, 58 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("BACKGROUND", (0, 0), (0, -1), LIGHT),
                ("BACKGROUND", (2, 0), (2, -1), LIGHT),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def render_bill(spec: ClaimSpec, path: Path) -> None:
    doc = SimpleDocTemplate(
        str(path), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=12 * mm, bottomMargin=12 * mm,
        invariant=1, title=f"Hospital Bill {spec.bill_no}",
    )  # fmt: skip
    p = spec.patient
    story = _header(spec)
    story += [Paragraph("FINAL INPATIENT BILL", TITLE), Spacer(1, 5)]
    story.append(
        _info_table(
            [
                ("Patient Name", p.name, "Bill No.", spec.bill_no),
                ("Age / Sex", f"{p.age} / {p.gender}", "Bill Date", _fmt_date(spec.discharge_date)),
                ("UHID", p.uhid, "Date of Admission", _fmt_date(spec.admission_date)),
                ("Treating Doctor", spec.doctor, "Date of Discharge", _fmt_date(spec.discharge_date)),
                ("Diagnosis", spec.diagnosis_name, "Patient Phone", p.phone),
            ]
        )
    )
    story.append(Spacer(1, 8))

    data = [["#", "Description", "Qty", "Rate (Rs)", "Amount (Rs)"]]
    for i, line in enumerate(spec.bill_lines, start=1):
        data.append([str(i), Paragraph(line.description, BODY), str(line.qty), _money(line.rate), _money(line.amount)])
    total_blank = "hospital_bill.total_amount" in spec.blank_fields
    data.append(["", "", "", "Sub Total", _money(spec.bill_subtotal)])
    if spec.gst_on_room:
        data.append(["", "", "", "GST @5% (room rent)", _money(spec.gst_on_room)])
    data.append(["", "", "", "TOTAL", BLANK if total_blank else _money(spec.bill_total)])

    n = len(data)
    table = Table(data, colWidths=[10 * mm, 82 * mm, 14 * mm, 38 * mm, 36 * mm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
                ("GRID", (0, 0), (-1, n - 1), 0.25, colors.lightgrey),
                ("LINEABOVE", (3, n - (3 if spec.gst_on_room else 2)), (-1, n - (3 if spec.gst_on_room else 2)), 0.8, colors.black),
                ("FONTNAME", (3, n - 1), (-1, n - 1), "Helvetica-Bold"),
                ("BACKGROUND", (3, n - 1), (-1, n - 1), LIGHT),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story += [table, Spacer(1, 14)]
    story.append(Paragraph("This is a computer-generated bill. Synthetic document for software testing only.", SMALL))
    story += [Spacer(1, 18), Paragraph("Authorised Signatory", BODY)]
    doc.build(story)


def render_summary(spec: ClaimSpec, narrative: Narrative, path: Path) -> None:
    doc = SimpleDocTemplate(
        str(path), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=12 * mm, bottomMargin=12 * mm,
        invariant=1, title=f"Discharge Summary {spec.claim_id}",
    )  # fmt: skip
    p = spec.patient
    adm = None if "discharge_summary.admission_date" in spec.blank_fields else spec.summary_admission_date
    story = _header(spec)
    story += [Paragraph("DISCHARGE SUMMARY", TITLE), Spacer(1, 5)]
    story.append(
        _info_table(
            [
                ("Patient Name", p.name, "UHID", p.uhid),
                ("Age / Sex", f"{p.age} / {p.gender}", "Phone", p.phone),
                ("Date of Admission", _fmt_date(adm), "Date of Discharge", _fmt_date(spec.summary_discharge_date)),
                ("Treating Doctor", spec.doctor, "Length of Stay", f"{spec.stay_days} day(s)"),
            ]
        )
    )

    story += [Paragraph("Diagnosis", HEAD), Paragraph(f"<b>{spec.diagnosis_name}</b> &nbsp; (ICD-10: {spec.icd10})", BODY)]
    if spec.ped_conditions:
        story.append(Paragraph(f"Associated / past history: {', '.join(spec.ped_conditions)}", BODY))
    story += [Paragraph("Procedure Performed", HEAD), Paragraph(spec.procedure or "Conservative medical management", BODY)]
    story += [Paragraph("History of Presenting Illness", HEAD), Paragraph(narrative.history, BODY)]
    story += [Paragraph("Course in Hospital", HEAD), Paragraph(narrative.course_in_hospital, BODY)]
    story += [Paragraph("Condition at Discharge", HEAD), Paragraph("Stable, afebrile, tolerating oral diet, ambulant.", BODY)]
    story.append(Paragraph("Discharge Advice", HEAD))
    for item in narrative.discharge_advice:
        story.append(Paragraph(f"&bull; {item}", BODY))
    story += [Spacer(1, 24), Paragraph(f"{spec.doctor}<br/>Consultant, {spec.hospital.name}", BODY)]
    story += [Spacer(1, 10), Paragraph("Synthetic document for software testing only.", SMALL)]
    doc.build(story)
