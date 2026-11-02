"""Render structured CV data into realistic-looking documents.

Templates
- classic_pdf : centred header, ruled section headings, right-aligned dates (typical ATS-friendly CV)
- modern_pdf  : coloured header band, accent headings (typical designer template)
- docx        : Word CV with styled headings, bottom borders and right tab-stop dates
- txt         : plain-text CV (recruiters still receive these via job portals)
"""
from __future__ import annotations

from pathlib import Path

import docx
from docx.enum.text import WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from fpdf import FPDF

NAVY = (31, 58, 95)
ACCENT = (42, 120, 214)
GREY = (90, 90, 90)


def _latin1(s: str) -> str:
    """Core PDF fonts are Latin-1 only; map common typography to safe equivalents."""
    table = {"–": "-", "—": "-", "‘": "'", "’": "'", "“": '"', "”": '"', "•": "-", "…": "..."}
    for k, v in table.items():
        s = s.replace(k, v)
    return s.encode("latin-1", "replace").decode("latin-1")


# ---------------------------------------------------------------- plain text
def cv_to_text(cv: dict) -> str:
    out = [cv["name"], cv["title"], " | ".join([cv["location"], cv["email"], cv["phone"], *cv["links"]]), ""]
    out += ["PROFESSIONAL SUMMARY", cv["summary"], ""]
    out += ["SKILLS"] + [f"{k}: {v}" for k, v in cv["skills"].items()] + [""]
    out += ["PROFESSIONAL EXPERIENCE"]
    for e in cv["experience"]:
        out.append(f"{e['title']} | {e['company']}, {e['location']} | {e['start']} - {e['end']}")
        out += [f"- {b}" for b in e["bullets"]]
        out.append("")
    if cv["projects"]:
        out += ["PROJECTS"] + [f"- {n}: {d}" for n, d in cv["projects"]] + [""]
    out += ["EDUCATION"] + [f"{d}, {i} | {y}" + (f" | {g}" if g else "") for d, i, y, g in cv["education"]] + [""]
    if cv["certifications"]:
        out += ["CERTIFICATIONS"] + [f"- {c}" for c in cv["certifications"]] + [""]
    if cv["achievements"]:
        out += ["ACHIEVEMENTS"] + [f"- {a}" for a in cv["achievements"]] + [""]
    return "\n".join(out).strip() + "\n"


def write_txt(cv: dict, path: Path) -> None:
    path.write_text(cv_to_text(cv), encoding="utf-8")


# ---------------------------------------------------------------- PDF helpers
class _CV(FPDF):
    def bullet(self, text: str, indent: float = 4) -> None:
        y = self.get_y()
        self.set_fill_color(60, 60, 60)
        self.circle(x=self.l_margin + indent - 2.2, y=y + 1.6, radius=0.6, style="F")
        self.set_x(self.l_margin + indent)
        self.multi_cell(0, 4.6, _latin1(text), new_x="LMARGIN", new_y="NEXT")


def _role_row(pdf: _CV, left: str, right: str) -> None:
    pdf.set_font("Helvetica", "B", 10.5)
    pdf.set_text_color(20, 20, 20)
    width = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.cell(width * 0.72, 5.5, _latin1(left))
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(*GREY)
    pdf.cell(width * 0.28, 5.5, _latin1(right), align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(20, 20, 20)


def _body(pdf: _CV, cv: dict, heading) -> None:
    heading("Professional Summary")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 4.8, _latin1(cv["summary"]), new_x="LMARGIN", new_y="NEXT")

    heading("Skills")
    for k, v in cv["skills"].items():
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 4.8, _latin1(f"**{k}:** {v}"), markdown=True, new_x="LMARGIN", new_y="NEXT")

    heading("Professional Experience")
    for e in cv["experience"]:
        _role_row(pdf, f"{e['title']} | {e['company']}, {e['location']}", f"{e['start']} - {e['end']}")
        pdf.set_font("Helvetica", "", 10)
        for b in e["bullets"]:
            pdf.bullet(b)
        pdf.ln(1.5)

    if cv["projects"]:
        heading("Projects")
        pdf.set_font("Helvetica", "", 10)
        for n, d in cv["projects"]:
            pdf.bullet(f"{n}: {d}")

    heading("Education")
    for d, i, y, g in cv["education"]:
        _role_row(pdf, f"{d}, {i}", y)
        if g:
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 4.8, _latin1(g), new_x="LMARGIN", new_y="NEXT")

    if cv["certifications"]:
        heading("Certifications")
        pdf.set_font("Helvetica", "", 10)
        for c in cv["certifications"]:
            pdf.bullet(c)
    if cv["achievements"]:
        heading("Achievements")
        pdf.set_font("Helvetica", "", 10)
        for a in cv["achievements"]:
            pdf.bullet(a)


def write_classic_pdf(cv: dict, path: Path) -> None:
    pdf = _CV(format="A4")
    pdf.set_margins(16, 14, 16)
    pdf.set_auto_page_break(True, margin=14)
    pdf.add_page()
    pdf.set_title(f"{cv['name']} - CV")
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 9, _latin1(cv["name"]), align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11.5)
    pdf.set_text_color(*ACCENT)
    pdf.cell(0, 6, _latin1(cv["title"]), align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*GREY)
    pdf.cell(0, 5, _latin1(" | ".join([cv["location"], cv["email"], cv["phone"], *cv["links"]])), align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(20, 20, 20)

    def heading(t: str) -> None:
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(*NAVY)
        pdf.cell(0, 6, t.upper(), new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(*NAVY)
        pdf.set_line_width(0.4)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
        pdf.ln(1.8)
        pdf.set_text_color(20, 20, 20)

    _body(pdf, cv, heading)
    pdf.output(str(path))


def write_modern_pdf(cv: dict, path: Path) -> None:
    pdf = _CV(format="A4")
    pdf.set_margins(18, 12, 18)
    pdf.set_auto_page_break(True, margin=14)
    pdf.add_page()
    pdf.set_title(f"{cv['name']} - Resume")
    pdf.set_fill_color(*NAVY)
    pdf.rect(0, 0, pdf.w, 38, style="F")
    pdf.set_xy(18, 9)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 22)
    pdf.cell(0, 10, _latin1(cv["name"]), new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(18)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(200, 220, 245)
    pdf.cell(0, 6, _latin1(cv["title"]), new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(18)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, _latin1(" | ".join([cv["location"], cv["email"], cv["phone"], *cv["links"]])), new_x="LMARGIN", new_y="NEXT")
    pdf.set_y(44)
    pdf.set_text_color(20, 20, 20)

    def heading(t: str) -> None:
        pdf.ln(3)
        y = pdf.get_y()
        pdf.set_fill_color(*ACCENT)
        pdf.rect(pdf.l_margin, y + 1, 1.6, 4.5, style="F")
        pdf.set_x(pdf.l_margin + 3.5)
        pdf.set_font("Helvetica", "B", 11.5)
        pdf.set_text_color(*ACCENT)
        pdf.cell(0, 6.5, t.upper(), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        pdf.set_text_color(20, 20, 20)

    _body(pdf, cv, heading)
    pdf.output(str(path))


# ---------------------------------------------------------------- DOCX
def _bottom_border(paragraph, color: str = "1F3A5F") -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    borders = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    for k, v in {"w:val": "single", "w:sz": "8", "w:space": "1", "w:color": color}.items():
        bottom.set(qn(k), v)
    borders.append(bottom)
    p_pr.append(borders)


def write_docx(cv: dict, path: Path) -> None:
    d = docx.Document()
    sec = d.sections[0]
    sec.left_margin = sec.right_margin = Inches(0.75)
    sec.top_margin = sec.bottom_margin = Inches(0.6)
    normal = d.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(2)
    d.core_properties.title = f"{cv['name']} - CV"
    d.core_properties.author = cv["name"]

    p = d.add_paragraph()
    r = p.add_run(cv["name"]); r.bold = True; r.font.size = Pt(24); r.font.color.rgb = RGBColor(*NAVY)
    p = d.add_paragraph()
    r = p.add_run(cv["title"]); r.font.size = Pt(12.5); r.font.color.rgb = RGBColor(*ACCENT)
    p = d.add_paragraph()
    r = p.add_run(" | ".join([cv["location"], cv["email"], cv["phone"], *cv["links"]])); r.font.size = Pt(9); r.font.color.rgb = RGBColor(*GREY)

    right_tab = sec.page_width - sec.left_margin - sec.right_margin

    def heading(t: str) -> None:
        hp = d.add_paragraph()
        hp.paragraph_format.space_before = Pt(10)
        hp.paragraph_format.space_after = Pt(4)
        hr = hp.add_run(t.upper()); hr.bold = True; hr.font.size = Pt(11.5); hr.font.color.rgb = RGBColor(*NAVY)
        _bottom_border(hp)

    def role(left: str, right: str) -> None:
        rp = d.add_paragraph()
        rp.paragraph_format.tab_stops.add_tab_stop(right_tab, WD_TAB_ALIGNMENT.RIGHT)
        rp.paragraph_format.space_before = Pt(4)
        lr = rp.add_run(left); lr.bold = True
        rr = rp.add_run("\t" + right); rr.font.color.rgb = RGBColor(*GREY); rr.font.size = Pt(9.5)

    heading("Professional Summary")
    d.add_paragraph(cv["summary"])
    heading("Skills")
    for k, v in cv["skills"].items():
        sp = d.add_paragraph()
        sp.add_run(f"{k}: ").bold = True
        sp.add_run(v)
    heading("Professional Experience")
    for e in cv["experience"]:
        role(f"{e['title']} | {e['company']}, {e['location']}", f"{e['start']} - {e['end']}")
        for b in e["bullets"]:
            d.add_paragraph(b, style="List Bullet")
    if cv["projects"]:
        heading("Projects")
        for n, desc in cv["projects"]:
            bp = d.add_paragraph(style="List Bullet")
            bp.add_run(f"{n}: ").bold = True
            bp.add_run(desc)
    heading("Education")
    for deg, inst, years, grade in cv["education"]:
        role(f"{deg}, {inst}", years)
        if grade:
            d.add_paragraph(grade)
    if cv["certifications"]:
        heading("Certifications")
        for c in cv["certifications"]:
            d.add_paragraph(c, style="List Bullet")
    if cv["achievements"]:
        heading("Achievements")
        for a in cv["achievements"]:
            d.add_paragraph(a, style="List Bullet")
    d.save(path)


RENDERERS = {
    "classic_pdf": (".pdf", write_classic_pdf),
    "modern_pdf": (".pdf", write_modern_pdf),
    "docx": (".docx", write_docx),
    "txt": (".txt", write_txt),
}
