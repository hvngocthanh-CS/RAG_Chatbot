"""
Generate realistic TechViet corporate PDF documents for RAG chatbot testing.
All table cells use Paragraph objects to ensure proper text wrapping.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "techviet_docs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Page dimensions ---
PAGE_W, PAGE_H = A4  # 595.27 x 841.89
MARGIN_L, MARGIN_R = 42, 42
MARGIN_T, MARGIN_B = 55, 50
USABLE_W = PAGE_W - MARGIN_L - MARGIN_R  # ~511pt

# --- Colors ---
BRAND = colors.HexColor('#1a5276')
BRAND_LIGHT = colors.HexColor('#2e86c1')
HEADER_BG = colors.HexColor('#1a5276')
ROW_ALT = colors.HexColor('#eaf2f8')
GRID_COLOR = colors.HexColor('#d5dbdb')
TEXT_DARK = colors.HexColor('#2c3e50')
TEXT_MED = colors.HexColor('#5d6d7e')

# --- Styles ---
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='DocTitle', parent=styles['Title'], fontSize=20, spaceAfter=6,
                          textColor=BRAND, leading=24))
styles.add(ParagraphStyle(name='DocSubtitle', parent=styles['Normal'], fontSize=10, spaceAfter=4,
                          textColor=TEXT_MED, alignment=TA_CENTER, leading=13))
styles.add(ParagraphStyle(name='SectionHead', parent=styles['Heading1'], fontSize=14, spaceAfter=8,
                          spaceBefore=18, textColor=BRAND, leading=17))
styles.add(ParagraphStyle(name='SubHead', parent=styles['Heading2'], fontSize=11, spaceAfter=6,
                          spaceBefore=12, textColor=BRAND_LIGHT, leading=14))
styles.add(ParagraphStyle(name='Body', parent=styles['Normal'], fontSize=9, leading=13,
                          spaceAfter=6, alignment=TA_JUSTIFY, textColor=TEXT_DARK))
styles.add(ParagraphStyle(name='BodyBold', parent=styles['Normal'], fontSize=9, leading=13,
                          spaceAfter=6, textColor=TEXT_DARK, fontName='Helvetica-Bold'))
styles.add(ParagraphStyle(name='CellText', parent=styles['Normal'], fontSize=8, leading=11,
                          textColor=TEXT_DARK))
styles.add(ParagraphStyle(name='CellHeader', parent=styles['Normal'], fontSize=8, leading=11,
                          textColor=colors.white, fontName='Helvetica-Bold'))
styles.add(ParagraphStyle(name='CellBold', parent=styles['Normal'], fontSize=8, leading=11,
                          textColor=TEXT_DARK, fontName='Helvetica-Bold'))
styles.add(ParagraphStyle(name='TVBullet', parent=styles['Normal'], fontSize=9, leading=13,
                          leftIndent=18, bulletIndent=6, spaceAfter=3, textColor=TEXT_DARK))
styles.add(ParagraphStyle(name='Note', parent=styles['Normal'], fontSize=8, leading=11,
                          textColor=TEXT_MED, fontStyle='italic', spaceAfter=4))

P = lambda t, s='CellText': Paragraph(str(t), styles[s])
PH = lambda t: Paragraph(str(t), styles['CellHeader'])
PB = lambda t: Paragraph(str(t), styles['CellBold'])


def header_footer(canvas, doc):
    canvas.saveState()
    # Top bar
    canvas.setFillColor(BRAND)
    canvas.rect(0, PAGE_H - 28, PAGE_W, 28, fill=1, stroke=0)
    canvas.setFont('Helvetica-Bold', 8)
    canvas.setFillColor(colors.white)
    canvas.drawString(MARGIN_L, PAGE_H - 20, "TECHVIET SOLUTIONS JSC")
    canvas.setFont('Helvetica', 7)
    canvas.drawRightString(PAGE_W - MARGIN_R, PAGE_H - 20, "CONFIDENTIAL — INTERNAL USE ONLY")
    # Bottom
    canvas.setFont('Helvetica', 7)
    canvas.setFillColor(colors.grey)
    canvas.drawString(MARGIN_L, 22, f"Page {doc.page}")
    canvas.drawRightString(PAGE_W - MARGIN_R, 22, "\u00a9 2026 TechViet Solutions JSC — All Rights Reserved")
    canvas.setStrokeColor(GRID_COLOR)
    canvas.line(MARGIN_L, 35, PAGE_W - MARGIN_R, 35)
    canvas.restoreState()


def build_pdf(filename, story):
    path = os.path.join(OUTPUT_DIR, filename)
    doc = SimpleDocTemplate(path, pagesize=A4,
                            topMargin=MARGIN_T, bottomMargin=MARGIN_B,
                            leftMargin=MARGIN_L, rightMargin=MARGIN_R)
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(f"  Created: {filename}")


def make_table(headers, rows, col_widths=None):
    """Build a table with proper wrapping. headers & rows are lists of strings.
    All cells are wrapped in Paragraph objects."""
    # Convert to Paragraph objects
    header_row = [PH(h) for h in headers]
    body_rows = []
    for row in rows:
        body_rows.append([P(cell) for cell in row])

    data = [header_row] + body_rows

    # Auto-calculate widths if not provided, distributing evenly
    if col_widths is None:
        n = len(headers)
        col_widths = [USABLE_W / n] * n
    else:
        # Scale widths to fit USABLE_W
        total = sum(col_widths)
        if total > USABLE_W:
            scale = USABLE_W / total
            col_widths = [w * scale for w in col_widths]

    style_cmds = [
        ('BACKGROUND', (0, 0), (-1, 0), HEADER_BG),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.4, GRID_COLOR),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style_cmds.append(('BACKGROUND', (0, i), (-1, i), ROW_ALT))

    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle(style_cmds))
    return t


def title_page(title, subtitle, dept, version_info=None):
    """Generate a consistent title block for each document."""
    s = []
    s.append(Spacer(1, 10))
    s.append(Paragraph(title, styles['DocTitle']))
    s.append(Paragraph(subtitle, styles['DocSubtitle']))
    s.append(Paragraph(dept, styles['DocSubtitle']))
    if version_info:
        s.append(Paragraph(version_info, styles['DocSubtitle']))
    s.append(Spacer(1, 14))
    return s


def bullet(text):
    return Paragraph(f"\u2022  {text}", styles['TVBullet'])


# ============================================================
# DOCUMENT 1 — Employee Handbook (comprehensive)
# ============================================================
def doc_employee_handbook():
    s = title_page(
        "Employee Handbook",
        "Comprehensive Guide to Policies, Benefits, and Workplace Standards",
        "Human Resources Department",
        "Version 3.2 | Effective January 1, 2026 | Approved by CEO & CHRO"
    )

    # --- 1. Company Overview ---
    s.append(Paragraph("1. Company Overview", styles['SectionHead']))
    s.append(Paragraph(
        "TechViet Solutions Joint Stock Company (\"TechViet\") was founded in 2018 and is headquartered in "
        "District 7, Ho Chi Minh City, Vietnam. We are a technology company specializing in enterprise "
        "software development, cloud infrastructure consulting, artificial intelligence solutions, and data "
        "engineering services. As of January 2026, TechViet employs 458 full-time staff and 32 long-term "
        "contractors across three offices in Ho Chi Minh City (headquarters, 320 staff), Hanoi (95 staff), "
        "and Da Nang (75 staff).", styles['Body']))
    s.append(Paragraph(
        "Our client portfolio includes 45 active enterprise accounts spanning banking and financial services "
        "(VietBank, FinServe VN, MekongPay), healthcare (SaigonHealth, MediCare Plus), logistics and supply "
        "chain (VNLogistics, TransViet), retail and e-commerce (MegaMart, ShopNow), energy (PetroVN, "
        "GreenEnergy Corp), and government (GovTech VN, Ministry of Digital Transformation). Our annual "
        "revenue for FY2025 was approximately USD 18.2 million, representing 28% year-over-year growth.", styles['Body']))

    s.append(Paragraph("1.1 Mission, Vision, and Core Values", styles['SubHead']))
    s.append(Paragraph(
        "Our mission is to empower businesses across Southeast Asia through innovative, reliable, and "
        "scalable technology solutions. Our vision is to become the leading technology partner in the ASEAN "
        "region by 2030, recognized for technical excellence, client success, and engineering culture.", styles['Body']))
    s.append(make_table(
        ["Core Value", "What It Means in Practice"],
        [
            ["Innovation First", "We dedicate 15% of engineering time to experimentation. Every team runs at least one innovation sprint per quarter. Failed experiments are celebrated if they produce learning."],
            ["Client Success", "We measure ourselves by client outcomes, not deliverables. Account health reviews happen monthly. Every engineer spends at least one day per quarter shadowing client operations."],
            ["Integrity & Transparency", "We share bad news early. Project status reports are honest, not optimistic. We never misrepresent capabilities to win deals. All code reviews are frank and constructive."],
            ["Collaboration", "Cross-functional squads are our default org structure. Knowledge hoarding is a performance issue. We document decisions and share context proactively."],
            ["Continuous Learning", "Every employee has a VND 15M annual learning budget. We run internal tech talks weekly. Senior engineers are expected to mentor at least one junior colleague."],
        ],
        [130, USABLE_W - 130]
    ))
    s.append(Spacer(1, 6))

    s.append(Paragraph("1.2 Organizational Structure", styles['SubHead']))
    s.append(make_table(
        ["Department", "Head", "Headcount", "Key Responsibilities"],
        [
            ["Engineering", "Le Van Hung (VP)", "255", "Product development, platform engineering, DevOps, QA"],
            ["AI & Data Science", "Dr. Tran Minh Khoa", "52", "ML models, data pipelines, AI product features, research"],
            ["Product Management", "Nguyen Thi Hanh", "18", "Roadmaps, requirements, client discovery, prioritization"],
            ["Design (UI/UX)", "Do Thanh Tam", "18", "User research, interaction design, design systems"],
            ["Sales & Partnerships", "Hoang Duc Anh", "25", "Business development, account management, proposals"],
            ["People & Culture (HR)", "Nguyen Thi Mai", "22", "Recruitment, L&D, compensation, employee experience"],
            ["Finance & Operations", "Vo Minh Tuan", "20", "Accounting, procurement, legal, office management"],
            ["IT & Security", "Le Hoang Nam (CISO)", "28", "Infrastructure, security operations, compliance, IT support"],
            ["Executive Office", "Dr. Pham Quang Minh (CEO)", "5", "Strategy, governance, investor relations"],
        ],
        [90, 120, 55, USABLE_W - 265]
    ))
    s.append(Spacer(1, 8))

    # --- 2. Employment Policies ---
    s.append(Paragraph("2. Employment Policies", styles['SectionHead']))
    s.append(Paragraph("2.1 Working Hours and Flexibility", styles['SubHead']))
    s.append(Paragraph(
        "Standard working hours are Monday through Friday, 8:30 AM to 5:30 PM (ICT), with a one-hour "
        "lunch break from 12:00 PM to 1:00 PM. TechViet observes a 40-hour work week in compliance with "
        "the Vietnamese Labor Code 2019 (Law No. 45/2019/QH14). Overtime must be pre-approved by the "
        "department head and is compensated at the rates mandated by law (150% for weekdays, 200% for "
        "weekends, 300% for public holidays).", styles['Body']))
    s.append(Paragraph(
        "TechViet offers a hybrid work model for eligible positions. Remote work is permitted up to 2 days "
        "per week, subject to team agreements and manager approval. During remote days, employees must be "
        "reachable during core hours (10:00 AM — 4:00 PM ICT) via Slack and Google Meet. Certain roles "
        "(office management, hardware IT support, reception) are not eligible for remote work. Teams in "
        "active client-facing delivery phases may temporarily require full in-office presence with 2 weeks' "
        "notice.", styles['Body']))
    s.append(Paragraph(
        "A 4-day work week pilot program ran from January to March 2026 with 3 volunteer teams (42 "
        "employees). Preliminary results showed maintained productivity with improved employee satisfaction "
        "scores. HR is analyzing full results for presentation at the April all-hands meeting.", styles['Body']))

    s.append(Paragraph("2.2 Leave Entitlements", styles['SubHead']))
    s.append(make_table(
        ["Leave Type", "Entitlement", "Carry Over", "Conditions & Notes"],
        [
            ["Annual Leave", "15 days/year", "Max 5 days to following year", "Accrued monthly (1.25 days/month). Usable after 6-month probation. For employees with 5+ years tenure: 16 days. 10+ years: 18 days."],
            ["Sick Leave", "10 days/year (paid)", "No carry over", "Medical certificate from licensed physician required for absences of 3+ consecutive days. First 2 days may be self-certified."],
            ["Personal Leave", "3 days/year", "No carry over", "For personal emergencies, family events (weddings, funerals), or urgent household matters. Manager notification required by 9 AM."],
            ["Maternity Leave", "6 months (180 days)", "N/A", "Per Vietnamese Labor Code Article 139. Full salary paid by social insurance. Additional 1 month for multiple births."],
            ["Paternity Leave", "7 days (paid)", "N/A", "Must be taken within 30 days of child's birth. Applies to legal father or same-sex partner (TechViet policy, exceeding legal minimum)."],
            ["Bereavement Leave", "3-5 days (paid)", "N/A", "3 days for immediate family (parents, spouse, children). 5 days if travel to another province is required. 1 day for extended family."],
            ["Study/Exam Leave", "5 days/year", "No carry over", "For approved professional certifications and examinations. Must submit certification plan to manager 30 days in advance."],
            ["Marriage Leave", "3 days (paid)", "N/A", "For the employee's own wedding. 1 day for child's wedding. Per Labor Code Article 115."],
            ["Unpaid Leave", "Up to 30 days/year", "N/A", "Requires VP approval. Benefits continue for first 15 days. Extended unpaid leave (30+ days) requires CHRO approval."],
        ],
        [70, 70, 60, USABLE_W - 200]
    ))
    s.append(Spacer(1, 6))
    s.append(Paragraph(
        "Leave requests must be submitted through the TechViet HR Portal (hr.techviet.vn) at least 3 "
        "business days in advance for planned absences. For urgent/emergency leave, employees must notify "
        "their direct supervisor by phone or Slack before 9:00 AM on the first day of absence and submit "
        "the formal request within 24 hours. Unapproved absences will be treated as unauthorized leave and "
        "may affect performance evaluations.", styles['Body']))

    s.append(Paragraph("2.3 Public Holidays (2026)", styles['SubHead']))
    s.append(make_table(
        ["Holiday", "Date(s) in 2026", "Days Off", "Notes"],
        [
            ["New Year's Day", "January 1 (Thursday)", "1", "If falls on weekend, following Monday is off"],
            ["Lunar New Year (Tet)", "January 26 — January 30", "5", "Office closed Jan 24 — Feb 2 (including weekends). Exact dates follow government announcement."],
            ["Hung Kings' Commemoration", "April 6 (Monday)", "1", "10th day of 3rd lunar month"],
            ["Reunification Day", "April 30 (Thursday)", "1", "Bridge day policy: if Thu/Tue, company may grant Fri/Mon off (announced annually)"],
            ["International Labour Day", "May 1 (Friday)", "1", "Combined with Reunification Day for potential 4-day weekend"],
            ["National Day", "September 2-3 (Wed-Thu)", "2", "Per Decree 145/2024: Sep 2 + 1 additional day"],
        ],
        [110, 120, 40, USABLE_W - 270]
    ))

    s.append(PageBreak())

    # --- 3. Compensation & Benefits ---
    s.append(Paragraph("3. Compensation and Benefits", styles['SectionHead']))
    s.append(Paragraph("3.1 Salary Structure and Career Ladder", styles['SubHead']))
    s.append(Paragraph(
        "TechViet operates a market-competitive salary structure benchmarked annually against the top 25th "
        "percentile of the Vietnamese IT industry using data from Mercer, Robert Walters, and TopDev salary "
        "surveys. Salaries are reviewed during the annual Performance Review Cycle (December) with "
        "adjustments effective January 1. All salaries are paid on the last business day of each month via "
        "bank transfer to the employee's registered account.", styles['Body']))
    s.append(make_table(
        ["Level", "Title Examples", "Monthly Gross (VND)", "Typical Experience", "Annual Bonus Target"],
        [
            ["L1", "Junior Engineer, Associate Designer, Analyst", "15,000,000 — 25,000,000", "0-2 years", "1.0 month salary"],
            ["L2", "Mid-level Engineer, Specialist, Senior Analyst", "25,000,000 — 40,000,000", "2-5 years", "1.5 months salary"],
            ["L3", "Senior Engineer, Lead Designer, Senior PM", "40,000,000 — 60,000,000", "5-8 years", "2.0 months salary"],
            ["L4", "Principal Engineer, Engineering Manager, Director", "60,000,000 — 85,000,000", "8-12 years", "2.5 months salary"],
            ["L5", "Distinguished Engineer, VP, Senior Director", "85,000,000 — 130,000,000", "12+ years", "3.0 months salary"],
            ["L6", "CTO, CEO, C-Suite Executives", "130,000,000+", "Per contract", "Per executive agreement"],
        ],
        [28, 140, 100, 80, USABLE_W - 348]
    ))
    s.append(Spacer(1, 4))
    s.append(Paragraph(
        "The 13th-month salary (Tet bonus) is mandatory under Vietnamese law and paid in full in the "
        "January payroll. The performance-based bonus above is in addition to the 13th-month salary. "
        "Bonus payouts are tied to both individual performance ratings and company financial performance, "
        "with a company multiplier ranging from 0.8x to 1.2x applied to individual bonus targets.", styles['Body']))

    s.append(Paragraph("3.2 Benefits Package", styles['SubHead']))
    s.append(make_table(
        ["Benefit", "Details", "Eligibility"],
        [
            ["Health Insurance", "Premium plan (PVI Insurance) covering employee + up to 2 dependents. Includes inpatient, outpatient, and emergency coverage up to VND 500M/year. Dental (VND 10M/year) and vision (VND 5M/year) included for L3+.", "All FTE after probation"],
            ["Life Insurance", "Coverage equal to 24x monthly gross salary. Accidental death & dismemberment: 48x monthly salary.", "All FTE"],
            ["Lunch Allowance", "VND 50,000/working day loaded onto meal card (Sodexo). Valid at 200+ partner restaurants near each office.", "All FTE"],
            ["Transportation", "VND 1,000,000/month parking or commuting allowance. Electric vehicle charging available at HCM office (free).", "All FTE"],
            ["Learning & Development", "VND 15,000,000/year for courses, conferences, certifications, and books. Pre-approval required for amounts >VND 5M. Conference travel covered separately per travel policy.", "All FTE"],
            ["Gym & Wellness", "50% subsidy on gym membership at partner fitness centers (California Fitness, Elite Fitness). Annual health checkup at premium clinic (fully covered).", "All FTE"],
            ["Employee Stock Option Plan", "ESOP available for L3+ employees after 2 years of continuous service. 4-year vesting with 1-year cliff. Pool size: 5% of total equity.", "L3+ after 2 years"],
            ["Equipment", "Company-issued laptop (MacBook Pro 14\" or Dell XPS 15, refreshed every 3 years). External monitor (27\" 4K) and ergonomic accessories provided. WFH setup allowance: VND 5,000,000 one-time.", "All FTE"],
            ["Relocation Assistance", "VND 20,000,000 relocation package for employees transferring between offices. Includes 1 month temporary housing.", "Inter-office transfers"],
            ["Referral Bonus", "VND 10,000,000 for successful engineering referrals (paid after 3-month probation of referred employee). VND 5,000,000 for non-engineering roles.", "All employees"],
        ],
        [80, USABLE_W - 160, 80]
    ))

    s.append(PageBreak())

    # --- 4. Performance Management ---
    s.append(Paragraph("4. Performance Management", styles['SectionHead']))
    s.append(Paragraph(
        "TechViet uses a combination of OKRs (Objectives and Key Results) for goal-setting and a calibrated "
        "performance review process for evaluation. The system is designed to be fair, transparent, and "
        "growth-oriented. Performance data directly influences compensation adjustments, bonus payouts, "
        "promotion decisions, and learning & development investments.", styles['Body']))

    s.append(Paragraph("4.1 Performance Cycle", styles['SubHead']))
    s.append(make_table(
        ["Phase", "Timing", "Activities", "Participants"],
        [
            ["Goal Setting", "January (Q1) and July (Q3)", "Set 3-5 OKRs aligned with team and company goals. Include at least 1 growth/learning objective.", "Employee + Manager"],
            ["Continuous Feedback", "Ongoing (minimum monthly)", "1-on-1 meetings, peer feedback via Culture Amp, project retrospectives. Managers document notable contributions and areas for growth.", "Employee + Manager + Peers"],
            ["Mid-Year Check-in", "June", "Formal progress review. Adjust OKRs if business priorities shifted. Identify blockers and support needed. Not rated.", "Employee + Manager"],
            ["Annual Review", "December", "Comprehensive evaluation against OKRs. Self-assessment, manager assessment, and 360 peer feedback (minimum 3 peers). Performance rating assigned.", "Employee + Manager + Peers + HR"],
            ["Calibration", "Late December", "Department heads review all ratings to ensure consistency and reduce bias. HR facilitates cross-department calibration.", "Department Heads + HR"],
            ["Outcome Communication", "January", "Ratings, compensation adjustments, and development plans communicated in 1-on-1. Appeal process available.", "Employee + Manager"],
        ],
        [75, 85, USABLE_W - 235, 75]
    ))
    s.append(Spacer(1, 6))

    s.append(Paragraph("4.2 Performance Rating Scale", styles['SubHead']))
    s.append(make_table(
        ["Rating", "Label", "Description", "Salary Adj.", "Bonus Mult.", "Distribution Guide"],
        [
            ["5", "Exceptional", "Consistently exceeds all expectations. Recognized impact beyond own team. Role model for others.", "12-18%", "1.5x", "~5% of employees"],
            ["4", "Exceeds Expectations", "Frequently delivers above expectations. Takes initiative on complex challenges. Strong peer feedback.", "8-12%", "1.25x", "~20% of employees"],
            ["3", "Meets Expectations", "Reliably delivers quality work. Meets all OKRs. Solid contributor to team goals.", "5-8%", "1.0x", "~55% of employees"],
            ["2", "Developing", "Partially meets expectations. Specific areas need improvement. Performance Improvement Plan (PIP) if consecutive.", "0-3%", "0.5x", "~15% of employees"],
            ["1", "Below Expectations", "Consistently fails to meet core expectations. Immediate PIP required. If no improvement in 90 days, may lead to separation.", "0%", "0x", "~5% of employees"],
        ],
        [30, 65, USABLE_W - 270, 45, 45, 85]
    ))
    s.append(Spacer(1, 6))
    s.append(Paragraph(
        "Distribution guides are targets, not quotas. In exceptional years, distributions may skew higher. "
        "However, managers must justify deviations greater than 10 percentage points from the guide. All "
        "rating 1 and rating 5 decisions require VP-level approval and HR review.", styles['Body']))

    # --- 5. Code of Conduct ---
    s.append(Paragraph("5. Code of Conduct", styles['SectionHead']))
    s.append(Paragraph(
        "All employees, contractors, and temporary staff are expected to uphold TechViet's ethical standards "
        "and professional conduct at all times, whether in the office, at client sites, at company events, "
        "or in any context where they represent TechViet.", styles['Body']))

    s.append(Paragraph("5.1 Anti-Harassment and Inclusion", styles['SubHead']))
    s.append(Paragraph(
        "TechViet maintains a zero-tolerance policy toward harassment, discrimination, bullying, and "
        "retaliation in any form — including verbal, physical, visual, and digital. This applies regardless "
        "of the individuals' positions or seniority. Protected categories include but are not limited to: "
        "gender, gender identity, sexual orientation, ethnicity, religion, age, disability, marital status, "
        "and pregnancy status.", styles['Body']))
    s.append(Paragraph(
        "Reports can be made to your manager, HR Business Partner, or anonymously via the Ethics Hotline "
        "(ethics@techviet.vn or ext. 999). All reports are investigated within 5 business days. Retaliation "
        "against anyone who reports a concern in good faith is itself a terminable offense.", styles['Body']))

    s.append(Paragraph("5.2 Confidentiality and Intellectual Property", styles['SubHead']))
    s.append(Paragraph(
        "All employees sign a Non-Disclosure Agreement (NDA) and an Intellectual Property Assignment "
        "Agreement upon joining. The NDA covers all confidential information including client data, trade "
        "secrets, business strategies, unreleased products, and internal financial information. The NDA "
        "remains in effect for 24 months after separation from TechViet.", styles['Body']))
    s.append(Paragraph(
        "Any invention, design, code, or creative work produced during employment and related to TechViet's "
        "business is the property of TechViet. Personal open-source contributions are encouraged but must "
        "not use TechViet proprietary code or client data. Employees must disclose any potential conflicts "
        "of interest, including consulting arrangements, board memberships, or significant financial "
        "interests in competitors or clients.", styles['Body']))

    s.append(Paragraph("5.3 Disciplinary Process", styles['SubHead']))
    s.append(make_table(
        ["Step", "Action", "Duration", "Documentation"],
        [
            ["1. Verbal Warning", "Manager discusses concern with employee. Clear expectations set.", "Immediate", "Manager's notes (shared with employee)"],
            ["2. Written Warning", "Formal letter detailing the issue, expected improvement, and timeline. Copy to HR.", "30-day review period", "HR file; signed acknowledgment"],
            ["3. Final Warning / PIP", "Performance Improvement Plan with specific, measurable goals. Weekly check-ins.", "60-90 days", "PIP document; weekly progress notes"],
            ["4. Termination", "If no improvement after PIP, employment may be terminated per Labor Code procedures.", "Per legal requirements", "Full documentation package; HR review"],
        ],
        [75, USABLE_W - 235, 65, 95]
    ))
    s.append(Spacer(1, 4))
    s.append(Paragraph(
        "Gross misconduct (theft, fraud, violence, data breach, substance abuse at work) may result in "
        "immediate termination without progression through the above steps, subject to investigation and "
        "Labor Code requirements.", styles['Body']))

    build_pdf("01_Employee_Handbook.pdf", s)


# ============================================================
# DOCUMENT 2 — IT Security Policy
# ============================================================
def doc_it_security():
    s = title_page(
        "Information Security Policy",
        "Comprehensive Security Framework for Systems, Data, and Personnel",
        "IT Security & Compliance Department",
        "Document ID: TVSC-SEC-001 | Version 2.5 | Classification: Internal"
    )

    s.append(Paragraph("1. Purpose and Scope", styles['SectionHead']))
    s.append(Paragraph(
        "This policy establishes the information security requirements for all personnel who access TechViet "
        "systems, networks, or data — including full-time employees, part-time staff, contractors, interns, "
        "and third-party vendors. It covers all TechViet-owned and managed information assets regardless of "
        "location: office premises, remote work environments, client sites, and cloud infrastructure. "
        "Compliance with this policy is mandatory and is audited quarterly by the internal security team and "
        "annually by an external auditor (currently Deloitte Vietnam).", styles['Body']))
    s.append(Paragraph(
        "This policy is aligned with ISO 27001:2022, the Vietnam Cybersecurity Law (2018), and the "
        "CIS Controls v8 framework. TechViet achieved ISO 27001 certification in March 2024 and maintains "
        "SOC 2 Type II compliance for client-facing services.", styles['Body']))

    s.append(Paragraph("2. Data Classification Framework", styles['SectionHead']))
    s.append(Paragraph(
        "All data created, processed, stored, or transmitted by TechViet must be classified by the data "
        "owner according to the following framework. Misclassification or failure to classify is a policy "
        "violation. When in doubt, apply the higher classification level.", styles['Body']))
    s.append(make_table(
        ["Level", "Label", "Examples", "Storage", "Sharing", "Disposal"],
        [
            ["C4", "Top Secret", "Encryption keys, pen-test results, M&A plans, executive financial models", "Hardware-encrypted vault; dedicated server", "Need-to-know; CISO approval; no copies", "Cryptographic erasure; physical destruction for media"],
            ["C3", "Confidential", "Client source code, PII, salary data, financial reports, security configs", "Encrypted at rest (AES-256) and in transit (TLS 1.3)", "VP approval required; encrypted transfer only; logged", "Secure deletion (3-pass overwrite); certificate of destruction"],
            ["C2", "Internal", "Internal emails, project plans, meeting notes, architecture docs, Jira tickets", "Standard access controls; company-managed storage only", "Internal only; do not forward to personal email", "Standard deletion; no special procedures"],
            ["C1", "Public", "Marketing materials, blog posts, open-source contributions, job postings", "No restrictions", "May be shared freely", "No restrictions"],
        ],
        [28, 50, 110, 100, 110, USABLE_W - 398]
    ))
    s.append(Spacer(1, 6))

    s.append(Paragraph("3. Access Control", styles['SectionHead']))
    s.append(Paragraph("3.1 Authentication Standards", styles['SubHead']))
    s.append(Paragraph(
        "TechViet enforces a defense-in-depth approach to authentication. All systems accessible from "
        "the internet require multi-factor authentication (MFA). Single sign-on (SSO) via Google Workspace "
        "SAML is the standard authentication method for all SaaS applications.", styles['Body']))
    s.append(make_table(
        ["Requirement", "Standard", "Enforcement Mechanism"],
        [
            ["Minimum Password Length", "14 characters", "Okta password policy; rejects shorter passwords at creation"],
            ["Password Complexity", "Must include: uppercase, lowercase, digit, and special character (!@#$%^&*)", "Okta policy; real-time feedback during password creation"],
            ["Password Rotation", "Every 90 days; last 12 passwords cannot be reused", "Okta policy; automated expiry notification at 14, 7, and 1 day(s)"],
            ["Account Lockout", "Locked after 5 consecutive failed attempts", "Auto-unlock after 30 min; IT can manually unlock with identity verification"],
            ["MFA Method", "Primary: Hardware key (YubiKey 5). Fallback: TOTP authenticator app (Google/Authy)", "Okta MFA policy; SMS and phone call MFA are explicitly disabled"],
            ["Session Timeout", "Web applications: 8 hours idle. VPN: 12 hours idle. Admin consoles: 30 min idle.", "Application-level config; enforced via Okta session policies"],
            ["Service Accounts", "No interactive login. API key rotation every 30 days. Scoped to minimum required permissions.", "Vault-managed secrets; automated rotation via HashiCorp Vault"],
        ],
        [100, 165, USABLE_W - 265]
    ))
    s.append(Spacer(1, 6))

    s.append(Paragraph("3.2 Role-Based Access Control (RBAC)", styles['SubHead']))
    s.append(Paragraph(
        "Access to systems and data is granted based on the principle of least privilege. Access rights are "
        "tied to job roles, not individuals. When an employee changes roles, their access is reviewed and "
        "adjusted within 2 business days. Quarterly access reviews are conducted by each department head.", styles['Body']))
    s.append(make_table(
        ["Role", "Systems Access", "Data Classification Access", "Approval Required", "Review Cycle"],
        [
            ["Standard Employee", "Email, Slack, Google Drive, Confluence, HR Portal", "C1, C2 only", "Auto-provisioned at onboarding", "Annual"],
            ["Developer", "Above + GitHub, Jira, staging environments, CI/CD, Sentry", "C1, C2, C3 (own project scope only)", "Team Lead approval", "Quarterly"],
            ["DevOps/SRE", "Above + AWS Console, Terraform Cloud, ArgoCD, production monitoring (read-only)", "C1, C2, C3", "Engineering Manager + IT Security", "Monthly"],
            ["Database Admin", "Above + production database access (write), backup systems, Vault admin", "C1, C2, C3, C4", "CTO approval", "Monthly"],
            ["Security Team", "All systems including SIEM, WAF, vulnerability scanners, forensics tools", "All classifications", "CISO approval", "Monthly"],
            ["Executive", "Business analytics, financial systems, board materials, all Confluence spaces", "C1, C2, C3, C4 (business scope)", "Auto-provisioned per role", "Quarterly"],
            ["Contractor", "Scoped to specific project. No access to internal HR, finance, or other project data.", "C1, C2, C3 (project scope only)", "Project Manager + IT Security", "Per contract renewal"],
        ],
        [65, 125, 80, 100, USABLE_W - 370]
    ))
    s.append(Spacer(1, 6))

    s.append(Paragraph("4. Network Security", styles['SectionHead']))
    s.append(Paragraph(
        "All TechViet office networks are segmented into isolated zones to contain potential breaches. "
        "Inter-zone traffic is filtered by next-generation firewalls (Palo Alto PA-3260 series). All "
        "outbound web traffic is inspected by Zscaler Internet Access (ZIA) for malware, DLP, and "
        "unauthorized data exfiltration. DNS resolution uses Cloudflare Gateway with threat intelligence "
        "feeds enabled.", styles['Body']))
    s.append(make_table(
        ["Network Zone", "Purpose", "Access Policy", "Monitoring"],
        [
            ["Corporate LAN (VLAN 10)", "Employee workstations, printers, meeting room equipment", "NAC (802.1X); domain-joined devices only", "Datadog Network Monitoring; 90-day NetFlow retention"],
            ["Development VLAN (VLAN 20)", "Dev servers, container hosts, build agents", "Developer role required; MFA for SSH access", "Full packet capture retained 7 days; flow data 90 days"],
            ["Server DMZ (VLAN 30)", "Production-facing services, load balancers, reverse proxies", "IP whitelist + WAF (Cloudflare); no direct internet egress", "24/7 SOC monitoring; all traffic logged to SIEM"],
            ["Guest Wi-Fi (VLAN 99)", "Visitors, personal devices", "Internet only; completely isolated from corporate network; captive portal", "Basic logging; 30-day retention"],
            ["Management VLAN (VLAN 5)", "Network equipment, IPMI/iDRAC, security appliances", "IT Security team only; hardware MFA + VPN required", "All access logged; alerts on any unauthorized attempt"],
        ],
        [85, 110, 150, USABLE_W - 345]
    ))
    s.append(Spacer(1, 6))

    s.append(Paragraph("5. Endpoint Security", styles['SectionHead']))
    s.append(make_table(
        ["Control", "Requirement", "Enforcement", "Compliance Check"],
        [
            ["Endpoint Detection (EDR)", "CrowdStrike Falcon on all endpoints. Real-time protection enabled. Cannot be disabled by user.", "Jamf (macOS) / Intune (Windows); auto-remediation if agent stops", "Daily compliance scan; non-compliant devices lose network access in 24 hours"],
            ["Disk Encryption", "FileVault 2 (macOS) or BitLocker (Windows) with AES-256. Recovery keys escrowed in Jamf/Intune.", "MDM-enforced at enrollment; cannot be deferred", "Weekly audit; HR notified for persistent non-compliance"],
            ["OS Patching", "Critical/high CVEs: within 72 hours. Medium: within 14 days. Low: next maintenance window.", "WSUS (Windows) / Jamf (macOS); staged rollout with 10% canary", "Weekly compliance dashboard; monthly report to CISO"],
            ["Screen Lock", "Auto-lock after 5 minutes of inactivity. Lock on lid close.", "MDM policy; user cannot override timeout", "Enforced at MDM level; verified during quarterly audits"],
            ["USB/External Devices", "Blocked by default. Exceptions require IT Security ticket with business justification. Approved for max 30 days.", "CrowdStrike Device Control; whitelist by serial number", "Monthly audit of active exceptions"],
            ["Browser Extensions", "Only approved extensions from TechViet Software Center. Unapproved extensions auto-removed within 24 hours.", "Chrome Enterprise policy via Google Admin", "Weekly scan; report to IT Security"],
        ],
        [75, 130, 145, USABLE_W - 350]
    ))

    s.append(PageBreak())
    s.append(Paragraph("6. Incident Response Framework", styles['SectionHead']))
    s.append(Paragraph(
        "All security incidents must be reported to the Security Operations Center (SOC) within 1 hour "
        "of discovery. Reporting channels: #security-incidents on Slack (preferred), email to soc@techviet.vn, "
        "or phone extension 911 (after hours). The SOC operates 24/7 with a team of 4 analysts across "
        "2 shifts.", styles['Body']))
    s.append(make_table(
        ["Severity", "Definition", "Examples", "Response SLA", "Escalation Path"],
        [
            ["P1 — Critical", "Active breach, data exfiltration, or complete service compromise", "Ransomware infection, confirmed data breach, compromised admin credentials", "15 minutes to triage; 1 hour to contain", "CISO + CEO within 30 min; Legal within 1 hour; external IR firm on standby"],
            ["P2 — High", "Suspected breach, active exploitation attempt, or significant vulnerability", "Malware detected on endpoints, unauthorized access to C3+ data, zero-day in production stack", "1 hour to triage; 4 hours to contain", "CISO within 2 hours; affected team leads immediately"],
            ["P3 — Medium", "Policy violation, phishing attempt, or low-impact vulnerability", "Employee clicked phishing link (no credentials entered), unpatched non-critical system, minor policy deviation", "4 hours to triage; 24 hours to remediate", "Security Manager next business day"],
            ["P4 — Low", "False positive, informational finding, or minor anomaly", "Legitimate software flagged by EDR, benign port scan from known scanner, cosmetic security header missing", "24 hours to triage; best-effort resolution", "SOC Analyst; no escalation required"],
        ],
        [55, 90, 115, 85, USABLE_W - 345]
    ))
    s.append(Spacer(1, 8))

    s.append(Paragraph("7. Data Backup and Disaster Recovery", styles['SectionHead']))
    s.append(make_table(
        ["System / Data", "Backup Strategy", "Frequency", "Retention", "RTO", "RPO", "Test Frequency"],
        [
            ["Production Databases (PostgreSQL)", "Continuous WAL archiving + daily pg_dump", "WAL: real-time; Full: daily 2AM", "90 days (daily); 1 year (monthly)", "< 1 hour", "< 5 min", "Monthly restore drill"],
            ["Source Code (GitHub Enterprise)", "Real-time replication to secondary AWS region (ap-southeast-3)", "Real-time", "Indefinite (Git history)", "< 15 min", "0 (real-time)", "Quarterly failover test"],
            ["Google Workspace (Email/Drive)", "Daily incremental via Veeam Backup for Google", "Daily at 1AM", "1 year", "< 4 hours", "< 24 hours", "Quarterly restore test"],
            ["Kubernetes Configs & Secrets", "Velero backup to S3; Vault auto-backup", "Every 6 hours", "180 days", "< 30 min", "< 6 hours", "Monthly restore drill"],
            ["CI/CD Pipelines (GitHub Actions)", "Infrastructure-as-Code in Git; secrets in Vault", "Real-time (Git)", "Indefinite", "< 30 min", "0 (Git)", "After every infra change"],
            ["Monitoring Data (Datadog)", "N/A (SaaS retention)", "Continuous", "15 months (Datadog plan)", "N/A", "N/A", "N/A"],
        ],
        [80, 90, 70, 75, 40, 45, USABLE_W - 400]
    ))
    s.append(Spacer(1, 6))
    s.append(Paragraph(
        "Disaster recovery procedures are documented in the TechViet Business Continuity Plan (TVSC-BCP-001). "
        "The DR site is AWS ap-southeast-3 (Jakarta). Full DR drills are conducted semi-annually, with the "
        "last successful drill completed on November 14, 2025 (RTO achieved: 47 minutes, target: 60 minutes).", styles['Body']))

    build_pdf("02_IT_Security_Policy.pdf", s)


# ============================================================
# DOCUMENT 3 — Onboarding Guide
# ============================================================
def doc_onboarding():
    s = title_page(
        "New Employee Onboarding Guide",
        "Your Comprehensive Guide to the First 90 Days at TechViet",
        "People & Culture Team",
        "Version 2.3 | Last Updated: February 2026"
    )

    s.append(Paragraph(
        "Welcome to TechViet! We are thrilled to have you join our team. This guide contains everything "
        "you need for a successful start. Your onboarding buddy and direct manager are your primary "
        "contacts — never hesitate to ask questions. We believe the first 90 days are critical for your "
        "long-term success, and we've designed this program to set you up for it.", styles['Body']))

    s.append(Paragraph("1. Pre-Start Preparation", styles['SectionHead']))
    s.append(Paragraph(
        "The following tasks should be completed before your first day. HR and IT will reach out to you "
        "via your personal email to coordinate. If any items are incomplete by your start date, notify "
        "your HR Business Partner immediately.", styles['Body']))
    s.append(make_table(
        ["Task", "Owner", "Deadline", "Instructions"],
        [
            ["Sign employment contract and NDA", "HR", "5 days before start", "Digital signature via DocuSign. Review carefully; ask HR for clarification on any clause."],
            ["Submit bank details for payroll", "New Hire", "3 days before start", "Form sent to personal email. Must be a Vietnamese bank account in your name."],
            ["Complete background check", "HR (vendor)", "Before start", "You'll receive an email from our vendor (First Advantage). Respond within 48 hours."],
            ["Prepare laptop and peripherals", "IT", "1 day before start", "Laptop pre-configured with standard software. Monitor and accessories at your desk."],
            ["Create corporate accounts", "IT", "1 day before start", "Google Workspace, Slack, Jira, GitHub. Credentials sent to personal email."],
            ["Assign onboarding buddy", "Manager", "Before start", "Your buddy is a team member who joined 6-12 months ago. They'll reach out to introduce themselves."],
            ["Set up desk and welcome kit", "Office Mgmt", "1 day before start", "Welcome kit includes: notebook, branded merchandise, office map, emergency contacts card."],
        ],
        [110, 60, 70, USABLE_W - 240]
    ))
    s.append(Spacer(1, 8))

    s.append(Paragraph("2. Day 1 Detailed Schedule", styles['SectionHead']))
    s.append(make_table(
        ["Time", "Activity", "Location / Link", "Contact Person", "What to Bring"],
        [
            ["8:30", "Arrive at reception. Security will issue your temporary badge. Your buddy meets you at lobby.", "Ground floor lobby, main entrance", "Buddy / Reception", "Government ID"],
            ["9:00", "HR welcome session: company overview, org chart, culture, policies summary, Q&A", "Meeting Room A3, 3rd Floor (or Zoom link for remote)", "HR Business Partner", "Nothing — materials provided"],
            ["9:45", "IT setup: laptop handoff, account activation, VPN config, MFA enrollment (bring phone for authenticator)", "IT Help Desk, 2nd Floor", "IT Support (Tran Van Duc)", "Personal phone for MFA setup"],
            ["10:30", "Office tour: floors, emergency exits, kitchen, gym, prayer room, mother's room, bike storage", "All floors", "Office Manager", "Comfortable shoes"],
            ["11:00", "Team introduction: meet your direct manager and team members, see the team board, learn about current sprint", "Team area", "Direct Manager", "Curiosity and questions!"],
            ["12:00", "Welcome lunch with team — on the company. Your buddy will guide you to the restaurant.", "Partner restaurant, ground floor", "Buddy", "Nothing"],
            ["13:30", "Company strategy and product overview presentation by CEO or COO", "Auditorium, 5th Floor", "CEO or COO", "Notebook for notes"],
            ["14:30", "Benefits enrollment: health insurance, meal card, transportation, emergency contacts, tax forms", "HR Office, 3rd Floor", "HR Coordinator", "Dependent info (for insurance)"],
            ["15:30", "Mandatory security awareness training (1 hour, interactive)", "Training Room B2 / Zoom", "Security Team", "Laptop"],
            ["16:30", "Development environment setup (Engineering) or tool-specific training (Non-engineering)", "Your desk", "Buddy / Team Lead", "Laptop"],
            ["17:00", "Day 1 debrief with manager: first impressions, answer questions, plan for rest of week", "Manager's desk or 1-on-1 room", "Direct Manager", "Your questions!"],
        ],
        [32, 140, 100, 80, USABLE_W - 352]
    ))

    s.append(PageBreak())
    s.append(Paragraph("3. Week 1 Mandatory Training Program", styles['SectionHead']))
    s.append(Paragraph(
        "All training modules must be completed by the end of your first week. Training is tracked via "
        "the TechViet LMS (learn.techviet.vn). Your manager and HR will be notified if any modules are "
        "overdue. Engineering-specific training is in addition to the general modules.", styles['Body']))
    s.append(make_table(
        ["Module", "Duration", "Platform", "Day", "Assessment", "Passing Score"],
        [
            ["Information Security Fundamentals", "2 hours", "LMS (self-paced)", "Day 2", "Quiz (20 questions)", "80%"],
            ["Code of Conduct & Business Ethics", "1 hour", "LMS (self-paced)", "Day 2", "Acknowledgment + Quiz", "100% ack, 80% quiz"],
            ["Anti-Harassment, Diversity & Inclusion", "1.5 hours", "LMS (video + scenarios)", "Day 3", "Scenario-based quiz", "80%"],
            ["Data Privacy & Compliance (PDPD/GDPR)", "1 hour", "LMS (self-paced)", "Day 3", "Quiz + certification", "85%"],
            ["TechViet Product & Service Overview", "1.5 hours", "Live session (recorded)", "Day 4", "None (informational)", "N/A"],
            ["Git Workflow & Code Review (Eng only)", "2 hours", "Live workshop", "Day 4", "Submit practice PR", "PR approved by mentor"],
            ["CI/CD & Deployment (Eng only)", "1.5 hours", "Live workshop", "Day 4", "Deploy to dev env", "Successful deploy"],
            ["Architecture & Tech Stack (Eng only)", "2 hours", "Live session + self-study", "Day 5", "Architecture quiz", "75%"],
            ["Client Communication Standards", "1 hour", "LMS (self-paced)", "Day 5", "Scenario quiz", "80%"],
        ],
        [110, 45, 80, 35, 105, USABLE_W - 375]
    ))
    s.append(Spacer(1, 8))

    s.append(Paragraph("4. 30-60-90 Day Milestone Framework", styles['SectionHead']))
    s.append(Paragraph(
        "Your onboarding success is measured against the milestones below. Your manager will review progress "
        "at the 30, 60, and 90-day marks. These conversations are documented and shared with HR. Successful "
        "completion of the 90-day milestone marks the end of your probation period.", styles['Body']))
    s.append(make_table(
        ["Period", "Engineering Track", "Non-Engineering Track", "Evaluation Method"],
        [
            ["Day 1-30: Learn", "Complete dev environment and all training. Read team's top 10 architecture docs. Merge first PR (bug fix or small feature). Shadow 3+ code reviews. Understand team's on-call rotation.", "Complete all training modules. Shadow 3+ client meetings or stakeholder sessions. Deliver first small assignment. Learn team's tools and workflows end-to-end.", "30-day check-in with manager. Self-assessment form. Buddy feedback (informal)."],
            ["Day 31-60: Contribute", "Own a feature or module end-to-end (design, implement, test, deploy). Participate actively in sprint planning and retros. Conduct at least 2 code reviews. Give a 15-min tech talk to team.", "Own a workstream or deliverable independently. Lead or co-lead a meeting with stakeholders. Contribute meaningfully to team OKRs. Present work at team meeting.", "60-day check-in with manager. Peer feedback from 2 colleagues. Review of deliverables."],
            ["Day 61-90: Own", "Deliver feature with minimal guidance. Begin mentoring newer members. Propose at least 1 process or technical improvement. Participate in on-call rotation. Complete probation review.", "Manage deliverables fully independently. Drive cross-team collaboration on at least 1 initiative. Propose process improvement. Complete probation review.", "90-day formal review (manager + HR). 360 feedback from 3+ peers. Probation decision communicated."],
        ],
        [55, 150, 150, USABLE_W - 355]
    ))
    s.append(Spacer(1, 8))

    s.append(Paragraph("5. Key Tools, Systems, and Access", styles['SectionHead']))
    s.append(make_table(
        ["Tool", "Purpose", "URL", "Login Method", "Help Channel"],
        [
            ["Google Workspace", "Email, calendar, documents, video calls", "mail.techviet.vn", "SSO (primary identity)", "#general"],
            ["Slack", "Real-time messaging, team channels, incident response", "techviet.slack.com", "SSO via Google", "#it-helpdesk"],
            ["Jira", "Sprint planning, issue tracking, project management", "jira.techviet.vn", "SSO", "#eng-support"],
            ["Confluence", "Documentation wiki, knowledge base, runbooks", "wiki.techviet.vn", "SSO", "#eng-support"],
            ["GitHub Enterprise", "Source code, pull requests, CI/CD, code review", "github.techviet.vn", "SSO + MFA (YubiKey)", "#eng-support"],
            ["AWS Console", "Cloud infrastructure (Engineering/DevOps only)", "aws.techviet.vn", "SSO + MFA + VPN required", "#devops"],
            ["HR Portal", "Leave requests, payslips, benefits, personal info", "hr.techviet.vn", "SSO", "#hr-support"],
            ["TechViet LMS", "Training modules, certifications, learning paths", "learn.techviet.vn", "SSO", "#learning"],
            ["1Password Teams", "Password manager (shared vaults for team credentials)", "Desktop app", "SSO + Master password", "#it-helpdesk"],
            ["Datadog", "Application monitoring, logs, dashboards (Eng only)", "app.datadoghq.com", "SSO + VPN for some dashboards", "#observability"],
        ],
        [72, 115, 78, 90, USABLE_W - 355]
    ))
    s.append(Spacer(1, 6))

    s.append(Paragraph("6. Key Contacts", styles['SectionHead']))
    s.append(make_table(
        ["Department", "Contact Person", "Email", "Slack", "Office Hours"],
        [
            ["HR / People & Culture", "Nguyen Thi Mai (CHRO)", "mai.nguyen@techviet.vn", "#hr-support", "Mon-Fri 9-5"],
            ["IT Help Desk", "Tran Van Duc (IT Manager)", "it-support@techviet.vn", "#it-helpdesk", "Mon-Fri 8-6"],
            ["Security Team (SOC)", "Le Hoang Nam (CISO)", "soc@techviet.vn", "#security-incidents", "24/7"],
            ["Office Management", "Pham Thanh Hoa", "office@techviet.vn", "#office-hcm", "Mon-Fri 8:30-5:30"],
            ["Finance / Payroll", "Vo Minh Tuan (CFO)", "payroll@techviet.vn", "#finance-support", "Mon-Fri 9-5"],
            ["Learning & Development", "Dang Thu Huong", "learning@techviet.vn", "#learning", "Mon-Fri 9-5"],
        ],
        [85, 100, 110, 90, USABLE_W - 385]
    ))

    build_pdf("03_Onboarding_Guide.pdf", s)


# ============================================================
# DOCUMENT 4 — Engineering Standards & Guidelines
# ============================================================
def doc_engineering_standards():
    s = title_page(
        "Engineering Standards & Guidelines",
        "Technical Standards, Code Quality, API Design, and Deployment Practices",
        "Engineering Department",
        "Document ID: TVSC-ENG-003 | Version 4.1 | Last Updated: February 2026"
    )

    s.append(Paragraph("1. Approved Technology Stack", styles['SectionHead']))
    s.append(Paragraph(
        "TechViet maintains a curated set of approved technologies to ensure security, maintainability, and "
        "knowledge sharing across teams. Using technologies outside this list requires an Architecture "
        "Decision Record (ADR) approved by the Architecture Review Board (ARB), which meets bi-weekly. "
        "The ARB consists of the CTO, VP Engineering, Principal Engineers, and the CISO.", styles['Body']))
    s.append(make_table(
        ["Layer", "Primary Choice", "Approved Alternatives", "Decision Rationale"],
        [
            ["Frontend Web", "React 18 + TypeScript 5.x", "Next.js 14 (SSR needs), Vue 3 (legacy projects)", "React ecosystem maturity; TypeScript mandatory for all new frontend code since 2024"],
            ["Backend API", "Python 3.12 + FastAPI 0.110+", "Go 1.22 (high-perf services), Node.js 20 (BFF)", "FastAPI for ML-heavy services and rapid prototyping; Go for latency-critical microservices"],
            ["Mobile", "React Native 0.73+", "Flutter 3.19+ (approved for new projects Q1 2026)", "React Native for code sharing with web; Flutter approved after successful ShopSmart pilot"],
            ["Relational DB", "PostgreSQL 16", "MySQL 8.0 (client requirement only)", "PostgreSQL for JSON support, extensions (pgvector, PostGIS), and performance at scale"],
            ["NoSQL", "MongoDB 7.0", "Redis 7.2 (cache/queue), DynamoDB (serverless)", "MongoDB for document workloads; DynamoDB only for pure serverless architectures on AWS"],
            ["Message Queue", "Apache Kafka 3.7", "RabbitMQ 3.13 (simple task queues)", "Kafka for event streaming and audit logs; RabbitMQ for simple producer-consumer patterns"],
            ["Search Engine", "Elasticsearch 8.12", "OpenSearch 2.12 (AWS-native)", "Elasticsearch default; OpenSearch when client requires AWS-native and avoids Elastic licensing"],
            ["Vector Database", "Qdrant 1.8 (self-hosted)", "Pinecone (prototyping), pgvector (small datasets <100K vectors)", "Qdrant selected after 4-week PoC (see ADR-001); best cost-performance for our workloads"],
            ["Containers", "Docker + Kubernetes (EKS 1.29)", "ECS Fargate (simple services)", "K8s for all production workloads; ECS only for isolated scheduled tasks"],
            ["CI/CD", "GitHub Actions", "None (Jenkins being decommissioned Q2 2026)", "Standardization decision (see ADR-003); shared workflow templates in techviet/ci-templates"],
            ["IaC", "Terraform 1.7+ (OpenTofu compatible)", "AWS CDK (complex AWS-native infra)", "Terraform default for multi-cloud compatibility; CDK approved for pure-AWS projects"],
            ["Monitoring", "Datadog (APM + Logs + Infra)", "Prometheus + Grafana (on-prem client deployments)", "Datadog for SaaS simplicity; Prometheus stack when data residency prevents SaaS"],
            ["LLM Provider", "Anthropic Claude 3.5 Sonnet (primary)", "OpenAI GPT-4o (fallback), local Llama 3 (offline/edge)", "Claude for accuracy and safety; GPT-4o as redundancy; Llama for data-sovereign clients"],
        ],
        [60, 100, 120, USABLE_W - 280]
    ))

    s.append(PageBreak())
    s.append(Paragraph("2. Code Quality Standards", styles['SectionHead']))
    s.append(Paragraph("2.1 Pull Request and Code Review Requirements", styles['SubHead']))
    s.append(Paragraph(
        "Code review is one of our most important quality practices. Every change to a shared codebase "
        "must go through a pull request, regardless of the author's seniority. The following requirements "
        "apply to all repositories in the TechViet GitHub organization.", styles['Body']))
    s.append(make_table(
        ["Requirement", "Standard", "Rationale"],
        [
            ["Minimum Reviewers", "2 approvals for main/develop branches; 1 approval for feature branches", "Two-person review catches more issues; single review OK for WIP branches to reduce friction"],
            ["Review Turnaround", "First review within 4 business hours; final approval within 24 hours", "Fast feedback loops prevent context-switching costs; measured and reported in weekly eng metrics"],
            ["Maximum PR Size", "400 lines changed (excluding auto-generated files, tests, migrations)", "Smaller PRs get better reviews. PRs >400 lines must be split or get explicit EM exception."],
            ["PR Description", "Must include: What changed, Why (link to Jira ticket), How to test, Rollback plan", "Structured descriptions enable async review and serve as documentation for future developers"],
            ["CI Checks Required", "Linting (ruff/eslint), unit tests, integration tests, type check, security scan (Snyk)", "Automated checks must all pass before merge. Manual override requires 2 reviewer approvals."],
            ["Branch Naming", "Format: {type}/{ticket}-{description}. Types: feat, fix, refactor, chore, docs, test", "Consistent naming enables automated changelog generation and branch cleanup scripts"],
            ["Commit Messages", "Conventional Commits format. 72-char subject line. Body explains why, not what.", "Enables automated release notes; enforced by commitlint in pre-commit hook"],
            ["Stale PR Policy", "PRs with no activity for 7 days are flagged. 14 days: auto-closed with comment.", "Prevents PR backlog buildup; author is notified and can reopen with /reopen command"],
        ],
        [80, 170, USABLE_W - 250]
    ))
    s.append(Spacer(1, 6))

    s.append(Paragraph("2.2 Testing Standards", styles['SubHead']))
    s.append(make_table(
        ["Test Type", "Coverage / Target", "When Executed", "Owner", "Tooling"],
        [
            ["Unit Tests", "≥80% line coverage (enforced by CI); ≥90% for core business logic modules", "Every commit via GitHub Actions", "Feature developer", "pytest (Python), Jest (JS/TS), go test (Go)"],
            ["Integration Tests", "All API endpoints; all database operations; all external service integrations", "Every PR; nightly full suite", "Feature developer + QA", "pytest + testcontainers; Playwright for API"],
            ["End-to-End (E2E)", "All critical user journeys (top 20 flows); top 5 error/edge-case scenarios", "Nightly build; pre-release gate", "QA Team", "Playwright (migrating from Selenium; 60% done)"],
            ["Performance Tests", "P95 latency within SLA; no regressions >10% vs. baseline; memory leak detection", "Weekly automated; pre-release mandatory", "DevOps + QA", "k6 (load); Vegeta (HTTP); custom profilers"],
            ["Security Scans (SAST)", "Zero critical/high findings; medium findings triaged within 5 business days", "Every PR (Snyk); weekly full scan (Veracode)", "Security Team + Developer", "Snyk (PR), Veracode (weekly), CodeQL (GitHub)"],
            ["Security Scans (DAST)", "Zero critical findings on production-facing endpoints", "Weekly against staging environment", "Security Team", "OWASP ZAP; Burp Suite (manual quarterly)"],
            ["Chaos Engineering", "Production services survive: single node failure, network partition, dependency timeout", "Monthly automated; quarterly game day", "SRE Team", "Litmus Chaos; custom failure injection"],
        ],
        [70, 105, 80, 70, USABLE_W - 325]
    ))

    s.append(PageBreak())
    s.append(Paragraph("3. API Design Standards", styles['SectionHead']))
    s.append(Paragraph(
        "All public and internal APIs must follow the TechViet API Design Guide. REST is the default "
        "protocol for client-facing APIs. gRPC is approved for internal service-to-service communication "
        "where low latency is critical. GraphQL is approved only for Backend-for-Frontend (BFF) layers.", styles['Body']))
    s.append(make_table(
        ["Standard", "Requirement", "Example / Notes"],
        [
            ["Versioning", "URL path versioning: /api/v1/resource. No breaking changes within a major version. Deprecation notice 6 months before sunset.", "/api/v1/users, /api/v2/users. Header versioning explicitly not used."],
            ["Authentication", "OAuth 2.0 + JWT (RS256) for external APIs. mTLS for internal service mesh (Istio). API keys for webhooks (HMAC-SHA256 signed).", "JWT expiry: 15 min access, 7 day refresh. Rotate signing keys quarterly."],
            ["Rate Limiting", "Default: 100 req/min per client (token bucket). Configurable per endpoint and client tier. Rate limit headers in every response.", "X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset. 429 response with Retry-After header."],
            ["Pagination", "Cursor-based for list endpoints (offset-based deprecated). Max page size: 100 items. Default: 20 items.", "?cursor=abc123&limit=20. Response includes next_cursor and has_more fields."],
            ["Error Format", "RFC 7807 Problem Details JSON. Include type, title, status, detail, instance, and correlation_id.", "{\"type\": \"/errors/not-found\", \"title\": \"User not found\", \"status\": 404, \"correlation_id\": \"abc-123\"}"],
            ["Idempotency", "All POST/PUT/PATCH endpoints must support Idempotency-Key header. Keys valid for 24 hours.", "Client generates UUID as idempotency key. Server returns cached response for duplicate keys."],
            ["Timeouts", "Client timeout: 30s. Server processing: 60s. Long-running operations: return 202 Accepted + poll endpoint.", "Async operations return {\"status_url\": \"/api/v1/jobs/123\"} for polling."],
            ["Documentation", "OpenAPI 3.1 spec required. All endpoints documented with examples. Spec auto-generated from code annotations.", "Swagger UI at /docs (dev/staging only). Redoc at /api-docs (all environments)."],
        ],
        [65, 170, USABLE_W - 235]
    ))
    s.append(Spacer(1, 8))

    s.append(Paragraph("4. Deployment and Release Standards", styles['SectionHead']))
    s.append(make_table(
        ["Environment", "Branch", "Deploy Method", "Approval Gate", "Rollback Method", "Monitoring"],
        [
            ["Development", "feature/* branches", "Auto-deploy on push (Argo CD)", "None (automated)", "Redeploy previous commit", "Basic logging (stdout)"],
            ["Staging", "develop branch", "Auto-deploy on merge to develop", "1 code reviewer", "Argo CD rollback (1-click)", "Full Datadog APM + logs"],
            ["UAT", "release/* branches", "Manual trigger (GitHub Actions)", "QA Lead + Product Owner", "Argo CD rollback", "Full monitoring + Slack alerts"],
            ["Production", "main branch (tag-based)", "Blue/green via Argo CD Rollouts", "2 code reviewers + QA sign-off + EM approval", "Automated rollback if error rate >1% in 10 min", "Full monitoring + PagerDuty + status page"],
        ],
        [60, 75, 85, 100, 85, USABLE_W - 405]
    ))
    s.append(Spacer(1, 4))
    s.append(Paragraph(
        "All production deployments must include a documented rollback plan. Canary deployments (10% traffic "
        "for 15 minutes) are required for changes affecting >10% of users. Feature flags via LaunchDarkly "
        "are mandatory for all new user-facing features — the flag must be in 'off' state at deploy time "
        "and enabled gradually after deploy verification.", styles['Body']))
    s.append(Paragraph(
        "Deploy windows: Production deploys are permitted Monday through Thursday, 9 AM — 4 PM ICT. No "
        "production deploys on Fridays, weekends, public holidays, or during client-designated freeze periods. "
        "Emergency hotfixes exempt from deploy window restrictions but require VP Engineering approval.", styles['Body']))

    build_pdf("04_Engineering_Standards.pdf", s)


# ============================================================
# DOCUMENT 5 — Quarterly Project Status Report
# ============================================================
def doc_project_status():
    s = title_page(
        "Quarterly Project Status Report",
        "Q1 2026 (January — March) | Prepared by Project Management Office",
        "PMO & Delivery Management",
        "Distribution: Executive Committee, Department Heads, Account Managers"
    )

    s.append(Paragraph("1. Executive Summary", styles['SectionHead']))
    s.append(Paragraph(
        "This report provides a comprehensive overview of TechViet's project portfolio for Q1 2026. The "
        "company manages 12 active projects across 4 business units with a combined contract value of "
        "USD 5.1 million. Overall delivery performance stands at 87% on-time delivery (OTD), an improvement "
        "from 82% in Q4 2025. Revenue recognition is on track at 94% of quarterly forecast.", styles['Body']))
    s.append(Paragraph(
        "Three projects are flagged at-risk: ShopSmart Reco Engine (RED — significant budget overrun due to "
        "uncontrolled scope changes), MediTrack v2 (YELLOW — integration delays with legacy EMR system), and "
        "HR Analytics (YELLOW — data quality issues and model accuracy below target). Detailed mitigation "
        "plans are provided in Section 3. All other projects are on track (GREEN).", styles['Body']))

    s.append(Paragraph("2. Project Portfolio Dashboard", styles['SectionHead']))
    s.append(make_table(
        ["Project", "Client", "Type", "Status", "Contract (USD)", "Spent (USD)", "% Budget", "Timeline", "Health"],
        [
            ["Phoenix Platform", "VietBank", "Core Banking", "Active", "850,000", "520,000", "61%", "Jan-Sep '26", "GREEN"],
            ["MediTrack v2", "SaigonHealth", "HealthTech", "Active", "420,000", "310,000", "74%", "Oct '25-Jun '26", "YELLOW"],
            ["LogiFlow AI", "VNLogistics", "AI/ML", "Active", "650,000", "180,000", "28%", "Feb-Nov '26", "GREEN"],
            ["ShopSmart Reco", "MegaMart", "AI/ML", "Active", "280,000", "250,000", "89%", "Sep '25-Apr '26", "RED"],
            ["CloudMigrate", "SteelCorp VN", "Cloud", "Active", "1,200,000", "400,000", "33%", "Jan-Dec '26", "GREEN"],
            ["DataLake 360", "PetroVN", "Data Eng", "Planning", "550,000", "25,000", "5%", "Apr-Oct '26", "GREEN"],
            ["ChatAssist Bot", "Internal", "AI/ML", "Active", "180,000", "95,000", "53%", "Nov '25-May '26", "GREEN"],
            ["HR Analytics", "Internal", "Analytics", "Active", "120,000", "88,000", "73%", "Oct '25-Mar '26", "YELLOW"],
            ["SecureGate 2.0", "GovTech VN", "Security", "UAT", "380,000", "360,000", "95%", "Jul '25-Mar '26", "GREEN"],
            ["PayFlow Integ.", "FinServe VN", "FinTech", "Active", "320,000", "145,000", "45%", "Dec '25-Jul '26", "GREEN"],
            ["GreenDash", "GreenEnergy", "IoT/Data", "Active", "200,000", "60,000", "30%", "Jan-Aug '26", "GREEN"],
            ["EduPortal", "Ministry DT", "GovTech", "Planning", "410,000", "15,000", "4%", "Apr-Dec '26", "GREEN"],
        ],
        [65, 55, 42, 35, 50, 50, 30, 60, USABLE_W - 387]
    ))
    s.append(Spacer(1, 8))

    s.append(Paragraph("3. At-Risk Project Analysis", styles['SectionHead']))

    s.append(Paragraph("3.1 ShopSmart Recommendation Engine — RED", styles['SubHead']))
    s.append(Paragraph(
        "The ShopSmart recommendation engine project has consumed 89% of its budget with only approximately "
        "60% of features delivered. The root cause is three major scope changes requested by MegaMart after "
        "sprint 4 — including a real-time personalization engine that was not in the original SOW. These "
        "scope additions were accepted by the previous project manager (who has since left TechViet) without "
        "corresponding budget adjustments or formal change orders.", styles['Body']))
    s.append(make_table(
        ["Risk Factor", "Impact", "Probability", "Mitigation Action", "Owner", "Deadline"],
        [
            ["Budget overrun", "High — estimated USD 45K additional cost to complete", "Confirmed", "Scope renegotiation meeting with MegaMart on March 15. Propose descoping real-time features to Phase 2 with separate SOW.", "Account Manager + PM", "Mar 15"],
            ["Resource shortage", "Medium — team needs 2 additional ML engineers", "High", "Request 2 ML engineers from AI team for 8-week rotation. If unavailable, evaluate contractor options.", "VP Engineering", "Mar 10"],
            ["Timeline slip", "High — currently 4 weeks behind revised schedule", "Confirmed", "Revised timeline to be agreed with client. If descoping accepted, original deadline achievable.", "PM", "Mar 20"],
            ["Client satisfaction", "Medium — client frustrated with pace and communication", "Medium", "Weekly executive-level status calls. CTO to join next client meeting to rebuild confidence.", "CTO + AM", "Mar 8"],
        ],
        [65, 95, 50, 140, 65, USABLE_W - 415]
    ))
    s.append(Spacer(1, 6))

    s.append(Paragraph("3.2 MediTrack v2 — YELLOW", styles['SubHead']))
    s.append(Paragraph(
        "MediTrack v2 is experiencing delays in the HL7 FHIR integration module due to incomplete and "
        "inconsistent API documentation from SaigonHealth's legacy EMR system (built by a vendor who is "
        "no longer contracted). Our integration team has had to reverse-engineer several endpoints. Currently "
        "3 weeks behind the original schedule. Additionally, the QA team has logged 12 critical bugs in the "
        "medication interaction checker — the algorithm was imported from a research paper without sufficient "
        "adaptation for Vietnamese pharmaceutical naming conventions.", styles['Body']))
    s.append(Paragraph(
        "Mitigation: A dedicated integration squad (3 engineers) has been assigned with daily standups "
        "with the SaigonHealth IT team since February 20. For the medication module, we've engaged a "
        "pharmacist consultant to validate the drug interaction rules against Vietnamese formulary data. "
        "Revised delivery date: June 30 (from original June 1).", styles['Body']))

    s.append(Paragraph("3.3 HR Analytics Dashboard — YELLOW", styles['SubHead']))
    s.append(Paragraph(
        "The internal HR Analytics dashboard is behind schedule primarily due to data quality issues in "
        "the legacy HR system (BambooHR). Key problems: inconsistent employee ID formats across systems "
        "(3 different formats discovered), missing termination dates for 15% of departed employees, and "
        "salary history stored in multiple currencies without conversion rates. The ETL pipeline took 40% "
        "longer than estimated due to data cleansing requirements.", styles['Body']))
    s.append(Paragraph(
        "The attrition prediction model is currently at 71% accuracy (F1 score), below the 80% target. "
        "The data science team believes this is due to insufficient feature variety (only HR data; no "
        "performance review or engagement survey data integrated yet). Mitigation: Additional 2-week sprint "
        "for data quality remediation. Exploring integration with Culture Amp engagement data to improve "
        "model accuracy.", styles['Body']))

    s.append(PageBreak())
    s.append(Paragraph("4. Resource Utilization", styles['SectionHead']))
    s.append(make_table(
        ["Department", "Total HC", "Allocated", "Available", "Utilization", "Trend vs Q4", "Key Constraint"],
        [
            ["Frontend Engineering", "45", "42", "3", "93%", "+2pp", "2 engineers on parental leave; 1 open req"],
            ["Backend Engineering", "68", "65", "3", "96%", "+3pp", "Phoenix Platform consuming most bandwidth"],
            ["AI / Machine Learning", "32", "31", "1", "97%", "+5pp", "CRITICAL: Only 1 ML eng available for new work"],
            ["DevOps / Platform (SRE)", "22", "20", "2", "91%", "+1pp", "K8s migration consuming planned capacity"],
            ["QA / Testing", "35", "30", "5", "86%", "-2pp", "Playwright migration freeing some capacity"],
            ["UI/UX Design", "18", "15", "3", "83%", "0pp", "Design system work completed; capacity returning"],
            ["Data Engineering", "20", "19", "1", "95%", "+4pp", "DataLake 360 staffing starts April; tight until then"],
            ["Project Management", "15", "14", "1", "93%", "+3pp", "2 new PMs joining in April"],
        ],
        [75, 40, 50, 48, 48, 48, USABLE_W - 309]
    ))
    s.append(Spacer(1, 6))
    s.append(Paragraph(
        "Overall engineering utilization is at 93%, exceeding the 85% healthy target. The AI/ML and Backend "
        "teams are at critical capacity, which limits ability to absorb new project work or handle unexpected "
        "issues. HR has approved 8 new engineering positions for Q2 2026 hiring (3 Backend, 2 ML, 2 DevOps, "
        "1 QA). Average time-to-hire is currently 45 days.", styles['Body']))

    s.append(Paragraph("5. Key Performance Metrics", styles['SectionHead']))
    s.append(make_table(
        ["Metric", "Q3 2025", "Q4 2025", "Q1 2026", "Target", "Status"],
        [
            ["On-Time Delivery (OTD)", "79%", "82%", "87%", "90%", "Improving — on track to hit target by Q3"],
            ["Client Satisfaction (CSAT)", "4.0/5.0", "4.2/5.0", "4.4/5.0", "4.5/5.0", "Improving — ShopSmart is the main detractor"],
            ["Sprint Velocity (avg across teams)", "38 pts", "42 pts", "45 pts", "45 pts", "On Target"],
            ["Defect Escape Rate (prod bugs/sprint)", "4.1%", "3.2%", "2.8%", "< 2%", "Improving — Playwright migration helping"],
            ["Employee Voluntary Turnover (quarterly)", "5.2%", "4.1%", "3.5%", "< 3%", "Improving — new benefits package impact"],
            ["Revenue per Engineer (monthly)", "$11,800", "$12,400", "$13,100", "$14,000", "Improving — efficiency gains from K8s/CI"],
            ["Gross Margin", "38%", "41%", "43%", "45%", "Improving — ops cost reduction via automation"],
        ],
        [105, 55, 55, 55, 55, USABLE_W - 325]
    ))

    build_pdf("05_Q1_2026_Project_Status.pdf", s)


# ============================================================
# DOCUMENT 6 — Product Requirements Document (ChatAssist)
# ============================================================
def doc_prd():
    s = title_page(
        "Product Requirements Document",
        "ChatAssist — TechViet Internal AI Knowledge Assistant",
        "Product Management Team",
        "Document ID: TVSC-PRD-047 | Version 1.3 | Author: Nguyen Thi Hanh (Head of Product)"
    )

    s.append(Paragraph("1. Executive Overview", styles['SectionHead']))
    s.append(Paragraph(
        "ChatAssist is an AI-powered internal knowledge assistant that enables TechViet employees to "
        "instantly find information from internal documentation, policies, technical wikis, and knowledge "
        "bases using natural language queries. The system uses Retrieval Augmented Generation (RAG) to "
        "provide accurate, sourced answers grounded in TechViet's actual documentation — eliminating "
        "hallucination risks associated with pure LLM approaches.", styles['Body']))
    s.append(Paragraph(
        "This project was initiated by the CTO based on findings from the 2025 Employee Experience Survey, "
        "where \"difficulty finding internal information\" was the #2 pain point (after \"meeting overload\"). "
        "The project has executive sponsorship from the CEO and a dedicated cross-functional team of 6 "
        "(2 ML engineers, 2 backend engineers, 1 frontend engineer, 1 product manager).", styles['Body']))

    s.append(Paragraph("1.1 Problem Statement", styles['SubHead']))
    s.append(Paragraph(
        "TechViet employees currently spend an estimated 5.2 hours per week searching for internal "
        "information across 8+ systems (Confluence, Google Drive, Slack history, HR Portal, GitHub wikis, "
        "email archives, shared folders, and LMS). This results in approximately 2,380 lost productivity "
        "hours per week company-wide (458 employees × 5.2 hours). At an average loaded cost of USD 22/hour, "
        "this represents approximately USD 52,000 per week or USD 2.7 million per year in lost productivity.", styles['Body']))
    s.append(Paragraph(
        "New employees are disproportionately affected: our onboarding survey shows they spend 8.3 hours/week "
        "searching for information in their first 3 months, and cite \"not knowing where to find things\" as "
        "the primary frustration. The average time-to-productivity for new hires is 90 days, and we believe "
        "ChatAssist can reduce this to 60 days.", styles['Body']))

    s.append(Paragraph("1.2 Success Metrics", styles['SubHead']))
    s.append(make_table(
        ["Metric", "Current Baseline", "3-Month Target", "6-Month Target", "12-Month Target", "Measurement Method"],
        [
            ["Time to find info", "15 min avg", "< 3 min", "< 2 min", "< 1 min", "In-app analytics (query to answer)"],
            ["Employee satisfaction (info access)", "3.1/5.0 (2025 survey)", "3.8/5.0", "4.2/5.0", "4.5/5.0", "Quarterly pulse survey"],
            ["IT/HR FAQ ticket volume", "120 tickets/week", "90/week (-25%)", "60/week (-50%)", "30/week (-75%)", "Jira Service Desk analytics"],
            ["New hire time-to-productivity", "90 days", "75 days", "60 days", "45 days", "Manager assessment at milestones"],
            ["Answer accuracy (human-rated)", "N/A", "> 82%", "> 88%", "> 93%", "Weekly sampling: 50 queries human-rated"],
            ["Daily active users", "N/A", "100+ (22%)", "200+ (45%)", "350+ (78%)", "Application analytics (unique users)"],
            ["Query volume", "N/A", "500+/day", "1,000+/day", "2,000+/day", "Application analytics"],
            ["User satisfaction per query", "N/A", "> 3.5/5", "> 4.0/5", "> 4.3/5", "In-app thumbs up/down + periodic survey"],
        ],
        [80, 65, 60, 60, 60, USABLE_W - 325]
    ))
    s.append(Spacer(1, 8))

    s.append(Paragraph("2. Functional Requirements", styles['SectionHead']))
    s.append(make_table(
        ["ID", "Requirement", "Description", "Priority", "Sprint", "Acceptance Criteria"],
        [
            ["FR-001", "Chat Interface", "Web-based conversational UI with message history, typing indicators, and source citation display", "P0", "1-2", "User can type question, receive streaming response with clickable source links"],
            ["FR-002", "RAG Pipeline", "Retrieval-augmented generation with semantic search, reranking, and grounded answer generation", "P0", "2-4", "Answers include relevant source passages; hallucination rate <5% on test set"],
            ["FR-003", "Document Ingestion", "Support PDF, DOCX, PPTX, Markdown, HTML, and plain text. Handle tables, images with OCR, and headers.", "P0", "1-3", "All supported formats parsed correctly; table data preserved; >95% text extraction accuracy"],
            ["FR-004", "Source Attribution", "Every answer must cite specific documents with page/section references. Clickable links to originals.", "P0", "3-4", "Each answer has 1-5 source citations; clicking opens original document at relevant section"],
            ["FR-005", "Multi-turn Conversation", "System maintains context across messages within a session. Handles follow-up questions and pronoun resolution.", "P1", "4-5", "System correctly resolves 'it', 'that policy', 'the same team' in follow-up questions"],
            ["FR-006", "Feedback System", "Thumbs up/down on each response. Optional free-text feedback. Feedback stored for model improvement.", "P1", "5", "Feedback captured and visible in admin dashboard; weekly feedback report auto-generated"],
            ["FR-007", "Admin Dashboard", "Document management (upload, delete, re-index), usage analytics, query analytics, feedback review", "P1", "5-6", "Admin can upload docs, see top queries, review low-rated responses, trigger re-indexing"],
            ["FR-008", "Access Control", "Role-based document access. Users only see answers from documents they're authorized to access.", "P0", "3-4", "HR-only docs not surfaced to non-HR users; access synced with Google Workspace groups"],
            ["FR-009", "Slack Integration", "Query ChatAssist via Slack DM or /chatassist command in any channel. Thread-based responses.", "P2", "7-8", "Users can query from Slack; responses appear in thread with sources; supports follow-ups"],
            ["FR-010", "Auto Re-indexing", "Daily scheduled re-indexing of connected document sources. Incremental updates for changed/new docs.", "P1", "4", "New docs indexed within 24 hours; changed docs updated; deleted docs removed from index"],
            ["FR-011", "Multilingual", "Support Vietnamese and English queries and documents. Auto-detect query language.", "P1", "6-7", "Equivalent accuracy for VN and EN queries; mixed-language docs handled correctly"],
            ["FR-012", "Conversation Export", "Export conversation as PDF or shareable link (read-only, with expiry).", "P2", "8", "Exported PDF matches conversation; shared links expire after 7 days"],
        ],
        [35, 65, 160, 25, 28, USABLE_W - 313]
    ))

    s.append(PageBreak())
    s.append(Paragraph("3. Non-Functional Requirements", styles['SectionHead']))
    s.append(make_table(
        ["ID", "Category", "Requirement", "Target", "Measurement"],
        [
            ["NFR-001", "Performance", "Time to first token (streaming response)", "< 1.5 seconds", "P95 measured by Datadog"],
            ["NFR-002", "Performance", "Full response generation time", "< 6 seconds for typical queries", "P95 measured by Datadog"],
            ["NFR-003", "Performance", "Document ingestion throughput", "100 pages/minute", "Benchmark with standard test corpus"],
            ["NFR-004", "Availability", "System uptime", "99.5% (planned maintenance excluded)", "Datadog SLO monitoring"],
            ["NFR-005", "Scalability", "Concurrent active users", "100+ simultaneous sessions", "Load test with k6"],
            ["NFR-006", "Scalability", "Document corpus size", "50,000+ documents (10M+ chunks)", "Performance test at target scale"],
            ["NFR-007", "Security", "Data residency", "All data in Vietnam (AWS ap-southeast-1)", "AWS resource tagging audit"],
            ["NFR-008", "Security", "Audit logging", "All queries, admin actions, and access logged", "Centralized in Datadog; 1-year retention"],
            ["NFR-009", "Compliance", "Accessibility", "WCAG 2.1 AA compliance", "Automated scan (axe) + manual audit"],
            ["NFR-010", "Reliability", "Data durability", "Zero data loss for indexed documents", "Qdrant replication factor 2; daily backup"],
        ],
        [45, 60, 150, 95, USABLE_W - 350]
    ))
    s.append(Spacer(1, 8))

    s.append(Paragraph("4. System Architecture", styles['SectionHead']))
    s.append(make_table(
        ["Component", "Technology Choice", "Deployment", "Purpose & Notes"],
        [
            ["Chat Frontend", "React 18 + TypeScript", "S3 + CloudFront", "Responsive chat UI with streaming support, markdown rendering, source citation cards"],
            ["API Gateway", "Kong 3.6", "EKS (2 replicas)", "Rate limiting (50 req/min per user), JWT validation, request routing, CORS"],
            ["Backend API", "Python 3.12 + FastAPI", "EKS (3 replicas, HPA)", "Query processing, RAG orchestration, session management, admin endpoints"],
            ["Query Understanding", "Custom NLP pipeline", "Within Backend API", "Query classification, intent detection, query rewriting for better retrieval"],
            ["Embedding Model", "multilingual-e5-large (560M params)", "EKS (GPU node, 2 replicas)", "Generates 1024-dim embeddings for documents and queries. Multilingual support."],
            ["Reranker", "bge-reranker-v2-m3", "EKS (GPU node)", "Cross-encoder reranking of top-50 candidates to top-5. Significantly improves relevance."],
            ["Vector Database", "Qdrant 1.8 (self-hosted)", "EKS (3-node cluster, NVMe SSD)", "Stores ~10M vector embeddings. HNSW index. Payload filtering for access control."],
            ["Document Store", "PostgreSQL 16 + S3", "RDS (Multi-AZ) + S3", "Metadata, user sessions, feedback in Postgres. Raw documents and chunks in S3."],
            ["LLM Provider", "Claude 3.5 Sonnet (Anthropic API)", "External API", "Answer generation with retrieved context. 200K token context window. Streaming enabled."],
            ["Document Processor", "Custom pipeline (Python)", "EKS (async workers)", "PDF/DOCX/HTML parsing, semantic chunking, embedding generation, Qdrant upsert"],
            ["Task Queue", "Redis 7.2 Streams", "ElastiCache (cluster mode)", "Async document processing jobs, scheduled re-indexing triggers"],
            ["Monitoring", "Datadog (APM + Logs + Custom Metrics)", "SaaS", "E2E latency tracking, LLM cost monitoring, retrieval quality metrics, user analytics"],
        ],
        [65, 95, 80, USABLE_W - 240]
    ))
    s.append(Spacer(1, 8))

    s.append(Paragraph("5. Project Timeline", styles['SectionHead']))
    s.append(make_table(
        ["Milestone", "Date", "Key Deliverables", "Dependencies", "Exit Criteria"],
        [
            ["M1: Foundation", "Dec 15, 2025", "Document ingestion pipeline, vector store, basic embedding service, S3 storage", "AWS infra provisioned (DevOps)", "500+ docs ingested; embedding quality validated"],
            ["M2: Core RAG", "Jan 31, 2026", "Query pipeline, RAG orchestration, basic chat UI, streaming responses", "M1 complete; Anthropic API access", "Demo: 10 test queries answered correctly with sources"],
            ["M3: Security & Quality", "Feb 28, 2026", "RBAC, document permissions, reranker, query understanding, feedback system", "M2 complete; IT security review passed", "Access control verified; reranker improves MRR by >15%"],
            ["M4: Internal Beta", "Mar 31, 2026", "Beta launch with 50 users (HR, IT, Engineering leads). Monitoring and alerting.", "M3 complete; QA sign-off; 2,400+ docs indexed", "50 beta users; >82% accuracy; <3s TTFT; feedback loop active"],
            ["M5: Integrations", "Apr 30, 2026", "Slack bot, admin dashboard, analytics, auto-reindexing, conversation export", "M4 feedback incorporated; Slack admin approval", "Slack bot functional; admin can manage docs; analytics dashboard live"],
            ["M6: GA Launch", "May 31, 2026", "Company-wide rollout, multilingual support, performance optimization, documentation", "M5 complete; load test passed; executive sign-off", "Available to all 458 employees; >88% accuracy; SLA met for 2 weeks"],
        ],
        [60, 60, 135, 95, USABLE_W - 350]
    ))

    build_pdf("06_ChatAssist_PRD.pdf", s)


# ============================================================
# DOCUMENT 7 — Business Travel & Expense Policy
# ============================================================
def doc_expense_policy():
    s = title_page(
        "Business Travel & Expense Policy",
        "Guidelines for Travel Approval, Accommodation, Meals, and Reimbursement",
        "Finance Department",
        "Document ID: TVSC-FIN-005 | Version 2.1 | Effective January 1, 2026"
    )

    s.append(Paragraph("1. Purpose and Scope", styles['SectionHead']))
    s.append(Paragraph(
        "This policy governs all business-related travel and expense reimbursement for TechViet employees, "
        "contractors (if specified in their contract), and interns on company business. It ensures consistent, "
        "fair, and fiscally responsible use of company funds while enabling employees to travel effectively. "
        "All business travel must have a clear business purpose documented in the travel request.", styles['Body']))
    s.append(Paragraph(
        "This policy does not cover relocation expenses (see Relocation Policy TVSC-HR-012), client-billable "
        "travel (governed by individual client SOWs), or daily commuting costs (covered by the monthly "
        "transportation allowance in the Employee Handbook).", styles['Body']))

    s.append(Paragraph("2. Travel Approval Matrix", styles['SectionHead']))
    s.append(Paragraph(
        "All business travel must be pre-approved before booking. Approval level is determined by the "
        "estimated total trip cost (including flights, accommodation, meals, and ground transport). Travel "
        "requests must be submitted via the TechViet Finance Portal (finance.techviet.vn).", styles['Body']))
    s.append(make_table(
        ["Travel Type", "Estimated Total Cost", "Approver(s)", "Min Lead Time", "Notes"],
        [
            ["Local (same city)", "< VND 2,000,000", "Direct Manager", "2 business days", "e.g., client meetings, conferences within HCMC/Hanoi/Da Nang"],
            ["Domestic (other city)", "VND 2M — 20M", "Department Head", "5 business days", "e.g., inter-office travel, client site visits, domestic conferences"],
            ["International (ASEAN)", "VND 20M — 50M", "VP + Finance Manager", "10 business days", "e.g., Singapore, Bangkok, Jakarta. Visa lead time may require earlier request."],
            ["International (outside ASEAN)", "> VND 50M", "C-Suite + CFO", "15 business days", "e.g., US, Europe, Japan, Korea, Australia. Requires business case document."],
            ["Group travel (3+ people)", "Any amount", "VP + CFO", "10 business days", "Must justify why multiple people need to travel. Video conference alternatives considered first."],
        ],
        [75, 85, 85, 60, USABLE_W - 305]
    ))
    s.append(Spacer(1, 6))

    s.append(Paragraph("3. Accommodation", styles['SectionHead']))
    s.append(Paragraph(
        "Employees should book accommodation through the TechViet Travel Portal (travel.techviet.vn), which "
        "has negotiated corporate rates with major hotel chains. Booking outside the portal is permitted but "
        "reimbursement is capped at the rates below. Airbnb is permitted for stays of 5+ nights with "
        "manager approval.", styles['Body']))
    s.append(make_table(
        ["Destination", "Max Rate/Night (VND)", "Max Rate/Night (USD)", "Max Star Rating", "Exceptions"],
        [
            ["Ho Chi Minh City", "2,500,000", "100", "4-star", "5-star permitted for client-facing events with VP approval"],
            ["Hanoi", "2,500,000", "100", "4-star", "Same as above"],
            ["Da Nang / Other VN", "1,800,000", "72", "4-star", "Resort properties not permitted unless conference venue"],
            ["Singapore", "N/A", "250", "4-star", "Higher rates during major events (F1, etc.) with pre-approval"],
            ["Bangkok / Jakarta / KL", "N/A", "180", "4-star", "—"],
            ["Tokyo / Seoul / Sydney", "N/A", "300", "4-star", "—"],
            ["San Francisco / New York", "N/A", "400", "4-star", "US tech hubs have higher rates; reviewed annually"],
            ["Other US / Europe", "N/A", "350", "4-star", "—"],
        ],
        [90, 80, 80, 60, USABLE_W - 310]
    ))
    s.append(Spacer(1, 6))

    s.append(Paragraph("4. Meals and Per Diem", styles['SectionHead']))
    s.append(make_table(
        ["Meal", "Domestic (VND)", "ASEAN (USD)", "Northeast Asia (USD)", "US/Europe/AU (USD)"],
        [
            ["Breakfast", "100,000", "15", "20", "25"],
            ["Lunch", "200,000", "25", "35", "40"],
            ["Dinner", "300,000", "40", "50", "60"],
            ["Daily Total", "600,000", "80", "105", "125"],
        ],
        [70, 80, 80, 90, USABLE_W - 320]
    ))
    s.append(Spacer(1, 4))
    s.append(Paragraph(
        "Per diem rates cover meals and incidental expenses only. Alcoholic beverages are not reimbursable "
        "under any circumstances. If a meal is provided by the event, client, or hotel (e.g., conference "
        "lunch, hotel breakfast included), the corresponding per diem is not claimable. Tips are included "
        "in the per diem and not separately reimbursable except in the US (up to 18% at restaurants).", styles['Body']))

    s.append(Paragraph("5. Transportation", styles['SectionHead']))
    s.append(make_table(
        ["Type", "Policy", "Receipt Required", "Booking Method"],
        [
            ["Flights (Domestic)", "Economy class. Book ≥7 days in advance for best rates.", "Yes (e-ticket)", "TechViet Travel Portal (preferred) or direct"],
            ["Flights (Intl < 6hrs)", "Economy class. Vietnam Airlines, Vietjet, or approved carriers.", "Yes", "Travel Portal or travel agent"],
            ["Flights (Intl ≥ 6hrs)", "Premium economy. Business class requires VP approval and is limited to L4+ or medical need.", "Yes", "Travel Portal or travel agent"],
            ["Taxi / Ride-hailing", "Grab or Be preferred. Actual cost reimbursed for business trips only.", "Yes (app receipt)", "Direct booking; save receipt from app"],
            ["Airport transfers", "Taxi or ride-hailing to/from airport. No black car services unless 3+ travelers sharing.", "Yes", "Direct booking"],
            ["Car rental", "Requires pre-approval. Max VND 1,500,000/day. Insurance mandatory.", "Yes (rental agreement + receipts)", "Via Travel Portal preferred"],
            ["Public transit", "Reimbursed at actual cost. Encouraged where practical.", "No (if < VND 200,000)", "Direct; claim on expense report"],
            ["Personal vehicle", "VND 5,000/km. Parking and highway tolls reimbursed. Log mileage.", "Toll receipts; mileage log", "Submit mileage log with expense report"],
        ],
        [70, 170, 85, USABLE_W - 325]
    ))
    s.append(Spacer(1, 6))

    s.append(Paragraph("6. Expense Report Submission", styles['SectionHead']))
    s.append(make_table(
        ["Step", "Action", "Responsible", "Deadline", "Notes"],
        [
            ["1", "Create expense report in Finance Portal with trip details and business purpose", "Employee", "Within 5 business days of trip", "Late submissions (>30 days) will be rejected"],
            ["2", "Upload all receipts (photos acceptable; must be legible with date, vendor, amount)", "Employee", "With report submission", "Missing receipts: attach written explanation"],
            ["3", "Manager reviews and approves/rejects with comments", "Direct Manager", "Within 3 business days", "Manager liable for policy compliance"],
            ["4", "Finance team audits for policy compliance and processes payment", "Finance", "Within 5 business days", "10% of reports randomly selected for detailed audit"],
            ["5", "Reimbursement deposited to employee bank account", "Finance (auto)", "Next payroll cycle", "If rejected, employee notified with reason"],
        ],
        [25, 170, 60, 85, USABLE_W - 340]
    ))
    s.append(Spacer(1, 4))
    s.append(Paragraph(
        "Expenses exceeding policy limits by more than 20% without prior written approval will not be "
        "reimbursed. Fraudulent expense claims are a terminable offense and may result in legal action. "
        "Repeated policy violations (3+ in a 12-month period) will result in suspension of corporate credit "
        "card and travel booking privileges.", styles['Body']))

    build_pdf("07_Travel_Expense_Policy.pdf", s)


# ============================================================
# DOCUMENT 8 — Architecture Decision Records
# ============================================================
def doc_adr():
    s = title_page(
        "Architecture Decision Records",
        "Key Technical Decisions — 2025-2026 Collection",
        "Engineering Architecture Review Board",
        "Maintainer: Dr. Tran Minh Khoa, Principal Engineer"
    )

    # ADR-001
    s.append(Paragraph("ADR-001: Adoption of Qdrant as Primary Vector Database", styles['SectionHead']))
    s.append(Paragraph("Status: Accepted | Date: October 15, 2025 | Deciders: ARB (CTO, VP Eng, Principal Engineers, CISO)", styles['Note']))

    s.append(Paragraph("Context and Problem", styles['SubHead']))
    s.append(Paragraph(
        "TechViet is building multiple AI-powered products requiring high-performance vector similarity "
        "search: ChatAssist (internal RAG chatbot, ~10M vectors), ShopSmart Reco (product embeddings, ~5M "
        "vectors), and LogiFlow AI (location/route embeddings, ~2M vectors). We need a vector database "
        "that can handle production workloads with low latency, supports metadata filtering for access "
        "control, and can be self-hosted in our AWS ap-southeast-1 infrastructure for data residency.", styles['Body']))

    s.append(Paragraph("Evaluation Results", styles['SubHead']))
    s.append(Paragraph(
        "We conducted a 4-week proof-of-concept evaluation with 1M vectors (1024 dimensions, matching our "
        "multilingual-e5-large embedding model). Each solution was tested on the same hardware (r6g.xlarge "
        "instances) with identical query patterns. The evaluation criteria and weights were agreed upon by "
        "the ARB before testing began.", styles['Body']))
    s.append(make_table(
        ["Criteria (Weight)", "Qdrant", "Pinecone", "Weaviate", "Milvus", "pgvector"],
        [
            ["Query Latency P95 (25%)", "4.2ms — 9/10", "3.8ms — 9/10", "8.1ms — 7/10", "5.5ms — 8/10", "22ms — 5/10"],
            ["Scalability to 10M+ (20%)", "Sharding + replication — 8/10", "Fully managed, unlimited — 10/10", "Auto-sharding — 8/10", "Distributed native — 9/10", "Limited by PG — 4/10"],
            ["Cost at Target Scale (20%)", "~$180/mo self-hosted — 9/10", "~$850/mo managed — 4/10", "~$250/mo self-hosted — 7/10", "~$220/mo self-hosted — 7/10", "~$50/mo (in existing PG) — 10/10"],
            ["Metadata Filtering (15%)", "Native payload filters, fast — 9/10", "Basic filters, limited ops — 7/10", "GraphQL-style filters — 8/10", "Boolean expressions — 8/10", "SQL WHERE clause — 6/10"],
            ["Self-hosted Capable (10%)", "Docker/K8s, easy — 10/10", "SaaS only — 0/10", "Docker/K8s — 10/10", "K8s, complex setup — 10/10", "Same as PG — 10/10"],
            ["Docs & Community (10%)", "Good docs, growing — 8/10", "Excellent — 9/10", "Decent — 7/10", "Large but fragmented — 7/10", "PG ecosystem — 8/10"],
            ["WEIGHTED TOTAL", "8.75 / 10", "6.55 / 10", "7.60 / 10", "7.90 / 10", "6.30 / 10"],
        ],
        [90, 85, 85, 85, 85, USABLE_W - 430]
    ))
    s.append(Spacer(1, 6))

    s.append(Paragraph("Decision", styles['SubHead']))
    s.append(Paragraph(
        "Adopt Qdrant as the primary vector database, deployed as a 3-node cluster on EKS with NVMe-backed "
        "EBS volumes. Pinecone is approved as a secondary option for rapid prototyping and PoC work where "
        "self-hosting overhead is not justified. pgvector is permitted for projects with small vector "
        "collections (<100K vectors) that don't warrant a separate vector database.", styles['Body']))
    s.append(Paragraph("Consequences:", styles['BodyBold']))
    s.append(bullet("DevOps team must create Helm charts and runbooks for Qdrant cluster operations"))
    s.append(bullet("ML team to develop shared Python SDK for Qdrant interactions (techviet-vectordb library)"))
    s.append(bullet("Monitoring: Qdrant metrics exported to Datadog; alerting on cluster health and query latency"))
    s.append(bullet("Backup: Qdrant snapshots to S3 every 6 hours; tested restore procedure monthly"))

    s.append(Spacer(1, 16))

    # ADR-002
    s.append(Paragraph("ADR-002: Strangler Fig Migration from Django Monolith to Microservices", styles['SectionHead']))
    s.append(Paragraph("Status: Accepted | Date: November 5, 2025 | Deciders: ARB + VietBank Project Leads", styles['Note']))

    s.append(Paragraph("Context and Problem", styles['SubHead']))
    s.append(Paragraph(
        "The Phoenix Platform (VietBank's core banking modernization project) has grown to 280,000 lines "
        "of code in a single Django 4.2 monolith over 18 months of development. The monolith has become a "
        "significant impediment to delivery speed and reliability:", styles['Body']))
    s.append(bullet("Deployment frequency dropped from daily to bi-weekly (entire app must be deployed together)"))
    s.append(bullet("A single failing test in any module blocks all deployments for all teams"))
    s.append(bullet("Database schema changes require coordination across 5 teams, causing frequent merge conflicts"))
    s.append(bullet("Build time: 25 minutes (was 4 minutes 12 months ago). Test suite: 47 minutes."))
    s.append(bullet("Scaling is all-or-nothing: cannot scale the transaction processing module independently"))

    s.append(Paragraph("Decision", styles['SubHead']))
    s.append(Paragraph(
        "Adopt the Strangler Fig pattern to incrementally extract high-value services from the monolith. "
        "The monolith remains operational throughout migration — no big-bang rewrite. Services are extracted "
        "based on a prioritization score combining business criticality, change frequency, team coupling, "
        "and scaling requirements.", styles['Body']))
    s.append(make_table(
        ["Phase", "Service to Extract", "Size", "Target Stack", "Timeline", "Team", "Priority Score"],
        [
            ["1", "Authentication & Authorization", "12K LOC", "Go + gRPC + Redis", "Q1 2026", "Platform Team (4 eng)", "9.2 — highest change frequency; security-critical"],
            ["2", "Notification Service", "8K LOC", "Python + FastAPI + Kafka", "Q1 2026", "Integrations (3 eng)", "8.5 — low coupling; easy extraction; blocking email reliability"],
            ["3", "Transaction Processing", "45K LOC", "Go + Kafka + PostgreSQL", "Q2 2026", "Core Banking (6 eng)", "8.8 — scaling bottleneck; highest business value"],
            ["4", "Reporting & Analytics", "35K LOC", "Python + FastAPI + ClickHouse", "Q2-Q3 2026", "Data Team (4 eng)", "7.5 — performance issues; separate data store beneficial"],
            ["5", "Customer Management", "28K LOC", "Go + gRPC + PostgreSQL", "Q3 2026", "Core Banking (4 eng)", "7.0 — moderate coupling; benefits from independent scaling"],
            ["6", "Remaining Monolith", "152K LOC", "Django 4.2 (maintained)", "Ongoing", "All teams (shared)", "N/A — gradually shrinks as services extracted"],
        ],
        [30, 90, 40, 85, 50, 85, USABLE_W - 380]
    ))
    s.append(Spacer(1, 6))
    s.append(Paragraph("Consequences:", styles['BodyBold']))
    s.append(bullet("Need service mesh (Istio) for inter-service communication, observability, and mTLS"))
    s.append(bullet("API gateway (Kong) to route traffic between monolith and new services during migration"))
    s.append(bullet("Distributed tracing (Datadog APM) critical for debugging cross-service requests"))
    s.append(bullet("Each extraction requires: data migration plan, backward-compatible API, 2-week parallel run"))

    s.append(PageBreak())

    # ADR-003
    s.append(Paragraph("ADR-003: Standardizing on GitHub Actions for All CI/CD", styles['SectionHead']))
    s.append(Paragraph("Status: Accepted | Date: December 1, 2025 | Deciders: ARB + DevOps Team", styles['Note']))

    s.append(Paragraph("Context and Problem", styles['SubHead']))
    s.append(Paragraph(
        "TechViet currently operates 3 separate CI/CD systems: Jenkins (8 projects, self-hosted on EC2), "
        "GitLab CI (2 projects, SaaS), and GitHub Actions (14 projects). This fragmentation creates "
        "significant operational overhead: 3 systems to maintain and secure, inconsistent pipeline "
        "definitions, different secret management approaches, and difficulty enforcing company-wide "
        "security scanning standards. The Jenkins server requires approximately 8 hours/month of DevOps "
        "maintenance and has suffered 3 outages in the last 6 months.", styles['Body']))

    s.append(make_table(
        ["Evaluation Criteria", "Jenkins", "GitLab CI", "GitHub Actions"],
        [
            ["Maintenance Effort", "HIGH: Self-hosted on EC2; 8 hrs/mo maintenance; frequent plugin updates; Java upgrades required", "LOW: SaaS managed; but separate from our primary Git platform", "LOW: SaaS managed; native to our Git platform (GitHub Enterprise)"],
            ["Integration Quality", "MEDIUM: Plugin-based GitHub integration; can be unreliable; requires webhooks", "LOW: Requires mirroring repos from GitHub to GitLab; adds complexity and latency", "HIGH: Native integration; status checks, PR annotations, Dependabot, CodeQL built-in"],
            ["Ecosystem / Marketplace", "LARGE but aging: Many plugins are unmaintained; security risk from unvetted plugins", "GROWING: Good built-in features but smaller marketplace than GitHub Actions", "LARGEST: 15,000+ actions; active community; first-party actions for major tools"],
            ["Security Scanning", "MANUAL: Must configure Snyk, SonarQube plugins separately; inconsistent across projects", "BUILT-IN: Basic SAST/DAST; limited compared to GitHub's offering", "EXCELLENT: CodeQL, Dependabot, secret scanning native; Snyk integration seamless"],
            ["Self-hosted Runners", "N/A (all self-hosted)", "AVAILABLE: Works but management overhead", "AVAILABLE: Using for builds; well-documented; auto-scaling with Actions Runner Controller"],
            ["Cost (Annual)", "~$12,000 (EC2 + maintenance time)", "~$5,400 (SaaS plan for 2 projects)", "Included in GitHub Enterprise license ($0 incremental)"],
            ["Current Projects", "8 (all legacy)", "2 (will migrate to GitHub)", "14 (all new projects)"],
        ],
        [80, 145, 145, USABLE_W - 370]
    ))
    s.append(Spacer(1, 6))

    s.append(Paragraph("Decision", styles['SubHead']))
    s.append(Paragraph(
        "Standardize on GitHub Actions for all CI/CD pipelines. Create a shared library of reusable "
        "workflow templates in the techviet/ci-templates repository. Migration plan: GitLab CI projects "
        "migrated by end of Q1 2026 (2 projects, low risk). Jenkins projects migrated by end of Q2 2026 "
        "(8 projects, staged migration). Jenkins EC2 instances decommissioned by July 2026.", styles['Body']))

    build_pdf("08_Architecture_Decision_Records.pdf", s)


# ============================================================
# DOCUMENT 9 — Incident Post-Mortem
# ============================================================
def doc_incident_postmortem():
    s = title_page(
        "Incident Post-Mortem Report",
        "INC-2026-0042: Production Database Outage — Phoenix Platform",
        "Engineering Reliability Team",
        "Severity: P1 Critical | Date: February 18, 2026 | Duration: 2h 47m"
    )

    s.append(Paragraph("1. Incident Summary", styles['SectionHead']))
    s.append(make_table(
        ["Field", "Details"],
        [
            ["Incident ID", "INC-2026-0042"],
            ["Severity", "P1 — Critical (complete service outage)"],
            ["Duration", "2 hours 47 minutes (14:23 — 17:10 ICT, February 18, 2026)"],
            ["Affected Service", "Phoenix Platform — core banking application for VietBank"],
            ["User Impact", "Complete outage: 12,000 VietBank end users unable to access online banking, mobile app, and internal teller system"],
            ["Financial Impact", "Estimated VND 450M in lost transaction processing revenue for VietBank. SLA penalty clause triggered: TechViet liable for 2% of monthly contract value (~USD 1,700)."],
            ["Root Cause", "Developer accidentally ran database migration against production instead of staging, causing ACCESS EXCLUSIVE lock on 85M-row transactions table"],
            ["Detection", "Automated (Datadog alert: P95 latency exceeded 30s threshold on /api/v1/transactions)"],
            ["Incident Commander", "Le Van Hung, VP Engineering"],
            ["Post-Mortem Author", "Nguyen Quoc Bao, DevOps Lead"],
        ],
        [100, USABLE_W - 100]
    ))
    s.append(Spacer(1, 8))

    s.append(Paragraph("2. Detailed Timeline", styles['SectionHead']))
    s.append(make_table(
        ["Time (ICT)", "Event", "Actor / System"],
        [
            ["14:12", "Developer (Dev A) opens terminal to run staging migration. Has both staging and production database connection strings in local .env file.", "Dev A"],
            ["14:15", "Dev A runs 'alembic upgrade head' — unknowingly connected to production database. The migration contains CREATE INDEX on transactions.created_at column.", "Dev A"],
            ["14:18", "PostgreSQL begins building B-tree index on transactions table (85M rows, ~42GB). Acquires ACCESS EXCLUSIVE lock, blocking ALL reads and writes.", "PostgreSQL"],
            ["14:23", "Datadog alert fires: P95 latency on /api/v1/transactions exceeds 30s threshold. PagerDuty page sent to on-call engineer (Dev B).", "Datadog + PagerDuty"],
            ["14:25", "Dev B acknowledges PagerDuty alert. Checks Datadog dashboard — sees all transaction-related endpoints timing out. Connection pool at 100% utilization.", "Dev B (On-call)"],
            ["14:28", "Dev B SSHs into production database monitoring server. Runs pg_stat_activity query — discovers long-running CREATE INDEX with ACCESS EXCLUSIVE lock.", "Dev B"],
            ["14:32", "Dev B identifies the migration as unplanned (not in today's deploy manifest). Escalates to P1 severity. Creates #inc-2026-0042 Slack channel.", "Dev B"],
            ["14:35", "VP Engineering (Le Van Hung) assumes Incident Commander role. DevOps Lead and DBA paged.", "IC (Le Van Hung)"],
            ["14:38", "DBA assesses situation: migration is 35% complete. Killing the query would leave a partially-built invalid index requiring manual cleanup (estimated 30+ min).", "DBA (Senior)"],
            ["14:40", "IC decision: Allow migration to complete naturally (estimated 40 min remaining). Killing it risks longer total outage and potential data corruption.", "IC + DBA"],
            ["14:45", "Account Manager notifies VietBank IT Director via phone. Provides estimated resolution time: 1 hour. VietBank activates their fallback procedures.", "Account Manager"],
            ["15:08", "Migration reaches 75%. Datadog shows first signs of some non-transaction endpoints recovering (those not touching the locked table).", "Monitoring"],
            ["15:22", "Migration completes. ACCESS EXCLUSIVE lock released. But PgBouncer connection pool remains exhausted — all 200 connections held by stale/retry connections.", "PostgreSQL / PgBouncer"],
            ["15:28", "DevOps manually resets PgBouncer connection pool. Connections begin recovering.", "DevOps Engineer"],
            ["15:35", "50% of requests succeeding. Application pods health checks passing. Gradual traffic restoration via load balancer.", "DevOps + Monitoring"],
            ["16:00", "80% capacity restored. Remaining issues: some user sessions expired during outage requiring re-login.", "Monitoring"],
            ["16:45", "100% service restored. All health checks passing. Error rate back to baseline (<0.1%).", "Monitoring"],
            ["17:10", "25-minute stability observation period completed successfully. Incident declared RESOLVED.", "IC"],
            ["17:30", "Detailed timeline and initial findings shared with VietBank IT team.", "IC + Account Manager"],
        ],
        [40, USABLE_W - 115, 75]
    ))

    s.append(PageBreak())
    s.append(Paragraph("3. Root Cause Analysis (5 Whys)", styles['SectionHead']))
    s.append(Paragraph(
        "We applied the 5 Whys technique combined with a contributing factors analysis to identify both "
        "the immediate cause and the systemic issues that allowed this incident to occur.", styles['Body']))
    s.append(make_table(
        ["Level", "Question", "Answer"],
        [
            ["Why 1", "Why did the production database lock up for 62 minutes?", "A CREATE INDEX operation on an 85M-row table acquired an ACCESS EXCLUSIVE lock, blocking all other operations."],
            ["Why 2", "Why was CREATE INDEX run instead of CREATE INDEX CONCURRENTLY?", "The migration was written by a junior developer and not reviewed for PostgreSQL-specific best practices. Our migration review checklist doesn't include index creation method verification."],
            ["Why 3", "Why did this migration run against production?", "The developer had both staging and production database connection strings in their local .env file and selected the wrong one. There was no confirmation prompt or environment verification."],
            ["Why 4", "Why does a developer have production database credentials locally?", "Our current setup stores all environment configs in a single .env file downloaded from the shared secrets vault. There's no separation between staging and production credentials."],
            ["Why 5", "Why wasn't this caught by any safety mechanism?", "We had no pre-migration safety checks: no table size estimation, no lock impact analysis, no environment confirmation prompt, and no requirement to test against production-scale data."],
        ],
        [40, 130, USABLE_W - 170]
    ))
    s.append(Spacer(1, 6))

    s.append(Paragraph("Contributing Factors", styles['SubHead']))
    s.append(make_table(
        ["ID", "Factor", "Category", "Severity"],
        [
            ["F1", "Production and staging credentials stored in same .env file with no visual distinction or safety check", "Configuration / Credential Management", "Critical"],
            ["F2", "No confirmation prompt, environment banner, or guard rail for production database operations", "Process / Tooling", "Critical"],
            ["F3", "Migration not tested against production-sized dataset (staging has 10K rows vs. 85M in production)", "Testing", "High"],
            ["F4", "CREATE INDEX used instead of CREATE INDEX CONCURRENTLY — developer was unaware of the distinction", "Knowledge / Training", "High"],
            ["F5", "PgBouncer connection pool had no circuit breaker or surge protection; exhaustion cascaded the failure", "Infrastructure", "Medium"],
            ["F6", "Migration code review process does not include DBA review for schema changes on large tables", "Process", "Medium"],
        ],
        [25, USABLE_W - 180, 85, 70]
    ))
    s.append(Spacer(1, 8))

    s.append(Paragraph("4. Action Items", styles['SectionHead']))
    s.append(make_table(
        ["ID", "Action", "Owner", "Priority", "Deadline", "Status"],
        [
            ["A1", "Separate production credentials into isolated HashiCorp Vault namespace. Developers cannot access production DB credentials without explicit request + DBA approval.", "DevOps Lead", "P0", "Feb 25", "DONE"],
            ["A2", "Add production migration safety tool: requires --confirm-production flag, shows environment banner with 10s countdown, logs operator identity.", "Platform Team", "P0", "Feb 28", "DONE"],
            ["A3", "Engineering standard update: All index operations on tables >1M rows MUST use CREATE INDEX CONCURRENTLY. Added to migration review checklist.", "Engineering", "P0", "Mar 1", "DONE"],
            ["A4", "Pre-migration analyzer: estimates lock duration based on table size + operation type. Warns if estimated lock >30 seconds. Blocks if >5 minutes.", "Platform Team", "P1", "Mar 15", "IN PROGRESS"],
            ["A5", "PgBouncer circuit breaker: auto-reset connection pool if utilization stays at 100% for >60 seconds. Implement health-check endpoint for PgBouncer.", "DevOps", "P1", "Mar 15", "IN PROGRESS"],
            ["A6", "Create production-mirror environment (shadow DB with anonymized production-scale data) for migration testing. Automated nightly refresh.", "DevOps + DBA", "P2", "Apr 15", "PLANNED"],
            ["A7", "Runbook: database migration rollback procedures for common scenarios (failed index, locked table, bad data migration). Published to Confluence.", "DBA Team", "P1", "Mar 10", "IN PROGRESS"],
            ["A8", "Training session: safe database migration practices. Mandatory for all engineers. Covers: CONCURRENTLY, lock types, large table strategies, testing.", "Engineering + DBA", "P2", "Mar 31", "PLANNED"],
        ],
        [20, 185, 55, 20, 40, USABLE_W - 320]
    ))
    s.append(Spacer(1, 8))

    s.append(Paragraph("5. What Went Well", styles['SectionHead']))
    s.append(bullet("Detection was fast: automated monitoring caught the issue within 5 minutes of lock acquisition."))
    s.append(bullet("Incident response was well-coordinated: IC assigned within 12 minutes; clear roles throughout."))
    s.append(bullet("Client communication was timely and transparent: VietBank informed within 22 minutes."))
    s.append(bullet("Post-incident: 3 of 8 action items completed within 10 days. No repeat incidents."))

    s.append(Paragraph("6. What Needs Improvement", styles['SectionHead']))
    s.append(bullet("22-minute gap between lock release and PgBouncer reset — automated pool recovery would have saved this."))
    s.append(bullet("Junior developer had production credentials with no additional safeguards — principle of least privilege violated."))
    s.append(bullet("Migration review process had no DBA checkpoint for schema changes on large tables."))
    s.append(bullet("No production-scale test environment exists — migrations tested only against small staging data."))

    build_pdf("09_Incident_Postmortem_INC0042.pdf", s)


# ============================================================
# DOCUMENT 10 — IT Support Knowledge Base
# ============================================================
def doc_it_faq():
    s = title_page(
        "IT Support Knowledge Base",
        "Frequently Asked Questions, Troubleshooting Guides, and Self-Service Instructions",
        "IT Department",
        "Last Updated: March 2026 | Contact: #it-helpdesk on Slack | it-support@techviet.vn"
    )

    s.append(Paragraph("1. Account & Access Management", styles['SectionHead']))

    s.append(Paragraph("1.1 How do I reset my password?", styles['SubHead']))
    s.append(Paragraph(
        "Navigate to https://sso.techviet.vn/reset and enter your company email address "
        "(firstname.lastname@techviet.vn). A password reset link will be sent within 5 minutes. Check "
        "your spam/junk folder if not received. The link expires after 1 hour. Your new password must "
        "meet all security requirements (see table below). If you're completely locked out and cannot "
        "access email, contact IT Help Desk directly — you'll need to verify your identity with your "
        "employee ID number and date of birth.", styles['Body']))
    s.append(make_table(
        ["Requirement", "Details", "Common Mistakes to Avoid"],
        [
            ["Minimum Length", "14 characters", "Using common patterns like 'Password123!' — blocked by our dictionary checker"],
            ["Character Types", "Must include all 4: uppercase, lowercase, number, special character", "Putting the special char at the end only — distribute throughout for better security"],
            ["Cannot Reuse", "Last 12 passwords are remembered", "Incrementing numbers (Password1!, Password2!) — our system detects sequential patterns"],
            ["Rotation", "Must change every 90 days", "You'll get reminders at 14, 7, and 1 day(s) before expiry via email and Slack"],
            ["Banned Words", "Your name, email prefix, 'techviet', 'password', '123456' and 10,000+ common passwords", "Any variation of these words is blocked, including leetspeak (p@ssw0rd)"],
        ],
        [80, 160, USABLE_W - 240]
    ))
    s.append(Spacer(1, 6))

    s.append(Paragraph("1.2 Setting Up Multi-Factor Authentication (MFA)", styles['SubHead']))
    s.append(Paragraph(
        "MFA is mandatory for all TechViet accounts. You'll be prompted to set up MFA during your first "
        "login. We support two methods — hardware security keys are strongly preferred for their phishing "
        "resistance. SMS-based MFA is NOT supported due to SIM-swapping risks.", styles['Body']))
    s.append(make_table(
        ["Method", "Setup Steps", "Best For", "Recovery if Lost"],
        [
            ["YubiKey (Hardware Key)", "1. Insert YubiKey into USB port. 2. When prompted during login, tap the gold sensor. 3. Key is automatically registered. You receive 2 keys — keep the backup in a safe place.", "All employees (most secure). Required for: AWS access, production systems, admin roles.", "Use backup YubiKey. If both lost: contact IT with employee ID for emergency reset (identity verification required in person)."],
            ["Authenticator App (TOTP)", "1. Download Google Authenticator or Authy on your phone. 2. Scan QR code displayed during MFA setup. 3. Enter the 6-digit code to verify. 4. Save the recovery codes in your password manager.", "Remote workers, mobile-first users. Acceptable for: standard systems, email, Slack.", "Use recovery codes (saved during setup). If codes lost: IT can reset MFA with manager approval + identity verification."],
        ],
        [80, USABLE_W / 2 - 80, 110, USABLE_W / 2 - 110]
    ))
    s.append(Spacer(1, 6))

    s.append(Paragraph("1.3 Account Lockout and Recovery", styles['SubHead']))
    s.append(Paragraph(
        "Your account is locked after 5 consecutive failed login attempts. This is a security measure to "
        "prevent brute-force attacks. Accounts auto-unlock after 30 minutes. If you cannot wait, or if "
        "you suspect your account was compromised (you didn't cause the lockout), follow these steps:", styles['Body']))
    s.append(make_table(
        ["Situation", "Action", "Contact"],
        [
            ["Normal lockout (you entered wrong password)", "Wait 30 minutes for auto-unlock, then reset your password at sso.techviet.vn/reset", "Self-service"],
            ["Urgent: need access immediately", "Contact IT Help Desk on Slack (#it-helpdesk) or call ext. 100. Verify with employee ID.", "#it-helpdesk"],
            ["Suspected compromise (you didn't cause the lockout)", "Immediately report to Security Team. Do NOT attempt to log in. They will investigate and secure your account.", "#security-incidents or soc@techviet.vn"],
            ["Lockout during off-hours (after 6 PM / weekends)", "Email it-support@techviet.vn with subject 'URGENT: Account Lockout'. On-call IT responds within 1 hour.", "it-support@techviet.vn"],
        ],
        [115, USABLE_W - 215, 100]
    ))
    s.append(Spacer(1, 8))

    s.append(Paragraph("2. VPN & Remote Access", styles['SectionHead']))

    s.append(Paragraph("2.1 Connecting to the Company VPN", styles['SubHead']))
    s.append(Paragraph(
        "VPN connection is required to access internal systems when working remotely. TechViet uses "
        "WireGuard VPN for its superior performance and security compared to traditional VPN solutions. "
        "Split tunneling is disabled — all traffic routes through the VPN when connected.", styles['Body']))
    s.append(make_table(
        ["Step", "Action", "Notes / Troubleshooting"],
        [
            ["1", "Install WireGuard client from TechViet Software Center (pre-installed on company laptops). For personal devices: download from wireguard.com.", "macOS: also available via App Store. Windows: requires admin rights for first install."],
            ["2", "Go to https://vpn.techviet.vn and log in with your SSO credentials + MFA.", "This site is accessible WITHOUT VPN (public-facing). Bookmark it."],
            ["3", "Download your personal VPN configuration file (.conf). This is unique to you — do not share it.", "File is regenerated every 90 days. You'll get a Slack reminder before expiry."],
            ["4", "In WireGuard app: click 'Import tunnel(s) from file' and select the downloaded .conf file.", "On macOS, you may need to drag the file into the WireGuard app window."],
            ["5", "Click 'Activate' to connect. A green indicator means you're connected.", "First connection may take 5-10 seconds. Subsequent connections are near-instant."],
            ["6", "Verify by navigating to an internal site (e.g., jira.techviet.vn). If it loads, you're connected.", "If it doesn't load, see troubleshooting table below."],
        ],
        [30, 195, USABLE_W - 225]
    ))
    s.append(Spacer(1, 6))

    s.append(Paragraph("2.2 VPN Troubleshooting", styles['SubHead']))
    s.append(make_table(
        ["Symptom", "Likely Cause", "Solution"],
        [
            ["VPN connects but internal sites don't load", "DNS resolution issue. Your system may be using a public DNS server instead of the corporate one.", "Flush DNS: macOS: 'sudo dscacheutil -flushcache && sudo killall -HUP mDNSResponder'. Windows: 'ipconfig /flushdns'. Then reconnect VPN."],
            ["VPN frequently disconnects (every few minutes)", "Unstable internet connection or firewall blocking WireGuard UDP traffic (port 51820).", "Try wired connection. If on hotel/airport Wi-Fi, their firewall may block UDP. Solution: use the TCP fallback endpoint (see vpn.techviet.vn/tcp-fallback)."],
            ["Very slow speeds through VPN", "Connected to a distant VPN server or ISP throttling.", "Change server: HCM office users → sgn.vpn.techviet.vn. Hanoi → han.vpn.techviet.vn. International → sgp.vpn.techviet.vn (Singapore)."],
            ["'Handshake timeout' error", "Config file expired (older than 90 days) or clock skew on your device.", "Download fresh config from vpn.techviet.vn. Also verify your system clock is accurate (Settings > Date & Time > auto-set)."],
            ["Cannot access specific service (e.g., AWS console) but VPN otherwise works", "You may not have the required access group for that service.", "Submit an access request via IT ticket. Include: service name, business justification, and manager approval."],
        ],
        [95, 100, USABLE_W - 195]
    ))
    s.append(Spacer(1, 8))

    s.append(Paragraph("3. Software & Development Environment", styles['SectionHead']))

    s.append(Paragraph("3.1 Pre-installed Development Tools", styles['SubHead']))
    s.append(make_table(
        ["Category", "macOS (MacBook Pro)", "Windows (Dell XPS/Precision)", "Notes"],
        [
            ["IDE / Editor", "VS Code 1.87+, IntelliJ IDEA Ultimate 2024.1", "VS Code 1.87+, IntelliJ IDEA Ultimate 2024.1", "JetBrains license managed via company account"],
            ["Terminal", "iTerm2, Oh My Zsh (preconfigured)", "Windows Terminal, Git Bash", "zsh is default shell on macOS"],
            ["Containers", "Docker Desktop (4 CPU, 8GB RAM default), kubectl, Helm", "Docker Desktop (WSL2 backend), kubectl, Helm", "Request more Docker resources via IT if needed"],
            ["Version Control", "Git 2.43+, GitHub CLI (gh), Git LFS", "Git 2.43+, GitHub CLI (gh), Git LFS", "Pre-configured with company SSH keys"],
            ["Languages", "Python 3.12 (pyenv), Node.js 20 (nvm), Go 1.22 (goenv)", "Python 3.12 (pyenv-win), Node.js 20 (nvm-windows), Go 1.22", "Use version managers; don't install globally"],
            ["Database Tools", "DBeaver Community, pgcli, mongosh", "DBeaver Community, pgcli, mongosh", "For prod DB access: see Access Control policy"],
            ["API Testing", "Postman, httpie, curl", "Postman, httpie, curl", "Postman collections in team shared workspace"],
            ["Security", "CrowdStrike Falcon, 1Password Teams", "CrowdStrike Falcon, 1Password Teams, BitLocker", "Do NOT uninstall CrowdStrike — instant compliance violation"],
        ],
        [65, 135, 135, USABLE_W - 335]
    ))
    s.append(Spacer(1, 6))

    s.append(Paragraph("3.2 Requesting Additional Software", styles['SubHead']))
    s.append(Paragraph(
        "Submit requests via the TechViet Software Portal (software.techviet.vn). Software on the approved "
        "list is auto-installed within 1 business day. Software NOT on the approved list requires a security "
        "review (3-5 business days) by IT Security before installation. Open-source tools with active "
        "maintenance and no known CVEs are generally approved within 2 business days.", styles['Body']))

    s.append(Paragraph("4. Hardware Support", styles['SectionHead']))

    s.append(Paragraph("4.1 Laptop Performance Issues", styles['SubHead']))
    s.append(make_table(
        ["Step", "Action", "Expected Impact", "Difficulty"],
        [
            ["1", "Restart your laptop fully (Shut Down, not just close lid / Sleep). Many issues are caused by months of uptime.", "Fixes 50%+ of performance issues. Clears RAM, resets file handles, applies pending updates.", "Easy"],
            ["2", "Close unnecessary browser tabs. Chrome uses 100-300MB per tab. Close tabs with video or complex web apps first.", "Can free 2-6 GB RAM. Use OneTab extension to save tabs without keeping them open.", "Easy"],
            ["3", "Check for and install pending OS updates (macOS: System Settings > Software Update; Windows: Settings > Windows Update).", "Security patches + performance fixes. Some updates require restart.", "Easy"],
            ["4", "Quit Docker Desktop if not actively using containers. Docker reserves 4-8 GB RAM by default even when idle.", "Frees 4-8 GB RAM immediately. Restart Docker when you need it.", "Easy"],
            ["5", "Check Activity Monitor (macOS) / Task Manager (Windows) for unexpected high-CPU processes. Screenshot and submit IT ticket if unfamiliar.", "Identifies rogue processes. CrowdStrike scan can temporarily spike CPU — this is normal.", "Medium"],
            ["6", "Run disk cleanup. macOS: Apple menu > About > Storage > Manage. Windows: Disk Cleanup utility. Clear caches, old downloads.", "Frees disk space. Low disk space (<10% free) significantly degrades performance.", "Medium"],
            ["7", "If still slow after above steps, submit IT ticket. Include: laptop model, macOS/Windows version, and Activity Monitor screenshot.", "IT will run hardware diagnostics: SSD health, RAM test, battery check. May replace components.", "IT Support"],
        ],
        [25, 195, 170, USABLE_W - 390]
    ))
    s.append(Spacer(1, 6))

    s.append(Paragraph("4.2 Laptop Replacement Program", styles['SubHead']))
    s.append(make_table(
        ["Model", "Configuration", "Typical Role / Use Case", "Approval Level"],
        [
            ["MacBook Pro 14\" M3 Pro", "18GB RAM, 512GB SSD, 14\" Liquid Retina XDR", "Frontend engineers, backend engineers, designers, PMs, business roles", "Standard (auto-approved at 3-year cycle)"],
            ["MacBook Pro 16\" M3 Max", "36GB RAM, 1TB SSD, 16\" Liquid Retina XDR", "ML engineers, senior engineers, DevOps, architects, anyone running local LLMs/training", "Manager approval + IT confirmation"],
            ["Dell XPS 15 9530", "32GB RAM, 1TB SSD, i9-13900H, 15.6\" OLED", "Engineers on Windows-required projects (e.g., .NET, specific client tools)", "Standard"],
            ["Dell Precision 5690", "64GB RAM, 2TB SSD, i9-14900HX, RTX 4000 Ada", "ML engineers doing local model training, heavy data processing, GPU-required workloads", "VP approval + IT confirmation"],
        ],
        [85, 130, 165, USABLE_W - 380]
    ))

    s.append(Paragraph("5. Contact Information", styles['SectionHead']))
    s.append(make_table(
        ["Channel", "Contact Details", "Best For", "Response Time"],
        [
            ["Slack (preferred)", "#it-helpdesk channel", "General IT questions, quick issues, software requests", "< 30 minutes during business hours"],
            ["Email", "it-support@techviet.vn", "Detailed issues, attachment-heavy requests, after-hours", "< 2 hours (business hours); < 4 hours (after hours)"],
            ["Phone", "+84 28 3823 XXXX ext. 100", "Urgent issues only (locked out before important meeting, laptop dead)", "Immediate during business hours"],
            ["Walk-in", "IT Help Desk, 2nd Floor, HCM Office", "Hardware issues, peripheral setup, in-person troubleshooting", "First-come, first-served (15-30 min wait typical)"],
            ["Self-Service Portal", "support.techviet.vn", "Track ticket status, browse knowledge base, submit requests 24/7", "Immediate (auto-responses); tickets triaged next business day"],
        ],
        [65, 110, 175, USABLE_W - 350]
    ))

    build_pdf("10_IT_Support_FAQ.pdf", s)


# ============================================================
# DOCUMENT 11 — Data Privacy & Compliance Policy
# ============================================================
def doc_data_privacy():
    s = title_page(
        "Data Privacy & Compliance Policy",
        "Personal Data Protection Framework — PDPD, GDPR, and PDPA Compliance",
        "Legal & Compliance Department",
        "Document ID: TVSC-LEG-002 | Version 1.4 | Effective January 1, 2026"
    )

    s.append(Paragraph("1. Scope and Legal Framework", styles['SectionHead']))
    s.append(Paragraph(
        "This policy applies to all TechViet employees, contractors, and third-party partners who process, "
        "store, or transmit personal data on behalf of TechViet or its clients. TechViet operates under "
        "multiple data protection regulations depending on the data subjects and jurisdictions involved:", styles['Body']))
    s.append(make_table(
        ["Regulation", "Jurisdiction", "Applies When", "Key Requirements", "Penalty for Non-Compliance"],
        [
            ["PDPD (Decree 13/2023/ND-CP)", "Vietnam", "Processing personal data of individuals in Vietnam", "Consent, data localization, breach notification within 72 hours, Data Protection Officer required", "Up to VND 100M per violation; potential criminal liability"],
            ["GDPR (Regulation 2016/679)", "European Union", "Processing data of EU residents (applies to 3 TechViet clients)", "Lawful basis, DPIA, DPO, 72-hour breach notification, right to erasure, data portability", "Up to €20M or 4% of global annual turnover"],
            ["PDPA (Act 26 of 2012)", "Singapore", "Processing data via TechViet's Singapore operations and SG-based clients", "Consent, purpose limitation, access and correction rights, 3-day breach notification", "Up to SGD 1M per violation"],
        ],
        [85, 55, 95, 135, USABLE_W - 370]
    ))
    s.append(Spacer(1, 4))
    s.append(Paragraph(
        "TechViet's Data Protection Officer (DPO) is Tran Thi Kim Anh (dpo@techviet.vn). The DPO reports "
        "directly to the CEO and has independent authority to escalate data protection concerns. All "
        "employees must complete annual data privacy training (1.5 hours, via LMS) and pass with ≥85%.", styles['Body']))

    s.append(Paragraph("2. Data Processing Principles", styles['SectionHead']))
    s.append(make_table(
        ["Principle", "Description", "How We Implement It"],
        [
            ["Lawfulness & Fairness", "Process personal data only with a valid legal basis (consent, contract, legal obligation, or legitimate interest)", "Legal review for all new data processing activities. Consent management system for marketing. Legitimate interest assessments documented."],
            ["Purpose Limitation", "Collect data only for specified, explicit purposes. Do not use for incompatible purposes.", "Privacy notices specify all purposes. Data inventory tracks purpose per dataset. New uses require DPO review."],
            ["Data Minimization", "Collect only the personal data that is necessary for the stated purpose", "Data fields reviewed during system design (Privacy by Design checklist). 'Nice to have' fields rejected by DPO."],
            ["Accuracy", "Keep personal data accurate and up to date", "Self-service update portal for employees and users. Annual data quality audit. Automated address verification for client data."],
            ["Storage Limitation", "Retain data only as long as necessary for the purpose (see Retention Schedule)", "Automated deletion workflows. Retention schedule per data type (Section 4). Annual retention audit by Legal."],
            ["Integrity & Confidentiality", "Protect data with appropriate technical and organizational security measures", "Encryption at rest and in transit. Access controls per IT Security Policy. Regular penetration testing."],
            ["Accountability", "Be able to demonstrate compliance with all principles", "Data Protection Impact Assessments (DPIA) for high-risk processing. Processing records maintained. Annual external audit."],
        ],
        [80, 150, USABLE_W - 230]
    ))
    s.append(Spacer(1, 6))

    s.append(Paragraph("3. Personal Data Categories and Handling", styles['SectionHead']))
    s.append(make_table(
        ["Category", "Examples", "Sensitivity", "Storage Requirements", "Access Control"],
        [
            ["Basic Identity", "Full name, email, phone, address, date of birth", "Standard", "Encrypted at rest (AES-256); standard backup procedures", "HR team + direct manager for employees; authorized staff for clients"],
            ["Employment Data", "Employee ID, job title, salary, performance reviews, disciplinary records", "High", "Encrypted + separate database schema; enhanced audit logging", "HR team only; manager sees only own reports; employee sees own data"],
            ["Financial Data", "Bank accounts, tax IDs, payment card numbers, salary history", "High", "PCI DSS Level 1 compliant storage; tokenization for card data; separate encryption keys", "Finance team only; tokenized for all other systems; PCI-scoped access"],
            ["Health Data", "Insurance claims, medical certificates, disability status, COVID vaccination", "Sensitive", "Physically separate encrypted database; own encryption keys; no cloud replication", "HR health admin (1 person) + DPO; manager sees only leave dates, not medical details"],
            ["Biometric Data", "Fingerprint (office access), facial recognition (visitor management)", "Sensitive", "On-premises processing and storage ONLY; templates stored, not raw biometrics; no cloud", "Security team only; 30-day auto-deletion; cannot be exported"],
            ["Client End-User PII", "Data processed on behalf of clients (varies per contract)", "Variable (per DPA)", "Per client Data Processing Agreement; data segregation mandatory; client-specific encryption", "Project team only (scoped); access revoked at project end; no cross-project access"],
        ],
        [65, 100, 55, 140, USABLE_W - 360]
    ))
    s.append(Spacer(1, 6))

    s.append(Paragraph("4. Data Retention Schedule", styles['SectionHead']))
    s.append(make_table(
        ["Data Type", "Retention Period", "Action After Retention", "Legal Basis", "Automated?"],
        [
            ["Employee records (active)", "Duration of employment", "Move to archive upon separation", "Employment contract", "N/A"],
            ["Employee records (separated)", "5 years after separation", "Anonymize (aggregated for analytics) or delete", "Vietnamese Labor Code + tax law", "Annual review; batch process in January"],
            ["Client project data", "Contract duration + 3 years", "Delete per client DPA terms; certificate of destruction", "Contractual obligation", "Automated reminder at contract end + 2.5 years"],
            ["Financial / accounting records", "10 years from creation", "Move to read-only archive for 2 additional years, then delete", "Vietnamese tax regulations (Decree 123/2020)", "Automated archival; manual deletion"],
            ["Application / system logs", "90 days", "Auto-deleted from all systems including backups", "Legitimate interest (security monitoring)", "Fully automated (Datadog log pipeline)"],
            ["Security camera footage", "30 days", "Auto-overwritten by recording system", "Legitimate interest (physical security)", "Hardware-level auto-overwrite"],
            ["Email communications", "3 years active + 2 years archive", "Permanent deletion from all backups", "Business operations", "Google Workspace retention policy + Veeam"],
            ["Marketing consent records", "Duration of consent + 2 years", "Delete consent record and all associated data", "PDPD Article 11 + GDPR Article 7", "Consent management system auto-tracks"],
            ["Job applicant data (not hired)", "6 months after hiring decision", "Delete unless candidate consents to talent pool retention (max 2 years)", "Consent (talent pool) / Legitimate interest (rejected)", "Automated via ATS (Greenhouse) pipeline"],
            ["Website analytics / cookies", "26 months from collection", "Auto-deleted from analytics platform", "Consent (cookie banner)", "Google Analytics 4 auto-deletion setting"],
        ],
        [80, 80, 105, 95, USABLE_W - 360]
    ))

    s.append(PageBreak())
    s.append(Paragraph("5. Data Subject Rights", styles['SectionHead']))
    s.append(Paragraph(
        "TechViet respects and facilitates the rights of all data subjects under applicable privacy laws. "
        "Requests can be submitted via the Privacy Portal (privacy.techviet.vn), email (privacy@techviet.vn), "
        "or in writing to the DPO. All requests are logged, tracked, and must be responded to within the "
        "deadlines specified below.", styles['Body']))
    s.append(make_table(
        ["Right", "Description", "Response Time", "Process", "Exceptions"],
        [
            ["Right of Access", "Obtain a complete copy of all personal data we hold about you, in a readable format", "30 days (PDPD/GDPR); 21 days (PDPA)", "Submit request via Privacy Portal. Identity verification required. Data compiled by DPO office.", "May be restricted if disclosure would compromise security investigation or legal proceedings"],
            ["Right to Rectification", "Correct or update inaccurate or incomplete personal data", "15 days", "Self-service via HR Portal (employees) or Privacy Portal (others). Complex changes reviewed by DPO.", "None — always honored"],
            ["Right to Erasure", "Request deletion of personal data when no longer necessary or consent withdrawn", "30 days", "DPO reviews request against retention obligations. If approved, deletion across all systems + backups.", "Cannot override legal retention requirements (tax, labor law). Will explain which data and why."],
            ["Right to Data Portability", "Receive your data in a structured, machine-readable format (JSON/CSV)", "30 days", "Export generated by automated system. Delivered via secure download link (encrypted, 7-day expiry).", "Applies only to data provided by the subject, not derived/inferred data"],
            ["Right to Restrict Processing", "Pause processing of your data while a dispute or objection is being resolved", "15 days to implement restriction", "DPO implements technical restriction. Data retained but not processed until resolved.", "Essential processing (legal obligations, contract performance) may continue"],
            ["Right to Object", "Object to processing based on legitimate interest grounds", "30 days for response", "DPO conducts balancing test: individual rights vs. TechViet legitimate interest. Decision documented.", "Cannot object to processing required by law or contract"],
            ["Right to Withdraw Consent", "Withdraw any previously given consent at any time, without affecting past processing", "Immediate effect", "Self-service via consent management (for marketing). Privacy Portal for other consent-based processing.", "Withdrawal does not affect lawfulness of prior processing based on consent"],
        ],
        [60, 110, 55, 130, USABLE_W - 355]
    ))
    s.append(Spacer(1, 8))

    s.append(Paragraph("6. Data Breach Response Procedure", styles['SectionHead']))
    s.append(make_table(
        ["Phase", "Actions", "Timeline", "Responsible", "Documentation"],
        [
            ["1. Detection & Containment", "Identify breach vector. Isolate affected systems. Preserve forensic evidence. Stop ongoing data loss.", "Immediate upon discovery. Must be reported to SOC within 1 hour.", "SOC / IT Security (first responder); CISO (escalation)", "Incident ticket in Jira; real-time updates in Slack incident channel"],
            ["2. Assessment", "Determine: scope (how many records), type (what data), root cause, affected individuals, ongoing risk.", "Within 4 hours of detection", "DPO + CISO + affected team leads", "Breach assessment report (template in Confluence)"],
            ["3. Internal Notification", "Brief CEO, Legal, affected department heads. Prepare external notification if required.", "Within 8 hours", "DPO", "Internal briefing document"],
            ["4. Authority Notification", "Report to relevant data protection authority if required (mandatory for breaches affecting >1000 individuals or sensitive data).", "Within 72 hours (PDPD/GDPR); 3 days (PDPA)", "DPO + Legal Counsel", "Authority notification form (per regulation)"],
            ["5. Individual Notification", "Notify affected data subjects if breach poses high risk to their rights. Include: what happened, what data, what we're doing, what they should do.", "Without undue delay after authority notification", "DPO + Communications team", "Notification letter/email (Legal-reviewed template)"],
            ["6. Remediation", "Implement technical and organizational fixes. Patch vulnerabilities. Update access controls.", "Per remediation action plan (typically 1-4 weeks)", "IT Security + Engineering", "Remediation plan with deadlines; tracked in Jira"],
            ["7. Post-Incident Review", "Document lessons learned. Update policies and procedures. Conduct additional training if needed.", "Within 14 days of incident closure", "DPO + CISO + all involved parties", "Post-incident report (published to Legal & Security teams)"],
        ],
        [70, 160, 60, 80, USABLE_W - 370]
    ))

    build_pdf("11_Data_Privacy_Policy.pdf", s)


# ============================================================
# DOCUMENT 12 — Engineering All-Hands Meeting Minutes
# ============================================================
def doc_meeting_minutes():
    s = title_page(
        "Engineering All-Hands Meeting Minutes",
        "Quarterly Engineering Update — Q1 2026 Review",
        "Engineering Leadership Team",
        "Date: March 14, 2026 | Time: 2:00 — 3:30 PM ICT | Location: Auditorium + Zoom"
    )

    s.append(Paragraph("Attendees", styles['SectionHead']))
    s.append(make_table(
        ["Name", "Title", "Attendance", "Role in Meeting"],
        [
            ["Dr. Pham Quang Minh", "CEO / Co-Founder", "In-person", "Opening remarks"],
            ["Le Van Hung", "VP Engineering", "In-person", "Q1 review presentation"],
            ["Nguyen Quoc Bao", "DevOps Lead / SRE Manager", "In-person", "Platform modernization update"],
            ["Dr. Tran Minh Khoa", "Principal Engineer, AI/ML", "In-person", "AI team update"],
            ["Vo Thi Lan", "QA Director", "Zoom (Hanoi)", "Quality and testing update"],
            ["Tran Van Duc", "IT Manager", "In-person", "IT infrastructure announcements"],
            ["All engineering staff (~180)", "Various roles", "Mixed (120 in-person, 60 Zoom)", "Audience + Q&A"],
        ],
        [100, 115, 75, USABLE_W - 290]
    ))
    s.append(Spacer(1, 8))

    s.append(Paragraph("1. CEO Opening Remarks (Dr. Pham Quang Minh)", styles['SectionHead']))
    s.append(Paragraph(
        "Dr. Minh opened the meeting by congratulating the engineering team on a strong Q1. He highlighted "
        "that TechViet has been shortlisted for the Vietnam Technology Awards 2026 in the \"Enterprise "
        "Software\" category, largely based on the Phoenix Platform and ChatAssist projects. He emphasized "
        "that engineering is the heart of TechViet and that the company will continue to invest heavily in "
        "technical excellence, hiring, and learning & development.", styles['Body']))
    s.append(Paragraph(
        "He also shared a company-wide update: TechViet closed a Series B funding round of USD 8 million "
        "from Vertex Ventures and Do Ventures in February 2026. A significant portion of this funding is "
        "earmarked for: expanding the AI/ML team (target: 50 headcount by Q4 2026), building a dedicated "
        "GPU cluster for model training, and establishing a small R&D lab in Singapore.", styles['Body']))

    s.append(Paragraph("2. Q1 2026 Engineering Performance (Le Van Hung)", styles['SectionHead']))
    s.append(Paragraph(
        "Le Van Hung presented the quarterly engineering metrics. Overall, the team showed strong improvement "
        "across all key indicators, with particularly notable gains in deployment frequency and incident "
        "response time (the latter driven by improvements made after the February database outage).", styles['Body']))
    s.append(make_table(
        ["Metric", "Q3 2025", "Q4 2025", "Q1 2026", "Q1 Target", "YoY Change", "Commentary"],
        [
            ["Deployments/Week", "32", "45", "62", "50", "+94%", "GitHub Actions migration + ArgoCD automation driving increase"],
            ["Mean Time to Recovery (MTTR)", "62 min", "45 min", "28 min", "30 min", "-55%", "Post-INC-0042 improvements + automated rollback"],
            ["P1 Incidents", "4", "3", "1", "≤2", "-75%", "Single incident (INC-0042). Zero in last 30 days."],
            ["Change Failure Rate", "8.2%", "5.1%", "3.8%", "<5%", "-54%", "Better testing + canary deployments reducing bad deploys"],
            ["PR Merge Time (avg)", "24 hrs", "18 hrs", "11 hrs", "12 hrs", "-54%", "Review bot reminders + culture shift toward faster reviews"],
            ["Test Coverage (avg)", "68%", "74%", "79%", "80%", "+16%", "Close to target. New projects starting above 85%."],
            ["Build Success Rate", "88%", "91%", "96%", "95%", "+9%", "CI template standardization eliminating flaky infra issues"],
            ["Developer Satisfaction", "3.6/5", "3.9/5", "4.2/5", "4.0/5", "+17%", "Quarterly pulse survey. Improved tooling cited most."],
        ],
        [70, 40, 40, 42, 42, 42, USABLE_W - 276]
    ))
    s.append(Spacer(1, 8))

    s.append(Paragraph("3. Platform Modernization Update (Nguyen Quoc Bao)", styles['SectionHead']))
    s.append(Paragraph(
        "Quoc Bao provided a detailed status update on the platform modernization initiatives that began "
        "in Q3 2025. The overall program is on track, with most initiatives ahead of their original "
        "timelines due to strong team execution and the dedicated platform engineering squad.", styles['Body']))
    s.append(make_table(
        ["Initiative", "Target State", "Current Status", "% Complete", "ETA", "Blockers / Notes"],
        [
            ["Kubernetes Migration", "All 12 production services on EKS", "8 of 12 services migrated and stable in production", "70%", "May 2026", "Phoenix Platform (largest) is next; requires careful coordination with VietBank"],
            ["Jenkins Decommission", "Zero Jenkins pipelines; all on GitHub Actions", "6 of 8 legacy pipelines migrated; 2 remaining (Phoenix, SecureGate)", "75%", "Apr 2026", "Phoenix pipeline is complex (47 stages). SecureGate has GovTech compliance constraints."],
            ["Terraform Coverage", "100% infrastructure-as-code; no manual resource creation", "85% of AWS resources managed by Terraform", "85%", "Jun 2026", "Legacy VPC configs and some IAM roles still need import. Sprint planned for May."],
            ["Datadog Full Rollout", "All services instrumented with APM, logs, and custom metrics", "10 of 12 services fully instrumented", "83%", "Apr 2026", "Remaining 2 services (legacy) need code changes for APM auto-instrumentation."],
            ["Secrets Auto-Rotation", "All secrets in HashiCorp Vault with automated rotation", "60% of secrets migrated; rotation policies defined", "60%", "May 2026", "Database credentials rotation requires app-level connection pool changes."],
            ["Cost Optimization", "20% reduction in AWS monthly spend via right-sizing and reserved instances", "15% reduction achieved ($4,200/mo saved)", "75%", "Jun 2026", "Savings Commitments purchased for EC2. Next: S3 lifecycle policies + unused EBS cleanup."],
        ],
        [65, 100, 115, 35, 40, USABLE_W - 355]
    ))
    s.append(Spacer(1, 8))

    s.append(Paragraph("4. AI/ML Team Update (Dr. Tran Minh Khoa)", styles['SectionHead']))
    s.append(Paragraph(
        "Dr. Khoa shared updates on the AI team's three active projects and the team's research initiatives.", styles['Body']))

    s.append(Paragraph("ChatAssist (Internal RAG Chatbot):", styles['BodyBold']))
    s.append(Paragraph(
        "ChatAssist is entering internal beta on March 31 with 50 users (selected from HR, IT, and "
        "Engineering leadership). The team has processed 2,400 internal documents (Confluence pages, PDF "
        "policies, Slack FAQ threads) into the Qdrant vector store, totaling approximately 180,000 text "
        "chunks. Early accuracy testing on a curated set of 200 questions shows 88% accuracy for HR/policy "
        "questions and 84% for IT/technical questions. The semantic chunking pipeline with overlap has "
        "significantly improved retrieval quality compared to the initial fixed-size chunking approach. "
        "Key challenge: handling tables in PDF documents — current extraction loses structure. The team is "
        "evaluating document-understanding models (LayoutLM) to improve table extraction.", styles['Body']))

    s.append(Paragraph("ShopSmart Recommendation Engine:", styles['BodyBold']))
    s.append(Paragraph(
        "After the scope renegotiation (real-time personalization moved to Phase 2 with separate SOW), the "
        "project is back on track. The new recommendation pipeline using Kafka Streams achieves P95 latency "
        "of 120ms, well within the 200ms SLA. A/B testing with MegaMart shows a 7.3% increase in click-through "
        "rate and 4.1% increase in basket size compared to MegaMart's previous rule-based system. Full "
        "rollout to all MegaMart users expected by April 15.", styles['Body']))

    s.append(Paragraph("LogiFlow AI Route Optimization:", styles['BodyBold']))
    s.append(Paragraph(
        "The route optimization model for VNLogistics is in training phase. Using a combination of "
        "graph neural networks and reinforcement learning. Early results on historical data show 12% "
        "reduction in fuel costs and 8% improvement in delivery time vs. VNLogistics' current heuristic "
        "system. However, the model struggles with real-time traffic integration — the team is exploring "
        "integration with Google Maps Traffic API for live traffic data during inference.", styles['Body']))

    s.append(Paragraph("5. Quality & Testing Update (Vo Thi Lan)", styles['SectionHead']))
    s.append(Paragraph(
        "Lan announced the progress on the Playwright migration from Selenium. Three teams have fully "
        "migrated their E2E test suites, reporting 60% faster test execution and 40% reduction in flaky "
        "tests. The QA team has published a comprehensive Playwright migration guide on Confluence with "
        "code examples for common patterns. Full migration across all teams expected by end of Q2 2026.", styles['Body']))
    s.append(Paragraph(
        "She also launched the 'Quality Champion' program: one engineer per team will be trained as a "
        "testing advocate, responsible for: reviewing test quality in code reviews, running monthly test "
        "health assessments, and sharing testing best practices. 8 volunteers have signed up. The first "
        "Quality Champion training session is scheduled for March 28.", styles['Body']))

    s.append(Paragraph("6. Open Q&A Session", styles['SectionHead']))
    s.append(make_table(
        ["Question", "Asked By", "Answer (Summarized)"],
        [
            ["When will we adopt Python 3.13?", "Backend Team member", "CTO: Python 3.13 is being evaluated in Q2. Key concern is dependency compatibility (some ML libraries lag). Target adoption: Q3 2026. An RFC will be published for community input."],
            ["Can we get GPU instances for local ML development?", "AI Team member", "DevOps Lead: Approved! SageMaker notebook instances available starting next week. Budget: 4x ml.g5.xlarge instances shared across AI team. Local GPU laptops also available (Dell Precision 5690)."],
            ["Are we considering Rust for performance-critical services?", "Platform Team member", "CTO: Interested but cautious. Rust expertise is scarce in our team. Proposal: submit an RFC to ARB with a specific use case. If approved, we'll fund Rust training for a 3-person pilot squad."],
            ["Will there be a hackathon this year?", "Multiple (chat)", "VP Engineering: Yes! 'TechViet Innovation Days' — a 48-hour hackathon — is confirmed for May 16-17. Theme: 'AI-Powered Internal Tools'. Cross-functional teams of 3-5. Prizes: 1st VND 30M, 2nd VND 20M, 3rd VND 10M."],
            ["What's the status of the 4-day work week pilot?", "Multiple (chat)", "CTO: HR is analyzing full data from the Q1 pilot (3 teams, 42 participants). Preliminary data is positive (productivity maintained, satisfaction up). Full results at April all-hands. Decision on expansion expected by May."],
            ["Can we hire more ML engineers? The team is at 97% utilization.", "AI Team Lead", "VP Eng: 2 ML engineer positions opened last week. Also approved: 1 ML platform engineer and 1 research scientist. Total 4 new AI team hires in Q2. Referral bonus doubled to VND 20M for ML roles."],
        ],
        [105, 70, USABLE_W - 175]
    ))
    s.append(Spacer(1, 8))

    s.append(Paragraph("7. Action Items from This Meeting", styles['SectionHead']))
    s.append(make_table(
        ["Action Item", "Owner", "Deadline", "Status"],
        [
            ["Publish Q1 Engineering Metrics Report on Confluence", "Le Van Hung", "March 18, 2026", "Pending"],
            ["Publish Playwright migration guide (v2) with advanced patterns", "Vo Thi Lan / QA Team", "March 21, 2026", "In Progress"],
            ["Open ChatAssist beta registration for 50 internal users", "Dr. Tran Minh Khoa", "March 25, 2026", "Pending"],
            ["Schedule Architecture Review Board meeting for Rust RFC", "Nguyen Quoc Bao", "April 1, 2026", "Pending"],
            ["Announce TechViet Innovation Days (hackathon) registration", "Le Van Hung", "March 28, 2026", "Pending"],
            ["Publish ML hiring positions on all channels; activate referral program", "HR / Nguyen Thi Mai", "March 18, 2026", "In Progress"],
            ["Share SageMaker notebook access guide with AI team", "DevOps Team", "March 18, 2026", "In Progress"],
            ["Distribute 4-day work week pilot preliminary findings to engineering leads", "CTO Office", "March 21, 2026", "Pending"],
        ],
        [USABLE_W - 230, 100, 70, 60]
    ))
    s.append(Spacer(1, 8))

    s.append(Paragraph("Next Engineering All-Hands: June 13, 2026, 2:00 PM ICT", styles['BodyBold']))

    build_pdf("12_Engineering_AllHands_Minutes.pdf", s)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("Generating TechViet corporate documents...\n")
    doc_employee_handbook()
    doc_it_security()
    doc_onboarding()
    doc_engineering_standards()
    doc_project_status()
    doc_prd()
    doc_expense_policy()
    doc_adr()
    doc_incident_postmortem()
    doc_it_faq()
    doc_data_privacy()
    doc_meeting_minutes()
    print(f"\nDone! {len(os.listdir(OUTPUT_DIR))} documents generated in: {OUTPUT_DIR}")
