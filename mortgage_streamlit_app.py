"""
Australian Mortgage Refinance Analyser v1.2
Real-time · RBA/ASX · Individualised fees · Extra repayments/redraws
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, datetime, timedelta
import calendar
import requests
import re, io, json

try:
    from scipy.optimize import brentq
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="AU Mortgage Refinance Analyser",
                   layout="wide", initial_sidebar_state="collapsed")

# ═══════════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
*, html, body, [class*="css"] { font-family:'Inter',system-ui,sans-serif; }
.stApp { background:#07090f; color:#d4dbe8; }
section[data-testid="stSidebar"] { background:#0b0f1a; }
.stNumberInput input,.stTextInput input{background:#111827!important;border:1px solid #1e2d4a!important;border-radius:5px!important;color:#d4dbe8!important;font-size:0.875rem!important;}
.stDateInput input{background:#111827!important;border:1px solid #1e2d4a!important;color:#d4dbe8!important;font-size:0.875rem!important;}
.stSelectbox>div>div{background:#111827!important;border:1px solid #1e2d4a!important;color:#d4dbe8!important;}
input:disabled,.stNumberInput input:disabled{background:#0b0f1a!important;color:#4a5568!important;border-color:#0f1929!important;}
.stTabs [data-baseweb="tab-list"]{background:#0b0f1a;border-bottom:1px solid #1e2d4a;gap:0;padding:0 4px;}
.stTabs [data-baseweb="tab"]{background:transparent;color:#64748b;border-radius:0;padding:10px 16px;font-size:0.78rem;font-weight:500;border-bottom:2px solid transparent;}
.stTabs [aria-selected="true"]{background:transparent!important;color:#4a9af5!important;border-bottom:2px solid #4a9af5!important;}
.streamlit-expanderHeader{background:#0b0f1a!important;border:1px solid #1e2d4a!important;border-radius:6px!important;color:#d4dbe8!important;font-size:0.875rem!important;font-weight:500!important;}
.streamlit-expanderContent{background:#07090f!important;border:1px solid #1e2d4a!important;border-top:none!important;border-radius:0 0 6px 6px!important;padding:16px!important;}
.stButton>button{background:#1a2744;color:#d4dbe8;border:1px solid #1e2d4a;border-radius:5px;font-size:0.82rem;font-weight:500;padding:6px 14px;transition:all .15s;}
.stButton>button:hover{background:#243560;border-color:#4a9af5;color:#4a9af5;}
.btn-danger>button{background:transparent!important;border-color:#e94560!important;color:#e94560!important;}
.m-card{background:#0b0f1a;border:1px solid #1e2d4a;border-radius:7px;padding:11px 13px;margin-bottom:5px;}
.m-label{color:#64748b;font-size:0.66rem;font-weight:500;text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px;}
.m-value{color:#d4dbe8;font-size:1.2rem;font-weight:600;line-height:1.2;}
.m-diff{color:#64748b;font-size:0.68rem;margin-top:2px;}
.m-diff-pos{color:#30d996;font-size:0.68rem;margin-top:2px;}
.m-diff-neg{color:#e94560;font-size:0.68rem;margin-top:2px;}
.sec-title{color:#64748b;font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.08em;padding:12px 0 6px;border-bottom:1px solid #1e2d4a;margin-bottom:10px;}
.sub-title{color:#8892b0;font-size:0.78rem;font-weight:600;padding:8px 0 4px;}
.list-hdr{color:#64748b;font-size:0.69rem;font-weight:600;text-transform:uppercase;letter-spacing:.04em;padding-bottom:3px;}
.note{background:#0b0f1a;border-left:2px solid #4a9af5;border-radius:0 4px 4px 0;padding:7px 11px;font-size:0.78rem;color:#64748b;margin:6px 0;}
.note-warn{background:#0b0f1a;border-left:2px solid #f5a94a;border-radius:0 4px 4px 0;padding:7px 11px;font-size:0.78rem;color:#a0896a;margin:6px 0;}
.note-ok{background:#0b0f1a;border-left:2px solid #30d996;border-radius:0 4px 4px 0;padding:7px 11px;font-size:0.78rem;color:#4a9a74;margin:6px 0;}
.cf{background:#0d1626;border:1px solid #1e2d4a;border-radius:5px;padding:8px 11px;font-size:0.875rem;margin-bottom:4px;}
.cf-lbl{color:#64748b;font-size:0.69rem;margin-bottom:2px;}
.cf-val{color:#4a9af5;font-weight:600;font-size:1rem;}
.cf-sub{color:#64748b;font-size:0.7rem;margin-top:1px;}
.rate-up{color:#e94560;font-weight:600;}
.rate-dn{color:#30d996;font-weight:600;}
.rate-nc{color:#64748b;}
.data-panel{background:#0b0f1a;border:1px solid #1e2d4a;border-radius:8px;padding:14px 16px;margin-bottom:10px;}
.data-panel-title{color:#4a9af5;font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px;}
hr{border-color:#1e2d4a!important;margin:16px 0!important;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

C_ORIG,C_CURR,C_VAR,C_FIX,C_SPLIT="#4a9af5","#f5a94a","#30d996","#e94560","#c47af5"
C_PAPER,C_PLOT,C_GRID,C_TEXT="#0b0f1a","#07090f","#1e2d4a","#d4dbe8"

PLOT_BASE = dict(
    paper_bgcolor=C_PAPER, plot_bgcolor=C_PLOT,
    font=dict(family="Inter", color=C_TEXT, size=11),
    xaxis=dict(gridcolor=C_GRID, zerolinecolor=C_GRID, linecolor=C_GRID),
    yaxis=dict(gridcolor=C_GRID, zerolinecolor=C_GRID, linecolor=C_GRID),
    legend=dict(bgcolor=C_PAPER, bordercolor=C_GRID, borderwidth=1, font=dict(size=11)),
    margin=dict(t=42, b=36, l=60, r=20),
    hovermode="x unified",
    hoverlabel=dict(bgcolor=C_PAPER, bordercolor=C_GRID, font=dict(color=C_TEXT, size=11)),
)

TODAY = date.today()

HDRS = {"User-Agent": "Mozilla/5.0 (compatible; AU-Mortgage-Analyser/1.2)",
        "Accept": "text/html,text/csv,application/json,*/*"}

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def fc(v, d=0):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "N/A"
    return ("-" if v < 0 else "") + f"${abs(v):,.{d}f}"

def fp(v, d=2):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "N/A"
    return f"{v:.{d}f}%"

def parse_dt(v) -> date:
    if isinstance(v, date): return v
    if isinstance(v, datetime): return v.date()
    try: return datetime.strptime(str(v), "%Y-%m-%d").date()
    except: return TODAY

def months_between(d1: date, d2: date) -> int:
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)

def add_months(d: date, n: int) -> date:
    m = d.month - 1 + n
    y = d.year + m // 12
    mo = m % 12 + 1
    return date(y, mo, min(d.day, calendar.monthrange(y, mo)[1]))

def sec(t: str):
    st.markdown(f'<div class="sec-title">{t}</div>', unsafe_allow_html=True)

def computed(lbl: str, val: str, sub: str = ""):
    s = f'<div class="cf-sub">{sub}</div>' if sub else ""
    st.markdown(f'<div class="cf"><div class="cf-lbl">{lbl}</div>'
                f'<div class="cf-val">{val}</div>{s}</div>', unsafe_allow_html=True)

def metric_card(lbl: str, val: str, diff: str = "", diff_pos: bool = True,
                diff_neutral: bool = False, sub_diff: str = ""):
    if diff_neutral or not diff: dc = "m-diff"
    elif diff_pos: dc = "m-diff-pos"
    else: dc = "m-diff-neg"
    dh = f'<div class="{dc}">{diff}</div>' if diff else ""
    sh = f'<div class="{("m-diff-pos" if diff_pos else "m-diff-neg")}">{sub_diff}</div>' if sub_diff else ""
    return (f'<div class="m-card"><div class="m-label">{lbl}</div>'
            f'<div class="m-value">{val}</div>{dh}{sh}</div>')

# ═══════════════════════════════════════════════════════════════════════════════
# FINANCIAL ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def calc_payment(principal: float, rate_pct: float, term_mo: int) -> float:
    if principal <= 0 or term_mo <= 0: return 0.0
    if rate_pct <= 0: return principal / term_mo
    r = rate_pct / 100 / 12
    return principal * r * (1 + r) ** term_mo / ((1 + r) ** term_mo - 1)

def build_rate_schedule(base: float, deltas: list) -> list:
    sched = [(date(1900, 1, 1), base)]
    cum = base
    for row in sorted([(parse_dt(r[0]), float(r[1])) for r in deltas if r[0]],
                      key=lambda x: x[0]):
        cum = round(cum + row[1], 4)
        sched.append((row[0], cum))
    return sched

def get_rate(sched: list, d: date) -> float:
    r = sched[0][1]
    for rd, rv in sched:
        if rd <= d: r = rv
        else: break
    return r

@st.cache_data(show_spinner=False)
def amortize_cached(
    principal: float, start_str: str, term_mo: int,
    rate_sched_t: tuple,
    off_init: float, off_monthly: float, off_lumps_t: tuple,
    extra_repay_t: tuple,
    monthly_fee: float, maintain_pmt: bool, min_pmt_floor: float = 0.0
) -> pd.DataFrame:
    """
    Daily-interest amortisation.

    Rules:
    - Interest = (balance - offset) × (rate/365) × days
    - Rate rise → payment INCREASES to maintain remaining term
    - Rate fall + maintain_pmt=True → keep higher payment (term shortens)
    - Rate fall + maintain_pmt=False → reduce to minimum (term preserved)
    - min_pmt_floor: optional floor (forces payment ≥ this amount in all periods)
    - extra_repay_t: ((month_offset, amount), ...) positive = extra repayment, negative = redraw
    """
    if principal <= 0 or term_mo <= 0:
        return pd.DataFrame()

    start = date.fromisoformat(start_str)
    rate_sched = [(date.fromisoformat(ds), r) for ds, r in rate_sched_t]

    def offset_at(mo: int) -> float:
        bal = off_init + mo * off_monthly
        for lmo, amt in off_lumps_t:
            if lmo <= mo: bal += amt
        return max(0.0, bal)

    # Index extra repayments by month
    extra_map = {}
    for mo, amt in extra_repay_t:
        extra_map[mo] = extra_map.get(mo, 0.0) + amt

    bal = principal
    prev_rate = get_rate(rate_sched, start)
    cur_pmt = calc_payment(principal, prev_rate, term_mo)
    # Apply floor if larger
    if min_pmt_floor > 0 and min_pmt_floor > cur_pmt:
        cur_pmt = min_pmt_floor

    cum_int = cum_paid = cum_saved = cum_extra = 0.0
    rows = []
    period = start

    for mo in range(1, term_mo + 121):
        if bal <= 0.01: break
        next_d = add_months(period, 1)
        days = (next_d - period).days
        ann_rate = get_rate(rate_sched, period)

        if abs(ann_rate - prev_rate) > 1e-9:
            rem = max(1, term_mo - mo + 1)
            req = calc_payment(bal, ann_rate, rem)
            if ann_rate > prev_rate:
                # Rate rise - must increase payment
                cur_pmt = req
            else:
                # Rate fall
                if maintain_pmt:
                    cur_pmt = max(cur_pmt, req)
                else:
                    cur_pmt = req
            prev_rate = ann_rate

        # Enforce floor every period (for scenarios)
        effective_pmt = max(cur_pmt, min_pmt_floor) if min_pmt_floor > 0 else cur_pmt

        avg_off = min(offset_at(mo - 1), bal)
        net = max(0.0, bal - avg_off)
        dr = ann_rate / 100 / 365
        interest = net * dr * days
        saved = (bal * dr * days) - interest
        cum_int += interest
        cum_saved += saved

        opening = bal
        bal += interest
        pmt = min(effective_pmt, bal)
        bal = max(0.0, bal - pmt)
        cum_paid += pmt + monthly_fee

        # Extra repayment / redraw on this month
        extra = extra_map.get(mo, 0.0)
        if extra != 0:
            if extra > 0:
                # Extra repayment: reduce balance, cap at balance
                applied = min(extra, bal)
                bal -= applied
                cum_extra += applied
            else:
                # Redraw: increase balance
                bal += abs(extra)
                cum_extra += extra  # negative

        # Principal paid = scheduled payment less interest, plus any extra repayment this period
        principal_paid = max(0.0, pmt - interest + (extra if extra > 0 else 0))

        rows.append({
            "Month": mo, "Date": next_d,
            "Opening Balance": opening,
            "Avg Offset": avg_off, "Net Debt": net,
            "Rate %": ann_rate,
            "Interest": interest, "Interest Saved": saved,
            "Principal": principal_paid,
            "Extra Repayment": extra,
            "Fees": monthly_fee, "Payment": pmt,
            "Closing Balance": bal,
            "Cum Interest": cum_int, "Cum Paid": cum_paid,
            "Cum Interest Saved": cum_saved,
            "Cum Extra Repayment": cum_extra,
        })
        period = next_d

    return pd.DataFrame(rows)

def amortize(principal, start, term_mo, rate_sched, off_init, off_monthly, off_lumps_t,
             extra_repay_t, monthly_fee, maintain_pmt, min_pmt_floor=0.0) -> pd.DataFrame:
    rs_t = tuple((d.isoformat(), r) for d, r in rate_sched)
    return amortize_cached(principal, start.isoformat(), term_mo, rs_t,
                           off_init, off_monthly, off_lumps_t,
                           extra_repay_t,
                           monthly_fee, maintain_pmt, min_pmt_floor)

def fast_partial(principal, rate, total, n):
    if principal <= 0 or n <= 0: return 0.0, 0.0
    n = min(n, total)
    if rate <= 0:
        pmt = principal / total
        return max(0.0, principal - pmt * n), 0.0
    r = rate / 100 / 12
    pmt = principal * r * (1 + r) ** total / ((1 + r) ** total - 1)
    bal = principal * (1 + r) ** n - pmt * ((1 + r) ** n - 1) / r
    bal = max(0.0, bal)
    return bal, max(0.0, pmt * n - (principal - bal))

@st.cache_data(show_spinner=False)
def calc_optimal_split(loan, var_r, fix_r, rev_r, fix_yrs, total_mo):
    """0.1% increment sweep across 1001 splits. Objective computed over FIXED period only."""
    n_fix = min(fix_yrs * 12, total_mo)
    best_obj, best_pct = float("inf"), 50.0
    rows = []
    for i in range(1001):
        pct_f = i / 10.0
        p_f = loan * pct_f / 100
        p_v = loan * (100 - pct_f) / 100
        bf, cif = fast_partial(p_f, fix_r, total_mo, n_fix) if p_f > 0 else (0.0, 0.0)
        bv, civ = fast_partial(p_v, var_r, total_mo, n_fix) if p_v > 0 else (0.0, 0.0)
        obj = (cif + civ) + (bf + bv)
        if obj < best_obj:
            best_obj, best_pct = obj, pct_f
        rows.append({"Fixed %": pct_f, "Variable %": 100 - pct_f,
                     "Fixed Interest": cif, "Variable Interest": civ,
                     "Fixed Balance": bf, "Variable Balance": bv,
                     "Cum Interest": cif + civ,
                     "End Balance": bf + bv,
                     "Objective": obj})
    return best_pct, pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def comparison_rate_asic(setup_fee, monthly_fee, rate_pct):
    pv, n = 150_000.0, 300
    pmt = calc_payment(pv, rate_pct, n) + monthly_fee
    target = pv - setup_fee
    if not HAS_SCIPY or target <= 0: return rate_pct
    def f(i):
        if abs(i) < 1e-12: return pmt * n - target
        return pmt * (1 - (1 + i) ** -n) / i - target
    try: return brentq(f, 1e-8, 0.5) * 12 * 100
    except: return rate_pct

@st.cache_data(show_spinner=False)
def effective_rate_calc(loan, setup_fee, monthly_fee, rate_pct, term):
    if not HAS_SCIPY or loan <= 0: return rate_pct
    pmt = calc_payment(loan, rate_pct, term) + monthly_fee
    target = loan - setup_fee
    if target <= 0: return rate_pct
    def f(i):
        if abs(i) < 1e-12: return pmt * term - target
        return pmt * (1 - (1 + i) ** -term) / i - target
    try: return brentq(f, 1e-8, 0.5) * 12 * 100
    except: return rate_pct

def merge_schedules(df_v, df_f):
    if df_v is None or df_v.empty:
        return df_f.copy() if df_f is not None and not df_f.empty else pd.DataFrame()
    if df_f is None or df_f.empty: return df_v.copy()
    n = max(len(df_v), len(df_f))
    num_cols = ["Opening Balance", "Avg Offset", "Net Debt", "Interest", "Interest Saved",
                "Principal", "Extra Repayment", "Fees", "Payment", "Closing Balance",
                "Cum Interest", "Cum Paid", "Cum Interest Saved", "Cum Extra Repayment"]
    rows = []
    for i in range(n):
        rv = df_v.iloc[i].to_dict() if i < len(df_v) else None
        rf = df_f.iloc[i].to_dict() if i < len(df_f) else None
        if rv and rf:
            row = {"Month": rv["Month"], "Date": rv["Date"]}
            for c in num_cols:
                row[c] = rv.get(c, 0.0) + rf.get(c, 0.0)
            ob = rv.get("Opening Balance", 0) + rf.get("Opening Balance", 0)
            row["Rate %"] = ((rv["Rate %"] * rv.get("Opening Balance", 0)
                             + rf["Rate %"] * rf.get("Opening Balance", 0)) / ob
                            if ob > 0 else 0.0)
            rows.append(row)
        else:
            r = (rv or rf).copy()
            r["Month"] = i + 1
            rows.append(r)
    return pd.DataFrame(rows)

def eff_rate_from_deltas(base, deltas, as_of=None):
    r = base
    for row in sorted([(parse_dt(row[0]), float(row[1])) for row in deltas if row[0]],
                      key=lambda x: x[0]):
        if as_of is None or row[0] <= as_of:
            r = round(r + row[1], 4)
    return r

def deltas_to_lumps_t(lumps: list, loan_start: date) -> tuple:
    result = []
    for row in lumps:
        d = parse_dt(row[0])
        mo = months_between(loan_start, d)
        if mo >= 0:
            result.append((mo, float(row[1])))
    return tuple(sorted(result))

# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCHERS (cached)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_rba_rate_cached():
    try:
        r = requests.get("https://www.rba.gov.au/statistics/tables/csv/f1-data.csv",
                         headers=HDRS, timeout=10)
        if r.status_code == 200:
            lines = r.text.split("\n")
            tcol = None
            ds = None
            for li, line in enumerate(lines):
                if "Cash Rate Target" in line:
                    for ci, p in enumerate(line.split(",")):
                        if "Cash Rate Target" in p:
                            tcol = ci; break
                    ds = li + 1; break
            if tcol is not None:
                for line in reversed(lines[ds or 0:]):
                    parts = line.split(",")
                    if len(parts) > tcol:
                        try:
                            v = float(parts[tcol].strip().strip('"'))
                            if 0 < v < 30: return v
                        except: pass
    except: pass

    try:
        r = requests.get("https://www.rba.gov.au/statistics/cash-rate/",
                         headers=HDRS, timeout=10)
        if r.status_code == 200:
            for pat in [r"(\d+\.\d{2})\s*per cent", r"(\d+\.\d{2})%"]:
                for m in re.findall(pat, r.text, re.IGNORECASE):
                    try:
                        v = float(m)
                        if 0 < v < 20: return v
                    except: pass
    except: pass
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_rba_history():
    """
    Fetch full RBA cash rate target history.
    Returns list of {"date", "rate", "delta"} sorted ascending.
    Used by: Original loan auto-fill button; RBA history chart.
    """
    records = []

    # ── Attempt 1: F1 full CSV ─────────────────────────────────────────
    try:
        r = requests.get("https://www.rba.gov.au/statistics/tables/csv/f1-data.csv",
                         headers=HDRS, timeout=15)
        if r.status_code == 200:
            lines = r.text.split("\n")
            tcol = None
            ds = None
            for li, line in enumerate(lines):
                if "Cash Rate Target" in line:
                    parts = line.split(",")
                    for ci, p in enumerate(parts):
                        if "Cash Rate Target" in p:
                            tcol = ci; break
                    ds = li + 1; break
            if tcol is not None and ds is not None:
                prev_rate = None
                for line in lines[ds:]:
                    parts = line.split(",")
                    if len(parts) <= tcol: continue
                    try:
                        dstr = parts[0].strip().strip('"')
                        rstr = parts[tcol].strip().strip('"')
                        if not dstr or not rstr: continue
                        # RBA dates: try multiple formats
                        dt = None
                        for fmt in ["%d-%b-%Y", "%d/%m/%Y", "%Y-%m-%d", "%b-%Y", "%d %b %Y"]:
                            try:
                                dt = datetime.strptime(dstr, fmt).date(); break
                            except: pass
                        if dt is None: continue
                        rv = float(rstr)
                        if not (0 < rv < 30): continue
                        # Only record when rate changes
                        if prev_rate is None or abs(rv - prev_rate) > 0.001:
                            delta = round(rv - (prev_rate if prev_rate is not None else rv), 4)
                            records.append({"date": dt, "rate": rv, "delta": delta})
                            prev_rate = rv
                    except: pass
    except: pass

    # ── Attempt 2: cash-rate page HTML table ──────────────────────────
    if not records:
        try:
            r = requests.get("https://www.rba.gov.au/statistics/cash-rate/",
                             headers=HDRS, timeout=12)
            if r.status_code == 200:
                rows = re.findall(r'<tr[^>]*>(.*?)</tr>', r.text, re.DOTALL)
                prev = None
                parsed_rows = []
                for row in rows:
                    cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row, re.DOTALL)
                    cells = [re.sub(r'<[^>]+>', '', c).replace("&nbsp;", " ").strip() for c in cells]
                    if len(cells) >= 2:
                        try:
                            dt = None
                            for fmt in ["%d %b %Y", "%d %B %Y", "%d/%m/%Y"]:
                                try:
                                    dt = datetime.strptime(cells[0], fmt).date(); break
                                except: pass
                            if dt is None: continue
                            rv = float(cells[1].replace('%', '').strip())
                            if 0 < rv < 30:
                                parsed_rows.append((dt, rv))
                        except: pass
                # RBA page typically shows newest first — sort ascending
                parsed_rows.sort(key=lambda x: x[0])
                for dt, rv in parsed_rows:
                    if prev is None or abs(rv - prev) > 0.001:
                        delta = round(rv - (prev if prev is not None else rv), 4)
                        records.append({"date": dt, "rate": rv, "delta": delta})
                        prev = rv
        except: pass

    return sorted(records, key=lambda x: x["date"])

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_rba_next_meeting():
    """
    Scrape https://www.rba.gov.au/schedules-events/board-meeting-schedules.html
    using BeautifulSoup, extract the Meeting column from year tables, handle
    same-month ("5-6 May") and cross-month ("31 March - 1 April") ranges,
    and return the announcement (end) date of the next future meeting.
    """
    candidates = []
    browser_hdrs = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    # ── Primary: BeautifulSoup scrape of board-meeting-schedules.html ─────
    if HAS_BS4:
        try:
            r = requests.get(
                "https://www.rba.gov.au/schedules-events/board-meeting-schedules.html",
                headers=browser_hdrs, timeout=12)
            if r.status_code == 200:
                soup = BeautifulSoup(r.content, "html.parser")
                tables = soup.find_all("table")
                for table in tables:
                    # Determine year from summary, caption, or preceding header
                    summary = (table.get("summary", "") or "").lower()
                    cap = table.find("caption")
                    cap_txt = (cap.get_text() if cap else "").lower()
                    context = summary + " " + cap_txt
                    is_meeting_table = any(k in context for k in
                        ["monetary", "board meeting", "meeting dates"])
                    year = None
                    ym = re.search(r"(20\d{2})", context)
                    if ym: year = int(ym.group(1))
                    if not year or not is_meeting_table:
                        prev = table.find_previous(["h1", "h2", "h3", "h4"])
                        if prev:
                            ptxt = prev.get_text()
                            if re.search(r"meeting|monetary|board", ptxt, re.I):
                                is_meeting_table = True
                            ym = re.search(r"(20\d{2})", ptxt)
                            if ym and not year: year = int(ym.group(1))
                    if not year or not is_meeting_table: continue

                    # Find the Meeting column index from header row
                    rows = table.find_all("tr")
                    if not rows: continue
                    meeting_col = 1  # default
                    header_cells = rows[0].find_all(["th", "td"])
                    for ci, hc in enumerate(header_cells):
                        htxt = hc.get_text(strip=True).lower()
                        if htxt == "meeting" or htxt.startswith("meeting"):
                            meeting_col = ci; break

                    for row in rows[1:]:
                        cells = row.find_all(["td", "th"])
                        if len(cells) <= meeting_col: continue
                        text = cells[meeting_col].get_text(strip=True)
                        # Cross-month range: "31 March - 1 April"  → "1 April"
                        m = re.search(
                            r"\d{1,2}\s+[A-Z][a-z]+\s*[–\-−]\s*(\d{1,2})\s+([A-Z][a-z]+)",
                            text)
                        if m:
                            try:
                                dt = datetime.strptime(
                                    f"{m.group(1)} {m.group(2)} {year}",
                                    "%d %B %Y").date()
                                if dt >= TODAY: candidates.append(dt)
                                continue
                            except: pass
                        # Same-month range: "5-6 May" → "6 May"
                        m = re.search(
                            r"(\d{1,2})\s*[–\-−]\s*(\d{1,2})\s+([A-Z][a-z]+)", text)
                        if m:
                            try:
                                dt = datetime.strptime(
                                    f"{m.group(2)} {m.group(3)} {year}",
                                    "%d %B %Y").date()
                                if dt >= TODAY: candidates.append(dt)
                                continue
                            except: pass
                        # Single date: "6 May"
                        m = re.match(r"^\s*(\d{1,2})\s+([A-Z][a-z]+)\s*$", text)
                        if m:
                            try:
                                dt = datetime.strptime(
                                    f"{m.group(1)} {m.group(2)} {year}",
                                    "%d %B %Y").date()
                                if dt >= TODAY: candidates.append(dt)
                            except: pass
        except Exception: pass

    # ── Fallback: ICS calendar feed ───────────────────────────────────────
    if not candidates:
        for url in [
            "https://www.rba.gov.au/schedules-events/calendar.ics",
            "https://www.rba.gov.au/schedules-events/calendar/"
            "?topics=monetary-policy-board&view=list&format=ics",
        ]:
            try:
                r = requests.get(url, headers=browser_hdrs, timeout=10)
                if r.status_code == 200 and "BEGIN:VEVENT" in r.text:
                    for e in re.findall(r"BEGIN:VEVENT(.*?)END:VEVENT",
                                         r.text, re.DOTALL):
                        sm = re.search(r"SUMMARY:(.+)", e)
                        if sm and ("monetary" in sm.group(1).lower() or
                                   "board" in sm.group(1).lower()):
                            dm = re.search(r"DTSTART[^:]*:(\d{8})", e)
                            if dm:
                                try:
                                    dt = datetime.strptime(dm.group(1), "%Y%m%d").date()
                                    if dt >= TODAY: candidates.append(dt)
                                except: pass
            except: pass

    return min(candidates).strftime("%d %B %Y") if candidates else None

# ASX month codes for 30-day Interbank Cash Rate Futures
ASX_MONTH_CODES = {2: "G", 3: "H", 5: "K", 6: "M",
                   8: "Q", 9: "U", 11: "X", 12: "Z"}

def _asx_ticker_for_meeting(meeting_date: date) -> str | None:
    """Return '.AX' ticker for the IB futures contract covering the meeting month.
       RBA only meets in Feb/Mar/May/Jun/Aug/Sep/Nov/Dec, which map 1-to-1 to
       ASX IB month codes. Format: IB<letter><YY>.AX e.g. IBK26.AX for May 2026."""
    letter = ASX_MONTH_CODES.get(meeting_date.month)
    if not letter:
        # Pick next available contract month
        for m in sorted(ASX_MONTH_CODES):
            if m > meeting_date.month:
                letter = ASX_MONTH_CODES[m]; break
        if not letter:
            letter = ASX_MONTH_CODES[min(ASX_MONTH_CODES)]
    return f"IB{letter}{meeting_date.year % 100:02d}.AX"

@st.cache_data(ttl=600, show_spinner=False)
def fetch_asx_rba_data(meeting_iso: str | None = None):
    """
    Fetch ASX 30-day Interbank Cash Rate Futures price for the contract
    covering the next RBA meeting month, via yfinance. Falls back to
    rba.isaacgross.net if yfinance unavailable or request fails.
    """
    result = {}
    meeting_date = None
    if meeting_iso:
        try: meeting_date = date.fromisoformat(meeting_iso)
        except: pass

    # ── Primary: yfinance ─────────────────────────────────────────────────
    if HAS_YF and meeting_date:
        ticker = _asx_ticker_for_meeting(meeting_date)
        if ticker:
            try:
                hist = yf.Ticker(ticker).history(period="5d")
                if not hist.empty:
                    price = float(hist["Close"].iloc[-1])
                    if 90 < price < 100:
                        result = {
                            "ticker": ticker,
                            "futures_price": round(price, 4),
                            "implied_yield": round(100 - price, 4),
                            "data_date": hist.index[-1].strftime("%Y-%m-%d"),
                            "source": "yfinance",
                        }
            except Exception as e:
                result["yfinance_error"] = str(e)[:120]

    # ── Fallback: rba.isaacgross.net ─────────────────────────────────────
    if "futures_price" not in result:
        ig_base = "https://rba.isaacgross.net"
        try:
            r = requests.get(ig_base, headers=HDRS, timeout=8)
            if r.status_code == 200:
                for pat in [r'"futures_price"\s*:\s*([\d.]+)',
                            r'"ib_price"\s*:\s*([\d.]+)',
                            r'"price"\s*:\s*([\d.]+)',
                            r'(9[4-9]\.\d{2})']:
                    mm = re.search(pat, r.text)
                    if mm:
                        try:
                            v = float(mm.group(1))
                            if 90 < v < 100:
                                result.update({
                                    "futures_price": v,
                                    "implied_yield": round(100 - v, 4),
                                    "source": "isaacgross",
                                })
                                break
                        except: pass
                for pat in [r'"probability"\s*:\s*([\d.]+)',
                            r'"prob"\s*:\s*([\d.]+)']:
                    mm = re.search(pat, r.text)
                    if mm:
                        try: result["probability"] = float(mm.group(1)); break
                        except: pass
        except: pass

    return result

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

def _d(k, v):
    if k not in st.session_state: st.session_state[k] = v

def init_state():
    # ── Original Loan ──
    _d("o_prop_val", 800_000.0); _d("o_prop_date", date(2020, 1, 15))
    _d("o_loan_amt", 640_000.0); _d("o_loan_date", date(2020, 1, 15))
    _d("o_use_dates", False); _d("o_end_date", date(2045, 1, 15))
    _d("o_term_mo", 300); _d("o_balance", 580_000.0)
    _d("o_balance_date", TODAY); _d("o_rate", 6.50)
    _d("o_rate_deltas", [])
    _d("o_off_init", 0.0); _d("o_off_date", TODAY)
    _d("o_off_monthly", 0.0); _d("o_off_lumps", [])
    _d("o_extra_repay", [])  # NEW
    _d("o_fee_mo", 10.0); _d("o_fee_setup", 0.0)
    _d("o_fee_break", 0.0); _d("o_fee_other", 0.0)

    # ── Current Loan ──
    _d("c_is_cont", True); _d("c_prop_val", 800_000.0); _d("c_prop_date", TODAY)
    _d("c_balance", 580_000.0); _d("c_rate", 6.50)
    _d("c_use_dates", False); _d("c_end_date", date(2045, 1, 15)); _d("c_term_mo", 300)
    _d("c_rate_deltas", [])
    _d("c_off_init", 0.0); _d("c_off_date", TODAY)
    _d("c_off_monthly", 0.0); _d("c_off_lumps", [])
    _d("c_extra_repay", [])  # NEW
    _d("c_fee_mo", 10.0); _d("c_fee_setup", 0.0); _d("c_fee_other", 0.0)

    # ── Shared anticipated/future rate changes ──
    _d("future_var_deltas", [])

    # ── Proposed Loan ──
    _d("p_auto_amount", True); _d("p_loan_amt", 580_000.0)
    _d("p_start_date", TODAY); _d("p_use_dates", False)
    _d("p_end_date", add_months(TODAY, 300)); _d("p_term_mo", 300)
    _d("p_adv_var_rate", 6.20); _d("p_adv_fix_rate", 5.89)
    _d("p_fix_yrs", 3)
    _d("p_rev_rate_override", False); _d("p_rev_rate", 6.20)
    _d("p_split_auto", True); _d("p_split_pct", 50.0)

    # Individualised fees — variable
    _d("p_var_fee_mo", 10.0); _d("p_var_fee_setup", 800.0)
    _d("p_var_fee_break", 0.0); _d("p_var_fee_other", 0.0)
    # Individualised fees — fixed
    _d("p_fix_fee_mo", 10.0); _d("p_fix_fee_setup", 800.0)
    _d("p_fix_fee_break", 0.0); _d("p_fix_fee_other", 0.0)
    # Match toggle
    _d("p_fees_match", True)

    # Offset
    _d("p_off_match", False)  # match to current loan offset
    _d("p_off_init", 0.0); _d("p_off_date", TODAY)
    _d("p_off_monthly", 0.0); _d("p_off_lumps", [])
    _d("p_extra_repay", [])  # NEW

    # ── Strategy ──
    _d("strategy", "Balanced (Optimal Split)")
    _d("maintain_pmt", True); _d("rba_bps", 0)

    # ── Live data ──
    _d("_rba_rate", None); _d("_rba_history", [])
    _d("_rba_next_meeting", None); _d("_asx_data", {})
    _d("_data_loaded", False)
    _d("_rba_fetch_status", "")

def load_live_data():
    ss = st.session_state
    if ss._data_loaded: return
    with st.spinner("Loading live RBA and market data..."):
        ss._rba_rate = fetch_rba_rate_cached()
        ss._rba_history = fetch_rba_history()
        ss._rba_next_meeting = fetch_rba_next_meeting()
        # Derive ISO date of next meeting for the ASX fetcher
        meeting_iso = None
        if ss._rba_next_meeting:
            try:
                meeting_iso = datetime.strptime(
                    ss._rba_next_meeting, "%d %B %Y").date().isoformat()
            except: pass
        ss._asx_data = fetch_asx_rba_data(meeting_iso)
        # Status flag for UI feedback
        if ss._rba_history:
            ss._rba_fetch_status = f"✓ Loaded {len(ss._rba_history)} RBA rate changes"
        else:
            ss._rba_fetch_status = "⚠ Could not retrieve RBA history — network or format issue"
    ss._data_loaded = True

# ═══════════════════════════════════════════════════════════════════════════════
# DYNAMIC LIST WIDGETS
# ═══════════════════════════════════════════════════════════════════════════════

def rate_delta_list(state_key: str, base_rate: float, title: str,
                    max_rows: int = 20, show_autofill: bool = False,
                    autofill_data: list = None, autofill_info: str = ""):
    ss = st.session_state
    if state_key not in ss: ss[state_key] = []

    sec(title)

    if show_autofill:
        cols = st.columns([1, 1, 1.5, 2])
    else:
        cols = st.columns([1, 1, 5])
    with cols[0]:
        if st.button("Add change", key=f"add_{state_key}") and len(ss[state_key]) < max_rows:
            ss[state_key].append([TODAY, 0.25])
    with cols[1]:
        if ss[state_key] and st.button("Clear", key=f"clr_{state_key}"):
            ss[state_key] = []; st.rerun()

    if show_autofill:
        with cols[2]:
            avail = len(autofill_data) if autofill_data else 0
            disabled = avail == 0
            btn_label = f"Auto-fill from RBA ({avail})" if avail else "Auto-fill from RBA (none available)"
            if st.button(btn_label, key=f"auto_{state_key}", disabled=disabled,
                         help="Populate rate change list from RBA F1 Cash Rate Target history. "
                              "Only changes AFTER loan start date are included."):
                ss[state_key] = [[r["date"], r["delta"]] for r in autofill_data if r["delta"] != 0]
                st.rerun()
        with cols[3]:
            if autofill_info:
                st.markdown(f'<div style="padding:8px 4px;color:#64748b;font-size:0.75rem">'
                            f'{autofill_info}</div>', unsafe_allow_html=True)

    if ss[state_key]:
        h = st.columns([2, 1.6, 2.2, 0.5])
        for i, lbl in enumerate(["Effective Date", "Change (±%)", "Result"]):
            h[i].markdown(f'<div class="list-hdr">{lbl}</div>', unsafe_allow_html=True)
        cum = base_rate
        to_del = None
        for i, row in enumerate(ss[state_key]):
            c1, c2, c3, c4 = st.columns([2, 1.6, 2.2, 0.5])
            with c1:
                nd = st.date_input(f"Rate date {i+1}", value=parse_dt(row[0]),
                                   key=f"{state_key}_d_{i}", label_visibility="collapsed")
            with c2:
                nv = st.number_input(f"Rate delta {i+1}", value=float(row[1]),
                                     min_value=-10.0, max_value=10.0, step=0.25, format="%.2f",
                                     key=f"{state_key}_v_{i}", label_visibility="collapsed")
            res = round(cum + nv, 4)
            with c3:
                sign = "+" if nv >= 0 else ""
                clr = "#e94560" if nv > 0 else ("#30d996" if nv < 0 else "#64748b")
                st.markdown(
                    f'<div style="padding:8px 3px;color:{clr};font-size:.875rem;font-weight:500">'
                    f'{sign}{nv:.2f}% → <strong>{res:.2f}%</strong></div>',
                    unsafe_allow_html=True)
            with c4:
                if st.button("✕", key=f"{state_key}_del_{i}"): to_del = i
            ss[state_key][i] = [nd, nv]
            cum = res
        if to_del is not None:
            ss[state_key].pop(to_del); st.rerun()

def lump_list(state_key: str, title: str, max_rows: int = 100):
    ss = st.session_state
    if state_key not in ss: ss[state_key] = []
    sec(title)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Add lump sum", key=f"add_{state_key}") and len(ss[state_key]) < max_rows:
            ss[state_key].append([TODAY, 0.0])
    with c2:
        if ss[state_key] and st.button("Clear", key=f"clr_{state_key}"):
            ss[state_key] = []; st.rerun()
    if ss[state_key]:
        h = st.columns([2, 2, 0.5])
        h[0].markdown('<div class="list-hdr">Date</div>', unsafe_allow_html=True)
        h[1].markdown('<div class="list-hdr">Amount ($)</div>', unsafe_allow_html=True)
        to_del = None
        for i, row in enumerate(ss[state_key]):
            c1, c2, c3 = st.columns([2, 2, 0.5])
            with c1:
                nd = st.date_input(f"Lump date {i+1}", value=parse_dt(row[0]),
                                   key=f"{state_key}_d_{i}", label_visibility="collapsed")
            with c2:
                na = st.number_input(f"Lump amt {i+1}", value=float(row[1]), step=1_000.0,
                                     key=f"{state_key}_a_{i}", label_visibility="collapsed")
            with c3:
                if st.button("✕", key=f"{state_key}_del_{i}"): to_del = i
            ss[state_key][i] = [nd, na]
        if to_del is not None:
            ss[state_key].pop(to_del); st.rerun()

def extra_repay_list(state_key: str, title: str, max_rows: int = 100):
    """Extra repayments (positive) and redraws (negative) with dates."""
    ss = st.session_state
    if state_key not in ss: ss[state_key] = []
    sec(title)
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("Add extra repayment", key=f"add_ex_{state_key}") and len(ss[state_key]) < max_rows:
            ss[state_key].append([TODAY, 1_000.0])
    with c2:
        if st.button("Add redraw", key=f"add_rd_{state_key}") and len(ss[state_key]) < max_rows:
            ss[state_key].append([TODAY, -1_000.0])
    with c3:
        if ss[state_key] and st.button("Clear", key=f"clr_{state_key}"):
            ss[state_key] = []; st.rerun()
    if ss[state_key]:
        st.markdown(
            '<div class="note">Positive amounts are extra repayments (reduce balance). '
            'Negative amounts are redraws (increase balance).</div>',
            unsafe_allow_html=True)
        h = st.columns([2, 2, 1.5, 0.5])
        h[0].markdown('<div class="list-hdr">Date</div>', unsafe_allow_html=True)
        h[1].markdown('<div class="list-hdr">Amount ($)</div>', unsafe_allow_html=True)
        h[2].markdown('<div class="list-hdr">Type</div>', unsafe_allow_html=True)
        to_del = None
        for i, row in enumerate(ss[state_key]):
            c1, c2, c3, c4 = st.columns([2, 2, 1.5, 0.5])
            with c1:
                nd = st.date_input(f"Extra date {i+1}", value=parse_dt(row[0]),
                                   key=f"{state_key}_d_{i}", label_visibility="collapsed")
            with c2:
                na = st.number_input(f"Extra amt {i+1}", value=float(row[1]),
                                     step=500.0, key=f"{state_key}_a_{i}",
                                     label_visibility="collapsed")
            with c3:
                typ = "Repayment" if na >= 0 else "Redraw"
                clr = "#30d996" if na >= 0 else "#f5a94a"
                st.markdown(f'<div style="padding:8px 3px;color:{clr};font-weight:500;'
                            f'font-size:0.85rem">{typ}</div>',
                            unsafe_allow_html=True)
            with c4:
                if st.button("✕", key=f"{state_key}_del_{i}"): to_del = i
            ss[state_key][i] = [nd, na]
        if to_del is not None:
            ss[state_key].pop(to_del); st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# RBA / ASX PANEL
# ═══════════════════════════════════════════════════════════════════════════════

def render_rba_panel():
    """Render RBA Cash Rate and Market Indicators inline (no own expander).
       Called from section_scenarios(). History chart omitted per user request
       (data still used in background for Original Loan auto-fill)."""
    ss = st.session_state
    rate = ss._rba_rate
    meeting = ss._rba_next_meeting
    asx = ss._asx_data

    st.markdown(
        '<div class="sec-title">RBA Cash Rate and Market Indicators</div>',
        unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="data-panel">'
                    '<div class="data-panel-title">Current RBA Cash Rate</div>',
                    unsafe_allow_html=True)
        if rate:
            st.markdown(f'<div style="font-size:2rem;font-weight:700;color:#4a9af5">'
                        f'{fp(rate)}</div>', unsafe_allow_html=True)
            history = ss._rba_history
            if len(history) >= 2:
                last = history[-1]
                d_str = last["date"].strftime("%d %b %Y")
                if last["delta"] > 0:
                    st.markdown(f'<div class="rate-up">▲ +{last["delta"]:.2f}% on {d_str}</div>',
                                unsafe_allow_html=True)
                elif last["delta"] < 0:
                    st.markdown(f'<div class="rate-dn">▼ {last["delta"]:.2f}% on {d_str}</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="rate-nc">Unchanged as of {d_str}</div>',
                                unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#64748b">Unable to fetch — see rba.gov.au</div>',
                        unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="data-panel">'
                    '<div class="data-panel-title">Next Monetary Policy Decision</div>',
                    unsafe_allow_html=True)
        if meeting:
            st.markdown(f'<div style="font-size:1.1rem;font-weight:600;color:#d4dbe8">'
                        f'{meeting}</div>', unsafe_allow_html=True)
            try:
                dt = datetime.strptime(meeting, "%d %B %Y").date()
                days = (dt - TODAY).days
                if days > 0:
                    st.markdown(f'<div style="color:#64748b;font-size:0.78rem">'
                                f'{days} days from today</div>', unsafe_allow_html=True)
            except: pass
        else:
            st.markdown(
                '<div style="color:#64748b;font-size:0.78rem">Unable to parse. Visit '
                '<a href="https://www.rba.gov.au/schedules-events/board-meeting-schedules.html" '
                'target="_blank" style="color:#4a9af5">RBA Board Meeting Schedules</a></div>',
                unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="data-panel">'
                    '<div class="data-panel-title">ASX Rate Tracker (IB Futures)</div>',
                    unsafe_allow_html=True)
        if asx and "futures_price" in asx:
            fp_val = asx["futures_price"]
            iy_val = asx.get("implied_yield", round(100 - fp_val, 4))
            ticker = asx.get("ticker", "")
            src = asx.get("source", "")
            data_date = asx.get("data_date", "")
            st.markdown(
                f'<div style="color:#30d996;font-size:0.82rem;font-weight:600">'
                f'{ticker} → {fp_val:.2f}</div>'
                f'<div style="color:#4a9af5;font-size:1.1rem;font-weight:700">'
                f'Implied Rate: {iy_val:.2f}%</div>'
                f'<div style="color:#64748b;font-size:0.7rem">'
                f'Source: {src}{" · " + data_date if data_date else ""}</div>',
                unsafe_allow_html=True)
            if "probability" in asx:
                prob = asx["probability"] * 100 if asx["probability"] <= 1 else asx["probability"]
                st.markdown(
                    f'<div style="color:#f5a94a;font-size:0.85rem;margin-top:4px">'
                    f'Rate change probability: <strong>{prob:.1f}%</strong></div>',
                    unsafe_allow_html=True)
        else:
            err = asx.get("yfinance_error", "") if asx else ""
            err_line = (f'<div style="color:#64748b;font-size:0.68rem;margin-top:3px">'
                       f'yfinance: {err}</div>') if err else ""
            st.markdown(
                f'<div style="color:#64748b;font-size:0.78rem">Auto-fetch unavailable — '
                f'enter IB Futures Price manually below. '
                f'<a href="https://www.asx.com.au/markets/trade-our-derivatives-market/'
                f'futures-market/rba-rate-tracker" target="_blank" '
                f'style="color:#4a9af5">ASX Rate Tracker</a></div>{err_line}',
                unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Probability Calculator
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<div style="color:#64748b;font-size:0.72rem;font-weight:600;text-transform:uppercase;'
        'letter-spacing:.06em;margin-bottom:10px">ASX Target Rate Probability '
        '(30-Day Interbank Cash Rate Futures)</div>',
        unsafe_allow_html=True)

    # Default IB price = live value if available, else 95.65
    _def_ib = float(asx.get("futures_price", 95.65)) if asx else 95.65
    # Determine default days-before-meeting
    _def_days = 5
    if meeting:
        try:
            dt = datetime.strptime(meeting, "%d %B %Y").date()
            _def_days = max(1, min(30, dt.day))
        except: pass

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ib_price = st.number_input("IB Futures Price", value=_def_ib,
                                   min_value=90.0, max_value=100.0,
                                   step=0.01, format="%.2f", key="rba_ib_price",
                                   help="30-Day Interbank Cash Rate Futures price "
                                        "(e.g. 95.65 → implied 4.35%).")
        implied_yield = round(100 - ib_price, 4)
        computed("Implied Yield", fp(implied_yield), "= 100 − Futures Price")
    with c2:
        rt = st.number_input("Current Target Rate (%)",
                             value=float(rate) if rate else 4.35,
                             min_value=0.0, max_value=30.0, step=0.25,
                             format="%.2f", key="rba_rt",
                             help="Current RBA Target Cash Rate.")
    with c3:
        rt1 = st.number_input("Expected New Rate (%)",
                              value=round((rate or 4.35) - 0.25, 2),
                              min_value=0.0, max_value=30.0, step=0.25,
                              format="%.2f", key="rba_rt1",
                              help="Expected rate if RBA moves (typically ±0.25%).")
    with c4:
        nb_days = st.number_input("Day of month of meeting", value=_def_days,
                                  min_value=1, max_value=30, key="rba_nb_days",
                                  help="Day of the month on which the RBA Board meets.")
    try:
        # Use calendar.monthrange for accurate days in meeting month if possible
        days_in_month = 30
        meeting_date = None
        if meeting:
            try:
                meeting_date = datetime.strptime(meeting, "%d %B %Y").date()
                days_in_month = calendar.monthrange(meeting_date.year,
                                                    meeting_date.month)[1]
            except: pass
        # Per ASX formula: nb = days before meeting / month length; na = rest
        nb = (nb_days - 1) / days_in_month
        na = (days_in_month - nb_days + 1) / days_in_month
        X = implied_yield
        denom = na * (rt1 - rt)
        if abs(denom) > 1e-9:
            p = (X - rt * nb - rt * na) / denom
            p = max(0.0, min(1.0, p))
            pct = p * 100
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                trend_clr = "#e94560" if rt1 > rt else "#30d996"
                direction = "increase" if rt1 > rt else "decrease"
                st.markdown(
                    f'<div class="cf"><div class="cf-lbl">Probability of rate {direction}</div>'
                    f'<div class="cf-val" style="color:{trend_clr}">{pct:.1f}%</div>'
                    f'<div class="cf-sub">p = (X − rt·nb − rt·na) / (na·(r(t+1) − rt))</div></div>',
                    unsafe_allow_html=True)
            with col_b:
                outlook = ("Increasing" if (pct > 60 and rt1 > rt)
                          else ("Decreasing" if (pct > 60 and rt1 < rt)
                                else "Stable / Uncertain"))
                oclr = {"Increasing": "#e94560", "Decreasing": "#30d996",
                       "Stable / Uncertain": "#64748b"}[outlook]
                st.markdown(
                    f'<div class="cf"><div class="cf-lbl">Rate Outlook</div>'
                    f'<div class="cf-val" style="color:{oclr}">{outlook}</div></div>',
                    unsafe_allow_html=True)
            with col_c:
                st.markdown(
                    f'<div class="cf"><div class="cf-lbl">nb / na (fraction of month)</div>'
                    f'<div class="cf-val">{nb:.2f} / {na:.2f}</div>'
                    f'<div class="cf-sub">{days_in_month} days in meeting month</div></div>',
                    unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Calculation error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# INPUT: ORIGINAL LOAN
# ═══════════════════════════════════════════════════════════════════════════════

def section_original():
    ss = st.session_state
    loan_start = parse_dt(ss.o_loan_date)
    rba_after_start = [r for r in ss._rba_history
                        if r["date"] > loan_start and abs(r["delta"]) > 0.001]

    sec("Property")
    c1, c2, c3 = st.columns(3)
    with c1:
        ss.o_prop_val = st.number_input("Property Valuation ($)", value=ss.o_prop_val,
            min_value=0.0, step=10_000.0, key="w_o_pv",
            help="Assessed value of the property at time of loan origination.")
    with c2:
        ss.o_prop_date = st.date_input("Valuation Date", value=ss.o_prop_date, key="w_o_pd")
    with c3:
        if ss.o_prop_val > 0 and ss.o_loan_amt > 0:
            computed("Original LVR", fp(ss.o_loan_amt / ss.o_prop_val * 100),
                     "Original Loan Amount ÷ Original Property Valuation")
        else:
            computed("Original LVR", "—")

    sec("Loan Details")
    c1, c2, c3 = st.columns(3)
    with c1:
        ss.o_loan_amt = st.number_input("Original Loan Amount ($)", value=ss.o_loan_amt,
            min_value=0.0, step=10_000.0, key="w_o_la")
    with c2:
        ss.o_balance = st.number_input("Remaining Balance ($)", value=ss.o_balance,
            min_value=0.0, step=1_000.0, key="w_o_bal",
            help="Outstanding principal as at the balance date below.")
    with c3:
        ss.o_balance_date = st.date_input("Balance As At", value=ss.o_balance_date, key="w_o_bd")
    c1, c2 = st.columns(2)
    with c1:
        ss.o_rate = st.number_input("Original Interest Rate (% p.a.)", value=ss.o_rate,
            min_value=0.0, max_value=30.0, step=0.01, format="%.4f", key="w_o_r",
            help="Interest rate at loan inception.")
    with c2:
        ss.o_loan_date = st.date_input("Loan Start Date", value=ss.o_loan_date, key="w_o_ld",
            help="Used to filter historical RBA rate changes relevant to this loan.")

    sec("Loan Term")
    ss.o_use_dates = st.toggle("Calculate term from start and end dates",
                                value=ss.o_use_dates, key="w_o_ud")
    if ss.o_use_dates:
        c1, c2, c3 = st.columns(3)
        with c1:
            ss.o_end_date = st.date_input("Loan End Date", value=ss.o_end_date, key="w_o_ed")
        with c2:
            if ss.o_end_date > ss.o_loan_date:
                ss.o_term_mo = months_between(ss.o_loan_date, ss.o_end_date)
                computed("Calculated Term", f"{ss.o_term_mo} months", f"{ss.o_term_mo/12:.1f} years")
        with c3:
            rem = max(0, months_between(TODAY, ss.o_end_date)) if ss.o_end_date > TODAY else 0
            computed("Remaining Term", f"{rem} months", f"{rem/12:.1f} years from today")
    else:
        c1, c2 = st.columns(2)
        with c1:
            ss.o_term_mo = st.number_input("Loan Term (months)", value=ss.o_term_mo,
                min_value=1, max_value=600, step=12, key="w_o_tm")
        with c2:
            computed("Equivalent", f"{ss.o_term_mo/12:.1f} years")

    # Historical Rate Changes with improved autofill
    info = ""
    if rba_after_start:
        first_d = rba_after_start[0]["date"].strftime("%b %Y")
        last_d = rba_after_start[-1]["date"].strftime("%b %Y")
        info = f"{len(rba_after_start)} RBA changes since loan start ({first_d} – {last_d})"
    elif not ss._rba_history:
        info = ss._rba_fetch_status
    else:
        info = "No RBA changes after loan start date"

    rate_delta_list("o_rate_deltas", ss.o_rate,
                    "Historical Rate Changes (± from previous rate)",
                    show_autofill=True, autofill_data=rba_after_start,
                    autofill_info=info)

    sec("Offset Account")
    c1, c2, c3 = st.columns(3)
    with c1:
        ss.o_off_init = st.number_input("Current Balance ($)", value=ss.o_off_init,
            min_value=0.0, step=1_000.0, key="w_o_oi",
            help="Balance in the offset account. Reduces the interest-bearing principal daily.")
    with c2:
        ss.o_off_date = st.date_input("Offset Start Date", value=ss.o_off_date, key="w_o_od")
    with c3:
        ss.o_off_monthly = st.number_input("Monthly Addition ($)", value=ss.o_off_monthly,
            min_value=0.0, step=100.0, key="w_o_om")
    lump_list("o_off_lumps", "Offset Lump Sum Deposits")

    # NEW: Extra Repayments and Redraws
    extra_repay_list("o_extra_repay", "Extra Repayments and Redraws")

    sec("Fees")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ss.o_fee_mo = st.number_input("Monthly Fee ($)", value=ss.o_fee_mo,
            min_value=0.0, step=1.0, key="w_o_fm")
    with c2:
        ss.o_fee_setup = st.number_input("Setup / Establishment Fee ($)", value=ss.o_fee_setup,
            min_value=0.0, step=100.0, key="w_o_fs")
    with c3:
        ss.o_fee_break = st.number_input("Breakage Fee ($)", value=ss.o_fee_break,
            min_value=0.0, step=100.0, key="w_o_fb")
    with c4:
        ss.o_fee_other = st.number_input("Other One-off Fee ($)", value=ss.o_fee_other,
            min_value=0.0, step=100.0, key="w_o_fo")

# ═══════════════════════════════════════════════════════════════════════════════
# INPUT: CURRENT LOAN
# ═══════════════════════════════════════════════════════════════════════════════

def section_current():
    ss = st.session_state
    ss.c_is_cont = st.toggle(
        "Treat as continuation of original loan", value=ss.c_is_cont, key="w_c_ic",
        help="When enabled, Current Remaining Balance, Current Interest Rate, and Monthly Fee "
             "are auto-filled from the Original Loan. Setup fee is disabled for continuation.")

    sec("Property")
    c1, c2, c3 = st.columns(3)
    prop_val_key = "w_c_pv" if ss.c_is_cont else "w_c_pv2"
    with c1:
        ss.c_prop_val = st.number_input("Current Property Valuation ($)", value=ss.c_prop_val,
            min_value=0.0, step=10_000.0, key=prop_val_key,
            help="Current market value (may differ from original if market moved).")
    with c2:
        ss.c_prop_date = st.date_input("Valuation Date", value=ss.c_prop_date,
            key="w_c_pd" if ss.c_is_cont else "w_c_pd2")
    with c3:
        bal_for_lvr = ss.o_balance if ss.c_is_cont else ss.c_balance
        if ss.c_prop_val > 0 and bal_for_lvr > 0:
            computed("Current LVR", fp(bal_for_lvr / ss.c_prop_val * 100),
                     "Current Remaining Balance ÷ Current Property Valuation")
        else:
            computed("Current LVR", "—")

    if ss.c_is_cont:
        latest_r = eff_rate_from_deltas(ss.o_rate, ss.o_rate_deltas)
        rem = max(0, ss.o_term_mo - months_between(ss.o_balance_date, TODAY))
        sec("Auto-filled from Original Loan")
        a1, a2, a3 = st.columns(3)
        with a1: computed("Current Remaining Balance", fc(ss.o_balance), "as at today")
        with a2: computed("Current Interest Rate", fp(latest_r), "after all original rate changes")
        with a3: computed("Remaining Term", f"{rem} months", f"{rem/12:.1f} years")

        sec("Fees (auto-filled from Original Loan)")
        c1, c2 = st.columns(2)
        with c1:
            ss.c_fee_mo = ss.o_fee_mo
            computed("Monthly Fee (auto-filled)", fc(ss.c_fee_mo), "inherited from original loan")
        with c2:
            ss.c_fee_setup = 0.0
            computed("Setup Fee", "$0", "does not apply for continuation")

    else:
        sec("Current Loan Details")
        c1, c2, c3 = st.columns(3)
        with c1:
            ss.c_balance = st.number_input("Current Remaining Balance ($)", value=ss.c_balance,
                min_value=0.0, step=1_000.0, key="w_c_bal",
                help="Outstanding principal as at today.")
        with c2:
            computed("Balance Date", TODAY.strftime("%d %b %Y"))
        with c3:
            ss.c_rate = st.number_input("Current Interest Rate (% p.a.)", value=ss.c_rate,
                min_value=0.0, max_value=30.0, step=0.01, format="%.4f", key="w_c_r")

        sec("Remaining Term")
        ss.c_use_dates = st.toggle("Calculate from loan end date",
                                    value=ss.c_use_dates, key="w_c_ud")
        if ss.c_use_dates:
            c1, c2 = st.columns(2)
            with c1:
                ss.c_end_date = st.date_input("Loan End Date", value=ss.c_end_date, key="w_c_ed")
            with c2:
                if ss.c_end_date > TODAY:
                    ss.c_term_mo = months_between(TODAY, ss.c_end_date)
                    computed("Remaining Term", f"{ss.c_term_mo} months",
                             f"{ss.c_term_mo/12:.1f} years")
        else:
            c1, c2 = st.columns(2)
            with c1:
                ss.c_term_mo = st.number_input("Remaining Term (months)", value=ss.c_term_mo,
                    min_value=1, max_value=600, step=12, key="w_c_tm")
            with c2:
                computed("Equivalent", f"{ss.c_term_mo/12:.1f} years")

        sec("Fees")
        c1, c2, c3 = st.columns(3)
        with c1:
            ss.c_fee_mo = st.number_input("Monthly Fee ($)", value=ss.c_fee_mo,
                min_value=0.0, step=1.0, key="w_c_fm")
        with c2:
            ss.c_fee_setup = st.number_input("Setup Fee ($)", value=ss.c_fee_setup,
                min_value=0.0, step=100.0, key="w_c_fs")
        with c3:
            ss.c_fee_other = st.number_input("Other Fee ($)", value=ss.c_fee_other,
                min_value=0.0, step=100.0, key="w_c_fo")

    # Shared anticipated rate changes
    rate_delta_list(
        "future_var_deltas",
        eff_rate_from_deltas(ss.o_rate, ss.o_rate_deltas) if ss.c_is_cont else ss.c_rate,
        "Anticipated Rate Changes (shared with Proposed Variable rate)", max_rows=20)
    st.markdown(
        '<div class="note">These anticipated changes apply to both Current Loan and '
        'Proposed Variable component simultaneously.</div>',
        unsafe_allow_html=True)

    sec("Offset Account")
    c1, c2, c3 = st.columns(3)
    with c1:
        ss.c_off_init = st.number_input("Current Balance ($)", value=ss.c_off_init,
            min_value=0.0, step=1_000.0, key="w_c_oi")
    with c2:
        ss.c_off_date = st.date_input("Offset Start Date", value=ss.c_off_date, key="w_c_od")
    with c3:
        ss.c_off_monthly = st.number_input("Monthly Addition ($)", value=ss.c_off_monthly,
            min_value=0.0, step=100.0, key="w_c_om")
    lump_list("c_off_lumps", "Offset Lump Sum Deposits")

    extra_repay_list("c_extra_repay", "Extra Repayments and Redraws")

# ═══════════════════════════════════════════════════════════════════════════════
# INPUT: PROPOSED LOAN
# ═══════════════════════════════════════════════════════════════════════════════

def section_proposed():
    ss = st.session_state
    curr_balance = ss.o_balance if ss.c_is_cont else ss.c_balance

    # ── Loan Amount and Term ──────────────────────────────────────────────
    sec("Loan Amount and Term")
    ss.p_auto_amount = st.toggle(
        "Auto-fill loan amount from current remaining balance",
        value=ss.p_auto_amount, key="w_p_aa",
        help="When enabled, the proposed loan amount matches the current outstanding balance.")
    if ss.p_auto_amount:
        ss.p_loan_amt = curr_balance
        c1, c2 = st.columns(2)
        with c1:
            computed("Proposed Loan Amount (auto-filled)", fc(ss.p_loan_amt),
                     "from current remaining balance")
        with c2:
            ss.p_start_date = st.date_input("Proposed Settlement Date",
                value=ss.p_start_date, key="w_p_sd")
    else:
        c1, c2 = st.columns(2)
        with c1:
            ss.p_loan_amt = st.number_input("Proposed Loan Amount ($)", value=ss.p_loan_amt,
                min_value=0.0, step=1_000.0, key="w_p_la")
        with c2:
            ss.p_start_date = st.date_input("Proposed Settlement Date",
                value=ss.p_start_date, key="w_p_sd2")

    ss.p_use_dates = st.toggle("Calculate term from start and end dates",
                                value=ss.p_use_dates, key="w_p_ud")
    if ss.p_use_dates:
        c1, c2 = st.columns(2)
        with c1:
            ss.p_end_date = st.date_input("Loan End Date", value=ss.p_end_date, key="w_p_ed")
        with c2:
            if ss.p_end_date > ss.p_start_date:
                ss.p_term_mo = months_between(ss.p_start_date, ss.p_end_date)
                computed("Calculated Term", f"{ss.p_term_mo} months",
                         f"{ss.p_term_mo/12:.1f} years")
    else:
        c1, c2 = st.columns(2)
        with c1:
            ss.p_term_mo = st.number_input("Loan Term (months)", value=ss.p_term_mo,
                min_value=1, max_value=600, step=12, key="w_p_tm")
        with c2:
            computed("Equivalent", f"{ss.p_term_mo/12:.1f} years")

    # ── Interest Rates (two-column layout; each column contains rate, fees, ──
    # ── computed rates. Fixed column also contains Fixed Period & Reversion) ──
    sec("Interest Rates")

    # Match fees toggle — placed prominently above the two columns
    ss.p_fees_match = st.toggle(
        "Match Fixed Rate fees to Variable Rate fees",
        value=ss.p_fees_match, key="w_p_fm",
        help="When enabled (default), Fixed fees mirror Variable fees and their input "
             "fields are disabled. Disable this toggle to set Fixed fees independently.")

    # Apply fee matching: when matched, Fixed fees follow Variable fees
    if ss.p_fees_match:
        ss.p_fix_fee_mo = ss.p_var_fee_mo
        ss.p_fix_fee_setup = ss.p_var_fee_setup
        ss.p_fix_fee_break = ss.p_var_fee_break
        ss.p_fix_fee_other = ss.p_var_fee_other

    # Compute comparison and effective rates using each component's own fees
    comp_var = comparison_rate_asic(ss.p_var_fee_setup, ss.p_var_fee_mo, ss.p_adv_var_rate)
    eff_var = effective_rate_calc(ss.p_loan_amt, ss.p_var_fee_setup,
                                   ss.p_var_fee_mo, ss.p_adv_var_rate, ss.p_term_mo)
    comp_fix = comparison_rate_asic(ss.p_fix_fee_setup, ss.p_fix_fee_mo, ss.p_adv_fix_rate)
    fix_mo_period = ss.p_fix_yrs * 12
    eff_fix = effective_rate_calc(ss.p_loan_amt, ss.p_fix_fee_setup, ss.p_fix_fee_mo,
                                   ss.p_adv_fix_rate, fix_mo_period)

    if not ss.p_rev_rate_override:
        ss.p_rev_rate = round(eff_var, 4)

    # Column headers
    h1, h2 = st.columns(2)
    h1.markdown('<div style="text-align:center;color:#30d996;font-size:0.75rem;font-weight:600;'
                'padding:6px;background:#071a0f;border-radius:5px;margin-bottom:8px">'
                'VARIABLE RATE</div>', unsafe_allow_html=True)
    h2.markdown('<div style="text-align:center;color:#e94560;font-size:0.75rem;font-weight:600;'
                'padding:6px;background:#1a0709;border-radius:5px;margin-bottom:8px">'
                'FIXED RATE</div>', unsafe_allow_html=True)

    # Two columns: each contains rate → fees → computed rates
    # Fixed column ALSO contains Fixed Period and Reversion Rate
    c_v, c_f = st.columns(2)

    with c_v:
        # Variable Rate inputs
        ss.p_adv_var_rate = st.number_input(
            "Advertised Variable Rate (% p.a.)", value=ss.p_adv_var_rate,
            min_value=0.0, max_value=30.0, step=0.01, format="%.4f", key="w_p_avr",
            help="Headline variable rate advertised by the lender, before fees.")

        # Variable Rate fees (under the Variable column)
        st.markdown('<div class="sub-title">Variable Rate Fees</div>', unsafe_allow_html=True)
        ss.p_var_fee_mo = st.number_input("Monthly Fee ($)", value=ss.p_var_fee_mo,
            min_value=0.0, step=1.0, key="w_p_var_fm",
            help="Monthly account-keeping fee for the Variable component.")
        ss.p_var_fee_setup = st.number_input("Setup / Establishment Fee ($)",
            value=ss.p_var_fee_setup, min_value=0.0, step=100.0, key="w_p_var_fs",
            help="One-off establishment fee for the Variable component.")
        ss.p_var_fee_break = st.number_input("Breakage Fee ($)", value=ss.p_var_fee_break,
            min_value=0.0, step=100.0, key="w_p_var_fb",
            help="Exit or discharge fee if breaking the Variable component early.")
        ss.p_var_fee_other = st.number_input("Other One-off Fee ($)", value=ss.p_var_fee_other,
            min_value=0.0, step=100.0, key="w_p_var_fo",
            help="Other one-off fees (valuation, legal, etc.) for the Variable component.")

        # Computed rates for Variable
        computed("Comparison Variable Rate", fp(comp_var),
                 "ASIC: $150k over 25yr, using Variable fees")
        computed("Effective Variable Rate", fp(eff_var),
                 f"For ${ss.p_loan_amt:,.0f} over {ss.p_term_mo} months, Variable fees")

    with c_f:
        # Fixed Rate inputs
        ss.p_adv_fix_rate = st.number_input(
            "Advertised Fixed Rate (% p.a.)", value=ss.p_adv_fix_rate,
            min_value=0.0, max_value=30.0, step=0.01, format="%.4f", key="w_p_afr",
            help="Headline fixed rate advertised for the Fixed Period.")

        # Fixed Period lives under the FIXED column (applies only to Fixed component)
        ss.p_fix_yrs = st.number_input(
            "Fixed Period (years)", value=ss.p_fix_yrs, min_value=1, max_value=30, step=1,
            key="w_p_fy",
            help="Duration that the Fixed Rate applies. Applies ONLY to the Fixed "
                 "component. Variable component uses the variable rate throughout.")

        # Fixed Rate fees (under the Fixed column) — disabled when matched to Variable
        st.markdown('<div class="sub-title">Fixed Rate Fees</div>', unsafe_allow_html=True)
        if ss.p_fees_match:
            # Show as read-only computed cards
            computed("Monthly Fee", fc(ss.p_fix_fee_mo), "matched to Variable")
            computed("Setup / Establishment Fee", fc(ss.p_fix_fee_setup),
                     "matched to Variable")
            computed("Breakage Fee", fc(ss.p_fix_fee_break), "matched to Variable")
            computed("Other One-off Fee", fc(ss.p_fix_fee_other), "matched to Variable")
        else:
            ss.p_fix_fee_mo = st.number_input("Monthly Fee ($)", value=ss.p_fix_fee_mo,
                min_value=0.0, step=1.0, key="w_p_fix_fm",
                help="Monthly account-keeping fee for the Fixed component.")
            ss.p_fix_fee_setup = st.number_input("Setup / Establishment Fee ($)",
                value=ss.p_fix_fee_setup, min_value=0.0, step=100.0, key="w_p_fix_fs",
                help="One-off establishment fee for the Fixed component.")
            ss.p_fix_fee_break = st.number_input("Breakage Fee ($)",
                value=ss.p_fix_fee_break, min_value=0.0, step=100.0, key="w_p_fix_fb",
                help="Exit or discharge fee if breaking the Fixed component early.")
            ss.p_fix_fee_other = st.number_input("Other One-off Fee ($)",
                value=ss.p_fix_fee_other, min_value=0.0, step=100.0, key="w_p_fix_fo",
                help="Other one-off fees for the Fixed component.")

        # Computed rates for Fixed
        computed("Comparison Fixed Rate", fp(comp_fix),
                 "ASIC: $150k over 25yr, using Fixed fees")
        computed("Effective Fixed Rate", fp(eff_fix),
                 f"For ${ss.p_loan_amt:,.0f} over Fixed Period ({fix_mo_period} months), "
                 "Fixed fees")

        # Reversion rate (after Fixed Period ends, Fixed component reverts to variable)
        ss.p_rev_rate_override = st.toggle(
            "Override reversion rate", value=ss.p_rev_rate_override, key="w_p_rro",
            help="By default, reversion rate = Effective Variable Rate. Enable to set "
                 "a custom rate.")
        if ss.p_rev_rate_override:
            ss.p_rev_rate = st.number_input("Reversion Rate (% p.a.)",
                value=ss.p_rev_rate, min_value=0.0, max_value=30.0, step=0.01,
                format="%.4f", key="w_p_rr_ov",
                help="Rate the Fixed component reverts to after the Fixed Period expires.")
        else:
            computed("Reversion Rate", fp(ss.p_rev_rate),
                     "Auto-filled from Effective Variable Rate")

    # ── Refinancing Strategy (moved from Scenarios — now a Proposed Loan subsection) ──
    sec("Refinancing Strategy")
    strategies = ["Conservative (80% Fixed)", "Balanced (Optimal Split)",
                  "Aggressive (0% Fixed)", "Manual"]
    if ss.strategy not in strategies:
        ss.strategy = "Balanced (Optimal Split)"
    ss.strategy = st.radio(
        "Select strategy", strategies, index=strategies.index(ss.strategy),
        key="w_strat", horizontal=True,
        help="Determines the Fixed/Variable split. Balanced finds the optimal split "
             "minimising interest + remaining balance at end of Fixed Period. "
             "Manual enables the slider below for custom allocation.")

    # Strategy determines split %
    if ss.strategy == "Conservative (80% Fixed)":
        ss.p_split_pct = 80.0; ss.p_split_auto = False
        computed("Fixed Component (locked by strategy)", fp(ss.p_split_pct, 1),
                 f"Variable ${ss.p_loan_amt*0.2:,.0f} / Fixed ${ss.p_loan_amt*0.8:,.0f}")
    elif ss.strategy == "Balanced (Optimal Split)":
        ss.p_split_auto = True
        st.markdown('<div class="note">Optimal fixed percentage is computed below '
                    'from the objective curve. Adjust Fixed Period to see how the '
                    'optimum shifts.</div>', unsafe_allow_html=True)
    elif ss.strategy == "Aggressive (0% Fixed)":
        ss.p_split_pct = 0.0; ss.p_split_auto = False
        computed("Fixed Component (locked by strategy)", fp(ss.p_split_pct, 1),
                 f"Variable ${ss.p_loan_amt:,.0f} / Fixed $0")
    else:  # Manual
        ss.p_split_auto = False
        c1, c2 = st.columns(2)
        with c1:
            ss.p_split_pct = st.slider("Fixed Component (%)", 0.0, 100.0,
                ss.p_split_pct, 0.5, key="w_p_sp",
                help="Manual allocation percentage for the Fixed component.")
        with c2:
            computed("Allocation",
                     f"Variable ${ss.p_loan_amt*(100-ss.p_split_pct)/100:,.0f} / "
                     f"Fixed ${ss.p_loan_amt*ss.p_split_pct/100:,.0f}")

    # Variable rate changes (shared, read-only display)
    st.markdown('<div class="sub-title">Variable Rate Changes '
                '(synced with Anticipated Rate Changes in Current Loan)</div>',
                unsafe_allow_html=True)
    if ss.future_var_deltas:
        h = st.columns([2, 1.6, 2.2, 0.5])
        for i, lbl in enumerate(["Effective Date", "Change (±%)", "Result"]):
            h[i].markdown(f'<div class="list-hdr">{lbl}</div>', unsafe_allow_html=True)
        cum = ss.p_adv_var_rate
        for row in ss.future_var_deltas:
            c1, c2, c3, c4 = st.columns([2, 1.6, 2.2, 0.5])
            nv = float(row[1])
            res = round(cum + nv, 4)
            sign = "+" if nv >= 0 else ""
            clr = "#e94560" if nv > 0 else ("#30d996" if nv < 0 else "#64748b")
            c1.markdown(f'<div style="padding:8px 3px;color:#8892b0;font-size:.82rem">'
                        f'{parse_dt(row[0]).strftime("%d %b %Y")}</div>',
                        unsafe_allow_html=True)
            c2.markdown(f'<div style="padding:8px 3px;color:{clr};font-size:.875rem;font-weight:500">'
                        f'{sign}{nv:.2f}%</div>', unsafe_allow_html=True)
            c3.markdown(f'<div style="padding:8px 3px;color:{clr};font-size:.875rem;font-weight:600">'
                        f'{res:.4f}%</div>', unsafe_allow_html=True)
            cum = res
    else:
        st.markdown('<div style="color:#64748b;font-size:0.8rem;padding:6px 0">'
                    'No anticipated rate changes — add them in the Current Loan section above.</div>',
                    unsafe_allow_html=True)

    # ── Offset Account ────────────────────────────────────────────────────
    sec("Offset Account (applied to Variable component)")
    ss.p_off_match = st.toggle(
        "Match Current Balance to Current Loan Offset Account",
        value=ss.p_off_match, key="w_p_om_match",
        help="When enabled, the Current Balance of the proposed offset equals the "
             "Current Loan's offset initial balance.")
    c1, c2, c3 = st.columns(3)
    with c1:
        if ss.p_off_match:
            ss.p_off_init = ss.c_off_init
            computed("Current Balance ($)", fc(ss.p_off_init),
                     "matched to Current Loan offset")
        else:
            ss.p_off_init = st.number_input("Current Balance ($)", value=ss.p_off_init,
                min_value=0.0, step=1_000.0, key="w_p_oi",
                help="Balance of the offset account at proposed loan settlement.")
    with c2:
        # Default offset start date to settlement date if not already set
        if ss.p_off_date < ss.p_start_date:
            ss.p_off_date = ss.p_start_date
        ss.p_off_date = st.date_input("Offset Start Date",
            value=ss.p_off_date, key="w_p_od",
            help="Defaults to Proposed Settlement Date.")
    with c3:
        ss.p_off_monthly = st.number_input("Monthly Addition ($)", value=ss.p_off_monthly,
            min_value=0.0, step=100.0, key="w_p_om",
            help="Automatic monthly contribution to the offset account.")
    lump_list("p_off_lumps", "Offset Lump Sum Deposits")

    # Extra Repayments / Redraws
    extra_repay_list("p_extra_repay", "Extra Repayments and Redraws")

# ═══════════════════════════════════════════════════════════════════════════════
# INPUT: SCENARIOS (formerly Strategy and Scenarios)
# ═══════════════════════════════════════════════════════════════════════════════

def section_scenarios():
    """Scenarios section — merges RBA Cash Rate & Market Indicators panel with
       the RBA cash-rate scenario slider and the payment-behaviour toggle.
       Refinancing Strategy has been moved into the Proposed Loan section."""
    ss = st.session_state
    rba_rate = ss._rba_rate

    # ── RBA Cash Rate and Market Indicators (embedded, not its own expander) ──
    render_rba_panel()

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Payment Behaviour toggle ──────────────────────────────────────────
    sec("Payment Behaviour on Rate Changes")
    ss.maintain_pmt = st.toggle(
        "When rates fall, maintain current repayment (pays off faster)",
        value=ss.maintain_pmt, key="w_mp",
        help="When rates rise, repayments always increase to maintain the remaining term "
             "(term can only decrease). When rates fall: ON → keep paying the current "
             "amount, loan shortens. OFF → drop to the new minimum, loan term preserved.")
    st.markdown(
        '<div class="note">Applies to scenarios where rates fall — negative RBA '
        'movements below, negative anticipated rate changes in Current Loan, and '
        'the −0.25% / −0.50% scenarios on the Rate Scenarios dashboard tab.</div>',
        unsafe_allow_html=True)

    # ── RBA Cash Rate Scenario ────────────────────────────────────────────
    sec("RBA Cash Rate Scenario")
    st.markdown(
        '<div class="note">Applies on top of any changes already entered in Current '
        'Loan or Proposed Loan. Flows into all Variable-rate calculations — '
        'including the Original Loan when Current is treated as a continuation. '
        'Does not affect Fixed rates during the Fixed Period (but does apply to '
        'the Fixed component after it reverts to variable).</div>',
        unsafe_allow_html=True)
    c1, c2 = st.columns([4, 1])
    with c1:
        ss.rba_bps = st.slider(
            "RBA cash rate change (basis points)", -300, 300, ss.rba_bps, 25, key="w_rba",
            help="Additional change applied from today onwards on top of any rate "
                 "changes already entered in Current and Proposed Loan sections.")
        if ss.rba_bps != 0:
            d = "increase" if ss.rba_bps > 0 else "decrease"
            st.markdown(
                f'<div class="note">Applying an additional {abs(ss.rba_bps)} bps '
                f'({abs(ss.rba_bps)/100:.2f}%) {d} immediately.</div>',
                unsafe_allow_html=True)
    with c2:
        if rba_rate:
            new_r = round(rba_rate + ss.rba_bps / 100, 2)
            computed("Implied RBA Rate", fp(new_r), "after scenario")

# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all():
    ss = st.session_state
    if ss.p_loan_amt <= 0 or ss.o_balance <= 0 or ss.o_term_mo <= 0:
        return None
    maintain = ss.maintain_pmt
    far = add_months(TODAY, max(ss.o_term_mo, ss.p_term_mo) + 12)

    def lumps_t(lumps, loan_start):
        return deltas_to_lumps_t(lumps, loan_start)

    def extra_t(lumps, loan_start):
        return deltas_to_lumps_t(lumps, loan_start)

    # RBA scenario delta is applied at the START of the current month so the
    # amortization loop's period-reference picks it up in the period containing
    # TODAY. This ensures the first forward-looking row (Date >= today) reflects
    # the scenario on the Dashboard Overview.
    rba_delta_pct = ss.rba_bps / 100.0 if ss.rba_bps != 0 else 0.0
    _start_of_month = date(TODAY.year, TODAY.month, 1)

    # ── Original ──
    # If Current is a continuation of Original, the RBA scenario also affects Original
    # (since Original is still this loan, just before refinance). Apply at start-of-month.
    o_deltas = list(ss.o_rate_deltas)
    if ss.c_is_cont and rba_delta_pct != 0:
        o_deltas = o_deltas + [[max(_start_of_month, ss.o_balance_date), rba_delta_pct]]
    o_rsched = build_rate_schedule(ss.o_rate, o_deltas)
    df_orig = amortize(
        ss.o_balance, ss.o_balance_date, ss.o_term_mo,
        o_rsched, ss.o_off_init, ss.o_off_monthly,
        lumps_t(ss.o_off_lumps, ss.o_balance_date),
        extra_t(ss.o_extra_repay, ss.o_balance_date),
        ss.o_fee_mo, maintain)

    # ── Current ──
    if ss.c_is_cont:
        c_base_rate = ss.o_rate
        c_hist = ss.o_rate_deltas
        c_term = ss.o_term_mo
        c_start = ss.o_balance_date
        c_bal = ss.o_balance
        c_fee_mo = ss.o_fee_mo
    else:
        c_base_rate = ss.c_rate
        c_hist = ss.c_rate_deltas
        c_term = ss.c_term_mo
        c_start = TODAY
        c_bal = ss.c_balance
        c_fee_mo = ss.c_fee_mo
    all_c_deltas = list(c_hist) + list(ss.future_var_deltas)
    if rba_delta_pct != 0:
        all_c_deltas = all_c_deltas + [[max(_start_of_month, c_start), rba_delta_pct]]
    c_rsched = build_rate_schedule(c_base_rate, all_c_deltas)
    df_curr = amortize(
        c_bal, c_start, c_term, c_rsched,
        ss.c_off_init, ss.c_off_monthly,
        lumps_t(ss.c_off_lumps, c_start),
        extra_t(ss.c_extra_repay, c_start),
        c_fee_mo, maintain)

    # ── Proposed: rates with individualised fees ──
    if ss.p_fees_match:
        fix_mo_fee, fix_setup_fee = ss.p_var_fee_mo, ss.p_var_fee_setup
    else:
        fix_mo_fee, fix_setup_fee = ss.p_fix_fee_mo, ss.p_fix_fee_setup

    comp_var = comparison_rate_asic(ss.p_var_fee_setup, ss.p_var_fee_mo, ss.p_adv_var_rate)
    eff_var = effective_rate_calc(ss.p_loan_amt, ss.p_var_fee_setup,
                                   ss.p_var_fee_mo, ss.p_adv_var_rate, ss.p_term_mo)
    comp_fix = comparison_rate_asic(fix_setup_fee, fix_mo_fee, ss.p_adv_fix_rate)
    fix_mo_period = ss.p_fix_yrs * 12
    eff_fix = effective_rate_calc(ss.p_loan_amt, fix_setup_fee, fix_mo_fee,
                                   ss.p_adv_fix_rate, fix_mo_period)

    # Optimal split uses EFFECTIVE rates for realism
    if ss.p_split_auto:
        best_pct, split_df = calc_optimal_split(
            ss.p_loan_amt, eff_var, ss.p_adv_fix_rate,
            ss.p_rev_rate, ss.p_fix_yrs, ss.p_term_mo)
        ss.p_split_pct = round(best_pct, 1)
    else:
        _, split_df = calc_optimal_split(
            ss.p_loan_amt, eff_var, ss.p_adv_fix_rate,
            ss.p_rev_rate, ss.p_fix_yrs, ss.p_term_mo)
        best_pct = ss.p_split_pct

    p_f = ss.p_loan_amt * best_pct / 100
    p_v = ss.p_loan_amt * (100 - best_pct) / 100

    # Variable component: uses shared future_var_deltas + RBA scenario (applied
    # at start-of-month or settlement, whichever is later, for consistent forward
    # payment reflection on Dashboard metrics)
    p_var_deltas = list(ss.future_var_deltas)
    if rba_delta_pct != 0:
        p_var_deltas = p_var_deltas + [[max(_start_of_month, ss.p_start_date), rba_delta_pct]]
    p_vsched = build_rate_schedule(ss.p_adv_var_rate, p_var_deltas)

    # Fixed component: uses fixed rate for fix_mo_period, then reverts to p_rev_rate.
    # After revert, the Fixed component effectively becomes variable — so we also
    # apply the RBA delta to the reversion rate (Fixed is not affected DURING the
    # fixed period, but IS affected after revert).
    fix_rev_date = add_months(TODAY, fix_mo_period)
    eff_rev_rate = ss.p_rev_rate + (rba_delta_pct if rba_delta_pct != 0 else 0.0)
    p_fsched = [(date(1900, 1, 1), ss.p_adv_fix_rate), (fix_rev_date, eff_rev_rate)]

    p_lumps_t = lumps_t(ss.p_off_lumps, TODAY)
    p_extra_t = extra_t(ss.p_extra_repay, TODAY)

    df_pv = amortize(p_v, TODAY, ss.p_term_mo, p_vsched,
                     ss.p_off_init, ss.p_off_monthly, p_lumps_t, p_extra_t,
                     ss.p_var_fee_mo * (100 - best_pct) / 100, maintain) if p_v > 1 else pd.DataFrame()
    df_pf = amortize(p_f, TODAY, ss.p_term_mo, p_fsched,
                     0.0, 0.0, (), (),
                     fix_mo_fee * best_pct / 100, maintain) if p_f > 1 else pd.DataFrame()
    df_ps = merge_schedules(df_pv, df_pf)

    # ── Payment scenarios ──
    # Key fix: apply maintain_pmt by setting min_pmt_floor to the base scenario payment
    base_scen_pmt = calc_payment(ss.p_loan_amt, ss.p_adv_var_rate, ss.p_term_mo)
    scen_deltas_list = [0.0, 0.25, 0.50, 1.00, -0.25, -0.50, ss.rba_bps / 100.0]
    scen_labels = ["Base", "+0.25%", "+0.50%", "+1.00%", "-0.25%", "-0.50%",
                   f"RBA {'+' if ss.rba_bps >= 0 else ''}{ss.rba_bps/100:.2f}%"]
    scenarios = {}
    for lbl, delta in zip(scen_labels, scen_deltas_list):
        sr = ss.p_adv_var_rate + delta
        extra = list(ss.future_var_deltas)
        sched = build_rate_schedule(sr, extra)
        # For rate-fall scenarios, apply maintain_pmt via floor
        floor = base_scen_pmt if (maintain and delta < 0 and lbl != "Base") else 0.0
        df_s = amortize(ss.p_loan_amt, TODAY, ss.p_term_mo, sched,
                        ss.p_off_init, ss.p_off_monthly, p_lumps_t, p_extra_t,
                        ss.p_var_fee_mo, maintain, min_pmt_floor=floor)
        if not df_s.empty:
            scenarios[lbl] = {"rate": sr, "payment": df_s["Payment"].iloc[0],
                              "total_interest": df_s["Cum Interest"].iloc[-1],
                              "term_months": len(df_s), "df": df_s}

    return {
        "df_orig": df_orig, "df_curr": df_curr,
        "df_pv": df_pv, "df_pf": df_pf, "df_ps": df_ps,
        "split_df": split_df, "best_pct": best_pct,
        "eff_var": eff_var, "eff_fix": eff_fix,
        "comp_var": comp_var, "comp_fix": comp_fix,
        "scenarios": scenarios,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# FORENSIC ANALYSIS DASHBOARD — 5 Themes (I–V) per Gemini-style audit framework
# ═══════════════════════════════════════════════════════════════════════════════

def _tier(value, tiers):
    """Given value and list of (threshold, label, colour) tuples sorted ascending
       by threshold, return the (label, colour) for the first threshold v <= thr."""
    for thr, lbl, clr in tiers:
        if value <= thr:
            return lbl, clr
    return tiers[-1][1], tiers[-1][2]

def _days_saved_snowball(df_curr, df_proposed):
    """Months saved when maintain-payment=True (proposed amortised at higher of
       (proposed min pmt, current pmt))."""
    if df_curr is None or df_curr.empty: return 0
    if df_proposed is None or df_proposed.empty: return 0
    return max(0, len(df_curr) - len(df_proposed))

def forensic_compute(R):
    """Compute every metric the 5 themes need, in one pass, using ONLY
       existing session-state inputs + the compute_all() outputs."""
    ss = st.session_state
    df_o, df_c, df_ps = R["df_orig"], R["df_curr"], R["df_ps"]
    df_pv, df_pf = R["df_pv"], R["df_pf"]

    rba_rate = ss._rba_rate if ss._rba_rate else 4.35  # sensible fallback

    # Current loan headline rate (after deltas to today)
    if ss.c_is_cont:
        c_rate = eff_rate_from_deltas(ss.o_rate, ss.o_rate_deltas, as_of=TODAY)
        c_bal = ss.o_balance
        c_fee_mo = ss.o_fee_mo
        c_setup = 0.0
        c_break = ss.o_fee_break
        c_term_rem = max(0, ss.o_term_mo - months_between(ss.o_balance_date, TODAY))
    else:
        c_rate = ss.c_rate
        c_bal = ss.c_balance
        c_fee_mo = ss.c_fee_mo
        c_setup = ss.c_fee_setup
        c_break = 0.0
        c_term_rem = ss.c_term_mo

    # Proposed headline — weighted by split
    best_pct = R["best_pct"]
    p_blended_rate = (best_pct/100)*ss.p_adv_fix_rate + (1-best_pct/100)*ss.p_adv_var_rate
    p_eff_var = R["eff_var"]
    p_eff_fix = R["eff_fix"]

    # Offset balances (today)
    off_curr = ss.c_off_init if not ss.c_is_cont else ss.o_off_init
    off_prop = ss.p_off_init

    # ── THEME I: Spread & Proximity ──────────────────────────────────────
    # Current effective rate (after offset): net_debt × rate ÷ gross_debt
    def effective_rate_after_offset(rate_pct, loan, offset):
        if loan <= 0: return rate_pct
        net = max(0, loan - offset)
        return rate_pct * (net / loan)

    curr_eff_after_off = effective_rate_after_offset(c_rate, c_bal, off_curr)
    prop_eff_after_off = effective_rate_after_offset(p_blended_rate, ss.p_loan_amt, off_prop)
    curr_spread_bps = (c_rate - rba_rate) * 100
    prop_spread_bps = (p_blended_rate - rba_rate) * 100
    curr_eff_spread_bps = (curr_eff_after_off - rba_rate) * 100
    prop_eff_spread_bps = (prop_eff_after_off - rba_rate) * 100

    # Tiers for spread classification (bps above RBA)
    SPREAD_TIERS = [
        (100,  "Elite",       "#30d996"),
        (170,  "Competitive", "#4a9af5"),
        (220,  "Market",      "#f5a94a"),
        (270,  "Expensive",   "#e94560"),
        (9999, "Very Expensive", "#e94560"),
    ]
    curr_tier = _tier(curr_spread_bps, SPREAD_TIERS)
    prop_tier = _tier(prop_spread_bps, SPREAD_TIERS)
    curr_eff_tier = _tier(curr_eff_spread_bps, SPREAD_TIERS)
    prop_eff_tier = _tier(prop_eff_spread_bps, SPREAD_TIERS)

    # Annual interest at proposed effective rate
    prop_annual_interest = max(0, ss.p_loan_amt - off_prop) * p_blended_rate / 100

    # ── THEME II: Strategy Stress-Test ──────────────────────────────────
    # Run three configurations over the fixed period (or 24 months whichever shorter)
    horizon_mo = min(ss.p_fix_yrs * 12, ss.p_term_mo)

    def stress_interest(pct_fixed, with_offset, fix_rate, var_rate, fix_period_mo,
                          full_term_mo, horizon):
        """Return cumulative interest over `horizon` months for a given config."""
        loan = ss.p_loan_amt
        p_f = loan * pct_fixed / 100
        p_v = loan * (1 - pct_fixed / 100)
        # Fixed side: no offset (offset lives on variable side only)
        bf, cif = fast_partial(p_f, fix_rate, full_term_mo,
                               min(horizon, fix_period_mo)) if p_f > 0 else (0, 0)
        # Variable side: optionally with offset benefit
        if p_v > 0:
            bv, civ = fast_partial(p_v, var_rate, full_term_mo, horizon)
            if with_offset and off_prop > 0:
                # Estimate offset interest saving over horizon
                avg_net_v = max(0, (p_v + bv) / 2 - off_prop)
                avg_gross_v = max(1, (p_v + bv) / 2)
                civ = civ * (avg_net_v / avg_gross_v)
        else:
            civ = 0
        return cif + civ

    scen_split = stress_interest(best_pct/100, True, ss.p_adv_fix_rate, ss.p_adv_var_rate,
                                  ss.p_fix_yrs*12, ss.p_term_mo, horizon_mo)
    scen_var_off = stress_interest(0.0, True, ss.p_adv_fix_rate, ss.p_adv_var_rate,
                                    ss.p_fix_yrs*12, ss.p_term_mo, horizon_mo)
    scen_all_fix = stress_interest(1.0, False, ss.p_adv_fix_rate, ss.p_adv_var_rate,
                                    ss.p_fix_yrs*12, ss.p_term_mo, horizon_mo)
    # The "Offset Disaster" delta
    disaster_cost = scen_all_fix - scen_split
    freedom_cost = scen_var_off - scen_split

    # ── THEME III: Execution — Break-even ─────────────────────────────────
    # Current vs Proposed monthly payment (forward-looking)
    def fwd_pmt(df):
        if df is None or df.empty: return 0.0
        mask = df["Date"].apply(lambda d: d >= TODAY)
        fwd = df[mask]
        return fwd["Payment"].iloc[0] if not fwd.empty else df["Payment"].iloc[0]

    curr_pmt = fwd_pmt(df_c)
    prop_pmt = fwd_pmt(df_ps)
    monthly_saving = curr_pmt - prop_pmt
    # Switching costs: proposed setup + breakage on current (if continuation, use original break)
    switch_cost = (ss.p_var_fee_setup + ss.p_fix_fee_setup
                   + ss.p_var_fee_other + ss.p_fix_fee_other
                   + c_break)
    if monthly_saving > 1:
        break_even_mo = switch_cost / monthly_saving
    else:
        break_even_mo = float("inf")

    # ── THEME IV: Life-of-Loan Snowball ──────────────────────────────────
    total_int_curr = df_c["Cum Interest"].iloc[-1] if (df_c is not None and not df_c.empty) else 0
    total_int_prop = df_ps["Cum Interest"].iloc[-1] if (df_ps is not None and not df_ps.empty) else 0
    lifetime_saving = total_int_curr - total_int_prop
    months_snowball = _days_saved_snowball(df_c, df_ps)

    # Recompute scenario: same proposed BUT floor payment at current payment
    # to show the true snowball
    snowball_months_saved = 0
    snowball_interest_saved = 0
    try:
        if curr_pmt > prop_pmt and ss.p_loan_amt > 0:
            p_var_deltas = list(ss.future_var_deltas)
            vsched = build_rate_schedule(ss.p_adv_var_rate, p_var_deltas)
            df_snow = amortize(ss.p_loan_amt, ss.p_start_date, ss.p_term_mo,
                               vsched, off_prop, ss.p_off_monthly,
                               deltas_to_lumps_t(ss.p_off_lumps, ss.p_start_date),
                               deltas_to_lumps_t(ss.p_extra_repay, ss.p_start_date),
                               ss.p_var_fee_mo, True, min_pmt_floor=curr_pmt)
            if not df_snow.empty:
                snowball_months_saved = ss.p_term_mo - len(df_snow)
                snowball_interest_saved = total_int_prop - df_snow["Cum Interest"].iloc[-1]
    except Exception:
        pass

    # ── THEME V: Checklist conditions ────────────────────────────────────
    has_split = 0 < best_pct < 100
    has_offset = off_prop > 0
    has_fixed = best_pct > 0
    rate_drop = max(0, c_rate - p_blended_rate)

    return {
        "rba_rate": rba_rate,
        "c_rate": c_rate, "c_bal": c_bal, "c_term_rem": c_term_rem,
        "p_blended_rate": p_blended_rate, "p_eff_var": p_eff_var, "p_eff_fix": p_eff_fix,
        "off_curr": off_curr, "off_prop": off_prop,
        "curr_eff_after_off": curr_eff_after_off,
        "prop_eff_after_off": prop_eff_after_off,
        "curr_spread_bps": curr_spread_bps, "prop_spread_bps": prop_spread_bps,
        "curr_eff_spread_bps": curr_eff_spread_bps,
        "prop_eff_spread_bps": prop_eff_spread_bps,
        "curr_tier": curr_tier, "prop_tier": prop_tier,
        "curr_eff_tier": curr_eff_tier, "prop_eff_tier": prop_eff_tier,
        "prop_annual_interest": prop_annual_interest,
        "horizon_mo": horizon_mo,
        "scen_split": scen_split, "scen_var_off": scen_var_off, "scen_all_fix": scen_all_fix,
        "disaster_cost": disaster_cost, "freedom_cost": freedom_cost,
        "curr_pmt": curr_pmt, "prop_pmt": prop_pmt, "monthly_saving": monthly_saving,
        "switch_cost": switch_cost, "break_even_mo": break_even_mo,
        "total_int_curr": total_int_curr, "total_int_prop": total_int_prop,
        "lifetime_saving": lifetime_saving,
        "snowball_months_saved": snowball_months_saved,
        "snowball_interest_saved": snowball_interest_saved,
        "has_split": has_split, "has_offset": has_offset, "has_fixed": has_fixed,
        "rate_drop": rate_drop,
        "best_pct": best_pct,
    }

def _hero_card(tier_lbl: str, clr: str, big_text: str, sub_text: str = ""):
    """Large coloured hero-style summary card used across themes."""
    sub_html = f'<div style="color:#8892b0;font-size:0.84rem;margin-top:6px">{sub_text}</div>' if sub_text else ""
    return (f'<div style="background:#0b0f1a;border:1px solid {clr};border-left:4px solid {clr};'
            f'border-radius:7px;padding:16px 20px;margin-bottom:14px">'
            f'<div style="color:{clr};font-size:0.7rem;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:.08em;margin-bottom:6px">{tier_lbl}</div>'
            f'<div style="color:#d4dbe8;font-size:1.3rem;font-weight:600;line-height:1.35">{big_text}</div>'
            f'{sub_html}</div>')

# ── THEME I: Macro-Economic Anchor ───────────────────────────────────────
def theme_i_anchor(R, F):
    ss = st.session_state
    rba = F["rba_rate"]

    st.markdown('<h3 style="color:#4a9af5;font-size:1rem;font-weight:600;margin:4px 0 8px">'
                'Theme I · The Macro-Economic Anchor</h3>', unsafe_allow_html=True)
    st.markdown(
        '<div class="note">Your loan\'s "cost of capital" measured against the '
        'RBA Cash Rate. The tighter this spread, the more efficient your debt.</div>',
        unsafe_allow_html=True)

    # Hero card: the proximity outcome
    prop_tier_lbl, prop_tier_clr = F["prop_eff_tier"]
    hero_text = (
        f'Your proposed loan operates at <strong style="color:{prop_tier_clr}">'
        f'{F["prop_eff_spread_bps"]:.0f} bps</strong> above the RBA Cash Rate '
        f'of {fp(rba)} after offset benefit.'
    )
    hero_sub = (
        f'Effective rate after offset: <strong>{fp(F["prop_eff_after_off"])}</strong>'
        f' on net debt of <strong>{fc(max(0, ss.p_loan_amt - F["off_prop"]))}</strong>'
        f' (${ss.p_loan_amt:,.0f} loan − ${F["off_prop"]:,.0f} offset).'
    )
    st.markdown(_hero_card(prop_tier_lbl + " PROXIMITY", prop_tier_clr, hero_text, hero_sub),
                unsafe_allow_html=True)

    # Comparison grid
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card("RBA Cash Rate (live)", fp(rba)),
                    unsafe_allow_html=True)
    with c2:
        tierL, tierC = F["curr_tier"]
        st.markdown(metric_card(f"Current Headline Rate",
                                 fp(F["c_rate"]),
                                 diff=f"+{F['curr_spread_bps']:.0f} bps ({tierL})",
                                 diff_pos=False, diff_neutral=(F['curr_spread_bps']<170)),
                    unsafe_allow_html=True)
    with c3:
        tierL, tierC = F["prop_tier"]
        st.markdown(metric_card(f"Proposed Headline (blended)",
                                 fp(F["p_blended_rate"]),
                                 diff=f"+{F['prop_spread_bps']:.0f} bps ({tierL})",
                                 diff_pos=False, diff_neutral=(F['prop_spread_bps']<170)),
                    unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("Proposed Effective After Offset",
                                 fp(F["prop_eff_after_off"]),
                                 diff=f"+{F['prop_eff_spread_bps']:.0f} bps vs RBA",
                                 diff_neutral=True),
                    unsafe_allow_html=True)

    # Spread chart — current vs proposed (headline & effective) vs RBA
    fig = go.Figure()
    cats = ["RBA Cash Rate", "Current Headline", "Current After Offset",
            "Proposed Headline", "Proposed After Offset"]
    vals = [rba, F["c_rate"], F["curr_eff_after_off"],
            F["p_blended_rate"], F["prop_eff_after_off"]]
    clrs = [C_ORIG, "#e94560", "#f5a94a", C_CURR, "#30d996"]
    fig.add_trace(go.Bar(x=cats, y=vals, marker_color=clrs,
                         text=[f"{v:.2f}%" for v in vals], textposition="outside",
                         hovertemplate="%{x}<br>%{y:.2f}%<extra></extra>"))
    # RBA baseline
    fig.add_shape(type="line", x0=-0.5, x1=4.5, xref="x", y0=rba, y1=rba, yref="y",
                  line=dict(color=C_ORIG, dash="dash", width=1.5))
    fig.add_annotation(x=4.5, y=rba, text=f"RBA {fp(rba)}", xref="x", yref="y",
                       showarrow=False, xanchor="right", yanchor="bottom",
                       font=dict(color=C_ORIG, size=10))
    fig.update_layout(**PLOT_BASE, title="Rate Stack: Your Loan vs the RBA Benchmark",
                       yaxis_title="Rate (%)", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Tier reference
    st.markdown(
        '<div class="sub-title">Spread Tier Reference</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:6px;font-size:0.75rem">'
        '<div style="background:#0b0f1a;border-left:3px solid #30d996;padding:8px 10px">'
        '<div style="color:#30d996;font-weight:600">Elite</div><div style="color:#8892b0">≤100 bps</div></div>'
        '<div style="background:#0b0f1a;border-left:3px solid #4a9af5;padding:8px 10px">'
        '<div style="color:#4a9af5;font-weight:600">Competitive</div><div style="color:#8892b0">101-170 bps</div></div>'
        '<div style="background:#0b0f1a;border-left:3px solid #f5a94a;padding:8px 10px">'
        '<div style="color:#f5a94a;font-weight:600">Market</div><div style="color:#8892b0">171-220 bps</div></div>'
        '<div style="background:#0b0f1a;border-left:3px solid #e94560;padding:8px 10px">'
        '<div style="color:#e94560;font-weight:600">Expensive</div><div style="color:#8892b0">221-270 bps</div></div>'
        '<div style="background:#0b0f1a;border-left:3px solid #e94560;padding:8px 10px">'
        '<div style="color:#e94560;font-weight:600">Very Expensive</div><div style="color:#8892b0">>270 bps</div></div>'
        '</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="note" style="margin-top:16px">Annual interest at proposed blended rate: '
        f'<strong>{fc(F["prop_annual_interest"])}</strong>'
        f' (on net debt after offset, pre-compounding).</div>',
        unsafe_allow_html=True)

# ── THEME II: Strategy Stress-Test ──────────────────────────────────────
def theme_ii_strategy(R, F):
    ss = st.session_state
    horizon_mo = F["horizon_mo"]
    horizon_yrs = horizon_mo / 12

    st.markdown('<h3 style="color:#4a9af5;font-size:1rem;font-weight:600;margin:4px 0 8px">'
                'Theme II · The Strategic Conflict — Variable vs Fixed vs Split</h3>',
                unsafe_allow_html=True)
    st.markdown(
        f'<div class="note">Stress-test of three configurations over your {horizon_yrs:.1f}-year '
        f'Fixed Period ({horizon_mo} months), holding all other inputs constant. '
        f'Anticipated rate changes and RBA scenario are included.</div>',
        unsafe_allow_html=True)

    # The three scenarios
    scenarios = [
        ("The Optimized Split", f"{F['best_pct']:.0f}% Fixed + {100-F['best_pct']:.0f}% Variable with Offset",
         F["scen_split"], C_SPLIT, "Structural winner — Rate insurance + 100% Offset efficiency"),
        ("100% Variable with Offset", f"Full ${ss.p_loan_amt:,.0f} at Variable with Offset",
         F["scen_var_off"], C_VAR, "The Flexibility Play — no break fees, but exposed to rises"),
        ("100% Fixed, No Offset", f"Full ${ss.p_loan_amt:,.0f} at Fixed, offset disabled",
         F["scen_all_fix"], C_FIX, "Offset Disaster — pay interest on your own cash"),
    ]
    # Sort ascending by interest (winner first)
    scenarios_ranked = sorted(scenarios, key=lambda x: x[2])
    winner = scenarios_ranked[0]
    loser = scenarios_ranked[-1]

    # Hero: disaster delta
    st.markdown(_hero_card(
        "STRUCTURAL WINNER",
        winner[3],
        f'{winner[0]} — <strong>{fc(winner[2])}</strong> interest over {horizon_yrs:.1f} years.',
        f'That\'s <strong>{fc(loser[2] - winner[2])}</strong> less than "{loser[0]}" — '
        f'the cost of choosing the wrong structure.'),
        unsafe_allow_html=True)

    # Chart
    fig = go.Figure()
    names = [s[0] for s in scenarios]
    values = [s[2] for s in scenarios]
    colors = [s[3] for s in scenarios]
    fig.add_trace(go.Bar(
        x=names, y=values, marker_color=colors,
        text=[fc(v) for v in values], textposition="outside",
        hovertemplate="%{x}<br>%{y:$,.0f}<extra></extra>"))
    fig.update_layout(**PLOT_BASE,
                       title=f"Cumulative Interest over {horizon_yrs:.1f}-Year Fixed Period",
                       yaxis_title="Cumulative Interest ($)", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Scenario detail cards
    for name, config, interest, clr, verdict in scenarios:
        delta = interest - winner[2]
        delta_str = "" if delta < 1 else f' — <strong style="color:#e94560">+{fc(delta)}</strong> vs winner'
        st.markdown(
            f'<div style="background:#0b0f1a;border:1px solid #1e2d4a;border-left:3px solid {clr};'
            f'border-radius:6px;padding:12px 14px;margin-bottom:8px">'
            f'<div style="color:{clr};font-size:0.9rem;font-weight:600;margin-bottom:4px">{name}</div>'
            f'<div style="color:#8892b0;font-size:0.82rem;margin-bottom:4px">{config}</div>'
            f'<div style="color:#d4dbe8;font-size:0.88rem"><strong>{fc(interest)}</strong> interest{delta_str}</div>'
            f'<div style="color:#64748b;font-size:0.78rem;margin-top:4px;font-style:italic">{verdict}</div>'
            f'</div>', unsafe_allow_html=True)

    if not F["has_offset"]:
        st.markdown(
            '<div class="note-warn">You have no offset balance entered. The "Offset Disaster" '
            'difference shrinks to zero — but so does the benefit of variable flexibility. '
            'If you have even modest savings, an offset-enabled variable portion materially '
            'reduces effective interest.</div>', unsafe_allow_html=True)

# ── THEME III: Execution & Break-Even ────────────────────────────────────
def theme_iii_execution(R, F):
    ss = st.session_state

    st.markdown('<h3 style="color:#4a9af5;font-size:1rem;font-weight:600;margin:4px 0 8px">'
                'Theme III · Execution — Break-Even Analysis</h3>', unsafe_allow_html=True)
    st.markdown(
        '<div class="note">How many months before the switch pays for itself, and what '
        'you\'d gain after that point.</div>', unsafe_allow_html=True)

    be = F["break_even_mo"]
    if be == float("inf") or be > 240:
        hero_tier = "NO BREAK-EVEN"; hero_clr = "#e94560"
        hero_text = "The proposed loan does not save money over the current loan."
        hero_sub = ("Check that proposed rates are lower than current, or that switching "
                    "costs aren't too high relative to monthly savings.")
    elif be <= 6:
        hero_tier = "RAPID BREAK-EVEN"; hero_clr = "#30d996"
        hero_text = f'Switch pays for itself in <strong>{be:.1f} months</strong>.'
        hero_sub = "Anything after this point is pure interest saving."
    elif be <= 18:
        hero_tier = "GOOD BREAK-EVEN"; hero_clr = "#4a9af5"
        hero_text = f'Switch pays for itself in <strong>{be:.1f} months</strong>.'
        hero_sub = "Within a typical stay-time for the loan."
    elif be <= 36:
        hero_tier = "SLOW BREAK-EVEN"; hero_clr = "#f5a94a"
        hero_text = f'Switch pays for itself in <strong>{be:.1f} months</strong>.'
        hero_sub = "Only worthwhile if you plan to keep this loan for several more years."
    else:
        hero_tier = "MARGINAL"; hero_clr = "#e94560"
        hero_text = f'Break-even after <strong>{be:.1f} months</strong> — longer than typical loan tenure.'
        hero_sub = "Reconsider unless fees can be negotiated down."

    st.markdown(_hero_card(hero_tier, hero_clr, hero_text, hero_sub), unsafe_allow_html=True)

    # Component breakdown
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card("Current Monthly Payment", fc(F["curr_pmt"])),
                    unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("Proposed Monthly Payment", fc(F["prop_pmt"])),
                    unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("Monthly Saving",
                                 fc(F["monthly_saving"]) if F["monthly_saving"] > 0 else "None",
                                 diff_pos=True,
                                 diff="—" if F["monthly_saving"] <= 0 else ""),
                    unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("Total Switch Cost", fc(F["switch_cost"]),
                                 diff_neutral=True),
                    unsafe_allow_html=True)

    # Payback curve
    if F["monthly_saving"] > 0 and be != float("inf"):
        months = np.arange(0, min(60, int(be * 3) + 12))
        cum_saving = np.maximum(0, months * F["monthly_saving"] - F["switch_cost"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=months, y=cum_saving,
            name="Cumulative Net Saving",
            line=dict(color=C_VAR, width=2.5),
            fill="tozeroy", fillcolor="rgba(48,217,150,0.08)",
            hovertemplate="Month %{x}<br>Net: %{y:$,.0f}<extra></extra>"))
        # Break-even marker
        fig.add_shape(type="line", x0=be, x1=be, y0=0, y1=1, yref="paper",
                       line=dict(color=C_FIX, dash="dash", width=1.5))
        fig.add_annotation(x=be, y=0.95, yref="paper", xref="x",
                            text=f"<b>Break-even month {be:.1f}</b>",
                            showarrow=False, font=dict(color=C_FIX, size=11),
                            bgcolor=C_PAPER, bordercolor=C_FIX, borderwidth=1)
        # Zero line
        fig.add_shape(type="line", x0=0, x1=months[-1], y0=0, y1=0,
                       line=dict(color=C_GRID, width=1))
        fig.update_layout(**PLOT_BASE, title="Payback Curve",
                           xaxis_title="Months after Switch", yaxis_title="Cumulative Net Saving ($)")
        st.plotly_chart(fig, use_container_width=True)

    # Switching cost composition
    cost_breakdown = [
        ("Proposed Variable Setup", ss.p_var_fee_setup),
        ("Proposed Fixed Setup", ss.p_fix_fee_setup if not ss.p_fees_match else 0),
        ("Proposed Variable Other", ss.p_var_fee_other),
        ("Proposed Fixed Other", ss.p_fix_fee_other if not ss.p_fees_match else 0),
    ]
    if ss.c_is_cont:
        cost_breakdown.append(("Original Breakage", ss.o_fee_break))
    cost_breakdown = [(n, v) for n, v in cost_breakdown if v > 0]

    if cost_breakdown:
        st.markdown('<div class="sub-title">Switch Cost Breakdown</div>', unsafe_allow_html=True)
        cols = st.columns(len(cost_breakdown))
        for col, (nm, val) in zip(cols, cost_breakdown):
            with col:
                st.markdown(metric_card(nm, fc(val), diff_neutral=True),
                            unsafe_allow_html=True)

# ── THEME IV: Life-of-Loan Forecast ─────────────────────────────────────
def theme_iv_forecast(R, F):
    ss = st.session_state

    st.markdown('<h3 style="color:#4a9af5;font-size:1rem;font-weight:600;margin:4px 0 8px">'
                'Theme IV · Life-of-Loan Forecast — The Snowball Effect</h3>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="note">Compound impact of the rate difference over the entire loan life, '
        'and what happens if you hold your current payment amount after refinancing.</div>',
        unsafe_allow_html=True)

    # Hero: snowball result
    if F["snowball_months_saved"] > 0:
        yrs = F["snowball_months_saved"] / 12
        hero = _hero_card(
            "SNOWBALL RESULT", "#30d996",
            f'By holding your current payment of <strong>{fc(F["curr_pmt"])}</strong>/mo '
            f'after switching, you pay the loan off <strong>{F["snowball_months_saved"]} months '
            f'({yrs:.1f} years) earlier</strong>.',
            f'Saving an extra <strong>{fc(F["snowball_interest_saved"])}</strong> in interest '
            f'beyond the baseline refinance.'
        )
        st.markdown(hero, unsafe_allow_html=True)
    elif F["lifetime_saving"] > 0:
        hero = _hero_card(
            "LIFETIME SAVING", "#4a9af5",
            f'Refinancing saves <strong>{fc(F["lifetime_saving"])}</strong> in total interest over the life of the loan.',
            'If you also hold your current payment after switching, you\'d save more time and interest — adjust inputs to see.'
        )
        st.markdown(hero, unsafe_allow_html=True)
    else:
        hero = _hero_card(
            "NO LIFETIME SAVING", "#e94560",
            'Refinancing doesn\'t reduce lifetime interest given current inputs.',
            'Check that proposed rates are materially below current, and that offset utilisation is realistic.'
        )
        st.markdown(hero, unsafe_allow_html=True)

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card("Current Loan Total Interest", fc(F["total_int_curr"])),
                    unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("Proposed Loan Total Interest", fc(F["total_int_prop"]),
                                 diff=f"{'−' if F['lifetime_saving']>0 else '+'}{fc(abs(F['lifetime_saving']))}",
                                 diff_pos=(F['lifetime_saving']>0)),
                    unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("With Snowball (pay current amount)",
                                 fc(F["total_int_prop"] - F["snowball_interest_saved"])
                                    if F["snowball_interest_saved"] > 0 else fc(F["total_int_prop"]),
                                 diff=f"−{fc(F['snowball_interest_saved'])}" if F["snowball_interest_saved"] > 0 else "",
                                 diff_pos=True),
                    unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("Months Saved (snowball)",
                                 f"{F['snowball_months_saved']} mo" if F["snowball_months_saved"] > 0 else "—",
                                 diff=f"{F['snowball_months_saved']/12:.1f} yrs earlier" if F["snowball_months_saved"] > 0 else "",
                                 diff_pos=True),
                    unsafe_allow_html=True)

    # Balance over time: current vs proposed vs proposed+snowball
    df_c, df_ps = R["df_curr"], R["df_ps"]
    if df_c is not None and not df_c.empty and df_ps is not None and not df_ps.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_c["Date"], y=df_c["Closing Balance"],
                                  name="Current Loan", line=dict(color=C_CURR, width=2),
                                  hovertemplate="Current<br>%{x|%b %Y}<br>$%{y:,.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=df_ps["Date"], y=df_ps["Closing Balance"],
                                  name="Proposed Loan (min payment)",
                                  line=dict(color=C_SPLIT, width=2),
                                  fill="tozeroy", fillcolor="rgba(196,122,245,0.05)",
                                  hovertemplate="Proposed<br>%{x|%b %Y}<br>$%{y:,.0f}<extra></extra>"))
        # Snowball overlay
        try:
            if F["curr_pmt"] > F["prop_pmt"]:
                p_var_deltas = list(ss.future_var_deltas)
                vsched = build_rate_schedule(ss.p_adv_var_rate, p_var_deltas)
                df_snow = amortize(ss.p_loan_amt, ss.p_start_date, ss.p_term_mo,
                                    vsched, F["off_prop"], ss.p_off_monthly,
                                    deltas_to_lumps_t(ss.p_off_lumps, ss.p_start_date),
                                    deltas_to_lumps_t(ss.p_extra_repay, ss.p_start_date),
                                    ss.p_var_fee_mo, True, min_pmt_floor=F["curr_pmt"])
                if not df_snow.empty:
                    fig.add_trace(go.Scatter(x=df_snow["Date"], y=df_snow["Closing Balance"],
                                              name="Proposed + Snowball (hold current payment)",
                                              line=dict(color=C_VAR, width=2, dash="dot"),
                                              hovertemplate="Snowball<br>%{x|%b %Y}<br>$%{y:,.0f}<extra></extra>"))
        except Exception:
            pass
        fig.update_layout(**PLOT_BASE,
                           title="Balance Paydown Over Time",
                           yaxis_title="Outstanding Balance ($)")
        fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.04))
        st.plotly_chart(fig, use_container_width=True)

# ── THEME V: Interrogation Checklist ────────────────────────────────────
def theme_v_checklist(R, F):
    ss = st.session_state

    st.markdown('<h3 style="color:#4a9af5;font-size:1rem;font-weight:600;margin:4px 0 8px">'
                'Theme V · Lender Interrogation Checklist</h3>', unsafe_allow_html=True)
    st.markdown(
        '<div class="note">Questions you must get "Yes" answers to, derived from your '
        'specific loan configuration. Copy/paste these into your broker or lender brief.</div>',
        unsafe_allow_html=True)

    # Build a dynamic list
    questions = []

    # Q1: Offset
    if F["has_offset"]:
        questions.append({
            "q": "Is the offset dollar-for-dollar?",
            "rationale": f'You have ${F["off_prop"]:,.0f} in the proposed offset. '
                         f'Confirm interest is calculated daily on the net balance (loan − offset), '
                         f'not just a partial benefit.',
            "critical": True,
        })
    # Q2: Term preservation
    orig_term = ss.o_term_mo
    rem_years = F["c_term_rem"] / 12
    questions.append({
        "q": f'Can I fix the remaining term at {rem_years:.0f} years (≈ {F["c_term_rem"]} months)?',
        "rationale": f'Lenders often reset to 25 or 30 years, which lowers minimum payment '
                     f'but extends total interest. Your current loan has {F["c_term_rem"]} months '
                     f'remaining. If they reset, manually set your direct debit to {fc(F["curr_pmt"])}.',
        "critical": True,
    })
    # Q3: Split with offset
    if F["has_split"] and F["has_offset"]:
        questions.append({
            "q": f'Does your {ss.p_adv_fix_rate:.2f}% Fixed rate support a split facility '
                 f'with a separate Offset account on the variable side?',
            "rationale": f'Your optimal strategy is {F["best_pct"]:.0f}% Fixed + '
                         f'{100-F["best_pct"]:.0f}% Variable-with-Offset. Many lenders '
                         f'offer the headline fixed rate BUT exclude offset entirely. '
                         f'This is the deal-breaker question.',
            "critical": True,
        })
    # Q4: Extra repayments on fixed
    if F["has_fixed"]:
        questions.append({
            "q": f'What is the annual cap on extra repayments for the Fixed component?',
            "rationale": f'Ensure at least $10,000/year (or 5% of the fixed balance) can be paid '
                         f'extra without penalty. Needed if your offset "overflows" during the fixed period.',
            "critical": False,
        })
    # Q5: Establishment waiver
    if ss.p_var_fee_setup > 0 or ss.p_fix_fee_setup > 0:
        total_setup = ss.p_var_fee_setup + (0 if ss.p_fees_match else ss.p_fix_fee_setup)
        questions.append({
            "q": f'Will you waive the Establishment / Setup Fee ({fc(total_setup)}) given a '
                 f'{fc(ss.p_loan_amt)} refinance?',
            "rationale": f'High-value borrowers should not pay for the privilege of switching. '
                         f'This is routinely waived for loans over $500k.',
            "critical": False,
        })
    # Q6: Breakage fee clarity (if currently fixed)
    if ss.c_is_cont and ss.o_fee_break > 0:
        questions.append({
            "q": f'Break-fee payable ({fc(ss.o_fee_break)}) — request written confirmation of '
                 f'exact figure before switching.',
            "rationale": f'Break fees on fixed loans vary with market movement and can spike at signing. '
                         f'Get the figure in writing within 48h of your refinance settlement date.',
            "critical": True,
        })
    # Q7: Rate lock
    if F["has_fixed"]:
        questions.append({
            "q": f'Do you offer rate-lock on the {ss.p_adv_fix_rate:.2f}% Fixed rate between '
                 f'application and settlement? Fee?',
            "rationale": f'Fixed rates can move between application (today) and settlement '
                         f'({ss.p_start_date.strftime("%d %b %Y")}). Rate lock typically costs '
                         f'$500-$1000 but can be worth it if the market is trending up.',
            "critical": False,
        })
    # Q8: Rate match / negotiation
    questions.append({
        "q": f'If I bring a competitor\'s offer of {fp(F["p_blended_rate"])} in writing, '
             f'will you match or beat it?',
        "rationale": f'Lenders frequently match competitive offers to retain business. '
                     f'Always get a written competing quote before signing.',
        "critical": False,
    })

    # Render as styled checklist
    for i, q in enumerate(questions, 1):
        border = "#e94560" if q["critical"] else "#4a9af5"
        tag = "CRITICAL" if q["critical"] else "IMPORTANT"
        tag_clr = "#e94560" if q["critical"] else "#4a9af5"
        st.markdown(
            f'<div style="background:#0b0f1a;border:1px solid #1e2d4a;border-left:3px solid {border};'
            f'border-radius:6px;padding:12px 14px;margin-bottom:10px">'
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">'
            f'<span style="background:{tag_clr};color:#fff;font-size:0.65rem;font-weight:700;'
            f'padding:2px 8px;border-radius:3px;letter-spacing:.05em">{tag}</span>'
            f'<span style="color:#8892b0;font-size:0.75rem">Q{i}</span></div>'
            f'<div style="color:#d4dbe8;font-size:0.92rem;font-weight:500;margin-bottom:6px">{q["q"]}</div>'
            f'<div style="color:#8892b0;font-size:0.82rem;line-height:1.5">{q["rationale"]}</div>'
            f'</div>', unsafe_allow_html=True)

    # Final forensic conclusion
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<h3 style="color:#4a9af5;font-size:1rem;font-weight:600;margin:14px 0 8px">'
                'Final Forensic Conclusion</h3>', unsafe_allow_html=True)

    # Build recommendation narrative
    rec_parts = []
    if F["lifetime_saving"] > 0 and F["break_even_mo"] < 24:
        rec_parts.append(
            f'<strong>Proceed with the refinance.</strong> Your proposed structure saves '
            f'<strong>{fc(F["lifetime_saving"])}</strong> in lifetime interest with break-even '
            f'at <strong>{F["break_even_mo"]:.1f} months</strong>.')
    elif F["lifetime_saving"] > 0:
        rec_parts.append(
            f'<strong>Refinance is marginally positive.</strong> Lifetime saving '
            f'<strong>{fc(F["lifetime_saving"])}</strong> but break-even is slow at '
            f'<strong>{F["break_even_mo"]:.1f} months</strong>. Only proceed if you plan to '
            f'keep this loan for 3+ years.')
    else:
        rec_parts.append(
            f'<strong>Reconsider.</strong> Proposed structure does not save lifetime interest '
            f'given current inputs. Review proposed rates and offset utilisation.')

    if F["has_split"]:
        rec_parts.append(
            f'Execute a <strong>{F["best_pct"]:.0f}% Fixed / {100-F["best_pct"]:.0f}% Variable</strong> '
            f'split — this is mathematically optimal given your rate inputs and Fixed Period.')

    prop_tier_lbl, prop_tier_clr = F["prop_eff_tier"]
    rec_parts.append(
        f'This minimises your RBA spread to <strong style="color:{prop_tier_clr}">'
        f'{F["prop_eff_spread_bps"]:.0f} bps ({prop_tier_lbl})</strong>, '
        f'an effective rate of <strong>{fp(F["prop_eff_after_off"])}</strong> after offset.')

    if F["snowball_months_saved"] > 0:
        rec_parts.append(
            f'By holding your current payment of <strong>{fc(F["curr_pmt"])}</strong>/mo, '
            f'you pay off <strong>{F["snowball_months_saved"]/12:.1f} years earlier</strong> '
            f'and save an additional <strong>{fc(F["snowball_interest_saved"])}</strong>.')

    narrative = " ".join(rec_parts)
    st.markdown(
        f'<div style="background:#0b0f1a;border:2px solid #4a9af5;border-radius:8px;'
        f'padding:18px 22px;margin-top:10px">'
        f'<div style="color:#d4dbe8;font-size:0.95rem;line-height:1.7">{narrative}</div>'
        f'</div>', unsafe_allow_html=True)

    # Copy-ready text version
    with st.expander("Plain-text version (copy to broker email / document)", expanded=False):
        import re as _re
        plain = _re.sub(r'<[^>]+>', '', narrative)
        plain = plain.replace("&nbsp;", " ")
        st.code(plain, language=None)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    init_state()
    ss = st.session_state
    load_live_data()

    rba_str = (f" · RBA: <span style='color:#4a9af5;font-weight:600'>{fp(ss._rba_rate)}</span>"
               if ss._rba_rate else "")
    meeting_str = (f" · Next Decision: <span style='color:#64748b'>{ss._rba_next_meeting}</span>"
                   if ss._rba_next_meeting else "")
    st.markdown(f"""
    <div style="padding:20px 0 16px;border-bottom:1px solid #1e2d4a;margin-bottom:20px;">
        <h1 style="color:#d4dbe8;font-size:1.5rem;font-weight:600;margin:0 0 4px;">
            Australian Mortgage Refinance Analyser
        </h1>
        <p style="color:#64748b;font-size:0.79rem;margin:0;">
            Daily-interest amortisation &nbsp;·&nbsp; Real-time analysis
            &nbsp;·&nbsp; Optimal split (0.1% increments)
            &nbsp;·&nbsp; ASIC comparison rates{rba_str}{meeting_str}
        </p>
    </div>""", unsafe_allow_html=True)

    with st.expander("Original Loan", expanded=True):
        section_original()
    with st.expander("Current Loan", expanded=True):
        section_current()
    with st.expander("Proposed Loan", expanded=True):
        section_proposed()
    with st.expander("Scenarios", expanded=True):
        section_scenarios()

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    c1, _ = st.columns([1, 7])
    with c1:
        st.markdown('<div class="btn-danger">', unsafe_allow_html=True)
        if st.button("Reset All", key="btn_rst"):
            for k in list(ss.keys()): del ss[k]
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<h2 style="color:#d4dbe8;font-size:1.1rem;font-weight:600;margin:0 0 14px">'
                'Analysis Dashboard</h2>', unsafe_allow_html=True)

    with st.spinner("Computing..."):
        R = compute_all()

    if R is None:
        st.markdown("""
        <div style="text-align:center;padding:48px 0;color:#64748b;">
            <div style="font-size:.92rem;font-weight:500;color:#8892b0;margin-bottom:6px;">
                Enter loan details above — analysis updates in real time
            </div>
            <div style="font-size:.8rem">Ensure Original Loan Amount and Remaining Balance are positive</div>
        </div>""", unsafe_allow_html=True)
        return

    F = forensic_compute(R)

    tabs = st.tabs([
        "I · Macro Anchor",
        "II · Strategic Conflict",
        "III · Execution & Break-Even",
        "IV · Life-of-Loan Forecast",
        "V · Lender Checklist",
    ])
    with tabs[0]: theme_i_anchor(R, F)
    with tabs[1]: theme_ii_strategy(R, F)
    with tabs[2]: theme_iii_execution(R, F)
    with tabs[3]: theme_iv_forecast(R, F)
    with tabs[4]: theme_v_checklist(R, F)


if __name__ == "__main__":
    main()
