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
    Fetch next RBA Monetary Policy Board meeting date.
    Tries ICS calendar, then HTML list page; parses only future dates.
    """
    candidates = []

    # ── Attempt 1: ICS feed for monetary-policy-board ─────────────────
    ics_urls = [
        "https://www.rba.gov.au/schedules-events/calendar.ics",
        "https://www.rba.gov.au/schedules-events/calendar/?topics=monetary-policy-board&view=list&format=ics",
    ]
    for url in ics_urls:
        try:
            r = requests.get(url, headers=HDRS, timeout=10)
            if r.status_code == 200 and "BEGIN:VEVENT" in r.text:
                events = re.findall(r'BEGIN:VEVENT(.*?)END:VEVENT', r.text, re.DOTALL)
                for e in events:
                    summary = re.search(r'SUMMARY:(.+)', e)
                    if not summary: continue
                    s = summary.group(1).strip().lower()
                    if "monetary" not in s and "board" not in s: continue
                    dstart = re.search(r'DTSTART[^:]*:(\d{8})', e)
                    if dstart:
                        try:
                            dt = datetime.strptime(dstart.group(1), "%Y%m%d").date()
                            if dt >= TODAY: candidates.append(dt)
                        except: pass
        except: pass

    # ── Attempt 2: HTML parse of the calendar list page ───────────────
    if not candidates:
        try:
            r = requests.get(
                "https://www.rba.gov.au/schedules-events/calendar/?topics=monetary-policy-board&view=list",
                headers=HDRS, timeout=10)
            if r.status_code == 200:
                # Events in list view are wrapped in article/section with date inside
                # Look for pattern: date followed by "Monetary Policy Board"
                # Dates appear as "1 April 2026" or "Tuesday 1 April 2026"
                months = (r"January|February|March|April|May|June|July|August|"
                         r"September|October|November|December")
                patterns = [
                    rf"(\d{{1,2}})\s+({months})\s+(\d{{4}})[^<]*?(?:Monetary\s+Policy|Board)",
                    rf"(?:Monetary\s+Policy|Board)[^<]*?(\d{{1,2}})\s+({months})\s+(\d{{4}})",
                    rf"<time[^>]*datetime=\"(\d{{4}}-\d{{2}}-\d{{2}})",
                ]
                for pat in patterns:
                    for m in re.findall(pat, r.text):
                        try:
                            if isinstance(m, tuple) and len(m) == 3:
                                day, mon, yr = m
                                dt = datetime.strptime(f"{day} {mon} {yr}", "%d %B %Y").date()
                            else:
                                ds = m if isinstance(m, str) else m[0]
                                dt = date.fromisoformat(ds)
                            if dt >= TODAY: candidates.append(dt)
                        except: pass
        except: pass

    # ── Attempt 3: Scrape schedules page for board meeting schedule ──
    if not candidates:
        try:
            r = requests.get("https://www.rba.gov.au/schedules-events/",
                             headers=HDRS, timeout=10)
            if r.status_code == 200:
                # Very permissive — find any future Tuesday/Wednesday that could be a meeting
                months = (r"January|February|March|April|May|June|July|August|"
                         r"September|October|November|December")
                for m in re.findall(rf"(\d{{1,2}}\s+(?:{months})\s+\d{{4}})", r.text):
                    try:
                        dt = datetime.strptime(m, "%d %B %Y").date()
                        if dt >= TODAY and dt <= add_months(TODAY, 6):
                            candidates.append(dt)
                    except: pass
        except: pass

    if candidates:
        return min(candidates).strftime("%d %B %Y")
    return None

@st.cache_data(ttl=300, show_spinner=False)
def fetch_asx_rba_data():
    """
    Fetch ASX 30-day interbank cash rate futures / RBA probability data.
    Priorities: rba.isaacgross.net (aggregated data), ASX JSON, fallback empty.
    """
    result = {}
    ig_base = "https://rba.isaacgross.net"
    ig_hdrs = {**HDRS, "Accept": "application/json",
               "Referer": "https://rba.isaacgross.net/"}

    for path in ["/api/rates", "/api", "/api/v1/rates", "/api/cash-rate",
                 "/api/futures", "/api/probability", "/rates", "/data"]:
        try:
            r = requests.get(f"{ig_base}{path}", headers=ig_hdrs, timeout=6)
            if r.status_code == 200 and "json" in r.headers.get("Content-Type", ""):
                data = r.json()
                result.update({"source": "isaacgross", "data": data})
                if isinstance(data, dict):
                    for key in ["futures_price", "ib_price", "price", "implied_rate", "rate"]:
                        if key in data:
                            try: result["futures_price"] = float(data[key]); break
                            except: pass
                    for key in ["probability", "prob", "rate_change_prob"]:
                        if key in data:
                            try: result["probability"] = float(data[key]); break
                            except: pass
                elif isinstance(data, list) and data:
                    result["raw_list"] = data
                break
        except: pass

    if "futures_price" not in result:
        try:
            r = requests.get(ig_base, headers=HDRS, timeout=8)
            if r.status_code == 200:
                patterns = [r'"futures_price"\s*:\s*([\d.]+)',
                           r'"ib_price"\s*:\s*([\d.]+)',
                           r'"price"\s*:\s*([\d.]+)',
                           r'(9[4-9]\.\d{2})']
                for pat in patterns:
                    mm = re.search(pat, r.text)
                    if mm:
                        try:
                            v = float(mm.group(1))
                            if 90 < v < 100:
                                result["futures_price"] = v
                                result["implied_yield"] = round(100 - v, 4)
                                result["source"] = "isaacgross_page"
                                break
                        except: pass
                for pat in [r'"probability"\s*:\s*([\d.]+)', r'"prob"\s*:\s*([\d.]+)']:
                    mm = re.search(pat, r.text)
                    if mm:
                        try: result["probability"] = float(mm.group(1)); break
                        except: pass
        except: pass

    if not result:
        for url in ["https://www.asx.com.au/asx/1/exchange/IB/prices",
                    "https://www.asx.com.au/data/trendlens_snapshot/IB.json"]:
            try:
                r = requests.get(url, headers=HDRS, timeout=6)
                if r.status_code == 200:
                    data = r.json()
                    result = {"source": "asx_api", "data": data}; break
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
    _d("strategy", "Balanced (optimal split)")
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
        ss._asx_data = fetch_asx_rba_data()
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
    ss = st.session_state
    rate = ss._rba_rate
    meeting = ss._rba_next_meeting

    with st.expander("RBA Cash Rate and Market Indicators", expanded=False):
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
                        st.markdown(f'<div style="color:#64748b;font-size:0.78rem">{days} days from today</div>',
                                    unsafe_allow_html=True)
                except: pass
            else:
                st.markdown(
                    '<div style="color:#64748b;font-size:0.78rem">Unable to parse. Visit '
                    '<a href="https://www.rba.gov.au/schedules-events/" target="_blank" '
                    'style="color:#4a9af5">RBA Schedules and Events</a></div>',
                    unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c3:
            st.markdown('<div class="data-panel">'
                        '<div class="data-panel-title">ASX Rate Tracker</div>',
                        unsafe_allow_html=True)
            asx = ss._asx_data
            if asx and "futures_price" in asx:
                fp_val = asx["futures_price"]
                iy_val = round(100 - fp_val, 4)
                src = asx.get("source", "")
                st.markdown(
                    f'<div style="color:#30d996;font-size:0.82rem;font-weight:600">'
                    f'IB Futures: {fp_val:.2f}</div>'
                    f'<div style="color:#4a9af5;font-size:1.1rem;font-weight:700">'
                    f'Implied Rate: {iy_val:.2f}%</div>'
                    f'<div style="color:#64748b;font-size:0.7rem">Source: {src}</div>',
                    unsafe_allow_html=True)
                if "probability" in asx:
                    prob = asx["probability"] * 100 if asx["probability"] <= 1 else asx["probability"]
                    st.markdown(
                        f'<div style="color:#f5a94a;font-size:0.85rem;margin-top:4px">'
                        f'Rate change probability: <strong>{prob:.1f}%</strong></div>',
                        unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div style="color:#64748b;font-size:0.78rem">Auto-fetch unavailable. '
                    'Enter IB futures price manually below, or visit '
                    '<a href="https://www.asx.com.au/markets/trade-our-derivatives-market/'
                    'futures-market/rba-rate-tracker" target="_blank" '
                    'style="color:#4a9af5">ASX Rate Tracker</a> or '
                    '<a href="https://rba.isaacgross.net/" target="_blank" '
                    'style="color:#4a9af5">rba.isaacgross.net</a></div>',
                    unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Probability Calculator
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            '<div style="color:#64748b;font-size:0.72rem;font-weight:600;text-transform:uppercase;'
            'letter-spacing:.06em;margin-bottom:10px">ASX Target Rate Probability '
            '(30-Day Interbank Cash Rate Futures)</div>',
            unsafe_allow_html=True)
        _asx = ss._asx_data
        _def_ib = float(_asx.get("futures_price", 95.65)) if _asx else 95.65
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            ib_price = st.number_input("IB Futures Price", value=_def_ib,
                                       min_value=90.0, max_value=100.0,
                                       step=0.01, format="%.2f",
                                       help="30-Day Interbank Cash Rate Futures price (e.g. 95.65 → implied 4.35%).")
            implied_yield = round(100 - ib_price, 4)
            computed("Implied Yield", fp(implied_yield), "= 100 − Futures Price")
        with c2:
            rt = st.number_input("Current Target Rate (%)", value=float(rate) if rate else 4.35,
                                 min_value=0.0, max_value=30.0, step=0.25, format="%.2f",
                                 help="Current RBA Target Cash Rate (less overnight differential).")
        with c3:
            rt1 = st.number_input("Expected New Rate (%)", value=round((rate or 4.35) - 0.25, 2),
                                  min_value=0.0, max_value=30.0, step=0.25, format="%.2f",
                                  help="Expected rate if RBA moves (typically ±0.25%).")
        with c4:
            nb_days = st.number_input("Days before RBA meeting", value=5,
                                      min_value=1, max_value=30,
                                      help="Days in current month before the RBA Board meeting.")
        try:
            days_in_month = 30
            nb = nb_days / days_in_month
            na = (days_in_month - nb_days) / days_in_month
            X = implied_yield
            denom = na * (rt1 - rt)
            if abs(denom) > 1e-9:
                p = round((X - rt * (nb + na)) / denom, 4)
                p = max(0.0, min(1.0, p))
                pct = p * 100
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    trend_clr = "#e94560" if rt1 > rt else "#30d996"
                    direction = "increase" if rt1 > rt else "decrease"
                    st.markdown(
                        f'<div class="cf"><div class="cf-lbl">Probability of rate {direction}</div>'
                        f'<div class="cf-val" style="color:{trend_clr}">{pct:.1f}%</div>'
                        f'<div class="cf-sub">p = (X − rt(nb+na)) / (na×(r(t+1)−rt))</div></div>',
                        unsafe_allow_html=True)
                with col_b:
                    outlook = ("Increasing" if (pct > 60 and rt1 > rt)
                              else ("Decreasing" if (pct > 60 and rt1 < rt) else "Stable / Uncertain"))
                    oclr = {"Increasing": "#e94560", "Decreasing": "#30d996",
                           "Stable / Uncertain": "#64748b"}[outlook]
                    st.markdown(
                        f'<div class="cf"><div class="cf-lbl">Rate Outlook</div>'
                        f'<div class="cf-val" style="color:{oclr}">{outlook}</div></div>',
                        unsafe_allow_html=True)
                with col_c:
                    st.markdown(
                        f'<div class="cf"><div class="cf-lbl">nb / na fraction of month</div>'
                        f'<div class="cf-val">{nb:.2f} / {na:.2f}</div></div>',
                        unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Calculation error: {e}")

        # History chart
        if ss._rba_history:
            hist = ss._rba_history[-40:] if len(ss._rba_history) > 40 else ss._rba_history
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[r["date"] for r in hist], y=[r["rate"] for r in hist],
                name="Cash Rate Target", line=dict(color=C_ORIG, width=2, shape="hv"),
                fill="tozeroy", fillcolor="rgba(74,154,245,0.06)",
                hovertemplate="<b>%{y:.2f}%</b><br>%{x|%d %b %Y}<extra></extra>"))
            fig.update_layout(**PLOT_BASE, title="RBA Cash Rate History",
                              yaxis_title="Rate (%)")
            fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.04))
            st.plotly_chart(fig, use_container_width=True)

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

    # ── Interest Rates ────────────────────────────────────────────────────
    sec("Interest Rates")

    # Match fee toggle ABOVE rates display (so computed rates reflect chosen fees)
    ss.p_fees_match = st.toggle(
        "Match Fixed Rate fees to Variable Rate fees",
        value=ss.p_fees_match, key="w_p_fm",
        help="When enabled, Fixed component fees equal Variable fees. Disable to set them individually.")

    # Compute both sets of rates using per-component fees
    if ss.p_fees_match:
        fix_mo, fix_setup = ss.p_var_fee_mo, ss.p_var_fee_setup
    else:
        fix_mo, fix_setup = ss.p_fix_fee_mo, ss.p_fix_fee_setup

    comp_var = comparison_rate_asic(ss.p_var_fee_setup, ss.p_var_fee_mo, ss.p_adv_var_rate)
    eff_var = effective_rate_calc(ss.p_loan_amt, ss.p_var_fee_setup,
                                   ss.p_var_fee_mo, ss.p_adv_var_rate, ss.p_term_mo)
    comp_fix = comparison_rate_asic(fix_setup, fix_mo, ss.p_adv_fix_rate)
    # Fixed-rate effective: calculated over fixed period only
    fix_mo_period = ss.p_fix_yrs * 12
    eff_fix = effective_rate_calc(ss.p_loan_amt, fix_setup, fix_mo,
                                   ss.p_adv_fix_rate, fix_mo_period)

    if not ss.p_rev_rate_override:
        ss.p_rev_rate = round(eff_var, 4)

    h1, h2 = st.columns(2)
    h1.markdown('<div style="text-align:center;color:#30d996;font-size:0.75rem;font-weight:600;'
                'padding:6px;background:#071a0f;border-radius:5px;margin-bottom:8px">'
                'VARIABLE RATE</div>', unsafe_allow_html=True)
    h2.markdown('<div style="text-align:center;color:#e94560;font-size:0.75rem;font-weight:600;'
                'padding:6px;background:#1a0709;border-radius:5px;margin-bottom:8px">'
                'FIXED RATE</div>', unsafe_allow_html=True)

    c_v, c_f = st.columns(2)
    with c_v:
        ss.p_adv_var_rate = st.number_input(
            "Advertised Variable Rate (% p.a.)", value=ss.p_adv_var_rate,
            min_value=0.0, max_value=30.0, step=0.01, format="%.4f", key="w_p_avr",
            help="Headline variable rate advertised by the lender, before fees.")
        computed("Comparison Variable Rate", fp(comp_var),
                 "ASIC standard: $150k over 25yr, using Variable fees")
        computed("Effective Variable Rate", fp(eff_var),
                 f"For ${ss.p_loan_amt:,.0f} over {ss.p_term_mo} months, using Variable fees")
    with c_f:
        ss.p_adv_fix_rate = st.number_input(
            "Advertised Fixed Rate (% p.a.)", value=ss.p_adv_fix_rate,
            min_value=0.0, max_value=30.0, step=0.01, format="%.4f", key="w_p_afr",
            help="Headline fixed rate advertised for the fixed period.")
        computed("Comparison Fixed Rate", fp(comp_fix),
                 "ASIC standard: $150k over 25yr, using Fixed fees")
        computed("Effective Fixed Rate", fp(eff_fix),
                 f"For ${ss.p_loan_amt:,.0f} over Fixed Period ({fix_mo_period} months), using Fixed fees")

    c1, c2 = st.columns(2)
    with c1:
        ss.p_fix_yrs = st.number_input(
            "Fixed Period (years)", value=ss.p_fix_yrs, min_value=1, max_value=30, step=1,
            key="w_p_fy",
            help="Duration that the Fixed Rate applies. Applies ONLY to the Fixed component "
                 "of the loan. Variable component uses the variable rate throughout.")
    with c2:
        ss.p_rev_rate_override = st.toggle(
            "Override reversion rate", value=ss.p_rev_rate_override, key="w_p_rro",
            help="By default the reversion rate equals the Effective Variable Rate. "
                 "Enable to set a custom (typically lower) rate.")
        if ss.p_rev_rate_override:
            ss.p_rev_rate = st.number_input("Reversion Rate (% p.a.)", value=ss.p_rev_rate,
                min_value=0.0, max_value=30.0, step=0.01, format="%.4f", key="w_p_rr_ov",
                help="Rate the Fixed component reverts to after the Fixed Period expires.")
        else:
            computed("Reversion Rate", fp(ss.p_rev_rate),
                     "Auto-filled to Effective Variable Rate")

    # ── Fees (MOVED UP: before Variable/Fixed Split) ──────────────────────
    sec("Fees")
    st.markdown('<div class="sub-title">Variable Rate Fees</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ss.p_var_fee_mo = st.number_input("Monthly Fee ($)", value=ss.p_var_fee_mo,
            min_value=0.0, step=1.0, key="w_p_var_fm")
    with c2:
        ss.p_var_fee_setup = st.number_input("Setup Fee ($)", value=ss.p_var_fee_setup,
            min_value=0.0, step=100.0, key="w_p_var_fs")
    with c3:
        ss.p_var_fee_break = st.number_input("Breakage Fee ($)", value=ss.p_var_fee_break,
            min_value=0.0, step=100.0, key="w_p_var_fb")
    with c4:
        ss.p_var_fee_other = st.number_input("Other One-off Fee ($)", value=ss.p_var_fee_other,
            min_value=0.0, step=100.0, key="w_p_var_fo")

    st.markdown('<div class="sub-title">Fixed Rate Fees</div>', unsafe_allow_html=True)
    if ss.p_fees_match:
        ss.p_fix_fee_mo = ss.p_var_fee_mo
        ss.p_fix_fee_setup = ss.p_var_fee_setup
        ss.p_fix_fee_break = ss.p_var_fee_break
        ss.p_fix_fee_other = ss.p_var_fee_other
        c1, c2, c3, c4 = st.columns(4)
        with c1: computed("Monthly Fee", fc(ss.p_fix_fee_mo), "matched to Variable")
        with c2: computed("Setup Fee", fc(ss.p_fix_fee_setup), "matched to Variable")
        with c3: computed("Breakage Fee", fc(ss.p_fix_fee_break), "matched to Variable")
        with c4: computed("Other Fee", fc(ss.p_fix_fee_other), "matched to Variable")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            ss.p_fix_fee_mo = st.number_input("Monthly Fee ($)", value=ss.p_fix_fee_mo,
                min_value=0.0, step=1.0, key="w_p_fix_fm")
        with c2:
            ss.p_fix_fee_setup = st.number_input("Setup Fee ($)", value=ss.p_fix_fee_setup,
                min_value=0.0, step=100.0, key="w_p_fix_fs")
        with c3:
            ss.p_fix_fee_break = st.number_input("Breakage Fee ($)", value=ss.p_fix_fee_break,
                min_value=0.0, step=100.0, key="w_p_fix_fb")
        with c4:
            ss.p_fix_fee_other = st.number_input("Other One-off Fee ($)", value=ss.p_fix_fee_other,
                min_value=0.0, step=100.0, key="w_p_fix_fo")

    # ── Variable / Fixed Split ────────────────────────────────────────────
    sec("Variable / Fixed Split")
    ss.p_split_auto = st.toggle(
        "Auto-calculate optimal split", value=ss.p_split_auto, key="w_p_sa",
        help="The optimal split minimises cumulative interest + closing balance at end of "
             "the Fixed Period. Evaluated in 0.1% increments across 1,001 combinations.")
    if not ss.p_split_auto:
        c1, c2 = st.columns(2)
        with c1:
            ss.p_split_pct = st.slider("Fixed Component (%)", 0.0, 100.0,
                ss.p_split_pct, 0.5, key="w_p_sp")
        with c2:
            computed("Allocation",
                     f"Variable {fc(ss.p_loan_amt*(100-ss.p_split_pct)/100)} / "
                     f"Fixed {fc(ss.p_loan_amt*ss.p_split_pct/100)}")

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
        help="When enabled, the Current Balance of the proposed offset equals the Current "
             "Loan's offset initial balance.")
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
            min_value=0.0, step=100.0, key="w_p_om")
    lump_list("p_off_lumps", "Offset Lump Sum Deposits")

    # Extra Repayments / Redraws
    extra_repay_list("p_extra_repay", "Extra Repayments and Redraws")

# ═══════════════════════════════════════════════════════════════════════════════
# INPUT: STRATEGY
# ═══════════════════════════════════════════════════════════════════════════════

def section_strategy():
    ss = st.session_state
    rba_rate = ss._rba_rate

    if rba_rate:
        st.markdown(
            f'<div class="data-panel">'
            f'<div class="data-panel-title">Current RBA Cash Rate</div>'
            f'<span style="font-size:1.8rem;font-weight:700;color:#4a9af5">{fp(rba_rate)}</span>'
            f'</div>', unsafe_allow_html=True)

    if ss._rba_next_meeting:
        st.markdown(
            f'<div class="note">Next RBA Monetary Policy Decision: '
            f'<strong>{ss._rba_next_meeting}</strong></div>',
            unsafe_allow_html=True)

    sec("Strategy (User Defined)")
    strategies = ["Conservative (80% fixed)", "Balanced (optimal split)", "Aggressive (0% fixed)"]
    if ss.strategy not in strategies:
        ss.strategy = "Balanced (optimal split)"
    ss.strategy = st.radio(
        "Select refinancing strategy", strategies, index=strategies.index(ss.strategy),
        key="w_strat", horizontal=True,
        help="User-defined preference. Dashboard shows all three strategies regardless of selection.")

    sec("Payment Behaviour on Rate Changes")
    ss.maintain_pmt = st.toggle(
        "When rates fall, maintain current repayment (pays off faster)",
        value=ss.maintain_pmt, key="w_mp",
        help="When rates rise, repayments always increase to maintain remaining term (term can only "
             "decrease). When rates fall: ON → keep paying current amount, loan shortens. "
             "OFF → drop to new minimum, loan term preserved.")
    st.markdown(
        '<div class="note">This setting takes effect in scenarios where rates fall '
        '(e.g. −0.25%, −0.50% scenarios on the Rate Scenarios tab, or negative RBA movements, '
        'or negative anticipated rate changes in the Current Loan section).</div>',
        unsafe_allow_html=True)

    sec("RBA Cash Rate Scenario (on top of any changes in Current / Proposed sections)")
    c1, c2 = st.columns([4, 1])
    with c1:
        ss.rba_bps = st.slider(
            "RBA cash rate change (basis points)", -300, 300, ss.rba_bps, 25, key="w_rba",
            help="Additional change applied on top of any rate changes already entered in "
                 "Current and Proposed Loan sections.")
        if ss.rba_bps != 0:
            d = "increase" if ss.rba_bps > 0 else "decrease"
            st.markdown(
                f'<div class="note">Applies an additional {abs(ss.rba_bps)} bps '
                f'({abs(ss.rba_bps)/100:.2f}%) {d} on top of existing anticipated changes.</div>',
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

    # ── Original ──
    o_rsched = build_rate_schedule(ss.o_rate, ss.o_rate_deltas)
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
    if ss.rba_bps != 0:
        all_c_deltas = all_c_deltas + [[add_months(TODAY, 1), ss.rba_bps / 100]]
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

    # Variable component: uses shared future_var_deltas + RBA scenario
    p_var_deltas = list(ss.future_var_deltas)
    if ss.rba_bps != 0:
        p_var_deltas = p_var_deltas + [[add_months(TODAY, 1), ss.rba_bps / 100]]
    p_vsched = build_rate_schedule(ss.p_adv_var_rate, p_var_deltas)

    # Fixed component: uses fixed rate for fix_mo_period, then reverts to p_rev_rate
    fix_rev_date = add_months(TODAY, fix_mo_period)
    p_fsched = [(date(1900, 1, 1), ss.p_adv_fix_rate), (fix_rev_date, ss.p_rev_rate)]

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
# DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def dash_overview(R):
    ss = st.session_state
    dfs = [R["df_orig"], R["df_curr"], R["df_ps"]]
    labels = ["Original Loan", "Current Loan", "Proposed (Split)"]
    colors = [C_ORIG, C_CURR, C_SPLIT]

    def val(df, field):
        if df is None or df.empty: return None
        if field == "payment": return df["Payment"].iloc[0]
        if field == "int": return df["Cum Interest"].iloc[-1]
        if field == "cost": return df["Cum Paid"].iloc[-1]
        if field == "term": return len(df)

    fields = [("payment", "Monthly Payment"), ("int", "Total Interest"),
              ("cost", "Total Cost"), ("term", "Loan Term")]

    # Collect values for each (field, loan)
    values = {f: [val(df, f) for df in dfs] for f, _ in fields}

    cols = st.columns(3)
    for ci, (col, df, lbl, clr) in enumerate(zip(cols, dfs, labels, colors)):
        with col:
            if df is None or df.empty:
                st.info(f"No data for {lbl}"); continue
            st.markdown(
                f'<div style="color:{clr};font-weight:600;font-size:0.8rem;'
                f'margin-bottom:8px;padding-bottom:4px;border-bottom:1px solid #1e2d4a">{lbl}</div>',
                unsafe_allow_html=True)
            for f, flbl in fields:
                v = val(df, f)
                if v is None: continue
                # Build diffs
                diff_orig = ""
                diff_curr = ""
                dpos_orig = True
                dpos_curr = True
                if ci >= 1:  # Current or Proposed → show vs Original
                    v0 = values[f][0]
                    if v0 is not None and v0 != 0:
                        d = v - v0
                        if f == "term":
                            diff_orig = f"{'−' if d<0 else '+'}{abs(int(d))} mo vs Orig"
                            dpos_orig = d < 0
                        else:
                            pct = d / v0 * 100
                            diff_orig = f"{'−' if d<0 else '+'}${abs(d):,.0f} ({abs(pct):.1f}%) vs Orig"
                            dpos_orig = d < 0
                if ci == 2:  # Proposed → ALSO show vs Current
                    v1 = values[f][1]
                    if v1 is not None and v1 != 0:
                        d = v - v1
                        if f == "term":
                            diff_curr = f"{'−' if d<0 else '+'}{abs(int(d))} mo vs Curr"
                            dpos_curr = d < 0
                        else:
                            pct = d / v1 * 100
                            diff_curr = f"{'−' if d<0 else '+'}${abs(d):,.0f} ({abs(pct):.1f}%) vs Curr"
                            dpos_curr = d < 0

                dformat = fc(v) if f != "term" else f"{v} mo / {v/12:.1f} yr"
                # Render: primary is diff vs Orig (or blank), sub is diff vs Curr (Proposed only)
                if ci == 0:
                    st.markdown(metric_card(flbl, dformat, "", True, diff_neutral=True),
                                unsafe_allow_html=True)
                elif ci == 1:
                    st.markdown(metric_card(flbl, dformat, diff_orig, dpos_orig),
                                unsafe_allow_html=True)
                else:  # ci == 2
                    # Show both vs Original AND vs Current
                    st.markdown(metric_card(flbl, dformat, diff_orig, dpos_orig,
                                             sub_diff=diff_curr if dpos_curr else f"{diff_curr}"),
                                unsafe_allow_html=True)
                    # Hack: metric_card doesn't distinguish sub_diff color independently;
                    # re-render if needed. Actually metric_card handles it via separate CSS classes
                    # — but we only pass one dpos flag. Let's render a second line manually:
                    # (replace above with a custom rendering)

    # Key metrics summary
    df_c, df_s = R["df_curr"], R["df_ps"]
    if df_c is not None and not df_c.empty and df_s is not None and not df_s.empty:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("**Proposed vs Current — Key Metrics**")
        c1, c2, c3, c4, c5 = st.columns(5)
        items = [
            ("Interest Saved", fc(df_c["Cum Interest"].iloc[-1] - df_s["Cum Interest"].iloc[-1])),
            ("Total Cost Saved", fc(df_c["Cum Paid"].iloc[-1] - df_s["Cum Paid"].iloc[-1])),
            ("Optimal Fixed Split", fp(R["best_pct"], 1)),
            ("Effective Var Rate", fp(R["eff_var"])),
            ("ASIC Comparison Rate", fp(R["comp_var"])),
        ]
        for col, (lbl, vv) in zip([c1, c2, c3, c4, c5], items):
            with col: st.markdown(metric_card(lbl, vv), unsafe_allow_html=True)

def dash_payments(R):
    df_o, df_c = R["df_orig"], R["df_curr"]
    df_ps, df_pv, df_pf = R["df_ps"], R["df_pv"], R["df_pf"]
    fig = go.Figure()
    for df, nm, clr, dash in [(df_o, "Original", C_ORIG, "dot"),
                               (df_c, "Current", C_CURR, "dash"),
                               (df_ps, "Proposed Split", C_SPLIT, "solid"),
                               (df_pv, "Variable Component", C_VAR, "dot"),
                               (df_pf, "Fixed Component", C_FIX, "dot")]:
        if df is not None and not df.empty:
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Payment"], name=nm,
                line=dict(color=clr, width=2, dash=dash),
                hovertemplate=f"<b>{nm}</b><br>%{{x|%b %Y}}<br>$%{{y:,.0f}}/mo<extra></extra>"))
    fig.update_layout(**PLOT_BASE, title="Monthly Repayments", yaxis_title="Repayment ($)")
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.04))
    st.plotly_chart(fig, use_container_width=True)

    if df_ps is not None and not df_ps.empty:
        s = df_ps.iloc[::12]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=s["Date"], y=s["Principal"], name="Principal",
                              marker_color=C_VAR,
                              hovertemplate="Principal: $%{y:,.0f}<extra></extra>"))
        fig2.add_trace(go.Bar(x=s["Date"], y=s["Interest"], name="Interest",
                              marker_color=C_FIX,
                              hovertemplate="Interest: $%{y:,.0f}<extra></extra>"))
        if df_ps["Interest Saved"].sum() > 0:
            fig2.add_trace(go.Scatter(x=s["Date"], y=s["Interest Saved"],
                name="Interest Saved (Offset)", mode="lines+markers",
                line=dict(color=C_ORIG, width=2), yaxis="y2",
                hovertemplate="Saved: $%{y:,.0f}<extra></extra>"))
            fig2.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False,
                                           title="Saved ($)", color=C_ORIG,
                                           tickfont=dict(color=C_ORIG)))
        fig2.update_layout(**PLOT_BASE, barmode="stack",
                           title="Annual Repayment Breakdown — Proposed Split",
                           yaxis_title="Amount ($)")
        st.plotly_chart(fig2, use_container_width=True)

def dash_balance(R):
    ss = st.session_state
    df_o, df_c = R["df_orig"], R["df_curr"]
    df_ps, df_pv, df_pf = R["df_ps"], R["df_pv"], R["df_pf"]
    fig = go.Figure()
    for df, nm, clr, dash in [(df_o, "Original", C_ORIG, "dot"),
                               (df_c, "Current", C_CURR, "dash"),
                               (df_ps, "Proposed Split", C_SPLIT, "solid"),
                               (df_pv, "Variable Component", C_VAR, "dot"),
                               (df_pf, "Fixed Component", C_FIX, "dot")]:
        if df is not None and not df.empty:
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Closing Balance"], name=nm,
                line=dict(color=clr, width=2, dash=dash),
                hovertemplate=f"<b>{nm}</b><br>%{{x|%b %Y}}<br>$%{{y:,.0f}}<extra></extra>",
                fill="tozeroy" if nm == "Proposed Split" else "none",
                fillcolor="rgba(196,122,245,0.04)" if nm == "Proposed Split" else "rgba(0,0,0,0)"))
    # Reversion date marker
    rev_date = add_months(TODAY, ss.p_fix_yrs * 12)
    rev_str = str(rev_date)
    fig.add_shape(type="line", x0=rev_str, x1=rev_str, y0=0, y1=1, yref="paper",
                  xref="x", line=dict(color=C_FIX, dash="dash", width=1))
    fig.add_annotation(x=rev_str, y=0.97, yref="paper", xref="x",
                       text="Fixed rate expires", showarrow=False,
                       font=dict(color=C_FIX, size=10),
                       bgcolor=C_PAPER, bordercolor=C_FIX, borderwidth=1)
    fig.update_layout(**PLOT_BASE, title="Outstanding Balance Over Time",
                      yaxis_title="Balance ($)")
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.04))
    st.plotly_chart(fig, use_container_width=True)

    curr_val = ss.c_prop_val if not ss.c_is_cont else ss.o_prop_val
    if curr_val > 0 and df_ps is not None and not df_ps.empty:
        lvr = df_ps["Closing Balance"] / curr_val * 100
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_ps["Date"], y=lvr, name="LVR (%)",
            line=dict(color=C_SPLIT, width=2),
            fill="tozeroy", fillcolor="rgba(196,122,245,0.07)",
            hovertemplate="%{x|%b %Y}<br>LVR: %{y:.1f}%<extra></extra>"))
        fig2.add_shape(type="line", x0=0, x1=1, xref="paper", y0=80, y1=80,
                       line=dict(color=C_FIX, dash="dash", width=1))
        fig2.add_annotation(x=1, xref="paper", y=80, yref="y",
                            text="80% — LMI threshold", showarrow=False,
                            font=dict(color=C_FIX, size=10), xanchor="right", yanchor="bottom")
        fig2.add_shape(type="line", x0=0, x1=1, xref="paper", y0=60, y1=60,
                       line=dict(color=C_VAR, dash="dot", width=1))
        fig2.add_annotation(x=1, xref="paper", y=60, yref="y",
                            text="60%", showarrow=False,
                            font=dict(color=C_VAR, size=10), xanchor="right", yanchor="bottom")
        fig2.update_layout(**PLOT_BASE, title="LVR Over Time — Proposed Split",
                           yaxis_title="LVR (%)")
        st.plotly_chart(fig2, use_container_width=True)

def dash_interest(R):
    df_o, df_c, df_ps = R["df_orig"], R["df_curr"], R["df_ps"]
    fig = go.Figure()
    for df, nm, clr in [(df_o, "Original", C_ORIG), (df_c, "Current", C_CURR),
                        (df_ps, "Proposed Split", C_SPLIT)]:
        if df is not None and not df.empty:
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Cum Interest"], name=nm,
                line=dict(color=clr, width=2),
                hovertemplate=f"<b>{nm}</b><br>%{{x|%b %Y}}<br>$%{{y:,.0f}}<extra></extra>"))
    fig.update_layout(**PLOT_BASE, title="Cumulative Interest Paid",
                      yaxis_title="Cumulative Interest ($)")
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.04))
    st.plotly_chart(fig, use_container_width=True)

    if df_c is not None and not df_c.empty and df_ps is not None and not df_ps.empty:
        int_sav = df_c["Cum Interest"].iloc[-1] - df_ps["Cum Interest"].iloc[-1]
        fee_diff = df_c["Fees"].sum() - df_ps["Fees"].sum()
        fig2 = go.Figure(go.Waterfall(
            orientation="h",
            measure=["absolute", "relative", "relative", "total"],
            x=[df_c["Cum Paid"].iloc[-1], -int_sav, -fee_diff, df_ps["Cum Paid"].iloc[-1]],
            y=["Current Total Cost", "Interest Δ", "Fee Δ", "Proposed Total Cost"],
            connector=dict(line=dict(color=C_GRID)),
            decreasing=dict(marker=dict(color=C_VAR)),
            increasing=dict(marker=dict(color=C_FIX)),
            totals=dict(marker=dict(color=C_SPLIT)),
            text=[fc(df_c["Cum Paid"].iloc[-1]), f"−{fc(abs(int_sav))}",
                  f"−{fc(abs(fee_diff))}", fc(df_ps["Cum Paid"].iloc[-1])],
            textposition="outside"))
        fig2.update_layout(**PLOT_BASE, title="Cost Waterfall — Current vs Proposed",
                           xaxis_title="Total Cost ($)")
        st.plotly_chart(fig2, use_container_width=True)

    if df_ps is not None and not df_ps.empty and df_ps["Cum Interest Saved"].iloc[-1] > 1:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df_ps["Date"], y=df_ps["Cum Interest Saved"],
            name="Cumulative Offset Savings", line=dict(color=C_VAR, width=2),
            fill="tozeroy", fillcolor="rgba(48,217,150,0.08)",
            hovertemplate="%{x|%b %Y}<br>Saved: $%{y:,.0f}<extra></extra>"))
        fig3.update_layout(**PLOT_BASE, title="Cumulative Interest Saved via Offset Account",
                           yaxis_title="Savings ($)")
        st.plotly_chart(fig3, use_container_width=True)

def dash_split(R):
    """
    Revamped Optimal Split visualisation.
    Three clearer visuals: (1) objective curve with optimal, (2) stacked cost composition,
    (3) compare key percentages bar chart.
    """
    ss = st.session_state
    sdf = R["split_df"]
    best = R["best_pct"]

    st.markdown(
        f'<div class="note">Optimal fixed component: <strong>{best:.1f}%</strong> '
        f'({100-best:.1f}% variable). Evaluated across 1,001 scenarios '
        f'(0.0% → 100.0% in 0.1% steps) over the {ss.p_fix_yrs}-year fixed period.</div>',
        unsafe_allow_html=True)

    # Summary metrics
    p_f = ss.p_loan_amt * best / 100
    p_v = ss.p_loan_amt * (100 - best) / 100
    best_row = sdf.iloc[sdf["Objective"].idxmin()]
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(metric_card("Optimal Fixed %", fp(best, 1)), unsafe_allow_html=True)
    with c2: st.markdown(metric_card("Fixed Amount", fc(p_f)), unsafe_allow_html=True)
    with c3: st.markdown(metric_card("Variable Amount", fc(p_v)), unsafe_allow_html=True)
    with c4: st.markdown(metric_card("Min Objective", fc(best_row["Objective"])),
                         unsafe_allow_html=True)

    # ── Chart 1: Single clear objective curve with optimal marked ──
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=sdf["Fixed %"], y=sdf["Objective"], name="Total Cost (Interest + End Balance)",
        line=dict(color=C_SPLIT, width=2.5),
        fill="tozeroy", fillcolor="rgba(196,122,245,0.08)",
        hovertemplate="Fixed: %{x:.1f}%<br>Total: $%{y:,.0f}<extra></extra>"))
    fig1.add_trace(go.Scatter(
        x=[best], y=[best_row["Objective"]], mode="markers+text",
        marker=dict(symbol="diamond", size=18, color=C_FIX,
                    line=dict(color="#ffffff", width=2)),
        name=f"Optimal: {best:.1f}%",
        text=[f"Optimal {best:.1f}%"], textposition="top center",
        textfont=dict(color=C_FIX, size=12, family="Inter")))
    fig1.update_layout(**PLOT_BASE,
        title="Total Cost by Fixed Component % (Cumulative Interest + End Balance at end of Fixed Period)",
        xaxis_title="Fixed Component (%)", yaxis_title="Total Cost ($)")
    st.plotly_chart(fig1, use_container_width=True)

    # ── Chart 2: Stacked area — composition of cost (Variable interest + Fixed interest + End balance) ──
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=sdf["Fixed %"], y=sdf["Variable Interest"],
        name="Variable Interest", mode="lines",
        line=dict(width=0), stackgroup="one",
        fillcolor="rgba(48,217,150,0.5)",
        hovertemplate="Fixed: %{x:.1f}%<br>Var Interest: $%{y:,.0f}<extra></extra>"))
    fig2.add_trace(go.Scatter(
        x=sdf["Fixed %"], y=sdf["Fixed Interest"],
        name="Fixed Interest", mode="lines",
        line=dict(width=0), stackgroup="one",
        fillcolor="rgba(233,69,96,0.5)",
        hovertemplate="Fixed: %{x:.1f}%<br>Fixed Interest: $%{y:,.0f}<extra></extra>"))
    fig2.add_trace(go.Scatter(
        x=sdf["Fixed %"], y=sdf["End Balance"],
        name="Remaining Balance", mode="lines",
        line=dict(width=0), stackgroup="one",
        fillcolor="rgba(74,154,245,0.4)",
        hovertemplate="Fixed: %{x:.1f}%<br>Remaining Bal: $%{y:,.0f}<extra></extra>"))
    # Mark optimal
    fig2.add_shape(type="line", x0=best, x1=best, y0=0, y1=1, yref="paper",
                   line=dict(color="#ffffff", dash="dash", width=1.5))
    fig2.add_annotation(x=best, y=1.02, yref="paper", xref="x",
                        text=f"<b>Optimal {best:.1f}%</b>", showarrow=False,
                        font=dict(color=C_FIX, size=11))
    fig2.update_layout(**PLOT_BASE,
        title="Cost Composition at End of Fixed Period",
        xaxis_title="Fixed Component (%)", yaxis_title="Amount ($)")
    st.plotly_chart(fig2, use_container_width=True)

    # ── Chart 3: Key percentages bar comparison ──
    key_pcts = sorted(set([0.0, 25.0, 50.0, 75.0, 100.0, round(best, 1)]))
    key_rows = [sdf[sdf["Fixed %"] == p].iloc[0] for p in key_pcts if p in sdf["Fixed %"].values]

    fig3 = go.Figure()
    x_labels = [f"{r['Fixed %']:.1f}%" + (" ★" if abs(r['Fixed %'] - best) < 0.05 else "") for r in key_rows]
    fig3.add_trace(go.Bar(
        x=x_labels, y=[r["Variable Interest"] for r in key_rows],
        name="Variable Interest", marker_color=C_VAR,
        hovertemplate="Var Int: $%{y:,.0f}<extra></extra>"))
    fig3.add_trace(go.Bar(
        x=x_labels, y=[r["Fixed Interest"] for r in key_rows],
        name="Fixed Interest", marker_color=C_FIX,
        hovertemplate="Fix Int: $%{y:,.0f}<extra></extra>"))
    fig3.add_trace(go.Bar(
        x=x_labels, y=[r["End Balance"] for r in key_rows],
        name="End Balance", marker_color=C_ORIG,
        hovertemplate="Balance: $%{y:,.0f}<extra></extra>"))
    fig3.update_layout(**PLOT_BASE, barmode="stack",
        title="Key Split Ratios — Comparison (★ = optimal)",
        xaxis_title="Fixed Component", yaxis_title="Amount ($)")
    st.plotly_chart(fig3, use_container_width=True)

    # ── Summary table ──
    tbl = sdf[sdf["Fixed %"] % 5 == 0].copy()
    st.dataframe(pd.DataFrame({
        "Fixed %": tbl["Fixed %"].apply(lambda x: f"{x:.0f}%"),
        "Variable %": tbl["Variable %"].apply(lambda x: f"{x:.0f}%"),
        "Variable Interest": tbl["Variable Interest"].apply(fc),
        "Fixed Interest": tbl["Fixed Interest"].apply(fc),
        "Total Interest": tbl["Cum Interest"].apply(fc),
        "End Balance": tbl["End Balance"].apply(fc),
        "Objective": tbl["Objective"].apply(fc),
    }), use_container_width=True, hide_index=True)

def dash_strategy_tab(R):
    ss = st.session_state
    loan, term = ss.p_loan_amt, ss.p_term_mo
    fix_mo = ss.p_fix_yrs * 12
    best = R["best_pct"]

    strat_map = {"Conservative (80% fixed)": 80.0,
                 "Balanced (optimal split)": best,
                 "Aggressive (0% fixed)": 0.0}
    sel_pct = strat_map.get(ss.strategy, best)
    st.markdown(
        f'<div class="note-ok">Selected strategy: <strong>{ss.strategy}</strong> — '
        f'{sel_pct:.1f}% fixed / {100-sel_pct:.1f}% variable</div>',
        unsafe_allow_html=True)

    strats = {"Conservative\n(80% Fixed)": (80.0, C_FIX),
              "Balanced\n(Optimal)": (best, C_SPLIT),
              "Aggressive\n(0% Fixed)": (0.0, C_VAR)}
    totals_ci, totals_pmt = {}, {}
    for nm, (pct_f, _) in strats.items():
        p_f_ = loan * pct_f / 100
        p_v_ = loan * (100 - pct_f) / 100
        bf, cif = fast_partial(p_f_, ss.p_adv_fix_rate, term, fix_mo) if p_f_ > 0 else (0, 0)
        bv, civ = fast_partial(p_v_, R["eff_var"], term, fix_mo) if p_v_ > 0 else (0, 0)
        rem = max(0, term - fix_mo)
        _, cif2 = fast_partial(bf, ss.p_rev_rate, rem, rem) if rem > 0 and bf > 0 else (0, 0)
        _, civ2 = fast_partial(bv, R["eff_var"], rem, rem) if rem > 0 and bv > 0 else (0, 0)
        k = nm.split("\n")[0]
        totals_ci[k] = cif + civ + cif2 + civ2
        totals_pmt[k] = ((calc_payment(p_f_, ss.p_adv_fix_rate, term) if p_f_ > 0 else 0)
                        + (calc_payment(p_v_, R["eff_var"], term) if p_v_ > 0 else 0))

    c1, c2 = st.columns(2)
    clrs = [C_FIX, C_SPLIT, C_VAR]
    with c1:
        fig = go.Figure(go.Bar(x=list(totals_ci.keys()), y=list(totals_ci.values()),
            marker_color=clrs, text=[fc(v) for v in totals_ci.values()],
            textposition="outside",
            hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>"))
        fig.update_layout(**PLOT_BASE, title="Total Interest by Strategy",
                          yaxis_title="Total Interest ($)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = go.Figure(go.Bar(x=list(totals_pmt.keys()), y=list(totals_pmt.values()),
            marker_color=clrs, text=[fc(v) for v in totals_pmt.values()],
            textposition="outside",
            hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>"))
        fig2.update_layout(**PLOT_BASE, title="Initial Monthly Repayment by Strategy",
                           yaxis_title="Monthly Repayment ($)")
        st.plotly_chart(fig2, use_container_width=True)

def dash_scenarios(R):
    scenarios = R["scenarios"]
    if not scenarios:
        st.warning("No scenario data.")
        return
    base = scenarios.get("Base", {})
    base_pmt = base.get("payment", 0)
    base_int = base.get("total_interest", 0)
    base_term = base.get("term_months", 0)

    st.markdown("**Monthly Repayments Under Rate Scenarios**")
    st.markdown(
        '<div class="note">Rate rises increase repayments (term never extends). '
        'Rate falls reduce repayments unless the "maintain repayment" toggle is on '
        '(in which case the loan pays off faster instead).</div>',
        unsafe_allow_html=True)

    rows = []
    for lbl, data in scenarios.items():
        dp = data["payment"] - base_pmt
        di = data["total_interest"] - base_int
        dt = data["term_months"] - base_term
        rows.append({"Scenario": lbl, "Variable Rate": fp(data["rate"]),
                     "Monthly Repayment": fc(data["payment"]),
                     "vs Base": f"{'+' if dp>=0 else ''}{fc(dp)}",
                     "Total Interest": fc(data["total_interest"]),
                     "Interest vs Base": f"{'+' if di>=0 else ''}{fc(di)}",
                     "Term (mo)": str(data["term_months"]),
                     "Term vs Base": f"{'+' if dt>=0 else ''}{dt} mo"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    scen_colors = [C_SPLIT, C_FIX, "#f59e0b", "#ef4444", C_VAR, "#22d3ee", C_ORIG]
    fig_pmt = go.Figure()
    for (lbl, data), clr in zip(scenarios.items(), scen_colors):
        df_s = data.get("df")
        if df_s is not None and not df_s.empty:
            fig_pmt.add_trace(go.Scatter(x=df_s["Date"], y=df_s["Payment"], name=lbl,
                line=dict(color=clr, width=2 if lbl == "Base" else 1.5,
                          dash="solid" if lbl == "Base" else "dash"),
                hovertemplate=f"<b>{lbl}</b><br>%{{x|%b %Y}}<br>$%{{y:,.0f}}/mo<extra></extra>"))
    fig_pmt.update_layout(**PLOT_BASE, title="Monthly Repayments — Rate Scenarios",
                          yaxis_title="Monthly Repayment ($)")
    fig_pmt.update_xaxes(rangeslider=dict(visible=True, thickness=0.04))
    st.plotly_chart(fig_pmt, use_container_width=True)

    fig_bal = go.Figure()
    for (lbl, data), clr in zip(scenarios.items(), scen_colors):
        df_s = data.get("df")
        if df_s is not None and not df_s.empty:
            fig_bal.add_trace(go.Scatter(x=df_s["Date"], y=df_s["Closing Balance"], name=lbl,
                line=dict(color=clr, width=2 if lbl == "Base" else 1.5,
                          dash="solid" if lbl == "Base" else "dash"),
                hovertemplate=f"<b>{lbl}</b><br>%{{x|%b %Y}}<br>$%{{y:,.0f}}<extra></extra>"))
    fig_bal.update_layout(**PLOT_BASE, title="Outstanding Balance — Rate Scenarios",
                          yaxis_title="Balance ($)")
    fig_bal.update_xaxes(rangeslider=dict(visible=True, thickness=0.04))
    st.plotly_chart(fig_bal, use_container_width=True)

def dash_schedules(R):
    options = {"Original Loan": R["df_orig"],
               "Current Loan": R["df_curr"],
               "Proposed Variable Component": R["df_pv"],
               "Proposed Fixed Component": R["df_pf"],
               "Proposed Split (Combined)": R["df_ps"]}
    avail = {k: v for k, v in options.items() if v is not None and not v.empty}
    if not avail:
        st.warning("No schedules.")
        return
    sel = st.selectbox("Select schedule", list(avail.keys()), key="sch_sel")
    df = avail[sel].copy()
    disp = df.copy()
    money_cols = ["Opening Balance", "Avg Offset", "Net Debt", "Interest", "Interest Saved",
                  "Principal", "Extra Repayment", "Fees", "Payment", "Closing Balance",
                  "Cum Interest", "Cum Paid", "Cum Interest Saved", "Cum Extra Repayment"]
    for c in money_cols:
        if c in disp.columns:
            disp[c] = disp[c].apply(fc)
    if "Rate %" in disp.columns:
        disp["Rate %"] = disp["Rate %"].apply(fp)
    st.dataframe(disp, use_container_width=True, hide_index=True, height=440)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    st.download_button(f"Download {sel} as CSV", buf.getvalue(),
                       f"schedule_{sel.lower().replace(' ','_').replace('(','').replace(')','')}.csv",
                       "text/csv", key=f"dl_{sel}")

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

    render_rba_panel()

    with st.expander("Original Loan", expanded=True):
        section_original()
    with st.expander("Current Loan", expanded=True):
        section_current()
    with st.expander("Proposed Loan", expanded=True):
        section_proposed()
    with st.expander("Strategy and Scenarios", expanded=True):
        section_strategy()

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

    tabs = st.tabs(["Overview", "Monthly Payments", "Loan Balance", "Interest Analysis",
                    "Optimal Split", "Strategy", "Rate Scenarios", "Schedules"])
    with tabs[0]: dash_overview(R)
    with tabs[1]: dash_payments(R)
    with tabs[2]: dash_balance(R)
    with tabs[3]: dash_interest(R)
    with tabs[4]: dash_split(R)
    with tabs[5]: dash_strategy_tab(R)
    with tabs[6]: dash_scenarios(R)
    with tabs[7]: dash_schedules(R)


if __name__ == "__main__":
    main()
