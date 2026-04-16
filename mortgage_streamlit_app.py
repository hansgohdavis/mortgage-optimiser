"""
Australian Mortgage Refinance Analyser
Complete single-page Streamlit application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, datetime, timedelta
import calendar
import requests
import re
import io

try:
    from scipy.optimize import brentq
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AU Mortgage Refinance Analyser",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS — MINIMALIST DARK
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}

.stApp { background-color: #07090f; color: #d4dbe8; }

section[data-testid="stSidebar"] { background: #0b0f1a; }

.stNumberInput input, .stTextInput input {
    background: #111827 !important;
    border: 1px solid #1e2d4a !important;
    border-radius: 5px !important;
    color: #d4dbe8 !important;
    font-size: 0.875rem !important;
}
.stDateInput input {
    background: #111827 !important;
    border: 1px solid #1e2d4a !important;
    color: #d4dbe8 !important;
    font-size: 0.875rem !important;
}
.stSelectbox > div > div {
    background: #111827 !important;
    border: 1px solid #1e2d4a !important;
    color: #d4dbe8 !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: #0b0f1a;
    border-bottom: 1px solid #1e2d4a;
    gap: 0;
    padding: 0 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #64748b;
    border-radius: 0;
    padding: 10px 18px;
    font-size: 0.8rem;
    font-weight: 500;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #4a9af5 !important;
    border-bottom: 2px solid #4a9af5 !important;
}

.streamlit-expanderHeader {
    background: #0b0f1a !important;
    border: 1px solid #1e2d4a !important;
    border-radius: 6px !important;
    color: #d4dbe8 !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
}
.streamlit-expanderContent {
    background: #07090f !important;
    border: 1px solid #1e2d4a !important;
    border-top: none !important;
    border-radius: 0 0 6px 6px !important;
    padding: 16px !important;
}

.stButton > button {
    background: #1a2744;
    color: #d4dbe8;
    border: 1px solid #1e2d4a;
    border-radius: 5px;
    font-size: 0.85rem;
    font-weight: 500;
    padding: 7px 16px;
    transition: all 0.15s;
}
.stButton > button:hover {
    background: #243560;
    border-color: #4a9af5;
    color: #4a9af5;
}
.btn-primary > button {
    background: #1a3a6b !important;
    border-color: #4a9af5 !important;
    color: #4a9af5 !important;
}
.btn-primary > button:hover { background: #1e4480 !important; }
.btn-danger > button {
    background: transparent !important;
    border-color: #e94560 !important;
    color: #e94560 !important;
}

.m-card {
    background: #0b0f1a;
    border: 1px solid #1e2d4a;
    border-radius: 7px;
    padding: 14px 16px;
    margin-bottom: 6px;
}
.m-label {
    color: #64748b;
    font-size: 0.7rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 5px;
}
.m-value { color: #d4dbe8; font-size: 1.3rem; font-weight: 600; line-height: 1.2; }
.m-delta-pos { color: #30d996; font-size: 0.75rem; margin-top: 2px; }
.m-delta-neg { color: #e94560; font-size: 0.75rem; margin-top: 2px; }

.sec-title {
    color: #64748b;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 14px 0 6px;
    border-bottom: 1px solid #1e2d4a;
    margin-bottom: 12px;
}

.list-header {
    color: #64748b;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding-bottom: 4px;
}

.note {
    background: #0b0f1a;
    border-left: 2px solid #4a9af5;
    border-radius: 0 4px 4px 0;
    padding: 8px 12px;
    font-size: 0.8rem;
    color: #64748b;
    margin: 8px 0;
}

.computed-field {
    background: #0d1626;
    border: 1px solid #1e2d4a;
    border-radius: 5px;
    padding: 9px 12px;
    font-size: 0.875rem;
}
.computed-label { color: #64748b; font-size: 0.72rem; margin-bottom: 3px; }
.computed-value { color: #4a9af5; font-weight: 600; font-size: 1.05rem; }

hr { border-color: #1e2d4a !important; margin: 20px 0 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

C_ORIG, C_CURR, C_VAR, C_FIX, C_SPLIT = "#4a9af5", "#f5a94a", "#30d996", "#e94560", "#c47af5"
C_PAPER, C_PLOT, C_GRID, C_TEXT = "#0b0f1a", "#07090f", "#1e2d4a", "#d4dbe8"

PLOT_BASE = dict(
    paper_bgcolor=C_PAPER, plot_bgcolor=C_PLOT,
    font=dict(family="Inter", color=C_TEXT, size=11),
    xaxis=dict(gridcolor=C_GRID, zerolinecolor=C_GRID, linecolor=C_GRID),
    yaxis=dict(gridcolor=C_GRID, zerolinecolor=C_GRID, linecolor=C_GRID),
    legend=dict(bgcolor=C_PAPER, bordercolor=C_GRID, borderwidth=1, font=dict(size=11)),
    margin=dict(t=36, b=36, l=56, r=16),
    hovermode="x unified",
    hoverlabel=dict(bgcolor=C_PAPER, bordercolor=C_GRID, font=dict(color=C_TEXT)),
)

TODAY = date.today()

# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def fc(v, d=0):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    neg = v < 0
    return ("-" if neg else "") + f"${abs(v):,.{d}f}"

def fp(v, d=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.{d}f}%"

def parse_dt(v) -> date:
    if isinstance(v, date):
        return v
    if isinstance(v, datetime):
        return v.date()
    try:
        return datetime.strptime(str(v), "%Y-%m-%d").date()
    except Exception:
        return TODAY

def months_between(d1: date, d2: date) -> int:
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)

def add_months(d: date, n: int) -> date:
    m = d.month - 1 + n
    y = d.year + m // 12
    mo = m % 12 + 1
    return date(y, mo, min(d.day, calendar.monthrange(y, mo)[1]))

def metric_card(label: str, value: str, delta: str = "", delta_pos: bool = True):
    dc = "m-delta-pos" if delta_pos else "m-delta-neg"
    dh = f'<div class="{dc}">{delta}</div>' if delta else ""
    return f'<div class="m-card"><div class="m-label">{label}</div><div class="m-value">{value}</div>{dh}</div>'

def sec(title: str):
    st.markdown(f'<div class="sec-title">{title}</div>', unsafe_allow_html=True)

def computed(label: str, value: str):
    st.markdown(
        f'<div class="computed-field"><div class="computed-label">{label}</div>'
        f'<div class="computed-value">{value}</div></div>',
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════════════════════════
# FINANCIAL ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def calc_payment(principal: float, rate_pct: float, term_mo: int) -> float:
    if principal <= 0 or term_mo <= 0:
        return 0.0
    if rate_pct <= 0:
        return principal / term_mo
    r = rate_pct / 100 / 12
    return principal * r * (1 + r) ** term_mo / ((1 + r) ** term_mo - 1)

def build_rate_schedule(base: float, deltas: list) -> list:
    """[(date, effective_rate)] — deltas are +/- cumulative from base."""
    schedule = [(date(1900, 1, 1), base)]
    cum = base
    for row in sorted([(parse_dt(r[0]), float(r[1])) for r in deltas if r[0]], key=lambda x: x[0]):
        cum = round(cum + row[1], 4)
        schedule.append((row[0], cum))
    return schedule

def get_rate(sched: list, d: date) -> float:
    r = sched[0][1]
    for rd, rv in sched:
        if rd <= d:
            r = rv
        else:
            break
    return r

def build_offset_schedule(start: date, end: date, init: float, monthly: float, lumps: list) -> dict:
    if init <= 0 and monthly <= 0 and not lumps:
        return {}
    lmap = {}
    for row in lumps:
        d = parse_dt(row[0])
        lmap[d] = lmap.get(d, 0.0) + float(row[1])
    bal = init
    sched = {}
    prev_mo = start.month - 1
    cur = start
    while cur <= end:
        if cur.month != prev_mo:
            if cur != start:
                bal += monthly
            prev_mo = cur.month
        if cur in lmap:
            bal += lmap[cur]
        sched[cur] = max(0.0, bal)
        cur += timedelta(days=1)
    return sched

def get_offset(sched: dict, d: date) -> float:
    if not sched:
        return 0.0
    if d in sched:
        return sched[d]
    keys = [k for k in sched if k <= d]
    return sched[max(keys)] if keys else 0.0

def amortize(
    principal: float,
    start: date,
    term_mo: int,
    rate_sched: list,
    offset_sched: dict,
    monthly_fee: float = 0.0,
    maintain_payment: bool = True,
) -> pd.DataFrame:
    """
    Daily-interest amortisation.
    - Rate rise  → payment increases to maintain remaining term (term never extends)
    - Rate fall + maintain_payment=True  → keep payment (term shortens)
    - Rate fall + maintain_payment=False → payment decreases to minimum for remaining term
    """
    if principal <= 0 or term_mo <= 0:
        return pd.DataFrame()

    bal = principal
    prev_rate = get_rate(rate_sched, start)
    cur_pmt = calc_payment(principal, prev_rate, term_mo)

    cum_int = cum_paid = cum_saved = 0.0
    rows = []
    period = start
    mo_num = 0

    while bal > 0.01 and mo_num < term_mo + 120:
        mo_num += 1
        next_d = add_months(period, 1)
        days = (next_d - period).days

        ann_rate = get_rate(rate_sched, period)

        if abs(ann_rate - prev_rate) > 1e-9:
            rem = max(1, term_mo - mo_num + 1)
            req_pmt = calc_payment(bal, ann_rate, rem)
            if ann_rate > prev_rate:
                cur_pmt = req_pmt                          # must increase
            else:
                if maintain_payment:
                    cur_pmt = max(cur_pmt, req_pmt)        # keep higher → faster payoff
                else:
                    cur_pmt = req_pmt                      # reduce to minimum
            prev_rate = ann_rate

        # Offset (sampled)
        avg_off = 0.0
        if offset_sched:
            pts = [period + timedelta(days=i) for i in range(0, days, max(1, days // 6))]
            avg_off = min(sum(get_offset(offset_sched, p) for p in pts) / len(pts), bal)

        net = max(0.0, bal - avg_off)
        dr = ann_rate / 100 / 365
        interest = net * dr * days
        int_no_off = bal * dr * days
        saved = int_no_off - interest

        cum_int += interest
        cum_saved += saved

        opening = bal
        bal += interest
        pmt = min(cur_pmt, bal)
        principal_paid = max(0.0, pmt - interest)
        bal = max(0.0, bal - pmt)
        cum_paid += pmt + monthly_fee

        rows.append({
            "Month": mo_num,
            "Date": next_d,
            "Opening Balance": opening,
            "Avg Offset": avg_off,
            "Net Debt": net,
            "Rate %": ann_rate,
            "Interest": interest,
            "Interest Saved": saved,
            "Principal": principal_paid,
            "Fees": monthly_fee,
            "Payment": pmt,
            "Closing Balance": bal,
            "Cum Interest": cum_int,
            "Cum Paid": cum_paid,
            "Cum Interest Saved": cum_saved,
        })

        period = next_d

    return pd.DataFrame(rows)

def fast_partial(principal: float, rate: float, total: int, n: int):
    """O(1) partial amortisation → (balance, cum_interest) after n months."""
    if principal <= 0 or n <= 0:
        return 0.0, 0.0
    n = min(n, total)
    if rate <= 0:
        pmt = principal / total
        return max(0.0, principal - pmt * n), 0.0
    r = rate / 100 / 12
    pmt = principal * r * (1 + r) ** total / ((1 + r) ** total - 1)
    bal = principal * (1 + r) ** n - pmt * ((1 + r) ** n - 1) / r
    bal = max(0.0, bal)
    return bal, max(0.0, pmt * n - (principal - bal))

def calc_optimal_split(loan, var_r, fix_r, rev_r, fix_yrs, total_mo):
    """
    1,001-scenario sweep in 0.1% increments over fixed period only.
    Objective = cum_interest + closing_balance at end of fixed period.
    """
    n_fix = min(fix_yrs * 12, total_mo)
    best_obj, best_pct = float("inf"), 50.0
    rows = []
    for i in range(1001):
        pct_f = i / 10.0
        p_f = loan * pct_f / 100
        p_v = loan * (100 - pct_f) / 100
        bal_f, ci_f = fast_partial(p_f, fix_r, total_mo, n_fix) if p_f > 0 else (0.0, 0.0)
        bal_v, ci_v = fast_partial(p_v, var_r, total_mo, n_fix) if p_v > 0 else (0.0, 0.0)
        obj = (ci_f + ci_v) + (bal_f + bal_v)
        if obj < best_obj:
            best_obj, best_pct = obj, pct_f
        rows.append({"Fixed %": pct_f, "Variable %": 100 - pct_f,
                     "Cum Interest": ci_f + ci_v, "End Balance": bal_f + bal_v, "Objective": obj})
    return best_pct, pd.DataFrame(rows)

def comparison_rate(loan, setup_fee, monthly_fee, rate_pct):
    pv, n = 150_000.0, 300
    pmt = calc_payment(pv, rate_pct, n) + monthly_fee
    target = pv - setup_fee
    if not HAS_SCIPY or target <= 0:
        return rate_pct
    def f(i):
        if abs(i) < 1e-12:
            return pmt * n - target
        return pmt * (1 - (1 + i) ** -n) / i - target
    try:
        return brentq(f, 1e-8, 0.5) * 12 * 100
    except Exception:
        return rate_pct

def effective_rate(loan, setup_fee, monthly_fee, rate_pct, term):
    if not HAS_SCIPY or loan <= 0:
        return rate_pct
    pmt = calc_payment(loan, rate_pct, term) + monthly_fee
    target = loan - setup_fee
    if target <= 0:
        return rate_pct
    def f(i):
        if abs(i) < 1e-12:
            return pmt * term - target
        return pmt * (1 - (1 + i) ** -term) / i - target
    try:
        return brentq(f, 1e-8, 0.5) * 12 * 100
    except Exception:
        return rate_pct

def merge_schedules(df_v, df_f):
    if df_v is None or df_v.empty:
        return df_f.copy() if df_f is not None and not df_f.empty else pd.DataFrame()
    if df_f is None or df_f.empty:
        return df_v.copy()
    n = max(len(df_v), len(df_f))
    num_cols = ["Opening Balance", "Avg Offset", "Net Debt", "Interest", "Interest Saved",
                "Principal", "Fees", "Payment", "Closing Balance",
                "Cum Interest", "Cum Paid", "Cum Interest Saved"]
    rows = []
    for i in range(n):
        rv = df_v.iloc[i].to_dict() if i < len(df_v) else None
        rf = df_f.iloc[i].to_dict() if i < len(df_f) else None
        if rv and rf:
            row = {"Month": rv["Month"], "Date": rv["Date"]}
            for c in num_cols:
                row[c] = rv.get(c, 0.0) + rf.get(c, 0.0)
            ob = rv.get("Opening Balance", 0) + rf.get("Opening Balance", 0)
            row["Rate %"] = (rv["Rate %"] * rv.get("Opening Balance", 0) +
                             rf["Rate %"] * rf.get("Opening Balance", 0)) / ob if ob > 0 else 0.0
            rows.append(row)
        else:
            r = (rv or rf).copy()
            r["Month"] = i + 1
            rows.append(r)
    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════════════════════
# RBA RATE FETCH
# ══════════════════════════════════════════════════════════════════════════════

def fetch_rba_rate():
    hdrs = {"User-Agent": "Mozilla/5.0 (compatible; AU-Mortgage-Analyser/2.0)",
            "Accept": "text/html,text/csv,application/json,*/*"}

    # Attempt 1: F1 CSV
    try:
        r = requests.get("https://www.rba.gov.au/statistics/tables/csv/f1-data.csv",
                         headers=hdrs, timeout=10)
        if r.status_code == 200:
            lines = r.text.split("\n")
            target_col = None
            data_start = None
            for li, line in enumerate(lines):
                if "Cash Rate Target" in line:
                    for ci, p in enumerate(line.split(",")):
                        if "Cash Rate Target" in p:
                            target_col = ci
                            break
                    data_start = li + 1
                    break
            if target_col is not None:
                for line in reversed(lines[data_start or 0:]):
                    parts = line.split(",")
                    if len(parts) > target_col:
                        try:
                            v = float(parts[target_col].strip().strip('"'))
                            if 0 < v < 30:
                                return v
                        except Exception:
                            pass
    except Exception:
        pass

    # Attempt 2: Scrape cash-rate page
    try:
        r = requests.get("https://www.rba.gov.au/statistics/cash-rate/",
                         headers=hdrs, timeout=10)
        if r.status_code == 200:
            for pat in [
                r"cash rate target[^<]{0,300}?(\d+\.\d{2})\s*(?:per cent|%)",
                r"(\d+\.\d{2})\s*(?:per cent|%) p\.a",
            ]:
                for m in re.findall(pat, r.text, re.IGNORECASE):
                    try:
                        v = float(m)
                        if 0 < v < 30:
                            return v
                    except Exception:
                        pass
    except Exception:
        pass

    # Attempt 3: RBA JSON / SDMX API
    try:
        r = requests.get(
            "https://api.rba.gov.au/statistics/key/F_1_D_M_AUD/all"
            "?detail=dataonly&lastNObservations=1",
            headers=hdrs, timeout=8)
        if r.status_code == 200:
            data = r.json()
            for series in data.get("dataSets", [{}])[0].get("series", {}).values():
                for obs in series.get("observations", {}).values():
                    try:
                        v = float(obs[0])
                        if 0 < v < 30:
                            return v
                    except Exception:
                        pass
    except Exception:
        pass

    return None

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

def _d(k, v):
    if k not in st.session_state:
        st.session_state[k] = v

def init_state():
    _d("o_prop_val", 800_000.0);   _d("o_prop_date", date(2020, 1, 15))
    _d("o_loan_amt", 640_000.0);   _d("o_loan_date", date(2020, 1, 15))
    _d("o_use_dates", False);      _d("o_end_date", date(2045, 1, 15))
    _d("o_term_mo", 300);          _d("o_balance", 580_000.0)
    _d("o_balance_date", TODAY);   _d("o_rate", 6.50)
    _d("o_rate_deltas", []);       _d("o_off_init", 0.0)
    _d("o_off_date", TODAY);       _d("o_off_monthly", 0.0)
    _d("o_off_lumps", []);         _d("o_fee_mo", 0.0)
    _d("o_fee_setup", 0.0);        _d("o_fee_break", 0.0);  _d("o_fee_other", 0.0)

    _d("c_is_cont", True);         _d("c_prop_val", 800_000.0)
    _d("c_prop_date", TODAY);      _d("c_balance", 580_000.0)
    _d("c_rate", 6.50);            _d("c_use_dates", False)
    _d("c_end_date", date(2045, 1, 15)); _d("c_term_mo", 300)
    _d("c_rate_deltas", []);       _d("c_off_init", 0.0)
    _d("c_off_date", TODAY);       _d("c_off_monthly", 0.0)
    _d("c_off_lumps", []);         _d("c_fee_mo", 10.0)
    _d("c_fee_setup", 0.0);        _d("c_fee_other", 0.0)

    _d("p_loan_amt", 580_000.0);   _d("p_start_date", TODAY)
    _d("p_use_dates", False);      _d("p_end_date", add_months(TODAY, 300))
    _d("p_term_mo", 300);          _d("p_var_rate", 6.20)
    _d("p_fix_rate", 5.89);        _d("p_adv_rate", 6.20)
    _d("p_comp_in", 6.35);         _d("p_fix_yrs", 3)
    _d("p_rev_rate", 6.50);        _d("p_split_auto", True)
    _d("p_split_pct", 50.0);       _d("p_var_deltas", [])
    _d("p_off_init", 0.0);         _d("p_off_date", TODAY)
    _d("p_off_monthly", 0.0);      _d("p_off_lumps", [])
    _d("p_fee_mo", 10.0);          _d("p_fee_setup", 800.0)
    _d("p_fee_break", 0.0);        _d("p_fee_other", 0.0)

    _d("strategy", "Balanced");    _d("maintain_pmt", True)
    _d("rba_bps", 0);              _d("rba_live", None)
    _d("results", None)

def eff_rate_from_deltas(base, deltas, as_of=None):
    r = base
    for row in sorted([(parse_dt(row[0]), float(row[1])) for row in deltas if row[0]], key=lambda x: x[0]):
        if as_of is None or row[0] <= as_of:
            r = round(r + row[1], 4)
    return r

# ══════════════════════════════════════════════════════════════════════════════
# DYNAMIC LIST WIDGETS (fixed labels)
# ══════════════════════════════════════════════════════════════════════════════

def rate_delta_list(state_key: str, base_rate: float, title: str, max_rows: int = 15):
    ss = st.session_state
    if state_key not in ss:
        ss[state_key] = []

    sec(title)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Add rate change", key=f"add_{state_key}") and len(ss[state_key]) < max_rows:
            ss[state_key].append([TODAY, 0.25])
    with c2:
        if ss[state_key] and st.button("Clear all", key=f"clr_{state_key}"):
            ss[state_key] = []
            st.rerun()

    if ss[state_key]:
        h = st.columns([2, 1.6, 1.8, 0.5])
        for i, lbl in enumerate(["Effective Date", "Change (%)", "Resulting Rate"]):
            h[i].markdown(f'<div class="list-header">{lbl}</div>', unsafe_allow_html=True)

        cum = base_rate
        to_del = None
        for i, row in enumerate(ss[state_key]):
            delta_val = float(row[1])
            c1, c2, c3, c4 = st.columns([2, 1.6, 1.8, 0.5])
            with c1:
                new_d = st.date_input(
                    f"Date for change {i+1}",
                    value=parse_dt(row[0]),
                    key=f"{state_key}_d_{i}",
                    label_visibility="collapsed"
                )
            with c2:
                new_delta = st.number_input(
                    f"Delta for change {i+1}",
                    value=delta_val,
                    min_value=-10.0, max_value=10.0, step=0.25, format="%.2f",
                    key=f"{state_key}_v_{i}",
                    label_visibility="collapsed"
                )
            cum_result = round(cum + new_delta, 4)
            with c3:
                sign = "+" if new_delta >= 0 else ""
                colour = "#e94560" if new_delta > 0 else ("#30d996" if new_delta < 0 else "#64748b")
                st.markdown(
                    f'<div style="padding:8px 4px; color:{colour}; font-size:0.875rem; font-weight:500">'
                    f'{sign}{fp(new_delta)} → {fp(cum_result)}</div>',
                    unsafe_allow_html=True
                )
            with c4:
                if st.button("✕", key=f"{state_key}_del_{i}"):
                    to_del = i
            ss[state_key][i] = [new_d, new_delta]
            cum = cum_result

        if to_del is not None:
            ss[state_key].pop(to_del)
            st.rerun()

def lump_sum_list(state_key: str, title: str, max_rows: int = 100):
    ss = st.session_state
    if state_key not in ss:
        ss[state_key] = []

    sec(title)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Add lump sum", key=f"add_{state_key}") and len(ss[state_key]) < max_rows:
            ss[state_key].append([TODAY, 0.0])
    with c2:
        if ss[state_key] and st.button("Clear all", key=f"clr_{state_key}"):
            ss[state_key] = []
            st.rerun()

    if ss[state_key]:
        h = st.columns([2, 2, 0.5])
        h[0].markdown('<div class="list-header">Date</div>', unsafe_allow_html=True)
        h[1].markdown('<div class="list-header">Amount ($)</div>', unsafe_allow_html=True)
        to_del = None
        for i, row in enumerate(ss[state_key]):
            c1, c2, c3 = st.columns([2, 2, 0.5])
            with c1:
                new_d = st.date_input(
                    f"Lump sum date {i+1}", value=parse_dt(row[0]),
                    key=f"{state_key}_d_{i}", label_visibility="collapsed"
                )
            with c2:
                new_a = st.number_input(
                    f"Lump sum amount {i+1}", value=float(row[1]), step=1_000.0,
                    key=f"{state_key}_a_{i}", label_visibility="collapsed"
                )
            with c3:
                if st.button("✕", key=f"{state_key}_del_{i}"):
                    to_del = i
            ss[state_key][i] = [new_d, new_a]
        if to_del is not None:
            ss[state_key].pop(to_del)
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# CALCULATION RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_calc():
    ss = st.session_state
    errors = []
    if ss.o_loan_amt <= 0: errors.append("Original loan amount must be positive.")
    if ss.o_balance <= 0: errors.append("Remaining balance must be positive.")
    if ss.o_term_mo <= 0: errors.append("Original loan term must be positive.")
    if ss.p_loan_amt <= 0: errors.append("Proposed loan amount must be positive.")
    if errors:
        for e in errors:
            st.error(e)
        return

    maintain = ss.maintain_pmt
    far = add_months(TODAY, max(ss.o_term_mo, ss.p_term_mo) + 12)

    # ── Original ──
    o_rsched = build_rate_schedule(ss.o_rate, ss.o_rate_deltas)
    o_osched = build_offset_schedule(ss.o_off_date, far, ss.o_off_init, ss.o_off_monthly, ss.o_off_lumps)
    df_orig = amortize(ss.o_balance, ss.o_balance_date, ss.o_term_mo, o_rsched, o_osched, ss.o_fee_mo, maintain)

    # ── Current ──
    if ss.c_is_cont:
        c_bal = ss.o_balance; c_rsched = build_rate_schedule(ss.o_rate, ss.o_rate_deltas)
        c_term = ss.o_term_mo; c_start = ss.o_balance_date
    else:
        c_bal = ss.c_balance; c_rsched = build_rate_schedule(ss.c_rate, ss.c_rate_deltas)
        c_term = ss.c_term_mo; c_start = TODAY
    c_osched = build_offset_schedule(ss.c_off_date, far, ss.c_off_init, ss.c_off_monthly, ss.c_off_lumps)
    df_curr = amortize(c_bal, c_start, c_term, c_rsched, c_osched, ss.c_fee_mo, maintain)

    # ── Proposed: optimal split ──
    if ss.p_split_auto:
        best_pct, split_df = calc_optimal_split(ss.p_loan_amt, ss.p_var_rate, ss.p_fix_rate,
                                                 ss.p_rev_rate, ss.p_fix_yrs, ss.p_term_mo)
        ss.p_split_pct = round(best_pct, 1)
    else:
        _, split_df = calc_optimal_split(ss.p_loan_amt, ss.p_var_rate, ss.p_fix_rate,
                                          ss.p_rev_rate, ss.p_fix_yrs, ss.p_term_mo)
        best_pct = ss.p_split_pct

    p_f = ss.p_loan_amt * best_pct / 100
    p_v = ss.p_loan_amt * (100 - best_pct) / 100
    p_osched = build_offset_schedule(ss.p_off_date, far, ss.p_off_init, ss.p_off_monthly, ss.p_off_lumps)
    p_vsched = build_rate_schedule(ss.p_var_rate, ss.p_var_deltas)
    fix_rev_date = add_months(TODAY, ss.p_fix_yrs * 12)
    p_fsched = [(date(1900, 1, 1), ss.p_fix_rate), (fix_rev_date, ss.p_rev_rate)]

    df_pv = amortize(p_v, TODAY, ss.p_term_mo, p_vsched, p_osched,
                     ss.p_fee_mo * (100 - best_pct) / 100, maintain) if p_v > 1 else pd.DataFrame()
    df_pf = amortize(p_f, TODAY, ss.p_term_mo, p_fsched, {},
                     ss.p_fee_mo * best_pct / 100, maintain) if p_f > 1 else pd.DataFrame()
    df_ps = merge_schedules(df_pv, df_pf)

    eff_r = effective_rate(ss.p_loan_amt, ss.p_fee_setup, ss.p_fee_mo, ss.p_var_rate, ss.p_term_mo)
    comp_r = comparison_rate(ss.p_loan_amt, ss.p_fee_setup, ss.p_fee_mo, ss.p_var_rate)

    # ── Payment scenarios (full variable, showing repayment sensitivity) ──
    scen_deltas = [0.0, 0.25, 0.50, 1.00, -0.25, -0.50, ss.rba_bps / 100.0]
    scen_labels = ["Base", "+0.25%", "+0.50%", "+1.00%", "-0.25%", "-0.50%",
                   f"RBA {'+' if ss.rba_bps >= 0 else ''}{ss.rba_bps/100:.2f}%"]
    scenarios = {}
    for lbl, delta in zip(scen_labels, scen_deltas):
        sr = ss.p_var_rate + delta
        df_s = amortize(ss.p_loan_amt, TODAY, ss.p_term_mo,
                        [(date(1900, 1, 1), sr)], p_osched, ss.p_fee_mo, maintain)
        if not df_s.empty:
            scenarios[lbl] = {
                "rate": sr, "payment": df_s["Payment"].iloc[0],
                "total_interest": df_s["Cum Interest"].iloc[-1],
                "term_months": len(df_s), "df": df_s,
            }

    ss.results = {
        "df_orig": df_orig, "df_curr": df_curr,
        "df_pv": df_pv, "df_pf": df_pf, "df_ps": df_ps,
        "split_df": split_df, "best_pct": best_pct,
        "eff_r": eff_r, "comp_r": comp_r,
        "scenarios": scenarios,
    }

# ══════════════════════════════════════════════════════════════════════════════
# INPUT SECTIONS
# ══════════════════════════════════════════════════════════════════════════════

def section_original():
    ss = st.session_state

    sec("Property")
    c1, c2, c3 = st.columns(3)
    with c1:
        ss.o_prop_val = st.number_input("Current Property Valuation ($)", value=ss.o_prop_val, min_value=0.0, step=10_000.0, key="w_o_pv")
    with c2:
        ss.o_prop_date = st.date_input("Valuation Date", value=ss.o_prop_date, key="w_o_pd")
    with c3:
        if ss.o_prop_val > 0 and ss.o_loan_amt > 0:
            computed("Original LVR (Loan Amount / Valuation)", fp(ss.o_loan_amt / ss.o_prop_val * 100))
        else:
            computed("Original LVR", "—")

    sec("Loan Details")
    c1, c2, c3 = st.columns(3)
    with c1:
        ss.o_loan_amt = st.number_input("Original Loan Amount ($)", value=ss.o_loan_amt, min_value=0.0, step=10_000.0, key="w_o_la")
    with c2:
        ss.o_balance = st.number_input("Remaining Balance ($)", value=ss.o_balance, min_value=0.0, step=1_000.0, key="w_o_bal")
    with c3:
        ss.o_balance_date = st.date_input("Balance As At", value=ss.o_balance_date, key="w_o_bd")

    c1, c2 = st.columns(2)
    with c1:
        ss.o_rate = st.number_input("Original Interest Rate (% p.a.)", value=ss.o_rate, min_value=0.0, max_value=30.0, step=0.01, format="%.4f", key="w_o_r")
    with c2:
        ss.o_loan_date = st.date_input("Loan Start Date", value=ss.o_loan_date, key="w_o_ld")

    sec("Loan Term")
    ss.o_use_dates = st.toggle("Calculate term from start and end dates", value=ss.o_use_dates, key="w_o_ud")
    if ss.o_use_dates:
        c1, c2, c3 = st.columns(3)
        with c1:
            ss.o_end_date = st.date_input("Loan End Date", value=ss.o_end_date, key="w_o_ed")
        with c2:
            if ss.o_end_date > ss.o_loan_date:
                ss.o_term_mo = months_between(ss.o_loan_date, ss.o_end_date)
                computed("Calculated Term", f"{ss.o_term_mo} months ({ss.o_term_mo/12:.1f} yrs)")
            else:
                st.warning("End date must be after start date.")
        with c3:
            rem = max(0, months_between(TODAY, ss.o_end_date)) if ss.o_end_date > TODAY else 0
            computed("Remaining Term (from today)", f"{rem} months ({rem/12:.1f} yrs)")
    else:
        c1, c2 = st.columns(2)
        with c1:
            ss.o_term_mo = st.number_input("Loan Term (months)", value=ss.o_term_mo, min_value=1, max_value=600, step=12, key="w_o_tm")
        with c2:
            computed("Equivalent", f"{ss.o_term_mo/12:.1f} years")

    rate_delta_list("o_rate_deltas", ss.o_rate, "Historical Rate Changes (enter as plus or minus change)")

    sec("Offset Account")
    c1, c2, c3 = st.columns(3)
    with c1:
        ss.o_off_init = st.number_input("Initial Balance ($)", value=ss.o_off_init, min_value=0.0, step=1_000.0, key="w_o_oi")
    with c2:
        ss.o_off_date = st.date_input("Offset Start Date", value=ss.o_off_date, key="w_o_od")
    with c3:
        ss.o_off_monthly = st.number_input("Monthly Addition ($)", value=ss.o_off_monthly, min_value=0.0, step=100.0, key="w_o_om")
    lump_sum_list("o_off_lumps", "Offset Lump Sum Deposits")

    sec("Fees")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ss.o_fee_mo = st.number_input("Monthly Fee ($)", value=ss.o_fee_mo, min_value=0.0, step=1.0, key="w_o_fm")
    with c2:
        ss.o_fee_setup = st.number_input("Setup Fee ($)", value=ss.o_fee_setup, min_value=0.0, step=100.0, key="w_o_fs")
    with c3:
        ss.o_fee_break = st.number_input("Breakage Fee ($)", value=ss.o_fee_break, min_value=0.0, step=100.0, key="w_o_fb")
    with c4:
        ss.o_fee_other = st.number_input("Other One-off Fee ($)", value=ss.o_fee_other, min_value=0.0, step=100.0, key="w_o_fo")

def section_current():
    ss = st.session_state
    ss.c_is_cont = st.toggle(
        "Treat as continuation of original loan (auto-fills balance and rate)",
        value=ss.c_is_cont, key="w_c_ic"
    )

    sec("Property")
    c1, c2, c3 = st.columns(3)
    if ss.c_is_cont:
        with c1:
            ss.c_prop_val = st.number_input("Current Property Valuation ($)", value=ss.c_prop_val, min_value=0.0, step=10_000.0, key="w_c_pv")
        with c2:
            ss.c_prop_date = st.date_input("Valuation Date", value=ss.c_prop_date, key="w_c_pd")
        with c3:
            if ss.c_prop_val > 0:
                computed("Current LVR (Balance / Valuation)", fp(ss.o_balance / ss.c_prop_val * 100))
            else:
                computed("Current LVR", "—")

        sec("Auto-filled from Original Loan")
        latest_r = eff_rate_from_deltas(ss.o_rate, ss.o_rate_deltas)
        rem = max(0, ss.o_term_mo - months_between(ss.o_balance_date, TODAY))
        a1, a2, a3 = st.columns(3)
        with a1:
            computed("Current Remaining Balance (as at today)", fc(ss.o_balance))
        with a2:
            computed("Current Interest Rate (after all changes)", fp(latest_r))
        with a3:
            computed("Remaining Term", f"{rem} months ({rem/12:.1f} yrs)")
    else:
        with c1:
            ss.c_prop_val = st.number_input("Current Property Valuation ($)", value=ss.c_prop_val, min_value=0.0, step=10_000.0, key="w_c_pv2")
        with c2:
            ss.c_prop_date = st.date_input("Valuation Date", value=ss.c_prop_date, key="w_c_pd2")
        with c3:
            if ss.c_prop_val > 0 and ss.c_balance > 0:
                computed("Current LVR (Balance / Valuation)", fp(ss.c_balance / ss.c_prop_val * 100))
            else:
                computed("Current LVR", "—")

        sec("Current Loan Details")
        st.markdown('<div class="note">Remaining Balance is always recorded as at today.</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            ss.c_balance = st.number_input("Current Remaining Balance ($)", value=ss.c_balance, min_value=0.0, step=1_000.0, key="w_c_bal")
        with c2:
            computed("Balance Date", TODAY.strftime("%d %b %Y"))
        with c3:
            ss.c_rate = st.number_input("Current Interest Rate (% p.a.)", value=ss.c_rate, min_value=0.0, max_value=30.0, step=0.01, format="%.4f", key="w_c_r")

        sec("Remaining Term")
        ss.c_use_dates = st.toggle("Calculate from loan end date", value=ss.c_use_dates, key="w_c_ud")
        if ss.c_use_dates:
            c1, c2 = st.columns(2)
            with c1:
                ss.c_end_date = st.date_input("Loan End Date", value=ss.c_end_date, key="w_c_ed")
            with c2:
                if ss.c_end_date > TODAY:
                    ss.c_term_mo = months_between(TODAY, ss.c_end_date)
                    computed("Remaining Term", f"{ss.c_term_mo} months ({ss.c_term_mo/12:.1f} yrs)")
                else:
                    st.warning("End date must be in the future.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                ss.c_term_mo = st.number_input("Remaining Term (months)", value=ss.c_term_mo, min_value=1, max_value=600, step=12, key="w_c_tm")
            with c2:
                computed("Equivalent", f"{ss.c_term_mo/12:.1f} years")

        rate_delta_list("c_rate_deltas", ss.c_rate, "Rate Changes (enter as plus or minus change)")

    sec("Offset Account")
    c1, c2, c3 = st.columns(3)
    with c1:
        ss.c_off_init = st.number_input("Initial Balance ($)", value=ss.c_off_init, min_value=0.0, step=1_000.0, key="w_c_oi")
    with c2:
        ss.c_off_date = st.date_input("Offset Start Date", value=ss.c_off_date, key="w_c_od")
    with c3:
        ss.c_off_monthly = st.number_input("Monthly Addition ($)", value=ss.c_off_monthly, min_value=0.0, step=100.0, key="w_c_om")
    lump_sum_list("c_off_lumps", "Offset Lump Sum Deposits")

    sec("Fees")
    c1, c2, c3 = st.columns(3)
    with c1:
        ss.c_fee_mo = st.number_input("Monthly Fee ($)", value=ss.c_fee_mo, min_value=0.0, step=1.0, key="w_c_fm")
    with c2:
        ss.c_fee_setup = st.number_input("Setup Fee ($)", value=ss.c_fee_setup, min_value=0.0, step=100.0, key="w_c_fs")
    with c3:
        ss.c_fee_other = st.number_input("Other Fee ($)", value=ss.c_fee_other, min_value=0.0, step=100.0, key="w_c_fo")

def section_proposed():
    ss = st.session_state

    sec("Loan Amount and Term")
    c1, c2 = st.columns(2)
    with c1:
        ss.p_loan_amt = st.number_input("Proposed Loan Amount ($)", value=ss.p_loan_amt, min_value=0.0, step=1_000.0, key="w_p_la")
    with c2:
        ss.p_start_date = st.date_input("Proposed Settlement Date", value=ss.p_start_date, key="w_p_sd")

    ss.p_use_dates = st.toggle("Calculate term from start and end dates", value=ss.p_use_dates, key="w_p_ud")
    if ss.p_use_dates:
        c1, c2 = st.columns(2)
        with c1:
            ss.p_end_date = st.date_input("Loan End Date", value=ss.p_end_date, key="w_p_ed")
        with c2:
            if ss.p_end_date > ss.p_start_date:
                ss.p_term_mo = months_between(ss.p_start_date, ss.p_end_date)
                computed("Calculated Term", f"{ss.p_term_mo} months ({ss.p_term_mo/12:.1f} yrs)")
    else:
        c1, c2 = st.columns(2)
        with c1:
            ss.p_term_mo = st.number_input("Loan Term (months)", value=ss.p_term_mo, min_value=1, max_value=600, step=12, key="w_p_tm")
        with c2:
            computed("Equivalent", f"{ss.p_term_mo/12:.1f} years")

    sec("Interest Rates")
    c1, c2, c3 = st.columns(3)
    with c1:
        ss.p_adv_rate = st.number_input("Advertised Rate (% p.a.)", value=ss.p_adv_rate, min_value=0.0, max_value=30.0, step=0.01, format="%.2f", key="w_p_ar")
    with c2:
        ss.p_comp_in = st.number_input("Lender Comparison Rate (% p.a.)", value=ss.p_comp_in, min_value=0.0, max_value=30.0, step=0.01, format="%.2f", key="w_p_cr")
    with c3:
        st.markdown('<div class="note">Effective and ASIC comparison rates are calculated automatically based on fees entered below.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        ss.p_var_rate = st.number_input("Variable Rate (% p.a.)", value=ss.p_var_rate, min_value=0.0, max_value=30.0, step=0.01, format="%.4f", key="w_p_vr")
    with c2:
        ss.p_fix_rate = st.number_input("Fixed Rate (% p.a.)", value=ss.p_fix_rate, min_value=0.0, max_value=30.0, step=0.01, format="%.4f", key="w_p_fr")

    c1, c2 = st.columns(2)
    with c1:
        ss.p_fix_yrs = st.number_input("Fixed Period (years)", value=ss.p_fix_yrs, min_value=1, max_value=30, step=1, key="w_p_fy")
    with c2:
        ss.p_rev_rate = st.number_input("Reversion Rate after Fixed Period (% p.a.)", value=ss.p_rev_rate, min_value=0.0, max_value=30.0, step=0.01, format="%.4f", key="w_p_rr")

    sec("Variable / Fixed Split")
    ss.p_split_auto = st.toggle(
        "Auto-calculate optimal split (0.1% increment sweep, fixed period only)",
        value=ss.p_split_auto, key="w_p_sa"
    )
    if not ss.p_split_auto:
        c1, c2 = st.columns(2)
        with c1:
            ss.p_split_pct = st.slider("Fixed Component (%)", 0.0, 100.0, ss.p_split_pct, 0.5, key="w_p_sp")
        with c2:
            computed(
                "Allocation",
                f"Variable {fc(ss.p_loan_amt * (100 - ss.p_split_pct) / 100)} / Fixed {fc(ss.p_loan_amt * ss.p_split_pct / 100)}"
            )
    else:
        st.markdown(
            '<div class="note">Optimal split minimises cumulative interest plus closing balance at end of the fixed period. '
            'Evaluated across all 1,001 combinations in 0.1% increments. Both components are treated as separate loans '
            'funded from the proposed loan amount.</div>',
            unsafe_allow_html=True
        )

    rate_delta_list("p_var_deltas", ss.p_var_rate, "Variable Rate Changes (enter as plus or minus change)")

    sec("Offset Account (applied to variable component)")
    c1, c2, c3 = st.columns(3)
    with c1:
        ss.p_off_init = st.number_input("Initial Balance ($)", value=ss.p_off_init, min_value=0.0, step=1_000.0, key="w_p_oi")
    with c2:
        ss.p_off_date = st.date_input("Offset Start Date", value=ss.p_off_date, key="w_p_od")
    with c3:
        ss.p_off_monthly = st.number_input("Monthly Addition ($)", value=ss.p_off_monthly, min_value=0.0, step=100.0, key="w_p_om")
    lump_sum_list("p_off_lumps", "Offset Lump Sum Deposits")

    sec("Fees")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ss.p_fee_mo = st.number_input("Monthly Fee ($)", value=ss.p_fee_mo, min_value=0.0, step=1.0, key="w_p_fm")
    with c2:
        ss.p_fee_setup = st.number_input("Establishment / Setup Fee ($)", value=ss.p_fee_setup, min_value=0.0, step=100.0, key="w_p_fs")
    with c3:
        ss.p_fee_break = st.number_input("Breakage Fee ($)", value=ss.p_fee_break, min_value=0.0, step=100.0, key="w_p_fb")
    with c4:
        ss.p_fee_other = st.number_input("Other One-off Fee ($)", value=ss.p_fee_other, min_value=0.0, step=100.0, key="w_p_fo")

def section_strategy():
    ss = st.session_state

    sec("Strategy")
    ss.strategy = st.radio(
        "Refinancing strategy",
        ["Conservative (80% fixed)", "Balanced (optimal split)", "Aggressive (0% fixed)"],
        index=["Conservative (80% fixed)", "Balanced (optimal split)", "Aggressive (0% fixed)"].index(
            ss.strategy if ss.strategy in ["Conservative (80% fixed)", "Balanced (optimal split)", "Aggressive (0% fixed)"]
            else "Balanced (optimal split)"
        ),
        key="w_strat", horizontal=True
    )
    st.markdown("""
    <div class="note">
    Conservative: 80% fixed — maximises payment certainty, protects against rate rises.<br>
    Balanced: mathematically optimal split — minimises total interest plus balance at end of fixed period.<br>
    Aggressive: 100% variable — maximum flexibility; benefits most from rate falls; best paired with offset account.
    </div>""", unsafe_allow_html=True)

    sec("Payment Behaviour on Rate Changes")
    ss.maintain_pmt = st.toggle(
        "When rates fall, maintain current repayment (pays loan off faster rather than reducing payment)",
        value=ss.maintain_pmt, key="w_mp"
    )
    st.markdown(
        '<div class="note">When rates rise, repayments always increase to maintain the remaining loan term. '
        'The term can only decrease — it cannot extend beyond the remaining term.</div>',
        unsafe_allow_html=True
    )

    sec("RBA Cash Rate Scenario")
    c1, c2 = st.columns([3, 1])
    with c1:
        ss.rba_bps = st.slider("RBA cash rate change (basis points)", -300, 300, ss.rba_bps, 25, key="w_rba")
        if ss.rba_bps != 0:
            d = "increase" if ss.rba_bps > 0 else "decrease"
            st.markdown(f'<div class="note">Applies a {abs(ss.rba_bps)} bps ({abs(ss.rba_bps)/100:.2f}%) {d} to the variable rate in scenarios.</div>', unsafe_allow_html=True)
    with c2:
        if st.button("Fetch live RBA rate", key="btn_rba"):
            with st.spinner("Fetching..."):
                r = fetch_rba_rate()
            if r:
                ss.rba_live = r
                st.success(f"RBA: {fp(r)}")
            else:
                st.warning("Could not fetch. Check rba.gov.au directly.")
        if ss.rba_live:
            computed("Current RBA Rate", fp(ss.rba_live))

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD TABS
# ══════════════════════════════════════════════════════════════════════════════

def dash_overview(R):
    ss = st.session_state
    dfs = [R["df_orig"], R["df_curr"], R["df_ps"]]
    labels = ["Original Loan", "Current Loan", "Proposed (Split)"]
    colors = [C_ORIG, C_CURR, C_SPLIT]

    cols = st.columns(3)
    for col, df, lbl, clr in zip(cols, dfs, labels, colors):
        with col:
            if df is None or df.empty:
                st.info(f"No data for {lbl}")
                continue
            st.markdown(
                f'<div style="color:{clr}; font-weight:600; font-size:0.82rem; '
                f'margin-bottom:8px; padding-bottom:4px; border-bottom:1px solid #1e2d4a">{lbl}</div>',
                unsafe_allow_html=True
            )
            st.markdown(metric_card("Initial Monthly Repayment", fc(df["Payment"].iloc[0])), unsafe_allow_html=True)
            st.markdown(metric_card("Total Interest", fc(df["Cum Interest"].iloc[-1])), unsafe_allow_html=True)
            st.markdown(metric_card("Total Cost", fc(df["Cum Paid"].iloc[-1])), unsafe_allow_html=True)
            st.markdown(metric_card("Loan Term", f"{len(df)} mo  /  {len(df)/12:.1f} yr"), unsafe_allow_html=True)

    df_c, df_s = R["df_curr"], R["df_ps"]
    if df_c is not None and not df_c.empty and df_s is not None and not df_s.empty:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("**Proposed vs Current**")
        int_sav = df_c["Cum Interest"].iloc[-1] - df_s["Cum Interest"].iloc[-1]
        cost_sav = df_c["Cum Paid"].iloc[-1] - df_s["Cum Paid"].iloc[-1]
        cols2 = st.columns(5)
        items = [
            ("Interest Saved", fc(int_sav)),
            ("Total Cost Saved", fc(cost_sav)),
            ("Optimal Fixed Split", fp(R["best_pct"], 1)),
            ("Effective Rate (Proposed)", fp(R["eff_r"])),
            ("ASIC Comparison Rate", fp(R["comp_r"])),
        ]
        for col, (lbl, val) in zip(cols2, items):
            with col:
                st.markdown(metric_card(lbl, val), unsafe_allow_html=True)

def dash_monthly_payments(R):
    df_o, df_c = R["df_orig"], R["df_curr"]
    df_ps, df_pv, df_pf = R["df_ps"], R["df_pv"], R["df_pf"]

    fig = go.Figure()
    for df, name, clr, dash in [
        (df_o, "Original", C_ORIG, "dot"),
        (df_c, "Current", C_CURR, "dash"),
        (df_ps, "Proposed Split", C_SPLIT, "solid"),
        (df_pv, "Variable Component", C_VAR, "dot"),
        (df_pf, "Fixed Component", C_FIX, "dot"),
    ]:
        if df is not None and not df.empty:
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Payment"],
                                     name=name, line=dict(color=clr, width=2, dash=dash)))
    fig.update_layout(**PLOT_BASE, title="Monthly Repayments Over Time", yaxis_title="Repayment ($)")
    st.plotly_chart(fig, use_container_width=True)

    if df_ps is not None and not df_ps.empty:
        s = df_ps.iloc[::12]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=s["Date"], y=s["Principal"], name="Principal", marker_color=C_VAR))
        fig2.add_trace(go.Bar(x=s["Date"], y=s["Interest"], name="Interest", marker_color=C_FIX))
        if df_ps["Interest Saved"].sum() > 0:
            fig2.add_trace(go.Scatter(x=s["Date"], y=s["Interest Saved"],
                                      name="Interest Saved (Offset)", mode="lines+markers",
                                      line=dict(color=C_ORIG, width=2), yaxis="y2"))
            fig2.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False,
                                           title="Saved ($)", color=C_ORIG))
        fig2.update_layout(**PLOT_BASE, barmode="stack",
                           title="Annual Breakdown — Proposed Split", yaxis_title="Amount ($)")
        st.plotly_chart(fig2, use_container_width=True)

def dash_loan_balance(R):
    ss = st.session_state
    df_o, df_c = R["df_orig"], R["df_curr"]
    df_ps, df_pv, df_pf = R["df_ps"], R["df_pv"], R["df_pf"]

    fig = go.Figure()
    for df, name, clr, dash in [
        (df_o, "Original", C_ORIG, "dot"),
        (df_c, "Current", C_CURR, "dash"),
        (df_ps, "Proposed Split", C_SPLIT, "solid"),
        (df_pv, "Variable Component", C_VAR, "dot"),
        (df_pf, "Fixed Component", C_FIX, "dot"),
    ]:
        if df is not None and not df.empty:
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Closing Balance"],
                                     name=name, line=dict(color=clr, width=2, dash=dash)))
    fig.update_layout(**PLOT_BASE, title="Outstanding Balance Over Time", yaxis_title="Balance ($)")
    st.plotly_chart(fig, use_container_width=True)

    curr_val = ss.c_prop_val if not ss.c_is_cont else ss.o_prop_val
    if curr_val > 0 and df_ps is not None and not df_ps.empty:
        lvr = df_ps["Closing Balance"] / curr_val * 100
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_ps["Date"], y=lvr, name="LVR",
                                   line=dict(color=C_SPLIT, width=2),
                                   fill="tozeroy", fillcolor="rgba(196,122,245,0.07)"))
        fig2.add_hline(y=80, line=dict(color=C_FIX, dash="dash", width=1), annotation_text="80% LMI threshold")
        fig2.add_hline(y=60, line=dict(color=C_VAR, dash="dot", width=1), annotation_text="60%")
        fig2.update_layout(**PLOT_BASE, title="LVR Over Time — Proposed Split", yaxis_title="LVR (%)")
        st.plotly_chart(fig2, use_container_width=True)

def dash_interest_analysis(R):
    df_o, df_c, df_ps = R["df_orig"], R["df_curr"], R["df_ps"]

    fig = go.Figure()
    for df, name, clr in [(df_o, "Original", C_ORIG), (df_c, "Current", C_CURR), (df_ps, "Proposed Split", C_SPLIT)]:
        if df is not None and not df.empty:
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Cum Interest"],
                                     name=name, line=dict(color=clr, width=2)))
    fig.update_layout(**PLOT_BASE, title="Cumulative Interest Paid", yaxis_title="Cumulative Interest ($)")
    st.plotly_chart(fig, use_container_width=True)

    if df_ps is not None and not df_ps.empty and df_ps["Cum Interest Saved"].iloc[-1] > 1:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_ps["Date"], y=df_ps["Cum Interest Saved"],
                                   name="Cumulative Offset Savings",
                                   line=dict(color=C_VAR, width=2),
                                   fill="tozeroy", fillcolor="rgba(48,217,150,0.08)"))
        fig2.update_layout(**PLOT_BASE, title="Cumulative Interest Saved via Offset Account",
                           yaxis_title="Savings ($)")
        st.plotly_chart(fig2, use_container_width=True)

def dash_optimal_split(R):
    ss = st.session_state
    sdf = R["split_df"]
    best = R["best_pct"]

    st.markdown(
        f'<div class="note">Optimal fixed component: <strong>{best:.1f}%</strong> '
        f'({100-best:.1f}% variable). Evaluated in 0.1% increments across 1,001 scenarios '
        f'covering the {ss.p_fix_yrs}-year fixed period only. Both components treated as '
        f'separate loans from the proposed balance.</div>',
        unsafe_allow_html=True
    )

    p_f = ss.p_loan_amt * best / 100
    p_v = ss.p_loan_amt * (100 - best) / 100
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(metric_card("Optimal Fixed %", fp(best, 1)), unsafe_allow_html=True)
    with c2: st.markdown(metric_card("Fixed Amount", fc(p_f)), unsafe_allow_html=True)
    with c3: st.markdown(metric_card("Variable Amount", fc(p_v)), unsafe_allow_html=True)
    with c4:
        best_row = sdf.iloc[sdf["Objective"].idxmin()]
        st.markdown(metric_card("Objective (Min)", fc(best_row["Objective"])), unsafe_allow_html=True)

    fig = make_subplots(rows=1, cols=2,
                         subplot_titles=["Objective: Cum. Interest + End Balance", "Components"],
                         horizontal_spacing=0.1)
    best_obj_val = sdf.iloc[sdf["Objective"].idxmin()]["Objective"]
    fig.add_trace(go.Scatter(x=sdf["Fixed %"], y=sdf["Objective"],
                              name="Objective", line=dict(color=C_SPLIT, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[best], y=[best_obj_val], mode="markers",
                              marker=dict(symbol="diamond", size=12, color=C_FIX),
                              name=f"Optimal {best:.1f}%"), row=1, col=1)
    fig.add_trace(go.Scatter(x=sdf["Fixed %"], y=sdf["Cum Interest"],
                              name="Cum Interest", line=dict(color=C_FIX, width=1.5)), row=1, col=2)
    fig.add_trace(go.Scatter(x=sdf["Fixed %"], y=sdf["End Balance"],
                              name="End Balance", line=dict(color=C_VAR, width=1.5)), row=1, col=2)
    fig.update_layout(**PLOT_BASE)
    fig.update_xaxes(title_text="Fixed Component (%)", gridcolor=C_GRID)
    fig.update_yaxes(title_text="$", gridcolor=C_GRID)
    st.plotly_chart(fig, use_container_width=True)

    tbl = sdf[sdf["Fixed %"] % 5 == 0].copy()
    st.dataframe(pd.DataFrame({
        "Fixed %": tbl["Fixed %"].apply(lambda x: f"{x:.0f}%"),
        "Variable %": tbl["Variable %"].apply(lambda x: f"{x:.0f}%"),
        "Cum Interest": tbl["Cum Interest"].apply(fc),
        "End Balance": tbl["End Balance"].apply(fc),
        "Objective": tbl["Objective"].apply(fc),
    }), use_container_width=True, hide_index=True)

def dash_strategy(R):
    ss = st.session_state
    loan, term = ss.p_loan_amt, ss.p_term_mo
    fix_mo = ss.p_fix_yrs * 12
    best = R["best_pct"]

    strats = {
        "Conservative\n(80% Fixed)": (80.0, C_FIX),
        f"Balanced\n({best:.1f}% Fixed)": (best, C_SPLIT),
        "Aggressive\n(0% Fixed)": (0.0, C_VAR),
    }
    totals_ci, totals_pmt = {}, {}

    for name, (pct_f, _) in strats.items():
        p_f, p_v = loan * pct_f / 100, loan * (100 - pct_f) / 100
        bal_f, ci_f = fast_partial(p_f, ss.p_fix_rate, term, fix_mo) if p_f > 0 else (0, 0)
        bal_v, ci_v = fast_partial(p_v, ss.p_var_rate, term, fix_mo) if p_v > 0 else (0, 0)
        rem = max(0, term - fix_mo)
        _, ci_f2 = fast_partial(bal_f, ss.p_rev_rate, rem, rem) if (rem > 0 and bal_f > 0) else (0, 0)
        _, ci_v2 = fast_partial(bal_v, ss.p_var_rate, rem, rem) if (rem > 0 and bal_v > 0) else (0, 0)
        n = name.split("\n")[0]
        totals_ci[n] = ci_f + ci_v + ci_f2 + ci_v2
        totals_pmt[n] = (calc_payment(p_f, ss.p_fix_rate, term) if p_f > 0 else 0) + \
                         (calc_payment(p_v, ss.p_var_rate, term) if p_v > 0 else 0)

    c1, c2 = st.columns(2)
    clrs = [C_FIX, C_SPLIT, C_VAR]
    with c1:
        fig = go.Figure(go.Bar(x=list(totals_ci.keys()), y=list(totals_ci.values()),
                                marker_color=clrs,
                                text=[fc(v) for v in totals_ci.values()], textposition="outside"))
        fig.update_layout(**PLOT_BASE, title="Estimated Total Interest by Strategy", yaxis_title="Total Interest ($)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = go.Figure(go.Bar(x=list(totals_pmt.keys()), y=list(totals_pmt.values()),
                                 marker_color=clrs,
                                 text=[fc(v) for v in totals_pmt.values()], textposition="outside"))
        fig2.update_layout(**PLOT_BASE, title="Initial Monthly Repayment by Strategy", yaxis_title="Monthly Repayment ($)")
        st.plotly_chart(fig2, use_container_width=True)

def dash_scenarios(R):
    scenarios = R["scenarios"]
    if not scenarios:
        st.warning("No scenario data available.")
        return

    base = scenarios.get("Base", {})
    base_pmt = base.get("payment", 0)
    base_int = base.get("total_interest", 0)
    base_term = base.get("term_months", 0)

    # ── Repayment scenario table ──────────────────────────────────────────
    st.markdown("**Monthly Repayments Under Rate Scenarios**")
    st.markdown(
        '<div class="note">'
        'Rate rises always increase repayments to maintain remaining loan term — the term is never extended. '
        'Rate falls reduce repayments (or shorten term if maintaining payment). '
        'Scenarios apply to the full proposed loan as variable-only for direct comparison.'
        '</div>',
        unsafe_allow_html=True
    )

    rows = []
    for label, data in scenarios.items():
        dpmt = data["payment"] - base_pmt
        dint = data["total_interest"] - base_int
        dterm = data["term_months"] - base_term
        rows.append({
            "Scenario": label,
            "Variable Rate": fp(data["rate"]),
            "Monthly Repayment": fc(data["payment"]),
            "vs Base": f"{'+' if dpmt >= 0 else ''}{fc(dpmt)}",
            "Total Interest": fc(data["total_interest"]),
            "Interest vs Base": f"{'+' if dint >= 0 else ''}{fc(dint)}",
            "Term (months)": str(data["term_months"]),
            "Term vs Base": f"{'+' if dterm >= 0 else ''}{dterm} mo",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Repayment over time per scenario ──────────────────────────────────
    fig_pmt = go.Figure()
    scen_colors = [C_SPLIT, C_FIX, "#f59e0b", "#ef4444", C_VAR, "#22d3ee", C_ORIG]
    for (label, data), clr in zip(scenarios.items(), scen_colors):
        df_s = data.get("df")
        if df_s is not None and not df_s.empty:
            fig_pmt.add_trace(go.Scatter(
                x=df_s["Date"], y=df_s["Payment"], name=label,
                line=dict(color=clr, width=2 if label == "Base" else 1.5,
                          dash="solid" if label == "Base" else "dash")
            ))
    fig_pmt.update_layout(**PLOT_BASE, title="Monthly Repayments — Rate Scenarios",
                          yaxis_title="Monthly Repayment ($)")
    st.plotly_chart(fig_pmt, use_container_width=True)

    # ── Balance over time per scenario ────────────────────────────────────
    fig_bal = go.Figure()
    for (label, data), clr in zip(scenarios.items(), scen_colors):
        df_s = data.get("df")
        if df_s is not None and not df_s.empty:
            fig_bal.add_trace(go.Scatter(
                x=df_s["Date"], y=df_s["Closing Balance"], name=label,
                line=dict(color=clr, width=2 if label == "Base" else 1.5,
                          dash="solid" if label == "Base" else "dash")
            ))
    fig_bal.update_layout(**PLOT_BASE, title="Outstanding Balance — Rate Scenarios",
                          yaxis_title="Balance ($)")
    st.plotly_chart(fig_bal, use_container_width=True)

def dash_schedules(R):
    options = {
        "Original Loan": R["df_orig"],
        "Current Loan": R["df_curr"],
        "Proposed Variable Component": R["df_pv"],
        "Proposed Fixed Component": R["df_pf"],
        "Proposed Split (Combined)": R["df_ps"],
    }
    available = {k: v for k, v in options.items() if v is not None and not v.empty}
    if not available:
        st.warning("No schedules to display.")
        return

    sel = st.selectbox("Select schedule to view", list(available.keys()), key="sch_sel")
    df = available[sel].copy()

    disp = df.copy()
    for c in ["Opening Balance", "Avg Offset", "Net Debt", "Interest", "Interest Saved",
              "Principal", "Fees", "Payment", "Closing Balance",
              "Cum Interest", "Cum Paid", "Cum Interest Saved"]:
        if c in disp.columns:
            disp[c] = disp[c].apply(fc)
    if "Rate %" in disp.columns:
        disp["Rate %"] = disp["Rate %"].apply(fp)

    st.dataframe(disp, use_container_width=True, hide_index=True, height=440)

    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    st.download_button(
        label=f"Download {sel} as CSV",
        data=buf.getvalue(),
        file_name=f"schedule_{sel.lower().replace(' ', '_').replace('(', '').replace(')', '')}.csv",
        mime="text/csv",
        key=f"dl_{sel}"
    )

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    init_state()
    ss = st.session_state

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="padding:24px 0 20px; border-bottom:1px solid #1e2d4a; margin-bottom:24px;">
        <h1 style="color:#d4dbe8; font-size:1.55rem; font-weight:600; margin:0 0 4px;">
            Australian Mortgage Refinance Analyser
        </h1>
        <p style="color:#64748b; font-size:0.82rem; margin:0;">
            Daily-interest amortisation &nbsp;&middot;&nbsp;
            Optimal variable/fixed split &nbsp;&middot;&nbsp;
            ASIC comparison rates &nbsp;&middot;&nbsp;
            RBA rate scenarios
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Input sections ────────────────────────────────────────────────────────
    with st.expander("Original Loan", expanded=True):
        section_original()

    with st.expander("Current Loan", expanded=True):
        section_current()

    with st.expander("Proposed Loan", expanded=True):
        section_proposed()

    with st.expander("Strategy and Scenarios", expanded=True):
        section_strategy()

    # ── Actions ───────────────────────────────────────────────────────────────
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    c1, c2, _ = st.columns([2, 1, 5])
    with c1:
        st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
        go_calc = st.button("Calculate and Analyse", key="btn_go")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="btn-danger">', unsafe_allow_html=True)
        do_reset = st.button("Reset", key="btn_rst")
        st.markdown("</div>", unsafe_allow_html=True)

    if do_reset:
        for k in list(ss.keys()):
            del ss[k]
        st.rerun()

    if go_calc:
        with st.spinner("Calculating..."):
            run_calc()

    # ── Dashboard ──────────────────────────────────────────────────────────────
    if ss.results is not None:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            '<h2 style="color:#d4dbe8; font-size:1.15rem; font-weight:600; margin:0 0 16px;">Analysis Dashboard</h2>',
            unsafe_allow_html=True
        )
        tabs = st.tabs([
            "Overview", "Monthly Payments", "Loan Balance",
            "Interest Analysis", "Optimal Split", "Strategy",
            "Rate Scenarios", "Schedules",
        ])
        R = ss.results
        with tabs[0]: dash_overview(R)
        with tabs[1]: dash_monthly_payments(R)
        with tabs[2]: dash_loan_balance(R)
        with tabs[3]: dash_interest_analysis(R)
        with tabs[4]: dash_optimal_split(R)
        with tabs[5]: dash_strategy(R)
        with tabs[6]: dash_scenarios(R)
        with tabs[7]: dash_schedules(R)

    else:
        st.markdown("""
        <div style="text-align:center; padding:64px 0; color:#64748b;">
            <div style="font-size:0.95rem; font-weight:500; color:#8892b0; margin-bottom:6px;">
                Enter loan details and click Calculate and Analyse
            </div>
            <div style="font-size:0.82rem;">Results will appear here</div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
