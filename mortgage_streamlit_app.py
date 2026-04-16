"""
Australian Mortgage Refinance Analyser
Streamlit single-file application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date, datetime, timedelta
import calendar
import requests
import io

# ── Optional scipy ────────────────────────────────────────────────────────────
try:
    from scipy.optimize import brentq
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & CSS
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AU Mortgage Refinance Analyser",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background-color: #080c14; color: #e8eaf6; }

section[data-testid="stSidebar"] { background-color: #0d1526; }

.stTabs [data-baseweb="tab-list"] {
    background-color: #0d1526;
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    color: #8892b0;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 500;
    font-size: 0.85rem;
}
.stTabs [aria-selected="true"] {
    background-color: #1a2744 !important;
    color: #4a9af5 !important;
}

.metric-card {
    background: #0d1526;
    border: 1px solid #1a2744;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.metric-label { color: #8892b0; font-size: 0.75rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-value { color: #e8eaf6; font-size: 1.5rem; font-weight: 700; margin-top: 4px; }
.metric-delta-pos { color: #30d996; font-size: 0.8rem; }
.metric-delta-neg { color: #e94560; font-size: 0.8rem; }

.section-header {
    color: #4a9af5;
    font-size: 0.9rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 8px 0 4px;
    border-bottom: 1px solid #1a2744;
    margin: 16px 0 12px;
}

div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input,
div[data-testid="stDateInput"] input,
div[data-testid="stSelectbox"] div {
    background-color: #0d1526 !important;
    border-color: #1a2744 !important;
    color: #e8eaf6 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #4a9af5 0%, #2d6bc4 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 24px;
    width: 100%;
}
.stButton > button:hover { opacity: 0.9; }

.reset-btn > button {
    background: transparent !important;
    border: 1px solid #e94560 !important;
    color: #e94560 !important;
}

.info-box {
    background: #0d1526;
    border-left: 3px solid #4a9af5;
    border-radius: 4px;
    padding: 10px 14px;
    margin: 8px 0;
    font-size: 0.85rem;
    color: #8892b0;
}

table { width: 100%; }
th { color: #4a9af5 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# COLOUR PALETTE
# ══════════════════════════════════════════════════════════════════════════════

C_ORIG   = "#4a9af5"
C_CURR   = "#f5a94a"
C_VAR    = "#30d996"
C_FIX    = "#e94560"
C_SPLIT  = "#c47af5"
C_PAPER  = "#0d1526"
C_PLOT   = "#080c14"
C_GRID   = "#1a2744"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=C_PAPER,
    plot_bgcolor=C_PLOT,
    font=dict(family="Inter", color="#e8eaf6"),
    xaxis=dict(gridcolor=C_GRID, zerolinecolor=C_GRID),
    yaxis=dict(gridcolor=C_GRID, zerolinecolor=C_GRID),
    legend=dict(bgcolor=C_PAPER, bordercolor=C_GRID),
    margin=dict(t=40, b=40, l=60, r=20),
)

# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def fmt_currency(v, decimals=0):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    neg = v < 0
    s = f"${abs(v):,.{decimals}f}"
    return f"-{s}" if neg else s

def fmt_pct(v, decimals=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.{decimals}f}%"

def next_monthly_date(anchor_day: int, from_date: date) -> date:
    """Return next monthly date at anchor_day on or after from_date."""
    y, m = from_date.year, from_date.month
    last = calendar.monthrange(y, m)[1]
    d = min(anchor_day, last)
    candidate = date(y, m, d)
    if candidate < from_date:
        m += 1
        if m > 12:
            m = 1
            y += 1
        last = calendar.monthrange(y, m)[1]
        candidate = date(y, m, min(anchor_day, last))
    return candidate

def days_between(d1: date, d2: date) -> int:
    return (d2 - d1).days

def parse_date(v):
    if isinstance(v, date):
        return v
    if isinstance(v, datetime):
        return v.date()
    try:
        return datetime.strptime(str(v), "%Y-%m-%d").date()
    except Exception:
        return date.today()

# ══════════════════════════════════════════════════════════════════════════════
# FINANCIAL ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def calc_monthly_payment(principal: float, annual_rate: float, term_months: int) -> float:
    if principal <= 0 or term_months <= 0:
        return 0.0
    if annual_rate == 0:
        return principal / term_months
    r = annual_rate / 100 / 12
    return principal * r * (1 + r) ** term_months / ((1 + r) ** term_months - 1)

def build_offset_schedule(
    start_date: date,
    end_date: date,
    initial_amount: float,
    monthly_addition: float,
    lump_sums: list,      # [[date, amount], ...]
) -> dict:               # {date: running_balance}
    """Build a dict mapping every date in range to offset balance."""
    schedule = {}
    balance = initial_amount
    lump_dict = {}
    for row in lump_sums:
        d = parse_date(row[0])
        amt = float(row[1])
        lump_dict[d] = lump_dict.get(d, 0) + amt

    current = start_date
    last_month_add = start_date.month
    while current <= end_date:
        if current.month != last_month_add:
            balance += monthly_addition
            last_month_add = current.month
        if current in lump_dict:
            balance += lump_dict[current]
        balance = max(0, balance)
        schedule[current] = balance
        current += timedelta(days=1)
    return schedule

def build_rate_schedule(base_rate: float, changes: list) -> list:
    """Return sorted list of (date, rate) from base + change list."""
    result = [(date(1900, 1, 1), base_rate)]
    for row in changes:
        if row[0] and row[1] is not None:
            result.append((parse_date(row[0]), float(row[1])))
    result.sort(key=lambda x: x[0])
    return result

def get_rate_on_date(rate_schedule: list, d: date) -> float:
    """Get the applicable rate on a given date."""
    r = rate_schedule[0][1]
    for rd, rv in rate_schedule:
        if rd <= d:
            r = rv
        else:
            break
    return r

def amortize(
    loan_amount: float,
    start_date: date,
    term_months: int,
    rate_schedule: list,        # [(date, annual_rate%), ...]
    offset_schedule: dict,      # {date: balance}
    monthly_fees: float = 0,
    maintain_payment: bool = True,
    max_extra_months: int = 120,
) -> pd.DataFrame:
    """
    Full amortisation engine.
    Interest = (balance - offset) * (annual_rate/365) * days_in_period
    Payment order: interest added first, then deduction.
    """
    if loan_amount <= 0:
        return pd.DataFrame()

    anchor_day = start_date.day
    balance = loan_amount
    cumulative_interest = 0.0
    cumulative_paid = 0.0
    cumulative_interest_saved = 0.0

    # Initial payment
    initial_rate = get_rate_on_date(rate_schedule, start_date)
    payment = calc_monthly_payment(loan_amount, initial_rate, term_months)
    current_payment = payment

    rows = []
    period_date = start_date
    month_num = 0
    max_months = term_months + max_extra_months

    while balance > 0.01 and month_num < max_months:
        month_num += 1
        period_start = period_date

        # Next payment date
        next_m = period_date.month + 1
        next_y = period_date.year
        if next_m > 12:
            next_m = 1
            next_y += 1
        last_day = calendar.monthrange(next_y, next_m)[1]
        period_end = date(next_y, next_m, min(anchor_day, last_day))

        days = days_between(period_start, period_end)

        # Rate for this period (use rate at period start)
        ann_rate = get_rate_on_date(rate_schedule, period_start)
        daily_rate = ann_rate / 100 / 365

        # Average offset over period
        offset_dates = [period_start + timedelta(days=i) for i in range(days)]
        if offset_schedule:
            offset_vals = [offset_schedule.get(d, offset_schedule.get(
                max((k for k in offset_schedule if k <= d), default=period_start), 0))
                for d in offset_dates]
            avg_offset = min(np.mean(offset_vals), balance)
        else:
            avg_offset = 0.0

        net_debt = max(0, balance - avg_offset)
        interest = net_debt * daily_rate * days

        # Interest saved vs no offset
        interest_no_offset = balance * daily_rate * days
        interest_saved = interest_no_offset - interest
        cumulative_interest_saved += interest_saved

        # Recalculate payment if rate changed (maintain term → payment changes)
        if not maintain_payment:
            months_remaining = term_months - month_num + 1
            if months_remaining > 0:
                current_payment = calc_monthly_payment(balance, ann_rate, months_remaining)

        opening_balance = balance
        balance = balance + interest
        principal = min(current_payment - interest, balance)
        principal = max(0, principal)
        payment_made = min(current_payment, balance)
        balance = max(0, balance - payment_made)

        cumulative_interest += interest
        cumulative_paid += payment_made + monthly_fees

        rows.append({
            "Month": month_num,
            "Date": period_end,
            "Opening_Balance": opening_balance,
            "Offset": avg_offset,
            "Net_Debt": net_debt,
            "Rate": ann_rate,
            "Interest": interest,
            "Interest_Saved": interest_saved,
            "Principal": principal,
            "Fees": monthly_fees,
            "Payment": payment_made,
            "Closing_Balance": balance,
            "Cum_Interest": cumulative_interest,
            "Cum_Paid": cumulative_paid,
            "Cum_Interest_Saved": cumulative_interest_saved,
        })

        period_date = period_end

    return pd.DataFrame(rows)

def fast_partial_amort(principal: float, annual_rate: float, term_months: int, n_months: int):
    """
    Analytical O(1) partial amortisation.
    Returns (closing_balance, cumulative_interest) after n_months.
    """
    if principal <= 0:
        return 0.0, 0.0
    if annual_rate == 0:
        pmt = principal / term_months
        balance = max(0, principal - pmt * n_months)
        cum_int = 0.0
        return balance, cum_int
    r = annual_rate / 100 / 12
    pmt = principal * r * (1 + r) ** term_months / ((1 + r) ** term_months - 1)
    balance_n = principal * (1 + r) ** n_months - pmt * ((1 + r) ** n_months - 1) / r
    balance_n = max(0, balance_n)
    cum_interest = pmt * n_months - (principal - balance_n)
    return balance_n, cum_interest

def comparison_rate(principal: float, setup_fee: float, monthly_fee: float,
                    annual_rate: float, term_months: int = 300) -> float:
    """
    ASIC standard: PV = $150,000, n = 300 months.
    Solve: 150000 - setup_fee = (R + C) × [1 - (1+i)^-n] / i
    """
    pv_std = 150_000.0
    n = 300
    r_monthly = annual_rate / 100 / 12
    pmt_std = calc_monthly_payment(pv_std, annual_rate, n)
    total_pmt = pmt_std + monthly_fee

    target = pv_std - setup_fee
    if not HAS_SCIPY or target <= 0:
        return annual_rate  # fallback

    def f(i):
        if i == 0:
            return total_pmt * n - target
        return total_pmt * (1 - (1 + i) ** -n) / i - target

    try:
        i_sol = brentq(f, 1e-8, 0.1)
        return i_sol * 12 * 100
    except Exception:
        return annual_rate

def effective_rate(principal: float, setup_fee: float, monthly_fee: float,
                   annual_rate: float, term_months: int) -> float:
    """Effective rate using actual loan amount and term."""
    if not HAS_SCIPY or principal <= 0:
        return annual_rate
    pmt = calc_monthly_payment(principal, annual_rate, term_months)
    total_pmt = pmt + monthly_fee
    target = principal - setup_fee
    if target <= 0:
        return annual_rate

    def f(i):
        if i == 0:
            return total_pmt * term_months - target
        return total_pmt * (1 - (1 + i) ** -term_months) / i - target

    try:
        i_sol = brentq(f, 1e-8, 0.5)
        return i_sol * 12 * 100
    except Exception:
        return annual_rate

def fetch_rba_rate() -> float | None:
    """Attempt to fetch current RBA cash rate."""
    try:
        url = "https://www.rba.gov.au/statistics/tables/json/f1-1.json"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            series = data.get("series", {})
            for k, v in series.items():
                if "cash" in k.lower() and "rate" in k.lower():
                    observations = v.get("observations", {})
                    if observations:
                        latest_key = sorted(observations.keys())[-1]
                        val = observations[latest_key][0]
                        return float(val)
    except Exception:
        pass
    return None

def calc_optimal_split(
    loan_amount: float,
    var_rate: float,
    fix_rate: float,
    reversion_rate: float,
    fix_period_years: int,
    total_term_months: int,
) -> tuple:
    """
    Sweep fixed% 0→100 in 0.1% increments.
    Objective = cumulative_interest + closing_balance after fixed period.
    Returns (optimal_fixed_pct, results_df)
    """
    n_fix = fix_period_years * 12
    n_fix = min(n_fix, total_term_months)

    results = []
    best_obj = float("inf")
    best_pct = 50.0

    for pct_fixed_int in range(0, 1001):
        pct_fixed = pct_fixed_int / 10.0
        pct_var = 100.0 - pct_fixed

        p_fix = loan_amount * pct_fixed / 100
        p_var = loan_amount * pct_var / 100

        # Fixed component
        if p_fix > 0:
            bal_fix, ci_fix = fast_partial_amort(p_fix, fix_rate, total_term_months, n_fix)
        else:
            bal_fix, ci_fix = 0.0, 0.0

        # Variable component
        if p_var > 0:
            bal_var, ci_var = fast_partial_amort(p_var, var_rate, total_term_months, n_fix)
        else:
            bal_var, ci_var = 0.0, 0.0

        total_bal = bal_fix + bal_var
        total_ci = ci_fix + ci_var
        objective = total_ci + total_bal

        if objective < best_obj:
            best_obj = objective
            best_pct = pct_fixed

        results.append({
            "Fixed_Pct": pct_fixed,
            "Var_Pct": pct_var,
            "Balance_End": total_bal,
            "Cum_Interest": total_ci,
            "Objective": objective,
        })

    return best_pct, pd.DataFrame(results)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════

def init_state():
    ss = st.session_state
    defaults = {
        # ── Original Loan ──
        "orig_property_value": 800000.0,
        "orig_property_date": date(2020, 1, 15),
        "orig_loan_amount": 640000.0,
        "orig_balance": 580000.0,
        "orig_balance_date": date.today(),
        "orig_lvr": 80.0,
        "orig_rate": 6.50,
        "orig_rate_date": date(2020, 1, 15),
        "orig_term_months": 300,
        "orig_rate_changes": [],        # [[date, rate], ...]
        "orig_offset_initial": 0.0,
        "orig_offset_date": date.today(),
        "orig_offset_monthly": 0.0,
        "orig_offset_lumps": [],
        "orig_monthly_fee": 0.0,
        "orig_setup_fee": 0.0,
        "orig_breakage_fee": 0.0,
        "orig_other_fee": 0.0,

        # ── Current Loan ──
        "curr_is_continuation": True,
        "curr_rate": 6.50,
        "curr_rate_date": date.today(),
        "curr_loan_amount": 580000.0,
        "curr_offset_initial": 0.0,
        "curr_offset_date": date.today(),
        "curr_offset_monthly": 0.0,
        "curr_offset_lumps": [],
        "curr_rate_changes": [],
        "curr_monthly_fee": 10.0,
        "curr_setup_fee": 0.0,
        "curr_other_fee": 0.0,

        # ── Proposed ──
        "prop_loan_amount": 580000.0,
        "prop_var_rate": 6.20,
        "prop_fix_rate": 5.89,
        "prop_adv_rate": 6.20,
        "prop_comp_rate": 6.35,
        "prop_fix_period_years": 3,
        "prop_reversion_rate": 6.50,
        "prop_split_auto": True,
        "prop_split_fixed_pct": 50.0,
        "prop_var_rate_changes": [],
        "prop_offset_initial": 0.0,
        "prop_offset_date": date.today(),
        "prop_offset_monthly": 0.0,
        "prop_offset_lumps": [],
        "prop_monthly_fee": 10.0,
        "prop_setup_fee": 800.0,
        "prop_breakage_fee": 0.0,
        "prop_other_fee": 0.0,
        "prop_term_months": 300,

        # ── Strategy ──
        "strategy": "Balanced",
        "rba_rate_change": 0.0,
        "maintain_payment": True,

        # ── Results ──
        "results": None,
        "rba_rate": None,
    }
    for k, v in defaults.items():
        if k not in ss:
            ss[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_inputs() -> list:
    ss = st.session_state
    errors = []
    if ss.orig_loan_amount <= 0:
        errors.append("Original loan amount must be positive.")
    if ss.orig_balance <= 0:
        errors.append("Original remaining balance must be positive.")
    if ss.orig_balance > ss.orig_loan_amount:
        errors.append("Remaining balance cannot exceed original loan amount.")
    if ss.orig_term_months <= 0:
        errors.append("Original term must be positive.")
    if ss.orig_rate <= 0:
        errors.append("Original interest rate must be positive.")
    if ss.prop_loan_amount <= 0:
        errors.append("Proposed loan amount must be positive.")
    if ss.prop_var_rate <= 0:
        errors.append("Proposed variable rate must be positive.")
    if ss.prop_fix_rate <= 0:
        errors.append("Proposed fixed rate must be positive.")
    if ss.prop_fix_period_years < 1 or ss.prop_fix_period_years > 30:
        errors.append("Fixed period must be between 1 and 30 years.")
    return errors

# ══════════════════════════════════════════════════════════════════════════════
# CALCULATION RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_calculations():
    ss = st.session_state
    errors = validate_inputs()
    if errors:
        for e in errors:
            st.error(e)
        return

    maintain = ss.maintain_payment

    # ── Original Loan ──
    orig_rate_sched = build_rate_schedule(ss.orig_rate, ss.orig_rate_changes)
    orig_offset_sched = build_offset_schedule(
        ss.orig_balance_date, ss.orig_balance_date + timedelta(days=ss.orig_term_months * 32),
        ss.orig_offset_initial, ss.orig_offset_monthly, ss.orig_offset_lumps
    ) if ss.orig_offset_initial > 0 or ss.orig_offset_monthly > 0 else {}

    df_orig = amortize(
        ss.orig_balance, ss.orig_balance_date, ss.orig_term_months,
        orig_rate_sched, orig_offset_sched, ss.orig_monthly_fee, maintain
    )

    # ── Current Loan ──
    if ss.curr_is_continuation:
        curr_balance = ss.orig_balance
        curr_rate = ss.orig_rate
        curr_rate_changes = ss.orig_rate_changes
        curr_loan_amount = ss.orig_balance
        curr_monthly_fee = ss.curr_monthly_fee
        curr_start = ss.orig_balance_date
        curr_term = ss.orig_term_months
        curr_offset_initial = ss.curr_offset_initial
        curr_offset_monthly = ss.curr_offset_monthly
        curr_offset_lumps = ss.curr_offset_lumps
    else:
        curr_balance = ss.curr_loan_amount
        curr_rate = ss.curr_rate
        curr_rate_changes = ss.curr_rate_changes
        curr_loan_amount = ss.curr_loan_amount
        curr_monthly_fee = ss.curr_monthly_fee
        curr_start = ss.curr_rate_date
        curr_term = ss.orig_term_months
        curr_offset_initial = ss.curr_offset_initial
        curr_offset_monthly = ss.curr_offset_monthly
        curr_offset_lumps = ss.curr_offset_lumps

    curr_rate_sched = build_rate_schedule(curr_rate, curr_rate_changes)
    curr_offset_sched = build_offset_schedule(
        curr_start, curr_start + timedelta(days=curr_term * 32),
        curr_offset_initial, curr_offset_monthly, curr_offset_lumps
    ) if curr_offset_initial > 0 or curr_offset_monthly > 0 else {}

    df_curr = amortize(
        curr_balance, curr_start, curr_term,
        curr_rate_sched, curr_offset_sched, curr_monthly_fee, maintain
    )

    # ── Proposed: Optimal Split ──
    if ss.prop_split_auto:
        best_pct, split_df = calc_optimal_split(
            ss.prop_loan_amount, ss.prop_var_rate, ss.prop_fix_rate,
            ss.prop_reversion_rate, ss.prop_fix_period_years, ss.prop_term_months
        )
        ss.prop_split_fixed_pct = best_pct
    else:
        _, split_df = calc_optimal_split(
            ss.prop_loan_amount, ss.prop_var_rate, ss.prop_fix_rate,
            ss.prop_reversion_rate, ss.prop_fix_period_years, ss.prop_term_months
        )
        best_pct = ss.prop_split_fixed_pct

    fixed_pct = best_pct
    var_pct = 100.0 - fixed_pct

    p_fix = ss.prop_loan_amount * fixed_pct / 100
    p_var = ss.prop_loan_amount * var_pct / 100
    prop_start = date.today()

    prop_var_rate_sched = build_rate_schedule(ss.prop_var_rate, ss.prop_var_rate_changes)
    prop_offset_sched = build_offset_schedule(
        prop_start, prop_start + timedelta(days=ss.prop_term_months * 32),
        ss.prop_offset_initial, ss.prop_offset_monthly, ss.prop_offset_lumps
    ) if ss.prop_offset_initial > 0 or ss.prop_offset_monthly > 0 else {}

    df_prop_var = amortize(
        p_var, prop_start, ss.prop_term_months,
        prop_var_rate_sched, prop_offset_sched, ss.prop_monthly_fee / 2, maintain
    ) if p_var > 1 else pd.DataFrame()

    # Fixed rate + reversion
    fix_change_date = prop_start + timedelta(days=ss.prop_fix_period_years * 365)
    fix_rate_sched = [(date(1900, 1, 1), ss.prop_fix_rate),
                      (fix_change_date, ss.prop_reversion_rate)]

    df_prop_fix = amortize(
        p_fix, prop_start, ss.prop_term_months,
        fix_rate_sched, {}, ss.prop_monthly_fee / 2, maintain
    ) if p_fix > 1 else pd.DataFrame()

    # Combined split schedule (merged monthly)
    df_prop_split = _merge_split_schedules(df_prop_var, df_prop_fix)

    # ── Rates ──
    eff_var = effective_rate(ss.prop_loan_amount, ss.prop_setup_fee, ss.prop_monthly_fee,
                             ss.prop_var_rate, ss.prop_term_months)
    comp_var = comparison_rate(ss.prop_loan_amount, ss.prop_setup_fee, ss.prop_monthly_fee,
                                ss.prop_var_rate, ss.prop_term_months)

    # ── Scenario: RBA change ──
    rba_chg = ss.rba_rate_change
    if rba_chg != 0:
        scen_rate_sched = build_rate_schedule(
            ss.prop_var_rate + rba_chg, ss.prop_var_rate_changes)
        df_scen = amortize(
            ss.prop_loan_amount, prop_start, ss.prop_term_months,
            scen_rate_sched, prop_offset_sched, ss.prop_monthly_fee, maintain
        )
    else:
        df_scen = df_prop_split.copy() if not df_prop_split.empty else pd.DataFrame()

    ss.results = {
        "df_orig": df_orig,
        "df_curr": df_curr,
        "df_prop_var": df_prop_var,
        "df_prop_fix": df_prop_fix,
        "df_prop_split": df_prop_split,
        "df_scen": df_scen,
        "split_analysis": split_df,
        "optimal_fixed_pct": fixed_pct,
        "eff_var": eff_var,
        "comp_var": comp_var,
        "rba_chg": rba_chg,
    }

def _merge_split_schedules(df_var: pd.DataFrame, df_fix: pd.DataFrame) -> pd.DataFrame:
    """Merge variable and fixed schedules into a combined split view."""
    if df_var.empty and df_fix.empty:
        return pd.DataFrame()
    if df_var.empty:
        return df_fix.copy()
    if df_fix.empty:
        return df_var.copy()

    max_months = max(len(df_var), len(df_fix))
    rows = []
    for i in range(max_months):
        rv = df_var.iloc[i] if i < len(df_var) else None
        rf = df_fix.iloc[i] if i < len(df_fix) else None

        if rv is not None and rf is not None:
            rows.append({
                "Month": rv["Month"],
                "Date": rv["Date"],
                "Opening_Balance": rv["Opening_Balance"] + rf["Opening_Balance"],
                "Offset": rv["Offset"] + rf["Offset"],
                "Net_Debt": rv["Net_Debt"] + rf["Net_Debt"],
                "Rate": (rv["Rate"] * rv["Opening_Balance"] + rf["Rate"] * rf["Opening_Balance"])
                        / (rv["Opening_Balance"] + rf["Opening_Balance"]) if (rv["Opening_Balance"] + rf["Opening_Balance"]) > 0 else 0,
                "Interest": rv["Interest"] + rf["Interest"],
                "Interest_Saved": rv["Interest_Saved"] + rf["Interest_Saved"],
                "Principal": rv["Principal"] + rf["Principal"],
                "Fees": rv["Fees"] + rf["Fees"],
                "Payment": rv["Payment"] + rf["Payment"],
                "Closing_Balance": rv["Closing_Balance"] + rf["Closing_Balance"],
                "Cum_Interest": rv["Cum_Interest"] + rf["Cum_Interest"],
                "Cum_Paid": rv["Cum_Paid"] + rf["Cum_Paid"],
                "Cum_Interest_Saved": rv["Cum_Interest_Saved"] + rf["Cum_Interest_Saved"],
            })
        elif rv is not None:
            r = rv.to_dict(); r["Month"] = i + 1; rows.append(r)
        elif rf is not None:
            r = rf.to_dict(); r["Month"] = i + 1; rows.append(r)

    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════════════════════
# DYNAMIC LIST HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def render_rate_changes(key_prefix: str, label: str = "Rate Changes", max_rows: int = 15):
    ss = st.session_state
    list_key = f"{key_prefix}_rate_changes"
    if list_key not in ss:
        ss[list_key] = []

    st.markdown(f'<div class="section-header">{label}</div>', unsafe_allow_html=True)
    col_add, col_clear = st.columns([1, 1])
    with col_add:
        if st.button(f"＋ Add Rate Change", key=f"btn_add_{list_key}") and len(ss[list_key]) < max_rows:
            ss[list_key].append([date.today(), 6.0])
    with col_clear:
        if st.button(f"✕ Clear All", key=f"btn_clr_{list_key}") and ss[list_key]:
            ss[list_key] = []

    for i, row in enumerate(ss[list_key]):
        c1, c2, c3 = st.columns([2, 2, 0.5])
        with c1:
            new_d = st.date_input(f"Date##{list_key}_{i}", value=parse_date(row[0]), key=f"{list_key}_d_{i}")
        with c2:
            new_r = st.number_input(f"Rate % p.a.##{list_key}_{i}", value=float(row[1]), min_value=0.0, max_value=30.0, step=0.01, key=f"{list_key}_r_{i}")
        with c3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑", key=f"{list_key}_del_{i}"):
                ss[list_key].pop(i)
                st.rerun()
        ss[list_key][i] = [new_d, new_r]

def render_lump_sums(key_prefix: str, label: str = "Offset Lump Sums", max_rows: int = 100):
    ss = st.session_state
    list_key = f"{key_prefix}_offset_lumps"
    if list_key not in ss:
        ss[list_key] = []

    st.markdown(f'<div class="section-header">{label}</div>', unsafe_allow_html=True)
    col_add, col_clear = st.columns([1, 1])
    with col_add:
        if st.button(f"＋ Add Lump Sum", key=f"btn_add_{list_key}") and len(ss[list_key]) < max_rows:
            ss[list_key].append([date.today(), 0.0])
    with col_clear:
        if st.button(f"✕ Clear", key=f"btn_clr_{list_key}") and ss[list_key]:
            ss[list_key] = []

    for i, row in enumerate(ss[list_key]):
        c1, c2, c3 = st.columns([2, 2, 0.5])
        with c1:
            new_d = st.date_input(f"Date##{list_key}_{i}", value=parse_date(row[0]), key=f"{list_key}_d_{i}")
        with c2:
            new_a = st.number_input(f"Amount ($)##{list_key}_{i}", value=float(row[1]), step=1000.0, key=f"{list_key}_a_{i}")
        with c3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑", key=f"{list_key}_del_{i}"):
                ss[list_key].pop(i)
                st.rerun()
        ss[list_key][i] = [new_d, new_a]

# ══════════════════════════════════════════════════════════════════════════════
# INPUT TAB RENDERERS
# ══════════════════════════════════════════════════════════════════════════════

def render_original_tab():
    ss = st.session_state
    st.markdown('<div class="section-header">Property</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        ss.orig_property_value = st.number_input("Property Valuation ($)", value=ss.orig_property_value, min_value=0.0, step=10000.0, key="w_orig_property_value")
    with c2:
        ss.orig_property_date = st.date_input("Valuation Date", value=ss.orig_property_date, key="w_orig_property_date")

    st.markdown('<div class="section-header">Loan Details</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        ss.orig_loan_amount = st.number_input("Original Loan Amount ($)", value=ss.orig_loan_amount, min_value=0.0, step=10000.0, key="w_orig_loan_amount")
    with c2:
        ss.orig_balance = st.number_input("Remaining Balance ($)", value=ss.orig_balance, min_value=0.0, step=1000.0, key="w_orig_balance")
    with c3:
        ss.orig_balance_date = st.date_input("Balance As At", value=ss.orig_balance_date, key="w_orig_balance_date")

    c1, c2, c3 = st.columns(3)
    with c1:
        ss.orig_rate = st.number_input("Interest Rate (% p.a.)", value=ss.orig_rate, min_value=0.0, max_value=30.0, step=0.01, format="%.4f", key="w_orig_rate")
    with c2:
        ss.orig_term_months = st.number_input("Loan Term (months)", value=ss.orig_term_months, min_value=1, max_value=600, step=12, key="w_orig_term_months")
    with c3:
        if ss.orig_property_value > 0:
            lvr = ss.orig_balance / ss.orig_property_value * 100
            ss.orig_lvr = lvr
            st.metric("LVR", f"{lvr:.2f}%")
        else:
            ss.orig_lvr = st.number_input("LVR (%)", value=ss.orig_lvr, min_value=0.0, max_value=200.0, step=0.1, key="w_orig_lvr")

    render_rate_changes("orig", "Historical Rate Changes")

    st.markdown('<div class="section-header">Offset Account</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        ss.orig_offset_initial = st.number_input("Initial Offset Balance ($)", value=ss.orig_offset_initial, min_value=0.0, step=1000.0, key="w_orig_offset_initial")
    with c2:
        ss.orig_offset_date = st.date_input("Offset Start Date", value=ss.orig_offset_date, key="w_orig_offset_date")
    with c3:
        ss.orig_offset_monthly = st.number_input("Monthly Addition ($)", value=ss.orig_offset_monthly, min_value=0.0, step=100.0, key="w_orig_offset_monthly")

    render_lump_sums("orig", "Offset Lump Sums")

    st.markdown('<div class="section-header">Fees</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ss.orig_monthly_fee = st.number_input("Monthly Fee ($)", value=ss.orig_monthly_fee, min_value=0.0, step=1.0, key="w_orig_monthly_fee")
    with c2:
        ss.orig_setup_fee = st.number_input("Setup Fee ($)", value=ss.orig_setup_fee, min_value=0.0, step=100.0, key="w_orig_setup_fee")
    with c3:
        ss.orig_breakage_fee = st.number_input("Breakage Fee ($)", value=ss.orig_breakage_fee, min_value=0.0, step=100.0, key="w_orig_breakage_fee")
    with c4:
        ss.orig_other_fee = st.number_input("Other One-off Fee ($)", value=ss.orig_other_fee, min_value=0.0, step=100.0, key="w_orig_other_fee")

def render_current_tab():
    ss = st.session_state
    ss.curr_is_continuation = st.toggle(
        "Treat current loan as continuation of original loan",
        value=ss.curr_is_continuation,
        key="w_curr_is_continuation"
    )

    if not ss.curr_is_continuation:
        st.markdown('<div class="section-header">Current Loan Details</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            ss.curr_loan_amount = st.number_input("Current Loan Amount ($)", value=ss.curr_loan_amount, min_value=0.0, step=1000.0, key="w_curr_loan_amount")
        with c2:
            ss.curr_rate = st.number_input("Current Rate (% p.a.)", value=ss.curr_rate, min_value=0.0, max_value=30.0, step=0.01, format="%.4f", key="w_curr_rate")
        with c3:
            ss.curr_rate_date = st.date_input("Rate Effective Date", value=ss.curr_rate_date, key="w_curr_rate_date")

        render_rate_changes("curr", "Rate Changes")
    else:
        st.info("Current loan uses the same parameters as the original loan. You can add an offset account below.")

    st.markdown('<div class="section-header">Offset Account</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        ss.curr_offset_initial = st.number_input("Initial Offset Balance ($)", value=ss.curr_offset_initial, min_value=0.0, step=1000.0, key="w_curr_offset_initial")
    with c2:
        ss.curr_offset_date = st.date_input("Offset Start Date", value=ss.curr_offset_date, key="w_curr_offset_date")
    with c3:
        ss.curr_offset_monthly = st.number_input("Monthly Addition ($)", value=ss.curr_offset_monthly, min_value=0.0, step=100.0, key="w_curr_offset_monthly")

    render_lump_sums("curr", "Offset Lump Sums")

    st.markdown('<div class="section-header">Fees</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        ss.curr_monthly_fee = st.number_input("Monthly Fee ($)", value=ss.curr_monthly_fee, min_value=0.0, step=1.0, key="w_curr_monthly_fee")
    with c2:
        ss.curr_setup_fee = st.number_input("Setup Fee ($)", value=ss.curr_setup_fee, min_value=0.0, step=100.0, key="w_curr_setup_fee")
    with c3:
        ss.curr_other_fee = st.number_input("Other Fee ($)", value=ss.curr_other_fee, min_value=0.0, step=100.0, key="w_curr_other_fee")

def render_proposed_tab():
    ss = st.session_state
    st.markdown('<div class="section-header">Loan Amount & Term</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        ss.prop_loan_amount = st.number_input("Proposed Loan Amount ($)", value=ss.prop_loan_amount, min_value=0.0, step=1000.0, key="w_prop_loan_amount")
    with c2:
        ss.prop_term_months = st.number_input("Loan Term (months)", value=ss.prop_term_months, min_value=1, max_value=600, step=12, key="w_prop_term_months")

    st.markdown('<div class="section-header">Interest Rates</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        ss.prop_adv_rate = st.number_input("Advertised Rate (% p.a.)", value=ss.prop_adv_rate, min_value=0.0, max_value=30.0, step=0.01, format="%.2f", key="w_prop_adv_rate")
    with c2:
        ss.prop_comp_rate = st.number_input("Comparison Rate (% p.a.)", value=ss.prop_comp_rate, min_value=0.0, max_value=30.0, step=0.01, format="%.2f", key="w_prop_comp_rate")
    with c3:
        st.markdown('<div class="info-box">Effective rate is calculated automatically based on fees and loan details.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        ss.prop_var_rate = st.number_input("Variable Rate (% p.a.)", value=ss.prop_var_rate, min_value=0.0, max_value=30.0, step=0.01, format="%.4f", key="w_prop_var_rate")
    with c2:
        ss.prop_fix_rate = st.number_input("Fixed Rate (% p.a.)", value=ss.prop_fix_rate, min_value=0.0, max_value=30.0, step=0.01, format="%.4f", key="w_prop_fix_rate")

    c1, c2 = st.columns(2)
    with c1:
        ss.prop_fix_period_years = st.number_input("Fixed Period (years)", value=ss.prop_fix_period_years, min_value=1, max_value=30, step=1, key="w_prop_fix_period_years")
    with c2:
        ss.prop_reversion_rate = st.number_input("Reversion Rate after Fixed (% p.a.)", value=ss.prop_reversion_rate, min_value=0.0, max_value=30.0, step=0.01, format="%.4f", key="w_prop_reversion_rate")

    st.markdown('<div class="section-header">Variable/Fixed Split</div>', unsafe_allow_html=True)
    ss.prop_split_auto = st.toggle("Auto-calculate optimal split ratio", value=ss.prop_split_auto, key="w_prop_split_auto")
    if not ss.prop_split_auto:
        ss.prop_split_fixed_pct = st.slider(
            "Fixed Component (%)", min_value=0.0, max_value=100.0,
            value=ss.prop_split_fixed_pct, step=0.5, key="w_prop_split_fixed_pct"
        )
        st.caption(f"Variable: {100 - ss.prop_split_fixed_pct:.1f}% | Fixed: {ss.prop_split_fixed_pct:.1f}%")
    else:
        st.info("Optimal split will be calculated automatically to minimise total interest + closing balance after the fixed period.")

    render_rate_changes("prop", "Proposed Variable Rate Changes")

    st.markdown('<div class="section-header">Offset Account</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        ss.prop_offset_initial = st.number_input("Initial Offset Balance ($)", value=ss.prop_offset_initial, min_value=0.0, step=1000.0, key="w_prop_offset_initial")
    with c2:
        ss.prop_offset_date = st.date_input("Offset Start Date", value=ss.prop_offset_date, key="w_prop_offset_date")
    with c3:
        ss.prop_offset_monthly = st.number_input("Monthly Addition ($)", value=ss.prop_offset_monthly, min_value=0.0, step=100.0, key="w_prop_offset_monthly")

    render_lump_sums("prop", "Offset Lump Sums")

    st.markdown('<div class="section-header">Fees</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ss.prop_monthly_fee = st.number_input("Monthly Fee ($)", value=ss.prop_monthly_fee, min_value=0.0, step=1.0, key="w_prop_monthly_fee")
    with c2:
        ss.prop_setup_fee = st.number_input("Setup / Establishment Fee ($)", value=ss.prop_setup_fee, min_value=0.0, step=100.0, key="w_prop_setup_fee")
    with c3:
        ss.prop_breakage_fee = st.number_input("Breakage Fee ($)", value=ss.prop_breakage_fee, min_value=0.0, step=100.0, key="w_prop_breakage_fee")
    with c4:
        ss.prop_other_fee = st.number_input("Other One-off Fee ($)", value=ss.prop_other_fee, min_value=0.0, step=100.0, key="w_prop_other_fee")

def render_strategy_tab():
    ss = st.session_state
    st.markdown('<div class="section-header">Strategy</div>', unsafe_allow_html=True)
    ss.strategy = st.radio(
        "Refinancing Strategy",
        ["Conservative (80% fixed)", "Balanced (optimal split)", "Aggressive (0% fixed / max variable)"],
        index=["Conservative (80% fixed)", "Balanced (optimal split)", "Aggressive (0% fixed / max variable)"].index(
            ss.strategy if ss.strategy in ["Conservative (80% fixed)", "Balanced (optimal split)", "Aggressive (0% fixed / max variable)"]
            else "Balanced (optimal split)"
        ),
        key="w_strategy"
    )
    ss.strategy = ss.strategy

    st.markdown("""
    <div class="info-box">
    <b>Conservative:</b> 80% fixed – maximises payment certainty, protects against rate rises.<br>
    <b>Balanced:</b> Optimal split – minimises total interest + closing balance at end of fixed period.<br>
    <b>Aggressive:</b> 100% variable with maximum offset – maximises flexibility, benefits most from rate falls.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Payment Mode</div>', unsafe_allow_html=True)
    ss.maintain_payment = st.toggle(
        "Maintain original payment (term changes when rate changes)",
        value=ss.maintain_payment, key="w_maintain_payment"
    )

    st.markdown('<div class="section-header">RBA Rate Scenarios</div>', unsafe_allow_html=True)

    col_rba, col_fetch = st.columns([3, 1])
    with col_rba:
        ss.rba_rate_change = st.slider(
            "RBA Cash Rate Change (basis points)",
            min_value=-300, max_value=300, value=int(ss.rba_rate_change * 100), step=25,
            key="w_rba_rate_change"
        ) / 100
    with col_fetch:
        if st.button("📡 Fetch RBA Rate"):
            rate = fetch_rba_rate()
            if rate:
                ss.rba_rate = rate
                st.success(f"RBA Rate: {rate:.2f}%")
            else:
                st.warning("Could not fetch RBA rate. Check connection.")

    if ss.rba_rate:
        st.caption(f"Current RBA Cash Rate: {ss.rba_rate:.2f}%")
    if ss.rba_rate_change != 0:
        direction = "increase" if ss.rba_rate_change > 0 else "decrease"
        st.info(f"Scenario: {abs(ss.rba_rate_change):.2f}% ({abs(ss.rba_rate_change * 100):.0f} bps) {direction} in variable rate")

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD RENDERERS
# ══════════════════════════════════════════════════════════════════════════════

def _summary_metric(label, value, delta=None, delta_prefix="saves"):
    delta_html = ""
    if delta is not None:
        cls = "metric-delta-pos" if delta <= 0 else "metric-delta-neg"
        sign = "-" if delta <= 0 else "+"
        delta_html = f'<div class="{cls}">{sign}{fmt_currency(abs(delta))} {delta_prefix}</div>'
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>"""

def render_overview(results):
    ss = st.session_state
    df_orig = results["df_orig"]
    df_curr = results["df_curr"]
    df_split = results["df_prop_split"]

    st.markdown("### Overview")

    labels = ["Original Loan", "Current Loan", "Proposed (Split)"]
    dfs = [df_orig, df_curr, df_split]
    colors = [C_ORIG, C_CURR, C_SPLIT]

    cols = st.columns(len(labels))
    for col, label, df, color in zip(cols, labels, dfs, colors):
        with col:
            if df.empty:
                st.warning(f"No data for {label}")
                continue
            total_interest = df["Cum_Interest"].iloc[-1]
            total_paid = df["Cum_Paid"].iloc[-1]
            term = len(df)
            pmt = df["Payment"].iloc[0]
            st.markdown(f'<div style="color:{color}; font-weight:700; font-size:1rem; margin-bottom:8px">{label}</div>', unsafe_allow_html=True)
            st.markdown(_summary_metric("Monthly Payment", fmt_currency(pmt)), unsafe_allow_html=True)
            st.markdown(_summary_metric("Total Interest", fmt_currency(total_interest)), unsafe_allow_html=True)
            st.markdown(_summary_metric("Total Paid", fmt_currency(total_paid)), unsafe_allow_html=True)
            st.markdown(_summary_metric("Term", f"{term} months ({term/12:.1f} yrs)"), unsafe_allow_html=True)

    if not df_curr.empty and not df_split.empty:
        st.markdown("---")
        st.markdown("### Proposed vs Current — Savings")
        int_curr = df_curr["Cum_Interest"].iloc[-1]
        int_prop = df_split["Cum_Interest"].iloc[-1]
        savings = int_curr - int_prop
        total_curr = df_curr["Cum_Paid"].iloc[-1]
        total_prop = df_split["Cum_Paid"].iloc[-1]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(_summary_metric("Interest Saved", fmt_currency(savings)), unsafe_allow_html=True)
        with c2:
            st.markdown(_summary_metric("Total Cost Saved", fmt_currency(total_curr - total_prop)), unsafe_allow_html=True)
        with c3:
            st.markdown(_summary_metric("Optimal Fixed Split", f"{results['optimal_fixed_pct']:.1f}%"), unsafe_allow_html=True)
        with c4:
            eff_r = results.get("eff_var", ss.prop_var_rate)
            comp_r = results.get("comp_var", ss.prop_comp_rate)
            st.markdown(_summary_metric("Effective Rate", fmt_pct(eff_r)), unsafe_allow_html=True)

def render_monthly_payments(results):
    df_orig = results["df_orig"]
    df_curr = results["df_curr"]
    df_split = results["df_prop_split"]

    fig = go.Figure()
    for df, name, color in [(df_orig, "Original", C_ORIG), (df_curr, "Current", C_CURR), (df_split, "Proposed Split", C_SPLIT)]:
        if not df.empty:
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Payment"], name=name, line=dict(color=color, width=2)))
    fig.update_layout(**PLOTLY_LAYOUT, title="Monthly Payments Over Time", yaxis_title="Payment ($)")
    st.plotly_chart(fig, use_container_width=True)

    # Interest vs Principal stacked bar (sampled)
    if not df_split.empty:
        sample = df_split.iloc[::12]  # annual
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=sample["Date"], y=sample["Principal"], name="Principal", marker_color=C_VAR))
        fig2.add_trace(go.Bar(x=sample["Date"], y=sample["Interest"], name="Interest", marker_color=C_FIX))
        fig2.update_layout(**PLOTLY_LAYOUT, barmode="stack", title="Annual Principal vs Interest (Proposed Split)",
                           yaxis_title="Amount ($)")
        st.plotly_chart(fig2, use_container_width=True)

def render_loan_balance(results):
    df_orig = results["df_orig"]
    df_curr = results["df_curr"]
    df_split = results["df_prop_split"]
    ss = st.session_state

    fig = go.Figure()
    for df, name, color in [(df_orig, "Original", C_ORIG), (df_curr, "Current", C_CURR), (df_split, "Proposed Split", C_SPLIT)]:
        if not df.empty:
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Closing_Balance"], name=name, line=dict(color=color, width=2)))
    fig.update_layout(**PLOTLY_LAYOUT, title="Outstanding Loan Balance Over Time", yaxis_title="Balance ($)")
    st.plotly_chart(fig, use_container_width=True)

    # LVR chart
    if ss.orig_property_value > 0 and not df_split.empty:
        lvr_vals = df_split["Closing_Balance"] / ss.orig_property_value * 100
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_split["Date"], y=lvr_vals, name="LVR", line=dict(color=C_SPLIT, width=2), fill="tozeroy", fillcolor="rgba(196,122,245,0.1)"))
        fig2.add_hline(y=80, line=dict(color=C_FIX, dash="dash"), annotation_text="80% LVR (LMI threshold)")
        fig2.update_layout(**PLOTLY_LAYOUT, title="LVR Over Time", yaxis_title="LVR (%)")
        st.plotly_chart(fig2, use_container_width=True)

def render_interest_analysis(results):
    df_orig = results["df_orig"]
    df_curr = results["df_curr"]
    df_split = results["df_prop_split"]

    fig = go.Figure()
    for df, name, color in [(df_orig, "Original", C_ORIG), (df_curr, "Current", C_CURR), (df_split, "Proposed Split", C_SPLIT)]:
        if not df.empty:
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Cum_Interest"], name=name, line=dict(color=color, width=2)))
    fig.update_layout(**PLOTLY_LAYOUT, title="Cumulative Interest Paid", yaxis_title="Cumulative Interest ($)")
    st.plotly_chart(fig, use_container_width=True)

    if not df_split.empty and df_split["Cum_Interest_Saved"].iloc[-1] > 0:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_split["Date"], y=df_split["Cum_Interest_Saved"],
                                  name="Interest Saved", line=dict(color=C_VAR, width=2), fill="tozeroy",
                                  fillcolor="rgba(48,217,150,0.1)"))
        fig2.update_layout(**PLOTLY_LAYOUT, title="Cumulative Interest Saved via Offset Account", yaxis_title="Savings ($)")
        st.plotly_chart(fig2, use_container_width=True)

def render_optimal_split(results):
    split_df = results["split_analysis"]
    best_pct = results["optimal_fixed_pct"]
    ss = st.session_state

    if split_df.empty:
        st.warning("No split analysis available.")
        return

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Objective (Interest + Balance)", "Components"))

    fig.add_trace(go.Scatter(x=split_df["Fixed_Pct"], y=split_df["Objective"],
                             name="Objective", line=dict(color=C_SPLIT, width=2)), row=1, col=1)
    best_row = split_df[split_df["Fixed_Pct"] == best_pct].iloc[0]
    fig.add_trace(go.Scatter(x=[best_pct], y=[best_row["Objective"]],
                             mode="markers", marker=dict(symbol="diamond", size=14, color=C_FIX),
                             name=f"Optimal ({best_pct:.1f}%)"), row=1, col=1)

    fig.add_trace(go.Scatter(x=split_df["Fixed_Pct"], y=split_df["Cum_Interest"],
                             name="Cum Interest", line=dict(color=C_FIX, width=1.5)), row=1, col=2)
    fig.add_trace(go.Scatter(x=split_df["Fixed_Pct"], y=split_df["Balance_End"],
                             name="End Balance", line=dict(color=C_VAR, width=1.5)), row=1, col=2)

    fig.update_layout(**PLOTLY_LAYOUT, title=f"Optimal Split Analysis — Best Fixed: {best_pct:.1f}%")
    fig.update_xaxes(title_text="Fixed %", gridcolor=C_GRID)
    fig.update_yaxes(title_text="$", gridcolor=C_GRID)
    st.plotly_chart(fig, use_container_width=True)

    # Summary table (every 5%)
    tbl = split_df[split_df["Fixed_Pct"] % 5 == 0].copy()
    tbl["Fixed_Pct"] = tbl["Fixed_Pct"].apply(lambda x: f"{x:.0f}%")
    tbl["Var_Pct"] = tbl["Var_Pct"].apply(lambda x: f"{x:.0f}%")
    tbl["Balance_End"] = tbl["Balance_End"].apply(fmt_currency)
    tbl["Cum_Interest"] = tbl["Cum_Interest"].apply(fmt_currency)
    tbl["Objective"] = tbl["Objective"].apply(fmt_currency)
    tbl.columns = ["Fixed %", "Variable %", "End Balance", "Cum Interest", "Objective"]
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    st.success(f"✅ Optimal fixed component: **{best_pct:.1f}%** (${ss.prop_loan_amount * best_pct / 100:,.0f}) | Variable: **{100 - best_pct:.1f}%** (${ss.prop_loan_amount * (100 - best_pct) / 100:,.0f})")

def render_strategy_tab_dashboard(results):
    ss = st.session_state
    df_split = results["df_prop_split"]
    if df_split.empty:
        st.warning("Run calculations first.")
        return

    loan = ss.prop_loan_amount
    term = ss.prop_term_months
    var_r = ss.prop_var_rate
    fix_r = ss.prop_fix_rate
    rev_r = ss.prop_reversion_rate
    fix_yr = ss.prop_fix_period_years
    fix_mo = fix_yr * 12

    strategies = {
        "Conservative\n(80% Fixed)": {"fix_pct": 80, "color": C_FIX},
        "Balanced\n(Optimal Split)": {"fix_pct": results["optimal_fixed_pct"], "color": C_SPLIT},
        "Aggressive\n(100% Variable)": {"fix_pct": 0, "color": C_VAR},
    }

    cols = st.columns(len(strategies))
    totals = {}

    for (name, cfg), col in zip(strategies.items(), cols):
        pct_f = cfg["fix_pct"]
        pct_v = 100 - pct_f
        p_f = loan * pct_f / 100
        p_v = loan * pct_v / 100

        _, ci_f = fast_partial_amort(p_f, fix_r, term, fix_mo) if p_f > 0 else (0, 0)
        _, ci_v = fast_partial_amort(p_v, var_r, term, fix_mo) if p_v > 0 else (0, 0)
        # After fixed period: both revert or continue at variable
        bal_f, _ = fast_partial_amort(p_f, fix_r, term, fix_mo) if p_f > 0 else (0, 0)
        bal_v, _ = fast_partial_amort(p_v, var_r, term, fix_mo) if p_v > 0 else (0, 0)
        remaining_mo = term - fix_mo
        if remaining_mo > 0:
            _, ci_f2 = fast_partial_amort(bal_f, rev_r, remaining_mo, remaining_mo) if bal_f > 0 else (0, 0)
            _, ci_v2 = fast_partial_amort(bal_v, var_r, remaining_mo, remaining_mo) if bal_v > 0 else (0, 0)
        else:
            ci_f2, ci_v2 = 0, 0

        total_ci = ci_f + ci_v + ci_f2 + ci_v2
        totals[name.replace("\n", " ")] = total_ci

        with col:
            st.markdown(f'<div style="color:{cfg["color"]}; font-weight:700; text-align:center; font-size:0.9rem; margin-bottom:8px">{name.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
            st.markdown(_summary_metric("Fixed Component", f"{pct_f:.1f}%"), unsafe_allow_html=True)
            st.markdown(_summary_metric("Variable Component", f"{pct_v:.1f}%"), unsafe_allow_html=True)
            st.markdown(_summary_metric("Est. Total Interest", fmt_currency(total_ci)), unsafe_allow_html=True)

    # Bar chart
    fig = go.Figure(go.Bar(
        x=list(totals.keys()),
        y=list(totals.values()),
        marker_color=[C_FIX, C_SPLIT, C_VAR],
        text=[fmt_currency(v) for v in totals.values()],
        textposition="outside",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title="Strategy Comparison — Estimated Total Interest",
                      yaxis_title="Total Interest ($)")
    st.plotly_chart(fig, use_container_width=True)

def render_scenarios(results):
    ss = st.session_state
    df_base = results["df_prop_split"]
    df_scen = results["df_scen"]
    rba_chg = results["rba_chg"]

    if df_base.empty:
        st.warning("No base scenario available.")
        return

    st.markdown(f"### RBA Rate Scenario: {'+' if rba_chg >= 0 else ''}{rba_chg:.2f}% ({'+' if rba_chg*100 >= 0 else ''}{rba_chg*100:.0f} bps)")

    c1, c2, c3 = st.columns(3)
    with c1:
        base_pmt = df_base["Payment"].iloc[0]
        scen_pmt = df_scen["Payment"].iloc[0] if not df_scen.empty else base_pmt
        delta_pmt = scen_pmt - base_pmt
        sign = "+" if delta_pmt >= 0 else ""
        st.markdown(_summary_metric("Monthly Payment Change",
                                    f"{sign}{fmt_currency(delta_pmt)}",
                                    ), unsafe_allow_html=True)
    with c2:
        base_int = df_base["Cum_Interest"].iloc[-1]
        scen_int = df_scen["Cum_Interest"].iloc[-1] if not df_scen.empty else base_int
        delta_int = scen_int - base_int
        sign = "+" if delta_int >= 0 else ""
        st.markdown(_summary_metric("Total Interest Change",
                                    f"{sign}{fmt_currency(delta_int)}"), unsafe_allow_html=True)
    with c3:
        base_term = len(df_base)
        scen_term = len(df_scen) if not df_scen.empty else base_term
        delta_term = scen_term - base_term
        sign = "+" if delta_term >= 0 else ""
        st.markdown(_summary_metric("Term Change", f"{sign}{delta_term} months"), unsafe_allow_html=True)

    if not df_scen.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_base["Date"], y=df_base["Closing_Balance"],
                                 name="Base (no change)", line=dict(color=C_SPLIT, width=2)))
        fig.add_trace(go.Scatter(x=df_scen["Date"], y=df_scen["Closing_Balance"],
                                 name=f"Scenario ({'+' if rba_chg >= 0 else ''}{rba_chg:.2f}%)",
                                 line=dict(color=C_FIX if rba_chg > 0 else C_VAR, width=2, dash="dash")))
        fig.update_layout(**PLOTLY_LAYOUT, title="Balance: Base vs Rate Scenario", yaxis_title="Balance ($)")
        st.plotly_chart(fig, use_container_width=True)

def render_schedules(results):
    dfs = {
        "Original Loan": results["df_orig"],
        "Current Loan": results["df_curr"],
        "Proposed Variable": results["df_prop_var"],
        "Proposed Fixed": results["df_prop_fix"],
        "Proposed Split (Combined)": results["df_prop_split"],
    }
    available = {k: v for k, v in dfs.items() if not v.empty}
    if not available:
        st.warning("No schedules available.")
        return

    sel = st.selectbox("Select Schedule", list(available.keys()))
    df = available[sel].copy()

    # Rename for display
    df = df.rename(columns={
        "Month": "Month",
        "Date": "Date",
        "Opening_Balance": "Opening Balance",
        "Offset": "Avg Offset",
        "Net_Debt": "Net Debt",
        "Rate": "Rate %",
        "Interest": "Interest",
        "Interest_Saved": "Interest Saved",
        "Principal": "Principal",
        "Fees": "Fees",
        "Payment": "Payment",
        "Closing_Balance": "Closing Balance",
        "Cum_Interest": "Cum Interest",
        "Cum_Paid": "Cum Paid",
        "Cum_Interest_Saved": "Cum Interest Saved",
    })

    currency_cols = ["Opening Balance", "Avg Offset", "Net Debt", "Interest", "Interest Saved",
                     "Principal", "Fees", "Payment", "Closing Balance", "Cum Interest",
                     "Cum Paid", "Cum Interest Saved"]

    display_df = df.copy()
    for c in currency_cols:
        if c in display_df.columns:
            display_df[c] = display_df[c].apply(lambda x: fmt_currency(x))
    if "Rate %" in display_df.columns:
        display_df["Rate %"] = display_df["Rate %"].apply(lambda x: fmt_pct(x))

    st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)

    # CSV download
    csv_buf = io.BytesIO()
    df.to_csv(csv_buf, index=False)
    csv_buf.seek(0)
    st.download_button(
        label=f"⬇ Download {sel} Schedule (CSV)",
        data=csv_buf.getvalue(),
        file_name=f"schedule_{sel.lower().replace(' ', '_')}.csv",
        mime="text/csv"
    )

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    init_state()
    ss = st.session_state

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0d1526 0%, #080c14 100%);
                border-bottom: 1px solid #1a2744; padding: 20px 0; margin-bottom: 20px;">
        <h1 style="color: #4a9af5; margin: 0; font-size: 1.8rem; font-weight: 700;">
            🏠 Australian Mortgage Refinance Analyser
        </h1>
        <p style="color: #8892b0; margin: 4px 0 0; font-size: 0.9rem;">
            Multi-variable amortisation engine · Optimal split ratio · ASIC comparison rates · RBA scenarios
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Input Tabs ─────────────────────────────────────────────────────────────
    in_tab1, in_tab2, in_tab3, in_tab4 = st.tabs([
        "📋 Original Loan", "🏦 Current Loan", "💡 Proposed Comparison", "⚙️ Strategy & Scenarios"
    ])

    with in_tab1:
        render_original_tab()
    with in_tab2:
        render_current_tab()
    with in_tab3:
        render_proposed_tab()
    with in_tab4:
        render_strategy_tab()

    # ── Action Buttons ─────────────────────────────────────────────────────────
    st.markdown("")
    c_calc, c_reset, _ = st.columns([2, 1, 3])
    with c_calc:
        if st.button("🔍 Calculate & Analyse", key="btn_calculate"):
            with st.spinner("Running amortisation calculations…"):
                run_calculations()

    with c_reset:
        st.markdown('<div class="reset-btn">', unsafe_allow_html=True)
        if st.button("↺ Reset All", key="btn_reset"):
            keys_to_del = [k for k in ss.keys() if not k.startswith("_") and k != "FormSubmitter"]
            for k in keys_to_del:
                del ss[k]
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Dashboard ──────────────────────────────────────────────────────────────
    if ss.results is not None:
        st.markdown("---")
        st.markdown("## 📊 Analysis Dashboard")

        dash_tabs = st.tabs([
            "📈 Overview",
            "💵 Monthly Payments",
            "📉 Loan Balance",
            "💰 Interest Analysis",
            "🎯 Optimal Split",
            "🏆 Strategy",
            "🌐 Scenarios",
            "📃 Schedules",
        ])

        with dash_tabs[0]:
            render_overview(ss.results)
        with dash_tabs[1]:
            render_monthly_payments(ss.results)
        with dash_tabs[2]:
            render_loan_balance(ss.results)
        with dash_tabs[3]:
            render_interest_analysis(ss.results)
        with dash_tabs[4]:
            render_optimal_split(ss.results)
        with dash_tabs[5]:
            render_strategy_tab_dashboard(ss.results)
        with dash_tabs[6]:
            render_scenarios(ss.results)
        with dash_tabs[7]:
            render_schedules(ss.results)

    else:
        st.markdown("""
        <div style="text-align:center; padding: 60px 20px; color: #8892b0;">
            <div style="font-size: 3rem; margin-bottom: 16px;">🏠</div>
            <h3 style="color: #4a9af5;">Enter your loan details and click <em>Calculate & Analyse</em></h3>
            <p>The dashboard will show amortisation schedules, optimal split ratio, interest savings,<br>
            strategy comparison, and RBA rate scenarios.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
