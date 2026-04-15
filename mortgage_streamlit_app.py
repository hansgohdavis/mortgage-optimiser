import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize_scalar, newton
import plotly.graph_objects as go

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Loan Refinance Optimiser",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== THEME (MODERN DARK UI) ======================
st.markdown("""
<style>
html, body {
    background-color: #0b0f17;
    color: #e8eefc;
    font-family: 'Inter', 'Space Grotesk', sans-serif;
}

.main-header {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(90deg,#7dd3fc,#a78bfa,#34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}

.stMetric {
    background: rgba(255,255,255,0.04);
    border-radius: 18px;
    padding: 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}

section[data-testid="stSidebar"] {
    background-color: #0a0d14;
}

.block-container {
    padding-top: 2rem;
}

/* subtle hover animation */
div:hover {
    transition: all 0.2s ease-in-out;
}

</style>
""", unsafe_allow_html=True)

# ====================== CORE FUNCTIONS ======================
@st.cache_data(ttl=3600)
def logistic(x, L, k, x0, b):
    return L / (1 + np.exp(-k * (x - x0))) + b

@st.cache_data(ttl=3600)
def calculate_monthly_payment(principal, annual_rate, term_months):
    if principal <= 0 or term_months <= 0:
        return 0.0
    r = annual_rate / 12 / 100
    if r == 0:
        return principal / term_months
    return principal * r * (1 + r) ** term_months / ((1 + r) ** term_months - 1)

# ===== inverse: solve term given payment =====
def solve_term(principal, annual_rate, payment):
    if principal <= 0 or payment <= 0:
        return 0
    r = annual_rate / 12 / 100
    if r == 0:
        return principal / payment
    return np.log(payment / (payment - principal * r)) / np.log(1 + r)

# ====================== SIMULATION ======================
@st.cache_data(ttl=3600)
def simulate_split_loan(var_p, fix_p, var_rate, fix_rate, revert_rate,
                        fixed_years, term_months,
                        offset_start_month=0,
                        monthly_offset_add=0,
                        monthly_fees=0,
                        one_time_fees=0,
                        keep_payment_fixed=True):
    fixed_months = int(fixed_years * 12)

    var_payment = calculate_monthly_payment(var_p, var_rate, term_months)
    fix_payment = calculate_monthly_payment(fix_p, fix_rate, term_months)

    var_bal, fix_bal = var_p, fix_p
    offset = 0

    schedule = []
    total_interest = 0
    total_paid = one_time_fees

    for m in range(1, term_months + 1):

        # activate offset
        if m >= offset_start_month:
            offset += monthly_offset_add

        # VARIABLE
        r_v = var_rate / 12 / 100
        interest_v = max(0, var_bal - offset) * r_v
        pay_v = var_payment - interest_v
        pay_v = min(pay_v, var_bal)
        var_bal -= pay_v

        # FIXED
        if fix_p > 0 and m == fixed_months + 1:
            remaining = max(1, term_months - m)
            fix_payment = calculate_monthly_payment(fix_bal, revert_rate, remaining)

        r_f = fix_rate / 12 / 100 if m <= fixed_months else revert_rate / 12 / 100
        interest_f = fix_bal * r_f
        pay_f = fix_payment - interest_f
        pay_f = min(pay_f, fix_bal)
        fix_bal -= pay_f

        total_interest += interest_v + interest_f
        total_paid += var_payment + fix_payment + monthly_fees

        schedule.append([
            m, interest_v + interest_f, pay_v + pay_f, var_bal + fix_bal, offset
        ])

        if var_bal + fix_bal <= 0.01:
            break

    df = pd.DataFrame(schedule, columns=["month","interest","principal","balance","offset"])
    return df, total_interest, total_paid

# ====================== OPTIMISATION ======================
@st.cache_data(ttl=3600)
def find_optimal_split(loan, var_r, fix_r, revert_r, fy, tm,
                       offset_start, offset_add):

    splits = np.arange(0, 101)
    interest = []
    debt = []

    for s in splits:
        vp = loan * (1 - s/100)
        fp = loan * s/100

        df, _, _ = simulate_split_loan(vp, fp, var_r, fix_r, revert_r,
                                       fy, tm, offset_start, offset_add)

        interest.append(df["interest"].sum())
        debt.append(df.iloc[-1]["balance"])

    interest = np.array(interest)
    debt = np.array(debt)

    def cost(s):
        i = np.interp(s, splits, interest)
        d = np.interp(s, splits, debt)
        return i + 0.5 * d

    res = minimize_scalar(cost, bounds=(0,100), method="bounded")
    opt = int(res.x)

    return opt, splits, interest, debt

# ====================== UI ======================
st.markdown('<div class="main-header">Loan Refinance Optimiser</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Original Loan")
    orig_val = st.number_input("House valuation", 850000.0)
    orig_loan = st.number_input("Loan amount", 620000.0)
    orig_term = st.number_input("Loan term (months)", 300)
    orig_rate = st.number_input("Interest rate (%)", 6.35)

    orig_lvr = orig_loan / orig_val * 100
    st.metric("Original LVR", f"{orig_lvr:.2f}%")

    st.divider()

    st.header("Refinance Inputs")
    loan_left = st.number_input("Remaining loan", orig_loan)
    var_rate = st.number_input("New variable rate", 5.5)
    fix_rate = st.number_input("New fixed rate", 4.9)
    revert_rate = st.number_input("Revert rate", 6.3)
    fixed_years = st.number_input("Fixed years", 2)

    st.divider()

    st.header("Strategy Controls")

    mode = st.radio(
        "Loan behaviour",
        ["Keep term fixed", "Keep monthly payment fixed"]
    )

    offset_start_orig = st.number_input("Offset start (original months)", 0)
    offset_start_new = st.number_input("Offset start (post-refinance months)", 0)
    monthly_offset_add = st.number_input("Monthly offset add", 1000)

    monthly_fees = st.number_input("Monthly fees", 0)
    one_time_fees = st.number_input("Upfront fees", 300)

# ====================== MAIN ======================
if loan_left > 0:

    term = orig_term

    df, interest, paid = simulate_split_loan(
        loan_left, 0, var_rate, 0, 0,
        fixed_years, term,
        offset_start_new, monthly_offset_add,
        monthly_fees, one_time_fees
    )

    opt, s, i, d = find_optimal_split(
        loan_left, var_rate, fix_rate, revert_rate,
        fixed_years, term,
        offset_start_new, monthly_offset_add
    )

    st.subheader("Results")

    c1, c2, c3 = st.columns(3)
    c1.metric("Optimal split % fixed", f"{opt}%")
    c2.metric("Total interest", f"${interest:,.0f}")
    c3.metric("Total paid", f"${paid:,.0f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["month"], y=df["balance"], name="Balance"))
    fig.add_trace(go.Scatter(x=df["month"], y=df["interest"], name="Interest"))
    fig.update_layout(
        template="plotly_dark",
        title="Loan Dynamics (Modern View)",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Optimisation Surface")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=s, y=i, name="Interest"))
    fig2.add_trace(go.Scatter(x=s, y=d, name="Debt"))
    fig2.update_layout(template="plotly_dark", height=400)

    st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("Enter loan details")
