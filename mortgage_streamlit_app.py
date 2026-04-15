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

# ====================== ULTRA-MODERN FINTECH UI ======================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@200;300;400;500&display=swap');

html, body, [class*='css'] {
    background-color: #070A0F;
    color: #E6EDF7;
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 300;
}

.main-header {
    font-size: 3.2rem;
    font-weight: 300;
    letter-spacing: -1.5px;
    background: linear-gradient(90deg,#7dd3fc,#a78bfa,#34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.stMetric {
    background: rgba(255,255,255,0.03);
    border-radius: 18px;
    padding: 16px;
    box-shadow: 0 12px 40px rgba(0,0,0,0.4);
}

section[data-testid='stSidebar'] {
    background-color: #05070B;
}

.block-container { padding-top: 2rem; }

</style>
""", unsafe_allow_html=True)

# ====================== CORE FINANCE FUNCTIONS ======================

def calculate_monthly_payment(principal, annual_rate, term_months):
    if principal <= 0 or term_months <= 0:
        return 0.0
    r = annual_rate / 12 / 100
    if r == 0:
        return principal / term_months
    return principal * r * (1 + r) ** term_months / ((1 + r) ** term_months - 1)


def solve_term(principal, annual_rate, payment):
    if principal <= 0 or payment <= 0:
        return 0
    r = annual_rate / 12 / 100
    if r == 0:
        return principal / payment
    return np.log(payment / (payment - principal * r)) / np.log(1 + r)

# ====================== FULL SIMULATION (RESTORED + EXTENDED) ======================

def simulate_split_loan(var_p, fix_p, var_rate, fix_rate, revert_rate,
                        fixed_years, term_months,
                        offset_start_orig=0,
                        offset_start_new=0,
                        monthly_offset_add=0,
                        monthly_fees=0,
                        one_time_fees=0,
                        mode="term"):

    fixed_months = int(fixed_years * 12)

    var_payment = calculate_monthly_payment(var_p, var_rate, term_months)
    fix_payment = calculate_monthly_payment(fix_p, fix_rate, term_months)

    var_bal, fix_bal = var_p, fix_p
    offset = 0

    schedule = []
    total_interest = 0
    total_paid = one_time_fees

    for m in range(1, int(term_months) + 1):

        # offset timing (dual regime)
        if m >= offset_start_orig or m >= offset_start_new:
            offset += monthly_offset_add

        # VARIABLE
        r_v = var_rate / 12 / 100
        interest_v = max(0, var_bal - offset) * r_v
        pay_v = min(var_payment - interest_v, var_bal)
        var_bal -= pay_v

        # FIXED
        if fix_p > 0 and m == fixed_months + 1:
            remaining = max(1, term_months - m)
            fix_payment = calculate_monthly_payment(fix_bal, revert_rate, remaining)

        r_f = fix_rate / 12 / 100 if m <= fixed_months else revert_rate / 12 / 100
        interest_f = fix_bal * r_f
        pay_f = min(fix_payment - interest_f, fix_bal)
        fix_bal -= pay_f

        total_interest += interest_v + interest_f
        total_paid += var_payment + fix_payment + monthly_fees

        schedule.append([m, interest_v + interest_f, pay_v + pay_f, var_bal + fix_bal, offset])

        if var_bal + fix_bal <= 0.01:
            break

    df = pd.DataFrame(schedule, columns=["month","interest","principal","balance","offset"])
    return df, total_interest, total_paid

# ====================== OPTIMISATION ENGINE ======================

def find_optimal_split(loan, var_r, fix_r, revert_r, fy, tm,
                       offset_start, offset_add):

    splits = np.linspace(0, 100, 101)
    interest = []
    debt = []

    for s in splits:
        vp = loan * (1 - s/100)
        fp = loan * s/100

        df, _, _ = simulate_split_loan(vp, fp, var_r, fix_r, revert_r,
                                       fy, tm, offset_start, offset_start,
                                       offset_add)

        interest.append(df["interest"].sum())
        debt.append(df.iloc[-1]["balance"])

    interest = np.array(interest)
    debt = np.array(debt)

    def cost(s):
        return np.interp(s, splits, interest) + 0.5 * np.interp(s, splits, debt)

    res = minimize_scalar(cost, bounds=(0,100), method="bounded")

    return int(res.x), splits, interest, debt

# ====================== UI ======================

st.markdown('<div class="main-header">Loan Refinance Optimiser</div>', unsafe_allow_html=True)

with st.sidebar:

    st.header("Original Loan")
    orig_val = st.number_input("House valuation")
    orig_loan = st.number_input("Loan amount")
    orig_term = st.number_input("Loan term (months)")
    orig_rate = st.number_input("Interest rate (%)")

    st.metric("Original LVR", f"{orig_loan / max(orig_val,1) * 100:.2f}%")

    st.divider()

    st.header("Refinance")
    loan_left = st.number_input("Remaining loan")
    var_rate = st.number_input("Variable rate")
    fix_rate = st.number_input("Fixed rate")
    revert_rate = st.number_input("Revert rate")
    fixed_years = st.number_input("Fixed years")

    st.divider()

    st.header("Strategy")

    mode = st.radio("Loan mode", ["Keep term fixed", "Keep payment fixed"])

    offset_start_orig = st.number_input("Offset start (orig)")
    offset_start_new = st.number_input("Offset start (new)")
    monthly_offset_add = st.number_input("Monthly offset add")

    monthly_fees = st.number_input("Monthly fees")
    one_time_fees = st.number_input("Upfront fees")

# ====================== MAIN ======================

if loan_left > 0:

    df, interest, paid = simulate_split_loan(
        loan_left, 0, var_rate, 0, 0,
        fixed_years, orig_term,
        offset_start_orig, offset_start_new,
        monthly_offset_add,
        monthly_fees, one_time_fees,
        mode
    )

    opt, s, i, d = find_optimal_split(
        loan_left, var_rate, fix_rate, revert_rate,
        fixed_years, orig_term,
        offset_start_new, monthly_offset_add
    )

    st.subheader("Core Results")

    c1, c2, c3 = st.columns(3)
    c1.metric("Optimal split % fixed", f"{opt}%")
    c2.metric("Total interest", f"${interest:,.0f}")
    c3.metric("Total paid", f"${paid:,.0f}")

    # ====================== 2D + 3D VISUALS ======================

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["month"], y=df["balance"], name="Balance"))
    fig.add_trace(go.Scatter(x=df["month"], y=df["interest"], name="Interest"))
    fig.update_layout(template="plotly_dark", height=450)
    st.plotly_chart(fig, use_container_width=True)

    # 3D SURFACE: SPLIT vs TIME vs INTEREST
    x = np.linspace(0, 100, 20)
    y = np.linspace(12, orig_term, 20)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(len(x)):
        for j in range(len(y)):
            vp = loan_left * (1 - x[i]/100)
            fp = loan_left * (x[i]/100)
            df2, _, _ = simulate_split_loan(
                vp, fp, var_rate, fix_rate, revert_rate,
                fixed_years, int(y[j]),
                offset_start_orig, offset_start_new,
                monthly_offset_add,
                monthly_fees, one_time_fees
            )
            Z[j,i] = df2["interest"].sum()

    fig3 = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])
    fig3.update_layout(
        title="3D Interest Surface (Split vs Term vs Cost)",
        template="plotly_dark",
        height=600
    )

    st.plotly_chart(fig3, use_container_width=True)

else:
    st.info("Enter loan details")
