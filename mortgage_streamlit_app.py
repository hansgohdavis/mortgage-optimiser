import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize_scalar
import plotly.graph_objects as go

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Loan Refinance Optimiser",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== ULTRA MINIMAL FINTECH UI ======================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@200;300;400;500&display=swap');

html, body, [class*="css"]  {
    background-color: #070A0F;
    color: #E6EDF7;
    font-family: 'IBM Plex Sans', 'Helvetica Neue', sans-serif;
    font-weight: 300;
}

.main-header {
    font-size: 3.2rem;
    font-weight: 300;
    letter-spacing: -1.5px;
    background: linear-gradient(90deg,#7dd3fc,#a78bfa,#34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.stMetric {
    background: rgba(255,255,255,0.03);
    border-radius: 20px;
    padding: 16px;
    box-shadow: 0 12px 40px rgba(0,0,0,0.4);
    border: 1px solid rgba(255,255,255,0.05);
}

section[data-testid="stSidebar"] {
    background-color: #05070B;
}

.block-container {
    padding-top: 2rem;
}

/* micro interaction */
button:hover {
    transform: translateY(-1px);
    transition: all 0.15s ease;
}

</style>
""", unsafe_allow_html=True)

# ====================== CORE FUNCTIONS ======================
@st.cache_data(ttl=3600)
def calculate_monthly_payment(principal, annual_rate, term_months):
    if principal <= 0 or term_months <= 0:
        return 0.0
    r = annual_rate / 12 / 100
    if r == 0:
        return principal / term_months
    return principal * r * (1 + r) ** term_months / ((1 + r) ** term_months - 1)

# ====================== SIMULATION ENGINE ======================
@st.cache_data(ttl=3600)
def simulate_split_loan(var_p, fix_p, var_rate, fix_rate, revert_rate,
                        fixed_years, term_months,
                        offset_start_month=0,
                        monthly_offset_add=0,
                        monthly_fees=0,
                        one_time_fees=0):

    fixed_months = int(fixed_years * 12)

    var_payment = calculate_monthly_payment(var_p, var_rate, term_months)
    fix_payment = calculate_monthly_payment(fix_p, fix_rate, term_months)

    var_bal, fix_bal = var_p, fix_p
    offset = 0

    schedule = []
    total_interest = 0
    total_paid = one_time_fees

    for m in range(1, int(term_months) + 1):

        if m >= offset_start_month:
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

# ====================== OPTIMISATION ======================
@st.cache_data(ttl=3600)
def find_optimal_split(loan, var_r, fix_r, revert_r, fy, tm,
                       offset_start, offset_add):

    splits = np.linspace(0, 100, 101)
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

    return int(res.x), splits, interest, debt

# ====================== UI ======================
st.markdown('<div class="main-header">Loan Refinance Optimiser</div>', unsafe_allow_html=True)

with st.sidebar:

    st.header("Original Loan")
    orig_val = st.number_input("House valuation", value=850000.0)
    orig_loan = st.number_input("Loan amount", value=620000.0)
    orig_term = st.number_input("Loan term (months)", value=300.0)
    orig_rate = st.number_input("Interest rate (%)", value=6.35)

    st.metric("Original LVR", f"{orig_loan / orig_val * 100:.2f}%")

    st.divider()

    st.header("Refinance")
    loan_left = st.number_input("Remaining loan", value=orig_loan)
    var_rate = st.number_input("Variable rate", value=5.5)
    fix_rate = st.number_input("Fixed rate", value=4.9)
    revert_rate = st.number_input("Revert rate", value=6.3)
    fixed_years = st.number_input("Fixed years", value=2.0)

    st.divider()

    st.header("Strategy")

    mode = st.radio("Loan mode", ["Keep term fixed", "Keep payment fixed"])

    offset_start_orig = st.number_input("Offset start (orig months)", value=0.0)
    offset_start_new = st.number_input("Offset start (new months)", value=0.0)
    monthly_offset_add = st.number_input("Monthly offset add", value=1000.0)

    monthly_fees = st.number_input("Monthly fees", value=0.0)
    one_time_fees = st.number_input("Upfront fees", value=300.0)

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

    st.subheader("Optimised Outcome")

    c1, c2, c3 = st.columns(3)
    c1.metric("Optimal split % fixed", f"{opt}%")
    c2.metric("Interest", f"${interest:,.0f}")
    c3.metric("Total paid", f"${paid:,.0f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["month"], y=df["balance"], name="Balance"))
    fig.add_trace(go.Scatter(x=df["month"], y=df["interest"], name="Interest"))

    fig.update_layout(
        template="plotly_dark",
        height=520,
        margin=dict(l=20,r=20,t=40,b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=s, y=i, name="Interest curve"))
    fig2.add_trace(go.Scatter(x=s, y=d, name="Debt curve"))

    fig2.update_layout(template="plotly_dark", height=420)

    st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("Enter loan details")
