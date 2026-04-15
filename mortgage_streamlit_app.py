"""
Mortgage Optimiser — Fintech UI Version
-------------------------------------
Clean, modern UI with:
- KPI cards
- Tabs
- Better layout
"""

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")

# -----------------------------
# STYLING
# -----------------------------
st.markdown("""
<style>
.metric-card {
    padding: 20px;
    border-radius: 12px;
    background-color: #111827;
    color: white;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.title("🏦 Mortgage Optimiser")

# -----------------------------
# INPUTS
# -----------------------------
with st.sidebar:
    st.header("⚙️ Inputs")

    scenario = st.selectbox(
        "Scenario",
        ["Custom", "RBA +0.5%", "Rates Hold", "Rates Rise Aggressively"]
    )

    loan_balance = st.number_input("Loan Balance", value=1000000.0)
    term_months = st.number_input("Loan Term (months)", value=360)

    base_rate = st.number_input("Base Variable Rate", value:=0.0589, step=0.0001, format="%0.4f")    

    if scenario == "RBA +0.5%":
        var_rate := 0.0639 , step=0.0001, format="%0.4f"
    elif scenario == "Rates Hold":
        var_rate := 0.0589 , step=0.0001, format="%0.4f"
    elif scenario == "Rates Rise Aggressively":
        var_rate := 0.0739 , step=0.0001, format="%0.4f"
    else:
        var_rate = st.number_input("Variable Rate", value:=0.0565 , step=0.0001, format="%0.4f")

    fix_rate = st.number_input("Fixed Rate", value:=0.0585 , step=0.0001, format="%0.4f")
    fixed_years = st.slider("Fixed Years", 1, 10, 2)

    offset = st.number_input("Offset", value=100000.0)
    offset_add = st.number_input("Monthly Offset Add", value=5000.0)

# -----------------------------
# CORE FUNCTIONS
# -----------------------------
def monthly_payment(P, r, n):
    r_m = r / 12
    if r_m == 0:
        return P / n
    return P * (r_m * (1 + r_m) ** n) / ((1 + r_m) ** n - 1)


def simulate_variable(P, r, n, offset, offset_add):
    balance = P
    offset_bal = offset
    total_interest = 0

    for _ in range(n):
        offset_bal += offset_add
        eff = max(balance - offset_bal, 0)
        interest = eff * (r / 12)
        payment = monthly_payment(P, r, n)
        principal = payment - interest
        balance -= principal
        total_interest += interest

        if balance <= 0:
            break

    return total_interest


def optimise_split(P, r_var, r_fix, years, offset, offset_add):
    results = []

    for split in range(101):
        fixed_P = P * split / 100
        var_P = P - fixed_P

        fixed_interest = fixed_P * r_fix * years
        var_interest = simulate_variable(var_P, r_var, years * 12, offset, offset_add)

        total = fixed_interest + var_interest
        results.append((split, total))

    df = pd.DataFrame(results, columns=["split", "total_cost"])
    best = df.loc[df.total_cost.idxmin()]

    return df, best

# -----------------------------
# RUN MODEL
# -----------------------------
df, best = optimise_split(
    loan_balance, var_rate, fix_rate, fixed_years, offset, offset_add
)

# -----------------------------
# KPI CARDS
# -----------------------------
col1, col2, col3 = st.columns(3)

col1.markdown(f"""
<div class="metric-card">
<h3>Optimal Split</h3>
<h1>{int(best['split'])}%</h1>
</div>
""", unsafe_allow_html=True)

col2.markdown(f"""
<div class="metric-card">
<h3>Total Cost</h3>
<h1>${best['total_cost']:,.0f}</h1>
</div>
""", unsafe_allow_html=True)

col3.markdown(f"""
<div class="metric-card">
<h3>Fixed Years</h3>
<h1>{fixed_years}</h1>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# TABS
# -----------------------------
tab1, tab2 = st.tabs(["📈 Optimisation", "💾 Scenarios"])

with tab1:
    st.subheader("Optimisation Curve")
    st.line_chart(df.set_index("split"))

with tab2:
    if "saved_runs" not in st.session_state:
        st.session_state.saved_runs = []

    if st.button("Save Scenario"):
        st.session_state.saved_runs.append(best.to_dict())

    st.write(st.session_state.saved_runs)

# -----------------------------
# FOOTER
# -----------------------------
st.caption("Fintech-style mortgage optimisation tool — decision support, not financial advice.")
