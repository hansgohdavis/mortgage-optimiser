import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# =========================
# UTILITIES
# =========================

def parse_date(date_str):
    try:
        return datetime.strptime(date_str, "%d/%m/%Y")
    except:
        return None

def monthly_dates(start_date, months):
    dates = []
    current = start_date
    for _ in range(months):
        dates.append(current)
        next_month = current.month + 1
        year = current.year + (next_month - 1) // 12
        month = ((next_month - 1) % 12) + 1
        current = current.replace(year=year, month=month)
    return dates

def monthly_payment(principal, rate, months):
    if rate == 0:
        return principal / months
    r = rate / 12
    return principal * (r * (1 + r)**months) / ((1 + r)**months - 1)

# =========================
# CORE AMORTISATION ENGINE
# =========================

def amortisation_schedule(
    loan_amount,
    rate,
    months,
    offset=0,
    monthly_offset_add=0,
    monthly_fee=0
):
    balance = loan_amount
    schedule = []

    payment = monthly_payment(loan_amount, rate, months)

    for m in range(months):
        net_balance = max(balance - offset, 0)

        interest = net_balance * rate / 12
        principal = payment - interest

        balance = balance + interest - payment
        offset += monthly_offset_add

        schedule.append({
            "Month": m+1,
            "Interest": interest,
            "Principal": principal,
            "Balance": max(balance, 0),
            "Offset": offset,
            "Payment": payment + monthly_fee
        })

        if balance <= 0:
            break

    return pd.DataFrame(schedule)

# =========================
# OPTIMISER
# =========================

def optimise_split(loan, var_rate, fix_rate, months):
    results = []

    for ratio in np.arange(0, 1.01, 0.01):
        var_part = loan * ratio
        fix_part = loan * (1 - ratio)

        var_sched = amortisation_schedule(var_part, var_rate, months)
        fix_sched = amortisation_schedule(fix_part, fix_rate, months)

        total_interest = var_sched["Interest"].sum() + fix_sched["Interest"].sum()
        results.append((ratio, total_interest))

    df = pd.DataFrame(results, columns=["Ratio", "Interest"])
    best = df.loc[df["Interest"].idxmin()]

    return best["Ratio"], df

# =========================
# UI — INPUTS
# =========================

st.title("🏠 Advanced Mortgage Optimisation Dashboard")

with st.sidebar:

    st.header("Original Loan")

    orig_amount = st.number_input("Loan Amount", 0.0, 30000000.0, 500000.0)
    orig_rate = st.number_input("Interest Rate (%)", 0.0, 20.0, 5.5) / 100
    orig_term = st.number_input("Term (months)", 1, 600, 360)
    orig_offset = st.number_input("Offset Amount", 0.0, 10000000.0, 0.0)
    orig_offset_add = st.number_input("Monthly Offset Add", 0.0, 100000.0, 0.0)

    st.header("Proposed Loan")

    new_rate_var = st.number_input("Variable Rate (%)", 0.0, 20.0, 6.0) / 100
    new_rate_fix = st.number_input("Fixed Rate (%)", 0.0, 20.0, 5.5) / 100

    strategy = st.selectbox(
        "Strategy",
        ["Conservative", "Balanced", "Aggressive"]
    )

    rba_change = st.number_input("RBA Change (%)", -5.0, 5.0, 0.5) / 100

# =========================
# CALCULATIONS
# =========================

baseline = amortisation_schedule(
    orig_amount,
    orig_rate,
    orig_term,
    orig_offset,
    orig_offset_add
)

opt_ratio, opt_curve = optimise_split(
    orig_amount,
    new_rate_var,
    new_rate_fix,
    orig_term
)

var_sched = amortisation_schedule(
    orig_amount * opt_ratio,
    new_rate_var,
    orig_term
)

fix_sched = amortisation_schedule(
    orig_amount * (1 - opt_ratio),
    new_rate_fix,
    orig_term
)

# =========================
# DASHBOARD
# =========================

st.header("📊 Analysis Dashboard")

col1, col2, col3 = st.columns(3)

baseline_cost = baseline["Payment"].sum()
new_cost = var_sched["Payment"].sum() + fix_sched["Payment"].sum()

col1.metric("Baseline Cost", f"${baseline_cost:,.0f}")
col2.metric("New Cost", f"${new_cost:,.0f}")
col3.metric("Savings", f"${baseline_cost - new_cost:,.0f}")

# =========================
# MONTHLY GRAPH
# =========================

st.subheader("Monthly Payments")

fig = go.Figure()

fig.add_trace(go.Scatter(
    y=baseline["Payment"],
    name="Baseline Monthly Payment"
))

fig.add_trace(go.Scatter(
    y=var_sched["Payment"],
    name="Variable Component"
))

fig.add_trace(go.Scatter(
    y=fix_sched["Payment"],
    name="Fixed Component"
))

st.plotly_chart(fig, use_container_width=True)

# =========================
# CUMULATIVE GRAPH
# =========================

st.subheader("Cumulative Cost")

baseline_cum = baseline["Payment"].cumsum()
new_cum = (var_sched["Payment"].cumsum() + fix_sched["Payment"].cumsum())

fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    y=baseline_cum,
    name="Baseline Total Cost"
))

fig2.add_trace(go.Scatter(
    y=new_cum,
    name="New Total Cost"
))

st.plotly_chart(fig2, use_container_width=True)

# =========================
# OPTIMISATION CURVE
# =========================

st.subheader("Optimal Split Ratio Curve")

fig3 = go.Figure()

fig3.add_trace(go.Scatter(
    x=opt_curve["Ratio"] * 100,
    y=opt_curve["Interest"],
    name="Interest vs Split Ratio"
))

fig3.add_vline(
    x=opt_ratio * 100,
    line_dash="dash"
)

st.plotly_chart(fig3, use_container_width=True)

st.write(f"Optimal Split Ratio: {opt_ratio*100:.2f}% variable")
