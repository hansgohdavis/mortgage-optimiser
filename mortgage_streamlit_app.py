import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# =========================
# DATE HANDLING
# =========================

def parse_date(d):
    try:
        return datetime.strptime(d, "%d/%m/%Y")
    except:
        return None

def month_range(start, end):
    dates = []
    current = start
    while current <= end:
        dates.append(current)
        next_month = current.month + 1
        year = current.year + (next_month - 1)//12
        month = ((next_month - 1)%12) + 1
        current = current.replace(year=year, month=month)
    return dates

# =========================
# EVENT SYSTEM
# =========================

class Event:
    def __init__(self, date, type, value):
        self.date = date
        self.type = type
        self.value = value

# =========================
# ENGINE
# =========================

def simulate_loan(
    start_date,
    loan,
    rate,
    term,
    offset,
    events,
    monthly_payment=None
):

    months = term
    dates = [start_date + pd.DateOffset(months=i) for i in range(months)]

    balance = loan
    current_rate = rate
    current_offset = offset

    if monthly_payment is None:
        r = rate/12
        monthly_payment = loan * (r*(1+r)**term)/((1+r)**term -1)

    results = []

    for i, d in enumerate(dates):

        # Apply events
        for e in events:
            if e.date == d:
                if e.type == "rate":
                    current_rate = e.value
                if e.type == "offset":
                    current_offset += e.value

        # DAILY INTEREST → MONTHLY
        days = 30
        net_balance = max(balance - current_offset, 0)

        interest = net_balance * current_rate * days/365

        balance += interest

        principal = monthly_payment - interest
        balance -= monthly_payment

        results.append({
            "Date": d,
            "Balance": max(balance,0),
            "Interest": interest,
            "Principal": principal,
            "Payment": monthly_payment,
            "Rate": current_rate,
            "Offset": current_offset
        })

        if balance <= 0:
            break

    return pd.DataFrame(results)

# =========================
# OPTIMISER
# =========================

def optimise_split(loan, var_rate, fix_rate, term, start_date):

    best = None
    best_ratio = 0
    curve = []

    for r in np.arange(0,1.01,0.01):

        var = simulate_loan(start_date, loan*r, var_rate, term, 0, [])
        fix = simulate_loan(start_date, loan*(1-r), fix_rate, term, 0, [])

        total = var["Interest"].sum() + fix["Interest"].sum()

        curve.append((r,total))

        if best is None or total < best:
            best = total
            best_ratio = r

    df = pd.DataFrame(curve, columns=["Ratio","Interest"])

    return best_ratio, df

# =========================
# UI
# =========================

st.title("🏠 Full Mortgage Optimisation Engine")

with st.sidebar:

    st.header("Original Loan")

    start_date_str = st.text_input("Start Date (DD/MM/YYYY)", "01/01/2024")
    start_date = parse_date(start_date_str)

    loan = st.number_input("Loan Amount", 0.0, 30000000.0, 800000.0)

    rate = st.number_input("Interest (%)", 0.0, 20.0, 6.0)/100

    term = st.number_input("Term (months)", 1, 600, 360)

    offset = st.number_input("Offset", 0.0, 10000000.0, 0.0)

    st.header("Rate Changes")

    num_changes = st.number_input("Number of Changes", 0, 15, 0)

    events = []

    for i in range(num_changes):
        d = st.text_input(f"Date {i}", f"01/01/202{i+5}")
        r = st.number_input(f"Rate {i} (%)", 0.0, 20.0, 6.0, key=i)/100
        events.append(Event(parse_date(d), "rate", r))

    st.header("Proposed Loan")

    var_rate = st.number_input("Variable Rate (%)", 0.0, 20.0, 6.5)/100
    fix_rate = st.number_input("Fixed Rate (%)", 0.0, 20.0, 5.5)/100

    st.header("Scenario")

    rba_change = st.number_input("RBA Change (%)", -5.0, 5.0, 0.5)/100

# =========================
# VALIDATION
# =========================

if start_date is None:
    st.error("Invalid start date")
    st.stop()

for e in events:
    if e.date < start_date:
        st.error("Error: rate change before loan start")
        st.stop()

# =========================
# RUN ENGINE
# =========================

baseline = simulate_loan(start_date, loan, rate, term, offset, events)

opt_ratio, curve = optimise_split(loan, var_rate, fix_rate, term, start_date)

var = simulate_loan(start_date, loan*opt_ratio, var_rate, term, 0, [])
fix = simulate_loan(start_date, loan*(1-opt_ratio), fix_rate, term, 0, [])

# =========================
# DASHBOARD
# =========================

st.header("📊 Dashboard")

col1, col2, col3 = st.columns(3)

base_cost = baseline["Payment"].sum()
new_cost = var["Payment"].sum() + fix["Payment"].sum()

col1.metric("Baseline Cost", f"${base_cost:,.0f}")
col2.metric("New Cost", f"${new_cost:,.0f}")
col3.metric("Savings", f"${base_cost-new_cost:,.0f}")

# =========================
# MONTHLY
# =========================

st.subheader("Monthly Payments")

fig = go.Figure()

fig.add_trace(go.Scatter(x=baseline["Date"], y=baseline["Payment"], name="Baseline"))
fig.add_trace(go.Scatter(x=var["Date"], y=var["Payment"], name="Variable"))
fig.add_trace(go.Scatter(x=fix["Date"], y=fix["Payment"], name="Fixed"))

st.plotly_chart(fig, use_container_width=True)

# =========================
# CUMULATIVE
# =========================

st.subheader("Cumulative Cost")

fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=baseline["Date"],
    y=baseline["Payment"].cumsum(),
    name="Baseline"
))

fig2.add_trace(go.Scatter(
    x=var["Date"],
    y=var["Payment"].cumsum() + fix["Payment"].cumsum(),
    name="New"
))

st.plotly_chart(fig2, use_container_width=True)

# =========================
# OPT CURVE
# =========================

st.subheader("Optimisation Curve")

fig3 = go.Figure()

fig3.add_trace(go.Scatter(
    x=curve["Ratio"]*100,
    y=curve["Interest"],
    name="Interest"
))

fig3.add_vline(x=opt_ratio*100)

st.plotly_chart(fig3, use_container_width=True)

st.write(f"Optimal Split: {opt_ratio*100:.2f}% variable")
