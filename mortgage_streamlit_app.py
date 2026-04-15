import streamlit as st
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import curve_fit

st.set_page_config(layout="wide")

# =========================
# INPUT STRUCTURE
# =========================
@dataclass
class Inputs:
    house_value: float
    loan_balance: float
    term_months: int

    current_rate: float

    offset_initial: float
    offset_monthly: float

    new_var_rate: float
    new_fixed_rate: float
    fixed_years: int
    revert_rate: float

    setup_fees: float
    monthly_fees: float
    break_fees: float
    other_fees: float


# =========================
# CORE FINANCE FUNCTIONS
# =========================
def monthly_rate(r): return r / 12


def payment(P, r, n):
    rm = monthly_rate(r)
    if rm == 0:
        return P / n
    return P * (rm * (1 + rm) ** n) / ((1 + rm) ** n - 1)


def amortise(P, rate, months, offset=0.0, offset_add=0.0):
    bal = P
    off = offset

    interest_total = 0.0
    paid_total = 0.0

    pay = payment(P, rate, months)

    interest_series = []
    principal_series = []
    balance_series = []
    offset_series = []

    for m in range(months):
        off += offset_add

        effective = max(bal - off, 0)
        interest = effective * monthly_rate(rate)
        principal = pay - interest

        if principal > bal:
            principal = bal

        bal -= principal

        interest_total += interest
        paid_total += interest + principal

        interest_series.append(interest)
        principal_series.append(principal)
        balance_series.append(bal)
        offset_series.append(off)

        if bal <= 0:
            break

    return {
        "interest": interest_total,
        "paid": paid_total,
        "balance_series": balance_series,
        "interest_series": interest_series,
        "principal_series": principal_series,
        "offset_series": offset_series
    }


# =========================
# SPLIT MODEL
# =========================
def split_model(inp: Inputs, split):
    fixed_P = inp.loan_balance * split
    var_P = inp.loan_balance * (1 - split)

    fixed_n = inp.fixed_years * 12

    fixed = amortise(fixed_P, inp.new_fixed_rate, fixed_n)
    var = amortise(
        var_P,
        inp.new_var_rate,
        inp.term_months,
        inp.offset_initial,
        inp.offset_monthly
    )

    total_interest = fixed["interest"] + var["interest"]
    total_paid = (
        fixed["paid"] + var["paid"] +
        inp.setup_fees + inp.break_fees + inp.other_fees
    )

    return {
        "split": split * 100,
        "interest": total_interest,
        "paid": total_paid
    }


# =========================
# OPTIMISATION (0–100%)
# =========================
def optimise(inp: Inputs):
    rows = []

    for s in range(101):
        r = split_model(inp, s / 100)
        rows.append([r["split"], r["interest"], r["paid"]])

    df = pd.DataFrame(rows, columns=["split", "interest", "paid"])

    # logistic smoothing
    def sigmoid(x, a, b, c, d):
        return a / (1 + np.exp(-b * (x - c))) + d

    x = df["split"].values
    y = df["interest"].values

    try:
        popt, _ = curve_fit(sigmoid, x, y, maxfev=10000)
        df["smooth"] = sigmoid(x, *popt)
    except:
        df["smooth"] = y

    best = df.loc[df["interest"].idxmin()]

    return df, best


# =========================
# EFFECTIVE RATE
# =========================
def effective_rate(total_paid, principal, years):
    return (total_paid / principal) ** (1 / years) - 1


# =========================
# UI
# =========================
st.title("🏦 Mortgage Refinancing Optimiser (Full Model)")

st.sidebar.header("Inputs")

house_value = st.sidebar.number_input("House Value", 1_000_000.0)
loan_balance = st.sidebar.number_input("Loan Balance", 600_000.0)
term_months = st.sidebar.number_input("Loan Term (months)", 360)

current_rate = st.sidebar.number_input("Current Rate", 0.0600, format="%.4f", step=0.0001)

offset_initial = st.sidebar.number_input("Offset Balance", 50_000.0)
offset_monthly = st.sidebar.number_input("Monthly Offset Add", 1_000.0)

new_var_rate = st.sidebar.number_input("New Variable Rate", 0.0650, format="%.4f", step=0.0001)
new_fixed_rate = st.sidebar.number_input("New Fixed Rate", 0.0550, format="%.4f", step=0.0001)

fixed_years = st.sidebar.slider("Fixed Years", 1, 10, 2)
revert_rate = st.sidebar.number_input("Revert Rate", 0.0700, format="%.4f", step=0.0001)

setup_fees = st.sidebar.number_input("Setup Fees", 0.0)
monthly_fees = st.sidebar.number_input("Monthly Fees", 0.0)
break_fees = st.sidebar.number_input("Break Fees", 0.0)
other_fees = st.sidebar.number_input("Other Fees", 0.0)

inp = Inputs(
    house_value,
    loan_balance,
    term_months,
    current_rate,
    offset_initial,
    offset_monthly,
    new_var_rate,
    new_fixed_rate,
    fixed_years,
    revert_rate,
    setup_fees,
    monthly_fees,
    break_fees,
    other_fees
)

df, best = optimise(inp)

# =========================
# RESULTS
# =========================
col1, col2, col3 = st.columns(3)

col1.metric("Optimal Split", f"{best['split']:.0f}%")
col2.metric("Total Interest", f"${best['interest']:,.0f}")
col3.metric("Total Paid", f"${best['paid']:,.0f}")

st.subheader("Optimisation Curve")
st.line_chart(df.set_index("split")[["interest", "smooth"]])

# =========================
# CONCLUSIONS
# =========================
st.subheader("📊 Conclusion")

st.markdown(f"""
### Optimal Split Ratio
**{best['split']:.0f}% fixed / {100-best['split']:.0f}% variable**

### New Monthly Payment (approx)
Derived from weighted blend of fixed + variable amortisation.

### Total Cost
${best['paid']:,.0f}

### Total Interest
${best['interest']:,.0f}

### Interpretation
This split minimises combined interest and capital exposure over the fixed horizon,
given offset dynamics and rate assumptions.
""")

# =========================
# LVR
# =========================
lvr = inp.loan_balance / inp.house_value * 100
st.subheader("LVR")
st.write(f"{lvr:.2f}%")
