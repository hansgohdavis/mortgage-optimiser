import streamlit as st
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import curve_fit

st.set_page_config(layout="wide")

# =========================
# INPUT MODEL
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
# CORE MATH
# =========================
def mrate(r):
    return r / 12


def payment(P, r, n):
    rm = mrate(r)
    if rm == 0:
        return P / n
    return P * (rm * (1 + rm) ** n) / ((1 + rm) ** n - 1)


# =========================
# AMORTISATION ENGINE
# =========================
def simulate(P, rate, months, offset=0.0, offset_add=0.0):
    bal = P
    off = offset

    total_interest = 0.0
    total_paid = 0.0

    pay = payment(P, rate, months)

    for _ in range(months):
        off += offset_add
        effective = max(bal - off, 0)

        interest = effective * mrate(rate)
        principal = pay - interest

        if principal > bal:
            principal = bal

        bal -= principal

        total_interest += interest
        total_paid += interest + principal

        if bal <= 0:
            break

    return total_interest, total_paid


# =========================
# SPLIT MODEL
# =========================
def split_model(inp: Inputs, split):
    fixed_P = inp.loan_balance * split
    var_P = inp.loan_balance * (1 - split)

    fixed_months = inp.fixed_years * 12

    fixed_i, fixed_paid = simulate(
        fixed_P,
        inp.new_fixed_rate,
        fixed_months
    )

    var_i, var_paid = simulate(
        var_P,
        inp.new_var_rate,
        inp.term_months,
        inp.offset_initial,
        inp.offset_monthly
    )

    total_interest = fixed_i + var_i
    total_paid = fixed_paid + var_paid + inp.setup_fees + inp.break_fees + inp.other_fees

    return {
        "split": split * 100,
        "interest": total_interest,
        "paid": total_paid
    }


# =========================
# OPTIMISATION
# =========================
def optimise(inp: Inputs):
    rows = []

    for s in range(101):
        res = split_model(inp, s / 100)
        rows.append([res["split"], res["interest"], res["paid"]])

    df = pd.DataFrame(rows, columns=["split", "interest", "paid"])

    # Smooth curve (logistic approximation)
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
# UI
# =========================
st.title("🏦 Mortgage Optimiser")

st.sidebar.header("Inputs")

house_value = st.sidebar.number_input("House Value", value=1_000_000.0)
loan_balance = st.sidebar.number_input("Loan Balance", value=600_000.0)
term_months = st.sidebar.number_input("Loan Term (months)", value=360)

current_rate = st.sidebar.number_input(
    "Current Rate",
    value=0.0600,
    format="%.4f",
    step=0.0001
)

offset_initial = st.sidebar.number_input("Offset", value=50_000.0)
offset_monthly = st.sidebar.number_input("Monthly Offset Add", value=1_000.0)

new_var_rate = st.sidebar.number_input(
    "New Variable Rate",
    value=0.0650,
    format="%.4f",
    step=0.0001
)

new_fixed_rate = st.sidebar.number_input(
    "New Fixed Rate",
    value=0.0550,
    format="%.4f",
    step=0.0001
)

fixed_years = st.sidebar.slider("Fixed Years", 1, 10, 2)

revert_rate = st.sidebar.number_input(
    "Revert Rate",
    value=0.0700,
    format="%.4f",
    step=0.0001
)

setup_fees = st.sidebar.number_input("Setup Fees", value=0.0)
monthly_fees = st.sidebar.number_input("Monthly Fees", value=0.0)
break_fees = st.sidebar.number_input("Break Fees", value=0.0)
other_fees = st.sidebar.number_input("Other Fees", value=0.0)


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
# OUTPUT
# =========================
col1, col2, col3 = st.columns(3)

col1.metric("Optimal Split", f"{best['split']:.0f}%")
col2.metric("Total Interest", f"${best['interest']:,.0f}")
col3.metric("Total Paid", f"${best['paid']:,.0f}")

st.subheader("Optimisation Curve")
st.line_chart(df.set_index("split")[["interest", "smooth"]])

st.subheader("Results Table")
st.dataframe(df)
