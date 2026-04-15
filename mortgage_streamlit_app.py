import streamlit as st
import numpy as np
import pandas as pd

from dataclasses import dataclass
from scipy.optimize import curve_fit

st.set_page_config(layout="wide")

# =========================
# ERROR WRAPPER
# =========================
def safe_run(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"Error- unable to calculate: {str(e)}")
            return None
    return wrapper


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
# BASIC FUNCTIONS
# =========================
def mrate(r):
    return r / 12


def payment(P, r, n):
    r_m = mrate(r)
    if r_m == 0:
        return P / n
    return P * (r_m * (1 + r_m) ** n) / ((1 + r_m) ** n - 1)


# =========================
# AMORTISATION ENGINE
# =========================
def simulate(P, rate, months, offset=0, offset_add=0):
    bal = P
    off = offset

    sch = []
    total_interest = 0
    total_paid = 0

    pay = payment(P, rate, months)

    for m in range(months):
        off += offset_add
        eff = max(bal - off, 0)

        i = eff * mrate(rate)
        p = pay - i

        if p > bal:
            p = bal

        bal -= p
        total_interest += i
        total_paid += i + p

        sch.append([m+1, bal, i, p, off])

        if bal <= 0:
            break

    df = pd.DataFrame(sch, columns=["month", "balance", "interest", "principal", "offset"])
    return df, total_interest, total_paid


# =========================
# SPLIT MODEL
# =========================
def split_model(inp: Inputs, split):
    fixed_P = inp.loan_balance * split
    var_P = inp.loan_balance - fixed_P

    fixed_n = inp.fixed_years * 12

    fixed_df, fixed_i, fixed_paid = simulate(
        fixed_P, inp.new_fixed_rate, fixed_n
    )

    var_df, var_i, var_paid = simulate(
        var_P, inp.new_var_rate, inp.term_months,
        inp.offset_initial, inp.offset_monthly
    )

    total_interest = fixed_i + var_i
    total_paid = fixed_paid + var_paid + inp.setup_fees + inp.break_fees

    return {
        "split": split,
        "interest": total_interest,
        "paid": total_paid,
        "fixed_df": fixed_df,
        "var_df": var_df
    }


# =========================
# OPTIMISATION
# =========================
@safe_run
def optimise(inp: Inputs):
    rows = []

    for s in range(101):
        r = split_model(inp, s/100)
        rows.append([s, r["interest"], r["paid"]])

    df = pd.DataFrame(rows, columns=["split", "interest", "paid"])

    # Logistic-style curve (sigmoid fit)
    def sigmoid(x, a, b, c, d):
        return a / (1 + np.exp(-b*(x-c))) + d

    x = df["split"].values
    y = df["interest"].values

    try:
        popt, _ = curve_fit(sigmoid, x, y, maxfev=10000)
        df["fit_interest"] = sigmoid(x, *popt)
    except:
        df["fit_interest"] = y

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
st.title("🏦 Mortgage Refinancing Optimiser")

st.sidebar.header("Inputs")

house_value = st.sidebar.number_input("House Value", 1000000.0)
loan_balance = st.sidebar.number_input("Loan Balance", 600000.0)
term_months = st.sidebar.number_input("Loan Term (months)", 360)

current_rate = st.sidebar.number_input("Current Rate", 0.06, format="%.4f")

offset_initial = st.sidebar.number_input("Offset", 50000.0)
offset_monthly = st.sidebar.number_input("Monthly Offset Add", 1000.0)

new_var_rate = st.sidebar.number_input("New Variable Rate", 0.065, format="%.4f")
new_fixed_rate = st.sidebar.number_input("New Fixed Rate", 0.055, format="%.4f")

fixed_years = st.sidebar.slider("Fixed Years", 1, 10, 2)

setup_fees = st.sidebar.number_input("Setup Fees", 0.0)
monthly_fees = st.sidebar.number_input("Monthly Fees", 0.0)
break_fees = st.sidebar.number_input("Break Fees", 0.0)
other_fees = st.sidebar.number_input("Other Fees", 0.0)

inp = Inputs(
    house_value, loan_balance, term_months,
    current_rate,
    offset_initial, offset_monthly,
    new_var_rate, new_fixed_rate,
    fixed_years, current_rate,
    setup_fees, monthly_fees, break_fees, other_fees
)

df, best = optimise(inp)

if df is not None:

    # =========================
    # KPI
    # =========================
    col1, col2, col3 = st.columns(3)

    col1.metric("Optimal Split", f"{best['split']}%")
    col2.metric("Total Interest", f"${best['interest']:,.0f}")
    col3.metric("Total Paid", f"${best['paid']:,.0f}")

    # =========================
    # CHART
    # =========================
    st.subheader("Optimisation Curve")
    st.line_chart(df.set_index("split")[["interest", "fit_interest"]])

    # =========================
    # MONTHLY SCHEDULE (OPTIMAL)
    # =========================
    st.subheader("Monthly Schedule (Optimal Split)")

    opt = split_model(inp, best["split"]/100)

    st.write("Fixed Loan")
    st.dataframe(opt["fixed_df"])

    st.write("Variable Loan")
    st.dataframe(opt["var_df"])

    # =========================
    # LVR
    # =========================
    lvr = inp.loan_balance / inp.house_value * 100
    st.subheader("LVR")
    st.write(f"{lvr:.2f}%")

    # =========================
    # EFFECTIVE RATE
    # =========================
    years = inp.fixed_years

    eff = effective_rate(best["paid"], inp.loan_balance, years)

    st.subheader("Effective Rate (Approx)")
    st.write(f"{eff*100:.2f}% per annum")

    st.caption("Model includes offset effects, fees, and split optimisation.")
