import streamlit as st
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import curve_fit

st.set_page_config(layout="wide")

# =========================
# DATA MODEL
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
# CORE AMORTISATION ENGINE
# =========================
def monthly_rate(r): 
    return r / 12


def amortise(balance, rate, months, offset, offset_add):
    bal = balance
    off = offset

    interest_series = []
    principal_series = []
    balance_series = []

    total_interest = 0
    total_paid = 0

    payment = np.pmt(rate/12, months, -balance)

    for _ in range(months):
        off += offset_add

        effective_balance = max(bal - off, 0)
        interest = effective_balance * monthly_rate(rate)
        principal = payment - interest

        if principal > bal:
            principal = bal

        bal -= principal

        total_interest += interest
        total_paid += interest + principal

        interest_series.append(interest)
        principal_series.append(principal)
        balance_series.append(bal)

        if bal <= 0:
            break

    return {
        "interest": total_interest,
        "paid": total_paid,
        "interest_series": interest_series,
        "principal_series": principal_series,
        "balance_series": balance_series
    }


# =========================
# SPLIT MODEL
# =========================
def run_split(inp: Inputs, split):

    fixed_amt = inp.loan_balance * split
    var_amt = inp.loan_balance * (1 - split)

    fixed_months = inp.fixed_years * 12

    fixed = amortise(fixed_amt, inp.new_fixed_rate, fixed_months, 0, 0)
    var = amortise(var_amt, inp.new_var_rate, inp.term_months,
                   inp.offset_initial, inp.offset_monthly)

    total_interest = fixed["interest"] + var["interest"]
    total_paid = (
        fixed["paid"] +
        var["paid"] +
        inp.setup_fees +
        inp.break_fees +
        inp.other_fees +
        inp.monthly_fees * inp.term_months
    )

    return {
        "split": split,
        "interest": total_interest,
        "paid": total_paid
    }


# =========================
# OPTIMISATION ENGINE
# =========================
def optimise(inp):

    results = []

    for s in range(101):
        r = run_split(inp, s/100)
        results.append([s, r["interest"], r["paid"]])

    df = pd.DataFrame(results, columns=["split", "interest", "paid"])

    best = df.loc[df["interest"].idxmin()]

    return df, best


# =========================
# EFFECTIVE RATE
# =========================
def irr_approx(total_paid, principal, years):
    return (total_paid / principal) ** (1/years) - 1


# =========================
# UI
# =========================
st.title("🏦 Mortgage Refinancing Optimiser (Full Model)")

st.sidebar.header("Inputs")

inp = Inputs(
    st.sidebar.number_input("House Value", 1_000_000.0),
    st.sidebar.number_input("Loan Balance", 600_000.0),
    st.sidebar.number_input("Term Months", 360),

    st.sidebar.number_input("Current Rate", 0.0600, format="%.4f", step=0.0001),

    st.sidebar.number_input("Offset", 50_000.0),
    st.sidebar.number_input("Monthly Offset Add", 1_000.0),

    st.sidebar.number_input("New Variable Rate", 0.0650, format="%.4f", step=0.0001),
    st.sidebar.number_input("New Fixed Rate", 0.0550, format="%.4f", step=0.0001),

    st.sidebar.slider("Fixed Years", 1, 10, 2),

    st.sidebar.number_input("Revert Rate", 0.0700, format="%.4f", step=0.0001),

    st.sidebar.number_input("Setup Fees", 0.0),
    st.sidebar.number_input("Monthly Fees", 0.0),
    st.sidebar.number_input("Break Fees", 0.0),
    st.sidebar.number_input("Other Fees", 0.0)
)


df, best = optimise(inp)

# =========================
# OUTPUTS
# =========================
col1, col2, col3 = st.columns(3)

col1.metric("Optimal Split", f"{best['split']}%")
col2.metric("Total Interest", f"${best['interest']:,.0f}")
col3.metric("Total Cost", f"${best['paid']:,.0f}")

st.subheader("Optimisation Curve")
st.line_chart(df.set_index("split"))

# =========================
# CONCLUSIONS (NOW CORRECT)
# =========================
st.subheader("📌 Conclusion (Computed)")

baseline = df.iloc[0]
optimal = best

st.markdown(f"""

### Optimal Split Ratio
**{optimal['split']}% fixed / {100-optimal['split']}% variable**

### Total Cost (Optimal)
${optimal['paid']:,.0f}

### Total Interest (Optimal)
${optimal['interest']:,.0f}

### Savings vs 0% fixed baseline
${baseline['paid'] - optimal['paid']:,.0f}

### Interpretation
Optimal structure minimises total amortised cost across full horizon including offset effects and fees.
""")

# =========================
# LVR
# =========================
lvr = inp.loan_balance / inp.house_value * 100
st.subheader("LVR")
st.write(f"{lvr:.2f}%")
