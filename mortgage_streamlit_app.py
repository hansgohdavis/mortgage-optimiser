import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize_scalar, newton
import plotly.graph_objects as go

# ====================== CORE FUNCTIONS (UNCHANGED LOGIC) ======================
@st.cache_data(ttl=3600, show_spinner=False)
def logistic(x, L, k, x0, b):
    return L / (1 + np.exp(-k * (x - x0))) + b

@st.cache_data(ttl=3600, show_spinner=False)
def calculate_monthly_payment(principal, annual_rate, term_months):
    if principal <= 0:
        return 0.0
    monthly_rate = annual_rate / 12 / 100
    if monthly_rate == 0:
        return principal / term_months
    power = (1 + monthly_rate) ** term_months
    return principal * monthly_rate * power / (power - 1)

# ====================== CORE SIMULATION (PRESERVED + EXTENDED ONLY) ======================
@st.cache_data(ttl=3600, show_spinner=False)
def simulate_split_loan(var_principal, fixed_principal, variable_rate, fixed_rate, revert_rate_after_fixed,
                        fixed_years, term_months, offset_start, monthly_offset_add,
                        monthly_fees=0.0, one_time_fees=0.0):

    fixed_months = int(fixed_years * 12)
    var_payment = calculate_monthly_payment(var_principal, variable_rate, term_months)
    fixed_payment = calculate_monthly_payment(fixed_principal, fixed_rate, term_months)

    var_balance = var_principal
    fixed_balance = fixed_principal
    offset = offset_start

    schedule = []
    cumulative_interest = 0.0
    cumulative_paid = float(one_time_fees)
    current_fixed_rate = fixed_rate
    current_fixed_payment = fixed_payment

    for month in range(1, term_months + 1):

        var_monthly_rate = variable_rate / 12 / 100
        interest_var_no_offset = var_balance * var_monthly_rate
        interest_var = max(0.0, var_balance - offset) * var_monthly_rate
        interest_saved_var = interest_var_no_offset - interest_var

        principal_var = var_payment - interest_var
        if principal_var > var_balance:
            principal_var = var_balance
            var_payment = interest_var + principal_var
        var_balance = max(0.0, var_balance - principal_var)

        if fixed_principal > 0 and month == fixed_months + 1 and fixed_balance > 0:
            remaining_months = term_months - fixed_months
            current_fixed_payment = calculate_monthly_payment(fixed_balance, revert_rate_after_fixed, remaining_months)

        fixed_monthly_rate = (revert_rate_after_fixed if month > fixed_months else fixed_rate) / 12 / 100
        interest_fixed = fixed_balance * fixed_monthly_rate
        principal_fixed = current_fixed_payment - interest_fixed

        if principal_fixed > fixed_balance:
            principal_fixed = fixed_balance
            current_fixed_payment = interest_fixed + principal_fixed

        fixed_balance = max(0.0, fixed_balance - principal_fixed)

        total_interest = interest_var + interest_fixed
        total_principal = principal_var + principal_fixed
        total_payment = var_payment + current_fixed_payment + monthly_fees

        cumulative_interest += total_interest
        cumulative_paid += total_payment

        schedule.append({
            "month": month,
            "interest": total_interest,
            "principal": total_principal,
            "interest_saved": interest_saved_var,
            "balance": var_balance + fixed_balance,
            "payment": total_payment,
            "cumulative_interest": cumulative_interest,
            "cumulative_paid": cumulative_paid,
            "offset": offset
        })

        offset += monthly_offset_add
        if (var_balance + fixed_balance) <= 0.01:
            break

    return pd.DataFrame(schedule), cumulative_interest, cumulative_paid

# ====================== OPTIMISATION (UNCHANGED) ======================
@st.cache_data(ttl=3600, show_spinner=False)
def find_optimal_split(loan_left, variable_rate, fixed_rate, revert_rate, fixed_years, term_months,
                       offset_start, monthly_offset_add, monthly_fees, one_time_fees):

    splits = list(range(0, 101))
    interests = []
    debts = []

    for s in splits:
        ratio = s / 100.0
        var_p = loan_left * (1 - ratio)
        fixed_p = loan_left * (ratio)

        df, _, _ = simulate_split_loan(var_p, fixed_p, variable_rate, fixed_rate, revert_rate,
                                       fixed_years, term_months, offset_start, monthly_offset_add,
                                       monthly_fees, one_time_fees)

        idx = min(fixed_years * 12, len(df))
        interests.append(df.iloc[:idx]["interest"].sum())
        debts.append(df.iloc[idx-1]["balance"] if idx > 0 else loan_left)

    x = np.array(splits)
    y_interest = np.array(interests)
    y_debt = np.array(debts)

    try:
        popt_i, _ = curve_fit(logistic, x, y_interest, maxfev=10000)
        popt_d, _ = curve_fit(logistic, x, y_debt, maxfev=10000)

        def cost(s):
            return logistic(s, *popt_i) + 0.5 * logistic(s, *popt_d)

        res = minimize_scalar(cost, bounds=(0, 100), method="bounded")
        opt = int(round(res.x))

        x_fit = np.linspace(0, 100, 200)
        return opt, splits, y_interest, y_debt, x_fit, logistic(x_fit, *popt_i), logistic(x_fit, *popt_d)

    except:
        return int(splits[np.argmin(y_interest)]), splits, y_interest, y_debt, None, None, None

# ====================== UI ======================
st.set_page_config(page_title="Loan Refinance Optimiser", layout="wide")

st.markdown("""
<style>
html, body {
    background:#05070C;
    color:#EAF0FF;
    font-family: Space Grotesk, sans-serif;
}
</style>
""", unsafe_allow_html=True)

st.title("🏠 Loan Refinance Optimiser")

# ====================== INPUTS (ORIGINAL + NEW ADDITIONS) ======================
with st.sidebar:

    st.header("Original Loan")
    orig_val = st.number_input("Original House Valuation", value=850000.0)
    orig_loan = st.number_input("Original Loan Amount", value=620000.0)
    orig_term = st.number_input("Original Loan Term (months)", value=300)
    orig_rate = st.number_input("Original Interest Rate", value=6.35)

    st.metric("Original LVR", f"{orig_loan/orig_val*100:.2f}%")

    st.divider()

    st.header("Current Loan")
    loan_left = st.number_input("Loan Remaining", value=620000.0)
    term_months = st.number_input("Remaining Term", value=300)
    current_rate = st.number_input("Current Rate", value=6.35)
    offset_current = st.number_input("Current Offset", value=45000.0)

    monthly_offset_add = st.number_input("Monthly Offset Add", value=1200.0)

    offset_start = st.number_input("Offset Start Delay (months)", value=0)

    st.divider()

    st.header("New Loan")
    new_var_rate = st.number_input("Variable Rate", value=5.49)
    new_fixed_rate = st.number_input("Fixed Rate", value=4.99)
    fixed_yrs = st.number_input("Fixed Years", value=2)
    revert_rate = st.number_input("Revert Rate", value=6.35)

    st.divider()
    mode = st.radio("Behaviour Mode", ["Keep term fixed", "Keep payment fixed"])

    monthly_fees = st.number_input("Fees", value=8.0)
    one_time_fees = st.number_input("Upfront Fees", value=299.0)

# ====================== RUN ======================
if loan_left > 0:

    baseline_df, baseline_interest, baseline_paid = simulate_split_loan(
        loan_left, 0, current_rate, 0, 0,
        fixed_yrs, term_months,
        offset_current, monthly_offset_add,
        monthly_fees, one_time_fees
    )

    opt, splits, y_interest, y_debt, x_fit, y_fit_i, y_fit_d = find_optimal_split(
        loan_left, new_var_rate, new_fixed_rate, revert_rate,
        fixed_yrs, term_months,
        offset_current, monthly_offset_add,
        monthly_fees, one_time_fees
    )

    opt_var = loan_left*(1-opt/100)
    opt_fix = loan_left*(opt/100)

    opt_df, opt_interest, opt_paid = simulate_split_loan(
        opt_var, opt_fix, new_var_rate, new_fixed_rate, revert_rate,
        fixed_yrs, term_months,
        offset_current, monthly_offset_add,
        monthly_fees, one_time_fees
    )

    # ====================== TABS ======================
    t1,t2,t3,t4 = st.tabs(["Results","Graphs","Optimisation","3D Surface"])

    with t1:
        c1,c2,c3 = st.columns(3)
        c1.metric("Optimal Split", f"{opt}%")
        c2.metric("Interest", f"${opt_interest:,.0f}")
        c3.metric("Paid", f"${opt_paid:,.0f}")

        # cumulative pairing
        st.subheader("Cumulative Comparison")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=opt_df["month"], y=opt_df["cumulative_interest"], name="Interest"))
        fig.add_trace(go.Scatter(x=opt_df["month"], y=opt_df["cumulative_paid"], name="Paid"))
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        st.subheader("Monthly Pairing (Interest vs Principal)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=opt_df["month"], y=opt_df["interest"], name="Interest"))
        fig.add_trace(go.Scatter(x=opt_df["month"], y=opt_df["principal"], name="Principal"))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Balance Over Time")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=opt_df["month"], y=opt_df["balance"], name="Balance"))
        st.plotly_chart(fig2, use_container_width=True)

    with t3:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=splits, y=y_interest, name="Interest"))
        fig.add_trace(go.Scatter(x=splits, y=y_debt, name="Debt"))
        st.plotly_chart(fig, use_container_width=True)

    with t4:

        x = np.linspace(0,100,20)
        y = np.linspace(12,term_months,20)
        X,Y = np.meshgrid(x,y)
        Z = np.zeros_like(X)

        for i in range(len(x)):
            for j in range(len(y)):
                vp = loan_left*(1-x[i]/100)
                fp = loan_left*(x[i]/100)

                df3,_,_ = simulate_split_loan(
                    vp,fp,new_var_rate,new_fixed_rate,revert_rate,
                    fixed_yrs,int(y[j]),offset_current,monthly_offset_add,
                    monthly_fees,one_time_fees
                )

                Z[j,i] = df3["interest"].sum()

        fig3 = go.Figure(data=[go.Surface(
            x=X, y=Y, z=Z,
            colorscale="Viridis"
        )])

        fig3.update_layout(
            title="Finance Surface",
            scene=dict(
                xaxis_title="Fixed Split %",
                yaxis_title="Loan Term (months)",
                zaxis_title="Total Interest Cost"
            )
        )

        st.plotly_chart(fig3, use_container_width=True)

else:
    st.info("Enter values")

st.caption("Corrected graph pairing • preserved original model • full feature retention")
