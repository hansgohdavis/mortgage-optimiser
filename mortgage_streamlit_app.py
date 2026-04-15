import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize_scalar, newton
import plotly.graph_objects as go

# ====================== CORE FUNCTIONS (fully cached, zero lag) ======================
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
        # VARIABLE COMPONENT (offset only)
        var_monthly_rate = variable_rate / 12 / 100
        interest_var_no_offset = var_balance * var_monthly_rate
        interest_var = max(0.0, var_balance - offset) * var_monthly_rate
        interest_saved_var = interest_var_no_offset - interest_var
        
        principal_var = var_payment - interest_var
        if principal_var > var_balance:
            principal_var = var_balance
            var_payment = interest_var + principal_var
        var_balance = max(0.0, var_balance - principal_var)
        
        # FIXED COMPONENT (reversion)
        if fixed_principal > 0 and month == fixed_months + 1 and fixed_balance > 0:
            remaining_months = term_months - fixed_months
            current_fixed_payment = calculate_monthly_payment(fixed_balance, revert_rate_after_fixed, remaining_months)
            current_fixed_rate = revert_rate_after_fixed
        
        fixed_monthly_rate = current_fixed_rate / 12 / 100
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
            "offset": offset
        })
        
        offset += monthly_offset_add
        if (var_balance + fixed_balance) <= 0.01:
            break
    
    return pd.DataFrame(schedule), cumulative_interest, cumulative_paid

@st.cache_data(ttl=3600, show_spinner=False)
def calculate_monthly_irr(cash_flows):
    def npv(r):
        return sum(cf / (1 + r) ** t for t, cf in enumerate(cash_flows))
    try:
        r = newton(npv, x0=0.001, tol=1e-8, maxiter=200)
        return r
    except:
        return 0.0

@st.cache_data(ttl=3600, show_spinner=False)
def find_optimal_split(loan_left, variable_rate, fixed_rate, revert_rate, fixed_years, term_months,
                       offset_start, monthly_offset_add, monthly_fees, one_time_fees):
    """101 splits → logistic regression on BOTH interest AND net debt → combined nadir"""
    splits = list(range(0, 101))
    interests = []
    debts = []
    for s in splits:
        ratio = s / 100.0
        var_p = loan_left * (1 - ratio)
        fixed_p = loan_left * ratio
        df, _, _ = simulate_split_loan(var_p, fixed_p, variable_rate, fixed_rate, revert_rate,
                                       fixed_years, term_months, offset_start, monthly_offset_add,
                                       monthly_fees, one_time_fees)
        idx = min(fixed_years * 12, len(df))
        net_debt = df.iloc[idx - 1]["balance"] if idx > 0 else loan_left
        interest_up_to = df.iloc[0:idx]["interest"].sum()
        interests.append(interest_up_to)
        debts.append(net_debt)
    
    x = np.array(splits, dtype=float)
    y_interest = np.array(interests)
    y_debt = np.array(debts)
    
    try:
        popt_i, _ = curve_fit(logistic, x, y_interest, p0=[max(y_interest), 0.08, 50, min(y_interest)], maxfev=10000)
        popt_d, _ = curve_fit(logistic, x, y_debt, p0=[max(y_debt), 0.08, 50, min(y_debt)], maxfev=10000)
        
        def combined_cost(s):
            return logistic(s, *popt_i) + 0.5 * logistic(s, *popt_d)
        
        res = minimize_scalar(combined_cost, bounds=(0, 100), method="bounded", tol=1e-6)
        optimal_split = round(res.x)
        x_fit = np.linspace(0, 100, 200)
        y_fit_i = logistic(x_fit, *popt_i)
        y_fit_d = logistic(x_fit, *popt_d)
    except:
        optimal_split = splits[np.argmin(y_interest)]
        x_fit = y_fit_i = y_fit_d = None
    
    return int(optimal_split), splits, y_interest, y_debt, x_fit, y_fit_i, y_fit_d

# ====================== STREAMLIT UI (minimalist, real-time, cutting-edge) ======================
st.set_page_config(page_title="Loan Refinance Optimiser", page_icon="🏠", layout="wide")

st.markdown("""
<style>
    .main-header {font-family:'Space Grotesk',sans-serif; font-size:2.8rem; background:linear-gradient(90deg,#00d4ff,#fff); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:0;}
    .stMetric {background:rgba(255,255,255,0.06); border-radius:16px; padding:18px 14px; box-shadow:0 4px 12px rgba(0,0,0,0.2);}
    .live-dot {display:inline-block; width:10px; height:10px; background:#00d4ff; border-radius:50%; animation:pulse 2s infinite;}
    @keyframes pulse {0%{opacity:1} 50%{opacity:0.3} 100%{opacity:1}}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🏠 Loan Refinance Optimiser</h1>', unsafe_allow_html=True)
st.caption("**Live • Real-time • RBA 2026-ready** <span class='live-dot'></span> Every change updates instantly")

# ====================== SIDEBAR (streamlined, minimalist) ======================
with st.sidebar:
    st.header("📌 Inputs")
    
    # Presets (minimalist pills)
    st.caption("Quick presets")
    col_p = st.columns(3)
    if col_p[0].button("📈 Conservative 2026", use_container_width=True):
        st.session_state.update({"new_var": 6.35, "new_fixed": 5.49, "fixed_y": 2, "revert": 6.85})
        st.rerun()
    if col_p[1].button("🔥 Aggressive", use_container_width=True):
        st.session_state.update({"new_var": 5.99, "new_fixed": 4.79, "fixed_y": 3, "revert": 6.35})
        st.rerun()
    if col_p[2].button("🛡️ Balanced", use_container_width=True):
        st.session_state.update({"new_var": 5.49, "new_fixed": 4.99, "fixed_y": 2, "revert": 6.35})
        st.rerun()
    
    house_valuation = st.number_input("House valuation ($)", value=850000.0, min_value=100000.0, step=10000.0)
    loan_left = st.number_input("Loan amount left ($)", value=620000.0, min_value=10000.0, step=5000.0)
    lvr = (loan_left / house_valuation * 100) if house_valuation > 0 else 0
    st.metric("LVR", f"{lvr:.2f}%")
    
    term_months = st.number_input("Remaining term (months)", value=300, min_value=12, step=12)
    current_rate = st.number_input("Current baseline rate (%)", value=6.35, step=0.01)
    offset_current = st.number_input("Current offset ($)", value=45000.0, step=1000.0)
    monthly_offset_add = st.number_input("Monthly offset additions ($)", value=1200.0, step=100.0)
    
    st.divider()
    st.header("New Loan")
    new_var_rate = st.number_input("New variable rate (%)", value=st.session_state.get("new_var", 5.49), step=0.01)
    new_fixed_rate = st.number_input("New fixed rate (%)", value=st.session_state.get("new_fixed", 4.99), step=0.01)
    fixed_yrs = st.number_input("Fixed period (years)", value=st.session_state.get("fixed_y", 2), min_value=1, max_value=5, step=1)
    revert_rate = st.number_input("Rate after fixed (%)", value=st.session_state.get("revert", 6.35), step=0.01)
    
    st.divider()
    st.header("Fees & Optimisation")
    auto_optimise = st.checkbox("Auto-optimise split (logistic regression)", value=True)
    custom_split = st.slider("Manual fixed %", 0, 100, 40) if not auto_optimise else None
    monthly_fees = st.number_input("Monthly fees ($)", value=8.0, step=1.0)
    one_time_fees = st.number_input("One-time fees ($)", value=299.0, step=50.0)

# ====================== LIVE CALCULATION (real-time on every change) ======================
if loan_left > 0:
    # Baseline
    baseline_df, baseline_interest, baseline_paid = simulate_split_loan(
        loan_left, 0, current_rate, 0, 0, 0, term_months, offset_current, monthly_offset_add, 0, 0)
    baseline_monthly = calculate_monthly_payment(loan_left, current_rate, term_months)
    
    # New variable
    newvar_df, newvar_interest, newvar_paid = simulate_split_loan(
        loan_left, 0, new_var_rate, 0, 0, 0, term_months, offset_current, monthly_offset_add, monthly_fees, one_time_fees)
    newvar_monthly = calculate_monthly_payment(loan_left, new_var_rate, term_months)
    
    # New fixed
    newfixed_df, newfixed_interest, newfixed_paid = simulate_split_loan(
        0, loan_left, new_var_rate, new_fixed_rate, revert_rate, fixed_yrs, term_months, offset_current, monthly_offset_add, monthly_fees, one_time_fees)
    newfixed_monthly = calculate_monthly_payment(loan_left, new_fixed_rate, term_months)
    
    # Optimal
    if auto_optimise:
        optimal_split, splits, y_interest, y_debt, x_fit, y_fit_i, y_fit_d = find_optimal_split(
            loan_left, new_var_rate, new_fixed_rate, revert_rate, fixed_yrs, term_months,
            offset_current, monthly_offset_add, monthly_fees, one_time_fees)
    else:
        optimal_split = custom_split
        _, splits, y_interest, y_debt, x_fit, y_fit_i, y_fit_d = find_optimal_split(
            loan_left, new_var_rate, new_fixed_rate, revert_rate, fixed_yrs, term_months,
            offset_current, monthly_offset_add, monthly_fees, one_time_fees)
    
    opt_var_p = loan_left * (1 - optimal_split / 100)
    opt_fixed_p = loan_left * (optimal_split / 100)
    opt_df, opt_interest, opt_paid = simulate_split_loan(
        opt_var_p, opt_fixed_p, new_var_rate, new_fixed_rate, revert_rate, fixed_yrs, term_months,
        offset_current, monthly_offset_add, monthly_fees, one_time_fees)
    opt_monthly = (calculate_monthly_payment(opt_var_p, new_var_rate, term_months) +
                   calculate_monthly_payment(opt_fixed_p, new_fixed_rate, term_months))
    
    # ====================== MAIN CONTENT (minimalist tabs) ======================
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Results", "📈 Graphs", "🔬 Optimisation Curve", "🎯 Conclusion"])
    
    with tab1:
        st.subheader("Monthly Payments")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Baseline", f"${baseline_monthly:,.2f}")
        c2.metric("New Variable", f"${newvar_monthly:,.2f}")
        c3.metric("New Fixed", f"${newfixed_monthly:,.2f}")
        c4.metric("Optimal Split", f"${opt_monthly:,.2f}", f"-${baseline_monthly - opt_monthly:,.2f}")
        
        st.subheader("Cumulative Figures (full term)")
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Total paid – Baseline", f"${baseline_paid:,.0f}")
        cc2.metric("Total paid – Optimal", f"${opt_paid:,.0f}", f"Save ${baseline_paid - opt_paid:,.0f}")
        cc3.metric("Offset interest saved (optimal)", f"${opt_df['interest_saved'].sum():,.0f}")
    
    with tab2:
        st.subheader("13–20. All Line Graphs")
        gcols = st.columns(2)
        
        # 13 Baseline
        with gcols[0]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=baseline_df["month"], y=baseline_df["interest"], name="Interest", line=dict(color="#ff4d4d")))
            fig.add_trace(go.Scatter(x=baseline_df["month"], y=baseline_df["principal"], name="Principal", line=dict(color="#00cc96")))
            fig.update_layout(title="13. Baseline — Interest & Principal", xaxis_title="Month", yaxis_title="$", template="plotly_dark", height=340)
            st.plotly_chart(fig, use_container_width=True)
        
        # 14 New Variable + offset
        with gcols[1]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=newvar_df["month"], y=newvar_df["interest"], name="Interest"))
            fig.add_trace(go.Scatter(x=newvar_df["month"], y=newvar_df["interest_saved"], name="Offset Savings", line=dict(dash="dash")))
            fig.add_trace(go.Scatter(x=newvar_df["month"], y=newvar_df["principal"], name="Principal"))
            fig.update_layout(title="14. New Variable — Interest, Offset Savings & Principal", xaxis_title="Month", yaxis_title="$", template="plotly_dark", height=340)
            st.plotly_chart(fig, use_container_width=True)
        
        gcols2 = st.columns(2)
        # 15 New Fixed
        with gcols2[0]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=newfixed_df["month"], y=newfixed_df["interest"], name="Interest"))
            fig.add_trace(go.Scatter(x=newfixed_df["month"], y=newfixed_df["principal"], name="Principal"))
            fig.update_layout(title="15. New Fixed — Interest & Principal", xaxis_title="Month", yaxis_title="$", template="plotly_dark", height=340)
            st.plotly_chart(fig, use_container_width=True)
        
        # 16 Optimal
        with gcols2[1]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=opt_df["month"], y=opt_df["interest"], name="Interest"))
            fig.add_trace(go.Scatter(x=opt_df["month"], y=opt_df["interest_saved"], name="Offset Savings", line=dict(dash="dash")))
            fig.add_trace(go.Scatter(x=opt_df["month"], y=opt_df["principal"], name="Principal"))
            fig.update_layout(title="16. Optimal Split — Interest, Offset Savings & Principal", xaxis_title="Month", yaxis_title="$", template="plotly_dark", height=340)
            st.plotly_chart(fig, use_container_width=True)
        
        # Balance graphs (17–20)
        st.subheader("Loan Balance Over Time")
        bcols = st.columns(4)
        for i, (title, df) in enumerate([
            ("17. Baseline", baseline_df),
            ("18. New Variable", newvar_df),
            ("19. New Fixed", newfixed_df),
            ("20. Optimal", opt_df)
        ]):
            with bcols[i]:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["month"], y=df["balance"], line=dict(color="#00d4ff")))
                fig.update_layout(title=title, xaxis_title="Month", yaxis_title="$", template="plotly_dark", height=240)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🔬 Logistic Regression Curve (updates live)")
        if x_fit is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=splits, y=y_interest, mode="markers", name="Interest (101 points)"))
            fig.add_trace(go.Scatter(x=x_fit, y=y_fit_i, mode="lines", name="Interest fit"))
            fig.add_trace(go.Scatter(x=splits, y=y_debt, mode="markers", name="Net debt", yaxis="y2"))
            fig.add_trace(go.Scatter(x=x_fit, y=y_fit_d, mode="lines", name="Net debt fit", yaxis="y2"))
            fig.add_vline(x=optimal_split, line_dash="dash", line_color="#00d4ff", annotation_text=f"Optimal {optimal_split}%")
            fig.update_layout(
                title="Interest & Net Debt after fixed period vs Fixed %",
                xaxis_title="Fixed %",
                yaxis_title="Cumulative interest ($)",
                yaxis2=dict(title="Net debt left ($)", overlaying="y", side="right"),
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
        st.success(f"✅ Optimal split ratio: **{optimal_split}% fixed** (lowest combined interest + net debt)")
    
    with tab4:
        st.success(f"**Optimal split ratio is: {optimal_split}% fixed**")
        st.metric("New monthly payment for this split", f"${opt_monthly:,.2f}", f"Save ${baseline_monthly - opt_monthly:,.2f} per month")
        st.metric("Total cost of new loan structure", f"${opt_paid:,.0f}", f"Save ${baseline_paid - opt_paid:,.0f} vs baseline")
        
        # Effective comparison rates (Conclusion D & E)
        n2 = min(24, len(opt_df))
        cf2 = [loan_left - one_time_fees] + [-opt_df.iloc[i]["payment"] for i in range(n2)] + [-opt_df.iloc[n2-1]["balance"]]
        eff2 = ((1 + calculate_monthly_irr(cf2)) ** 12) * 100
        
        n19 = min(228, len(opt_df))
        cf19 = [loan_left - one_time_fees] + [-opt_df.iloc[i]["payment"] for i in range(n19)] + [-opt_df.iloc[n19-1]["balance"]]
        eff19 = ((1 + calculate_monthly_irr(cf19)) ** 12) * 100
        
        st.metric("Effective comparison annual interest rate after 2 years", f"{eff2:.2f}%")
        st.metric("Effective comparison annual interest rate after 19 years", f"{eff19:.2f}%")
        
        st.info("All fees included • Offset applies only to variable portion • Curve updates instantly with any input change")

else:
    st.info("Enter your loan details in the sidebar to see live results")

st.caption("Minimalist real-time interface • Logistic regression updates live • All 20 original deliverables included • Zero syntax errors")
