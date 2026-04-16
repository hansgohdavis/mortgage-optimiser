"""
Australian Mortgage Refinance Analyser v2.0
Real-time calculations · RBA/ASX data · Enhanced visualisations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, datetime, timedelta
import calendar
import requests
import re, io, json

try:
    from scipy.optimize import brentq
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AU Mortgage Refinance Analyser",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
*, html, body, [class*="css"] { font-family:'Inter',system-ui,sans-serif; }
.stApp { background:#07090f; color:#d4dbe8; }
section[data-testid="stSidebar"] { background:#0b0f1a; }

/* inputs */
.stNumberInput input,.stTextInput input{background:#111827!important;border:1px solid #1e2d4a!important;border-radius:5px!important;color:#d4dbe8!important;font-size:0.875rem!important;}
.stDateInput input{background:#111827!important;border:1px solid #1e2d4a!important;color:#d4dbe8!important;font-size:0.875rem!important;}
.stSelectbox>div>div{background:#111827!important;border:1px solid #1e2d4a!important;color:#d4dbe8!important;}
input:disabled,.stNumberInput input:disabled{background:#0b0f1a!important;color:#4a5568!important;border-color:#0f1929!important;}

/* tabs */
.stTabs [data-baseweb="tab-list"]{background:#0b0f1a;border-bottom:1px solid #1e2d4a;gap:0;padding:0 4px;}
.stTabs [data-baseweb="tab"]{background:transparent;color:#64748b;border-radius:0;padding:10px 16px;font-size:0.78rem;font-weight:500;border-bottom:2px solid transparent;}
.stTabs [aria-selected="true"]{background:transparent!important;color:#4a9af5!important;border-bottom:2px solid #4a9af5!important;}

/* expanders */
.streamlit-expanderHeader{background:#0b0f1a!important;border:1px solid #1e2d4a!important;border-radius:6px!important;color:#d4dbe8!important;font-size:0.875rem!important;font-weight:500!important;}
.streamlit-expanderContent{background:#07090f!important;border:1px solid #1e2d4a!important;border-top:none!important;border-radius:0 0 6px 6px!important;padding:16px!important;}

/* buttons */
.stButton>button{background:#1a2744;color:#d4dbe8;border:1px solid #1e2d4a;border-radius:5px;font-size:0.82rem;font-weight:500;padding:6px 14px;transition:all .15s;}
.stButton>button:hover{background:#243560;border-color:#4a9af5;color:#4a9af5;}
.btn-danger>button{background:transparent!important;border-color:#e94560!important;color:#e94560!important;}
.btn-sm>button{padding:3px 9px!important;font-size:0.73rem!important;}

/* metric cards */
.m-card{background:#0b0f1a;border:1px solid #1e2d4a;border-radius:7px;padding:12px 14px;margin-bottom:5px;}
.m-label{color:#64748b;font-size:0.68rem;font-weight:500;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px;}
.m-value{color:#d4dbe8;font-size:1.25rem;font-weight:600;line-height:1.2;}
.m-diff{color:#64748b;font-size:0.7rem;margin-top:2px;}
.m-diff-pos{color:#30d996;font-size:0.7rem;margin-top:2px;}
.m-diff-neg{color:#e94560;font-size:0.7rem;margin-top:2px;}

/* section titles */
.sec-title{color:#64748b;font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.08em;padding:12px 0 6px;border-bottom:1px solid #1e2d4a;margin-bottom:10px;}

/* list headers */
.list-hdr{color:#64748b;font-size:0.69rem;font-weight:600;text-transform:uppercase;letter-spacing:.04em;padding-bottom:3px;}

/* note / info */
.note{background:#0b0f1a;border-left:2px solid #4a9af5;border-radius:0 4px 4px 0;padding:7px 11px;font-size:0.78rem;color:#64748b;margin:6px 0;}
.note-warn{background:#0b0f1a;border-left:2px solid #f5a94a;border-radius:0 4px 4px 0;padding:7px 11px;font-size:0.78rem;color:#a0896a;margin:6px 0;}
.note-ok{background:#0b0f1a;border-left:2px solid #30d996;border-radius:0 4px 4px 0;padding:7px 11px;font-size:0.78rem;color:#4a9a74;margin:6px 0;}

/* computed fields */
.cf{background:#0d1626;border:1px solid #1e2d4a;border-radius:5px;padding:8px 11px;font-size:0.875rem;margin-bottom:4px;}
.cf-lbl{color:#64748b;font-size:0.69rem;margin-bottom:2px;}
.cf-val{color:#4a9af5;font-weight:600;font-size:1rem;}
.cf-sub{color:#64748b;font-size:0.7rem;margin-top:1px;}

/* rate badge */
.rate-up{color:#e94560;font-weight:600;}
.rate-dn{color:#30d996;font-weight:600;}
.rate-nc{color:#64748b;}

/* RBA/ASX panel */
.data-panel{background:#0b0f1a;border:1px solid #1e2d4a;border-radius:8px;padding:14px 16px;margin-bottom:10px;}
.data-panel-title{color:#4a9af5;font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px;}

/* tooltip wrapper */
.tooltip-wrap{position:relative;display:inline-block;}
.tooltip-wrap .tooltip-txt{visibility:hidden;background:#1a2744;color:#d4dbe8;text-align:left;border-radius:5px;padding:8px 10px;font-size:0.78rem;width:260px;position:absolute;z-index:1000;bottom:125%;left:50%;margin-left:-130px;border:1px solid #1e2d4a;line-height:1.5;}
.tooltip-wrap:hover .tooltip-txt{visibility:visible;}
.tt-icon{color:#4a9af5;cursor:help;font-size:0.78rem;margin-left:4px;}

hr{border-color:#1e2d4a!important;margin:16px 0!important;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

C_ORIG,C_CURR,C_VAR,C_FIX,C_SPLIT="#4a9af5","#f5a94a","#30d996","#e94560","#c47af5"
C_PAPER,C_PLOT,C_GRID,C_TEXT="#0b0f1a","#07090f","#1e2d4a","#d4dbe8"

PLOT_BASE=dict(
    paper_bgcolor=C_PAPER,plot_bgcolor=C_PLOT,
    font=dict(family="Inter",color=C_TEXT,size=11),
    xaxis=dict(gridcolor=C_GRID,zerolinecolor=C_GRID,linecolor=C_GRID),
    yaxis=dict(gridcolor=C_GRID,zerolinecolor=C_GRID,linecolor=C_GRID),
    legend=dict(bgcolor=C_PAPER,bordercolor=C_GRID,borderwidth=1,font=dict(size=11)),
    margin=dict(t=40,b=36,l=60,r=20),
    hovermode="x unified",
    hoverlabel=dict(bgcolor=C_PAPER,bordercolor=C_GRID,font=dict(color=C_TEXT,size=11)),
)

TODAY = date.today()

RATE_TOOLTIPS = {
    "adv_var":  "The headline variable interest rate advertised by the lender, before fees.",
    "comp_var": "ASIC comparison rate: normalises fees into the rate using a $150,000 loan over 25 years (standard benchmark).",
    "eff_var":  "Effective variable rate for your specific loan amount and term — more accurate than the comparison rate for your situation.",
    "adv_fix":  "The headline fixed interest rate advertised by the lender for the fixed period.",
    "comp_fix": "ASIC comparison rate for the fixed component, using the $150,000 / 25-year benchmark.",
    "eff_fix":  "Effective fixed rate for your specific loan amount and term.",
    "rev":      "The variable rate your fixed loan reverts to after the fixed period expires. Defaults to the effective variable rate.",
    "split":    "The proportion allocated to fixed vs variable. Optimal split minimises cumulative interest + closing balance at the end of the fixed period, evaluated in 0.1% increments across all 1,001 combinations.",
    "lvr_orig": "Loan-to-Value Ratio at origination: Original Loan Amount ÷ Original Property Valuation × 100.",
    "lvr_curr": "Current Loan-to-Value Ratio: Current Remaining Balance ÷ Current Property Valuation × 100.",
    "offset":   "Funds in your offset account reduce the balance on which interest is calculated daily. Interest Saved = (Balance × Rate/365 × Days) − (Net Debt × Rate/365 × Days).",
    "comparison_rate_note": "Effective and ASIC comparison rates are calculated automatically from the fees you enter below.",
    "cont":     "When enabled, the current loan is treated as a continuation of the original loan. Current remaining balance and interest rate are auto-filled from the original loan details.",
}

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def fc(v,d=0):
    if v is None or (isinstance(v,float) and np.isnan(v)): return "N/A"
    return ("-" if v<0 else "") + f"${abs(v):,.{d}f}"

def fp(v,d=2):
    if v is None or (isinstance(v,float) and np.isnan(v)): return "N/A"
    return f"{v:.{d}f}%"

def parse_dt(v)->date:
    if isinstance(v,date): return v
    if isinstance(v,datetime): return v.date()
    try: return datetime.strptime(str(v),"%Y-%m-%d").date()
    except: return TODAY

def months_between(d1:date,d2:date)->int:
    return (d2.year-d1.year)*12+(d2.month-d1.month)

def add_months(d:date,n:int)->date:
    m=d.month-1+n; y=d.year+m//12; mo=m%12+1
    return date(y,mo,min(d.day,calendar.monthrange(y,mo)[1]))

def tt(tip_key:str)->str:
    tip = RATE_TOOLTIPS.get(tip_key,"")
    return f'<span class="tooltip-wrap"><span class="tt-icon">?</span><span class="tooltip-txt">{tip}</span></span>'

def sec(t:str):
    st.markdown(f'<div class="sec-title">{t}</div>',unsafe_allow_html=True)

def computed(lbl:str,val:str,sub:str=""):
    s = f'<div class="cf-sub">{sub}</div>' if sub else ""
    st.markdown(f'<div class="cf"><div class="cf-lbl">{lbl}</div><div class="cf-val">{val}</div>{s}</div>',unsafe_allow_html=True)

def metric_card(lbl:str,val:str,diff:str="",diff_pos:bool=True,diff_neutral:bool=False):
    if diff_neutral or not diff:
        dc,ds = "m-diff",diff
    elif diff_pos:
        dc,ds = "m-diff-pos",diff
    else:
        dc,ds = "m-diff-neg",diff
    dh = f'<div class="{dc}">{ds}</div>' if diff else ""
    return f'<div class="m-card"><div class="m-label">{lbl}</div><div class="m-value">{val}</div>{dh}</div>'

# ═══════════════════════════════════════════════════════════════════════════════
# FINANCIAL ENGINE  (cached for real-time performance)
# ═══════════════════════════════════════════════════════════════════════════════

def calc_payment(principal:float,rate_pct:float,term_mo:int)->float:
    if principal<=0 or term_mo<=0: return 0.0
    if rate_pct<=0: return principal/term_mo
    r=rate_pct/100/12
    return principal*r*(1+r)**term_mo/((1+r)**term_mo-1)

def build_rate_schedule(base:float,deltas:list)->list:
    sched=[(date(1900,1,1),base)]; cum=base
    for row in sorted([(parse_dt(r[0]),float(r[1])) for r in deltas if r[0]],key=lambda x:x[0]):
        cum=round(cum+row[1],4); sched.append((row[0],cum))
    return sched

def get_rate(sched:list,d:date)->float:
    r=sched[0][1]
    for rd,rv in sched:
        if rd<=d: r=rv
        else: break
    return r

@st.cache_data(show_spinner=False)
def amortize_cached(
    principal:float, start_str:str, term_mo:int,
    rate_sched_t:tuple,   # ((date_str,rate), ...)
    off_init:float, off_monthly:float, off_lumps_t:tuple,  # ((mo_offset,amount),...)
    monthly_fee:float, maintain_pmt:bool
)->pd.DataFrame:
    if principal<=0 or term_mo<=0: return pd.DataFrame()
    start=date.fromisoformat(start_str)
    rate_sched=[(date.fromisoformat(ds),r) for ds,r in rate_sched_t]

    # Build monthly offset array
    def offset_at(mo:int)->float:
        bal = off_init + mo * off_monthly
        for lmo,amt in off_lumps_t:
            if lmo <= mo: bal += amt
        return max(0.0,bal)

    bal=principal
    prev_rate=get_rate(rate_sched,start)
    cur_pmt=calc_payment(principal,prev_rate,term_mo)
    cum_int=cum_paid=cum_saved=0.0
    rows=[]; period=start

    for mo in range(1,term_mo+121):
        if bal<=0.01: break
        next_d=add_months(period,1)
        days=(next_d-period).days
        ann_rate=get_rate(rate_sched,period)
        if abs(ann_rate-prev_rate)>1e-9:
            rem=max(1,term_mo-mo+1); req=calc_payment(bal,ann_rate,rem)
            if ann_rate>prev_rate: cur_pmt=req
            else: cur_pmt=max(cur_pmt,req) if maintain_pmt else req
            prev_rate=ann_rate
        avg_off=min(offset_at(mo-1),bal)
        net=max(0.0,bal-avg_off)
        dr=ann_rate/100/365
        interest=net*dr*days; saved=(bal*dr*days)-interest
        cum_int+=interest; cum_saved+=saved
        opening=bal; bal+=interest
        pmt=min(cur_pmt,bal); pp=max(0.0,pmt-interest); bal=max(0.0,bal-pmt)
        cum_paid+=pmt+monthly_fee
        rows.append({"Month":mo,"Date":next_d,"Opening Balance":opening,
                     "Avg Offset":avg_off,"Net Debt":net,"Rate %":ann_rate,
                     "Interest":interest,"Interest Saved":saved,"Principal":pp,
                     "Fees":monthly_fee,"Payment":pmt,"Closing Balance":bal,
                     "Cum Interest":cum_int,"Cum Paid":cum_paid,"Cum Interest Saved":cum_saved})
        period=next_d
    return pd.DataFrame(rows)

def amortize(principal,start,term_mo,rate_sched,off_init,off_monthly,off_lumps_t,
             monthly_fee,maintain_pmt)->pd.DataFrame:
    """Wrapper converts types for cached function."""
    rs_t = tuple((d.isoformat(),r) for d,r in rate_sched)
    return amortize_cached(principal,start.isoformat(),term_mo,rs_t,
                           off_init,off_monthly,off_lumps_t,monthly_fee,maintain_pmt)

def fast_partial(principal,rate,total,n):
    if principal<=0 or n<=0: return 0.0,0.0
    n=min(n,total)
    if rate<=0:
        pmt=principal/total; return max(0.0,principal-pmt*n),0.0
    r=rate/100/12
    pmt=principal*r*(1+r)**total/((1+r)**total-1)
    bal=principal*(1+r)**n-pmt*((1+r)**n-1)/r
    bal=max(0.0,bal)
    return bal,max(0.0,pmt*n-(principal-bal))

@st.cache_data(show_spinner=False)
def calc_optimal_split(loan,var_r,fix_r,rev_r,fix_yrs,total_mo):
    n_fix=min(fix_yrs*12,total_mo)
    best_obj,best_pct=float("inf"),50.0
    rows=[]
    for i in range(1001):
        pct_f=i/10.0
        p_f=loan*pct_f/100; p_v=loan*(100-pct_f)/100
        bf,cif=fast_partial(p_f,fix_r,total_mo,n_fix) if p_f>0 else (0.0,0.0)
        bv,civ=fast_partial(p_v,var_r,total_mo,n_fix) if p_v>0 else (0.0,0.0)
        obj=(cif+civ)+(bf+bv)
        if obj<best_obj: best_obj,best_pct=obj,pct_f
        rows.append({"Fixed %":pct_f,"Variable %":100-pct_f,
                     "Cum Interest":cif+civ,"End Balance":bf+bv,"Objective":obj})
    return best_pct,pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def comparison_rate_asic(setup_fee,monthly_fee,rate_pct):
    pv,n=150_000.0,300
    pmt=calc_payment(pv,rate_pct,n)+monthly_fee; target=pv-setup_fee
    if not HAS_SCIPY or target<=0: return rate_pct
    def f(i):
        if abs(i)<1e-12: return pmt*n-target
        return pmt*(1-(1+i)**-n)/i-target
    try: return brentq(f,1e-8,0.5)*12*100
    except: return rate_pct

@st.cache_data(show_spinner=False)
def effective_rate_calc(loan,setup_fee,monthly_fee,rate_pct,term):
    if not HAS_SCIPY or loan<=0: return rate_pct
    pmt=calc_payment(loan,rate_pct,term)+monthly_fee; target=loan-setup_fee
    if target<=0: return rate_pct
    def f(i):
        if abs(i)<1e-12: return pmt*term-target
        return pmt*(1-(1+i)**-term)/i-target
    try: return brentq(f,1e-8,0.5)*12*100
    except: return rate_pct

def merge_schedules(df_v,df_f):
    if df_v is None or df_v.empty: return df_f.copy() if df_f is not None and not df_f.empty else pd.DataFrame()
    if df_f is None or df_f.empty: return df_v.copy()
    n=max(len(df_v),len(df_f))
    num_cols=["Opening Balance","Avg Offset","Net Debt","Interest","Interest Saved",
              "Principal","Fees","Payment","Closing Balance","Cum Interest","Cum Paid","Cum Interest Saved"]
    rows=[]
    for i in range(n):
        rv=df_v.iloc[i].to_dict() if i<len(df_v) else None
        rf=df_f.iloc[i].to_dict() if i<len(df_f) else None
        if rv and rf:
            row={"Month":rv["Month"],"Date":rv["Date"]}
            for c in num_cols: row[c]=rv.get(c,0.0)+rf.get(c,0.0)
            ob=rv.get("Opening Balance",0)+rf.get("Opening Balance",0)
            row["Rate %"]=(rv["Rate %"]*rv.get("Opening Balance",0)+rf["Rate %"]*rf.get("Opening Balance",0))/ob if ob>0 else 0.0
            rows.append(row)
        else:
            r=(rv or rf).copy(); r["Month"]=i+1; rows.append(r)
    return pd.DataFrame(rows)

def eff_rate_from_deltas(base,deltas,as_of=None):
    r=base
    for row in sorted([(parse_dt(row[0]),float(row[1])) for row in deltas if row[0]],key=lambda x:x[0]):
        if as_of is None or row[0]<=as_of: r=round(r+row[1],4)
    return r

def deltas_to_lumps_t(lumps:list,loan_start:date)->tuple:
    """Convert [(date,amount)] to ((month_offset, amount), ...) relative to loan_start."""
    result=[]
    for row in lumps:
        d=parse_dt(row[0]); mo=months_between(loan_start,d)
        if mo>=0: result.append((mo,float(row[1])))
    return tuple(sorted(result))

# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCHERS  (cached, auto-called on load)
# ═══════════════════════════════════════════════════════════════════════════════

HDRS={"User-Agent":"Mozilla/5.0 (compatible; AU-Mortgage-Analyser/2.0)",
      "Accept":"text/html,text/csv,application/json,*/*"}

@st.cache_data(ttl=3600,show_spinner=False)
def fetch_rba_rate_cached()->float|None:
    # F1 CSV
    try:
        r=requests.get("https://www.rba.gov.au/statistics/tables/csv/f1-data.csv",headers=HDRS,timeout=10)
        if r.status_code==200:
            lines=r.text.split("\n"); tcol=None; ds=None
            for li,line in enumerate(lines):
                if "Cash Rate Target" in line:
                    for ci,p in enumerate(line.split(",")):
                        if "Cash Rate Target" in p: tcol=ci; break
                    ds=li+1; break
            if tcol is not None:
                for line in reversed(lines[ds or 0:]):
                    parts=line.split(",")
                    if len(parts)>tcol:
                        try:
                            v=float(parts[tcol].strip().strip('"'))
                            if 0<v<30: return v
                        except: pass
    except: pass
    # Scrape cash-rate page
    try:
        r=requests.get("https://www.rba.gov.au/statistics/cash-rate/",headers=HDRS,timeout=10)
        if r.status_code==200:
            for pat in [r"(\d+\.\d{2})\s*per cent",r"(\d+\.\d{2})%"]:
                for m in re.findall(pat,r.text,re.IGNORECASE):
                    try:
                        v=float(m)
                        if 0<v<20: return v
                    except: pass
    except: pass
    return None

@st.cache_data(ttl=3600,show_spinner=False)
def fetch_rba_history()->list:
    """
    Returns list of {"date": date, "rate": float, "delta": float} sorted ascending.
    """
    records=[]
    try:
        r=requests.get("https://www.rba.gov.au/statistics/tables/csv/f1-data.csv",headers=HDRS,timeout=12)
        if r.status_code==200:
            lines=r.text.split("\n"); tcol=None; dcol=0; ds=None
            for li,line in enumerate(lines):
                if "Cash Rate Target" in line:
                    for ci,p in enumerate(line.split(",")): 
                        if "Cash Rate Target" in p: tcol=ci; break
                    ds=li+1; break
            if tcol is not None and ds is not None:
                prev_rate=None
                for line in lines[ds:]:
                    parts=line.split(",")
                    if len(parts)<=max(dcol,tcol): continue
                    try:
                        dstr=parts[dcol].strip().strip('"')
                        rstr=parts[tcol].strip().strip('"')
                        if not dstr or not rstr: continue
                        dt=datetime.strptime(dstr,"%b-%Y").date() if "-" in dstr else date.fromisoformat(dstr)
                        rv=float(rstr)
                        if not(0<rv<30): continue
                        if prev_rate is None or abs(rv-prev_rate)>0.001:
                            delta=round(rv-(prev_rate or rv),4) if prev_rate is not None else 0.0
                            records.append({"date":dt,"rate":rv,"delta":delta})
                            prev_rate=rv
                    except: pass
    except: pass

    if not records:
        # Fallback: scrape cash-rate page table
        try:
            r=requests.get("https://www.rba.gov.au/statistics/cash-rate/",headers=HDRS,timeout=10)
            if r.status_code==200:
                # Find table rows with date and rate
                rows=re.findall(r'<tr[^>]*>.*?</tr>',r.text,re.DOTALL)
                prev=None
                for row in rows:
                    cells=re.findall(r'<td[^>]*>(.*?)</td>',row,re.DOTALL)
                    cells=[re.sub(r'<[^>]+>','',c).strip() for c in cells]
                    if len(cells)>=2:
                        try:
                            dt=datetime.strptime(cells[0],"%d %b %Y").date()
                            rv=float(cells[1].replace('%','').strip())
                            if 0<rv<30:
                                delta=round(rv-(prev or rv),4) if prev is not None else 0.0
                                records.append({"date":dt,"rate":rv,"delta":delta})
                                prev=rv
                        except: pass
        except: pass
    return sorted(records,key=lambda x:x["date"])

@st.cache_data(ttl=3600,show_spinner=False)
def fetch_rba_next_meeting()->str|None:
    try:
        r=requests.get(
            "https://www.rba.gov.au/schedules-events/calendar/?topics=monetary-policy-board&view=list",
            headers=HDRS,timeout=10)
        if r.status_code==200:
            text=r.text
            # Look for date patterns near "Monetary Policy Board"
            patterns=[
                r'(\d{1,2}\s+\w+\s+\d{4})',
                r'(\w+\s+\d{1,2},?\s+\d{4})',
            ]
            for pat in patterns:
                matches=re.findall(pat,text)
                for m in matches:
                    try:
                        for fmt in ["%d %B %Y","%d %b %Y","%B %d %Y","%B %d, %Y"]:
                            try:
                                dt=datetime.strptime(m.strip(),fmt).date()
                                if dt>=TODAY: return dt.strftime("%d %B %Y")
                            except: pass
                    except: pass
    except: pass
    return None

@st.cache_data(ttl=300,show_spinner=False)
def fetch_asx_rba_data()->dict:
    """
    Attempts to fetch ASX 30-day interbank cash rate futures data.
    Returns dict with available fields, or empty dict on failure.
    ASX pages are JavaScript-rendered; direct fetch returns shell HTML only.
    """
    result={}
    # Attempt 1: ASX market data JSON endpoint for IB futures
    try:
        endpoints=[
            "https://www.asx.com.au/asx/1/exchange/IB/prices",
            "https://www.asx.com.au/data/trendlens_snapshot/IB.json",
        ]
        for url in endpoints:
            r=requests.get(url,headers=HDRS,timeout=6)
            if r.status_code==200:
                data=r.json()
                result["raw"]=data; result["source"]=url
                break
    except: pass

    # Attempt 2: Parse embedded JSON from rate tracker page
    if not result:
        try:
            r=requests.get("https://www.asx.com.au/markets/trade-our-derivatives-market/futures-market/rba-rate-tracker",
                          headers=HDRS,timeout=8)
            if r.status_code==200:
                # Look for JSON data blocks
                json_blks=re.findall(r'window\.__INITIAL_STATE__\s*=\s*({.*?});',r.text,re.DOTALL)
                if not json_blks:
                    json_blks=re.findall(r'window\.chartData\s*=\s*(\[.*?\]);',r.text,re.DOTALL)
                for blk in json_blks:
                    try:
                        data=json.loads(blk)
                        result["page_data"]=data; break
                    except: pass
        except: pass

    return result

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

def _d(k,v):
    if k not in st.session_state: st.session_state[k]=v

def init_state():
    # ── Original Loan ──────────────────────────────────────────────────────
    _d("o_prop_val",800_000.0); _d("o_prop_date",date(2020,1,15))
    _d("o_loan_amt",640_000.0); _d("o_loan_date",date(2020,1,15))
    _d("o_use_dates",False);    _d("o_end_date",date(2045,1,15))
    _d("o_term_mo",300);        _d("o_balance",580_000.0)
    _d("o_balance_date",TODAY); _d("o_rate",6.50)
    _d("o_rate_deltas",[]);     _d("o_off_init",0.0)
    _d("o_off_date",TODAY);     _d("o_off_monthly",0.0)
    _d("o_off_lumps",[]);       _d("o_fee_mo",0.0)
    _d("o_fee_setup",0.0);      _d("o_fee_break",0.0);  _d("o_fee_other",0.0)
    _d("o_rba_autofill",False)  # has RBA history been auto-filled?

    # ── Current Loan ───────────────────────────────────────────────────────
    _d("c_is_cont",True);       _d("c_prop_val",800_000.0)
    _d("c_prop_date",TODAY);    _d("c_balance",580_000.0)
    _d("c_rate",6.50);          _d("c_use_dates",False)
    _d("c_end_date",date(2045,1,15)); _d("c_term_mo",300)
    _d("c_rate_deltas",[]);     _d("c_off_init",0.0)
    _d("c_off_date",TODAY);     _d("c_off_monthly",0.0)
    _d("c_off_lumps",[]);       _d("c_fee_mo",10.0)
    _d("c_fee_setup",0.0);      _d("c_fee_other",0.0)

    # ── Shared anticipated / future rate changes ───────────────────────────
    # Used by both Current Loan "Anticipated Rate Changes"
    # and Proposed Loan "Variable Rate Changes" (always in sync)
    _d("future_var_deltas",[])

    # ── Proposed Loan ──────────────────────────────────────────────────────
    _d("p_auto_amount",True);   _d("p_loan_amt",580_000.0)
    _d("p_start_date",TODAY);   _d("p_use_dates",False)
    _d("p_end_date",add_months(TODAY,300)); _d("p_term_mo",300)
    _d("p_adv_var_rate",6.20);  _d("p_adv_fix_rate",5.89)
    _d("p_fix_yrs",3);          _d("p_rev_rate_override",False)
    _d("p_rev_rate",6.20);      _d("p_split_auto",True)
    _d("p_split_pct",50.0);     _d("p_off_init",0.0)
    _d("p_off_date",TODAY);     _d("p_off_monthly",0.0)
    _d("p_off_lumps",[]);       _d("p_fee_mo",10.0)
    _d("p_fee_setup",800.0);    _d("p_fee_break",0.0);  _d("p_fee_other",0.0)

    # ── Strategy ───────────────────────────────────────────────────────────
    _d("strategy","Balanced");  _d("maintain_pmt",True)
    _d("rba_bps",0)

    # ── Live data cache (populated on first render) ────────────────────────
    _d("_rba_rate",None);       _d("_rba_history",[])
    _d("_rba_next_meeting",None); _d("_asx_data",{})
    _d("_data_loaded",False)

def load_live_data():
    """Auto-fetch all live data on first load."""
    ss=st.session_state
    if ss._data_loaded: return
    with st.spinner("Loading live RBA and market data..."):
        ss._rba_rate    = fetch_rba_rate_cached()
        ss._rba_history = fetch_rba_history()
        ss._rba_next_meeting = fetch_rba_next_meeting()
        ss._asx_data    = fetch_asx_rba_data()
    ss._data_loaded = True

def auto_fill_rba_history():
    """
    Populate o_rate_deltas with RBA historical changes since loan start date,
    provided the user hasn't manually edited the list and the flag isn't set.
    """
    ss=st.session_state
    if ss.o_rba_autofill: return
    if not ss._rba_history: return
    loan_start=parse_dt(ss.o_loan_date)
    loan_rate=ss.o_rate
    # Find base rate at loan start from history
    base_rate=loan_rate
    for rec in ss._rba_history:
        if rec["date"]<=loan_start: base_rate=rec["rate"]
    # Build deltas for changes AFTER loan start
    deltas=[]
    prev=base_rate
    for rec in ss._rba_history:
        if rec["date"]>loan_start:
            delta=round(rec["rate"]-prev,4)
            if abs(delta)>0.001: deltas.append([rec["date"],delta])
            prev=rec["rate"]
    if deltas:
        ss.o_rate_deltas=deltas
        ss.o_rba_autofill=True

# ═══════════════════════════════════════════════════════════════════════════════
# DYNAMIC LIST WIDGETS
# ═══════════════════════════════════════════════════════════════════════════════

def rate_delta_list(state_key:str,base_rate:float,title:str,max_rows:int=20,
                    show_autofill:bool=False,autofill_data:list=None):
    ss=st.session_state
    if state_key not in ss: ss[state_key]=[]

    sec(title)
    cols=st.columns([1,1,1]) if show_autofill else st.columns([1,1,3])
    with cols[0]:
        if st.button("Add change",key=f"add_{state_key}") and len(ss[state_key])<max_rows:
            ss[state_key].append([TODAY,0.25])
    with cols[1]:
        if ss[state_key] and st.button("Clear",key=f"clr_{state_key}"):
            ss[state_key]=[]; st.rerun()
    if show_autofill and autofill_data is not None:
        with cols[2]:
            if st.button("Auto-fill from RBA",key=f"auto_{state_key}",
                         help="Populate from official RBA cash rate history"):
                ss[state_key]=[[r["date"],r["delta"]] for r in autofill_data if r["delta"]!=0]
                st.rerun()

    if ss[state_key]:
        h=st.columns([2,1.6,2.2,0.5])
        for i,lbl in enumerate(["Effective Date","Change (±%)","Result"]):
            h[i].markdown(f'<div class="list-hdr">{lbl}</div>',unsafe_allow_html=True)
        cum=base_rate; to_del=None
        for i,row in enumerate(ss[state_key]):
            c1,c2,c3,c4=st.columns([2,1.6,2.2,0.5])
            with c1:
                nd=st.date_input(f"Rate date {i+1}",value=parse_dt(row[0]),
                                 key=f"{state_key}_d_{i}",label_visibility="collapsed")
            with c2:
                nv=st.number_input(f"Rate delta {i+1}",value=float(row[1]),
                                   min_value=-10.0,max_value=10.0,step=0.25,format="%.2f",
                                   key=f"{state_key}_v_{i}",label_visibility="collapsed")
            res=round(cum+nv,4)
            with c3:
                sign="+" if nv>=0 else ""; clr="#e94560" if nv>0 else ("#30d996" if nv<0 else "#64748b")
                st.markdown(
                    f'<div style="padding:8px 3px;color:{clr};font-size:.875rem;font-weight:500">'
                    f'{sign}{nv:.2f}% → <strong>{res:.2f}%</strong></div>',
                    unsafe_allow_html=True)
            with c4:
                if st.button("✕",key=f"{state_key}_del_{i}"): to_del=i
            ss[state_key][i]=[nd,nv]; cum=res
        if to_del is not None: ss[state_key].pop(to_del); st.rerun()

def lump_list(state_key:str,title:str,max_rows:int=100):
    ss=st.session_state
    if state_key not in ss: ss[state_key]=[]
    sec(title)
    c1,c2=st.columns(2)
    with c1:
        if st.button("Add lump sum",key=f"add_{state_key}") and len(ss[state_key])<max_rows:
            ss[state_key].append([TODAY,0.0])
    with c2:
        if ss[state_key] and st.button("Clear",key=f"clr_{state_key}"):
            ss[state_key]=[]; st.rerun()
    if ss[state_key]:
        h=st.columns([2,2,0.5])
        h[0].markdown('<div class="list-hdr">Date</div>',unsafe_allow_html=True)
        h[1].markdown('<div class="list-hdr">Amount ($)</div>',unsafe_allow_html=True)
        to_del=None
        for i,row in enumerate(ss[state_key]):
            c1,c2,c3=st.columns([2,2,0.5])
            with c1:
                nd=st.date_input(f"Lump date {i+1}",value=parse_dt(row[0]),
                                 key=f"{state_key}_d_{i}",label_visibility="collapsed")
            with c2:
                na=st.number_input(f"Lump amt {i+1}",value=float(row[1]),step=1_000.0,
                                   key=f"{state_key}_a_{i}",label_visibility="collapsed")
            with c3:
                if st.button("✕",key=f"{state_key}_del_{i}"): to_del=i
            ss[state_key][i]=[nd,na]
        if to_del is not None: ss[state_key].pop(to_del); st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# RBA / ASX STATUS PANEL
# ═══════════════════════════════════════════════════════════════════════════════

def render_rba_panel():
    ss=st.session_state
    rate=ss._rba_rate; meeting=ss._rba_next_meeting

    with st.expander("RBA Cash Rate and Market Indicators", expanded=False):
        c1,c2,c3=st.columns(3)
        with c1:
            st.markdown('<div class="data-panel">'
                        '<div class="data-panel-title">Current RBA Cash Rate</div>',
                        unsafe_allow_html=True)
            if rate:
                st.markdown(f'<div style="font-size:2rem;font-weight:700;color:#4a9af5">{fp(rate)}</div>',
                           unsafe_allow_html=True)
                history=ss._rba_history
                if len(history)>=2:
                    last_chg=history[-1]["delta"]
                    d_str=history[-1]["date"].strftime("%d %b %Y")
                    if last_chg>0:
                        st.markdown(f'<div class="rate-up">▲ +{last_chg:.2f}% on {d_str}</div>',unsafe_allow_html=True)
                    elif last_chg<0:
                        st.markdown(f'<div class="rate-dn">▼ {last_chg:.2f}% on {d_str}</div>',unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="rate-nc">Unchanged as of {d_str}</div>',unsafe_allow_html=True)
            else:
                st.markdown('<div style="color:#64748b">Unable to fetch — check rba.gov.au</div>',unsafe_allow_html=True)
            st.markdown('</div>',unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="data-panel"><div class="data-panel-title">Next Monetary Policy Decision</div>',
                       unsafe_allow_html=True)
            if meeting:
                st.markdown(f'<div style="font-size:1.1rem;font-weight:600;color:#d4dbe8">{meeting}</div>',
                           unsafe_allow_html=True)
                # Days until meeting
                try:
                    dt=datetime.strptime(meeting,"%d %B %Y").date()
                    days=(dt-TODAY).days
                    if days>0:
                        st.markdown(f'<div style="color:#64748b;font-size:0.8rem">{days} days from today</div>',
                                   unsafe_allow_html=True)
                except: pass
            else:
                st.markdown('<div style="color:#64748b">Source: rba.gov.au — check for updates</div>',unsafe_allow_html=True)
            st.markdown('</div>',unsafe_allow_html=True)

        with c3:
            st.markdown('<div class="data-panel"><div class="data-panel-title">ASX Rate Tracker</div>',
                       unsafe_allow_html=True)
            asx=ss._asx_data
            if asx:
                st.markdown('<div style="color:#30d996;font-size:0.82rem">Data retrieved</div>',
                           unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div style="color:#64748b;font-size:0.78rem">ASX pages use JavaScript rendering — '
                    'live data unavailable via direct fetch. Use the calculator below for probability estimates, '
                    'or visit <a href="https://www.asx.com.au/markets/trade-our-derivatives-market/futures-market/rba-rate-tracker" '
                    'target="_blank" style="color:#4a9af5">ASX RBA Rate Tracker</a></div>',
                    unsafe_allow_html=True)
            st.markdown('</div>',unsafe_allow_html=True)

        # ASX Probability Calculator
        st.markdown("<hr>",unsafe_allow_html=True)
        st.markdown(
            '<div style="color:#64748b;font-size:0.72rem;font-weight:600;text-transform:uppercase;'
            'letter-spacing:.06em;margin-bottom:10px">ASX Target Rate Probability (30-Day Interbank Cash Rate Futures)</div>',
            unsafe_allow_html=True)

        c1,c2,c3,c4=st.columns(4)
        with c1:
            ib_price=st.number_input("IB Futures Price",value=95.65,min_value=90.0,max_value=100.0,
                                     step=0.01,format="%.2f",
                                     help="30-Day Interbank Cash Rate Futures price from ASX (e.g. 95.65 → implied rate 4.35%)")
            implied_yield=round(100-ib_price,4)
            computed("Implied Yield",fp(implied_yield),"= 100 − Futures Price")
        with c2:
            rt=st.number_input("Current Target Rate (%)",value=rate if rate else 4.35,
                               min_value=0.0,max_value=30.0,step=0.25,format="%.2f",
                               help="Current RBA Target Cash Rate (less any overnight differential)")
        with c3:
            rt1=st.number_input("Expected New Rate (%)",value=round((rate or 4.35)-0.25,2),
                                min_value=0.0,max_value=30.0,step=0.25,format="%.2f",
                                help="Expected new Target Cash Rate if RBA acts (typically ±0.25%)")
        with c4:
            nb_days=st.number_input("Days before RBA meeting",value=5,min_value=1,max_value=30,
                                    help="Number of days in the current month BEFORE the RBA board meeting")

        # Solve for p
        try:
            days_in_month=30
            nb=nb_days/days_in_month; na=(days_in_month-nb_days)/days_in_month
            X=implied_yield; denom=na*(rt1-rt)
            if abs(denom)>1e-9:
                p=round((X-rt*(nb+na))/denom,4)
                p=max(0.0,min(1.0,p))
                pct=p*100
                col_a,col_b,col_c=st.columns(3)
                with col_a:
                    trend_color="#e94560" if rt1>rt else "#30d996"
                    direction="increase" if rt1>rt else "decrease"
                    st.markdown(
                        f'<div class="cf"><div class="cf-lbl">Probability of rate {direction}</div>'
                        f'<div class="cf-val" style="color:{trend_color}">{pct:.1f}%</div>'
                        f'<div class="cf-sub">Formula: p = (X − rt(nb+na)) / (na×(r(t+1)−rt))</div></div>',
                        unsafe_allow_html=True)
                with col_b:
                    outlook="Increasing" if (pct>60 and rt1>rt) else ("Decreasing" if (pct>60 and rt1<rt) else "Stable / Uncertain")
                    oclr={"Increasing":"#e94560","Decreasing":"#30d996","Stable / Uncertain":"#64748b"}[outlook]
                    st.markdown(
                        f'<div class="cf"><div class="cf-lbl">Rate Outlook</div>'
                        f'<div class="cf-val" style="color:{oclr}">{outlook}</div></div>',
                        unsafe_allow_html=True)
                with col_c:
                    st.markdown(
                        f'<div class="cf"><div class="cf-lbl">nb (known) / na (unknown)</div>'
                        f'<div class="cf-val">{nb:.2f} / {na:.2f}</div>'
                        f'<div class="cf-sub">of month</div></div>',
                        unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Calculation error: {e}")

        # Rate history chart
        if ss._rba_history:
            hist=ss._rba_history[-40:] if len(ss._rba_history)>40 else ss._rba_history
            fig=go.Figure()
            fig.add_trace(go.Scatter(
                x=[r["date"] for r in hist], y=[r["rate"] for r in hist],
                name="Cash Rate Target", line=dict(color=C_ORIG,width=2,shape="hv"),
                fill="tozeroy",fillcolor="rgba(74,154,245,0.06)",
                hovertemplate="<b>%{y:.2f}%</b><br>%{x|%d %b %Y}<extra></extra>"))
            fig.update_layout(**PLOT_BASE,title="RBA Cash Rate History",yaxis_title="Rate (%)")
            fig.update_xaxes(rangeslider=dict(visible=True,thickness=0.04))
            st.plotly_chart(fig,use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# INPUT SECTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def section_original():
    ss=st.session_state

    # ── Auto-fill RBA history into rate changes ──
    auto_fill_rba_history()

    # ── Filter RBA history after loan start ──
    loan_start=parse_dt(ss.o_loan_date)
    rba_after_start=[r for r in ss._rba_history if r["date"]>loan_start and abs(r["delta"])>0.001]

    sec("Property")
    c1,c2,c3=st.columns(3)
    with c1:
        ss.o_prop_val=st.number_input("Property Valuation ($)",value=ss.o_prop_val,min_value=0.0,step=10_000.0,
            key="w_o_pv",help="The assessed value of the property at time of loan origination.")
    with c2:
        ss.o_prop_date=st.date_input("Valuation Date",value=ss.o_prop_date,key="w_o_pd")
    with c3:
        if ss.o_prop_val>0 and ss.o_loan_amt>0:
            computed("Original LVR "+tt("lvr_orig"),fp(ss.o_loan_amt/ss.o_prop_val*100),
                     "Loan Amount ÷ Valuation")
        else: computed("Original LVR","—")

    sec("Loan Details")
    c1,c2,c3=st.columns(3)
    with c1:
        ss.o_loan_amt=st.number_input("Original Loan Amount ($)",value=ss.o_loan_amt,min_value=0.0,step=10_000.0,key="w_o_la")
    with c2:
        ss.o_balance=st.number_input("Remaining Balance ($)",value=ss.o_balance,min_value=0.0,step=1_000.0,
            key="w_o_bal",help="Outstanding principal as at the balance date below.")
    with c3:
        ss.o_balance_date=st.date_input("Balance As At",value=ss.o_balance_date,key="w_o_bd")

    c1,c2=st.columns(2)
    with c1:
        ss.o_rate=st.number_input("Original Interest Rate (% p.a.)",value=ss.o_rate,min_value=0.0,max_value=30.0,
            step=0.01,format="%.4f",key="w_o_r",
            help="The rate at inception. Use the Rate Changes section below to track changes over time.")
    with c2:
        ss.o_loan_date=st.date_input("Loan Start Date",value=ss.o_loan_date,key="w_o_ld",
            help="Used to filter historical RBA rate changes relevant to this loan.")

    sec("Loan Term")
    ss.o_use_dates=st.toggle("Calculate term from start and end dates",value=ss.o_use_dates,key="w_o_ud")
    if ss.o_use_dates:
        c1,c2,c3=st.columns(3)
        with c1:
            ss.o_end_date=st.date_input("Loan End Date",value=ss.o_end_date,key="w_o_ed")
        with c2:
            if ss.o_end_date>ss.o_loan_date:
                ss.o_term_mo=months_between(ss.o_loan_date,ss.o_end_date)
                computed("Calculated Term",f"{ss.o_term_mo} months",f"{ss.o_term_mo/12:.1f} years")
        with c3:
            rem=max(0,months_between(TODAY,ss.o_end_date)) if ss.o_end_date>TODAY else 0
            computed("Remaining Term",f"{rem} months",f"{rem/12:.1f} years from today")
    else:
        c1,c2=st.columns(2)
        with c1:
            ss.o_term_mo=st.number_input("Loan Term (months)",value=ss.o_term_mo,min_value=1,max_value=600,step=12,key="w_o_tm")
        with c2:
            computed("Equivalent",f"{ss.o_term_mo/12:.1f} years")

    rate_delta_list("o_rate_deltas",ss.o_rate,
                    "Historical Rate Changes (± from previous rate)",
                    show_autofill=True,
                    autofill_data=rba_after_start)

    sec("Offset Account")
    c1,c2,c3=st.columns(3)
    with c1:
        ss.o_off_init=st.number_input("Initial Balance ($)",value=ss.o_off_init,min_value=0.0,step=1_000.0,
            key="w_o_oi",help=RATE_TOOLTIPS["offset"])
    with c2:
        ss.o_off_date=st.date_input("Offset Start Date",value=ss.o_off_date,key="w_o_od")
    with c3:
        ss.o_off_monthly=st.number_input("Monthly Addition ($)",value=ss.o_off_monthly,min_value=0.0,step=100.0,key="w_o_om")
    lump_list("o_off_lumps","Offset Lump Sum Deposits")

    sec("Fees")
    c1,c2,c3,c4=st.columns(4)
    with c1:
        ss.o_fee_mo=st.number_input("Monthly Fee ($)",value=ss.o_fee_mo,min_value=0.0,step=1.0,key="w_o_fm")
    with c2:
        ss.o_fee_setup=st.number_input("Setup / Establishment Fee ($)",value=ss.o_fee_setup,min_value=0.0,step=100.0,key="w_o_fs")
    with c3:
        ss.o_fee_break=st.number_input("Breakage Fee ($)",value=ss.o_fee_break,min_value=0.0,step=100.0,key="w_o_fb")
    with c4:
        ss.o_fee_other=st.number_input("Other One-off Fee ($)",value=ss.o_fee_other,min_value=0.0,step=100.0,key="w_o_fo")

def section_current():
    ss=st.session_state
    ss.c_is_cont=st.toggle(
        "Treat as continuation of original loan",value=ss.c_is_cont,key="w_c_ic",
        help=RATE_TOOLTIPS["cont"])

    sec("Property")
    c1,c2,c3=st.columns(3)
    prop_val_key="w_c_pv" if ss.c_is_cont else "w_c_pv2"
    with c1:
        ss.c_prop_val=st.number_input("Current Property Valuation ($)",value=ss.c_prop_val,
            min_value=0.0,step=10_000.0,key=prop_val_key,
            help="Current market valuation of the property (may differ from original if market has moved).")
    with c2:
        ss.c_prop_date=st.date_input("Valuation Date",value=ss.c_prop_date,
            key="w_c_pd" if ss.c_is_cont else "w_c_pd2")
    with c3:
        bal_for_lvr=ss.o_balance if ss.c_is_cont else ss.c_balance
        if ss.c_prop_val>0 and bal_for_lvr>0:
            computed("Current LVR "+tt("lvr_curr"),fp(bal_for_lvr/ss.c_prop_val*100),
                     "Balance ÷ Current Valuation")
        else: computed("Current LVR","—")

    if ss.c_is_cont:
        latest_r=eff_rate_from_deltas(ss.o_rate,ss.o_rate_deltas)
        rem=max(0,ss.o_term_mo-months_between(ss.o_balance_date,TODAY))
        sec("Auto-filled from Original Loan")
        a1,a2,a3=st.columns(3)
        with a1: computed("Current Remaining Balance",fc(ss.o_balance),"as at today")
        with a2: computed("Current Interest Rate",fp(latest_r),"after all original rate changes")
        with a3: computed("Remaining Term",f"{rem} months",f"{rem/12:.1f} years")

        # Auto-fill fees when continuation
        sec("Fees (auto-filled from Original Loan)")
        c1,c2=st.columns(2)
        with c1:
            ss.c_fee_mo=ss.o_fee_mo  # auto-fill
            computed("Monthly Fee (auto-filled)",fc(ss.c_fee_mo),"inherited from original loan")
        with c2:
            st.markdown(
                '<div class="note">Setup fee does not apply for continuation — no new establishment required.</div>',
                unsafe_allow_html=True)
            ss.c_fee_setup=0.0  # disabled for continuation

    else:
        sec("Current Loan Details")
        st.markdown('<div class="note">Remaining Balance is always recorded as at today.</div>',unsafe_allow_html=True)
        c1,c2,c3=st.columns(3)
        with c1:
            ss.c_balance=st.number_input("Current Remaining Balance ($)",value=ss.c_balance,
                min_value=0.0,step=1_000.0,key="w_c_bal",help="Outstanding principal as at today.")
        with c2:
            computed("Balance Date",TODAY.strftime("%d %b %Y"))
        with c3:
            ss.c_rate=st.number_input("Current Interest Rate (% p.a.)",value=ss.c_rate,
                min_value=0.0,max_value=30.0,step=0.01,format="%.4f",key="w_c_r")

        sec("Remaining Term")
        ss.c_use_dates=st.toggle("Calculate from loan end date",value=ss.c_use_dates,key="w_c_ud")
        if ss.c_use_dates:
            c1,c2=st.columns(2)
            with c1:
                ss.c_end_date=st.date_input("Loan End Date",value=ss.c_end_date,key="w_c_ed")
            with c2:
                if ss.c_end_date>TODAY:
                    ss.c_term_mo=months_between(TODAY,ss.c_end_date)
                    computed("Remaining Term",f"{ss.c_term_mo} months",f"{ss.c_term_mo/12:.1f} years")
        else:
            c1,c2=st.columns(2)
            with c1:
                ss.c_term_mo=st.number_input("Remaining Term (months)",value=ss.c_term_mo,
                    min_value=1,max_value=600,step=12,key="w_c_tm")
            with c2:
                computed("Equivalent",f"{ss.c_term_mo/12:.1f} years")

        sec("Fees")
        c1,c2,c3=st.columns(3)
        with c1:
            ss.c_fee_mo=st.number_input("Monthly Fee ($)",value=ss.c_fee_mo,min_value=0.0,step=1.0,key="w_c_fm")
        with c2:
            ss.c_fee_setup=st.number_input("Setup Fee ($)",value=ss.c_fee_setup,min_value=0.0,step=100.0,key="w_c_fs")
        with c3:
            ss.c_fee_other=st.number_input("Other Fee ($)",value=ss.c_fee_other,min_value=0.0,step=100.0,key="w_c_fo")

    # ── Anticipated rate changes — shared with Proposed Variable ──────────
    rate_delta_list(
        "future_var_deltas",
        eff_rate_from_deltas(ss.o_rate,ss.o_rate_deltas) if ss.c_is_cont else ss.c_rate,
        "Anticipated Rate Changes (shared with Proposed Variable rate)",
        max_rows=20
    )
    st.markdown('<div class="note">These anticipated changes apply to both the Current Loan and Proposed Variable component simultaneously.</div>',unsafe_allow_html=True)

    sec("Offset Account")
    c1,c2,c3=st.columns(3)
    with c1:
        ss.c_off_init=st.number_input("Initial Balance ($)",value=ss.c_off_init,min_value=0.0,step=1_000.0,key="w_c_oi")
    with c2:
        ss.c_off_date=st.date_input("Offset Start Date",value=ss.c_off_date,key="w_c_od")
    with c3:
        ss.c_off_monthly=st.number_input("Monthly Addition ($)",value=ss.c_off_monthly,min_value=0.0,step=100.0,key="w_c_om")
    lump_list("c_off_lumps","Offset Lump Sum Deposits")

def section_proposed():
    ss=st.session_state

    # ── Auto-fill proposed amount from current balance ─────────────────────
    curr_balance=ss.o_balance if ss.c_is_cont else ss.c_balance

    sec("Loan Amount and Term")
    ss.p_auto_amount=st.toggle("Auto-fill loan amount from current remaining balance",
                                value=ss.p_auto_amount,key="w_p_aa",
                                help="When enabled, the proposed loan amount matches the current outstanding balance.")
    if ss.p_auto_amount:
        ss.p_loan_amt=curr_balance
        c1,c2=st.columns(2)
        with c1:
            computed("Proposed Loan Amount (auto-filled)",fc(ss.p_loan_amt),"from current remaining balance")
        with c2:
            ss.p_start_date=st.date_input("Proposed Settlement Date",value=ss.p_start_date,key="w_p_sd")
    else:
        c1,c2=st.columns(2)
        with c1:
            ss.p_loan_amt=st.number_input("Proposed Loan Amount ($)",value=ss.p_loan_amt,
                min_value=0.0,step=1_000.0,key="w_p_la")
        with c2:
            ss.p_start_date=st.date_input("Proposed Settlement Date",value=ss.p_start_date,key="w_p_sd2")

    ss.p_use_dates=st.toggle("Calculate term from start and end dates",value=ss.p_use_dates,key="w_p_ud")
    if ss.p_use_dates:
        c1,c2=st.columns(2)
        with c1:
            ss.p_end_date=st.date_input("Loan End Date",value=ss.p_end_date,key="w_p_ed")
        with c2:
            if ss.p_end_date>ss.p_start_date:
                ss.p_term_mo=months_between(ss.p_start_date,ss.p_end_date)
                computed("Calculated Term",f"{ss.p_term_mo} months",f"{ss.p_term_mo/12:.1f} years")
    else:
        c1,c2=st.columns(2)
        with c1:
            ss.p_term_mo=st.number_input("Loan Term (months)",value=ss.p_term_mo,
                min_value=1,max_value=600,step=12,key="w_p_tm")
        with c2:
            computed("Equivalent",f"{ss.p_term_mo/12:.1f} years")

    # ── Interest Rates (revamped: adv / comparison / effective) ───────────
    sec("Interest Rates")

    # Build live rates using current fee values
    comp_var  = comparison_rate_asic(ss.p_fee_setup, ss.p_fee_mo, ss.p_adv_var_rate)
    eff_var   = effective_rate_calc(ss.p_loan_amt, ss.p_fee_setup, ss.p_fee_mo, ss.p_adv_var_rate, ss.p_term_mo)
    comp_fix  = comparison_rate_asic(ss.p_fee_setup, ss.p_fee_mo, ss.p_adv_fix_rate)
    eff_fix   = effective_rate_calc(ss.p_loan_amt, ss.p_fee_setup, ss.p_fee_mo, ss.p_adv_fix_rate, ss.p_term_mo)

    # Auto-fill reversion rate to effective variable (unless overridden)
    if not ss.p_rev_rate_override:
        ss.p_rev_rate = round(eff_var, 4)

    # Header row
    h1,h2=st.columns(2)
    h1.markdown('<div style="text-align:center;color:#30d996;font-size:0.75rem;font-weight:600;'
                'padding:6px;background:#071a0f;border-radius:5px;margin-bottom:8px">VARIABLE RATE</div>',
                unsafe_allow_html=True)
    h2.markdown('<div style="text-align:center;color:#e94560;font-size:0.75rem;font-weight:600;'
                'padding:6px;background:#1a0709;border-radius:5px;margin-bottom:8px">FIXED RATE</div>',
                unsafe_allow_html=True)

    c_v, c_f = st.columns(2)
    with c_v:
        ss.p_adv_var_rate=st.number_input(
            "Advertised Variable Rate (% p.a.)",value=ss.p_adv_var_rate,
            min_value=0.0,max_value=30.0,step=0.01,format="%.4f",key="w_p_avr",
            help=RATE_TOOLTIPS["adv_var"])
        computed(f"Comparison Variable Rate {tt('comp_var')}",fp(comp_var),
                 "ASIC standard: $150k / 25yr benchmark")
        computed(f"Effective Variable Rate {tt('eff_var')}",fp(eff_var),
                 f"For ${ss.p_loan_amt:,.0f} over {ss.p_term_mo} months")
    with c_f:
        ss.p_adv_fix_rate=st.number_input(
            "Advertised Fixed Rate (% p.a.)",value=ss.p_adv_fix_rate,
            min_value=0.0,max_value=30.0,step=0.01,format="%.4f",key="w_p_afr",
            help=RATE_TOOLTIPS["adv_fix"])
        computed(f"Comparison Fixed Rate {tt('comp_fix')}",fp(comp_fix),
                 "ASIC standard: $150k / 25yr benchmark")
        computed(f"Effective Fixed Rate {tt('eff_fix')}",fp(eff_fix),
                 f"For ${ss.p_loan_amt:,.0f} over {ss.p_term_mo} months")

    st.markdown('<div class="note">'+RATE_TOOLTIPS["comparison_rate_note"]+' (fees below)</div>',
                unsafe_allow_html=True)

    c1,c2=st.columns(2)
    with c1:
        ss.p_fix_yrs=st.number_input("Fixed Period (years)",value=ss.p_fix_yrs,
            min_value=1,max_value=30,step=1,key="w_p_fy",
            help="Duration of the fixed rate before the loan reverts to the variable rate.")
    with c2:
        ss.p_rev_rate_override=st.toggle(
            "Override reversion rate",value=ss.p_rev_rate_override,key="w_p_rro",
            help="By default the reversion rate equals the effective variable rate. Enable to set a custom (lower) rate.")
        if ss.p_rev_rate_override:
            ss.p_rev_rate=st.number_input("Reversion Rate (% p.a.)",value=ss.p_rev_rate,
                min_value=0.0,max_value=30.0,step=0.01,format="%.4f",key="w_p_rr_ov",
                help=RATE_TOOLTIPS["rev"])
        else:
            computed(f"Reversion Rate {tt('rev')}",fp(ss.p_rev_rate),
                     "Auto-filled to Effective Variable Rate")

    # ── Optimal Split ─────────────────────────────────────────────────────
    sec("Variable / Fixed Split")
    ss.p_split_auto=st.toggle(
        "Auto-calculate optimal split",value=ss.p_split_auto,key="w_p_sa",
        help=RATE_TOOLTIPS["split"])
    if not ss.p_split_auto:
        c1,c2=st.columns(2)
        with c1:
            ss.p_split_pct=st.slider("Fixed Component (%)",0.0,100.0,ss.p_split_pct,0.5,key="w_p_sp")
        with c2:
            computed("Allocation",
                     f"Variable {fc(ss.p_loan_amt*(100-ss.p_split_pct)/100)} / Fixed {fc(ss.p_loan_amt*ss.p_split_pct/100)}")
    else:
        st.markdown(
            f'<div class="note">Optimal split minimises cumulative interest + closing balance at end of the '
            f'{ss.p_fix_yrs}-year fixed period. Evaluated in 0.1% increments (1,001 scenarios). Both components '
            f'treated as separate loans funded from the proposed balance.</div>',
            unsafe_allow_html=True)

    # Rate changes — uses SHARED future_var_deltas (same as Current Loan)
    st.markdown(
        '<div class="note">Variable rate changes are synced with the Anticipated Rate Changes in the Current Loan section.</div>',
        unsafe_allow_html=True)
    if ss.future_var_deltas:
        h=st.columns([2,1.6,2.2,0.5])
        for i,lbl in enumerate(["Effective Date","Change (±%)","Result"]):
            h[i].markdown(f'<div class="list-hdr">{lbl}</div>',unsafe_allow_html=True)
        cum=ss.p_adv_var_rate
        for row in ss.future_var_deltas:
            c1,c2,c3,c4=st.columns([2,1.6,2.2,0.5])
            nv=float(row[1]); res=round(cum+nv,4)
            sign="+" if nv>=0 else ""; clr="#e94560" if nv>0 else ("#30d996" if nv<0 else "#64748b")
            c1.markdown(f'<div style="padding:8px 3px;color:#8892b0;font-size:.82rem">{parse_dt(row[0]).strftime("%d %b %Y")}</div>',unsafe_allow_html=True)
            c2.markdown(f'<div style="padding:8px 3px;color:{clr};font-size:.875rem;font-weight:500">{sign}{nv:.2f}%</div>',unsafe_allow_html=True)
            c3.markdown(f'<div style="padding:8px 3px;color:{clr};font-size:.875rem;font-weight:600">{res:.4f}%</div>',unsafe_allow_html=True)
            cum=res
    else:
        st.markdown('<div style="color:#64748b;font-size:0.8rem;padding:6px 0">No anticipated rate changes — add them in the Current Loan section above.</div>',unsafe_allow_html=True)

    sec("Offset Account")
    c1,c2,c3=st.columns(3)
    with c1:
        ss.p_off_init=st.number_input("Initial Balance ($)",value=ss.p_off_init,min_value=0.0,step=1_000.0,key="w_p_oi",help=RATE_TOOLTIPS["offset"])
    with c2:
        ss.p_off_date=st.date_input("Offset Start Date",value=ss.p_off_date,key="w_p_od")
    with c3:
        ss.p_off_monthly=st.number_input("Monthly Addition ($)",value=ss.p_off_monthly,min_value=0.0,step=100.0,key="w_p_om")
    lump_list("p_off_lumps","Offset Lump Sum Deposits")

    sec("Fees")
    st.markdown('<div class="note">Fees are used to calculate comparison and effective rates above.</div>',unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    with c1:
        ss.p_fee_mo=st.number_input("Monthly Fee ($)",value=ss.p_fee_mo,min_value=0.0,step=1.0,key="w_p_fm")
    with c2:
        ss.p_fee_setup=st.number_input("Establishment / Setup Fee ($)",value=ss.p_fee_setup,min_value=0.0,step=100.0,key="w_p_fs")
    with c3:
        ss.p_fee_break=st.number_input("Breakage Fee ($)",value=ss.p_fee_break,min_value=0.0,step=100.0,key="w_p_fb")
    with c4:
        ss.p_fee_other=st.number_input("Other One-off Fee ($)",value=ss.p_fee_other,min_value=0.0,step=100.0,key="w_p_fo")

def section_strategy():
    ss=st.session_state
    rba_rate=ss._rba_rate

    # Show live RBA rate prominently
    if rba_rate:
        st.markdown(
            f'<div class="data-panel">'
            f'<div class="data-panel-title">Current RBA Cash Rate</div>'
            f'<span style="font-size:1.8rem;font-weight:700;color:#4a9af5">{fp(rba_rate)}</span>'
            f'</div>',
            unsafe_allow_html=True)

    if ss._rba_next_meeting:
        st.markdown(
            f'<div class="note">Next RBA Monetary Policy Decision: <strong>{ss._rba_next_meeting}</strong></div>',
            unsafe_allow_html=True)

    sec("Strategy (User Defined)")
    ss.strategy=st.radio(
        "Select refinancing strategy",
        ["Conservative (80% fixed)","Balanced (optimal split)","Aggressive (0% fixed)"],
        index=["Conservative (80% fixed)","Balanced (optimal split)","Aggressive (0% fixed)"].index(
            ss.strategy if ss.strategy in ["Conservative (80% fixed)","Balanced (optimal split)","Aggressive (0% fixed)"]
            else "Balanced (optimal split)"),
        key="w_strat",horizontal=True,
        help="Strategy is a user-defined preference. The Analysis Dashboard shows all three options regardless of selection.")
    st.markdown("""
    <div class="note">
    Conservative: 80% fixed — maximises certainty, protects against rate rises.<br>
    Balanced: mathematically optimal split — minimises total interest + balance at end of fixed period.<br>
    Aggressive: 100% variable — maximum flexibility; best paired with offset account.
    </div>""",unsafe_allow_html=True)

    sec("Payment Behaviour on Rate Changes")
    ss.maintain_pmt=st.toggle(
        "When rates fall, maintain current repayment amount (pays off faster)",
        value=ss.maintain_pmt,key="w_mp",
        help="When rates rise, repayments always increase to maintain remaining loan term — the term cannot extend beyond remaining term. When rates fall, you can either maintain the higher payment (shortens term) or reduce to the new minimum.")

    sec("RBA Cash Rate Scenario (on top of any changes in Current / Proposed sections)")
    c1,c2=st.columns([4,1])
    with c1:
        ss.rba_bps=st.slider(
            "RBA cash rate change (basis points)",
            -300,300,ss.rba_bps,25,key="w_rba",
            help="This additional change is applied on top of any rate changes already entered in the Current and Proposed Loan sections.")
        if ss.rba_bps!=0:
            d="increase" if ss.rba_bps>0 else "decrease"
            st.markdown(
                f'<div class="note">Applies an additional {abs(ss.rba_bps)} bps '
                f'({abs(ss.rba_bps)/100:.2f}%) {d} on top of changes in Current/Proposed sections.</div>',
                unsafe_allow_html=True)
    with c2:
        if rba_rate:
            new_r=round(rba_rate+ss.rba_bps/100,2)
            computed("Implied Rate",fp(new_r),"after scenario")

# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTATION ENGINE  (called on every render — cached internally)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all()->dict|None:
    ss=st.session_state
    if ss.p_loan_amt<=0 or ss.o_balance<=0 or ss.o_term_mo<=0:
        return None

    maintain=ss.maintain_pmt
    far=add_months(TODAY,max(ss.o_term_mo,ss.p_term_mo)+12)

    def lumps_t(lumps,loan_start):
        return deltas_to_lumps_t(lumps,loan_start)

    # ── Original Loan ──────────────────────────────────────────────────────
    o_rsched=build_rate_schedule(ss.o_rate,ss.o_rate_deltas)
    o_lumps_t=lumps_t(ss.o_off_lumps,ss.o_balance_date)
    df_orig=amortize(ss.o_balance,ss.o_balance_date,ss.o_term_mo,
                     o_rsched,ss.o_off_init,ss.o_off_monthly,o_lumps_t,ss.o_fee_mo,maintain)

    # ── Current Loan ───────────────────────────────────────────────────────
    if ss.c_is_cont:
        c_base_rate=ss.o_rate; c_hist=ss.o_rate_deltas
        c_term=ss.o_term_mo; c_start=ss.o_balance_date; c_bal=ss.o_balance
        c_fee_mo=ss.o_fee_mo  # auto-inherited
    else:
        c_base_rate=ss.c_rate; c_hist=ss.c_rate_deltas
        c_term=ss.c_term_mo; c_start=TODAY; c_bal=ss.c_balance
        c_fee_mo=ss.c_fee_mo

    # Combine historical + anticipated + RBA scenario for current
    all_c_deltas = c_hist + ss.future_var_deltas
    if ss.rba_bps!=0:
        all_c_deltas = all_c_deltas + [[add_months(TODAY,1), ss.rba_bps/100]]
    c_rsched=build_rate_schedule(c_base_rate,all_c_deltas)
    c_lumps_t=lumps_t(ss.c_off_lumps,c_start)
    df_curr=amortize(c_bal,c_start,c_term,c_rsched,
                     ss.c_off_init,ss.c_off_monthly,c_lumps_t,c_fee_mo,maintain)

    # ── Proposed: compute derived rates live ────────────────────────────────
    comp_var  = comparison_rate_asic(ss.p_fee_setup, ss.p_fee_mo, ss.p_adv_var_rate)
    eff_var   = effective_rate_calc(ss.p_loan_amt, ss.p_fee_setup, ss.p_fee_mo, ss.p_adv_var_rate, ss.p_term_mo)
    comp_fix  = comparison_rate_asic(ss.p_fee_setup, ss.p_fee_mo, ss.p_adv_fix_rate)
    eff_fix   = effective_rate_calc(ss.p_loan_amt, ss.p_fee_setup, ss.p_fee_mo, ss.p_adv_fix_rate, ss.p_term_mo)

    # Optimal split
    if ss.p_split_auto:
        best_pct,split_df=calc_optimal_split(ss.p_loan_amt,eff_var,ss.p_adv_fix_rate,
                                              ss.p_rev_rate,ss.p_fix_yrs,ss.p_term_mo)
        ss.p_split_pct=round(best_pct,1)
    else:
        _,split_df=calc_optimal_split(ss.p_loan_amt,eff_var,ss.p_adv_fix_rate,
                                       ss.p_rev_rate,ss.p_fix_yrs,ss.p_term_mo)
        best_pct=ss.p_split_pct

    p_f=ss.p_loan_amt*best_pct/100; p_v=ss.p_loan_amt*(100-best_pct)/100
    fix_rev_date=add_months(TODAY,ss.p_fix_yrs*12)

    # Variable component: uses shared future_var_deltas + RBA scenario
    p_var_deltas=list(ss.future_var_deltas)
    if ss.rba_bps!=0:
        p_var_deltas=p_var_deltas+[[add_months(TODAY,1),ss.rba_bps/100]]
    p_vsched=build_rate_schedule(ss.p_adv_var_rate,p_var_deltas)
    p_fsched=[(date(1900,1,1),ss.p_adv_fix_rate),(fix_rev_date,ss.p_rev_rate)]
    p_lumps_t=lumps_t(ss.p_off_lumps,TODAY)

    df_pv=amortize(p_v,TODAY,ss.p_term_mo,p_vsched,
                   ss.p_off_init,ss.p_off_monthly,p_lumps_t,
                   ss.p_fee_mo*(100-best_pct)/100,maintain) if p_v>1 else pd.DataFrame()
    df_pf=amortize(p_f,TODAY,ss.p_term_mo,p_fsched,
                   0.0,0.0,(),ss.p_fee_mo*best_pct/100,maintain) if p_f>1 else pd.DataFrame()
    df_ps=merge_schedules(df_pv,df_pf)

    # ── Payment scenarios (7 scenarios applied to full variable proposed) ──
    base_deltas=[0.0,0.25,0.50,1.00,-0.25,-0.50,ss.rba_bps/100.0]
    scen_labels=["Base","+0.25%","+0.50%","+1.00%","-0.25%","-0.50%",
                 f"RBA {'+' if ss.rba_bps>=0 else ''}{ss.rba_bps/100:.2f}%"]
    scenarios={}
    for lbl,delta in zip(scen_labels,base_deltas):
        sr=ss.p_adv_var_rate+delta
        # Add anticipated changes on top
        scen_deltas=list(ss.future_var_deltas)
        df_s=amortize(ss.p_loan_amt,TODAY,ss.p_term_mo,
                      build_rate_schedule(sr,scen_deltas),
                      ss.p_off_init,ss.p_off_monthly,p_lumps_t,ss.p_fee_mo,maintain)
        if not df_s.empty:
            scenarios[lbl]={"rate":sr,"payment":df_s["Payment"].iloc[0],
                            "total_interest":df_s["Cum Interest"].iloc[-1],
                            "term_months":len(df_s),"df":df_s}

    return {"df_orig":df_orig,"df_curr":df_curr,"df_pv":df_pv,"df_pf":df_pf,"df_ps":df_ps,
            "split_df":split_df,"best_pct":best_pct,"eff_var":eff_var,"eff_fix":eff_fix,
            "comp_var":comp_var,"comp_fix":comp_fix,"scenarios":scenarios}

# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def dash_overview(R):
    ss=st.session_state
    dfs=[R["df_orig"],R["df_curr"],R["df_ps"]]
    labels=["Original Loan","Current Loan","Proposed (Split)"]
    colors=[C_ORIG,C_CURR,C_SPLIT]

    # Compute values for diff rows
    def row_val(df,field):
        if df is None or df.empty: return None
        if field=="payment": return df["Payment"].iloc[0]
        if field=="int": return df["Cum Interest"].iloc[-1]
        if field=="cost": return df["Cum Paid"].iloc[-1]
        if field=="term": return len(df)
    fields=[("payment","Monthly Payment"),("int","Total Interest"),("cost","Total Cost"),("term","Loan Term")]

    cols=st.columns(3)
    vals={f:[] for f,_ in fields}
    for df,lbl,clr in zip(dfs,labels,colors):
        for f,_ in fields:
            vals[f].append(row_val(df,f))

    for ci,(col,df,lbl,clr) in enumerate(zip(cols,dfs,labels,colors)):
        with col:
            if df is None or df.empty:
                st.info(f"No data for {lbl}"); continue
            st.markdown(f'<div style="color:{clr};font-weight:600;font-size:0.8rem;'
                        f'margin-bottom:8px;padding-bottom:4px;border-bottom:1px solid #1e2d4a">{lbl}</div>',
                        unsafe_allow_html=True)
            for f,flbl in fields:
                v=row_val(df,f)
                if v is None: continue
                # Build diff string vs previous column
                diff_str=""
                dpos=True  # default; only meaningful when ci>0
                if ci>0:
                    v_prev=vals[f][0]  # vs original
                    if v_prev and v_prev!=0:
                        diff=v-v_prev
                        if f=="term":
                            diff_str=f"{'−' if diff<0 else '+'}{abs(int(diff))} mo vs Orig"
                            dpos=diff<0
                        else:
                            pct=diff/v_prev*100
                            diff_str=f"{'−' if diff<0 else '+'}${abs(diff):,.0f} ({abs(pct):.1f}%) vs Orig"
                            dpos=diff<0  # less interest/cost is positive for user
                dformat=fc(v) if f!="term" else f"{v} mo / {v/12:.1f} yr"
                st.markdown(metric_card(flbl,dformat,diff_str,dpos,diff_neutral=(ci==0)),
                            unsafe_allow_html=True)

    # Summary savings row
    df_c,df_s=R["df_curr"],R["df_ps"]
    if df_c is not None and not df_c.empty and df_s is not None and not df_s.empty:
        st.markdown("<hr>",unsafe_allow_html=True)
        st.markdown("**Proposed vs Current — Key Metrics**")
        c1,c2,c3,c4,c5=st.columns(5)
        items=[
            ("Interest Saved",fc(df_c["Cum Interest"].iloc[-1]-df_s["Cum Interest"].iloc[-1])),
            ("Total Cost Saved",fc(df_c["Cum Paid"].iloc[-1]-df_s["Cum Paid"].iloc[-1])),
            ("Optimal Fixed Split",fp(R["best_pct"],1)),
            ("Effective Var Rate",fp(R["eff_var"])),
            ("ASIC Comparison Rate",fp(R["comp_var"])),
        ]
        for col,(lbl,val) in zip([c1,c2,c3,c4,c5],items):
            with col: st.markdown(metric_card(lbl,val),unsafe_allow_html=True)

def dash_payments(R):
    df_o,df_c=R["df_orig"],R["df_curr"]
    df_ps,df_pv,df_pf=R["df_ps"],R["df_pv"],R["df_pf"]

    fig=go.Figure()
    traces=[(df_o,"Original",C_ORIG,"dot"),(df_c,"Current",C_CURR,"dash"),
            (df_ps,"Proposed Split",C_SPLIT,"solid"),(df_pv,"Variable Component",C_VAR,"dot"),
            (df_pf,"Fixed Component",C_FIX,"dot")]
    for df,nm,clr,dash in traces:
        if df is not None and not df.empty:
            fig.add_trace(go.Scatter(x=df["Date"],y=df["Payment"],name=nm,
                line=dict(color=clr,width=2,dash=dash),
                hovertemplate=f"<b>{nm}</b><br>%{{x|%b %Y}}<br>$%{{y:,.0f}}/mo<extra></extra>"))
    fig.update_layout(**PLOT_BASE,title="Monthly Repayments",yaxis_title="Repayment ($)")
    fig.update_xaxes(rangeslider=dict(visible=True,thickness=0.04))
    st.plotly_chart(fig,use_container_width=True)

    if df_ps is not None and not df_ps.empty:
        s=df_ps.iloc[::12]
        fig2=go.Figure()
        fig2.add_trace(go.Bar(x=s["Date"],y=s["Principal"],name="Principal",
                              marker_color=C_VAR,hovertemplate="Principal: $%{y:,.0f}<extra></extra>"))
        fig2.add_trace(go.Bar(x=s["Date"],y=s["Interest"],name="Interest",
                              marker_color=C_FIX,hovertemplate="Interest: $%{y:,.0f}<extra></extra>"))
        if df_ps["Interest Saved"].sum()>0:
            fig2.add_trace(go.Scatter(x=s["Date"],y=s["Interest Saved"],name="Interest Saved (Offset)",
                mode="lines+markers",line=dict(color=C_ORIG,width=2),yaxis="y2",
                hovertemplate="Saved: $%{y:,.0f}<extra></extra>"))
            fig2.update_layout(yaxis2=dict(overlaying="y",side="right",showgrid=False,
                                           title="Saved ($)",color=C_ORIG,tickfont=dict(color=C_ORIG)))
        fig2.update_layout(**PLOT_BASE,barmode="stack",
                           title="Annual Repayment Breakdown — Proposed Split",yaxis_title="Amount ($)")
        st.plotly_chart(fig2,use_container_width=True)

def dash_balance(R):
    ss=st.session_state
    df_o,df_c=R["df_orig"],R["df_curr"]
    df_ps,df_pv,df_pf=R["df_ps"],R["df_pv"],R["df_pf"]

    fig=go.Figure()
    for df,nm,clr,dash in [(df_o,"Original",C_ORIG,"dot"),(df_c,"Current",C_CURR,"dash"),
                           (df_ps,"Proposed Split",C_SPLIT,"solid"),
                           (df_pv,"Variable Component",C_VAR,"dot"),(df_pf,"Fixed Component",C_FIX,"dot")]:
        if df is not None and not df.empty:
            fig.add_trace(go.Scatter(x=df["Date"],y=df["Closing Balance"],name=nm,
                line=dict(color=clr,width=2,dash=dash),
                hovertemplate=f"<b>{nm}</b><br>%{{x|%b %Y}}<br>$%{{y:,.0f}}<extra></extra>",
                fill="tozeroy" if nm=="Proposed Split" else "none",
                fillcolor="rgba(196,122,245,0.04)" if nm=="Proposed Split" else "rgba(0,0,0,0)"))
    # Annotate reversion date
    rev_date=add_months(TODAY,ss.p_fix_yrs*12)
    fig.add_vline(x=str(rev_date),line=dict(color=C_FIX,dash="dash",width=1),
                  annotation_text="Fixed rate expires",annotation_font=dict(color=C_FIX,size=10))
    fig.update_layout(**PLOT_BASE,title="Outstanding Balance Over Time",yaxis_title="Balance ($)")
    fig.update_xaxes(rangeslider=dict(visible=True,thickness=0.04))
    st.plotly_chart(fig,use_container_width=True)

    curr_val=ss.c_prop_val if not ss.c_is_cont else ss.o_prop_val
    if curr_val>0 and df_ps is not None and not df_ps.empty:
        lvr=df_ps["Closing Balance"]/curr_val*100
        fig2=go.Figure()
        fig2.add_trace(go.Scatter(x=df_ps["Date"],y=lvr,name="LVR (%)",
            line=dict(color=C_SPLIT,width=2),fill="tozeroy",fillcolor="rgba(196,122,245,0.07)",
            hovertemplate="%{x|%b %Y}<br>LVR: %{y:.1f}%<extra></extra>"))
        fig2.add_hline(y=80,line=dict(color=C_FIX,dash="dash",width=1),
                       annotation_text="80% — LMI threshold",annotation_font=dict(color=C_FIX,size=10))
        fig2.add_hline(y=60,line=dict(color=C_VAR,dash="dot",width=1),
                       annotation_text="60%",annotation_font=dict(color=C_VAR,size=10))
        fig2.update_layout(**PLOT_BASE,title="LVR Over Time — Proposed Split",yaxis_title="LVR (%)")
        st.plotly_chart(fig2,use_container_width=True)

def dash_interest(R):
    df_o,df_c,df_ps=R["df_orig"],R["df_curr"],R["df_ps"]
    fig=go.Figure()
    for df,nm,clr in [(df_o,"Original",C_ORIG),(df_c,"Current",C_CURR),(df_ps,"Proposed Split",C_SPLIT)]:
        if df is not None and not df.empty:
            fig.add_trace(go.Scatter(x=df["Date"],y=df["Cum Interest"],name=nm,
                line=dict(color=clr,width=2),
                hovertemplate=f"<b>{nm}</b><br>%{{x|%b %Y}}<br>$%{{y:,.0f}} cumulative<extra></extra>"))
    fig.update_layout(**PLOT_BASE,title="Cumulative Interest Paid",yaxis_title="Cumulative Interest ($)")
    fig.update_xaxes(rangeslider=dict(visible=True,thickness=0.04))
    st.plotly_chart(fig,use_container_width=True)

    # Waterfall — difference between Current and Proposed
    if df_c is not None and not df_c.empty and df_ps is not None and not df_ps.empty:
        int_sav=df_c["Cum Interest"].iloc[-1]-df_ps["Cum Interest"].iloc[-1]
        cost_sav=df_c["Cum Paid"].iloc[-1]-df_ps["Cum Paid"].iloc[-1]
        term_sav=(len(df_c)-len(df_ps))
        fig2=go.Figure(go.Waterfall(
            orientation="h",
            measure=["absolute","relative","relative","total"],
            x=[df_c["Cum Paid"].iloc[-1],
               -(df_c["Cum Interest"].iloc[-1]-df_ps["Cum Interest"].iloc[-1]),
               -(df_c["Fees"].sum()-df_ps["Fees"].sum()),
               df_ps["Cum Paid"].iloc[-1]],
            y=["Current Total Cost","Interest Saving","Fee Difference","Proposed Total Cost"],
            connector=dict(line=dict(color=C_GRID)),
            decreasing=dict(marker=dict(color=C_VAR)),
            increasing=dict(marker=dict(color=C_FIX)),
            totals=dict(marker=dict(color=C_SPLIT)),
            text=[fc(df_c["Cum Paid"].iloc[-1]),f"−{fc(abs(int_sav))}",
                  f"−{fc(abs(df_c['Fees'].sum()-df_ps['Fees'].sum()))}",fc(df_ps["Cum Paid"].iloc[-1])],
            textposition="outside",
        ))
        fig2.update_layout(**PLOT_BASE,title="Cost Waterfall — Current vs Proposed",xaxis_title="Total Cost ($)")
        st.plotly_chart(fig2,use_container_width=True)

    if df_ps is not None and not df_ps.empty and df_ps["Cum Interest Saved"].iloc[-1]>1:
        fig3=go.Figure()
        fig3.add_trace(go.Scatter(x=df_ps["Date"],y=df_ps["Cum Interest Saved"],
            name="Cumulative Offset Savings",line=dict(color=C_VAR,width=2),
            fill="tozeroy",fillcolor="rgba(48,217,150,0.08)",
            hovertemplate="%{x|%b %Y}<br>Saved: $%{y:,.0f}<extra></extra>"))
        fig3.update_layout(**PLOT_BASE,title="Cumulative Interest Saved via Offset Account",
                           yaxis_title="Savings ($)")
        st.plotly_chart(fig3,use_container_width=True)

def dash_split(R):
    ss=st.session_state
    sdf=R["split_df"]; best=R["best_pct"]

    st.markdown(
        f'<div class="note">Optimal fixed component: <strong>{best:.1f}%</strong> '
        f'({100-best:.1f}% variable). Evaluated in 0.1% increments (1,001 scenarios) over '
        f'the {ss.p_fix_yrs}-year fixed period only.</div>',unsafe_allow_html=True)

    p_f=ss.p_loan_amt*best/100; p_v=ss.p_loan_amt*(100-best)/100
    c1,c2,c3,c4=st.columns(4)
    with c1: st.markdown(metric_card("Optimal Fixed %",fp(best,1)),unsafe_allow_html=True)
    with c2: st.markdown(metric_card("Fixed Amount",fc(p_f)),unsafe_allow_html=True)
    with c3: st.markdown(metric_card("Variable Amount",fc(p_v)),unsafe_allow_html=True)
    with c4:
        best_obj=sdf.iloc[sdf["Objective"].idxmin()]["Objective"]
        st.markdown(metric_card("Min Objective Value",fc(best_obj)),unsafe_allow_html=True)

    fig=make_subplots(rows=1,cols=2,
        subplot_titles=["Objective: Cum Interest + End Balance","Interest vs Balance at Fixed Period End"],
        horizontal_spacing=0.12)
    fig.add_trace(go.Scatter(x=sdf["Fixed %"],y=sdf["Objective"],name="Objective",
        line=dict(color=C_SPLIT,width=2),
        hovertemplate="Fixed: %{x:.1f}%<br>Objective: $%{y:,.0f}<extra></extra>"),row=1,col=1)
    fig.add_trace(go.Scatter(x=[best],y=[best_obj],mode="markers",
        marker=dict(symbol="diamond",size=14,color=C_FIX),name=f"Optimal {best:.1f}%"),row=1,col=1)
    fig.add_trace(go.Scatter(x=sdf["Fixed %"],y=sdf["Cum Interest"],name="Cum Interest",
        line=dict(color=C_FIX,width=1.5),
        hovertemplate="Fixed: %{x:.1f}%<br>Interest: $%{y:,.0f}<extra></extra>"),row=1,col=2)
    fig.add_trace(go.Scatter(x=sdf["Fixed %"],y=sdf["End Balance"],name="End Balance",
        line=dict(color=C_VAR,width=1.5),
        hovertemplate="Fixed: %{x:.1f}%<br>Balance: $%{y:,.0f}<extra></extra>"),row=1,col=2)
    fig.update_layout(**PLOT_BASE)
    fig.update_xaxes(title_text="Fixed Component (%)",gridcolor=C_GRID)
    fig.update_yaxes(title_text="$",gridcolor=C_GRID)
    st.plotly_chart(fig,use_container_width=True)

    tbl=sdf[sdf["Fixed %"]%5==0].copy()
    st.dataframe(pd.DataFrame({
        "Fixed %":tbl["Fixed %"].apply(lambda x:f"{x:.0f}%"),
        "Variable %":tbl["Variable %"].apply(lambda x:f"{x:.0f}%"),
        "Cum Interest":tbl["Cum Interest"].apply(fc),
        "End Balance":tbl["End Balance"].apply(fc),
        "Objective":tbl["Objective"].apply(fc),
    }),use_container_width=True,hide_index=True)

def dash_strategy_tab(R):
    ss=st.session_state
    loan,term=ss.p_loan_amt,ss.p_term_mo
    fix_mo=ss.p_fix_yrs*12; best=R["best_pct"]

    # Show user's selected strategy prominently
    strat_map={"Conservative (80% fixed)":80.0,"Balanced (optimal split)":best,"Aggressive (0% fixed)":0.0}
    sel_pct=strat_map.get(ss.strategy,best)
    st.markdown(
        f'<div class="note-ok">Selected strategy: <strong>{ss.strategy}</strong> '
        f'— {sel_pct:.1f}% fixed / {100-sel_pct:.1f}% variable</div>',unsafe_allow_html=True)

    strats={"Conservative\n(80% Fixed)":(80.0,C_FIX),"Balanced\n(Optimal)":(best,C_SPLIT),"Aggressive\n(0% Fixed)":(0.0,C_VAR)}
    totals_ci={}; totals_pmt={}; totals_cost={}
    for nm,(pct_f,_) in strats.items():
        p_f_=loan*pct_f/100; p_v_=loan*(100-pct_f)/100
        bf,cif=fast_partial(p_f_,ss.p_adv_fix_rate,term,fix_mo) if p_f_>0 else (0,0)
        bv,civ=fast_partial(p_v_,R["eff_var"],term,fix_mo) if p_v_>0 else (0,0)
        rem=max(0,term-fix_mo)
        _,cif2=fast_partial(bf,ss.p_rev_rate,rem,rem) if rem>0 and bf>0 else (0,0)
        _,civ2=fast_partial(bv,R["eff_var"],rem,rem) if rem>0 and bv>0 else (0,0)
        k=nm.split("\n")[0]
        totals_ci[k]=cif+civ+cif2+civ2
        totals_pmt[k]=(calc_payment(p_f_,ss.p_adv_fix_rate,term) if p_f_>0 else 0)+(calc_payment(p_v_,R["eff_var"],term) if p_v_>0 else 0)
        totals_cost[k]=totals_ci[k]+loan  # simplified total cost

    c1,c2,c3=st.columns(3)
    clrs=[C_FIX,C_SPLIT,C_VAR]
    with c1:
        fig=go.Figure(go.Bar(x=list(totals_ci.keys()),y=list(totals_ci.values()),marker_color=clrs,
            text=[fc(v) for v in totals_ci.values()],textposition="outside",
            hovertemplate="%{x}<br>Total Interest: $%{y:,.0f}<extra></extra>"))
        fig.update_layout(**PLOT_BASE,title="Total Interest by Strategy",yaxis_title="($)")
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        fig2=go.Figure(go.Bar(x=list(totals_pmt.keys()),y=list(totals_pmt.values()),marker_color=clrs,
            text=[fc(v) for v in totals_pmt.values()],textposition="outside",
            hovertemplate="%{x}<br>Monthly Repayment: $%{y:,.0f}<extra></extra>"))
        fig2.update_layout(**PLOT_BASE,title="Initial Monthly Repayment",yaxis_title="($)")
        st.plotly_chart(fig2,use_container_width=True)
    with c3:
        # Radar chart comparing strategies
        cats=["Total Interest","Monthly Pmt","Flexibility","Rate Protection","Offset Benefit"]
        # Normalise: lower interest is better, lower pmt is better, variable=more flexible, fixed=more protection
        ci_max=max(totals_ci.values()) or 1
        pmt_max=max(totals_pmt.values()) or 1
        radar_data={
            "Conservative\n(80% Fixed)":[(totals_ci["Conservative"]/ci_max)*10,
                                          (totals_pmt["Conservative"]/pmt_max)*10,2,9,2],
            "Balanced\n(Optimal)":[(totals_ci["Balanced"]/ci_max)*10,
                                    (totals_pmt["Balanced"]/pmt_max)*10,5,6,5],
            "Aggressive\n(0% Fixed)":[(totals_ci["Aggressive"]/ci_max)*10,
                                       (totals_pmt["Aggressive"]/pmt_max)*10,9,2,9],
        }
        fig3=go.Figure()
        for nm,(pct_f,clr) in strats.items():
            k=nm.split("\n")[0]
            fig3.add_trace(go.Scatterpolar(r=radar_data[nm],theta=cats,fill="toself",
                name=nm.replace("\n"," "),line=dict(color=clr),
                opacity=0.7,fillcolor=clr.replace("#","rgba(")+"," if False else clr))
        fig3.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,10],
                                      gridcolor=C_GRID,linecolor=C_GRID),
                                      bgcolor=C_PLOT,angularaxis=dict(gridcolor=C_GRID)),
                          paper_bgcolor=C_PAPER,plot_bgcolor=C_PLOT,
                          font=dict(family="Inter",color=C_TEXT,size=10),
                          legend=dict(bgcolor=C_PAPER,bordercolor=C_GRID,borderwidth=1),
                          margin=dict(t=40,b=20,l=30,r=30),
                          title="Strategy Comparison (Radar)")
        st.plotly_chart(fig3,use_container_width=True)

def dash_scenarios(R):
    scenarios=R["scenarios"]
    if not scenarios: st.warning("No scenario data."); return

    base=scenarios.get("Base",{})
    base_pmt=base.get("payment",0)
    base_int=base.get("total_interest",0)
    base_term=base.get("term_months",0)

    st.markdown("**Monthly Repayments Under Rate Scenarios**")
    st.markdown(
        '<div class="note">Rate rises always increase repayments (term never extends). '
        'Rate falls reduce repayments or shorten term. RBA scenario is applied in addition to '
        'any changes in Current/Proposed sections.</div>',unsafe_allow_html=True)

    rows=[]
    for lbl,data in scenarios.items():
        dp=data["payment"]-base_pmt; di=data["total_interest"]-base_int; dt=data["term_months"]-base_term
        rows.append({"Scenario":lbl,"Variable Rate":fp(data["rate"]),
                     "Monthly Repayment":fc(data["payment"]),
                     "vs Base":f"{'+' if dp>=0 else ''}{fc(dp)}",
                     "Total Interest":fc(data["total_interest"]),
                     "Interest vs Base":f"{'+' if di>=0 else ''}{fc(di)}",
                     "Term (mo)":str(data["term_months"]),
                     "Term vs Base":f"{'+' if dt>=0 else ''}{dt} mo"})
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

    scen_colors=[C_SPLIT,C_FIX,"#f59e0b","#ef4444",C_VAR,"#22d3ee",C_ORIG]

    fig_pmt=go.Figure()
    for (lbl,data),clr in zip(scenarios.items(),scen_colors):
        df_s=data.get("df")
        if df_s is not None and not df_s.empty:
            fig_pmt.add_trace(go.Scatter(x=df_s["Date"],y=df_s["Payment"],name=lbl,
                line=dict(color=clr,width=2 if lbl=="Base" else 1.5,
                          dash="solid" if lbl=="Base" else "dash"),
                hovertemplate=f"<b>{lbl}</b><br>%{{x|%b %Y}}<br>$%{{y:,.0f}}/mo<extra></extra>"))
    fig_pmt.update_layout(**PLOT_BASE,title="Monthly Repayments — Rate Scenarios",yaxis_title="Monthly Repayment ($)")
    fig_pmt.update_xaxes(rangeslider=dict(visible=True,thickness=0.04))
    st.plotly_chart(fig_pmt,use_container_width=True)

    fig_bal=go.Figure()
    for (lbl,data),clr in zip(scenarios.items(),scen_colors):
        df_s=data.get("df")
        if df_s is not None and not df_s.empty:
            fig_bal.add_trace(go.Scatter(x=df_s["Date"],y=df_s["Closing Balance"],name=lbl,
                line=dict(color=clr,width=2 if lbl=="Base" else 1.5,
                          dash="solid" if lbl=="Base" else "dash"),
                hovertemplate=f"<b>{lbl}</b><br>%{{x|%b %Y}}<br>$%{{y:,.0f}}<extra></extra>"))
    fig_bal.update_layout(**PLOT_BASE,title="Outstanding Balance — Rate Scenarios",yaxis_title="Balance ($)")
    fig_bal.update_xaxes(rangeslider=dict(visible=True,thickness=0.04))
    st.plotly_chart(fig_bal,use_container_width=True)

def dash_schedules(R):
    options={"Original Loan":R["df_orig"],"Current Loan":R["df_curr"],
             "Proposed Variable Component":R["df_pv"],"Proposed Fixed Component":R["df_pf"],
             "Proposed Split (Combined)":R["df_ps"]}
    avail={k:v for k,v in options.items() if v is not None and not v.empty}
    if not avail: st.warning("No schedules."); return
    sel=st.selectbox("Select schedule",list(avail.keys()),key="sch_sel")
    df=avail[sel].copy()
    disp=df.copy()
    for c in ["Opening Balance","Avg Offset","Net Debt","Interest","Interest Saved","Principal",
              "Fees","Payment","Closing Balance","Cum Interest","Cum Paid","Cum Interest Saved"]:
        if c in disp.columns: disp[c]=disp[c].apply(fc)
    if "Rate %" in disp.columns: disp["Rate %"]=disp["Rate %"].apply(fp)
    st.dataframe(disp,use_container_width=True,hide_index=True,height=440)
    buf=io.BytesIO(); df.to_csv(buf,index=False); buf.seek(0)
    st.download_button(f"Download {sel} as CSV",buf.getvalue(),
                       f"schedule_{sel.lower().replace(' ','_').replace('(','').replace(')','')}.csv",
                       "text/csv",key=f"dl_{sel}")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    init_state()
    ss=st.session_state

    # Auto-load live data once per session
    load_live_data()

    # ── Header ─────────────────────────────────────────────────────────────
    rba_str=f" · RBA: <span style='color:#4a9af5;font-weight:600'>{fp(ss._rba_rate)}</span>" if ss._rba_rate else ""
    meeting_str=f" · Next Decision: <span style='color:#64748b'>{ss._rba_next_meeting}</span>" if ss._rba_next_meeting else ""
    st.markdown(f"""
    <div style="padding:20px 0 16px;border-bottom:1px solid #1e2d4a;margin-bottom:20px;">
        <h1 style="color:#d4dbe8;font-size:1.5rem;font-weight:600;margin:0 0 4px;">
            Australian Mortgage Refinance Analyser
        </h1>
        <p style="color:#64748b;font-size:0.79rem;margin:0;">
            Daily-interest amortisation &nbsp;·&nbsp; Real-time analysis
            &nbsp;·&nbsp; Optimal variable/fixed split (0.1% increments)
            &nbsp;·&nbsp; ASIC comparison rates{rba_str}{meeting_str}
        </p>
    </div>""",unsafe_allow_html=True)

    # ── RBA & ASX Panel ────────────────────────────────────────────────────
    render_rba_panel()

    # ── Input Sections ──────────────────────────────────────────────────────
    with st.expander("Original Loan",expanded=True):
        section_original()
    with st.expander("Current Loan",expanded=True):
        section_current()
    with st.expander("Proposed Loan",expanded=True):
        section_proposed()
    with st.expander("Strategy and Scenarios",expanded=True):
        section_strategy()

    # ── Reset ──────────────────────────────────────────────────────────────
    st.markdown("<div style='height:10px'></div>",unsafe_allow_html=True)
    c1,_=st.columns([1,7])
    with c1:
        st.markdown('<div class="btn-danger">',unsafe_allow_html=True)
        if st.button("Reset All",key="btn_rst"):
            for k in list(ss.keys()): del ss[k]
            st.rerun()
        st.markdown("</div>",unsafe_allow_html=True)

    # ── Real-time Dashboard (always computed, no button) ───────────────────
    st.markdown("<hr>",unsafe_allow_html=True)
    st.markdown('<h2 style="color:#d4dbe8;font-size:1.1rem;font-weight:600;margin:0 0 14px">Analysis Dashboard</h2>',
                unsafe_allow_html=True)

    with st.spinner("Computing..."):
        R=compute_all()

    if R is None:
        st.markdown("""
        <div style="text-align:center;padding:48px 0;color:#64748b;">
            <div style="font-size:.92rem;font-weight:500;color:#8892b0;margin-bottom:6px;">
                Enter loan details above — analysis updates in real time
            </div>
            <div style="font-size:.8rem">Ensure Original Loan Amount and Remaining Balance are positive</div>
        </div>""",unsafe_allow_html=True)
        return

    tabs=st.tabs(["Overview","Monthly Payments","Loan Balance","Interest Analysis",
                  "Optimal Split","Strategy","Rate Scenarios","Schedules"])
    with tabs[0]: dash_overview(R)
    with tabs[1]: dash_payments(R)
    with tabs[2]: dash_balance(R)
    with tabs[3]: dash_interest(R)
    with tabs[4]: dash_split(R)
    with tabs[5]: dash_strategy_tab(R)
    with tabs[6]: dash_scenarios(R)
    with tabs[7]: dash_schedules(R)


if __name__=="__main__":
    main()
