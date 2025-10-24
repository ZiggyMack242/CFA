“””
CFA v2.0 Interactive Console
“All Named, All Priced” - Now with knobs and dials

Usage: streamlit run app.py
“””

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Tuple, List
import json

# ============================================================================

# DATA MODEL & DEFAULTS

# ============================================================================

MDN_DEFAULT = {
“name”: “Methodological Naturalism”,
“bf_i”: {“axioms”: 6, “debts”: 4},
“levers”: {
“CCI”: 8.0,
“EDB”: 7.5,
“PF_instrumental”: 10.0,
“PF_existential”: 3.0,
“AR”: 7.0,
“MG”: 4.0
},
“admits_limits”: True  # for fallibilism bonus eligibility
}

CT_DEFAULT = {
“name”: “Classical Theism”,
“bf_i”: {“axioms”: 7, “debts”: 4},
“levers”: {
“CCI”: 7.5,
“EDB”: 8.5,
“PF_instrumental”: 7.0,
“PF_existential”: 8.0,
“AR”: 8.5,
“MG”: 8.5
},
“admits_limits”: True  # acknowledges mystery
}

PF_TYPES = [“Instrumental”, “Composite_70_30”, “Holistic_50_50”]
LEVER_NAMES = [“CCI”, “EDB”, “PF”, “AR”, “MG”]

# ============================================================================

# CORE MATH FUNCTIONS

# ============================================================================

def composite_pf(pf_inst: float, pf_exist: float, pf_type: str) -> float:
“”“Calculate composite PF based on type.”””
if pf_type == “Instrumental”:
return pf_inst
if pf_type == “Holistic_50_50”:
return 0.5 * pf_inst + 0.5 * pf_exist
# Default: Composite_70_30
return 0.7 * pf_inst + 0.3 * pf_exist

def apply_fallibilism_bonus(cci: float, bonus: str, admitted_limits: bool = True) -> float:
“”“Apply +0.3 CCI if framework admits limits and bonus is ON.”””
if bonus == “ON” and admitted_limits:
return min(cci + 0.3, 10.0)  # Cap at 10
return cci

def parity_weight(mg: float, parity: str) -> float:
“”“Apply parity weighting to MG (0.5× if OFF).”””
return mg if parity == “ON” else 0.5 * mg

def bfi_total(axioms: int, debts: int, debt_weight: str) -> float:
“”“Calculate total BFI with optional debt weighting.”””
w = 1.0 if debt_weight == “Equal_1.0x” else 1.2
return axioms + w * debts

def ypa_scenario_scores(fr: Dict, cfg: Dict) -> Tuple[Dict, Dict, float]:
“””
Compute YPA for all three scenarios.

```
Returns:
    - scenarios: Dict of {scenario_name: {total, YPA}}
    - final_levers: Dict of adjusted lever values
    - bfi: Total BFI
"""
# Apply adjustments
CCI = apply_fallibilism_bonus(
    fr["levers"]["CCI"], 
    cfg["fallibilism_bonus"],
    fr.get("admits_limits", True)
)
EDB = fr["levers"]["EDB"]
PF = composite_pf(
    fr["levers"]["PF_instrumental"],
    fr["levers"]["PF_existential"],
    cfg["pf_type"]
)
AR = fr["levers"]["AR"]
MG = parity_weight(fr["levers"]["MG"], cfg["lever_parity"])

# Weight scenarios
scenarios_weights = {
    "Neutral": {"CCI": 1.0, "EDB": 1.0, "PF": 1.0, "AR": 1.0, "MG": 1.0},
    "Existential": {"CCI": 1.0, "EDB": 2.0, "PF": 1.0, "AR": 1.0, "MG": 2.0},
    "Empirical": {"CCI": 1.5, "EDB": 1.0, "PF": 2.0, "AR": 1.0, "MG": 1.0},
}

lever_map = {"CCI": CCI, "EDB": EDB, "PF": PF, "AR": AR, "MG": MG}
bfi = bfi_total(fr["bf_i"]["axioms"], fr["bf_i"]["debts"], cfg["bfi_debt_weight"])

# Calculate YPA for each scenario
results = {}
for name, weights in scenarios_weights.items():
    total = sum(lever_map[k] * w for k, w in weights.items())
    results[name] = {"total": total, "YPA": total / bfi if bfi > 0 else 0}

return results, lever_map, bfi
```

# ============================================================================

# GUARDRAILS

# ============================================================================

def guardrail_lever_coupling(PF: float, CCI: float) -> Tuple[bool, str]:
“”“Check if PF ≥ 9 requires CCI ≥ 6.5.”””
if PF >= 9 and CCI < 6.5:
return False, f”⚠️ FAIL: PF={PF:.2f} ≥ 9 but CCI={CCI:.2f} < 6.5”
return True, f”✅ PASS: Lever-Coupling satisfied (PF={PF:.2f}, CCI={CCI:.2f})”

def guardrail_bfi_sensitivity(ypa_current: float, ypa_baseline: float,
bfi_current: float, bfi_baseline: float) -> Tuple[bool, str]:
“”“Check if YPA increases faster than BFI (suspicious efficiency gain).”””
if bfi_current == bfi_baseline:
return True, “✅ N/A: BFI unchanged”

```
slope = (ypa_current - ypa_baseline) / (bfi_current - bfi_baseline)
if slope > 0.4:
    return False, f"⚠️ FLAG: ΔYPA/ΔBFI = {slope:.3f} > 0.4 (suspicious efficiency)"
return True, f"✅ PASS: ΔYPA/ΔBFI = {slope:.3f} ≤ 0.4"
```

def guardrail_weight_bounds(weights: Dict) -> Tuple[bool, str]:
“”“Check if any scenario weight is <0.3× or >3×.”””
violations = [k for k, v in weights.items() if v < 0.3 or v > 3.0]
if violations:
return False, f”⚠️ FAIL: Extreme weights on {’, ’.join(violations)}”
return True, “✅ PASS: All weights within [0.3×, 3×]”

# ============================================================================

# SYMMETRY AUDIT

# ============================================================================

def symmetry_audit(fr: Dict, cfg: Dict) -> List[Tuple[str, float, float, float]]:
“””
Test toggle inversions and report ΔYPA.

```
Returns list of (toggle_name, baseline_ypa, flipped_ypa, delta)
"""
def get_ypa(framework, config):
    results, _, _ = ypa_scenario_scores(framework, config)
    return results["Neutral"]["YPA"]

baseline = get_ypa(fr, cfg)
reports = []

# Test 1: Lever-Parity flip
cfg_parity = cfg.copy()
cfg_parity["lever_parity"] = "OFF" if cfg["lever_parity"] == "ON" else "ON"
flipped = get_ypa(fr, cfg_parity)
reports.append(("Lever-Parity", baseline, flipped, flipped - baseline))

# Test 2: PF-Type variations
for pf_type in PF_TYPES:
    if pf_type == cfg["pf_type"]:
        continue
    cfg_pf = cfg.copy()
    cfg_pf["pf_type"] = pf_type
    flipped = get_ypa(fr, cfg_pf)
    reports.append((f"PF-Type→{pf_type}", baseline, flipped, flipped - baseline))

# Test 3: Fallibilism-Bonus flip
cfg_fall = cfg.copy()
cfg_fall["fallibilism_bonus"] = "OFF" if cfg["fallibilism_bonus"] == "ON" else "ON"
flipped = get_ypa(fr, cfg_fall)
reports.append(("Fallibilism-Bonus", baseline, flipped, flipped - baseline))

return reports
```

# ============================================================================

# VISUALIZATION HELPERS

# ============================================================================

def create_lever_comparison_chart(fa_levers: Dict, fb_levers: Dict,
fa_name: str, fb_name: str) -> go.Figure:
“”“Create grouped bar chart comparing lever scores.”””
levers = list(fa_levers.keys())

```
fig = go.Figure()
fig.add_trace(go.Bar(
    name=fa_name,
    x=levers,
    y=[fa_levers[l] for l in levers],
    marker_color='rgb(55, 83, 109)'
))
fig.add_trace(go.Bar(
    name=fb_name,
    x=levers,
    y=[fb_levers[l] for l in levers],
    marker_color='rgb(26, 118, 255)'
))

fig.update_layout(
    title="Lever-by-Lever Comparison",
    xaxis_title="Lever",
    yaxis_title="Score (0-10)",
    barmode='group',
    height=400
)
return fig
```

def create_ypa_trinity_chart(fa_results: Dict, fb_results: Dict,
fa_name: str, fb_name: str) -> go.Figure:
“”“Create grouped bar chart for YPA Trinity.”””
scenarios = [“Neutral”, “Existential”, “Empirical”]

```
fig = go.Figure()
fig.add_trace(go.Bar(
    name=fa_name,
    x=scenarios,
    y=[fa_results[s]["YPA"] for s in scenarios],
    marker_color='rgb(55, 83, 109)'
))
fig.add_trace(go.Bar(
    name=fb_name,
    x=scenarios,
    y=[fb_results[s]["YPA"] for s in scenarios],
    marker_color='rgb(26, 118, 255)'
))

fig.update_layout(
    title="YPA Trinity Comparison",
    xaxis_title="Scenario",
    yaxis_title="YPA (Efficiency)",
    barmode='group',
    height=400
)
return fig
```

# ============================================================================

# STREAMLIT UI

# ============================================================================

def main():
st.set_page_config(
page_title=“CFA v2.0 Console”,
page_icon=“⚖️”,
layout=“wide”,
initial_sidebar_state=“expanded”
)

```
# Custom CSS
st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1f77b4;}
    .sub-header {font-size: 1.5rem; font-weight: bold; color: #ff7f0e;}
    .metric-card {background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;}
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">⚖️ CFA v2.0 Interactive Console</p>', unsafe_allow_html=True)
st.markdown('**"All Named, All Priced" — Now with knobs and dials**')
st.markdown("---")

# ========================================================================
# SIDEBAR: CONFIGURATION TOGGLES
# ========================================================================

st.sidebar.header("🎛️ Configuration Toggles")
st.sidebar.markdown("*Adjust these to see how frameworks respond*")

lever_parity = st.sidebar.selectbox(
    "Lever-Parity",
    ["ON", "OFF"],
    index=0,
    help="ON = Moral norms weighted equal to epistemic | OFF = MG down-weighted 0.5×"
)

pf_type = st.sidebar.selectbox(
    "PF-Type",
    PF_TYPES,
    index=1,  # Composite default
    help="Instrumental = Tech only | Holistic = 50:50 mix | Composite = 70:30 mix"
)

fall_bonus = st.sidebar.selectbox(
    "Fallibilism-Bonus",
    ["ON", "OFF"],
    index=0,
    help="ON = +0.3 CCI for frameworks that admit limits"
)

bfi_weight = st.sidebar.selectbox(
    "BFI Debt Weight",
    ["Equal_1.0x", "Weighted_1.2x"],
    index=0,
    help="Equal = Axioms and debts count same | Weighted = Debts cost 1.2×"
)

cfg = {
    "lever_parity": lever_parity,
    "pf_type": pf_type,
    "fallibilism_bonus": fall_bonus,
    "bfi_debt_weight": bfi_weight
}

# Show current config
st.sidebar.markdown("---")
st.sidebar.markdown("**Current Configuration:**")
st.sidebar.json(cfg)

# ========================================================================
# MAIN AREA: FRAMEWORK EDITORS
# ========================================================================

col1, col2 = st.columns(2)

# Framework A (MdN default)
with col1:
    st.markdown('<p class="sub-header">📘 Framework A</p>', unsafe_allow_html=True)
    
    fa_name = st.text_input("Name", value=MDN_DEFAULT["name"], key="fa_name")
    
    with st.expander("🔢 BFI (Brute-Fact Index)", expanded=False):
        fa_axioms = st.number_input("Axioms", 1, 30, MDN_DEFAULT["bf_i"]["axioms"], key="fa_axioms")
        fa_debts = st.number_input("Debts", 0, 30, MDN_DEFAULT["bf_i"]["debts"], key="fa_debts")
        fa_admits = st.checkbox("Admits Limits (for Fallibilism bonus)", value=True, key="fa_admits")
    
    st.markdown("**Lever Scores (0-10)**")
    fa_cci = st.slider("CCI - Coherence & Closure", 0.0, 10.0, MDN_DEFAULT["levers"]["CCI"], 0.1, key="fa_cci")
    fa_edb = st.slider("EDB - Explanatory Depth & Breadth", 0.0, 10.0, MDN_DEFAULT["levers"]["EDB"], 0.1, key="fa_edb")
    fa_pf_inst = st.slider("PF Instrumental (Tech/Prediction)", 0.0, 10.0, MDN_DEFAULT["levers"]["PF_instrumental"], 0.1, key="fa_pf_inst")
    fa_pf_exist = st.slider("PF Existential (Meaning/Purpose)", 0.0, 10.0, MDN_DEFAULT["levers"]["PF_existential"], 0.1, key="fa_pf_exist")
    fa_ar = st.slider("AR - Aesthetic Resonance", 0.0, 10.0, MDN_DEFAULT["levers"]["AR"], 0.1, key="fa_ar")
    fa_mg = st.slider("MG - Moral Generativity", 0.0, 10.0, MDN_DEFAULT["levers"]["MG"], 0.1, key="fa_mg")

    fa = {
        "name": fa_name,
        "bf_i": {"axioms": fa_axioms, "debts": fa_debts},
        "levers": {
            "CCI": fa_cci,
            "EDB": fa_edb,
            "PF_instrumental": fa_pf_inst,
            "PF_existential": fa_pf_exist,
            "AR": fa_ar,
            "MG": fa_mg
        },
        "admits_limits": fa_admits
    }

# Framework B (CT default)
with col2:
    st.markdown('<p class="sub-header">📕 Framework B</p>', unsafe_allow_html=True)
    
    fb_name = st.text_input("Name", value=CT_DEFAULT["name"], key="fb_name")
    
    with st.expander("🔢 BFI (Brute-Fact Index)", expanded=False):
        fb_axioms = st.number_input("Axioms", 1, 30, CT_DEFAULT["bf_i"]["axioms"], key="fb_axioms")
        fb_debts = st.number_input("Debts", 0, 30, CT_DEFAULT["bf_i"]["debts"], key="fb_debts")
        fb_admits = st.checkbox("Admits Limits (for Fallibilism bonus)", value=True, key="fb_admits")
    
    st.markdown("**Lever Scores (0-10)**")
    fb_cci = st.slider("CCI - Coherence & Closure", 0.0, 10.0, CT_DEFAULT["levers"]["CCI"], 0.1, key="fb_cci")
    fb_edb = st.slider("EDB - Explanatory Depth & Breadth", 0.0, 10.0, CT_DEFAULT["levers"]["EDB"], 0.1, key="fb_edb")
    fb_pf_inst = st.slider("PF Instrumental (Tech/Prediction)", 0.0, 10.0, CT_DEFAULT["levers"]["PF_instrumental"], 0.1, key="fb_pf_inst")
    fb_pf_exist = st.slider("PF Existential (Meaning/Purpose)", 0.0, 10.0, CT_DEFAULT["levers"]["PF_existential"], 0.1, key="fb_pf_exist")
    fb_ar = st.slider("AR - Aesthetic Resonance", 0.0, 10.0, CT_DEFAULT["levers"]["AR"], 0.1, key="fb_ar")
    fb_mg = st.slider("MG - Moral Generativity", 0.0, 10.0, CT_DEFAULT["levers"]["MG"], 0.1, key="fb_mg")

    fb = {
        "name": fb_name,
        "bf_i": {"axioms": fb_axioms, "debts": fb_debts},
        "levers": {
            "CCI": fb_cci,
            "EDB": fb_edb,
            "PF_instrumental": fb_pf_inst,
            "PF_existential": fb_pf_exist,
            "AR": fb_ar,
            "MG": fb_mg
        },
        "admits_limits": fb_admits
    }

st.markdown("---")

# ========================================================================
# RESULTS SECTION
# ========================================================================

st.markdown('<p class="sub-header">📊 Results & Analysis</p>', unsafe_allow_html=True)

# Compute results for both frameworks
ya_results, ya_levers, ya_bfi = ypa_scenario_scores(fa, cfg)
yb_results, yb_levers, yb_bfi = ypa_scenario_scores(fb, cfg)

# Tab interface for different views
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Visual Comparison",
    "📋 Detailed Scores",
    "🛡️ Guardrails",
    "🔄 Symmetry Audit"
])

# TAB 1: VISUAL COMPARISON
with tab1:
    st.plotly_chart(
        create_lever_comparison_chart(ya_levers, yb_levers, fa["name"], fb["name"]),
        use_container_width=True
    )
    
    st.plotly_chart(
        create_ypa_trinity_chart(ya_results, yb_results, fa["name"], fb["name"]),
        use_container_width=True
    )

    # Winner summary
    st.markdown("### 🏆 Scenario Winners")
    winner_df = pd.DataFrame({
        "Scenario": ["Neutral", "Existential", "Empirical"],
        fa["name"]: [
            f"{ya_results['Neutral']['YPA']:.3f}",
            f"{ya_results['Existential']['YPA']:.3f}",
            f"{ya_results['Empirical']['YPA']:.3f}"
        ],
        fb["name"]: [
            f"{yb_results['Neutral']['YPA']:.3f}",
            f"{yb_results['Existential']['YPA']:.3f}",
            f"{yb_results['Empirical']['YPA']:.3f}"
        ],
        "Δ (B-A)": [
            f"{yb_results['Neutral']['YPA'] - ya_results['Neutral']['YPA']:.3f}",
            f"{yb_results['Existential']['YPA'] - ya_results['Existential']['YPA']:.3f}",
            f"{yb_results['Empirical']['YPA'] - ya_results['Empirical']['YPA']:.3f}"
        ]
    })
    st.dataframe(winner_df, use_container_width=True)

# TAB 2: DETAILED SCORES
with tab2:
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown(f"#### {fa['name']}")
        st.markdown(f"**BFI:** {ya_bfi:.2f} ({fa['bf_i']['axioms']} axioms + {fa['bf_i']['debts']} debts)")
        st.markdown("**Adjusted Levers:**")
        st.json(ya_levers)
        st.markdown("**YPA Trinity:**")
        st.table(pd.DataFrame({
            "Scenario": ["Neutral", "Existential", "Empirical"],
            "Total": [
                f"{ya_results['Neutral']['total']:.2f}",
                f"{ya_results['Existential']['total']:.2f}",
                f"{ya_results['Empirical']['total']:.2f}"
            ],
            "YPA": [
                f"{ya_results['Neutral']['YPA']:.3f}",
                f"{ya_results['Existential']['YPA']:.3f}",
                f"{ya_results['Empirical']['YPA']:.3f}"
            ]
        }))

    with col_b:
        st.markdown(f"#### {fb['name']}")
        st.markdown(f"**BFI:** {yb_bfi:.2f} ({fb['bf_i']['axioms']} axioms + {fb['bf_i']['debts']} debts)")
        st.markdown("**Adjusted Levers:**")
        st.json(yb_levers)
        st.markdown("**YPA Trinity:**")
        st.table(pd.DataFrame({
            "Scenario": ["Neutral", "Existential", "Empirical"],
            "Total": [
                f"{yb_results['Neutral']['total']:.2f}",
                f"{yb_results['Existential']['total']:.2f}",
                f"{yb_results['Empirical']['total']:.2f}"
            ],
            "YPA": [
                f"{yb_results['Neutral']['YPA']:.3f}",
                f"{yb_results['Existential']['YPA']:.3f}",
                f"{yb_results['Empirical']['YPA']:.3f}"
            ]
        }))

# TAB 3: GUARDRAILS
with tab3:
    st.markdown("### 🛡️ Automated Fairness Checks")
    
    col_ga, col_gb = st.columns(2)
    
    with col_ga:
        st.markdown(f"#### {fa['name']}")
        
        # Lever-Coupling
        pass_lc, msg_lc = guardrail_lever_coupling(ya_levers["PF"], ya_levers["CCI"])
        st.markdown(f"**Lever-Coupling:** {msg_lc}")
        
        # BFI-Sensitivity (compare to defaults)
        default_ya, _, default_bfi_a = ypa_scenario_scores(MDN_DEFAULT, cfg)
        pass_bfi, msg_bfi = guardrail_bfi_sensitivity(
            ya_results["Neutral"]["YPA"],
            default_ya["Neutral"]["YPA"],
            ya_bfi,
            default_bfi_a
        )
        st.markdown(f"**BFI-Sensitivity:** {msg_bfi}")

    with col_gb:
        st.markdown(f"#### {fb['name']}")
        
        # Lever-Coupling
        pass_lc, msg_lc = guardrail_lever_coupling(yb_levers["PF"], yb_levers["CCI"])
        st.markdown(f"**Lever-Coupling:** {msg_lc}")
        
        # BFI-Sensitivity
        default_yb, _, default_bfi_b = ypa_scenario_scores(CT_DEFAULT, cfg)
        pass_bfi, msg_bfi = guardrail_bfi_sensitivity(
            yb_results["Neutral"]["YPA"],
            default_yb["Neutral"]["YPA"],
            yb_bfi,
            default_bfi_b
        )
        st.markdown(f"**BFI-Sensitivity:** {msg_bfi}")

# TAB 4: SYMMETRY AUDIT
with tab4:
    st.markdown("### 🔄 Toggle Impact Analysis")
    st.markdown("*How does each framework respond to configuration changes?*")
    
    col_sa, col_sb = st.columns(2)
    
    with col_sa:
        st.markdown(f"#### {fa['name']}")
        audit_a = symmetry_audit(fa, cfg)
        df_a = pd.DataFrame(audit_a, columns=["Toggle", "Baseline", "Flipped", "ΔYPA"])
        df_a["Flag"] = df_a["ΔYPA"].apply(lambda x: "⚠️" if abs(x) > 0.3 else "✅")
        st.dataframe(df_a, use_container_width=True)

    with col_sb:
        st.markdown(f"#### {fb['name']}")
        audit_b = symmetry_audit(fb, cfg)
        df_b = pd.DataFrame(audit_b, columns=["Toggle", "Baseline", "Flipped", "ΔYPA"])
        df_b["Flag"] = df_b["ΔYPA"].apply(lambda x: "⚠️" if abs(x) > 0.3 else "✅")
        st.dataframe(df_b, use_container_width=True)

# ========================================================================
# FOOTER: EXPORT & INFO
# ========================================================================

st.markdown("---")
st.markdown("### 💾 Export Configuration")

export_data = {
    "config": cfg,
    "framework_a": fa,
    "framework_b": fb,
    "results": {
        "a": {
            "levers": ya_levers,
            "bfi": ya_bfi,
            "ypa": {k: v["YPA"] for k, v in ya_results.items()}
        },
        "b": {
            "levers": yb_levers,
            "bfi": yb_bfi,
            "ypa": {k: v["YPA"] for k, v in yb_results.items()}
        }
    }
}

st.download_button(
    label="📥 Download Full Report (JSON)",
    data=json.dumps(export_data, indent=2),
    file_name="cfa_v2_report.json",
    mime="application/json"
)

st.markdown("---")
st.caption("CFA v2.0 | 'All Named, All Priced' | Built with Streamlit")
```

if **name** == “**main**”:
main()
