"""
Streamlit app for the H2 vs Jet-A combustor model.
Run with:  streamlit run app.py
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# import the model (must be in same folder)
from combustor_model import (
    adiabatic_exhaust_temperature,
    fuel_mass_flow,
    lhv_molar,
    lh2_penalty_per_kg,
    LHV_KJ_PER_KG,
    MOLAR_MASS,
    LH2_PRECONDITIONING_KJ_PER_MOL,
)

# ---------------------------------------------------------------
# page config
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Combustor Sizing Tool",
    page_icon="🔥",
    layout="wide",
)

st.title("🔥 Combustor Sizing Tool — H₂ vs Jet-A")
st.caption("0D adiabatic combustor model with Peng-Robinson real-gas correction")
st.divider()

# ---------------------------------------------------------------
# sidebar inputs
# ---------------------------------------------------------------
st.sidebar.header("⚙️ Inputs")

fuel = st.sidebar.selectbox("Fuel type", ["H2", "JetA"],
                             format_func=lambda x: "Hydrogen (H₂)" if x == "H2" else "Jet-A (C₁₂H₂₃)")

P_bar = st.sidebar.slider("Combustor pressure (bar)", 1.0, 60.0, 30.0, 0.5)

lambda_air = st.sidebar.slider("Excess air ratio λ", 1.0, 10.0, 3.0, 0.1,
                                help="1.0 = stoichiometric, >1 = lean")

target_power = st.sidebar.number_input("Target power (kW)", min_value=1.0, value=201.0, step=10.0)

efficiency = st.sidebar.slider("Combustion efficiency η", 0.5, 1.0, 1.0, 0.01)

real_gas = st.sidebar.toggle("Real-gas correction (PR EOS)", value=True)

include_lh2_cond = False
if fuel == "H2":
    include_lh2_cond = st.sidebar.toggle("Include LH₂ conditioning penalty", value=False,
                                          help="Adds ~4415 kJ/kg penalty for cryogenic heating")

st.sidebar.divider()
st.sidebar.caption("Air inlet temperature: 600 K (compressor exit)")
st.sidebar.caption("Fuel inlet temperature: 298.15 K")

# ---------------------------------------------------------------
# compute outputs
# ---------------------------------------------------------------
try:
    T_out = adiabatic_exhaust_temperature(fuel, lambda_air, P_bar, real_gas)
    T_ideal = adiabatic_exhaust_temperature(fuel, lambda_air, P_bar, real_gas=False)
    mdot = fuel_mass_flow(target_power, fuel, efficiency, include_lh2_cond)
    solver_ok = True
except Exception as e:
    solver_ok = False
    st.error(f"Solver error: {e}")

# ---------------------------------------------------------------
# top metric cards
# ---------------------------------------------------------------
if solver_ok:
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Exhaust Temperature",
        f"{T_out:.1f} K",
        delta=f"{T_out - T_ideal:+.2f} K vs ideal" if real_gas else "ideal-gas mode",
        delta_color="off",
    )
    c2.metric(
        "Fuel Mass Flow",
        f"{mdot*1000:.4f} g/s",
        delta=f"{mdot*3600:.3f} kg/hr",
        delta_color="off",
    )
    c3.metric(
        "LHV (mass basis)",
        f"{LHV_KJ_PER_KG[fuel]/1000:.1f} MJ/kg",
    )
    c4.metric(
        "LHV (molar basis)",
        f"{lhv_molar(fuel):.1f} kJ/mol",
    )

    st.divider()

# ---------------------------------------------------------------
# plots
# ---------------------------------------------------------------
col_left, col_right = st.columns(2)

# Plot 1: T vs lambda
with col_left:
    st.subheader("Exhaust Temperature vs λ")

    lam_range = np.linspace(1.0, 8.0 if fuel == "H2" else 5.0, 40)

    T_real_arr  = [adiabatic_exhaust_temperature(fuel, l, P_bar, True)  for l in lam_range]
    T_ideal_arr = [adiabatic_exhaust_temperature(fuel, l, P_bar, False) for l in lam_range]

    fig1, ax1 = plt.subplots(figsize=(7, 4.5))
    color = "#2563EB" if fuel == "H2" else "#D97706"
    ax1.plot(lam_range, T_real_arr,  color=color, label="Real-gas")
    ax1.plot(lam_range, T_ideal_arr, color=color, linestyle="--", alpha=0.6, label="Ideal-gas")

    # mark selected lambda
    if solver_ok:
        ax1.axvline(lambda_air, color="gray", linestyle=":", linewidth=1)
        ax1.scatter([lambda_air], [T_out], color=color, zorder=5, s=60)
        ax1.annotate(f"{T_out:.0f} K", (lambda_air, T_out),
                     textcoords="offset points", xytext=(8, 4), fontsize=9, color=color)

    ax1.set_xlabel("Excess air ratio λ")
    ax1.set_ylabel("T_exhaust (K)")
    ax1.set_title(f"{fuel}  |  P = {P_bar:.0f} bar")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

# Plot 2: real vs ideal difference
with col_right:
    st.subheader("Real-gas vs Ideal-gas Deviation (ΔT)")

    delta_arr = np.array(T_real_arr) - np.array(T_ideal_arr)

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    ax2.plot(lam_range, delta_arr, color="#DC2626")
    ax2.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax2.fill_between(lam_range, delta_arr, 0, alpha=0.12, color="#DC2626")

    # mark selected lambda
    if solver_ok:
        dT_at_lam = T_out - T_ideal
        ax2.axvline(lambda_air, color="gray", linestyle=":", linewidth=1)
        ax2.scatter([lambda_air], [dT_at_lam], color="#DC2626", zorder=5, s=60)
        ax2.annotate(f"{dT_at_lam:.3f} K", (lambda_air, dT_at_lam),
                     textcoords="offset points", xytext=(8, 4), fontsize=9, color="#DC2626")

    ax2.set_xlabel("Excess air ratio λ")
    ax2.set_ylabel("T_real − T_ideal  (K)")
    ax2.set_title(f"PR-EOS correction  |  {fuel}  |  P = {P_bar:.0f} bar")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

st.divider()

# ---------------------------------------------------------------
# fuel comparison table
# ---------------------------------------------------------------
st.subheader("Fuel Comparison at Current Conditions")

T_h2   = adiabatic_exhaust_temperature("H2",   lambda_air, P_bar, real_gas)
T_jet  = adiabatic_exhaust_temperature("JetA",  lambda_air, P_bar, real_gas)
m_h2   = fuel_mass_flow(target_power, "H2",   efficiency, False)
m_jet  = fuel_mass_flow(target_power, "JetA",  efficiency, False)

comp_data = {
    "Property": [
        "Exhaust temperature (K)",
        "LHV (kJ/kg)",
        "LHV (kJ/mol)",
        f"Mass flow at {target_power:.0f} kW (g/s)",
        "Mass flow ratio (JetA / H₂)",
    ],
    "H₂": [
        f"{T_h2:.1f}",
        f"{LHV_KJ_PER_KG['H2']:,.0f}",
        f"{lhv_molar('H2'):.1f}",
        f"{m_h2*1000:.4f}",
        "1.000 ×",
    ],
    "Jet-A": [
        f"{T_jet:.1f}",
        f"{LHV_KJ_PER_KG['JetA']:,.0f}",
        f"{lhv_molar('JetA'):.1f}",
        f"{m_jet*1000:.4f}",
        f"{m_jet/m_h2:.3f} ×",
    ],
}

st.table(comp_data)

# ---------------------------------------------------------------
# LH2 conditioning box (shown only for H2)
# ---------------------------------------------------------------
if fuel == "H2":
    with st.expander("ℹ️ LH₂ Conditioning Penalty Details"):
        pen_kj_mol = LH2_PRECONDITIONING_KJ_PER_MOL
        pen_kj_kg  = lh2_penalty_per_kg()
        eff_lhv    = LHV_KJ_PER_KG["H2"] - pen_kj_kg
        pct        = 100 * pen_kj_kg / LHV_KJ_PER_KG["H2"]

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Penalty (kJ/mol)", f"{pen_kj_mol:.2f}")
        col_b.metric("Penalty (kJ/kg)",  f"{pen_kj_kg:.1f}")
        col_c.metric("% of LHV",         f"{pct:.2f}%")

        st.caption(
            "This is the energy needed to bring LH₂ from cryogenic storage (~20 K) "
            "up to gas phase at 298 K before entering the combustor. "
            "Breakdown: vaporisation ≈ 0.45 kJ/mol + sensible heating 20→298 K ≈ 8.45 kJ/mol."
        )

# ---------------------------------------------------------------
# footer
# ---------------------------------------------------------------
st.divider()
st.caption(
    "Model assumptions: steady-flow 0D control volume · constant pressure · "
    "complete combustion (no dissociation) · adiabatic · PR-EOS real-gas correction per species · "
    "air inlet T = 600 K · fuel inlet T = 298.15 K"
)
