"""
Combustor Sizing Tool — H2 vs Jet-A
Streamlit front-end.  Model logic lives in combustor_model.py.
Run:  streamlit run app.py
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st

from combustor_model import (
    adiabatic_exhaust_temperature,
    fuel_mass_flow,
    lhv_molar,
    LHV_KJ_PER_KG,
)

# ── page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Combustor Sizing Tool",
    layout="wide",
)

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.facecolor": "none",
    "axes.facecolor": "none",
})

# ── title ─────────────────────────────────────────────────────
st.markdown("## Combustor Sizing Tool")
st.markdown(
    "0D adiabatic model · Peng-Robinson real-gas correction · H₂ vs Jet-A comparison"
)
st.divider()

# ── input section ─────────────────────────────────────────────
st.markdown("### Inputs")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Operating conditions**")
    P_bar = st.number_input(
        "Combustor pressure (bar)",
        min_value=1.0, max_value=100.0, value=30.0, step=0.5,
    )
    lambda_air = st.number_input(
        "Excess air ratio λ  (reference point)",
        min_value=1.0, max_value=15.0, value=3.0, step=0.1,
        help="Used for the single-point metrics. Plots always sweep the full λ range.",
    )

with col2:
    st.markdown("**Power and efficiency**")
    target_power = st.number_input(
        "Target power (kW)",
        min_value=1.0, max_value=500_000.0, value=201.0, step=10.0,
    )
    efficiency = st.number_input(
        "Combustion efficiency η",
        min_value=0.01, max_value=1.0, value=1.0, step=0.01,
    )

with col3:
    st.markdown("**Model options**")
    real_gas = st.checkbox("Real-gas correction (PR-EOS)", value=True)
    include_lh2 = st.checkbox(
        "Include LH₂ conditioning penalty",
        value=False,
        help="Adds ~4 415 kJ/kg energy penalty for bringing cryogenic H₂ to 298 K.",
    )
    st.caption("Air inlet: 600 K · Fuel inlet: 298.15 K")

st.divider()

run = st.button("Calculate", type="primary", use_container_width=False)

# ── calculation ───────────────────────────────────────────────
if run:
    try:
        # single-point temperatures
        T_h2   = adiabatic_exhaust_temperature("H2",   lambda_air, P_bar, real_gas)
        T_jeta = adiabatic_exhaust_temperature("JetA", lambda_air, P_bar, real_gas)
        T_h2_ideal   = adiabatic_exhaust_temperature("H2",   lambda_air, P_bar, False)
        T_jeta_ideal = adiabatic_exhaust_temperature("JetA", lambda_air, P_bar, False)

        # fuel mass flows
        m_h2   = fuel_mass_flow(target_power, "H2",   efficiency, include_lh2)
        m_jeta = fuel_mass_flow(target_power, "JetA", efficiency, False)

        # lambda sweep for plots
        lam_h2   = np.linspace(1.05, 8.0, 60)
        lam_jeta = np.linspace(1.05, 5.0, 60)

        T_h2_sweep   = [adiabatic_exhaust_temperature("H2",   l, P_bar, real_gas) for l in lam_h2]
        T_jeta_sweep = [adiabatic_exhaust_temperature("JetA", l, P_bar, real_gas) for l in lam_jeta]
        T_h2_ig      = [adiabatic_exhaust_temperature("H2",   l, P_bar, False) for l in lam_h2]
        T_jeta_ig    = [adiabatic_exhaust_temperature("JetA", l, P_bar, False) for l in lam_jeta]

    except Exception as exc:
        st.error(f"Solver error — {exc}")
        st.stop()

    # ── metric cards ──────────────────────────────────────────
    st.markdown("### Results")
    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "H₂ exhaust temperature",
        f"{T_h2:.1f} K",
        delta=f"{T_h2 - T_h2_ideal:+.2f} K  (real vs ideal)",
        delta_color="off",
    )
    m2.metric(
        "Jet-A exhaust temperature",
        f"{T_jeta:.1f} K",
        delta=f"{T_jeta - T_jeta_ideal:+.2f} K  (real vs ideal)",
        delta_color="off",
    )
    m3.metric(
        "H₂ mass flow",
        f"{m_h2 * 1000:.4f} g/s",
        delta=f"LHV = {LHV_KJ_PER_KG['H2'] / 1000:.0f} MJ/kg",
        delta_color="off",
    )
    m4.metric(
        "Jet-A mass flow",
        f"{m_jeta * 1000:.4f} g/s",
        delta=f"LHV = {LHV_KJ_PER_KG['JetA'] / 1000:.1f} MJ/kg",
        delta_color="off",
    )

    st.divider()

    # ── plots ─────────────────────────────────────────────────
    p_left, p_right = st.columns(2)

    # Plot 1 — T vs λ both fuels
    with p_left:
        st.markdown("**Exhaust temperature vs excess air ratio**")
        fig, ax = plt.subplots(figsize=(6.5, 4.2))

        ax.plot(lam_h2,   T_h2_sweep,   color="#2563EB", lw=2,   label="H₂  (real-gas)")
        ax.plot(lam_h2,   T_h2_ig,      color="#2563EB", lw=1.2, ls="--", alpha=0.55, label="H₂  (ideal)")
        ax.plot(lam_jeta, T_jeta_sweep, color="#D97706", lw=2,   label="Jet-A  (real-gas)")
        ax.plot(lam_jeta, T_jeta_ig,    color="#D97706", lw=1.2, ls="--", alpha=0.55, label="Jet-A  (ideal)")

        # mark reference λ
        ax.axvline(lambda_air, color="#64748B", lw=0.9, ls=":")
        ax.scatter([lambda_air, lambda_air], [T_h2, T_jeta],
                   color=["#2563EB", "#D97706"], zorder=6, s=55)
        ax.annotate(f"{T_h2:.0f} K",   (lambda_air, T_h2),
                    xytext=(7, 4), textcoords="offset points", fontsize=8.5, color="#2563EB")
        ax.annotate(f"{T_jeta:.0f} K", (lambda_air, T_jeta),
                    xytext=(7, -14), textcoords="offset points", fontsize=8.5, color="#D97706")

        ax.set_xlabel("Excess air ratio  λ")
        ax.set_ylabel("T_exhaust  (K)")
        ax.set_title(f"P = {P_bar:.0f} bar")
        ax.legend(fontsize=8.5)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Plot 2 — real vs ideal deviation
    with p_right:
        st.markdown("**Real-gas vs ideal-gas deviation  (ΔT)**")
        delta_h2   = np.array(T_h2_sweep)   - np.array(T_h2_ig)
        delta_jeta = np.array(T_jeta_sweep) - np.array(T_jeta_ig)

        fig2, ax2 = plt.subplots(figsize=(6.5, 4.2))
        ax2.plot(lam_h2,   delta_h2,   color="#2563EB", lw=2,  label="H₂")
        ax2.plot(lam_jeta, delta_jeta, color="#D97706", lw=2,  label="Jet-A")
        ax2.axhline(0, color="#94A3B8", lw=0.8)
        ax2.fill_between(lam_h2,   delta_h2,   0, alpha=0.08, color="#2563EB")
        ax2.fill_between(lam_jeta, delta_jeta, 0, alpha=0.08, color="#D97706")

        ax2.set_xlabel("Excess air ratio  λ")
        ax2.set_ylabel("T_real − T_ideal  (K)")
        ax2.set_title(f"PR-EOS correction  |  P = {P_bar:.0f} bar")
        ax2.legend(fontsize=8.5)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    st.divider()

    # ── comparison table ──────────────────────────────────────
    st.markdown("### Fuel comparison at λ = {:.1f},  P = {:.0f} bar".format(lambda_air, P_bar))

    mass_ratio = m_jeta / m_h2
    reduction  = (1 - m_h2 / m_jeta) * 100

    table_data = {
        "Property": [
            "Exhaust temperature (K)",
            "Real-gas correction ΔT (K)",
            "LHV (MJ/kg)",
            f"Mass flow at {target_power:.0f} kW  (g/s)",
            "Mass flow ratio  (Jet-A / H₂)",
            "H₂ mass saving vs Jet-A",
        ],
        "H₂": [
            f"{T_h2:.1f}",
            f"{T_h2 - T_h2_ideal:+.3f}",
            f"{LHV_KJ_PER_KG['H2'] / 1000:.2f}",
            f"{m_h2 * 1000:.4f}",
            "1.00 ×  (reference)",
            "—",
        ],
        "Jet-A": [
            f"{T_jeta:.1f}",
            f"{T_jeta - T_jeta_ideal:+.3f}",
            f"{LHV_KJ_PER_KG['JetA'] / 1000:.2f}",
            f"{m_jeta * 1000:.4f}",
            f"{mass_ratio:.3f} ×",
            f"−{reduction:.1f} %",
        ],
    }

    st.table(table_data)

    st.divider()

    # ── conclusion ────────────────────────────────────────────
    st.markdown("### Observations")

    delta_T = T_h2 - T_jeta
    sign    = "higher" if delta_T > 0 else "lower"

    st.markdown(
        f"""
- **Exhaust temperature:** At λ = {lambda_air:.1f} and {P_bar:.0f} bar, hydrogen produces an exhaust
  temperature **{abs(delta_T):.0f} K {sign}** than Jet-A. Because H₂ has a higher adiabatic flame
  temperature, more excess air (higher λ) is needed to meet the same turbine inlet temperature limit.

- **Fuel mass flow:** Hydrogen requires **{reduction:.1f}% less mass** per unit power than Jet-A,
  directly reflecting its higher LHV ({LHV_KJ_PER_KG['H2']/1000:.0f} MJ/kg vs
  {LHV_KJ_PER_KG['JetA']/1000:.1f} MJ/kg). This is the primary weight advantage of H₂ in aviation.

- **Real-gas correction:** The PR-EOS correction shifts the H₂ exhaust temperature by
  **{T_h2 - T_h2_ideal:+.3f} K** and Jet-A by **{T_jeta - T_jeta_ideal:+.3f} K** at these conditions.
  The correction grows with pressure and becomes significant at combustor pressures above 30 bar.

- **Model basis:** Steady-flow adiabatic 0D control volume. No dissociation, no heat loss.
  Results represent an upper bound on exhaust temperature — real combustors will be slightly lower
  due to heat transfer and incomplete combustion.
        """
    )

else:
    st.info("Enter your inputs above and press **Calculate** to run the model.")
