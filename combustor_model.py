from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- constants ---
R = 8.314462618       # J/mol/K
T_REF = 298.15        # reference temp (K)
T_AIR_INLET = 600.0   # air comes in preheated from compressor
T_FUEL_INLET = 298.15 # fuel enters at ambient ref temp
AIR_O2_TO_N2 = 3.76   # standard air composition ratio

LH2_PRECONDITIONING_KJ_PER_MOL = 8.90  # energy to bring LH2 from cryo to gas phase

LHV_KJ_PER_KG = {
    "H2":   119_450.0,
    "JetA":  43_280.0,
}

MOLAR_MASS = {
    "H2":   2.01588e-3,
    "O2":  31.9988e-3,
    "N2":  28.0134e-3,
    "H2O": 18.01528e-3,
    "CO2": 44.0095e-3,
    "JetA": 167.316e-3,  # C12H23 pseudo molecule
}

# Jet-A treated as C12H23
JET_A_C = 12
JET_A_H = 23

# Shomate coefficients for sensible enthalpy (relative to 298.15 K)
# order: [A, B, C, D, E, F, H]
SHOMATE = {
    "N2":  [26.09200,  8.218801, -1.976141,  0.159274,  0.044434,  -7.989230,   0.0],
    "O2":  [29.65900,  6.777380, -1.688820,  0.112250, -0.088878, -11.320500,   0.0],
    "H2":  [33.066178,-11.363417, 11.432816, -2.772874, -0.158558,  -9.980797,  0.0],
    "H2O": [30.09200,  6.832514,  6.793435, -2.534480,  0.082139,-250.881000,-241.8264],
    "CO2": [24.99735,  55.18696, -33.69137,  7.948387, -0.136638,-403.607500,-393.5224],
}

# critical properties for PR EOS: (Tc K, Pc bar, acentric factor omega)
CRITICAL = {
    "N2":  (126.2,    33.98,   0.0372),
    "O2":  (154.6,    50.43,   0.0222),
    "H2":  ( 33.19,   12.96,  -0.2190),
    "H2O": (647.096, 220.64,   0.3440),
    "CO2": (304.1282, 73.773,  0.2250),
}


# ---------------------------------------------------------------
# Thermodynamic property functions
# ---------------------------------------------------------------

def shomate_sensible_enthalpy(species: str, T_K: float) -> float:
    """sensible enthalpy relative to 298.15 K, kJ/mol"""
    A, B, C, D, E, F, H = SHOMATE[species]
    t = T_K / 1000.0
    return A*t + B*t**2/2 + C*t**3/3 + D*t**4/4 - E/t + F - H


def pr_residual_enthalpy(species: str, T_K: float, P_bar: float) -> float:
    """Peng-Robinson residual enthalpy for real-gas correction, kJ/mol"""
    if species not in CRITICAL:
        return 0.0

    Tc, Pc_bar, omega = CRITICAL[species]
    P  = P_bar  * 1e5
    Pc = Pc_bar * 1e5

    kappa = 0.37464 + 1.54226*omega - 0.26992*omega**2
    m     = 1.0 + kappa * (1.0 - math.sqrt(T_K / Tc))
    alpha = m**2

    a0 = 0.45724 * R**2 * Tc**2 / Pc
    b  = 0.07780 * R  * Tc      / Pc
    a  = a0 * alpha

    dalpha_dT = -kappa * m / math.sqrt(T_K * Tc)
    da_dT     = a0 * dalpha_dT

    A = a * P / (R**2 * T_K**2)
    B = b * P / (R   * T_K)

    coeffs     = [1.0, -(1.0 - B), A - 2*B - 3*B**2, -(A*B - B**2 - B**3)]
    roots      = np.roots(coeffs)
    real_roots = roots[np.isreal(roots)].real

    if real_roots.size == 0:
        return 0.0

    Z      = float(np.max(real_roots))
    denom1 = Z + (1.0 + math.sqrt(2.0)) * B
    denom2 = Z + (1.0 - math.sqrt(2.0)) * B

    if denom1 <= 0 or denom2 <= 0 or B <= 0:
        return 0.0

    log_term       = math.log(denom1 / denom2)
    H_res_J_per_mol = R*T_K*(Z - 1) + ((T_K*da_dT - a) / (2*math.sqrt(2)*b)) * log_term
    return H_res_J_per_mol / 1000.0


def species_enthalpy(species: str, T_K: float, P_bar: float, real_gas: bool = True) -> float:
    """total molar enthalpy for one species, kJ/mol"""
    h = shomate_sensible_enthalpy(species, T_K)
    if real_gas:
        h += pr_residual_enthalpy(species, T_K, P_bar)
    return h


def mixture_enthalpy(moles: dict, T_K: float, P_bar: float, real_gas: bool = True) -> float:
    """total enthalpy of a gas mixture, kJ"""
    return sum(n * species_enthalpy(sp, T_K, P_bar, real_gas) for sp, n in moles.items())


# ---------------------------------------------------------------
# Stoichiometry helpers
# ---------------------------------------------------------------

def stoich_o2(fuel: str) -> float:
    if fuel == "H2":
        return 0.5
    if fuel == "JetA":
        return JET_A_C + JET_A_H / 4.0
    raise ValueError(f"Unknown fuel: {fuel}")


def combustion_products(fuel: str, lambda_air: float) -> dict:
    """products dict (moles) for complete combustion at given lambda"""
    if lambda_air < 1.0:
        raise ValueError("lambda must be >= 1 (lean/stoich only)")

    o2_in    = lambda_air * stoich_o2(fuel)
    n2_in    = AIR_O2_TO_N2 * o2_in
    excess   = o2_in - stoich_o2(fuel)

    if fuel == "H2":
        return {"H2O": 1.0, "O2": excess, "N2": n2_in}
    return {"CO2": float(JET_A_C), "H2O": JET_A_H / 2.0, "O2": excess, "N2": n2_in}


def air_moles(fuel: str, lambda_air: float) -> dict:
    o2_in = lambda_air * stoich_o2(fuel)
    return {"O2": o2_in, "N2": AIR_O2_TO_N2 * o2_in}


# ---------------------------------------------------------------
# LHV and fuel flow
# ---------------------------------------------------------------

def lhv_molar(fuel: str) -> float:
    """LHV in kJ/mol"""
    return LHV_KJ_PER_KG[fuel] * MOLAR_MASS[fuel]


def lh2_penalty_per_kg() -> float:
    """conditioning penalty for LH2, kJ/kg"""
    return LH2_PRECONDITIONING_KJ_PER_MOL / MOLAR_MASS["H2"]


def fuel_mass_flow(power_kW: float, fuel: str,
                   efficiency: float = 1.0,
                   include_lh2_cond: bool = False) -> float:
    """mass flow in kg/s for a target shaft power"""
    lhv = LHV_KJ_PER_KG[fuel]
    if fuel == "H2" and include_lh2_cond:
        lhv -= lh2_penalty_per_kg()
    return power_kW / (efficiency * lhv)


# ---------------------------------------------------------------
# Adiabatic exhaust temperature solver
# ---------------------------------------------------------------

def reactant_enthalpy(fuel: str, lambda_air: float, P_bar: float,
                      real_gas: bool = True) -> float:
    H_air  = mixture_enthalpy(air_moles(fuel, lambda_air), T_AIR_INLET, P_bar, real_gas)
    H_fuel = mixture_enthalpy({"H2": 1.0}, T_FUEL_INLET, P_bar, real_gas) if fuel == "H2" else 0.0
    return H_air + H_fuel


def adiabatic_exhaust_temperature(fuel: str, lambda_air: float, P_bar: float,
                                   real_gas: bool = True) -> float:
    """bisection solver: find T where H_products = H_reactants + LHV"""
    H_target = reactant_enthalpy(fuel, lambda_air, P_bar, real_gas) + lhv_molar(fuel)

    def residual(T):
        return mixture_enthalpy(combustion_products(fuel, lambda_air), T, P_bar, real_gas) - H_target

    lo, hi = 500.0, 4500.0
    if residual(lo) * residual(hi) > 0:
        lo, hi = 300.0, 6000.0
        if residual(lo) * residual(hi) > 0:
            raise ValueError(f"Solver bracket failed for {fuel} λ={lambda_air} P={P_bar}")

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm  = residual(mid)
        if abs(fm) < 1e-3:
            return mid
        if residual(lo) * fm <= 0:
            hi = mid
        else:
            lo = mid

    return 0.5 * (lo + hi)


# ---------------------------------------------------------------
# Plotting (saves to ./outputs/)
# ---------------------------------------------------------------

def get_output_dir() -> Path:
    base = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    d = base / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def plot_temperature_vs_lambda(P_bar: float = 30.0) -> Path:
    lam_h2  = np.linspace(1.0, 8.0, 30)
    lam_jet = np.linspace(1.0, 4.0, 30)

    h2_real  = [adiabatic_exhaust_temperature("H2",   l, P_bar, True)  for l in lam_h2]
    h2_ideal = [adiabatic_exhaust_temperature("H2",   l, P_bar, False) for l in lam_h2]
    jet_real  = [adiabatic_exhaust_temperature("JetA", l, P_bar, True)  for l in lam_jet]
    jet_ideal = [adiabatic_exhaust_temperature("JetA", l, P_bar, False) for l in lam_jet]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(lam_h2,  h2_real,  label="H₂  real-gas",  color="#2563EB")
    ax.plot(lam_h2,  h2_ideal, label="H₂  ideal-gas", color="#2563EB", linestyle="--")
    ax.plot(lam_jet, jet_real,  label="Jet-A real-gas",  color="#D97706")
    ax.plot(lam_jet, jet_ideal, label="Jet-A ideal-gas", color="#D97706", linestyle="--")
    ax.set_xlabel("Excess air ratio λ")
    ax.set_ylabel("Adiabatic exhaust temperature (K)")
    ax.set_title(f"Combustor exit temperature vs λ  (P = {P_bar:.1f} bar)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = get_output_dir() / "temperature_vs_lambda.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_real_vs_ideal_diff(P_bar: float = 30.0, fuel: str = "H2") -> Path:
    lams  = np.linspace(1.0, 8.0 if fuel == "H2" else 4.0, 30)
    T_r   = np.array([adiabatic_exhaust_temperature(fuel, l, P_bar, True)  for l in lams])
    T_i   = np.array([adiabatic_exhaust_temperature(fuel, l, P_bar, False) for l in lams])
    delta = T_r - T_i

    fig, ax = plt.subplots(figsize=(9, 5))
    color = "#2563EB" if fuel == "H2" else "#D97706"
    ax.plot(lams, delta, color=color)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Excess air ratio λ")
    ax.set_ylabel("T_real − T_ideal  (K)")
    ax.set_title(f"Real-gas vs ideal-gas deviation  —  {fuel}  (P = {P_bar:.1f} bar)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = get_output_dir() / f"real_vs_ideal_{fuel}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# ---------------------------------------------------------------
# CLI interface (forward solver + mass flow + plots)
# ---------------------------------------------------------------

def ask_float(prompt, default=None, lo=None, hi=None):
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return default
        try:
            v = float(raw)
        except ValueError:
            print("  Enter a valid number.")
            continue
        if lo is not None and v < lo:
            print(f"  Must be >= {lo}")
            continue
        if hi is not None and v > hi:
            print(f"  Must be <= {hi}")
            continue
        return v


def ask_choice(prompt, choices, default=None):
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return default
        if raw in choices:
            return raw
        print(f"  Choose from: {', '.join(sorted(choices))}")


def run_cli():
    print("=" * 60)
    print("  COMBUSTOR SIZING TOOL  —  H2 vs Jet-A")
    print("=" * 60)

    while True:
        print("\n  [1] Forward solver  (λ, P → T_exhaust)")
        print("  [2] Fuel mass flow  (power → kg/s)")
        print("  [3] Save plots")
        print("  [0] Exit")

        mode = ask_choice("\n  Choice: ", {"0", "1", "2", "3"}, default="0")

        if mode == "0":
            break

        elif mode == "1":
            fuel = "H2" if ask_choice("  Fuel [1=H2, 2=JetA]: ", {"1","2"}, "1") == "1" else "JetA"
            lam  = ask_float("  Excess air ratio λ: ", default=3.0, lo=1.0)
            P    = ask_float("  Pressure (bar): ", default=30.0, lo=0.1)
            rg   = ask_choice("  Real gas? [1=yes, 0=no]: ", {"0","1"}, "1") == "1"
            T    = adiabatic_exhaust_temperature(fuel, lam, P, rg)
            print(f"\n  → Exhaust temperature: {T:.2f} K")

        elif mode == "2":
            fuel = "H2" if ask_choice("  Fuel [1=H2, 2=JetA]: ", {"1","2"}, "1") == "1" else "JetA"
            pwr  = ask_float("  Target power (kW): ", default=201.0, lo=0.1)
            eff  = ask_float("  Efficiency (0-1): ",  default=1.0, lo=1e-4, hi=1.0)
            cond = False
            if fuel == "H2":
                cond = ask_choice("  Include LH2 conditioning? [1=yes, 0=no]: ", {"0","1"}, "0") == "1"
            mdot = fuel_mass_flow(pwr, fuel, eff, cond)
            print(f"\n  → Mass flow: {mdot:.6f} kg/s")

        elif mode == "3":
            P = ask_float("  Pressure (bar): ", default=30.0, lo=0.1)
            p1 = plot_temperature_vs_lambda(P)
            p2 = plot_real_vs_ideal_diff(P, "H2")
            p3 = plot_real_vs_ideal_diff(P, "JetA")
            print(f"\n  Saved:\n    {p1}\n    {p2}\n    {p3}")


if __name__ == "__main__":
    run_cli()
