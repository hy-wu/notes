from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from CoolProp.CoolProp import PropsSI
except ImportError as exc:
    raise SystemExit(
        "CoolProp is required for argon_real_reference.py. Install it with `python -m pip install CoolProp`."
    ) from exc


RESULT_DIR = Path("argon_results")
SUMMARY_CSV = RESULT_DIR / "argon_equilibrium_summary.csv"
SCAN_CSV = RESULT_DIR / "argon_compressibility_scan.csv"

ARGON_SIGMA_M = 3.405e-10
ARGON_EPSILON_OVER_K_K = 119.8
ARGON_MOLAR_MASS_KG_PER_MOL = 39.948e-3
AVOGADRO = 6.02214076e23
BOLTZMANN = 1.380649e-23

PRESSURE_SCALE_PA = BOLTZMANN * ARGON_EPSILON_OVER_K_K / (ARGON_SIGMA_M**3)
KAPPA_SCALE_PA_INV = 1.0 / PRESSURE_SCALE_PA
TARGET_RHO_STAR = 0.8
TARGET_T_K = ARGON_EPSILON_OVER_K_K


def reduced_density_to_mass_density(rho_star):
    particle_mass = ARGON_MOLAR_MASS_KG_PER_MOL / AVOGADRO
    return rho_star * particle_mass / (ARGON_SIGMA_M**3)


def reduced_temperature_to_kelvin(temp_star):
    return temp_star * ARGON_EPSILON_OVER_K_K


def reduced_pressure_to_mpa(pressure_star):
    return pressure_star * PRESSURE_SCALE_PA / 1.0e6


def reduced_kappa_to_gpa_inv(kappa_star):
    return kappa_star * KAPPA_SCALE_PA_INV * 1.0e9


def coolprop_state(temp_k, rho_mass):
    return {
        "P_MPa": PropsSI("P", "T", temp_k, "Dmass", rho_mass, "Argon") / 1.0e6,
        "Z": PropsSI("Z", "T", temp_k, "Dmass", rho_mass, "Argon"),
        "kappa_GPa_inv": PropsSI("ISOTHERMAL_COMPRESSIBILITY", "T", temp_k, "Dmass", rho_mass, "Argon")
        * 1.0e9,
    }


def get_summary_value(summary_df, observable, column):
    row = summary_df.loc[summary_df["observable"] == observable, column]
    if row.empty:
        raise KeyError(f"Missing observable '{observable}' in {SUMMARY_CSV}")
    return float(row.iloc[0])


def build_real_reference_tables():
    summary_df = pd.read_csv(SUMMARY_CSV)
    scan_df = pd.read_csv(SCAN_CSV)

    scan_real = scan_df.copy()
    scan_real["rho_mass_kgm3"] = scan_real["rho_actual"].map(reduced_density_to_mass_density)
    scan_real["T_K"] = scan_real["T_mean"].map(reduced_temperature_to_kelvin)
    scan_real["P_MPa"] = scan_real["P_mean"].map(reduced_pressure_to_mpa)

    coolprop_scan_rows = []
    for rho_star in sorted(scan_real["rho_actual"].unique()):
        rho_mass = reduced_density_to_mass_density(rho_star)
        state = coolprop_state(TARGET_T_K, rho_mass)
        coolprop_scan_rows.append(
            {
                "rho_actual": rho_star,
                "rho_mass_kgm3": rho_mass,
                "T_K": TARGET_T_K,
                "P_MPa": state["P_MPa"],
                "Z": state["Z"],
            }
        )

    coolprop_scan_df = pd.DataFrame(coolprop_scan_rows)

    statepoint_rows = []
    kappa_bamps = reduced_kappa_to_gpa_inv(get_summary_value(summary_df, "kappa_T_from_EOS", "BAMPS"))
    kappa_lammps = reduced_kappa_to_gpa_inv(get_summary_value(summary_df, "kappa_T_from_EOS", "LAMMPS"))

    for code, kappa_val in [("BAMPS", kappa_bamps), ("LAMMPS", kappa_lammps)]:
        state_row = scan_real[(scan_real["code"] == code) & np.isclose(scan_real["rho_target"], TARGET_RHO_STAR)].iloc[0]
        cp_ref = coolprop_state(float(state_row["T_K"]), float(state_row["rho_mass_kgm3"]))
        statepoint_rows.append(
            {
                "code": code,
                "T_K": state_row["T_K"],
                "rho_star": state_row["rho_actual"],
                "rho_mass_kgm3": state_row["rho_mass_kgm3"],
                "Z_code": state_row["Z_mean"],
                "Z_coolprop": cp_ref["Z"],
                "Z_rel_error_pct": 100.0 * abs(state_row["Z_mean"] - cp_ref["Z"]) / abs(cp_ref["Z"]),
                "kappa_code_GPa_inv": kappa_val,
                "kappa_coolprop_GPa_inv": cp_ref["kappa_GPa_inv"],
                "kappa_rel_error_pct": 100.0 * abs(kappa_val - cp_ref["kappa_GPa_inv"]) / abs(cp_ref["kappa_GPa_inv"]),
            }
        )

    statepoint_df = pd.DataFrame(statepoint_rows)
    return scan_real, coolprop_scan_df, statepoint_df


def plot_real_reference(scan_real, coolprop_scan_df, statepoint_df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax_p = axes[0, 0]
    for code, color in [("BAMPS", "red"), ("LAMMPS", "black")]:
        sub = scan_real[scan_real["code"] == code].sort_values("rho_mass_kgm3")
        ax_p.plot(sub["rho_mass_kgm3"], sub["P_MPa"], "o-", color=color, label=code)
    ax_p.plot(coolprop_scan_df["rho_mass_kgm3"], coolprop_scan_df["P_MPa"], "s--", color="steelblue", label="CoolProp Argon")
    ax_p.set_title(f"Mapped Argon Pressure Scan at {TARGET_T_K:.1f} K")
    ax_p.set_xlabel(r"Mass density [kg/m$^3$]")
    ax_p.set_ylabel("Pressure [MPa]")
    ax_p.grid(True, alpha=0.3)
    ax_p.legend()

    ax_z = axes[0, 1]
    for code, color in [("BAMPS", "red"), ("LAMMPS", "black")]:
        sub = scan_real[scan_real["code"] == code].sort_values("rho_mass_kgm3")
        ax_z.plot(sub["rho_mass_kgm3"], sub["Z_mean"], "o-", color=color, label=code)
    ax_z.plot(coolprop_scan_df["rho_mass_kgm3"], coolprop_scan_df["Z"], "s--", color="steelblue", label="CoolProp Argon")
    ax_z.set_title(f"Mapped Compressibility Factor at {TARGET_T_K:.1f} K")
    ax_z.set_xlabel(r"Mass density [kg/m$^3$]")
    ax_z.set_ylabel("Z")
    ax_z.grid(True, alpha=0.3)
    ax_z.legend()

    ax_bar = axes[1, 0]
    metrics = ["Z_code", "kappa_code_GPa_inv"]
    ref_metrics = ["Z_coolprop", "kappa_coolprop_GPa_inv"]
    labels = ["Z", r"$\kappa_T$ [GPa$^{-1}$]"]
    x = np.arange(len(labels))
    width = 0.22

    for idx, (_, row) in enumerate(statepoint_df.iterrows()):
        ax_bar.bar(x + (idx - 0.5) * width, [row[m] for m in metrics], width=width, label=row["code"])
    ax_bar.bar(
        x + 1.5 * width,
        [statepoint_df.iloc[0][m] for m in ref_metrics],
        width=width,
        label="CoolProp",
        color="steelblue",
    )
    ax_bar.set_xticks(x + 0.5 * width)
    ax_bar.set_xticklabels(labels)
    ax_bar.set_title("Mapped Argon Z & kappa_T Comparison")
    ax_bar.grid(True, alpha=0.3, axis="y")
    ax_bar.legend()

    ax_text = axes[1, 1]
    ax_text.axis("off")
    lines = [
        "Real-Argon mapping",
        "sigma = 3.405 A",
        "epsilon/k_B = 119.8 K",
        "",
        "CoolProp Argon reference:",
        "Tegeler, Span, Wagner (1999)",
        "",
        "Statepoint at reduced density rho* ~= 0.8",
        "",
        "code    Z err%   kappa err%",
    ]
    for _, row in statepoint_df.iterrows():
        lines.append(
            f"{row['code']:<6} {row['Z_rel_error_pct']:>8.2f} {row['kappa_rel_error_pct']:>12.2f}"
        )

    ax_text.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=11)
    ax_text.set_title("Reference Metadata")

    plt.tight_layout()
    plt.savefig(RESULT_DIR / "argon_real_reference.png")


if __name__ == "__main__":
    scan_real_df, coolprop_scan, statepoint_df = build_real_reference_tables()
    scan_real_df.to_csv(RESULT_DIR / "argon_real_reference_scan.csv", index=False)
    statepoint_df.to_csv(RESULT_DIR / "argon_real_reference_summary.csv", index=False)
    plot_real_reference(scan_real_df, coolprop_scan, statepoint_df)
