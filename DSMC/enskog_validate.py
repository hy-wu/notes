import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BIN_FILE = "enskog_validate.exe"
SRC_FILE = "enskog_validate.cu"
RESULT_DIR = Path("enskog_results")

BOX = 20.0
IX = 5
DT = 0.005
STEPS = 1600
BASE_RHO = 0.05
BASE_T = 1.0
BASE_D = 1.0
BASE_ANISOTROPY = 0.5

DENSITY_VALUES = np.round(np.linspace(0.02, 0.09, 8), 3)
TEMPERATURE_VALUES = [0.4, 0.7, 1.0, 1.5, 2.2]
DIAMETER_VALUES = [0.7, 0.85, 1.0, 1.15, 1.3]
ORDER_VALUES = [0, 1, 2]


def compile_code():
    subprocess.run(["nvcc", "-O3", "-arch=sm_89", "-std=c++17", SRC_FILE, "-o", BIN_FILE], check=True)


def run_case(name, order=0, rho=BASE_RHO, temp=BASE_T, diameter=BASE_D, steps=STEPS, anisotropy=BASE_ANISOTROPY):
    RESULT_DIR.mkdir(exist_ok=True)
    prefix = RESULT_DIR / name
    cmd = [
        BIN_FILE,
        str(prefix),
        str(order),
        str(rho),
        str(temp),
        str(diameter),
        str(steps),
        str(BOX),
        str(IX),
        str(DT),
        str(anisotropy),
    ]
    subprocess.run(cmd, check=True)
    summary = pd.read_csv(f"{prefix}_summary.csv")
    ts = pd.read_csv(f"{prefix}_timeseries.csv")
    final_energy = pd.read_csv(f"{prefix}_final_energy.csv")
    row = summary.iloc[0].to_dict()
    row.update({"case": name, "rho_input": rho, "temp_input": temp, "diameter_input": diameter, "steps": steps})
    return row, ts, final_energy


def relative_change(a, b):
    denom = max(abs(a), 1e-12)
    return abs(b - a) / denom


def build_suite():
    cases = []
    timeseries = {}
    energy_samples = {}

    for rho in DENSITY_VALUES:
        for order in ORDER_VALUES:
            row, ts, final_energy = run_case(f"rho_order{order}_{rho:.3f}", order=order, rho=float(rho))
            cases.append(row)
            timeseries[row["case"]] = ts
            energy_samples[row["case"]] = final_energy["Energy"].to_numpy()

    for temp in TEMPERATURE_VALUES:
        row, ts, final_energy = run_case(f"temp_order0_{temp:.2f}", order=0, temp=temp)
        cases.append(row)
        timeseries[row["case"]] = ts
        energy_samples[row["case"]] = final_energy["Energy"].to_numpy()

    for diameter in DIAMETER_VALUES:
        for order in ORDER_VALUES:
            row, ts, final_energy = run_case(f"diam_order{order}_{diameter:.2f}", order=order, diameter=diameter)
            cases.append(row)
            timeseries[row["case"]] = ts
            energy_samples[row["case"]] = final_energy["Energy"].to_numpy()

    for order in ORDER_VALUES:
        row, ts, final_energy = run_case(f"order_{order}", order=order)
        cases.append(row)
        timeseries[row["case"]] = ts
        energy_samples[row["case"]] = final_energy["Energy"].to_numpy()

    df = pd.DataFrame(cases)
    df["rate_enhancement_measured"] = df["MeasuredRate"] / df["TheoryRate0"]
    df["rate_enhancement_trunc"] = df["TheoryRateTrunc"] / df["TheoryRate0"]
    df["rate_enhancement_inf_cs"] = df["TheoryRateInfCS"] / df["TheoryRate0"]
    df["rate_enhancement_inf_ev"] = df["TheoryRateInfEV"] / df["TheoryRate0"]
    df["rate_ratio_measured_trunc"] = df["MeasuredRate"] / df["TheoryRateTrunc"]
    df["rate_ratio_measured_inf_cs"] = df["MeasuredRate"] / df["TheoryRateInfCS"]
    df["rate_ratio_measured_inf_ev"] = df["MeasuredRate"] / df["TheoryRateInfEV"]
    df["Z_measured"] = 1.0 + 4.0 * df["Eta"] * df["rate_enhancement_measured"]
    df["Z_trunc"] = 1.0 + 4.0 * df["Eta"] * df["ChiTrunc"]
    df["Z_inf_cs"] = 1.0 + 4.0 * df["Eta"] * df["ChiResummedCS"]
    df["Z_inf_ev"] = 1.0 + 4.0 * df["Eta"] * df["ChiResummedEV"]
    df["momentum_rel_change"] = (
        abs(df["PxFinal"] - df["PxInitial"])
        + abs(df["PyFinal"] - df["PyInitial"])
        + abs(df["PzFinal"] - df["PzInitial"])
    ) / (
        abs(df["PxInitial"]) + abs(df["PyInitial"]) + abs(df["PzInitial"]) + 1e-12
    )
    df["ke_rel_change"] = abs(df["KEFinal"] - df["KEInitial"]) / abs(df["KEInitial"])
    df.to_csv(RESULT_DIR / "enskog_validation_summary.csv", index=False)
    return df, timeseries, energy_samples


def add_scaling_metrics(df):
    metrics = []

    density_frames = []
    for order in ORDER_VALUES:
        sub = df[df["case"].str.startswith(f"rho_order{order}_")].sort_values("Eta").copy()
        if len(sub) >= 2:
            rmse = float(np.sqrt(np.mean((sub["rate_enhancement_measured"] - sub["rate_enhancement_trunc"]) ** 2)))
            metrics.append({"metric": f"order{order}_density_enhancement_rmse", "value": rmse})
            metrics.append(
                {
                    "metric": f"order{order}_eos_rmse",
                    "value": float(np.sqrt(np.mean((sub["Z_measured"] - sub["Z_trunc"]) ** 2))),
                }
            )
            density_frames.append(
                sub[[
                    "case", "Order", "rho_input", "Rho", "Eta", "ChiTrunc", "ChiResummedCS", "ChiResummedEV",
                    "rate_enhancement_measured", "rate_enhancement_trunc", "rate_enhancement_inf_cs", "rate_enhancement_inf_ev",
                    "Z_measured", "Z_trunc", "Z_inf_cs", "Z_inf_ev",
                    "rate_ratio_measured_trunc", "rate_ratio_measured_inf_cs", "rate_ratio_measured_inf_ev"
                ]]
            )
    if density_frames:
        eos_df = pd.concat(density_frames, ignore_index=True)
        eos_df.to_csv(RESULT_DIR / "enskog_chi_ratio.csv", index=False)
        eos_df.to_csv(RESULT_DIR / "enskog_eos_validation.csv", index=False)

    temp = df[df["case"].str.startswith("temp_order0_")].sort_values("temp_input")
    if len(temp) >= 2:
        y = temp["MeasuredRate"] / temp["MeasuredRate"].iloc[2]
        x = np.sqrt(temp["temp_input"] / temp["temp_input"].iloc[2])
        metrics.append({"metric": "sqrt_temperature_scaling_rmse", "value": float(np.sqrt(np.mean((y - x) ** 2)))})

    diam = df[df["case"].str.startswith("diam_order0_")].sort_values("diameter_input")
    if len(diam) >= 2:
        y = diam["MeasuredRate"] / diam["MeasuredRate"].iloc[2]
        x = (diam["diameter_input"] / diam["diameter_input"].iloc[2]) ** 2
        metrics.append({"metric": "diameter_squared_scaling_rmse", "value": float(np.sqrt(np.mean((y - x) ** 2)))})

    metrics.append({"metric": "order2_vs_cs_mean_ratio", "value": float(df[df["case"].str.startswith("rho_order2_")]["rate_ratio_measured_inf_cs"].mean())})
    metrics.append({"metric": "order2_vs_ev_mean_ratio", "value": float(df[df["case"].str.startswith("rho_order2_")]["rate_ratio_measured_inf_ev"].mean())})
    order2 = df[df["case"].str.startswith("rho_order2_")]
    metrics.append({"metric": "order2_Z_vs_cs_rmse", "value": float(np.sqrt(np.mean((order2["Z_measured"] - order2["Z_inf_cs"]) ** 2)))})
    metrics.append({"metric": "order2_Z_vs_ev_rmse", "value": float(np.sqrt(np.mean((order2["Z_measured"] - order2["Z_inf_ev"]) ** 2)))})
    metrics.append({"metric": "max_ke_rel_change", "value": float(df["ke_rel_change"].max())})
    metrics.append({"metric": "max_momentum_rel_change", "value": float(df["momentum_rel_change"].max())})

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(RESULT_DIR / "enskog_validation_metrics.csv", index=False)
    return metrics_df


def plot_density_panel(ax, df):
    styles = {0: ("o", "C0"), 1: ("s", "C1"), 2: ("^", "C2")}
    for order in ORDER_VALUES:
        sub = df[df["case"].str.startswith(f"rho_order{order}_")].sort_values("Eta")
        marker, color = styles[order]
        ax.plot(sub["Eta"], sub["rate_enhancement_measured"], marker + "-", color=color, label=f"Measured O({order})")
        ax.plot(sub["Eta"], sub["rate_enhancement_trunc"], "--", color=color, alpha=0.85, label=f"Theory O({order})")

    ref = df[df["case"].str.startswith("rho_order2_")].sort_values("Eta")
    ax.plot(ref["Eta"], ref["rate_enhancement_inf_cs"], "k-", linewidth=1.8, label=r"Resummed $\chi_\infty$ (Carnahan-Starling)")
    ax.plot(ref["Eta"], ref["rate_enhancement_inf_ev"], "k:", linewidth=1.8, label=r"Excluded-volume $\chi_\infty$")
    ax.set_title("Collision-rate enhancement vs packing fraction")
    ax.set_xlabel(r"$\eta = (\pi/6)\rho d^3$")
    ax.set_ylabel(r"$\nu / \nu_0$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)


def plot_eos_panel(ax, df):
    styles = {0: ("o", "C0"), 1: ("s", "C1"), 2: ("^", "C2")}
    for order in ORDER_VALUES:
        sub = df[df["case"].str.startswith(f"rho_order{order}_")].sort_values("Eta")
        marker, color = styles[order]
        ax.plot(sub["Eta"], sub["Z_measured"], marker + "-", color=color, label=f"Measured Z O({order})")
        ax.plot(sub["Eta"], sub["Z_trunc"], "--", color=color, alpha=0.85, label=f"Theory Z O({order})")

    ref = df[df["case"].str.startswith("rho_order2_")].sort_values("Eta")
    ax.plot(ref["Eta"], ref["Z_inf_cs"], "k-", linewidth=1.8, label=r"CS EOS $Z_\infty$")
    ax.plot(ref["Eta"], ref["Z_inf_ev"], "k:", linewidth=1.8, label=r"Excluded-volume EOS $Z_\infty$")
    ax.set_title("Equation-of-state validation")
    ax.set_xlabel(r"$\eta = (\pi/6)\rho d^3$")
    ax.set_ylabel(r"$Z = 1 + 4\eta\chi$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)


def plot_eos_error_panel(ax, df):
    order2 = df[df["case"].str.startswith("rho_order2_")].sort_values("Eta")
    cs_rel = (order2["Z_measured"] - order2["Z_inf_cs"]) / order2["Z_inf_cs"]
    ev_rel = (order2["Z_measured"] - order2["Z_inf_ev"]) / order2["Z_inf_ev"]
    ax.plot(order2["Eta"], cs_rel, "o-", label="Order 2 vs CS EOS")
    ax.plot(order2["Eta"], ev_rel, "s-", label="Order 2 vs excluded-volume EOS")
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1.0)
    ax.set_title("EOS relative deviation")
    ax.set_xlabel(r"$\eta = (\pi/6)\rho d^3$")
    ax.set_ylabel(r"$(Z_{\mathrm{meas}} - Z_{\mathrm{ref}})/Z_{\mathrm{ref}}$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)


def plot_diameter_eos_panel(ax, df):
    styles = {0: ("o", "C0"), 1: ("s", "C1"), 2: ("^", "C2")}
    for order in ORDER_VALUES:
        sub = df[df["case"].str.startswith(f"diam_order{order}_")].sort_values("diameter_input")
        marker, color = styles[order]
        ax.plot(sub["diameter_input"], sub["Z_measured"], marker + "-", color=color, label=f"Measured Z O({order})")
        ax.plot(sub["diameter_input"], sub["Z_trunc"], "--", color=color, alpha=0.85, label=f"Theory Z O({order})")
    ref = df[df["case"].str.startswith("diam_order2_")].sort_values("diameter_input")
    ax.plot(ref["diameter_input"], ref["Z_inf_cs"], "k-", linewidth=1.8, label=r"CS EOS $Z_\infty$")
    ax.plot(ref["diameter_input"], ref["Z_inf_ev"], "k:", linewidth=1.8, label=r"Excluded-volume EOS $Z_\infty$")
    ax.set_title("Diameter-sweep EOS validation")
    ax.set_xlabel("diameter")
    ax.set_ylabel("Z")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)


def boltzmann_energy_pdf(energy, temperature):
    temperature = max(float(temperature), 1.0e-8)
    prefactor = 2.0 / np.sqrt(np.pi)
    return prefactor * np.sqrt(np.maximum(energy, 0.0)) * np.exp(-energy / temperature) / (temperature ** 1.5)


def plot_energy_distribution_panel(ax, df, energy_samples):
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(DIAMETER_VALUES)))
    series = []
    for diameter, color in zip(DIAMETER_VALUES, colors):
        case = f"diam_order2_{diameter:.2f}"
        if case not in energy_samples:
            continue
        energies = np.asarray(energy_samples[case], dtype=float)
        if energies.size == 0:
            continue
        row = df[df["case"] == case].iloc[0]
        t_fit = 2.0 * energies.mean() / 3.0
        series.append((diameter, color, energies, t_fit, row["T_final"]))

    if not series:
        return

    max_energy = max(np.max(item[2]) for item in series)
    bins = np.linspace(0.0, max_energy * 1.05, 61)
    grid = np.linspace(0.0, max_energy * 1.05, 400)

    for diameter, color, energies, t_fit, _ in series:
        hist, edges = np.histogram(energies, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.step(centers, hist, where="mid", color=color, alpha=0.35)
        ax.plot(grid, boltzmann_energy_pdf(grid, t_fit), color=color, linewidth=1.8, label=f"d={diameter:.2f}, Tfit={t_fit:.2f}")

    ax.set_title("Final energy distribution with Boltzmann fit (order 2)")
    ax.set_xlabel("particle kinetic energy")
    ax.set_ylabel("probability density")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def plot_ke_trace_panel(ax, df, timeseries):
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(DIAMETER_VALUES)))
    for diameter, color in zip(DIAMETER_VALUES, colors):
        case = f"diam_order2_{diameter:.2f}"
        if case not in timeseries:
            continue
        ts = timeseries[case]
        row = df[df["case"] == case].iloc[0]
        ax.plot(ts["Time"], ts["KE"] / row["N"], color=color, label=f"d={diameter:.2f}")
    ax.set_title("Kinetic energy evolution (order 2)")
    ax.set_xlabel("time")
    ax.set_ylabel("KE/N")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def format_parameter_header():
    base_n = int(round(BASE_RHO * BOX ** 3))
    density_n_min = int(round(float(DENSITY_VALUES[0]) * BOX ** 3))
    density_n_max = int(round(float(DENSITY_VALUES[-1]) * BOX ** 3))
    density_text = f"rho sweep={DENSITY_VALUES[0]:.3f}..{DENSITY_VALUES[-1]:.3f} ({len(DENSITY_VALUES)} pts)"
    temp_text = "T sweep=" + ", ".join(f"{value:.2f}" for value in TEMPERATURE_VALUES)
    diameter_text = "d sweep=" + ", ".join(f"{value:.2f}" for value in DIAMETER_VALUES)
    order_text = "orders=" + ", ".join(str(order) for order in ORDER_VALUES)
    line1 = (
        "Parameters: "
        f"box={BOX:.1f}, IX={IX}, dt={DT:.4f}, steps={STEPS}, "
        f"Nbase={base_n}, N(rho sweep)={density_n_min}..{density_n_max}, "
        f"base(rho,T,d)=({BASE_RHO:.3f}, {BASE_T:.2f}, {BASE_D:.2f}), anisotropy={BASE_ANISOTROPY:.2f}"
    )
    line2 = f"{density_text}; {temp_text}; {diameter_text}; {order_text}"
    return line1 + "\n" + line2


def plot_results(df, timeseries, metrics_df, energy_samples):
    fig, axes = plt.subplots(3, 3, figsize=(18.5, 16.0))
    fig.suptitle("Enskog collision and EOS validation", fontsize=16, y=0.985)
    fig.text(0.5, 0.945, format_parameter_header(), ha="center", va="top", fontsize=10)

    plot_density_panel(axes[0, 0], df)
    plot_eos_panel(axes[0, 1], df)
    plot_eos_error_panel(axes[0, 2], df)

    ax = axes[1, 0]
    temp = df[df["case"].str.startswith("temp_order0_")].sort_values("temp_input")
    ax.plot(temp["temp_input"], temp["MeasuredRate"] / temp["MeasuredRate"].iloc[2], "o-", label="Measured")
    ax.plot(temp["temp_input"], np.sqrt(temp["temp_input"] / temp["temp_input"].iloc[2]), "k--", label=r"$\sqrt{T}$")
    ax.set_title("Hard-sphere rate temperature scaling")
    ax.set_xlabel("T")
    ax.set_ylabel("normalized rate")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    diam = df[df["case"].str.startswith("diam_order0_")].sort_values("diameter_input")
    ax.plot(diam["diameter_input"], diam["MeasuredRate"] / diam["MeasuredRate"].iloc[2], "o-", label="Measured")
    ax.plot(diam["diameter_input"], (diam["diameter_input"] / diam["diameter_input"].iloc[2]) ** 2, "k--", label=r"$d^2$")
    ax.set_title("Hard-sphere cross-section scaling")
    ax.set_xlabel("diameter")
    ax.set_ylabel("normalized rate")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 2]
    for case, label in [("order_0", "order 0"), ("order_1", "order 1"), ("order_2", "order 2")]:
        if case in timeseries:
            ts = timeseries[case]
            ax.plot(ts["Step"], np.abs(ts["Anisotropy"]), label=label)
    ax.set_yscale("log")
    ax.set_title("Anisotropy relaxation")
    ax.set_xlabel("step")
    ax.set_ylabel(r"$|A|$")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[2, 0]
    plot_diameter_eos_panel(ax, df)

    ax = axes[2, 1]
    plot_energy_distribution_panel(ax, df, energy_samples)

    ax = axes[2, 2]
    plot_ke_trace_panel(ax, df, timeseries)

    fig.tight_layout(rect=(0.02, 0.05, 0.98, 0.92))
    fig.savefig(RESULT_DIR / "enskog_validation.png")

    fig2, ax2 = plt.subplots(figsize=(11.5, 4.8))
    fig2.suptitle("Enskog validation metrics", fontsize=14, y=0.97)
    fig2.text(0.5, 0.92, format_parameter_header(), ha="center", va="top", fontsize=9)
    ax2.axis("off")
    lines = ["Validation metrics", "------------------"]
    for _, row in metrics_df.iterrows():
        lines.append(f"{row['metric']:<38} {row['value']:.6g}")
    ax2.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace")
    fig2.tight_layout(rect=(0.02, 0.04, 0.98, 0.83))
    fig2.savefig(RESULT_DIR / "enskog_validation_metrics.png")


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    compile_code()
    df, timeseries, energy_samples = build_suite()
    metrics_df = add_scaling_metrics(df)
    plot_results(df, timeseries, metrics_df, energy_samples)
    print(df[[
        "case", "Order", "Eta", "MeasuredRate", "TheoryRateTrunc", "TheoryRateInfCS",
        "Z_measured", "Z_trunc", "Z_inf_cs", "rate_ratio_measured_trunc",
        "rate_ratio_measured_inf_cs", "ke_rel_change"
    ]].to_string(index=False))
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
