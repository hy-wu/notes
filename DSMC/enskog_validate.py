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
BASE_RHO = 0.16
BASE_T = 1.0
BASE_D = 0.8
BASE_ANISOTROPY = 0.5
TARGET_SWEEP_N = 131072
TARGET_CELL_SIZE = 4.0

DENSITY_VALUES = np.round(np.linspace(0.08, 0.20, 7), 3)
TEMPERATURE_VALUES = [0.4, 0.7, 1.0, 1.5, 2.2]
DIAMETER_VALUES = [0.55, 0.675, 0.8, 0.925, 1.05]
ORDER_VALUES = [0, 1, 2]
SIZE_SWEEP = [
    (20.0, 5),
    (40.0, 10),
    (60.0, 15),
    (80.0, 20),
    (((TARGET_SWEEP_N / BASE_RHO) ** (1.0 / 3.0)), 23),
]


def compile_code():
    subprocess.run(["nvcc", "-O3", "-arch=sm_89", "-std=c++17", SRC_FILE, "-o", BIN_FILE], check=True)


def sweep_geometry_for_density(rho, target_n=TARGET_SWEEP_N):
    box = (target_n / float(rho)) ** (1.0 / 3.0)
    ix = max(1, int(round(box / TARGET_CELL_SIZE)))
    return float(box), ix


BASE_SWEEP_BOX, BASE_SWEEP_IX = sweep_geometry_for_density(BASE_RHO)


def run_case(
    name,
    order=0,
    rho=BASE_RHO,
    temp=BASE_T,
    diameter=BASE_D,
    steps=STEPS,
    anisotropy=BASE_ANISOTROPY,
    box=BOX,
    ix=IX,
):
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
        str(box),
        str(ix),
        str(DT),
        str(anisotropy),
    ]
    subprocess.run(cmd, check=True)
    summary = pd.read_csv(f"{prefix}_summary.csv")
    ts = pd.read_csv(f"{prefix}_timeseries.csv")
    final_energy = pd.read_csv(f"{prefix}_final_energy.csv")
    row = summary.iloc[0].to_dict()
    row.update(
        {
            "case": name,
            "rho_input": rho,
            "temp_input": temp,
            "diameter_input": diameter,
            "steps": steps,
            "box_input": box,
            "ix_input": ix,
        }
    )
    return row, ts, final_energy


def relative_change(a, b):
    denom = max(abs(a), 1e-12)
    return abs(b - a) / denom


def build_suite():
    cases = []
    timeseries = {}
    energy_samples = {}

    for rho in DENSITY_VALUES:
        sweep_box, sweep_ix = sweep_geometry_for_density(rho)
        for order in ORDER_VALUES:
            row, ts, final_energy = run_case(
                f"rho_order{order}_{rho:.3f}",
                order=order,
                rho=float(rho),
                box=sweep_box,
                ix=sweep_ix,
            )
            cases.append(row)
            timeseries[row["case"]] = ts
            energy_samples[row["case"]] = final_energy["Energy"].to_numpy()

    for temp in TEMPERATURE_VALUES:
        row, ts, final_energy = run_case(
            f"temp_order0_{temp:.2f}",
            order=0,
            temp=temp,
            box=BASE_SWEEP_BOX,
            ix=BASE_SWEEP_IX,
        )
        cases.append(row)
        timeseries[row["case"]] = ts
        energy_samples[row["case"]] = final_energy["Energy"].to_numpy()

    for diameter in DIAMETER_VALUES:
        for order in ORDER_VALUES:
            row, ts, final_energy = run_case(
                f"diam_order{order}_{diameter:.2f}",
                order=order,
                diameter=diameter,
                box=BASE_SWEEP_BOX,
                ix=BASE_SWEEP_IX,
            )
            cases.append(row)
            timeseries[row["case"]] = ts
            energy_samples[row["case"]] = final_energy["Energy"].to_numpy()

    for order in ORDER_VALUES:
        row, ts, final_energy = run_case(
            f"order_{order}",
            order=order,
            box=BASE_SWEEP_BOX,
            ix=BASE_SWEEP_IX,
        )
        cases.append(row)
        timeseries[row["case"]] = ts
        energy_samples[row["case"]] = final_energy["Energy"].to_numpy()

    for box, ix in SIZE_SWEEP:
        row, ts, final_energy = run_case(
            f"size_order2_box{box:.0f}_ix{ix}",
            order=2,
            box=box,
            ix=ix,
        )
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
    size = df[df["case"].str.startswith("size_order2_")].sort_values("N")
    if len(size) >= 2:
        size_summary = size[
            [
                "case",
                "N",
                "Box",
                "IX",
                "CellSize",
                "MeanStepMs",
                "ParticleStepsPerSec",
                "Z_measured",
                "Z_inf_cs",
                "rate_ratio_measured_inf_cs",
                "ke_rel_change",
            ]
        ].copy()
        size_summary["Z_rel_err_cs"] = (size_summary["Z_measured"] - size["Z_inf_cs"].to_numpy()) / size["Z_inf_cs"].to_numpy()
        size_summary.to_csv(RESULT_DIR / "enskog_size_validation.csv", index=False)
        metrics.append({"metric": "size_order2_Z_span", "value": float(size["Z_measured"].max() - size["Z_measured"].min())})
        metrics.append(
            {
                "metric": "size_order2_Z_vs_cs_rmse",
                "value": float(np.sqrt(np.mean((size["Z_measured"] - size["Z_inf_cs"]) ** 2))),
            }
        )
        metrics.append(
            {
                "metric": "size_order2_rate_ratio_span",
                "value": float(size["rate_ratio_measured_inf_cs"].max() - size["rate_ratio_measured_inf_cs"].min()),
            }
        )
        metrics.append({"metric": "size_order2_mean_step_ms_min", "value": float(size["MeanStepMs"].min())})
        metrics.append({"metric": "size_order2_mean_step_ms_max", "value": float(size["MeanStepMs"].max())})
        metrics.append({"metric": "size_order2_particles_max", "value": float(size["N"].max())})
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


def plot_size_validation(df):
    size = df[df["case"].str.startswith("size_order2_")].sort_values("N")
    if size.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.suptitle("Enskog size-sweep validation", fontsize=15, y=0.98)
    fig.text(0.5, 0.91, format_parameter_header(), ha="center", va="top", fontsize=8.5)

    ax = axes[0]
    ax.plot(size["N"], size["Z_measured"], "o-", label="Measured Z")
    ax.plot(size["N"], size["Z_inf_cs"], "k--", label=r"CS EOS $Z_\infty$")
    ax.plot(size["N"], size["Z_inf_ev"], "k:", label=r"Excluded-volume EOS $Z_\infty$")
    ax.set_title("EOS vs system size")
    ax.set_xlabel("N")
    ax.set_ylabel("Z")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(size["N"], size["rate_ratio_measured_inf_cs"], "o-", label="Rate / CS reference")
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1.0)
    ax.set_title("Collision-rate convergence")
    ax.set_xlabel("N")
    ax.set_ylabel(r"$\nu / \nu_{\mathrm{CS}}$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(size["N"], size["MeanStepMs"], "o-", label="Mean step time")
    ax2 = ax.twinx()
    ax2.plot(size["N"], size["ParticleStepsPerSec"] / 1.0e6, "s--", color="C1", label="Particle steps/s")
    ax.set_title("Runtime vs system size")
    ax.set_xlabel("N")
    ax.set_ylabel("ms/step")
    ax2.set_ylabel(r"particle-steps/s ($10^6$)")
    ax.grid(True, alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.83))
    fig.savefig(RESULT_DIR / "enskog_size_validation.png")


def plot_runtime_efficiency(df):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    fig.suptitle("Enskog runtime efficiency", fontsize=15, y=0.98)
    fig.text(0.5, 0.915, format_parameter_header(), ha="center", va="top", fontsize=8.5)

    ax = axes[0, 0]
    for order, marker in [(0, "o"), (1, "s"), (2, "^")]:
        sub = df[df["case"].str.startswith(f"rho_order{order}_")].sort_values("rho_input")
        ax.plot(sub["rho_input"], sub["MeanStepMs"], marker + "-", label=f"order {order}")
    ax.set_title("Mean step time vs density")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel("ms/step")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    temp = df[df["case"].str.startswith("temp_order0_")].sort_values("temp_input")
    ax.plot(temp["temp_input"], temp["MeanStepMs"], "o-")
    ax.set_title("Mean step time vs temperature")
    ax.set_xlabel("T")
    ax.set_ylabel("ms/step")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    for order, marker in [(0, "o"), (1, "s"), (2, "^")]:
        sub = df[df["case"].str.startswith(f"diam_order{order}_")].sort_values("diameter_input")
        ax.plot(sub["diameter_input"], sub["MeanStepMs"], marker + "-", label=f"order {order}")
    ax.set_title("Mean step time vs diameter")
    ax.set_xlabel("diameter")
    ax.set_ylabel("ms/step")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    order = df[df["case"].str.startswith("order_")].sort_values("Order")
    ax.plot(order["Order"], order["MeanStepMs"], "o-")
    ax.set_title("Mean step time vs Enskog order")
    ax.set_xlabel("order")
    ax.set_ylabel("ms/step")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    size = df[df["case"].str.startswith("size_order2_")].sort_values("N")
    ax.plot(size["N"], size["MeanStepMs"], "o-", label="Mean step time")
    ax.set_title("Mean step time vs system size")
    ax.set_xlabel("N")
    ax.set_ylabel("ms/step")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    ax.plot(size["N"], size["ParticleStepsPerSec"] / 1.0e6, "s-", color="C1")
    ax.set_title("Throughput vs system size")
    ax.set_xlabel("N")
    ax.set_ylabel(r"particle-steps/s ($10^6$)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.84))
    fig.savefig(RESULT_DIR / "enskog_runtime_efficiency.png")


def format_parameter_header():
    base_n = int(round(BASE_RHO * BASE_SWEEP_BOX ** 3))
    density_n_values = [int(round(float(rho) * sweep_geometry_for_density(rho)[0] ** 3)) for rho in DENSITY_VALUES]
    size_n_values = [int(round(BASE_RHO * box ** 3)) for box, _ in SIZE_SWEEP]
    size_box_ix_text = ", ".join(f"{box:.1f}/{ix}" for box, ix in SIZE_SWEEP)
    density_text = f"rho sweep={DENSITY_VALUES[0]:.3f}..{DENSITY_VALUES[-1]:.3f} ({len(DENSITY_VALUES)} pts)"
    temp_text = "T sweep=" + ", ".join(f"{value:.2f}" for value in TEMPERATURE_VALUES)
    diameter_text = "d sweep=" + ", ".join(f"{value:.3f}" for value in DIAMETER_VALUES)
    order_text = "orders=" + ", ".join(str(order) for order in ORDER_VALUES)
    size_text = "size sweep N=" + ", ".join(str(value) for value in size_n_values)
    line1 = (
        "Parameters: "
        f"base box/IX={BASE_SWEEP_BOX:.2f}/{BASE_SWEEP_IX}, dt={DT:.4f}, steps={STEPS}, "
        f"Nbase={base_n}, target N={TARGET_SWEEP_N}, anisotropy={BASE_ANISOTROPY:.2f}"
    )
    line2 = (
        f"density sweep: {density_text}; N={min(density_n_values)}..{max(density_n_values)}; "
        f"base(rho,T,d)=({BASE_RHO:.3f}, {BASE_T:.2f}, {BASE_D:.3f}); {temp_text}; {diameter_text}"
    )
    line3 = f"size sweep (same rho/T/d): {size_text}; size(box/IX)={size_box_ix_text}; {order_text}"
    return line1 + "\n" + line2 + "\n" + line3


def plot_results(df, timeseries, metrics_df, energy_samples):
    fig, axes = plt.subplots(3, 3, figsize=(18.5, 16.0))
    fig.suptitle("Enskog collision and EOS validation", fontsize=16, y=0.985)
    fig.text(0.5, 0.93, format_parameter_header(), ha="center", va="top", fontsize=8.8)

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

    fig.tight_layout(rect=(0.02, 0.05, 0.98, 0.84))
    fig.savefig(RESULT_DIR / "enskog_validation.png")

    fig2, ax2 = plt.subplots(figsize=(11.5, 4.8))
    fig2.suptitle("Enskog validation metrics", fontsize=14, y=0.97)
    fig2.text(0.5, 0.92, format_parameter_header(), ha="center", va="top", fontsize=9)
    ax2.axis("off")
    lines = ["Validation metrics", "------------------"]
    for _, row in metrics_df.iterrows():
        lines.append(f"{row['metric']:<38} {row['value']:.6g}")
    ax2.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace")
    fig2.tight_layout(rect=(0.02, 0.04, 0.98, 0.78))
    fig2.savefig(RESULT_DIR / "enskog_validation_metrics.png")


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    compile_code()
    df, timeseries, energy_samples = build_suite()
    metrics_df = add_scaling_metrics(df)
    plot_results(df, timeseries, metrics_df, energy_samples)
    plot_size_validation(df)
    plot_runtime_efficiency(df)
    print(df[[
        "case", "Order", "N", "MeanStepMs", "Eta", "MeasuredRate", "TheoryRateTrunc", "TheoryRateInfCS",
        "Z_measured", "Z_trunc", "Z_inf_cs", "rate_ratio_measured_trunc",
        "rate_ratio_measured_inf_cs", "ke_rel_change"
    ]].to_string(index=False))
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
