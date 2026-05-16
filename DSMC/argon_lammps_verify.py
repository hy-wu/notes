import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BIN_FILE = "bamps_compare.exe"
SRC_FILE = "src/thermostat_compare.cu"
RESULT_DIR = Path("argon_results")
SCAN_DIR = RESULT_DIR / "compressibility_scan"

NVE_PREFIX = RESULT_DIR / "nve"
NVT_PREFIX = RESULT_DIR / "nvt"
LAMMPS_PREFIX = RESULT_DIR / "lammps"

EQUIL_STEPS = 5000
PROD_STEPS = 5000
BOX_SIZE = 12.0
SIGMA = 1.0
EPSILON = 1.0
RHO = 0.8
T_TARGET = 1.0
RDF_BINS = 150
STRUCTURE_FACTOR_POINTS = 240
STRUCTURE_FACTOR_Q_MAX = 18.0
LOW_Q_FIT_POINTS = 8
SPEED_BINS = 60
DENSITY_SCAN = np.array([0.72, 0.76, 0.80, 0.84, 0.88], dtype=np.float64)


def volume():
    return BOX_SIZE**3


def actual_density(target_rho):
    n_particles = int(target_rho * volume())
    return n_particles, n_particles / volume()


def windows_path(path):
    return str(path).replace("/", "\\")


def compile_code():
    print("Compiling rigorous harness...")
    subprocess.run(["nvcc", "-O3", "-arch=sm_89", "-std=c++17", SRC_FILE, "-o", BIN_FILE], check=True)


def run_bamps_state(prefix, rho, thermostat_type=3):
    subprocess.run(
        [
            BIN_FILE,
            str(thermostat_type),
            "1",
            str(prefix),
            str(SIGMA),
            str(EPSILON),
            str(rho),
            str(T_TARGET),
            str(PROD_STEPS),
            str(BOX_SIZE),
        ],
        check=True,
    )


def build_lammps_input(prefix, rho, dump_final=False, include_rdf=False):
    n_particles, rho_actual = actual_density(rho)
    prefix = Path(prefix)
    csv_path = Path(f"{prefix}_timeseries.csv")
    dump_path = Path(f"{prefix}_final_state.dump")

    lines = [
        "units lj",
        "atom_style atomic",
        "dimension 3",
        "boundary p p p",
        "",
        f"region sim_box block 0.0 {BOX_SIZE} 0.0 {BOX_SIZE} 0.0 {BOX_SIZE}",
        "create_box 1 sim_box",
        f"create_atoms 1 random {n_particles} 12345 sim_box",
        "",
        "mass 1 1.0",
        "pair_style lj/cut 2.5",
        f"pair_coeff 1 1 {EPSILON} {SIGMA} 2.5",
        "pair_modify tail yes",
        "neighbor 0.3 bin",
        "neigh_modify delay 0 every 1 check yes",
        "",
        "minimize 1.0e-4 1.0e-6 100 1000",
        "reset_timestep 0",
        f"velocity all create {T_TARGET} 87287 dist gaussian mom yes rot yes",
        f"fix nvt_run all nvt temp {T_TARGET} {T_TARGET} 0.1",
        f"run {EQUIL_STEPS}",
        "",
        "reset_timestep 0",
    ]

    if include_rdf:
        lines.extend(
            [
                f"compute myRDF all rdf {RDF_BINS}",
                f"fix rdf_out all ave/time 10 {PROD_STEPS // 10} {PROD_STEPS} c_myRDF[*] file {windows_path(RESULT_DIR / 'lammps_rdf.txt')} mode vector",
            ]
        )

    lines.extend(
        [
        "variable v_step equal step",
        "variable v_temp equal temp",
        "variable v_press equal press",
        "variable v_pe equal pe",
        "variable v_etotal equal etotal",
        (
            f'fix csv_out all print 10 "${{v_step}},${{v_temp}},${{v_press}},${{v_pe}},${{v_etotal}}" '
            f'file {windows_path(csv_path)} screen no title "Step,T,P,PE,TotalE"'
        ),
        ]
    )

    if dump_final:
        lines.append(
            f"dump final_state all custom {PROD_STEPS} {windows_path(dump_path)} id type x y z vx vy vz"
        )

    lines.extend(["", f"run {PROD_STEPS}", ""])
    return "\n".join(lines), rho_actual


def run_lammps_state(prefix, rho, dump_final=False, include_rdf=False):
    RESULT_DIR.mkdir(exist_ok=True)
    prefix = Path(prefix)
    input_path = Path(f"{prefix}.lmp")
    content, rho_actual = build_lammps_input(prefix, rho, dump_final=dump_final, include_rdf=include_rdf)
    input_path.write_text(content, encoding="ascii")
    subprocess.run(["lmp", "-in", str(input_path)], check=True)
    return rho_actual


def run_primary_benchmarks():
    RESULT_DIR.mkdir(exist_ok=True)
    print("Running NVE energy conservation test...")
    run_bamps_state(NVE_PREFIX, RHO, thermostat_type=0)
    print("Running NVT equilibrium benchmark...")
    run_bamps_state(NVT_PREFIX, RHO, thermostat_type=3)
    print("Running LAMMPS equilibrium reference...")
    run_lammps_state(LAMMPS_PREFIX, RHO, dump_final=True, include_rdf=True)


def run_compressibility_scan():
    SCAN_DIR.mkdir(parents=True, exist_ok=True)
    scan_rows = []

    for rho_target in DENSITY_SCAN:
        n_particles, rho_actual = actual_density(rho_target)
        bamps_prefix = SCAN_DIR / f"bamps_rho_{n_particles}"
        lammps_prefix = SCAN_DIR / f"lammps_rho_{n_particles}"

        print(f"Running compressibility scan at rho={rho_actual:.6f} (N={n_particles})...")
        run_bamps_state(bamps_prefix, rho_target, thermostat_type=3)
        run_lammps_state(lammps_prefix, rho_target, dump_final=False)

        bamps_ts = pd.read_csv(f"{bamps_prefix}_timeseries.csv")
        lammps_ts = pd.read_csv(f"{lammps_prefix}_timeseries.csv")

        bamps_summary = summarize_thermo(bamps_ts, "P_virial", n_particles, rho_actual)
        lammps_summary = summarize_thermo(
            lammps_ts, "P", n_particles, rho_actual, energies_are_per_particle=True
        )

        scan_rows.extend(
            [
                {
                    "code": "BAMPS",
                    "rho_target": rho_target,
                    "rho_actual": rho_actual,
                    "N": n_particles,
                    "T_mean": bamps_summary["T_mean"],
                    "P_mean": bamps_summary["P_mean"],
                    "Z_mean": bamps_summary["Z_mean"],
                },
                {
                    "code": "LAMMPS",
                    "rho_target": rho_target,
                    "rho_actual": rho_actual,
                    "N": n_particles,
                    "T_mean": lammps_summary["T_mean"],
                    "P_mean": lammps_summary["P_mean"],
                    "Z_mean": lammps_summary["Z_mean"],
                },
            ]
        )

    scan_df = pd.DataFrame(scan_rows)
    scan_df.to_csv(RESULT_DIR / "argon_compressibility_scan.csv", index=False)
    return scan_df


def load_bamps_snapshot(state_csv):
    df = pd.read_csv(state_csv)
    positions = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    velocities = df[["px", "py", "pz"]].to_numpy(dtype=np.float64)
    return positions, velocities


def load_lammps_dump(dump_path):
    with open(dump_path, "r", encoding="ascii") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    atom_header = "ITEM: ATOMS id type x y z vx vy vz"
    atom_header_idx = max(idx for idx, line in enumerate(lines) if line == atom_header)

    records = []
    for line in lines[atom_header_idx + 1 :]:
        if line.startswith("ITEM:"):
            break
        parts = line.split()
        records.append(
            {
                "id": int(parts[0]),
                "type": int(parts[1]),
                "x": float(parts[2]),
                "y": float(parts[3]),
                "z": float(parts[4]),
                "vx": float(parts[5]),
                "vy": float(parts[6]),
                "vz": float(parts[7]),
            }
        )

    df = pd.DataFrame(records)
    positions = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    velocities = df[["vx", "vy", "vz"]].to_numpy(dtype=np.float64)
    return positions, velocities


def load_lammps_rdf(filepath):
    data = []
    with open(filepath, "r", encoding="ascii") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) == 2:
                continue
            data.append([float(part) for part in parts])

    data = np.array(data, dtype=np.float64)
    return data[-RDF_BINS:, 1], data[-RDF_BINS:, 2]


def calculate_rdf_from_positions(pos, box_size, n_bins=RDF_BINS):
    n_particles = len(pos)
    hist = np.zeros(n_bins, dtype=np.float64)
    r_edges = np.linspace(0.0, box_size / 2.0, n_bins + 1)
    dr = r_edges[1] - r_edges[0]

    for i in range(n_particles):
        diff = pos - pos[i]
        diff -= box_size * np.round(diff / box_size)
        dist = np.linalg.norm(diff, axis=1)
        dist = dist[dist > 1e-6]
        counts, _ = np.histogram(dist, bins=r_edges)
        hist += counts

    rho = n_particles / (box_size**3)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    shell_vol = 4.0 * np.pi * (r_centers**2) * dr
    g_r = hist / (n_particles * rho * shell_vol)
    return r_centers, g_r


def calculate_structure_factor_from_rdf(r_vals, g_r, rho, q_vals):
    if len(r_vals) < 2:
        return np.ones_like(q_vals)

    r_max = r_vals[-1]
    window = np.sinc(r_vals / r_max)
    h_r = (g_r - 1.0) * window
    s_vals = np.empty_like(q_vals)

    for i, q_val in enumerate(q_vals):
        kernel = np.sinc((q_val * r_vals) / np.pi)
        integrand = (r_vals**2) * h_r * kernel
        s_vals[i] = 1.0 + 4.0 * np.pi * rho * np.trapezoid(integrand, r_vals)

    return s_vals


def first_shell_metrics(r_vals, g_r, rho):
    peak_mask = (r_vals >= 0.8) & (r_vals <= 2.0)
    peak_indices = np.where(peak_mask)[0]
    peak_idx = peak_indices[np.argmax(g_r[peak_mask])]

    min_mask = (r_vals > r_vals[peak_idx]) & (r_vals <= 2.4)
    min_indices = np.where(min_mask)[0]
    min_idx = min_indices[np.argmin(g_r[min_mask])]

    coord_integrand = 4.0 * np.pi * rho * (r_vals[: min_idx + 1] ** 2) * g_r[: min_idx + 1]
    coordination = np.trapezoid(coord_integrand, r_vals[: min_idx + 1])

    return {
        "rdf_peak_r": r_vals[peak_idx],
        "rdf_peak_g": g_r[peak_idx],
        "rdf_first_min_r": r_vals[min_idx],
        "coordination_number": coordination,
    }


def steady_window(df):
    start_idx = len(df) // 2
    return df.iloc[start_idx:].reset_index(drop=True)


def summarize_thermo(df, pressure_col, n_particles, rho_value, energies_are_per_particle=False):
    prod = steady_window(df)
    mean_t = prod["T"].mean()
    mean_p = prod[pressure_col].mean()
    mean_pe = prod["PE"].mean()
    mean_total = prod["TotalE"].mean()
    energy_scale = 1.0 if energies_are_per_particle else n_particles

    return {
        "T_mean": mean_t,
        "P_mean": mean_p,
        "Z_mean": mean_p / (rho_value * mean_t),
        "PE_per_particle": mean_pe / energy_scale,
        "TotalE_per_particle": mean_total / energy_scale,
    }


def maxwell_boltzmann_pdf(speed, temperature):
    prefactor = 4.0 * np.pi * (1.0 / (2.0 * np.pi * temperature)) ** 1.5
    return prefactor * (speed**2) * np.exp(-(speed**2) / (2.0 * temperature))


def speed_distribution(velocities, temperature, max_speed):
    speeds = np.linalg.norm(velocities, axis=1)
    hist, edges = np.histogram(speeds, bins=SPEED_BINS, range=(0.0, max_speed), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    theory = maxwell_boltzmann_pdf(centers, temperature)
    return pd.DataFrame({"speed": centers, "hist": hist, "theory": theory}), speeds


def low_q_extrapolation(q_vals, s_vals, fit_points=LOW_Q_FIT_POINTS):
    q_fit = q_vals[:fit_points]
    s_fit = s_vals[:fit_points]
    coeffs = np.polyfit(q_fit**2, s_fit, 1)
    s0 = coeffs[1]
    fit_curve = coeffs[0] * (q_vals**2) + coeffs[1]
    return s0, fit_curve


def estimate_kappa_from_scan(scan_df, code, rho_target):
    sub = scan_df[scan_df["code"] == code].sort_values("rho_actual")
    coeffs = np.polyfit(sub["rho_actual"], sub["P_mean"], 2)
    derivative = 2.0 * coeffs[0] * rho_target + coeffs[1]
    kappa_t = 1.0 / (rho_target * derivative)
    fit = np.poly1d(coeffs)
    return {
        "dPdrho": derivative,
        "kappa_T_eos": kappa_t,
        "fit_pressure": fit(rho_target),
        "poly": fit,
    }


def build_summary_table(scan_df):
    n_particles, rho_actual = actual_density(RHO)

    bamps_ts = pd.read_csv(RESULT_DIR / "nvt_timeseries.csv")
    lammps_ts = pd.read_csv(RESULT_DIR / "lammps_timeseries.csv")

    bamps_thermo = summarize_thermo(bamps_ts, "P_virial", n_particles, rho_actual)
    lammps_thermo = summarize_thermo(
        lammps_ts, "P", n_particles, rho_actual, energies_are_per_particle=True
    )

    bamps_pos, bamps_vel = load_bamps_snapshot(RESULT_DIR / "nvt_final_state.csv")
    lammps_pos, lammps_vel = load_lammps_dump(RESULT_DIR / "lammps_final_state.dump")

    r_bamps, g_bamps = calculate_rdf_from_positions(bamps_pos, BOX_SIZE)
    r_lammps, g_lammps = load_lammps_rdf(RESULT_DIR / "lammps_rdf.txt")
    r_lammps_snapshot, g_lammps_snapshot = calculate_rdf_from_positions(lammps_pos, BOX_SIZE)

    bamps_shell = first_shell_metrics(r_bamps, g_bamps, rho_actual)
    lammps_shell = first_shell_metrics(r_lammps, g_lammps, rho_actual)

    q_vals = np.linspace(0.4, STRUCTURE_FACTOR_Q_MAX, STRUCTURE_FACTOR_POINTS)
    s_bamps = calculate_structure_factor_from_rdf(r_bamps, g_bamps, rho_actual, q_vals)
    s_lammps = calculate_structure_factor_from_rdf(
        r_lammps_snapshot, g_lammps_snapshot, rho_actual, q_vals
    )

    peak_bamps = np.argmax(s_bamps[5:]) + 5
    peak_lammps = np.argmax(s_lammps[5:]) + 5

    bamps_s0, bamps_lowq_fit = low_q_extrapolation(q_vals, s_bamps)
    lammps_s0, lammps_lowq_fit = low_q_extrapolation(q_vals, s_lammps)

    bamps_kappa_s0 = bamps_s0 / (rho_actual * bamps_thermo["T_mean"])
    lammps_kappa_s0 = lammps_s0 / (rho_actual * lammps_thermo["T_mean"])

    bamps_scan = estimate_kappa_from_scan(scan_df, "BAMPS", rho_actual)
    lammps_scan = estimate_kappa_from_scan(scan_df, "LAMMPS", rho_actual)

    max_speed = max(np.linalg.norm(bamps_vel, axis=1).max(), np.linalg.norm(lammps_vel, axis=1).max()) * 1.05
    speed_bamps, bamps_speeds = speed_distribution(bamps_vel, bamps_thermo["T_mean"], max_speed)
    speed_lammps, lammps_speeds = speed_distribution(lammps_vel, lammps_thermo["T_mean"], max_speed)

    speed_df = pd.DataFrame(
        {
            "speed": speed_bamps["speed"],
            "BAMPS_hist": speed_bamps["hist"],
            "BAMPS_theory": speed_bamps["theory"],
            "LAMMPS_hist": speed_lammps["hist"],
            "LAMMPS_theory": speed_lammps["theory"],
        }
    )

    rows = [
        ("T_mean", bamps_thermo["T_mean"], lammps_thermo["T_mean"]),
        ("P_mean", bamps_thermo["P_mean"], lammps_thermo["P_mean"]),
        ("Z_mean", bamps_thermo["Z_mean"], lammps_thermo["Z_mean"]),
        ("PE_per_particle", bamps_thermo["PE_per_particle"], lammps_thermo["PE_per_particle"]),
        ("TotalE_per_particle", bamps_thermo["TotalE_per_particle"], lammps_thermo["TotalE_per_particle"]),
        ("rdf_peak_r", bamps_shell["rdf_peak_r"], lammps_shell["rdf_peak_r"]),
        ("rdf_peak_g", bamps_shell["rdf_peak_g"], lammps_shell["rdf_peak_g"]),
        ("rdf_first_min_r", bamps_shell["rdf_first_min_r"], lammps_shell["rdf_first_min_r"]),
        ("coordination_number", bamps_shell["coordination_number"], lammps_shell["coordination_number"]),
        ("sk_peak_q", q_vals[peak_bamps], q_vals[peak_lammps]),
        ("sk_peak_value", s_bamps[peak_bamps], s_lammps[peak_lammps]),
        ("S0_lowq", bamps_s0, lammps_s0),
        ("kappa_T_from_S0", bamps_kappa_s0, lammps_kappa_s0),
        ("dPdrho_fit", bamps_scan["dPdrho"], lammps_scan["dPdrho"]),
        ("kappa_T_from_EOS", bamps_scan["kappa_T_eos"], lammps_scan["kappa_T_eos"]),
        ("v_rms", np.sqrt(np.mean(bamps_speeds**2)), np.sqrt(np.mean(lammps_speeds**2))),
        (
            "mb_rmse",
            np.sqrt(np.mean((speed_bamps["hist"] - speed_bamps["theory"]) ** 2)),
            np.sqrt(np.mean((speed_lammps["hist"] - speed_lammps["theory"]) ** 2)),
        ),
    ]

    summary_df = pd.DataFrame(rows, columns=["observable", "BAMPS", "LAMMPS"])
    denom = summary_df["LAMMPS"].replace(0.0, np.nan).abs()
    summary_df["rel_error_pct"] = 100.0 * (summary_df["BAMPS"] - summary_df["LAMMPS"]).abs() / denom
    summary_df["rel_error_pct"] = summary_df["rel_error_pct"].fillna(0.0)

    structure_df = pd.DataFrame(
        {
            "q": q_vals,
            "S_bamps": s_bamps,
            "S_bamps_lowq_fit": bamps_lowq_fit,
            "S_lammps": s_lammps,
            "S_lammps_lowq_fit": lammps_lowq_fit,
        }
    )
    return summary_df, structure_df, speed_df, (r_bamps, g_bamps), (r_lammps, g_lammps), rho_actual


def plot_validation():
    scan_df = pd.read_csv(RESULT_DIR / "argon_compressibility_scan.csv")
    summary_df, structure_df, speed_df, bamps_rdf, lammps_rdf, rho_actual = build_summary_table(scan_df)

    summary_df.to_csv(RESULT_DIR / "argon_equilibrium_summary.csv", index=False)
    structure_df.to_csv(RESULT_DIR / "argon_structure_factor.csv", index=False)
    speed_df.to_csv(RESULT_DIR / "argon_speed_distribution.csv", index=False)

    nve_df = pd.read_csv(RESULT_DIR / "nve_timeseries.csv")
    total_e = nve_df["TotalE"]
    rel_drift = (total_e - total_e.iloc[0]) / max(abs(total_e.iloc[0]), 1e-12)

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))

    ax_e = axes[0, 0]
    ax_e.plot(nve_df["Step"], rel_drift, color="navy", lw=2)
    ax_e.axhline(0.0, color="gray", linestyle=":", alpha=0.6)
    ax_e.set_title("NVE Relative Total-Energy Drift")
    ax_e.set_xlabel("Production Step")
    ax_e.set_ylabel(r"$(E - E_0) / |E_0|$")
    ax_e.grid(True, alpha=0.3)

    r_bamps, g_bamps = bamps_rdf
    r_lammps, g_lammps = lammps_rdf
    ax_g = axes[0, 1]
    ax_g.plot(r_bamps, g_bamps, "r-", lw=2, label="BAMPS GPU")
    ax_g.plot(r_lammps, g_lammps, "k--", lw=2, label="LAMMPS")
    ax_g.set_title(r"Radial Distribution Function $g(r)$")
    ax_g.set_xlabel(r"$r / \sigma$")
    ax_g.set_ylabel(r"$g(r)$")
    ax_g.set_xlim(0.0, 4.0)
    ax_g.axhline(1.0, color="gray", linestyle=":", alpha=0.6)
    ax_g.grid(True, alpha=0.3)
    ax_g.legend()

    ax_s = axes[1, 0]
    ax_s.plot(structure_df["q"], structure_df["S_bamps"], "r-", lw=2, label="BAMPS GPU")
    ax_s.plot(structure_df["q"], structure_df["S_lammps"], "k--", lw=2, label="LAMMPS")
    ax_s.plot(structure_df["q"], structure_df["S_bamps_lowq_fit"], color="salmon", linestyle=":", alpha=0.8)
    ax_s.plot(structure_df["q"], structure_df["S_lammps_lowq_fit"], color="gray", linestyle=":", alpha=0.8)
    ax_s.set_title(r"Static Structure Factor $S(k)$")
    ax_s.set_xlabel(r"$k \sigma$")
    ax_s.set_ylabel(r"$S(k)$")
    ax_s.grid(True, alpha=0.3)
    ax_s.legend()

    ax_v = axes[1, 1]
    ax_v.plot(speed_df["speed"], speed_df["BAMPS_hist"], "r-", lw=2, label="BAMPS hist")
    ax_v.plot(speed_df["speed"], speed_df["BAMPS_theory"], color="salmon", linestyle="--", lw=2, label="BAMPS MB")
    ax_v.plot(speed_df["speed"], speed_df["LAMMPS_hist"], "k-", lw=2, label="LAMMPS hist")
    ax_v.plot(speed_df["speed"], speed_df["LAMMPS_theory"], color="gray", linestyle="--", lw=2, label="LAMMPS MB")
    ax_v.set_title("Speed Distribution vs Maxwell-Boltzmann")
    ax_v.set_xlabel(r"$|v|$")
    ax_v.set_ylabel("Probability density")
    ax_v.grid(True, alpha=0.3)
    ax_v.legend()

    ax_k = axes[2, 0]
    for code, color in [("BAMPS", "red"), ("LAMMPS", "black")]:
        sub = scan_df[scan_df["code"] == code].sort_values("rho_actual")
        ax_k.plot(sub["rho_actual"], sub["P_mean"], "o-", color=color, label=f"{code} data")
        coeffs = np.polyfit(sub["rho_actual"], sub["P_mean"], 2)
        fit = np.poly1d(coeffs)
        rho_grid = np.linspace(sub["rho_actual"].min(), sub["rho_actual"].max(), 200)
        ax_k.plot(rho_grid, fit(rho_grid), color=color, linestyle="--", alpha=0.7, label=f"{code} fit")

    ax_k.axvline(rho_actual, color="steelblue", linestyle=":", alpha=0.7)
    ax_k.set_title(r"Local EoS Scan for $\kappa_T$")
    ax_k.set_xlabel(r"$\rho$")
    ax_k.set_ylabel(r"$P$")
    ax_k.grid(True, alpha=0.3)
    ax_k.legend()

    ax_text = axes[2, 1]
    ax_text.axis("off")
    lines = [
        "Observable              BAMPS      LAMMPS    RelErr%",
        "---------------------------------------------------",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"{row['observable']:<20} {row['BAMPS']:>8.4f}   {row['LAMMPS']:>8.4f}   {row['rel_error_pct']:>7.3f}"
        )

    ax_text.text(
        0.0,
        1.0,
        "\n".join(lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
    )
    ax_text.set_title("Equilibrium Summary including Compressibility")

    plt.tight_layout()
    plt.savefig(RESULT_DIR / "argon_equilibrium_validation.png")


if __name__ == "__main__":
    os.makedirs(RESULT_DIR, exist_ok=True)
    compile_code()
    run_primary_benchmarks()
    run_compressibility_scan()
    plot_validation()
