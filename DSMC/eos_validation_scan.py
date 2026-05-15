import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# --- Configuration ---
THERMOSTATS = {
    1: "Andersen",
    2: "Langevin",
    3: "NHC"
}
BIN_FILE = "bamps_compare.exe"
SRC_FILE = "src/thermostat_compare.cu"
STEPS = 2000

# Fixed Baseline Parameters
BASE_SIG = 0.2
BASE_EPS = 0.4
BASE_RHO = 4.0
BASE_T   = 1.2

# Scan Ranges
SCAN_RANGES = {
    "sigma":   np.linspace(0.1, 0.4, 10),
    "epsilon": np.linspace(0.0, 1.0, 10),
    "temp":    np.linspace(0.5, 2.5, 10),
    "rho":     np.linspace(1.0, 8.0, 10)
}

def compile_code():
    print("Compiling comparison harness...")
    subprocess.run(["nvcc", "-O3", "-arch=sm_89", SRC_FILE, "-o", BIN_FILE], check=True)

def run_sim(tid, prefix, sig, eps, rho, T, steps):
    cmd = [f"./{BIN_FILE}", str(tid), prefix, str(sig), str(eps), str(rho), str(T), str(steps)]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Read the time series to get the averaged P_wall and T_meas from the last 50% steps
        df = pd.read_csv(f"{prefix}_timeseries.csv")
        last_half = df.iloc[len(df)//2:]
        return last_half['T'].mean(), last_half['P_wall'].mean()
    except Exception as e:
        print(f"Error running sim: {e}")
        return None, None

def perform_scans():
    os.makedirs("eos_scans", exist_ok=True)
    results = {name: {var: [] for var in SCAN_RANGES} for name in THERMOSTATS.values()}

    for tid, tname in THERMOSTATS.items():
        print(f"\n--- Starting Scans for Thermostat: {tname} ---")
        
        # 1. Sigma Scan
        print(f"Scanning Sigma...")
        for sig in SCAN_RANGES["sigma"]:
            t, p = run_sim(tid, f"eos_scans/tmp_{tname}_sig", sig, BASE_EPS, BASE_RHO, BASE_T, STEPS)
            if t is not None: results[tname]["sigma"].append([sig, p/t])

        # 2. Epsilon Scan
        print(f"Scanning Epsilon...")
        for eps in SCAN_RANGES["epsilon"]:
            t, p = run_sim(tid, f"eos_scans/tmp_{tname}_eps", BASE_SIG, eps, BASE_RHO, BASE_T, STEPS)
            if t is not None: results[tname]["epsilon"].append([eps, p/t])

        # 3. Temperature Scan
        print(f"Scanning Temperature...")
        for T in SCAN_RANGES["temp"]:
            t, p = run_sim(tid, f"eos_scans/tmp_{tname}_T", BASE_SIG, BASE_EPS, BASE_RHO, T, STEPS)
            if t is not None: results[tname]["temp"].append([T, p/t])

        # 4. Density Scan
        print(f"Scanning Density...")
        for rho in SCAN_RANGES["rho"]:
            t, p = run_sim(tid, f"eos_scans/tmp_{tname}_rho", BASE_SIG, BASE_EPS, rho, BASE_T, STEPS)
            if t is not None: results[tname]["rho"].append([rho, p/t])

    return results

def get_theory_z(rho, T, sig, eps):
    # Z = P / (rho * T) = Z_cs - a*rho / T
    eta = (np.pi / 6.0) * rho * (sig**3)
    if eta >= 1.0: return np.nan
    Z_cs = (1 + eta + eta**2 - eta**3) / (1 - eta)**3
    a = (16.0 / 9.0) * np.pi * eps * (sig**3)
    return Z_cs - (a * rho / T)

def plot_results(results):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    vars_list = list(SCAN_RANGES.keys())
    
    titles = {
        "sigma":   f"Sigma Scan (fixed eps={BASE_EPS}, rho={BASE_RHO}, T={BASE_T})",
        "epsilon": f"Epsilon Scan (fixed sig={BASE_SIG}, rho={BASE_RHO}, T={BASE_T})",
        "temp":    f"Temperature Scan (fixed sig={BASE_SIG}, eps={BASE_EPS}, rho={BASE_RHO})",
        "rho":     f"Density Scan (fixed sig={BASE_SIG}, eps={BASE_EPS}, T={BASE_T})"
    }
    
    xlabels = {
        "sigma": r"$\sigma$",
        "epsilon": r"$\epsilon$",
        "temp": r"$T$",
        "rho": r"$\rho$"
    }

    for i, var in enumerate(vars_list):
        ax = axes[i//2, i%2]
        
        # Plot Theory (Corrected: P/T = rho * Z)
        x_theory = np.linspace(SCAN_RANGES[var][0], SCAN_RANGES[var][-1], 100)
        p_over_t_theory = []
        for x in x_theory:
            if var == "sigma":   
                rho_val = BASE_RHO
                z = get_theory_z(rho_val, BASE_T, x, BASE_EPS)
            elif var == "epsilon": 
                rho_val = BASE_RHO
                z = get_theory_z(rho_val, BASE_T, BASE_SIG, x)
            elif var == "temp":    
                rho_val = BASE_RHO
                z = get_theory_z(rho_val, x, BASE_SIG, BASE_EPS)
            elif var == "rho":     
                rho_val = x
                z = get_theory_z(rho_val, BASE_T, BASE_SIG, BASE_EPS)
            p_over_t_theory.append(rho_val * z)
        
        ax.plot(x_theory, p_over_t_theory, 'k--', alpha=0.6, label="Theory (rho * Z_cs_vlasov)")

        # Plot Data for each thermostat
        for tname in THERMOSTATS.values():
            data = np.array(results[tname][var])
            # P_wall/T = rho * Z_wall
            # We want to plot P_wall/T on y-axis, but normalized by rho to see Z? 
            # The user asked for P_wall/T.
            ax.plot(data[:, 0], data[:, 1], 'o-', label=f"{tname}")

        ax.set_title(titles[var])
        ax.set_xlabel(xlabels[var])
        ax.set_ylabel(r"$P_{wall}/T$")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("eos_scans/eos_validation_comprehensive.png")
    print("\nScan complete. Comprehensive result saved to eos_scans/eos_validation_comprehensive.png")

    # Document fixed parameters
    with open("eos_scans/scan_parameters.txt", "w") as f:
        f.write("BAMPS Modern EoS Validation Scan Parameters\n")
        f.write("============================================\n\n")
        f.write(f"Steps per point: {STEPS}\n")
        f.write(f"Density Baseline (rho): {BASE_RHO}\n")
        f.write(f"Temperature Baseline (T): {BASE_T}\n")
        f.write(f"Sigma Baseline (sig): {BASE_SIG}\n")
        f.write(f"Epsilon Baseline (eps): {BASE_EPS}\n\n")
        f.write("Scanned Ranges:\n")
        for var, rng in SCAN_RANGES.items():
            f.write(f"- {var}: [{rng[0]:.2f}, {rng[-1]:.2f}] with {len(rng)} points\n")

if __name__ == "__main__":
    compile_code()
    results = perform_scans()
    plot_results(results)
