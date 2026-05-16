import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import time
from scipy.integrate import quad

# --- Configuration ---
THERMOSTATS = {
    1: "Andersen",
    2: "Langevin",
    3: "NHC"
}
BIN_FILE = "bamps_compare.exe"
SRC_FILE = "src/thermostat_compare.cu"
STEPS = 10000  
BOX_SIZE = 14.0 

# Fixed Baseline Parameters
BASE_SIG = 0.2
BASE_EPS = 0.4
BASE_RHO = 4.0
BASE_T   = 1.2

# Scan Ranges
SCAN_RANGES = {
    "sigma":   np.linspace(0.1, 0.4, 8),
    "epsilon": np.linspace(0.0, 1.0, 8),
    "temp":    np.linspace(0.5, 2.5, 8),
    "rho":     np.linspace(1.0, 8.0, 8)
}

MASTER_LOG_PATH = "eos_scans/simulation_master_log.csv"

def compile_code():
    print("Compiling comparison harness...")
    subprocess.run(["nvcc", "-O3", "-arch=sm_89", "-std=c++17", SRC_FILE, "-o", BIN_FILE], check=True)

def run_sim(tid, prefix, sig, eps, rho, T, steps, box):
    start_time = time.time()
    cmd = [f"./{BIN_FILE}", str(tid), "0", prefix, str(sig), str(eps), str(rho), str(T), str(steps), str(box)]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        runtime = time.time() - start_time
        
        df = pd.read_csv(f"{prefix}_timeseries.csv")
        prod_df = df.iloc[len(df)//2:]
        half_idx = len(prod_df)//2
        q3 = prod_df.iloc[:half_idx]
        q4 = prod_df.iloc[half_idx:]
        
        t_err = abs(q3['T'].mean() - q4['T'].mean()) / (T + 1e-9)
        p_err = abs(q3['P_wall'].mean() - q4['P_wall'].mean()) / (q4['P_wall'].mean() + 1e-9)
        
        status = "OK"
        if t_err > 0.05 or p_err > 0.10:
            status = f"WARN"

        res = {
            "T_mean": prod_df['T'].mean(),
            "P_wall_mean": prod_df['P_wall'].mean(),
            "P_virial_mean": prod_df['P_virial'].mean(),
            "Runtime_sec": runtime,
            "Status": status
        }
        return res
    except Exception as e:
        print(f"Error running sim: {e}")
        return None

def log_to_master(tname, var, val, params, res):
    # params: [N, box, sig, eps, T_target, steps]
    data = {
        "Thermostat": tname,
        "ScanVar": var,
        "ScanValue": val,
        "N": params[0],
        "BoxSize": params[1],
        "Sigma": params[2],
        "Epsilon": params[3],
        "TargetT": params[4],
        "Steps": params[5],
        "MeasuredT": res["T_mean"],
        "MeasuredP_wall": res["P_wall_mean"],
        "MeasuredP_virial": res["P_virial_mean"],
        "Runtime_sec": res["Runtime_sec"],
        "Status": res["Status"]
    }
    df = pd.DataFrame([data])
    header = not os.path.exists(MASTER_LOG_PATH)
    df.to_csv(MASTER_LOG_PATH, mode='a', index=False, header=header)

# --- Rigorous Theory: B2 Virial Integration ---

def lj_potential(r, sigma, epsilon):
    if r < 1e-6: return 1e10
    s6 = (sigma / r)**6
    return 4 * epsilon * (s6**2 - s6)

def calculate_b2_lj(T, sig, eps):
    # B2 = -2*pi * integral (exp(-u/kT) - 1) * r^2 dr
    def integrand(r):
        u = lj_potential(r, sig, eps)
        return (np.exp(-u / T) - 1) * (r**2)
    
    # Integrate from 0 to 5*sig (beyond which potential is negligible)
    # Using 10*sig to be very safe
    integral, _ = quad(integrand, 0, 10 * sig)
    return -2 * np.pi * integral

def get_rigorous_z(rho, T, sig, eps):
    eta = (np.pi / 6.0) * rho * (sig**3)
    if eta >= 1.0: return np.nan
    # Carnahan-Starling for Hard Sphere repulsion
    Z_cs = (1 + eta + eta**2 - eta**3) / (1 - eta)**3
    
    # Correcting the attractive part using B2 integral
    # B2_total = B2_HS + B2_attractive_delta
    # B2_HS = 2/3 * pi * sigma^3
    # Z_approx = Z_cs + (B2_LJ - B2_HS) * rho
    b2_hs = (2.0/3.0) * np.pi * (sig**3)
    b2_lj = calculate_b2_lj(T, sig, eps)
    
    return Z_cs + (b2_lj - b2_hs) * rho

def perform_scans():
    os.makedirs("eos_scans", exist_ok=True)
    if os.path.exists(MASTER_LOG_PATH): os.remove(MASTER_LOG_PATH)
    
    results = {name: {var: [] for var in SCAN_RANGES} for name in THERMOSTATS.values()}

    for tid, tname in THERMOSTATS.items():
        print(f"\n--- Starting Scans for Thermostat: {tname} ---")
        
        for var in SCAN_RANGES:
            print(f"Scanning {var}...")
            for val in SCAN_RANGES[var]:
                sig, eps, rho, T = BASE_SIG, BASE_EPS, BASE_RHO, BASE_T
                if var == "sigma": sig = val
                elif var == "epsilon": eps = val
                elif var == "temp": T = val
                elif var == "rho": rho = val
                
                res = run_sim(tid, f"eos_scans/tmp_{tname}_{var}", sig, eps, rho, T, STEPS, BOX_SIZE)
                if res is not None:
                    n_particles = int(rho * BOX_SIZE**3)
                    log_to_master(tname, var, val, [n_particles, BOX_SIZE, sig, eps, T, STEPS], res)
                    
                    p_over_t = res["P_wall_mean"] / res["T_mean"]
                    results[tname][var].append([val, p_over_t, res["Status"]])
                    print(f"  {var}={val:.2f} | P/T={p_over_t:.4f} | {res['Status']} | {res['Runtime_sec']:.1f}s")

    return results

def plot_results(results):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    vars_list = list(SCAN_RANGES.keys())
    
    titles = {
        "sigma":   r"Sigma Scan ($\epsilon$=0.4, $\rho$=4.0, $T$=1.2)",
        "epsilon": r"Epsilon Scan ($\sigma$=0.2, $\rho$=4.0, $T$=1.2)",
        "temp":    r"Temperature Scan ($\sigma$=0.2, $\epsilon$=0.4, $\rho$=4.0)",
        "rho":     r"Density Scan ($\sigma$=0.2, $\epsilon$=0.4, $T$=1.2)"
    }
    
    xlabels = {"sigma": r"$\sigma$", "epsilon": r"$\epsilon$", "temp": r"$T$", "rho": r"$\rho$"}

    for i, var in enumerate(vars_list):
        ax = axes[i//2, i%2]
        
        # 1. Plot Rigorous Theory (B2 Integrated)
        x_theory = np.linspace(SCAN_RANGES[var][0], SCAN_RANGES[var][-1], 50)
        p_over_t_rigorous = []
        for x in x_theory:
            r_val = BASE_RHO if var != "rho" else x
            t_val = BASE_T if var != "temp" else x
            s_val = BASE_SIG if var != "sigma" else x
            e_val = BASE_EPS if var != "epsilon" else x
            z_rig = get_rigorous_z(r_val, t_val, s_val, e_val)
            p_over_t_rigorous.append(r_val * z_rig)
        
        ax.plot(x_theory, p_over_t_rigorous, 'r--', lw=2, label="Rigorous Theory (CS + B2 Integral)")

        # 2. Plot Data
        for tname in THERMOSTATS.values():
            data = results[tname][var]
            x_vals = [d[0] for d in data]
            y_vals = [d[1] for d in data]
            ax.plot(x_vals, y_vals, 'o-', alpha=0.7, label=f"{tname} Data")

        ax.set_title(titles[var])
        ax.set_xlabel(xlabels[var])
        ax.set_ylabel(r"$P_{wall}/T$")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("eos_scans/eos_validation_rigorous.png")
    print("\nScan complete. Comprehensive result saved to eos_scans/eos_validation_rigorous.png")
    print(f"Master simulation log saved to {MASTER_LOG_PATH}")

if __name__ == "__main__":
    compile_code()
    results = perform_scans()
    plot_results(results)
