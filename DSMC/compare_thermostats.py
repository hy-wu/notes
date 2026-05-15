import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import maxwell

# --- Configuration ---
THERMOSTATS = {
    0: "Null (NVE)",
    1: "Andersen",
    2: "Langevin",
    3: "NHC (Global)"
}
BIN_FILE = "bamps_compare.exe"
SRC_FILE = "src/thermostat_compare.cu"
TARGET_T = 1.2
MASS = 1.0

def compile_code():
    print("Compiling comparison harness...")
    subprocess.run(["nvcc", "-O3", "-arch=sm_89", SRC_FILE, "-o", BIN_FILE], check=True)

def run_simulations():
    os.makedirs("comparison_results", exist_ok=True)
    for tid, name in THERMOSTATS.items():
        prefix = f"comparison_results/{name.split()[0].lower()}"
        print(f"Running simulation: {name}...")
        subprocess.run([f"./{BIN_FILE}", str(tid), prefix], check=True)

def plot_results():
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Temperature Time Series
    ax_t = axes[0, 0]
    # 2. Pressure (Virial) Time Series
    ax_pv = axes[0, 1]
    # 3. Pressure (Wall) Time Series
    ax_pw = axes[1, 0]
    # 4. Momentum Distribution
    ax_dist = axes[1, 1]
    
    colors = ['black', 'blue', 'orange', 'green']
    
    for i, (tid, name) in enumerate(THERMOSTATS.items()):
        prefix = f"comparison_results/{name.split()[0].lower()}"
        ts_df = pd.read_csv(f"{prefix}_timeseries.csv")
        mom_df = pd.read_csv(f"{prefix}_momenta.csv")
        
        c = colors[i]
        label = name
        
        ax_t.plot(ts_df['Step'], ts_df['T'], color=c, label=label, alpha=0.8)
        ax_pv.plot(ts_df['Step'], ts_df['P_virial'], color=c, label=label, alpha=0.8)
        ax_pw.plot(ts_df['Step'], ts_df['P_wall'], color=c, label=label, alpha=0.8)
        
        # Momentum Distribution (Normalized Histogram)
        hist, bins = np.histogram(mom_df['p_mag'], bins=50, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax_dist.plot(bin_centers, hist, color=c, label=label)

    # Theoretical Maxwell-Boltzmann for Target T
    # P(p) dp = 4*pi*p^2 * (1/(2*pi*m*kT))^(3/2) * exp(-p^2/(2*m*kT)) dp
    # In scipy.stats.maxwell, scale = sqrt(kT/m)
    p_theory = np.linspace(0, 5, 200)
    scale = np.sqrt(TARGET_T / MASS)
    ax_dist.plot(p_theory, maxwell.pdf(p_theory, scale=scale), 'r--', lw=2, label=f'Theory (T={TARGET_T})')

    # Formatting
    ax_t.set_title("Temperature Evolution")
    ax_t.set_xlabel("Step")
    ax_t.set_ylabel("Measured T")
    ax_t.axhline(TARGET_T, color='red', linestyle=':', label='Target T')
    ax_t.legend()
    ax_t.grid(True, alpha=0.3)

    ax_pv.set_title("Virial Pressure Evolution")
    ax_pv.set_xlabel("Step")
    ax_pv.set_ylabel("P_virial")
    ax_pv.legend()
    ax_pv.grid(True, alpha=0.3)

    ax_pw.set_title("Wall Pressure Evolution")
    ax_pw.set_xlabel("Step")
    ax_pw.set_ylabel("P_wall")
    ax_pw.legend()
    ax_pw.grid(True, alpha=0.3)

    ax_dist.set_title("Terminal Momentum Distribution")
    ax_dist.set_xlabel("|p|")
    ax_dist.set_ylabel("Density")
    ax_dist.legend()
    ax_dist.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("comparison_results/thermostat_comparison.png")
    print("Comparison complete. Results saved to comparison_results/thermostat_comparison.png")

if __name__ == "__main__":
    compile_code()
    run_simulations()
    plot_results()
