import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# --- Configuration for Argon/LJ Benchmark ---
# Standard LJ Liquid State: rho* = 0.8, T* = 1.0
BIN_FILE = "bamps_compare.exe"
SRC_FILE = "src/thermostat_compare.cu"
STEPS = 15000
BOX_SIZE = 12.0
SIGMA = 1.0
EPSILON = 1.0
RHO = 0.8
T_TARGET = 1.0

def compile_code():
    print("Compiling comparison harness...")
    subprocess.run(["nvcc", "-O3", "-arch=sm_89", "-std=c++17", SRC_FILE, "-o", BIN_FILE], check=True)

def run_argon_bench():
    os.makedirs("argon_results", exist_ok=True)
    
    # 1. Run NVE for Energy Conservation
    print("Running NVE (Energy Conservation check)...")
    cmd_nve = [f"./{BIN_FILE}", "0", "1", "argon_results/nve", str(SIGMA), str(EPSILON), str(RHO), str(T_TARGET), str(STEPS), str(BOX_SIZE)]
    subprocess.run(cmd_nve, check=True)
    
    # 2. Run NVT (NHC) for RDF and Structural check
    print("Running NVT (NHC, Structural check)...")
    cmd_nvt = [f"./{BIN_FILE}", "3", "1", "argon_results/nvt", str(SIGMA), str(EPSILON), str(RHO), str(T_TARGET), str(STEPS), str(BOX_SIZE)]
    subprocess.run(cmd_nvt, check=True)

def calculate_rdf(state_csv, box_size, n_bins=150):
    df = pd.read_csv(state_csv)
    pos = df[['x', 'y', 'z']].values
    n = len(pos)
    
    # Distance calculation (optimized for large N)
    # Using a simple O(N^2) but limited to a subset for speed if needed
    # For N=1382, O(N^2) is fine (~2 million pairs)
    from scipy.spatial.distance import pdist
    distances = pdist(pos)
    
    # Apply MIC (approximate since we don't know the exact pair vectors here easily without re-coding)
    # But since we have the positions in a box centered at 0, we can adjust.
    # Actually, for pdist, we need to handle PBC manually or use a library.
    # Let's do a manual calculation for a subset to be precise with PBC.
    sample_n = min(n, 1000)
    subset_pos = pos[:sample_n]
    
    hist = np.zeros(n_bins)
    dr = (box_size / 2.0) / n_bins
    r_bins = np.linspace(0, box_size / 2.0, n_bins + 1)
    
    for i in range(sample_n):
        diff = subset_pos[i+1:] - subset_pos[i]
        diff -= box_size * np.round(diff / box_size)
        dist = np.linalg.norm(diff, axis=1)
        counts, _ = np.histogram(dist, bins=r_bins)
        hist += counts

    # Normalize
    # g(r) = (counts / sample_n) / (rho * shell_volume)
    rho = n / (box_size**3)
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2.0
    shell_vol = 4.0 * np.pi * (r_centers**2) * dr
    gr = (hist / sample_n) / (rho * shell_vol)
    
    return r_centers, gr

def plot_validation():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Energy Conservation (NVE)
    ax_e = axes[0]
    nve_df = pd.read_csv("argon_results/nve_timeseries.csv")
    e_init = nve_df['TotalE'].iloc[0]
    e_drift = (nve_df['TotalE'] - e_init) / abs(e_init)
    ax_e.plot(nve_df['Step'], e_drift, 'b-', label='NVE Energy Drift')
    ax_e.set_title("NVE Energy Conservation (Argon/LJ)")
    ax_e.set_xlabel("Step")
    ax_e.set_ylabel(r"$\Delta E / |E_0|$")
    ax_e.grid(True, alpha=0.3)
    ax_e.legend()
    
    # 2. RDF g(r) (NVT)
    ax_g = axes[1]
    r, gr = calculate_rdf("argon_results/nvt_final_state.csv", BOX_SIZE)
    ax_g.plot(r, gr, 'r-', lw=2, label='BAMPS GPU Engine')
    
    # Placeholder for LAMMPS Reference (Literature data for LJ rho=0.8, T=1.0)
    # Ref: Johnson et al., Mol. Phys. 1993
    # Typical first peak is at ~1.1 sigma with height ~2.4
    # We'll just mark the expected peak
    ax_g.axvline(1.122, color='gray', linestyle='--', alpha=0.5, label='Expected 1st Peak (2^(1/6))')
    
    ax_g.set_title(r"Radial Distribution Function $g(r)$ (NVT, $\rho^*=0.8, T^*=1.0$)")
    ax_g.set_xlabel(r"$r / \sigma$")
    ax_g.set_ylabel(r"$g(r)$")
    ax_g.set_xlim(0, 4.0)
    ax_g.grid(True, alpha=0.3)
    ax_g.legend()
    
    plt.tight_layout()
    plt.savefig("argon_results/argon_validation.png")
    print("Validation plots saved to argon_results/argon_validation.png")

if __name__ == "__main__":
    compile_code()
    run_argon_bench()
    plot_validation()
