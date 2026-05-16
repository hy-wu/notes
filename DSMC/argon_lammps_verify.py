import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# --- Configuration for Argon/LJ Benchmark ---
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
    sample_n = min(n, 1000)
    subset_pos = pos[:sample_n]
    
    dr = (box_size / 2.0) / n_bins
    r_bins = np.linspace(0, box_size / 2.0, n_bins + 1)
    hist = np.zeros(n_bins)
    
    for i in range(sample_n):
        diff = subset_pos[i+1:] - subset_pos[i]
        diff -= box_size * np.round(diff / box_size)
        dist = np.linalg.norm(diff, axis=1)
        counts, _ = np.histogram(dist, bins=r_bins)
        hist += counts

    rho = n / (box_size**3)
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2.0
    shell_vol = 4.0 * np.pi * (r_centers**2) * dr
    gr = (hist / sample_n) / (rho * shell_vol)
    return r_centers, gr

def load_lammps_rdf(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            parts = line.split()
            if len(parts) == 2: continue # Skip timestep/count line
            data.append([float(p) for p in parts])
    
    # LAMMPS outputs multiple snapshots if fix ave/time is used. 
    # We take the average or the last one.
    # Actually lammps_rdf.txt from fix ave/time contains blocks.
    # Let's just find the last block.
    data = np.array(data)
    # The columns are: Index, r, g(r), coord_num
    r = data[-150:, 1]
    gr = data[-150:, 2]
    return r, gr

def plot_validation():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Energy Conservation (NVE)
    ax_e = axes[0]
    nve_df = pd.read_csv("argon_results/nve_timeseries.csv")
    e_init = nve_df['TotalE'].iloc[0]
    e_drift = (nve_df['TotalE'] - e_init) / abs(e_init)
    ax_e.plot(nve_df['Step'], e_drift, 'b-', label='BAMPS GPU Engine (NVE)')
    ax_e.set_title("Energy Conservation (Argon/LJ)")
    ax_e.set_xlabel("Step")
    ax_e.set_ylabel(r"$\Delta E / |E_0|$")
    ax_e.grid(True, alpha=0.3)
    ax_e.legend()
    
    # 2. RDF g(r) (NVT)
    ax_g = axes[1]
    r_gpu, gr_gpu = calculate_rdf("argon_results/nvt_final_state.csv", BOX_SIZE)
    ax_g.plot(r_gpu, gr_gpu, 'r-', lw=2, label='BAMPS GPU Engine (NHC)')
    
    # Reference from LAMMPS
    try:
        r_ref, gr_ref = load_lammps_rdf("lammps_rdf.txt")
        ax_g.plot(r_ref, gr_ref, 'k--', alpha=0.7, label='LAMMPS Reference (Standard)')
    except Exception as e:
        print(f"Could not load LAMMPS reference: {e}")
    
    ax_g.set_title(r"Radial Distribution Function $g(r)$ 对比 ($\rho^*=0.8, T^*=1.0$)")
    ax_g.set_xlabel(r"$r / \sigma$")
    ax_g.set_ylabel(r"$g(r)$")
    ax_g.set_xlim(0, 4.0)
    ax_g.grid(True, alpha=0.3)
    ax_g.legend()
    
    plt.tight_layout()
    plt.savefig("argon_results/argon_validation.png")
    print("Validation plots updated with LAMMPS reference at argon_results/argon_validation.png")

if __name__ == "__main__":
    # Assuming code is already compiled and lammps data generated
    # If not, uncomment run_argon_bench()
    run_argon_bench()
    plot_validation()
