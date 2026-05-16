import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Configuration ---
BIN_FILE = "bamps_compare.exe"
SRC_FILE = "src/thermostat_compare.cu"
STEPS = 15000
BOX_SIZE = 12.0
SIGMA = 1.0
EPSILON = 1.0
RHO = 0.8
T_TARGET = 1.0

def compile_code():
    print("Compiling rigorous harness...")
    subprocess.run(["nvcc", "-O3", "-arch=sm_89", "-std=c++17", SRC_FILE, "-o", BIN_FILE], check=True)

def run_argon_bench():
    os.makedirs("argon_results", exist_ok=True)
    
    # 1. Run NVE Production (after NVT equilibration built-in to C++)
    # Note: C++ code now does 5000 steps NVT then 'steps' of production.
    # To test NVE energy conservation, we run a short production in NVE mode (type 0).
    print("Running NVE Energy Conservation test...")
    cmd_nve = [f"./{BIN_FILE}", "0", "1", "argon_results/nve", str(SIGMA), str(EPSILON), str(RHO), str(T_TARGET), "5000", str(BOX_SIZE)]
    subprocess.run(cmd_nve, check=True)
    
    # 2. Run NVT Production for RDF
    print("Running NVT Structural test...")
    cmd_nvt = [f"./{BIN_FILE}", "3", "1", "argon_results/nvt", str(SIGMA), str(EPSILON), str(RHO), str(T_TARGET), "5000", str(BOX_SIZE)]
    subprocess.run(cmd_nvt, check=True)

def calculate_rdf_correct(state_csv, box_size, n_bins=150):
    df = pd.read_csv(state_csv)
    pos = df[['x', 'y', 'z']].values
    n = len(pos)
    
    # Full sampling to fix normalization: every sampled atom against ALL other atoms
    sample_n = min(n, 500) 
    hist = np.zeros(n_bins)
    dr = (box_size / 2.0) / n_bins
    r_bins = np.linspace(0, box_size / 2.0, n_bins + 1)
    
    for i in range(sample_n):
        diff = pos - pos[i] # All atoms
        # Periodic boundary
        diff -= box_size * np.round(diff / box_size)
        dist = np.linalg.norm(diff, axis=1)
        # Exclude self
        dist = dist[dist > 1e-6]
        counts, _ = np.histogram(dist, bins=r_bins)
        hist += counts

    rho = n / (box_size**3)
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2.0
    # Normalization: hist / (sample_n * rho * 4*pi*r^2*dr)
    shell_vol = 4.0 * np.pi * (r_centers**2) * dr
    gr = hist / (sample_n * rho * shell_vol)
    return r_centers, gr

def load_lammps_rdf(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            parts = line.split()
            if len(parts) == 2: continue
            data.append([float(p) for p in parts])
    data = np.array(data)
    # Return last 150 bins
    return data[-150:, 1], data[-150:, 2]

def plot_validation():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Energy Conservation (NVE)
    ax_e = axes[0]
    df = pd.read_csv("argon_results/nve_timeseries.csv")
    steps = df['Step']
    ax_e.plot(steps, df['KE'], 'g--', alpha=0.5, label='Kinetic (KE)')
    ax_e.plot(steps, df['PE'], 'r--', alpha=0.5, label='Potential (PE)')
    ax_e.plot(steps, df['TotalE'], 'b-', lw=2, label='Total Energy (KE+PE)')
    
    ax_e.set_title("NVE Energy Conservation (Liquid Argon)")
    ax_e.set_xlabel("Production Step")
    ax_e.set_ylabel("Energy")
    ax_e.grid(True, alpha=0.3)
    ax_e.legend()
    
    # 2. RDF Comparison
    ax_g = axes[1]
    r_gpu, gr_gpu = calculate_rdf_correct("argon_results/nvt_final_state.csv", BOX_SIZE)
    ax_g.plot(r_gpu, gr_gpu, 'r-', lw=2, label='BAMPS GPU (NHC)')
    
    try:
        r_ref, gr_ref = load_lammps_rdf("lammps_rdf.txt")
        ax_g.plot(r_ref, gr_ref, 'k--', alpha=0.8, label='LAMMPS (tail yes)')
    except: pass

    ax_g.set_title(r"Radial Distribution Function $g(r)$ 对比")
    ax_g.set_xlabel(r"$r / \sigma$")
    ax_g.set_ylabel(r"$g(r)$")
    ax_g.set_xlim(0, 4.0)
    ax_g.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax_g.grid(True, alpha=0.3)
    ax_g.legend()
    
    plt.tight_layout()
    plt.savefig("argon_results/argon_validation.png")

if __name__ == "__main__":
    compile_code()
    # Regenerate LAMMPS data with tail correction
    subprocess.run(["lmp", "-in", "in.lj"], capture_output=True)
    run_argon_bench()
    plot_validation()
