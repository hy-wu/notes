import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# --- Configuration ---
BIN_FILE = "bamps_compare.exe"
SRC_FILE = "src/thermostat_compare.cu"
STEPS = 50000 
BOX_SIZE = 12.0
SIGMA = 1.0
EPSILON = 1.0
RHO = 0.8
T_TARGET = 1.0
DT = 0.002

def compile_code():
    print("Compiling MSD-enabled harness...")
    subprocess.run(["nvcc", "-O3", "-arch=sm_89", "-std=c++17", SRC_FILE, "-o", BIN_FILE], check=True)

def run_diffusion_bench():
    os.makedirs("diffusion_results", exist_ok=True)
    # Workflow: 5000 steps NVT (built-in) -> Reset -> STEPS NVE (type 0)
    print(f"Running rigorous NVE production for Diffusion ({STEPS} steps)...")
    cmd = [f"./{BIN_FILE}", "0", "1", "diffusion_results/diff", str(SIGMA), str(EPSILON), str(RHO), str(T_TARGET), str(STEPS), str(BOX_SIZE)]
    subprocess.run(cmd, check=True)

def analyze_diffusion():
    df = pd.read_csv("diffusion_results/diff_timeseries.csv")
    
    # Time = Step * DT
    time_vals = df['Step'] * DT
    msd_vals = df['MSD']
    
    # Use only the late linear regime (skip initial 20% to ensure decorrelation)
    start_idx = len(df) // 5
    linear_time = time_vals[start_idx:]
    linear_msd = msd_vals[start_idx:]
    
    slope, intercept, r_value, p_value, std_err = linregress(linear_time, linear_msd)
    D = slope / 6.0
    
    print(f"Rigorous Diffusion Coefficient D* = {D:.5f}")
    print(f"R-squared: {r_value**2:.5f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_vals, msd_vals, 'b-', alpha=0.6, label='NVE Production MSD')
    plt.plot(linear_time, slope * linear_time + intercept, 'r--', lw=2, label=f'Linear Fit (D*={D:.4f}, R2={r_value**2:.4f})')
    
    plt.title(r"Rigorous MSD (NVE Production after NVT Equil.)")
    plt.xlabel(r"Time ($\tau$)")
    plt.ylabel(r"MSD ($\sigma^2$)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("diffusion_results/msd_plot.png")
    
    return D

if __name__ == "__main__":
    compile_code()
    run_diffusion_bench()
    analyze_diffusion()
