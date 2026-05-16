import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# --- Configuration ---
BIN_FILE = "bamps_compare.exe"
SRC_FILE = "src/thermostat_compare.cu"
STEPS = 50000 # Longer for better D accuracy
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
    # Using NHC (type 3) and PBC (type 1)
    print(f"Running long NVT-NHC simulation ({STEPS} steps)...")
    cmd = [f"./{BIN_FILE}", "3", "1", "diffusion_results/diff", str(SIGMA), str(EPSILON), str(RHO), str(T_TARGET), str(STEPS), str(BOX_SIZE)]
    subprocess.run(cmd, check=True)

def analyze_diffusion():
    df = pd.read_csv("diffusion_results/diff_timeseries.csv")
    
    # MSD = 6Dt  => D = slope / 6
    # Time = Step * DT
    time_vals = df['Step'] * DT
    msd_vals = df['MSD']
    
    # Use only the linear regime (skip initial ballistic 10%)
    start_idx = len(df) // 10
    linear_time = time_vals[start_idx:]
    linear_msd = msd_vals[start_idx:]
    
    slope, intercept, r_value, p_value, std_err = linregress(linear_time, linear_msd)
    D = slope / 6.0
    
    print(f"Extracted Diffusion Coefficient D* = {D:.5f}")
    print(f"R-squared: {r_value**2:.5f}")
    
    # Comparison with Literature
    # Liquid Argon at rho*=0.8, T*=1.0 typically has D* ~ 0.04-0.05
    D_ref = 0.045 
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_vals, msd_vals, 'b-', label='Simulation MSD')
    plt.plot(linear_time, slope * linear_time + intercept, 'r--', label=f'Linear Fit (D*={D:.4f})')
    
    plt.title(r"Mean Square Displacement (MSD) for Liquid Argon ($\rho^*=0.8, T^*=1.0$)")
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
