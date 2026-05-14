import os
import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt

# --- User-Specified Alignment Configuration ---
# Match rho=6.0, T=1.2, sigma=0.1
# EPSILON_LIST = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
EPSILON_LIST = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
SIGMA_FIXED = 0.2
T_TARGET = 1.0
RHO_TARGET = 6.0

VOLUME = 1000.0 # L=10
N_PARTICLES = int(RHO_TARGET * VOLUME) # 6000 particles

STEPS = 50000
DT_VAL = 0.002 
TESTPARTCL_VAL = 1 

SRC_FILE = "code/bamps_gpu_ancient.cu"
BIN_FILE = "bamps_bin.exe"

def run_command(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, err = process.communicate()
    return process.returncode

def get_vdw_p(n, T, eps, sig):
    b = (2.0/3.0) * np.pi * (sig**3)
    a = (16.0/9.0) * np.pi * eps * (sig**3)
    # vdW: P = nT / (1-nb) - an^2
    return (n * T) / (1.0 - n * b + 1e-9) - a * (n**2)

def perform_lammps_aligned_scan():
    os.makedirs('validation_results', exist_ok=True)
    
    # 1. COMPILE
    print(f"Compiling for state point rho={RHO_TARGET}, T={T_TARGET}...")
    compile_cmd = ["nvcc", "-O3", "-arch=sm_89", "-DMODE_RELATIVISTIC=0", "-DMODE_LJ=1", "-DENSKOG_ORDER=0", f"-DTESTPARTCL={TESTPARTCL_VAL}", SRC_FILE, "-o", BIN_FILE]
    if run_command(compile_cmd) != 0: 
        print("Compilation failed.")
        return

    # 2. SCAN EPSILON
    print("Scanning Epsilon (Aligned with LAMMPS)...")
    results = []
    for eps in EPSILON_LIST:
        # Args: sig, eps, steps, dt, N, target_T
        run_cmd = [f"./{BIN_FILE}", f"{SIGMA_FIXED:.4f}", f"{eps:.4f}", str(STEPS), f"{DT_VAL:.4f}", str(N_PARTICLES), f"{T_TARGET:.4f}"]
        if run_command(run_cmd) != 0: continue
        
        log = np.genfromtxt('physics_log.txt', skip_header=1)
        if log.ndim == 1:
            t_f, p_v = log[2], log[6]
        else:
            # Average last 20% of the run for stability
            last_idx = int(len(log) * 0.8)
            t_f = np.mean(log[last_idx:, 2])
            p_v = np.mean(log[last_idx:, 6])
            
        p_th = get_vdw_p(N_PARTICLES/VOLUME, t_f, eps, SIGMA_FIXED)
        results.append([eps, t_f, p_v, p_th])
        print(f"  EPS={eps:.2f} | T={t_f:.3f} | P_virial={p_v:.4f} | P_vdw={p_th:.4f}")

    data = np.array(results)
    
    # --- PLOTTING ---
    plt.figure(figsize=(10, 7))
    plt.plot(data[:, 0], data[:, 2] / data[:, 1], 'bo-', label='only LJ')
    plt.plot(data[:, 0], data[:, 3] / data[:, 1], 'r--', label='vdW Theory')
    # plt.scatter(data[:, 0], data[:, 2] / data[:, 3], color='green', label='CUDA/vdW Ratio')
    
    # Ideal gas baseline
    plt.axhline(RHO_TARGET * T_TARGET, color='gray', linestyle=':', label='Ideal Gas (rho*T)')
    
    plt.title(f'Pressure vs Epsilon (rho={RHO_TARGET}, T={T_TARGET})\nCUDA vs vdW Theory')
    plt.xlabel('Epsilon')
    plt.ylabel('Pressure / Temperature')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('validation_results/lammps_aligned_verification.png')
    
    # Save CSV for numerical comparison
    np.savetxt('validation_results/lammps_aligned_data.csv', data, delimiter=',', header="Epsilon,Temp,P_measured,P_vdw", comments='')
    print("\nVerification complete. Result saved to validation_results/lammps_aligned_verification.png")

if __name__ == "__main__":
    perform_lammps_aligned_scan()
