import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt

# --- Fast Configuration ---
EPSILON_LIST = np.linspace(0.005, 0.04, 5)
SIG_FIXED = 0.15

N_TARGET = 10000 
STEPS = 5000 
DT_VAL = 0.01 
TESTPARTCL_VAL = 20 # High enough to test scaling fix
VOLUME = 1000.0

SRC_FILE = "code/bamps_gpu_ancient.cu"
BIN_FILE = "bamps_bin.exe"

def run_command(cmd):
    # Using a list for subprocess to avoid shell parsing issues on Windows
    print(f"Running: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, err = process.communicate()
    if process.returncode != 0:
        print(out.decode())
    return process.returncode

def perform_fast_scan():
    os.makedirs('fast_results', exist_ok=True)
    print(f"Starting Fast Validation Scan (N={N_TARGET})...")
    
    eps_data = []
    for eps in EPSILON_LIST:
        # Construct cmd as a list for subprocess.Popen
        cmd = [
            "nvcc", "-O3", "-arch=sm_89",
            "-DMODE_RELATIVISTIC=0", "-DMODE_LJ=1", "-DENSKOG_ORDER=0",
            f"-DTOTAL_STEPS={STEPS}", f"-DTESTPARTCL={TESTPARTCL_VAL}", f"-DDELTA_T={DT_VAL}",
            f"-DSIGMA_OVERRIDE={SIG_FIXED:.4f}", f"-DEPSILON_OVERRIDE={eps:.4f}",
            f"-DN_PARTICLES_OVERRIDE={N_TARGET}", SRC_FILE, "-o", BIN_FILE
        ]
        
        if run_command(cmd) != 0: 
            print(f"Failed to compile for EPS={eps}")
            continue
        if run_command([f"./{BIN_FILE}"]) != 0: 
            print(f"Failed to run for EPS={eps}")
            continue
        
        if os.path.exists('physics_log.txt'):
            log = np.genfromtxt('physics_log.txt', skip_header=1)
            if log.ndim == 1: # Only one record
                t_f, p_v = log[2], log[6]
            else:
                t_f, p_v = log[-1, 2], log[-1, 6]
            eps_data.append([eps, t_f, p_v])
            print(f"  EPS={eps:.4f} | P_meas={p_v:.4f}")
        else:
            print("physics_log.txt not found")

    if not eps_data:
        print("No data collected.")
        return

    ed = np.array(eps_data)

    # --- PLOTTING ---
    plt.figure(figsize=(8, 6))
    plt.plot(ed[:, 0], ed[:, 2], 'go-', label='Measured $P_{virial}$ (Fixed Scaling)')
    plt.title('Fast Validation: Pressure vs. $\\epsilon$\n(1/N^2 Scaling & Collisional Mapping)')
    plt.xlabel('$\\epsilon$ [GeV]'); plt.ylabel('Pressure [GeV/fm$^3$]')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig('fast_results/epsilon_trend_check.png')
    print("\nValidation complete. Plot saved: fast_results/epsilon_trend_check.png")

if __name__ == "__main__":
    perform_fast_scan()
