import os
import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt

# --- High-Resolution Configuration ---
SIGMA_LIST = np.linspace(0.05, 0.25, 40) # 40 points for Sigma
EPSILON_LIST = np.linspace(0.002, 0.04, 40) # 40 points for Epsilon
EPS_FIXED = 0.01
SIG_FIXED = 0.15

N_TARGET = 30000 
STEPS = 30000
DT_VAL = 0.002 # Smaller DT for LJ precision
TESTPARTCL_VAL = 1
VOLUME = 1000.0

SRC_FILE = "code/bamps_gpu_ancient.cu"
BIN_FILE = "./bamps_bin"

def run_command(cmd):
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while True:
        output = process.stdout.readline()
        if output == b'' and process.poll() is not None:
            break
        if output:
            line = output.decode().strip()
            if "Step" in line or "Performance" in line or "MODE:" in line:
                print(f"  [CUDA] {line}")
    return process.returncode

def get_vdw_p(n, T, eps, sig):
    b = (2.0/3.0) * np.pi * (sig**3)
    a = (16.0/9.0) * np.pi * eps * (sig**3)
    return (n * T) / (1.0 - n * b + 1e-9) - a * (n**2)

def perform_scan():
    print(f"Starting High-Precision Scan (N={N_TARGET}, 40 points total)...")
    
    # --- 1. SCAN SIGMA ---
    print("Scanning Sigma...")
    sig_data = []
    for sig in SIGMA_LIST:
        cmd = (f"nvcc -O3 -DMODE_RELATIVISTIC=0 -DMODE_LJ=1 -DENSKOG_ORDER=0 "
               f"-DTOTAL_STEPS={STEPS} -DTESTPARTCL={TESTPARTCL_VAL} -DDELTA_T={DT_VAL} "
               f"-DSIGMA_OVERRIDE={sig:.4f} -DEPSILON_OVERRIDE={EPS_FIXED:.4f} "
               f"-DN_PARTICLES_OVERRIDE={N_TARGET} {SRC_FILE} -o {BIN_FILE}")
        if run_command(cmd) != 0: continue
        if run_command(BIN_FILE) != 0: continue
        
        shutil.copy('physics_log.txt', f'parameter_scan_results/physics_log_sig_{sig:.4f}.txt')
        log = np.genfromtxt('physics_log.txt', skip_header=1)
        t_f, p_v = log[-1, 2], log[-1, 6]
        p_th = get_vdw_p(N_TARGET/VOLUME, t_f, EPS_FIXED, sig)
        sig_data.append([sig, EPS_FIXED, t_f, p_v, p_th])
        print(f"  SIG={sig:.3f} | EPS={EPS_FIXED:.4f} | P_meas={p_v:.4f} | P_th={p_th:.4f}", end='\r')

    # --- 2. SCAN EPSILON ---
    print("\nScanning Epsilon...")
    eps_data = []
    for eps in EPSILON_LIST:
        cmd = (f"nvcc -O3 -DMODE_RELATIVISTIC=0 -DMODE_LJ=1 -DENSKOG_ORDER=0 "
               f"-DTOTAL_STEPS={STEPS} -DTESTPARTCL={TESTPARTCL_VAL} -DDELTA_T={DT_VAL} "
               f"-DSIGMA_OVERRIDE={SIG_FIXED:.4f} -DEPSILON_OVERRIDE={eps:.4f} "
               f"-DN_PARTICLES_OVERRIDE={N_TARGET} {SRC_FILE} -o {BIN_FILE}")
        if run_command(cmd) != 0: continue
        if run_command(BIN_FILE) != 0: continue
        
        shutil.copy('physics_log.txt', f'parameter_scan_results/physics_log_eps_{eps:.4f}.txt')
        log = np.genfromtxt('physics_log.txt', skip_header=1)
        t_f, p_v = log[-1, 2], log[-1, 6]
        p_th = get_vdw_p(N_TARGET/VOLUME, t_f, eps, SIG_FIXED)
        eps_data.append([eps, SIG_FIXED, t_f, p_v, p_th])
        print(f"  EPS={eps:.4f} | SIG={SIG_FIXED:.4f} | P_meas={p_v:.4f} | P_th={p_th:.4f}", end='\r')

    sd = np.array(sig_data)
    ed = np.array(eps_data)

    # --- SAVE CSVs ---
    np.savetxt('scan_sigma.csv', sd, delimiter=',', header="Sigma,Epsilon,Temp,P_measured,P_theory", comments='')
    np.savetxt('scan_epsilon.csv', ed, delimiter=',', header="Epsilon,Sigma,Temp,P_measured,P_theory", comments='')

    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot 1: Sigma Scan
    ax1.scatter(sd[:, 0], sd[:, 2], color='blue', s=40, label='Measured $P_{virial}$')
    ax1.plot(sd[:, 0], sd[:, 3], 'r--', lw=1.5, label='vdW Theory (Point-wise)')
    ax1.set_title(f'Pressure vs. $\\sigma$\n(Fixed $\\epsilon$={EPS_FIXED}, N={N_TARGET})')
    ax1.set_xlabel('$\\sigma$ [fm]'); ax1.set_ylabel('Pressure [GeV/fm$^3$]')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # Plot 2: Epsilon Scan
    ax2.scatter(ed[:, 0], ed[:, 2], color='green', s=40, label='Measured $P_{virial}$')
    ax2.plot(ed[:, 0], ed[:, 3], 'r--', lw=1.5, label='vdW Theory (Point-wise)')
    ax2.set_title(f'Pressure vs. $\\epsilon$\n(Fixed $\\sigma$={SIG_FIXED}, N={N_TARGET})')
    ax2.set_xlabel('$\\epsilon$ [GeV]'); ax2.set_ylabel('Pressure [GeV/fm$^3$]')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('vdW_final_synchronized_scan.png', dpi=200)
    print("\nScan complete. Files saved: scan_sigma.csv, scan_epsilon.csv, vdW_final_synchronized_scan.png")

if __name__ == "__main__":
    perform_scan()
