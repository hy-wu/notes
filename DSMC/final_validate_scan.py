import os
import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt

# --- High Density Configuration ---
SIGMA_LIST = np.linspace(0.05, 0.25, 10) 
EPSILON_LIST = np.linspace(0.002, 0.04, 10)
EPS_FIXED = 0.01
SIG_FIXED = 0.15

N_TARGET = 16384 # High density (rho=8.0)
STEPS = 10000 
DT_VAL = 0.001 
TESTPARTCL_VAL = 1 
VOLUME = 1000.0

SRC_FILE = "code/bamps_gpu_ancient.cu"
BIN_FILE = "bamps_bin.exe"

def run_command(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, err = process.communicate()
    return process.returncode

def get_vdw_p(n, T, eps, sig):
    b = (2.0/3.0) * np.pi * (sig**3)
    a = (16.0/9.0) * np.pi * eps * (sig**3)
    return (n * T) / (1.0 - n * b + 1e-9) - a * (n**2)

def save_energy_spectrum(prefix, val):
    if not os.path.exists('energies.txt'): return
    try:
        energies = np.loadtxt('energies.txt')
        plt.figure(figsize=(8, 6))
        plt.hist(energies, bins=200, density=True, color='skyblue', edgecolor='black', alpha=0.7)
        
        # Fit Maxwell-Boltzmann-like curve if relevant
        T_eff = np.mean(energies) / 1.5 # Classical approximation
        x = np.linspace(0, np.max(energies), 100)
        # f(E) ~ sqrt(E) * exp(-E/T)
        y = 2.0 * np.sqrt(x/np.pi) * (1.0/T_eff**1.5) * np.exp(-x/T_eff)
        plt.plot(x, y, 'r-', lw=2, label=f'M-B Fit (T={T_eff:.3f})')

        plt.title(f'Energy Spectrum: {prefix}={val:.4f}')
        plt.xlabel('Energy [GeV]')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.savefig(f'parameter_scan_results/energy_spec_{prefix}_{val:.4f}.png')
        plt.close()
    except Exception as e:
        print(f"  [Error] Could not generate spectrum for {prefix}={val}: {e}")

def perform_fast_scan():
    os.makedirs('validation_results', exist_ok=True)
    os.makedirs('parameter_scan_results', exist_ok=True)
    
    # 1. COMPILE ONCE
    print("Compiling CUDA program (Once)...")
    compile_cmd = ["nvcc", "-O3", "-arch=sm_89", "-DMODE_RELATIVISTIC=0", "-DMODE_LJ=1", "-DENSKOG_ORDER=0", f"-DTESTPARTCL={TESTPARTCL_VAL}", SRC_FILE, "-o", BIN_FILE]
    if run_command(compile_cmd) != 0: 
        print("Compilation failed.")
        return

    # 2. SCAN SIGMA
    print("Scanning Sigma...")
    sig_data = []
    for sig in SIGMA_LIST:
        run_cmd = [f"./{BIN_FILE}", f"{sig:.4f}", f"{EPS_FIXED:.4f}", str(STEPS), f"{DT_VAL:.4f}", str(N_TARGET)]
        if run_command(run_cmd) != 0: continue
        
        # Archive Log and Spectrum
        shutil.copy('physics_log.txt', f'parameter_scan_results/physics_log_sig_{sig:.4f}.txt')
        save_energy_spectrum('sig', sig)
        
        log = np.genfromtxt('physics_log.txt', skip_header=1)
        if log.ndim == 1:
            t_f, p_w, p_v = log[2], log[3], log[6]
        else:
            t_f, p_w, p_v = log[-1, 2], log[-1, 3], log[-1, 6]
            
        p_th = get_vdw_p(N_TARGET/VOLUME, t_f, EPS_FIXED, sig)
        sig_data.append([sig, t_f, p_w, p_v, p_th])
        print(f"  SIG={sig:.3f} | P_wall={p_w:.4f} | P_virial={p_v:.4f}")

    # 3. SCAN EPSILON
    print("\nScanning Epsilon...")
    eps_data = []
    for eps in EPSILON_LIST:
        run_cmd = [f"./{BIN_FILE}", f"{SIG_FIXED:.4f}", f"{eps:.4f}", str(STEPS), f"{DT_VAL:.4f}", str(N_TARGET)]
        if run_command(run_cmd) != 0: continue
        
        # Archive Log and Spectrum
        shutil.copy('physics_log.txt', f'parameter_scan_results/physics_log_eps_{eps:.4f}.txt')
        save_energy_spectrum('eps', eps)
        
        log = np.genfromtxt('physics_log.txt', skip_header=1)
        if log.ndim == 1:
            t_f, p_w, p_v = log[2], log[3], log[6]
        else:
            t_f, p_w, p_v = log[-1, 2], log[-1, 3], log[-1, 6]
            
        p_th = get_vdw_p(N_TARGET/VOLUME, t_f, eps, SIG_FIXED)
        eps_data.append([eps, t_f, p_w, p_v, p_th])
        print(f"  EPS={eps:.4f} | P_wall={p_w:.4f} | P_virial={p_v:.4f}")

    sd = np.array(sig_data)
    ed = np.array(eps_data)

    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Sigma Scan Plot
    ax1.plot(sd[:, 0], sd[:, 2], 'mo-', alpha=0.6, label='Measured $P_{wall}$')
    ax1.plot(sd[:, 0], sd[:, 3], 'bo-', label='Measured $P_{virial}$')
    ax1.plot(sd[:, 0], sd[:, 4], 'r--', label='vdW Theory')
    ax1.set_title('Sigma Scan (Dual Pressure)')
    ax1.set_xlabel('$\sigma$')
    ax1.set_ylabel('Pressure')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Epsilon Scan Plot
    ax2.plot(ed[:, 0], ed[:, 2], 'mo-', alpha=0.6, label='Measured $P_{wall}$')
    ax2.plot(ed[:, 0], ed[:, 3], 'go-', label='Measured $P_{virial}$')
    ax2.plot(ed[:, 0], ed[:, 4], 'r--', label='vdW Theory')
    ax2.set_title('Epsilon Scan (Dual Pressure)')
    ax2.set_xlabel('$\epsilon$')
    ax2.set_ylabel('Pressure')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('validation_results/dual_pressure_scan.png')
    print("\nScan complete. Logs archived in parameter_scan_results/.")

if __name__ == "__main__":
    perform_fast_scan()
