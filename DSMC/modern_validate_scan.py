import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import re

# --- Configuration ---
SIGMA_LIST = np.linspace(0.05, 0.45, 21)
EPSILON_LIST = np.linspace(0.0, 1.0, 21)

FIXED_RHO = 6.0
FIXED_T = 1.0
FIXED_SIG = 0.1
FIXED_EPS = 0.4
STEPS = 20000

BIN_FILE = "bamps_modern.exe"

def run_sim(sig, eps, rho, T, steps=1000):
    cmd = [f"./{BIN_FILE}", f"{sig:.4f}", f"{eps:.4f}", f"{rho:.4f}", f"{T:.4f}", str(steps)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Parse all measured quantities
        p_virial = re.search(r"FINAL_P_VIRIAL: ([\d.e+-]+)", result.stdout)
        p_wall = re.search(r"FINAL_P_WALL: ([\d.e+-]+)", result.stdout)
        t_meas = re.search(r"FINAL_TEMPERATURE: ([\d.e+-]+)", result.stdout)
        
        if p_virial and p_wall and t_meas:
            return float(p_virial.group(1)), float(p_wall.group(1)), float(t_meas.group(1))
    except Exception as e:
        print(f"Error running sim: {e}")
    return None, None, None

def get_theory_p(rho, T, sig, eps):
    eta = (np.pi / 6.0) * rho * (sig**3)
    if eta >= 1.0: return np.nan
    Z_cs = (1 + eta + eta**2 - eta**3) / (1 - eta)**3
    p_rep = rho * T * Z_cs
    a = (16.0 / 9.0) * np.pi * eps * (sig**3)
    p_att = -a * (rho**2)
    return p_rep + p_att

def perform_scans():
    os.makedirs('validation_results', exist_ok=True)
    
    print("Compiling Modern Engine (Reflective Wall Dual-Measurement Mode)...")
    subprocess.run(["nvcc", "-O3", "-arch=sm_89", "src/main.cu", "-o", BIN_FILE], check=True)

    # 1. Sigma Scan
    print(f"Scanning Sigma (rho={FIXED_RHO}, T={FIXED_T}, eps={FIXED_EPS})...")
    sig_data = []
    for sig in SIGMA_LIST:
        p_vir, p_wall, t_meas = run_sim(sig, FIXED_EPS, FIXED_RHO, FIXED_T, STEPS)
        p_theory = get_theory_p(FIXED_RHO, FIXED_T, sig, FIXED_EPS)
        if p_vir is not None:
            sig_data.append([sig, p_vir, p_wall, t_meas, p_theory])
            print(f"  SIG={sig:.3f} | P_vir={p_vir:.4f} | P_wall={p_wall:.4f} | T_meas={t_meas:.4f}")
    
    sig_data = np.array(sig_data)

    # 2. Epsilon Scan
    print(f"\nScanning Epsilon (rho={FIXED_RHO}, T={FIXED_T}, sig={FIXED_SIG})...")
    eps_data = []
    for eps in EPSILON_LIST:
        p_vir, p_wall, t_meas = run_sim(FIXED_SIG, eps, FIXED_RHO, FIXED_T, STEPS)
        p_theory = get_theory_p(FIXED_RHO, FIXED_T, FIXED_SIG, eps)
        if p_vir is not None:
            eps_data.append([eps, p_vir, p_wall, t_meas, p_theory])
            print(f"  EPS={eps:.3f} | P_vir={p_vir:.4f} | P_wall={p_wall:.4f} | T_meas={t_meas:.4f}")
            
    eps_data = np.array(eps_data)

    # --- Plotting (The "No Laziness" 2x3 Dashboard) ---
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Row 0: Sigma Scan
    axes[0, 0].plot(sig_data[:, 0], sig_data[:, 1], 'bo-', label='Measured P_virial')
    axes[0, 0].plot(sig_data[:, 0], sig_data[:, 2], 'ms--', alpha=0.6, label='Measured P_wall')
    axes[0, 0].plot(sig_data[:, 0], sig_data[:, 4], 'r:', label='Theory (CS+Vlasov)')
    axes[0, 0].set_title('Sigma Scan: Pressure Comparison')
    axes[0, 0].set_xlabel('$\sigma$')
    axes[0, 0].legend()

    axes[0, 1].plot(sig_data[:, 0], sig_data[:, 3], 'ro-', label='Measured T')
    axes[0, 1].axhline(FIXED_T, color='gray', linestyle='--', label='Target T')
    axes[0, 1].set_title('Sigma Scan: Temperature Stability')
    axes[0, 1].set_xlabel('$\sigma$')
    axes[0, 1].legend()

    axes[0, 2].plot(sig_data[:, 0], sig_data[:, 1]/sig_data[:, 3], 'go-', label='P_virial / T')
    axes[0, 2].plot(sig_data[:, 0], sig_data[:, 2]/sig_data[:, 3], 'm^--', alpha=0.6, label='P_wall / T')
    axes[0, 2].plot(sig_data[:, 0], sig_data[:, 4]/FIXED_T, 'r:', label='Theory Z')
    axes[0, 2].set_title('Sigma Scan: Compressibility Indication')
    axes[0, 2].set_xlabel('$\sigma$')
    axes[0, 2].legend()

    # Row 1: Epsilon Scan
    axes[1, 0].plot(eps_data[:, 0], eps_data[:, 1], 'bo-', label='Measured P_virial')
    axes[1, 0].plot(eps_data[:, 0], eps_data[:, 2], 'ms--', alpha=0.6, label='Measured P_wall')
    axes[1, 0].plot(eps_data[:, 0], eps_data[:, 4], 'r:', label='Theory (CS+Vlasov)')
    axes[1, 0].set_title('Epsilon Scan: Pressure Comparison')
    axes[1, 0].set_xlabel('$\epsilon$')
    axes[1, 0].legend()

    axes[1, 1].plot(eps_data[:, 0], eps_data[:, 3], 'ro-', label='Measured T')
    axes[1, 1].axhline(FIXED_T, color='gray', linestyle='--', label='Target T')
    axes[1, 1].set_title('Epsilon Scan: Temperature Stability')
    axes[1, 1].set_xlabel('$\epsilon$')
    axes[1, 1].legend()

    axes[1, 2].plot(eps_data[:, 0], eps_data[:, 1]/eps_data[:, 3], 'go-', label='P_virial / T')
    axes[1, 2].plot(eps_data[:, 0], eps_data[:, 2]/eps_data[:, 3], 'm^--', alpha=0.6, label='P_wall / T')
    axes[1, 2].plot(eps_data[:, 0], eps_data[:, 4]/FIXED_T, 'r:', label='Theory Z')
    axes[1, 2].set_title('Epsilon Scan: Compressibility Indication')
    axes[1, 2].set_xlabel('$\epsilon$')
    axes[1, 2].legend()

    plt.tight_layout()
    plt.savefig('validation_results/modern_parameter_scan_dual.png')
    print("\nScan complete. Results saved to validation_results/modern_parameter_scan_dual.png")

if __name__ == "__main__":
    perform_scans()
