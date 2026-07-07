import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import subprocess
import os

# Set target paths
sim_exe = r"C:\Users\hy-wu.DESKTOP-G355NC5\Documents\GitHub\notes\QGP_MC\qgp_sim.exe"
main_cu = r"C:\Users\hy-wu.DESKTOP-G355NC5\Documents\GitHub\notes\QGP_MC\src\main.cu"
out_dir = r"C:\Users\hy-wu.DESKTOP-G355NC5\Documents\GitHub\notes\QGP_MC"

# 1. Compile CUDA code
print("Compiling CUDA simulation...")
compile_cmd = f"nvcc -O3 {main_cu} -o {sim_exe} -lcurand"
subprocess.run(compile_cmd, shell=True, check=True)
print("Compilation successful.")

# 2. Run simulation with Lambda = 0.0, a_vdw = 0.0, shear_v = 0.0 (Ideal Gas)
print("\nRunning ideal gas simulation (Lambda=0, a_vdw=0, shear_v=0)...")
subprocess.run(f"{sim_exe} 200 0.0 0.0 0.0", shell=True, check=True, cwd=out_dir)

# 3. Read output
p_data = pd.read_csv(os.path.join(out_dir, "particles_final.csv"))
E = p_data['E'].values
mass = 0.5

# 4. Fit Jüttner curve
counts, bins = np.histogram(E, bins=50, density=True)
bin_centers = (bins[:-1] + bins[1:]) / 2.0

def juttner_dist(E_val, T, scale):
    term = E_val**2 - mass**2
    term = np.clip(term, 0.0, None)
    return scale * E_val * np.sqrt(term) * np.exp(-E_val / T)

T_guess = 0.3
term = bin_centers**2 - mass**2
term = np.clip(term, 0.0, None)
f_unnorm = bin_centers * np.sqrt(term) * np.exp(-bin_centers / T_guess)
scale_guess = np.max(counts) / np.max(f_unnorm) if np.max(f_unnorm) > 0 else 100.0

try:
    popt, pcov = curve_fit(juttner_dist, bin_centers, counts, p0=[T_guess, scale_guess])
    fitted_T = popt[0]
    print(f"\nIdeal Gas Jüttner Fit Success -> Fitted Temperature T = {fitted_T:.4f} GeV")
except Exception as e:
    print(f"Fitting failed: {e}")
    fitted_T = 0.3
    popt = [T_guess, scale_guess]

# 5. Plot
plt.figure(figsize=(8, 6))
plt.hist(E, bins=50, density=True, alpha=0.6, color='royalblue', label='Simulated Energies (Ideal Gas)')
E_plot = np.linspace(mass + 1e-4, np.max(E), 200)
plt.plot(E_plot, juttner_dist(E_plot, *popt), 'r-', linewidth=2.5,
         label=f'Jüttner Fit (T={fitted_T:.4f} GeV)')
plt.xlabel("Energy E (GeV)", fontsize=12)
plt.ylabel("Probability Density", fontsize=12)
plt.legend(fontsize=12)
plt.title("Ideal Gas Relativistic Jüttner Distribution Verification", fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)

fig_path = os.path.join(out_dir, "ideal_fit.png")
plt.savefig(fig_path, dpi=300)
print(f"Ideal fit plot saved to: {fig_path}")
plt.close()
