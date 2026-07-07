import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import subprocess
import os

# Set target paths
sim_exe = r"C:\Users\hy-wu.DESKTOP-G355NC5\Documents\GitHub\notes\QGP_MC\qgp_sim.exe"
main_cu = r"C:\Users\hy-wu.DESKTOP-G355NC5\Documents\GitHub\notes\QGP_MC\src\main.cu"
out_dir = r"C:\Users\hy-wu.DESKTOP-G355NC5\Documents\GitHub\notes\QGP_MC"

# 1. Compile CUDA code
print("Compiling CUDA simulation...")
try:
    compile_cmd = f"nvcc -O3 {main_cu} -o {sim_exe} -lcurand"
    subprocess.run(compile_cmd, shell=True, check=True)
    print("Compilation successful.")
except subprocess.CalledProcessError as e:
    print(f"Compilation failed: {e}")
    exit(1)

def safe_rename(src, dst):
    if os.path.exists(dst):
        os.remove(dst)
    os.rename(src, dst)

# 2. Run simulation with Lambda = 0.0 (Standard)
print("\nRunning simulation with Lambda = 0.0 (Standard)...")
subprocess.run(f"{sim_exe} 200 0.0 0.0025 0.25", shell=True, check=True, cwd=out_dir)
safe_rename(os.path.join(out_dir, "particles_final.csv"), os.path.join(out_dir, "particles_lambda_0.csv"))
safe_rename(os.path.join(out_dir, "grid_final.csv"), os.path.join(out_dir, "grid_lambda_0.csv"))

# 3. Run simulation with Lambda = 0.6 (Two-level non-local expansion)
print("\nRunning simulation with Lambda = 0.6 (Two-level)...")
subprocess.run(f"{sim_exe} 200 0.6 0.0025 0.25", shell=True, check=True, cwd=out_dir)
safe_rename(os.path.join(out_dir, "particles_final.csv"), os.path.join(out_dir, "particles_lambda_6.csv"))
safe_rename(os.path.join(out_dir, "grid_final.csv"), os.path.join(out_dir, "grid_lambda_6.csv"))

print("\nRunning validation diagnostics and generating figures...")

# Read results
p0 = pd.read_csv(os.path.join(out_dir, "particles_lambda_0.csv"))
p6 = pd.read_csv(os.path.join(out_dir, "particles_lambda_6.csv"))
g0 = pd.read_csv(os.path.join(out_dir, "grid_lambda_0.csv"))
g6 = pd.read_csv(os.path.join(out_dir, "grid_lambda_6.csv"))

# Create output figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ----------------- Subplot 1: Jüttner Distribution Fit (Thermalization) -----------------
ax1 = axes[0, 0]
mass = 0.5 # GeV
# Energy list
E = p6['E'].values

# Jüttner distribution formula: P(E) = C * E * sqrt(E^2 - m^2) * exp(-E/T)
def juttner_dist(E_val, T, scale):
    term = E_val**2 - mass**2
    term = np.clip(term, 0.0, None)
    return scale * E_val * np.sqrt(term) * np.exp(-E_val / T)

# Plot histogram of energies
counts, bins = np.histogram(E, bins=100, density=True)
bin_centers = (bins[:-1] + bins[1:]) / 2.0
ax1.hist(E, bins=100, density=True, alpha=0.6, color='royalblue', label='Simulated Energies')

# Fit Jüttner curve
try:
    T_guess = 0.3
    term = bin_centers**2 - mass**2
    term = np.clip(term, 0.0, None)
    f_unnorm = bin_centers * np.sqrt(term) * np.exp(-bin_centers / T_guess)
    scale_guess = np.max(counts) / np.max(f_unnorm) if np.max(f_unnorm) > 0 else 100.0
    popt, pcov = curve_fit(juttner_dist, bin_centers, counts, p0=[T_guess, scale_guess])
    fitted_T = popt[0]
    E_plot = np.linspace(mass + 1e-4, np.max(E), 200)
    ax1.plot(E_plot, juttner_dist(E_plot, *popt), 'r-', linewidth=2.5,
             label=f'Jüttner Fit (T={fitted_T:.3f} GeV)')
except Exception as e:
    print(f"Jüttner fitting failed: {e}")
    fitted_T = 0.3

ax1.set_title("1. Thermalization Check (Relativistic Jüttner Fit)", fontsize=12, fontweight='bold')
ax1.set_xlabel("Energy E (GeV)", fontsize=10)
ax1.set_ylabel("Probability Density", fontsize=10)
ax1.legend(fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.5)

# ----------------- Subplot 2: Phase Boundary / Density Field -----------------
ax2 = axes[0, 1]
# We plot the density field in the X-Y plane
grid_size = 12
density_2d = np.zeros((grid_size, grid_size))
for ix in range(grid_size):
    for iy in range(grid_size):
        # Average density over z
        sub_df = g6[(g6['ix'] == ix) & (g6['iy'] == iy)]
        density_2d[iy, ix] = sub_df['n'].mean()

im = ax2.imshow(density_2d, origin='lower', extent=[-6, 6, -6, 6], cmap='YlOrRd', interpolation='gaussian')
fig.colorbar(im, ax=ax2, label='Density n (fm^-3)')
ax2.set_title("2. QGP Phase Separation & Droplet Boundaries", fontsize=12, fontweight='bold')
ax2.set_xlabel("x (fm)", fontsize=10)
ax2.set_ylabel("y (fm)", fontsize=10)

# Calculate Elliptic Flow v2
# v2 = < (px^2 - py^2) / (px^2 + py^2) >
v2_0 = np.mean((p0['px']**2 - p0['py']**2) / (p0['px']**2 + p0['py']**2))
v2_6 = np.mean((p6['px']**2 - p6['py']**2) / (p6['px']**2 + p6['py']**2))
print(f"Measured Elliptic Flow v2 (Lambda=0.0): {v2_0:.4f}")
print(f"Measured Elliptic Flow v2 (Lambda=0.6): {v2_6:.4f}")

# ----------------- Subplot 3: Azimuthal Spin Polarization P_z -----------------
ax3 = axes[1, 0]
# Bin particles by momentum azimuthal angle phi_p = arctan2(py, px)
phi_bins = np.linspace(-np.pi, np.pi, 20)
phi_centers = (phi_bins[:-1] + phi_bins[1:]) / 2.0

def get_polarization_profile(df, component='pol_z'):
    # Momentum azimuthal angle
    phi_p = np.arctan2(df['py'], df['px'])
    pol = df[component].values
    avg_pol = []
    for i in range(len(phi_bins)-1):
        idx = (phi_p >= phi_bins[i]) & (phi_p < phi_bins[i+1])
        if np.sum(idx) > 10:
            avg_pol.append(np.mean(pol[idx]))
        else:
            avg_pol.append(0.0)
    return np.array(avg_pol)

pz_0 = get_polarization_profile(p0, 'pol_z')
pz_6 = get_polarization_profile(p6, 'pol_z')

# Apply periodic 1D Gaussian smoothing to filter out bin-by-bin statistical fluctuations
pz_0_smooth = gaussian_filter1d(pz_0, sigma=1.0, mode='wrap')
pz_6_smooth = gaussian_filter1d(pz_6, sigma=1.0, mode='wrap')

ax3.plot(phi_centers, pz_0_smooth, 'o--', color='crimson', label=r'$\lambda = 0.0$ (Standard)')
ax3.plot(phi_centers, pz_6_smooth, 's-', color='teal', label=r'$\lambda = 0.6$ (Two-level Non-local)')
ax3.set_title("3. Longitudinal Spin Polarization $P_z(\\phi_p)$", fontsize=12, fontweight='bold')
ax3.set_xlabel(r"Momentum Azimuthal Angle $\phi_p$", fontsize=10)
ax3.set_ylabel(r"Polarization $P_z$", fontsize=10)
ax3.legend(fontsize=10)
ax3.grid(True, linestyle='--', alpha=0.5)

# ----------------- Subplot 4: Azimuthal Spin Polarization P_y -----------------
ax4 = axes[1, 1]
py_0 = get_polarization_profile(p0, 'pol_y')
py_6 = get_polarization_profile(p6, 'pol_y')

# Apply periodic 1D Gaussian smoothing
py_0_smooth = gaussian_filter1d(py_0, sigma=1.0, mode='wrap')
py_6_smooth = gaussian_filter1d(py_6, sigma=1.0, mode='wrap')

ax4.plot(phi_centers, py_0_smooth, 'o--', color='crimson', label=r'$\lambda = 0.0$ (Standard)')
ax4.plot(phi_centers, py_6_smooth, 's-', color='teal', label=r'$\lambda = 0.6$ (Two-level Non-local)')
ax4.set_title("4. Transverse Spin Polarization $P_y(\\phi_p)$", fontsize=12, fontweight='bold')
ax4.set_xlabel(r"Momentum Azimuthal Angle $\phi_p$", fontsize=10)
ax4.set_ylabel(r"Polarization $P_y$", fontsize=10)
ax4.legend(fontsize=10)
ax4.grid(True, linestyle='--', alpha=0.5)

# Tight layout and save figure
plt.tight_layout()
fig_path = os.path.join(out_dir, "qgp_validation_results.png")
plt.savefig(fig_path, dpi=300)
print(f"Validation dashboard saved to: {fig_path}")

plt.close()
