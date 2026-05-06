import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import os

# Detect mode
is_relativistic = True
try:
    with open('bamps_master.log', 'r') as f:
        content = f.read()
        is_relativistic = content.rfind("MODE: RELATIVISTIC") > content.rfind("MODE: CLASSICAL")
except: pass

# 1. Load Data
try:
    diag = np.genfromtxt('physics_log.txt', skip_header=1)
    energies = np.loadtxt('energies.txt')
    positions = np.loadtxt('positions.txt')
except Exception as e:
    print(f"Loading failed: {e}"); exit()

# --- PLOT 1: Physics Validation (Temp, EOS, Spectrum) ---
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
mode_str = "Relativistic" if is_relativistic else "Classical"
fig.suptitle(f'Physics Analysis: {mode_str}', fontsize=18)

# A. Temperature
axes[0, 0].plot(diag[:, 1], diag[:, 2], 'r-', lw=2)
axes[0, 0].set_title('Temperature Evolution'); axes[0, 0].set_ylabel('T [GeV]'); axes[0, 0].grid(True)

# B. EOS Check
TESTPARTCL = 100; VOLUME = 1000.0
real_P = diag[:, 3] / TESTPARTCL
theory_P = (diag[:, 5] / TESTPARTCL / VOLUME) * diag[:, 2]
axes[0, 1].plot(diag[:, 1], real_P, 'b-', label='Measured P_wall')
axes[0, 1].plot(diag[:, 1], theory_P, 'k--', label='nT')
axes[0, 1].set_title('EOS Check'); axes[0, 1].legend(); axes[0, 1].grid(True)

# C. Energy Spectrum (Fine Bins)
num_bins = 200
counts, bins, _ = axes[1, 0].hist(energies, bins=num_bins, density=True, alpha=0.5, color='cyan', label='Data')
def log_boltz(E, T, logA, n): return logA + n * np.log(E + 1e-9) - E/T
def phys_dist(E, T, logA, n): return np.exp(logA) * (E**n) * np.exp(-E/T)
bin_centers = (bins[:-1] + bins[1:]) / 2
mask = (counts > 1e-5) & (bin_centers < np.percentile(energies, 99))
try:
    popt, _ = curve_fit(log_boltz, bin_centers[mask], np.log(counts[mask]), p0=[np.mean(energies)/2, 0, 2.0 if is_relativistic else 0.5])
    T_fit, logA_fit, n_fit = popt
    e_plot = np.linspace(0.001, np.max(energies), 300)
    axes[1, 0].plot(e_plot, phys_dist(e_plot, *popt), 'r--', lw=2, label=f'Fit (n={n_fit:.2f})')
except:
    T_fit, n_fit = 0, 0

axes[1, 0].set_title('Energy Spectrum (Fine)'); axes[1, 0].set_yscale('log'); axes[1, 0].set_ylim(1e-4, 10); axes[1, 0].legend()
axes[1, 0].grid(True, which='both', alpha=0.2)

# D. Metrics
axes[1, 1].text(0.1, 0.4, f"Final Kinetic T: {diag[-1, 2]:.4f}\nFit T:         {T_fit:.4f}\nIndex n:       {n_fit:.2f}\nEOS Consistency: {abs(real_P[-1]-theory_P[-1])/theory_P[-1]*100:.2f}%", 
                fontsize=14, family='monospace', bbox=dict(facecolor='white', alpha=0.8))
axes[1, 1].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('physics_validation.png')
plt.close()

# --- PLOT 2: Spatial Distribution (Large Scatter Plot) ---
plt.figure(figsize=(12, 12))
# Use very small markers and alpha for high density
plt.scatter(positions[:, 0], positions[:, 1], s=0.2, c=positions[:, 2], cmap='viridis', alpha=0.4)
plt.title(f'Particle Spatial Distribution (XY Projection, color=Z)\n{mode_str}', fontsize=16)
plt.xlabel('X [fm]'); plt.ylabel('Y [fm]')
plt.xlim(-5.5, 5.5); plt.ylim(-5.5, 5.5)
plt.grid(True, alpha=0.3)
plt.colorbar(label='Z position [fm]')
plt.savefig('spatial_distribution.png', dpi=150)
plt.close()

print("Plots saved: physics_validation.png, spatial_distribution.png")
