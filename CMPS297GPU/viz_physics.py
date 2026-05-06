import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import os

# Detect parameters from bamps_master.log
is_relativistic = True
testpartcl = 100
try:
    with open('bamps_master.log', 'r') as f:
        lines = f.readlines()
        for line in reversed(lines):
            if "MODE: RELATIVISTIC" in line: is_relativistic = True
            if "MODE: CLASSICAL" in line: is_relativistic = False
            if "TESTPARTCL:" in line:
                # Extract TESTPARTCL: XX
                parts = line.split("TESTPARTCL:")
                if len(parts) > 1:
                    testpartcl = int(parts[1].split()[0].strip())
                break
except Exception as e:
    print(f"Log parsing failed, using defaults: {e}")

print(f"Detected Settings: {'RELATIVISTIC' if is_relativistic else 'CLASSICAL'}, TESTPARTCL={testpartcl}")

# 1. Load Data
try:
    diag = np.genfromtxt('physics_log.txt', skip_header=1)
    energies = np.loadtxt('energies.txt')
    positions = np.loadtxt('positions.txt')
except Exception as e:
    print(f"Loading failed: {e}"); exit()

# --- PLOT 1: Physics Validation ---
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
mode_str = "Relativistic" if is_relativistic else "Classical"
fig.suptitle(f'Physics Analysis: {mode_str} (testpartcl={testpartcl})', fontsize=18)

# A. Temperature
axes[0, 0].plot(diag[:, 1], diag[:, 2], 'r-', lw=2)
axes[0, 0].set_title('Temperature Evolution'); axes[0, 0].set_ylabel('T [GeV]'); axes[0, 0].grid(True)

# B. EOS Check (Scaled by actual testpartcl)
VOLUME = 1000.0
real_P = diag[:, 3] / testpartcl
theory_P = (diag[:, 5] / testpartcl / VOLUME) * diag[:, 2]
axes[0, 1].plot(diag[:, 1], real_P, 'b-', label='Measured P_wall')
axes[0, 1].plot(diag[:, 1], theory_P, 'k--', label='nT')
axes[0, 1].set_title('Equation of State (EOS)'); axes[0, 1].legend(); axes[0, 1].grid(True)

# C. Energy Spectrum
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
    axes[1, 0].plot(e_plot, phys_dist(e_plot, *popt), 'r--', lw=2, label=f'Fit (n={n_fit:.2f})\nT_fit={T_fit:.4f}')
except: T_fit, n_fit = 0, 0
axes[1, 0].set_title('Energy Spectrum (Fine)'); axes[1, 0].set_yscale('log'); axes[1, 0].set_ylim(1e-4, 10); axes[1, 0].legend(); axes[1, 0].grid(True, which='both', alpha=0.2)

# D. Metrics
axes[1, 1].text(0.1, 0.4, f"Final Kinetic T: {diag[-1, 2]:.4f} GeV\nFit T:         {T_fit:.4f} GeV\nIndex n:       {n_fit:.2f}\nEOS Consistency: {abs(real_P[-1]-theory_P[-1])/theory_P[-1]*100:.2f}%", 
                fontsize=14, family='monospace', bbox=dict(facecolor='white', alpha=0.8))
axes[1, 1].axis('off')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('physics_validation.png')
plt.close()

# --- PLOT 2: Spatial Distribution (Large Scatter Plot) ---
plt.figure(figsize=(15, 15)) # Ultra large as requested
plt.scatter(positions[:, 0], positions[:, 1], s=0.05, c=positions[:, 2], cmap='viridis', alpha=0.3)
plt.title(f'Microscopic Spatial Distribution\n{mode_str} | XY Projection', fontsize=20)
plt.xlabel('X [fm]', fontsize=14); plt.ylabel('Y [fm]', fontsize=14)
plt.xlim(-5.1, 5.1); plt.ylim(-5.1, 5.1)
plt.grid(True, alpha=0.2)
cb = plt.colorbar(); cb.set_label('Z position [fm]', fontsize=12)
plt.savefig('spatial_distribution.png', dpi=200)
plt.close()

print("Plots generated with dynamic parameters.")
