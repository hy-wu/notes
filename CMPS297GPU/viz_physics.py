import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import os

# --- Constants and Parameter Detection ---
TESTPARTCL = 100
VOLUME = 1000.0  # 10x10x10
EPS, SIG = 0.1, 0.8 # LJ Parameters

is_relativistic = False
mode_lj = False
dt = 0.01

try:
    with open('bamps_master.log', 'r') as f:
        lines = f.readlines()
        for line in reversed(lines):
            if "RUN MODE:" in line:
                is_relativistic = "REL" in line
                mode_lj = "LJ: 1" in line
                if "DT:" in line:
                    dt = float(line.split("DT:")[1].split(",")[0].strip())
                if "TESTPARTCL:" in line:
                    TESTPARTCL = int(line.split("TESTPARTCL:")[1].split(",")[0].strip())
                break
except: pass

print(f"Viz Config: {'REL' if is_relativistic else 'CLASS'}, LJ={mode_lj}, TP={TESTPARTCL}, DT={dt}")

# 1. Load Data
try:
    # Column indices: 0:Step, 1:Time, 2:Temp, 3:Pressure_Wall, 4:Energy, 5:Count, 6:Pressure_Virial
    diag = np.genfromtxt('physics_log.txt', skip_header=1)
    energies = np.loadtxt('energies.txt')
    positions = np.loadtxt('positions.txt')
except Exception as e:
    print(f"Loading failed: {e}"); exit()

# 2. Physics Quantities
time = diag[:, 1]
temp = diag[:, 2]
p_wall = diag[:, 3]
p_virial = diag[:, 6] if diag.shape[1] > 6 else np.zeros_like(p_wall)
n_total = diag[:, 5] / TESTPARTCL
n_dens = n_total / VOLUME

# 3. van der Waals Theory
# b = 2/3 * pi * sigma^3
# a = 16/9 * pi * epsilon * sigma^3
b_vdw = (2.0/3.0) * np.pi * (SIG**3)
a_vdw = (16.0/9.0) * np.pi * EPS * (SIG**3)
# P = (n*T)/(1 - n*b) - a*n^2
p_vdw = (n_dens * temp) / (1.0 - n_dens * b_vdw + 1e-9) - a_vdw * (n_dens**2)

# --- PLOTTING ---
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(2, 3)

# A. Temperature Evolution (Check Thermostat)
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(time, temp, 'r-', lw=2)
ax1.set_title('Temperature (Andersen Thermostat)'); ax1.set_ylabel('T [GeV]'); ax1.grid(True)

# B. Equation of State (EOS) Validation
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(time, p_wall, 'b-', label='Measured P_wall')
ax2.plot(time, p_virial, 'g:', label='Virial Pressure')
ax2.plot(time, n_dens * temp, 'k--', alpha=0.5, label='Ideal Gas (nT)')
if mode_lj:
    ax2.plot(time, p_vdw, 'r-', lw=1.5, label='vdW Theory')
    ax2.set_title('vdW EOS Validation')
else:
    ax2.set_title('Ideal/Enskog EOS Check')
ax2.legend(); ax2.grid(True)

# C. Spatial Distribution (Ultra-Fine Scatter)
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(positions[:, 0], positions[:, 1], s=0.01, c=positions[:, 2], cmap='viridis', alpha=0.2)
ax3.set_title('Spatial Distribution (XY Projection)'); ax3.set_xlim(-5.1, 5.1); ax3.set_ylim(-5.1, 5.1)

# D. Energy Spectrum & Boltzmann Fit
ax4 = fig.add_subplot(gs[1, 0])
counts, bins, _ = ax4.hist(energies, bins=200, density=True, alpha=0.5, color='cyan', label='Data')
def log_boltz(E, T, logA, n): return logA + n * np.log(E + 1e-9) - E/T
def phys_dist(E, T, logA, n): return np.exp(logA) * (E**n) * np.exp(-E/T)
bin_centers = (bins[:-1] + bins[1:]) / 2
mask = (counts > 1e-5) & (bin_centers < np.percentile(energies, 99))
try:
    popt, _ = curve_fit(log_boltz, bin_centers[mask], np.log(counts[mask]), p0=[np.mean(energies)/2, 0, 2.0 if is_relativistic else 0.5])
    ax4.plot(bin_centers, phys_dist(bin_centers, *popt), 'r--', lw=2, label=f'T_fit={popt[0]:.4f}')
except: pass
ax4.set_title('Energy Spectrum (Log)'); ax4.set_yscale('log'); ax4.set_ylim(1e-4, 10); ax4.legend(); ax4.grid(True, alpha=0.2)

# E. Low Energy Zoom
ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(energies, bins=200, range=(0, 1.5), density=True, alpha=0.5, color='cyan')
if 'popt' in locals(): ax5.plot(np.linspace(0, 1.5, 100), phys_dist(np.linspace(0, 1.5, 100), *popt), 'r--', lw=2)
ax5.set_title('Low Energy Region Detail')

# F. Summary Stats
ax6 = fig.add_subplot(gs[1, 2])
vdw_err = abs(p_wall[-1]-p_vdw[-1])/p_vdw[-1]*100 if mode_lj else 0.0
ax6.text(0.1, 0.3, f"Mode: {'REL' if is_relativistic else 'CLASS'}\n"
                  f"LJ Enabled: {bool(mode_lj)}\n"
                  f"Final P_wall:   {p_wall[-1]:.4f}\n"
                  f"Final P_virial: {p_virial[-1]:.4f}\n"
                  f"Final P_vdW:    {p_vdw[-1]:.4f}\n"
                  f"vdW Consistency Err: {vdw_err:.2f}%", 
         fontsize=14, family='monospace', bbox=dict(facecolor='white', alpha=0.8))
ax6.axis('off')

plt.tight_layout()
plt.savefig('physics_validation.png')
plt.figure(figsize=(15, 15)); plt.scatter(positions[:, 0], positions[:, 1], s=0.01, c=positions[:, 2], cmap='viridis', alpha=0.3)
plt.savefig('spatial_distribution.png', dpi=200); plt.close('all')
print("Refined vdW Validation Plots Generated.")
