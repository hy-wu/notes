import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import sys

# Detect mode from bamps_master.log last entry
is_relativistic = True
try:
    with open('bamps_master.log', 'r') as f:
        lines = f.readlines()
        for line in reversed(lines):
            if "MODE: RELATIVISTIC" in line:
                is_relativistic = True
                break
            if "MODE: CLASSICAL" in line:
                is_relativistic = False
                break
except:
    pass

print(f"Detecting Simulation Mode: {'RELATIVISTIC' if is_relativistic else 'CLASSICAL'}")

# 1. Load Diagnostics
try:
    diag = np.genfromtxt('physics_log.txt', skip_header=1)
    time = diag[:, 1]
    temp = diag[:, 2]
    pressure_wall = diag[:, 3]
except Exception as e:
    print(f"Error loading physics_log.txt: {e}")
    exit()

# 2. Load Energies
try:
    energies = np.loadtxt('energies.txt')
except Exception as e:
    print(f"Error loading energies.txt: {e}")
    exit()

# Setup Figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
mode_str = "Relativistic (BAMPS)" if is_relativistic else "Classical (Newtonian)"
fig.suptitle(f'Ancient-Hybrid Simulation: {mode_str}', fontsize=16)

# A. Temperature Evolution
axes[0, 0].plot(time, temp, 'r-', label='Measured T')
axes[0, 0].set_title('Temperature vs Time')
axes[0, 0].set_ylabel('T [GeV]')
axes[0, 0].grid(True)

# B. EOS Check: P_wall vs Theory
# Relativistic: P = nT
# Classical: P = nT (Note: in both cases P=nT holds for ideal gas, 
# but T is defined differently: E/3N vs E/1.5N)
V = 1000.0
N = 100000
nT = (N/V) * temp
axes[0, 1].plot(time, pressure_wall, 'b-', label='Measured P_wall')
axes[0, 1].plot(time, nT, 'k--', label='Theory P = nT')
axes[0, 1].set_title('Equation of State (EOS)')
axes[0, 1].set_ylabel('P [GeV/fm^3]')
axes[0, 1].legend()
axes[0, 1].grid(True)

# C. Energy Distribution & Boltzmann Fit
counts, bins, _ = axes[1, 0].hist(energies, bins=80, density=True, alpha=0.6, color='skyblue', label='Simulated')

# Fit Functions
def relativistic_dist(E, T, A):
    return A * (E**2) * np.exp(-E/T)

def classical_dist(E, T, A):
    return A * np.sqrt(E) * np.exp(-E/T)

bin_centers = (bins[:-1] + bins[1:]) / 2
if is_relativistic:
    func = relativistic_dist
    p0 = [np.mean(energies)/3, 1.0]
    fit_label = 'Rel. Boltzmann (E^2)'
else:
    func = classical_dist
    p0 = [np.mean(energies)/1.5, 1.0]
    fit_label = 'Classical Boltzmann (sqrt(E))'

try:
    popt, _ = curve_fit(func, bin_centers, counts, p0=p0)
    T_fit = popt[0]
    e_range = np.linspace(0.01, np.max(energies), 200)
    axes[1, 0].plot(e_range, func(e_range, *popt), 'r--', lw=2, label=f'Fit {fit_label}\n(T={T_fit:.3f})')
except Exception as e:
    print(f"Fit failed: {e}")
    T_fit = 0

axes[1, 0].set_title('Energy Spectrum Validation')
axes[1, 0].set_xlabel('Energy [GeV]')
axes[1, 0].set_yscale('log')
axes[1, 0].set_ylim(1e-4, 10.0 if not is_relativistic else 1.0)
axes[1, 0].legend()
axes[1, 0].grid(True, which='both', ls='-', alpha=0.2)

# D. Summary
axes[1, 1].text(0.1, 0.4, f"Final Kinetic T: {temp[-1]:.4f} GeV\n"
                          f"Final Fit T:     {T_fit:.4f} GeV\n"
                          f"N_particles:     {N}\n"
                          f"EOS Consistency: {abs(pressure_wall[-1]-nT[-1])/nT[-1]*100:.2f}%", 
                fontsize=12, family='monospace', bbox=dict(facecolor='white', alpha=0.5))
axes[1, 1].set_axis_off()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
out_name = 'validation_rel.png' if is_relativistic else 'validation_class.png'
plt.savefig(out_name)
print(f"Validation plot saved to {out_name}.")
