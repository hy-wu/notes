import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# 1. Load Diagnostics
try:
    diag = np.genfromtxt('physics_log.txt', skip_header=1)
    time = diag[:, 1]
    temp = diag[:, 2]
    pressure_wall = diag[:, 3]
except:
    print("Error: physics_log.txt missing")
    exit()

# 2. Load Energies
try:
    energies = np.loadtxt('energies.txt')
except:
    print("Error: energies.txt missing")
    exit()

# Setup Figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Ancient-Hybrid BAMPS: Mechanical Pressure & Energy Distribution', fontsize=16)

# A. Temperature & Pressure vs Time
axes[0, 0].plot(time, temp, 'r-', label='Kinetic T (E/3N)')
axes[0, 0].set_ylabel('Temperature [GeV]')
axes[0, 0].legend(loc='upper left')
axes[0, 0].grid(True)

ax_p = axes[0, 0].twinx()
ax_p.plot(time, pressure_wall, 'b--', label='Wall Pressure')
ax_p.set_ylabel('P_wall [GeV/fm^3]')
ax_p.legend(loc='upper right')

# B. EOS Check: P_wall vs nT
V = 1000.0
N = 100000
nT = (N/V) * temp
axes[0, 1].plot(time, pressure_wall, 'b-', label='Measured P_wall')
axes[0, 1].plot(time, nT, 'k--', label='Theory P = nT')
axes[0, 1].set_title('Equation of State (EOS) Validation')
axes[0, 1].set_xlabel('Time [fm/c]')
axes[0, 1].set_ylabel('P [GeV/fm^3]')
axes[0, 1].legend()
axes[0, 1].grid(True)

# C. Energy Distribution & Boltzmann Fit
counts, bins, _ = axes[1, 0].hist(energies, bins=80, density=True, alpha=0.6, color='skyblue', label='Simulated')

# Fit Boltzmann: f(E) = A * E^2 * exp(-E/T)
def boltzmann_dist(E, T, A):
    return A * (E**2) * np.exp(-E/T)

bin_centers = (bins[:-1] + bins[1:]) / 2
popt, _ = curve_fit(boltzmann_dist, bin_centers, counts, p0=[np.mean(energies)/3, 1.0])
T_fit = popt[0]

e_range = np.linspace(0.1, np.max(energies), 200)
axes[1, 0].plot(e_range, boltzmann_dist(e_range, *popt), 'r--', lw=2, label=f'Boltzmann Fit (T={T_fit:.3f})')
axes[1, 0].set_title('Energy Distribution (End of Run)')
axes[1, 0].set_xlabel('Energy [GeV]')
axes[1, 0].set_ylabel('Probability Density')
axes[1, 0].set_yscale('log')
axes[1, 0].set_ylim(1e-3, 1.0)
axes[1, 0].legend()
axes[1, 0].grid(True, which='both', ls='-', alpha=0.2)

# D. Fit vs Kinetic Temperature
axes[1, 1].text(0.1, 0.6, f"Kinetic T (E/3N): {temp[-1]:.4f} GeV\n"
                          f"Boltzmann Fit T:   {T_fit:.4f} GeV\n"
                          f"Discrepancy:       {abs(temp[-1]-T_fit)/temp[-1]*100:.2f}%", 
                fontsize=12, family='monospace', bbox=dict(facecolor='white', alpha=0.5))
axes[1, 1].set_title('Temperature Consistency Check')
axes[1, 1].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('final_validation.png')
print(f"Validation plot saved to final_validation.png. Fit T = {T_fit:.4f}")
