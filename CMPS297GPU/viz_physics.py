import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Parameter Sync
TESTPARTCL = 100
VOLUME = 1000.0  # 10x10x10

# Detect mode from bamps_master.log
is_relativistic = True
try:
    with open('bamps_master.log', 'r') as f:
        content = f.read()
        last_rel = content.rfind("MODE: RELATIVISTIC")
        last_class = content.rfind("MODE: CLASSICAL")
        is_relativistic = last_rel > last_class
except:
    pass

print(f"Detected Mode for Visualization: {'RELATIVISTIC' if is_relativistic else 'CLASSICAL'}")

# 1. Load Data
try:
    diag = np.genfromtxt('physics_log.txt', skip_header=1)
    time = diag[:, 1]
    temp = diag[:, 2]
    press_raw = diag[:, 3]
    count = diag[:, 5]
    energies = np.loadtxt('energies.txt')
except Exception as e:
    print(f"Data loading failed: {e}")
    exit()

# 2. Correct Pressure for Test-Particle scaling
# Pressure is momentum transfer / (time * area). 
# We need to scale by 1/testpartcl to get physical pressure.
pressure_physical = press_raw / TESTPARTCL
nT_theory = (count / TESTPARTCL / VOLUME) * temp

# 3. Improved Fitting: f(E) = A * E^n * exp(-E/T)
# We fit in log-space for better stability at the tail: log(f) = logA + n*logE - E/T
def log_boltzmann(E, T, logA, n):
    return logA + n * np.log(E + 1e-9) - E/T

def physical_dist(E, T, logA, n):
    return np.exp(logA) * (E**n) * np.exp(-E/T)

counts, bins = np.histogram(energies, bins=80, density=True)
bin_centers = (bins[:-1] + bins[1:]) / 2

# Filter for log fitting
mask = counts > 1e-5
x_fit = bin_centers[mask]
y_log = np.log(counts[mask])

# Initial Guess [T, logA, n]
n_init = 2.0 if is_relativistic else 0.5
p0 = [np.mean(energies)/n_init, 0, n_init]

try:
    popt, _ = curve_fit(log_boltzmann, x_fit, y_log, p0=p0)
    T_fit, logA_fit, n_fit = popt
except Exception as e:
    print(f"Curve fit failed: {e}")
    T_fit, logA_fit, n_fit = p0[0], 0, n_init

# 4. Plotting
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle(f'BAMPS Physics Validation ({"Relativistic" if is_relativistic else "Classical"})', fontsize=18)

# A. Temperature
axes[0, 0].plot(time, temp, 'r-', lw=2)
axes[0, 0].set_title('Temperature Evolution')
axes[0, 0].set_xlabel('Time [fm/c]')
axes[0, 0].set_ylabel('T [GeV]')
axes[0, 0].grid(True)

# B. EOS Verification
axes[0, 1].plot(time, pressure_physical, 'b-', label='Measured P_wall (Scaled)')
axes[0, 1].plot(time, nT_theory, 'k--', alpha=0.8, label='Ideal Gas Theory (P=nT)')
axes[0, 1].set_title('Equation of State (EOS)')
axes[0, 1].set_xlabel('Time [fm/c]')
axes[0, 1].set_ylabel('P [GeV/fm^3]')
axes[0, 1].legend()
axes[0, 1].grid(True)

# C. Energy Spectrum (Log Scale)
axes[1, 0].hist(energies, bins=80, density=True, alpha=0.5, color='cyan', label='Simulated Data')
e_plot = np.linspace(0.01, np.max(energies), 200)
axes[1, 0].plot(e_plot, physical_dist(e_plot, T_fit, logA_fit, n_fit), 'r--', lw=3, 
                label=f'Generalized Fit (n={n_fit:.2f})\nT_fit={T_fit:.4f}')
axes[1, 0].set_title('Energy Spectrum & Boltzmann Fit')
axes[1, 0].set_xlabel('Energy [GeV]')
axes[1, 0].set_yscale('log')
axes[1, 0].set_ylim(1e-4, 5.0)
axes[1, 0].legend()
axes[1, 0].grid(True, which='both', alpha=0.3)

# D. Metrics
axes[1, 1].text(0.1, 0.4, f"Final Kinetic T: {temp[-1]:.4f} GeV\n"
                          f"Final Fit T:     {T_fit:.4f} GeV\n"
                          f"Distribution Index n: {n_fit:.2f}\n"
                          f"EOS Consistency Error: {abs(pressure_physical[-1]-nT_theory[-1])/nT_theory[-1]*100:.2f}%",
                fontsize=14, family='monospace', bbox=dict(facecolor='white', alpha=0.8))
axes[1, 1].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('final_validation_optimized.png')
print("Optimized plot saved as final_validation_optimized.png")
