import matplotlib.pyplot as plt
import numpy as np

# Load energy data
try:
    energies = np.loadtxt('energies.txt')
    print(f"Loaded {len(energies)} particle energies.")
except:
    print("Error: energies.txt not found.")
    exit()

# Plot Histogram
plt.figure(figsize=(10, 6))
counts, bins, _ = plt.hist(energies, bins=300, density=True, alpha=0.7, color='skyblue', label='GPU Simulation')

# Theoretical Boltzmann Distribution: dN/dE ~ E^2 * exp(-E/T)
# Mean energy <E> = 3T for relativistic gas. 
# Our initial beam was at 2 GeV, it should spread.
avg_e = np.mean(energies)
T_eff = avg_e / 3.0
e_range = np.linspace(0.1, np.max(energies), 200)
boltzmann = (e_range**2 / (2 * T_eff**3)) * np.exp(-e_range / T_eff)

plt.plot(e_range, boltzmann, 'r--', lw=2, label=f'Boltzmann Fit (T={T_eff:.2f} GeV)')

plt.title('BAMPS GPU: Energy Distribution of Gluons')
plt.xlabel('Energy E [GeV]')
plt.ylabel('Probability Density')
plt.yscale('log')
plt.ylim(1e-4, 1.0)
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)

plt.savefig('energy_spectrum.png')
print("Plot saved to energy_spectrum.png")
