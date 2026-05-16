import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# --- Configuration ---
MASTER_LOG_PATH = "eos_scans/simulation_master_log.csv"
BASE_SIG = 0.2
BASE_EPS = 0.4
BASE_RHO = 4.0
BASE_T   = 1.2
STEPS = 10000

# --- Theory Functions ---

def lj_potential(r, sigma, epsilon):
    if r < 1e-6: return 1e10
    s6 = (sigma / r)**6
    return 4 * epsilon * (s6**2 - s6)

def calculate_b2_lj(T, sig, eps):
    def integrand(r):
        u = lj_potential(r, sig, eps)
        return (np.exp(-u / T) - 1) * (r**2)
    integral, _ = quad(integrand, 0, 10 * sig)
    return -2 * np.pi * integral

def get_theory_z_b2(rho, T, sig, eps):
    eta = (np.pi / 6.0) * rho * (sig**3)
    if eta >= 1.0: return np.nan
    Z_cs = (1 + eta + eta**2 - eta**3) / (1 - eta)**3
    b2_hs = (2.0/3.0) * np.pi * (sig**3)
    b2_lj = calculate_b2_lj(T, sig, eps)
    return Z_cs + (b2_lj - b2_hs) * rho

def get_theory_z_mean_field(rho, T, sig, eps):
    eta = (np.pi / 6.0) * rho * (sig**3)
    if eta >= 1.0: return np.nan
    Z_cs = (1 + eta + eta**2 - eta**3) / (1 - eta)**3
    a = (16.0 / 9.0) * np.pi * eps * (sig**3)
    return Z_cs - (a * rho / T)

def plot_all():
    df = pd.read_csv(MASTER_LOG_PATH)
    thermostats = df['Thermostat'].unique()
    scan_vars = ["sigma", "epsilon", "temp", "rho"]
    
    # 1. EoS Validation Comprehensive
    fig_eos, axes_eos = plt.subplots(2, 2, figsize=(16, 12))
    # 2. Temperature Stability
    fig_temp, axes_temp = plt.subplots(2, 2, figsize=(16, 12))
    # 3. Performance Benchmark (Scalability)
    fig_perf, axes_perf = plt.subplots(2, 2, figsize=(16, 12))
    
    titles = {
        "sigma":   r"Sigma Scan ($\epsilon$=0.4, $\rho$=4.0, $T$=1.2)",
        "epsilon": r"Epsilon Scan ($\sigma$=0.2, $\rho$=4.0, $T$=1.2)",
        "temp":    r"Temperature Scan ($\sigma$=0.2, $\epsilon$=0.4, $\rho$=4.0)",
        "rho":     r"Density Scan ($\sigma$=0.2, $\epsilon$=0.4, $T$=1.2)"
    }
    xlabels = {"sigma": r"$\sigma$", "epsilon": r"$\epsilon$", "temp": r"$T$", "rho": r"$\rho$"}

    for i, var in enumerate(scan_vars):
        ax_e = axes_eos[i//2, i%2]
        ax_t = axes_temp[i//2, i%2]
        ax_p = axes_perf[i//2, i%2]
        
        # --- Range and Theory Setup ---
        sub_all = df[df['ScanVar']==var]
        x_min = sub_all['ScanValue'].min()
        x_max = sub_all['ScanValue'].max()
        x_theory = np.linspace(x_min, x_max, 50)
        
        y_b2 = []
        y_mf = []
        for x in x_theory:
            r_v = BASE_RHO if var != "rho" else x
            t_v = BASE_T if var != "temp" else x
            s_v = BASE_SIG if var != "sigma" else x
            e_v = BASE_EPS if var != "epsilon" else x
            
            z_b2 = get_theory_z_b2(r_v, t_v, s_v, e_v)
            z_mf = get_theory_z_mean_field(r_v, t_v, s_v, e_v)
            
            mult = 1.0 if var == "rho" else r_v
            y_b2.append(mult * z_b2)
            y_mf.append(mult * z_mf)
        
        ax_e.plot(x_theory, y_mf, color='gray', linestyle='--', alpha=0.6, label="Theory (Mean-Field)")
        ax_e.plot(x_theory, y_b2, color='red', linestyle='--', lw=2, label="Theory (CS + B2 Integral)")

        # --- Plot Data ---
        for tname in thermostats:
            sub = df[(df['Thermostat']==tname) & (df['ScanVar']==var)].sort_values('ScanValue')
            if sub.empty: continue
            
            x_data = sub['ScanValue']
            t_meas = sub['MeasuredT']
            p_wall = sub['MeasuredP_wall']
            
            # EoS Plot
            if var == "rho":
                y_data = p_wall / (x_data * t_meas)
                ax_e.set_ylabel(r"$P_{wall}/(\rho T)$")
            else:
                y_data = p_wall / t_meas
                ax_e.set_ylabel(r"$P_{wall}/T$")
            ax_e.plot(x_data, y_data, 'o-', label=f"{tname}")
            
            # Temperature Stability Plot
            if var == "temp":
                y_t = t_meas / x_data  # Normalized by target T
                ax_t.set_ylabel(r"$T_{meas} / T_{target}$")
                ax_t.axhline(1.0, color='red', linestyle=':', alpha=0.3)
            else:
                y_t = t_meas
                ax_t.set_ylabel("Measured T")
                ax_t.axhline(BASE_T, color='red', linestyle=':', alpha=0.3)
            ax_t.plot(x_data, y_t, 'o-', label=f"{tname}")
            
            # Performance Plot
            ms_per_step = (sub['Runtime_sec'] / STEPS) * 1000
            ax_p.plot(x_data, ms_per_step, 's-', label=f"{tname}")

        ax_e.set_title(titles[var])
        ax_e.set_xlabel(xlabels[var])
        ax_e.legend()
        ax_e.grid(True, alpha=0.3)
        
        ax_t.set_title(f"Thermal Equilibrium: {var} Scan")
        ax_t.set_xlabel(xlabels[var])
        ax_t.legend()
        ax_t.grid(True, alpha=0.3)
        
        ax_p.set_title(f"Scaling Performance: {var} Scan")
        ax_p.set_xlabel(xlabels[var])
        ax_p.set_ylabel("ms / step")
        ax_p.legend()
        ax_p.grid(True, alpha=0.3)

    fig_eos.tight_layout()
    fig_eos.savefig("eos_scans/eos_validation_comprehensive.png")
    fig_temp.tight_layout()
    fig_temp.savefig("eos_scans/temp_stability_comprehensive.png")
    fig_perf.tight_layout()
    fig_perf.savefig("eos_scans/performance_benchmarks.png")

    print("All final plots updated successfully.")

if __name__ == "__main__":
    plot_all()
