# Validation Roadmap

This roadmap defines how to move from the current LJ/Argon credibility baseline
to a broader multi-physics validation suite.

## Current Baseline: `lj_argon_baseline`

Status: active.

The baseline is now represented by:

- Case config: `cases/lj_argon_baseline.json`
- Runner: `validate_suite.py`
- Normalized outputs: `results/lj_argon_baseline/`

Current checks from regenerated artifacts:

- NVE total-energy relative drift
- NVT steady-state temperature
- NVT virial-pressure sanity bounds
- MSD late-time linearity and diffusion coefficient bounds
- Required figure/artifact presence
- BAMPS vs LAMMPS summary comparisons for thermodynamics, RDF first shell,
  structure-factor peak, speed distribution, and compressibility diagnostics

Known gaps:

- Static `S(k)` is currently produced through an RDF transform in the Argon
  script; the low-k `S(0)` and `kappa_T_from_S0` comparisons fail against
  LAMMPS, while the structure-factor peak agrees well. A direct
  particle-coordinate structure-factor estimator should be added before using
  low-k behavior as a strong compressibility claim.
- Multi-seed uncertainty is not yet normalized into the report.

## Case Acceptance Contract

Every validation case should provide:

- A JSON case file under `validation/cases/`.
- A reproducible run command or script.
- A small set of required artifacts.
- Quantitative metrics with thresholds.
- A normalized `validation_report.md`, `metrics.csv`, and `artifacts.csv`.

Recommended metric classes:

- Numerical stability: NVE energy drift, NaN/overflow checks, grid overflow.
- Equilibrium statistics: temperature, Maxwell-Boltzmann or Juttner distribution.
- Structure: RDF peak position/height, coordination number, direct and RDF-derived `S(k)`.
- Thermodynamics: pressure, energy per particle, compressibility factor `Z`, EOS slope.
- Transport: MSD diffusion coefficient, VACF, viscosity or thermal conductivity where applicable.
- Convergence: time step, cutoff, system size, grid resolution, and random seed sensitivity.

## Next Case 1: `enskog_hard_sphere`

Goal: validate the DSMC/Enskog collision branch independent of LJ forces.

Inputs already available:

- `enskog_validate.py`
- `enskog_validate.cu`
- `enskog_results/enskog_validation_summary.csv`
- `enskog_results/enskog_validation_metrics.csv`
- `enskog_results/enskog_validation.png`
- `enskog_results/enskog_size_validation.png`
- `enskog_results/enskog_runtime_efficiency.png`

Proposed required checks:

- Temperature anisotropy relaxation is monotonic within a tolerance window.
- Enskog correction order approaches the expected contact-value trend.
- EOS/compressibility factor error remains under a chosen threshold.
- Runtime scales approximately linearly with particle count for fixed cells.
- Grid overflow stays zero.

## Next Case 2: `lj_state_grid`

Goal: expand beyond one Argon state point to a grid of reduced LJ fluid states.

Suggested states:

- `rho* = 0.3, 0.6, 0.8`
- `T* = 0.8, 1.0, 1.5, 2.0`

For each state:

- Run BAMPS NVT equilibrium.
- Run LAMMPS reference with matching cutoff, tail correction, and thermostat protocol.
- Compare `P`, `E/N`, `Z`, `g(r)`, direct `S(k)`, and speed distribution.
- Save per-state `summary_metrics.csv`.

Acceptance should use error bars from at least three seeds before claiming close agreement.

## Next Case 3: `transport_nonequilibrium`

Goal: validate dynamical transport beyond self-diffusion.

Initial targets:

- Self-diffusion from MSD over multiple state points.
- VACF consistency with the MSD-derived diffusion coefficient.
- Shear viscosity from Green-Kubo or controlled Couette flow.
- Thermal conductivity from a weak temperature-gradient setup.

Acceptance criteria should emphasize linear-response windows and uncertainty estimates.

## Next Case 4: `relativistic_ideal_gas`

Goal: validate relativistic kinematics without confounding interaction errors.

Initial checks:

- Ideal-gas EOS in classical, relativistic massive, and ultrarelativistic modes.
- Momentum distribution against the appropriate equilibrium distribution.
- Energy-momentum tensor consistency.
- Conservation in NVE mode.

Only after these pass should interacting relativistic/QGP-like cases be added.

## Operational Notes

- Keep expensive regenerated raw outputs outside the review-critical path when possible.
- Keep normalized CSV summaries small and committed when they support report claims.
- Prefer direct observable estimators and use RDF-derived transforms as cross-checks.
- Any report claim should cite a metric row, a threshold, and the artifact that generated it.
