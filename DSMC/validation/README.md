# DSMC/BAMPS Validation Workflow

This directory turns the existing validation scripts into a reproducible
workflow with explicit cases, artifacts, metrics, and pass/fail thresholds.

The first active case is `lj_argon_baseline`, which organizes the current
liquid-Argon/Lennard-Jones benchmark:

- BAMPS NVE energy drift
- BAMPS NVT temperature and virial-pressure sanity checks
- BAMPS vs LAMMPS RDF and static-structure-factor summaries when available
- MSD-based self-diffusion linearity and coefficient bounds
- Required figure/artifact presence checks

## Quick Summary From Existing Results

Run from the repository root:

```powershell
python DSMC\validation\validate_suite.py --case lj_argon_baseline --allow-fail
```

This does not launch simulations. It reads existing files under `DSMC/` and
writes normalized outputs to:

```text
DSMC/validation/results/lj_argon_baseline/
  artifacts.csv
  check_metrics.csv
  comparison_metrics.csv
  metrics.csv
  validation_report.md
```

`check_metrics.csv` contains one-sided health checks such as NVE energy drift,
temperature stability, and MSD fit quality. `comparison_metrics.csv` is the
BAMPS-vs-LAMMPS table with `BAMPS`, `LAMMPS`, `abs_error`, and `rel_error_pct`
columns. `metrics.csv` is a combined compatibility table that includes both row
types.

Use `--allow-fail` when you want a report even if required artifacts are missing.
Omit it in CI-like usage to make missing required checks fail the process. Missing
optional artifacts are reported but do not fail the case.

The current regenerated result set includes the LAMMPS reference CSV outputs and
fills `comparison_metrics.csv`. Most thermodynamic and structural checks pass;
the low-k RDF-derived `S(0)` and `kappa_T_from_S0` comparisons currently fail,
which flags the finite-size/RDF-transform compressibility estimate as a
diagnostic rather than a settled validation result.

## Running The Underlying Case

To execute the configured simulation scripts before summarizing:

```powershell
python DSMC\validation\validate_suite.py --case lj_argon_baseline --run
```

The Argon reference step requires:

- `nvcc`
- a LAMMPS executable named `lmp` on `PATH`

The MSD step requires `nvcc`.

Use a dry run to inspect commands:

```powershell
python DSMC\validation\validate_suite.py --case lj_argon_baseline --dry-run
```

## Case Contract

Each case JSON describes:

- `working_dir`: path used when resolving scripts and artifacts
- `run_steps`: commands that can regenerate the case
- `artifacts`: figures and data files expected after a run
- `metrics`: quantitative checks and thresholds

The runner currently supports:

- `csv_column`: compute statistics from a CSV column
- `summary_observable`: read relative errors from a BAMPS/LAMMPS summary table
- `msd_fit`: fit MSD vs time, report `D` and `R^2`

## Next Cases To Add

The registry in `cases/suite.json` lists planned cases:

- `enskog_hard_sphere`: hard-sphere/Enskog collision and EOS validation
- `lj_state_grid`: multi-state LJ fluid grid over density and temperature
- `transport_nonequilibrium`: diffusion, viscosity, heat conduction, Couette flow
- `relativistic_ideal_gas`: ideal relativistic and ultrarelativistic checks

The intended direction is: add one case config, wire its existing script outputs
into metrics, then only expand physics coverage once the case has stable
artifacts and thresholds.

See `VALIDATION_ROADMAP.md` for the staged expansion plan.
