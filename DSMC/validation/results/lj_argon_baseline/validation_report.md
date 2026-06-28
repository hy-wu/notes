# Validation Report: lj_argon_baseline

Liquid Argon/LJ credibility baseline with BAMPS vs LAMMPS structure, thermodynamics, energy drift, and MSD checks.

## Summary

- Passed metrics: 9
- Failed metrics: 0
- Missing required metrics: 0
- Missing optional metrics: 0
- Passed comparisons: 16
- Failed comparisons: 2
- Missing comparisons/reference data: 0

## Check Metrics

| Metric | Status | Value | Detail |
| --- | --- | ---: | --- |
| nve_energy_relative_drift_max | pass | 0.00097676501 | <= 0.01 |
| nvt_temperature_mean | pass | 0.99635633 | \|value - 1.0\| <= 0.05 |
| nvt_pressure_mean | pass | 1.0010698 | >= -5.0; <= 10.0 |
| diffusion_msd_r2.r2 | pass | 0.99878933 | >= 0.99 |
| diffusion_msd_r2.D | pass | 0.12302228 | >= 0.02; <= 0.15 |
| rdf_peak_position_error | pass | 1.3953488 | <= 5.0 |
| rdf_peak_height_error | pass | 1.8992324 | <= 15.0 |
| sk_peak_position_error | pass | 0 | <= 8.0 |
| pressure_error_vs_lammps | pass | 1.849653 | <= 10.0 |

## BAMPS vs LAMMPS Comparisons

| Observable | Status | BAMPS | LAMMPS | Abs Error | Rel Error % | Detail |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| T_mean | pass | 0.99635633 | 0.99934366 | 0.0029873285 | 0.29892905 | rel_error_pct <= 2.0 |
| P_mean | pass | 1.0010698 | 1.019935 | 0.018865259 | 1.849653 | rel_error_pct <= 10.0 |
| Z_mean | pass | 1.2562768 | 1.2761254 | 0.019848515 | 1.5553734 | rel_error_pct <= 10.0 |
| PE_per_particle | pass | -5.5397631 | -5.5365657 | 0.0031973875 | 0.057750375 | rel_error_pct <= 10.0 |
| TotalE_per_particle | pass | -4.0452292 | -4.0386349 | 0.0065943143 | 0.16328077 | rel_error_pct <= 10.0 |
| rdf_peak_r | pass | 1.06 | 1.075 | 0.015 | 1.3953488 | rel_error_pct <= 5.0 |
| rdf_peak_g | pass | 2.7008697 | 2.65053 | 0.050339723 | 1.8992324 | rel_error_pct <= 15.0 |
| rdf_first_min_r | pass | 1.5 | 1.55833 | 0.05833 | 3.7431096 | rel_error_pct <= 8.0 |
| coordination_number | pass | 11.267004 | 12.200738 | 0.93373396 | 7.6530939 | rel_error_pct <= 15.0 |
| sk_peak_q | pass | 6.7330544 | 6.7330544 | 0 | 0 | rel_error_pct <= 8.0 |
| sk_peak_value | pass | 2.0547342 | 2.1065863 | 0.051852114 | 2.4614284 | rel_error_pct <= 15.0 |
| S0_lowq | fail | 0.073754134 | 0.038286309 | 0.035467825 | 92.638402 | rel_error_pct <= 25.0 |
| kappa_T_from_S0 | fail | 0.092556597 | 0.047903179 | 0.044653418 | 93.215981 | rel_error_pct <= 30.0 |
| kappa_T_from_EOS | pass | 0.080902357 | 0.080042182 | 0.00086017494 | 1.074652 | rel_error_pct <= 30.0 |
| v_rms | pass | 1.6862741 | 1.7352463 | 0.048972185 | 2.8222037 | rel_error_pct <= 5.0 |
| mb_rmse | pass | 0.045019628 | 0.047190223 | 0.0021705944 | 4.5996698 | rel_error_pct <= 50.0 |
| thermo_timeseries_temperature | pass | 0.99635633 | 0.99934366 | 0.0029873285 | 0.29892905 | rel_error_pct <= 2.0 |
| thermo_timeseries_pressure | pass | 1.0010698 | 1.019935 | 0.018865259 | 1.849653 | rel_error_pct <= 10.0 |

## Artifacts

| Artifact | Status | Required | Path |
| --- | --- | --- | --- |
| argon_validation_figure | present | True | argon_results/argon_equilibrium_validation.png |
| argon_summary_table | present | False | argon_results/argon_equilibrium_summary.csv |
| argon_structure_factor_table | present | False | argon_results/argon_structure_factor.csv |
| argon_speed_distribution_table | present | False | argon_results/argon_speed_distribution.csv |
| msd_figure | present | True | diffusion_results/msd_plot.png |
