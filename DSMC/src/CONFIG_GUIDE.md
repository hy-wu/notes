# BAMPS Modern Configuration Guide

This guide explains how to configure and extend the modern BAMPS GPU engine using its template-based policy architecture.

## Core Simulation Setup
The simulation is instantiated in `src/main.cu` using the following template signature:
```cpp
Simulation<Kinematics, Boundary, Potential, Thermostat> sim(N, box, sigma, epsilon, target_T);
```

### 1. Kinematics (`include/kinematics.cuh`)
Defines how velocities and energies are calculated from momenta.
- `ClassicalKinematics`: Standard Newtonian physics ($v = p/m, E = p^2/2m$).
- `RelativisticKinematics`: Einsteinian relativity ($v = p/E, E = \sqrt{p^2 + m^2}$).

### 2. Boundary Conditions (`include/kernels.cuh`)
Defines particle wrapping/reflection and enables specific force logic.
- `PeriodicBoundary`: Full PBC. Enables Minimum Image Convention (MIC) in force kernels.
- `ReflectiveWall`: Hard walls. Momentum transfer is recorded as `d_wall_mom` for pressure calculation.

### 3. Interaction Potentials (`include/potentials.cuh`)
Defines the inter-particle forces.
- `LennardJones`: 12-6 potential with force capping and analytical tail corrections.
- `NullPotential`: No continuous forces. Used for pure DSMC or ideal gas runs.

### 4. Thermostats (`include/thermostats.cuh`)
Defines the heat bath coupling.
- `AndersenThermostat`: Stochastic velocity resets. Controlled by `nu` (collision frequency).
- `LangevinThermostat`: Continuous drag and random noise.
- `GlobalScalingThermostat`: Global velocity scaling (requires reduction step in NHC/Berendsen).
- `NullThermostat`: NVE ensemble (energy conservation testing).

## Common configurations
- **Pure LJ MD (Standard)**: `Simulation<ClassicalKinematics, PeriodicBoundary, LennardJones, AndersenThermostat>`
- **Relativistic Ideal Gas**: `Simulation<RelativisticKinematics, PeriodicBoundary, NullPotential, NullThermostat>`
- **Dense Fluid in a Box**: `Simulation<ClassicalKinematics, ReflectiveWall, LennardJones, AndersenThermostat>`
