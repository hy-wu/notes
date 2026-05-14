#pragma once
#include "common.cuh"

namespace bamps {

struct NullThermostat {
    __device__ inline static void apply(Particle& p, float target_T, float dt, float nu, curandState& state) {}
};

struct AndersenThermostat {
    __device__ inline static void apply(Particle& p, float target_T, float dt, float nu, curandState& state) {
        if (curand_uniform(&state) < (nu * dt)) {
            float sig = sqrtf(target_T / MASS);
            p.mom.x = curand_normal(&state) * sig * MASS;
            p.mom.y = curand_normal(&state) * sig * MASS;
            p.mom.z = curand_normal(&state) * sig * MASS;
        }
    }
};

struct LangevinThermostat {
    __device__ inline static void apply(Particle& p, float target_T, float dt, float gamma, curandState& state) {
        float coeff_drag = 1.0f - gamma * dt;
        float coeff_random = sqrtf(2.0f * gamma * target_T * MASS * dt);
        
        p.mom.x = p.mom.x * coeff_drag + curand_normal(&state) * coeff_random;
        p.mom.y = p.mom.y * coeff_drag + curand_normal(&state) * coeff_random;
        p.mom.z = p.mom.z * coeff_drag + curand_normal(&state) * coeff_random;
    }
};

/**
 * Global Scaling Thermostats (Berendsen, Nosé-Hoover)
 * Scaling is handled via apply_global_scaling_kernel in Simulation::step().
 * The local apply() here is a NO-OP to prevent using 'nu' as a scaling factor.
 */
struct GlobalScalingThermostat {
    __device__ inline static void apply(Particle& p, float target_T, float dt, float nu, curandState& state) {}
};

} // namespace bamps
