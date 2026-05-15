#pragma once
#include "common.cuh"

namespace bamps {

struct ClassicalKinematics {
    static constexpr bool is_relativistic = false;

    __device__ inline static float3 get_velocity(const float3& mom, float energy) {
        return mom * (1.0f / MASS);
    }

    __device__ inline static float calculate_energy(const float3& mom) {
        return dot(mom, mom) * (0.5f / MASS);
    }

    __device__ inline static float calculate_kinetic_energy(const float3& mom) {
        return calculate_energy(mom);
    }
};

struct RelativisticKinematics {
    static constexpr bool is_relativistic = true;

    __device__ inline static float3 get_velocity(const float3& mom, float energy) {
        // v = p / E
        if (energy > 1e-10f) return mom * (1.0f / energy);
        return make_float3(0, 0, 0);
    }

    __device__ inline static float calculate_energy(const float3& mom) {
        // E = sqrt(p^2 + m^2). Using m=MASS.
        return sqrtf(dot(mom, mom) + MASS * MASS);
    }

    __device__ inline static float calculate_kinetic_energy(const float3& mom) {
        return calculate_energy(mom) - MASS;
    }
};

struct UltrarelativisticKinematics {
    static constexpr bool is_relativistic = true;

    __device__ inline static float3 get_velocity(const float3& mom, float energy) {
        // v = c * p / |p| (assuming c=1)
        float p_mag = length(mom);
        if (p_mag > 1e-10f) return mom * (1.0f / p_mag);
        return make_float3(0, 0, 0);
    }

    __device__ inline static float calculate_energy(const float3& mom) {
        // E = |p|
        return length(mom);
    }

    __device__ inline static float calculate_kinetic_energy(const float3& mom) {
        // In massless limit, kinetic energy is total energy (E - 0 = E)
        return calculate_energy(mom);
    }
};

} // namespace bamps
