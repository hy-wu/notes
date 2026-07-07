#pragma once
#include "common.cuh"

namespace qgp {

struct RelativisticKinematics {
    static constexpr bool is_relativistic = true;

    __device__ inline static float3 get_velocity(const float3& mom, float energy) {
        if (energy > 1e-10f) return mom * (1.0f / energy);
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    __device__ inline static float calculate_energy(const float3& mom, float mass) {
        return sqrtf(dot(mom, mom) + mass * mass);
    }

    __device__ inline static float calculate_kinetic_energy(const float3& mom, float mass) {
        return calculate_energy(mom, mass) - mass;
    }
};

} // namespace qgp
