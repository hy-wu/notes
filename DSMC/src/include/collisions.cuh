#pragma once
#include "common.cuh"

namespace bamps {

struct NullCollision {
    static constexpr bool has_collision = false;
};

struct HardSphereCollision {
    static constexpr bool has_collision = true;

    __host__ __device__ inline static float cross_section(float diameter) {
        return PI * diameter * diameter;
    }

    __host__ __device__ inline static float packing_fraction(float number_density, float diameter) {
        return (PI / 6.0f) * number_density * diameter * diameter * diameter;
    }

    __host__ __device__ inline static float chi_resummed_cs(float number_density, float diameter) {
        float eta = packing_fraction(number_density, diameter);
        eta = fminf(fmaxf(eta, 0.0f), 0.49f);
        return (1.0f - 0.5f * eta) / ((1.0f - eta) * (1.0f - eta) * (1.0f - eta));
    }

    __host__ __device__ inline static float chi_resummed_ev(float number_density, float diameter) {
        float eta = packing_fraction(number_density, diameter);
        float x = fmaxf(1.0f - 4.0f * eta, 0.05f);
        return 1.0f / x;
    }

    __host__ __device__ inline static float chi_truncated(float number_density, float diameter, int order) {
        float eta = packing_fraction(number_density, diameter);
        if (order <= 0) return 1.0f;
        if (order == 1) return 1.0f + 2.5f * eta;
        return 1.0f + 2.5f * eta + 4.5f * eta * eta;
    }
};

using EnskogHardSphereCollision = HardSphereCollision;

__device__ inline int random_cell_particle(int* grid_indices, int cell_idx, int count, curandState& state) {
    int offset = (int)(curand_uniform(&state) * count);
    if (offset >= count) offset = count - 1;
    return grid_indices[cell_idx * MAX_PARTICLES_PER_CELL + offset];
}

__device__ inline void random_distinct_cell_pair(
    int* grid_indices, int cell_idx, int count, curandState& state, int& idx1, int& idx2)
{
    int off1 = (int)(curand_uniform(&state) * count);
    if (off1 >= count) off1 = count - 1;
    int off2 = (int)(curand_uniform(&state) * (count - 1));
    if (off2 >= count - 1) off2 = count - 2;
    if (off2 >= off1) off2 += 1;
    idx1 = grid_indices[cell_idx * MAX_PARTICLES_PER_CELL + off1];
    idx2 = grid_indices[cell_idx * MAX_PARTICLES_PER_CELL + off2];
}

__device__ inline void scatter_hard_sphere_pair_direct(Particle* particles, int idx1, int idx2, curandState& state) {
    float3 p1 = particles[idx1].mom;
    float3 p2 = particles[idx2].mom;
    float3 p_cm = (p1 + p2) * 0.5f;
    float3 p_rel = p1 - p2;
    float p_rel_mag = length(p_rel);
    if (p_rel_mag <= 1.0e-12f) return;

    float z = curand_uniform(&state) * 2.0f - 1.0f;
    float phi = curand_uniform(&state) * 2.0f * PI;
    float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    float3 p_rel_new = make_float3(
        p_rel_mag * r * cosf(phi),
        p_rel_mag * r * sinf(phi),
        p_rel_mag * z);

    particles[idx1].mom = p_cm + p_rel_new * 0.5f;
    particles[idx2].mom = p_cm - p_rel_new * 0.5f;
    particles[idx1].E = 0.5f * dot(particles[idx1].mom, particles[idx1].mom) / MASS;
    particles[idx2].E = 0.5f * dot(particles[idx2].mom, particles[idx2].mom) / MASS;
}

template <typename Coll>
__global__ void enskog_collision_kernel(
    Particle* particles, int* grid_indices, int* grid_counts, int num_cells, int IX,
    float diameter, float dt, float cell_volume, float number_density, int enskog_order,
    curandState* states, unsigned long long* collision_counts, int* collision_locks = nullptr)
{
    (void)IX;
    (void)collision_locks;

    int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_idx >= num_cells) return;
    int count = min(grid_counts[cell_idx], MAX_PARTICLES_PER_CELL);
    if (count < 2) return;

    if constexpr (Coll::has_collision) {
        curandState local_state = states[cell_idx];
        float sigma_coll = Coll::cross_section(diameter);
        float chi = Coll::chi_truncated(number_density, diameter, enskog_order);
        int trials = max(1, count / 2);
        float pair_weight = ((float)count * (float)(count - 1)) / (2.0f * (float)trials);

        for (int trial = 0; trial < trials; ++trial) {
            int idx1, idx2;
            random_distinct_cell_pair(grid_indices, cell_idx, count, local_state, idx1, idx2);
            if (!particles[idx1].is_alive || !particles[idx2].is_alive) continue;

            float g = length(particles[idx1].mom - particles[idx2].mom) / MASS;
            float prob = pair_weight * chi * sigma_coll * g * dt / cell_volume;
            if (collision_counts) atomicAdd(&collision_counts[2], 1ULL);
            if (prob > 1.0f) {
                prob = 1.0f;
                if (collision_counts) atomicAdd(&collision_counts[3], 1ULL);
            }
            if (curand_uniform(&local_state) < prob) {
                scatter_hard_sphere_pair_direct(particles, idx1, idx2, local_state);
                if (collision_counts) atomicAdd(&collision_counts[0], 1ULL);
            }
        }
        states[cell_idx] = local_state;
    }
}

template <typename Coll>
__global__ void collision_kernel(
    Particle* particles, int* grid_indices, int* grid_counts,
    int num_cells, float sigma, curandState* states)
{
    int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_idx >= num_cells) return;
    int count = min(grid_counts[cell_idx], MAX_PARTICLES_PER_CELL);
    if (count < 2) return;

    if constexpr (Coll::has_collision) {
        curandState local_state = states[cell_idx];
        int idx1, idx2;
        random_distinct_cell_pair(grid_indices, cell_idx, count, local_state, idx1, idx2);
        if (particles[idx1].is_alive && particles[idx2].is_alive) {
            scatter_hard_sphere_pair_direct(particles, idx1, idx2, local_state);
        }
        states[cell_idx] = local_state;
    }
}

} // namespace bamps
