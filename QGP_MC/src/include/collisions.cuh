#pragma once
#include "common.cuh"

namespace qgp {

__device__ inline void lorentz_boost(
    float3 p_in, float E_in, float3 beta, float gamma,
    float3& p_out, float& E_out)
{
    float beta2 = dot(beta, beta);
    if (beta2 < 1e-10f) {
        p_out = p_in;
        E_out = E_in;
        return;
    }
    float beta_dot_p = dot(beta, p_in);
    E_out = gamma * (E_in - beta_dot_p);
    float factor = ((gamma - 1.0f) / beta2) * beta_dot_p - gamma * E_in;
    p_out = p_in + beta * factor;
}

__device__ inline void lorentz_boost_back(
    float3 p_in, float E_in, float3 beta, float gamma,
    float3& p_out, float& E_out)
{
    float beta2 = dot(beta, beta);
    if (beta2 < 1e-10f) {
        p_out = p_in;
        E_out = E_in;
        return;
    }
    float beta_dot_p = dot(beta, p_in);
    E_out = gamma * (E_in + beta_dot_p);
    float factor = ((gamma - 1.0f) / beta2) * beta_dot_p + gamma * E_in;
    p_out = p_in + beta * factor;
}

__device__ inline float relativistic_relative_velocity(float3 p1, float E1, float3 p2, float E2, float mass) {
    float p1_dot_p2 = E1 * E2 - dot(p1, p2);
    float term = p1_dot_p2 * p1_dot_p2 - mass * mass * mass * mass;
    if (term < 0.0f) return 0.0f;
    return sqrtf(term) / (E1 * E2);
}

__device__ inline void scatter_relativistic_pair(Particle* particles, int idx1, int idx2, float mass, curandState& state) {
    float3 p1 = particles[idx1].mom;
    float E1 = particles[idx1].E;
    float3 p2 = particles[idx2].mom;
    float E2 = particles[idx2].E;

    // Total momentum and energy in lab frame
    float3 P_tot = p1 + p2;
    float E_tot = E1 + E2;

    // CM frame velocity beta = P_tot / E_tot
    float3 beta = P_tot * (1.0f / E_tot);
    float beta2 = dot(beta, beta);
    if (beta2 >= 1.0f) return; // Superluminal safety

    float gamma = 1.0f / sqrtf(1.0f - beta2);

    // Boost to CM frame
    float3 p1_cm, p2_cm;
    float E1_cm, E2_cm;
    lorentz_boost(p1, E1, beta, gamma, p1_cm, E1_cm);
    lorentz_boost(p2, E2, beta, gamma, p2_cm, E2_cm);

    // Magnitude of CM momentum
    float q_mag = length(p1_cm);
    if (q_mag < 1e-6f) return;

    // Generate random unit vector on sphere
    float z = curand_uniform(&state) * 2.0f - 1.0f;
    float phi = curand_uniform(&state) * 2.0f * PI;
    float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));

    float3 n = make_float3(r * cosf(phi), r * sinf(phi), z);
    float3 p1_cm_new = n * q_mag;
    float3 p2_cm_new = p1_cm_new * -1.0f;

    // Boost back to lab frame
    float3 p1_new, p2_new;
    float E1_new, E2_new;
    lorentz_boost_back(p1_cm_new, E1_cm, beta, gamma, p1_new, E1_new);
    lorentz_boost_back(p2_cm_new, E2_cm, beta, gamma, p2_new, E2_new);

    // Update particles
    particles[idx1].mom = p1_new;
    particles[idx1].E = E1_new;
    particles[idx2].mom = p2_new;
    particles[idx2].E = E2_new;
}

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

__global__ void relativistic_collision_kernel(
    Particle* particles, int* grid_indices, int* grid_counts,
    int num_cells, float sigma_cross, float dt, float cell_volume, float mass,
    curandState* states)
{
    int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_idx >= num_cells) return;

    int count = min(grid_counts[cell_idx], MAX_PARTICLES_PER_CELL);
    if (count < 2) return;

    curandState local_state = states[cell_idx];
    
    // Standard DSMC collision trial count per timestep: count * (count - 1) / 2
    // We sample max(1, count / 2) pairs and weight the collision probability accordingly
    int trials = max(1, count / 2);
    float pair_weight = ((float)count * (float)(count - 1)) / (2.0f * (float)trials);

    for (int trial = 0; trial < trials; ++trial) {
        int idx1, idx2;
        random_distinct_cell_pair(grid_indices, cell_idx, count, local_state, idx1, idx2);
        if (!particles[idx1].is_alive || !particles[idx2].is_alive) continue;

        float g = relativistic_relative_velocity(particles[idx1].mom, particles[idx1].E,
                                                particles[idx2].mom, particles[idx2].E, mass);
        
        // Probability = cross_section * relative_velocity * dt / cell_volume
        float prob = pair_weight * sigma_cross * g * dt / cell_volume;
        if (prob > 1.0f) prob = 1.0f;

        if (curand_uniform(&local_state) < prob) {
            scatter_relativistic_pair(particles, idx1, idx2, mass, local_state);
        }
    }

    states[cell_idx] = local_state;
}

} // namespace qgp
