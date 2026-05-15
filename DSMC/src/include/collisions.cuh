#pragma once
#include "common.cuh"

namespace bamps {

struct NullCollision {
    static constexpr bool has_collision = false;
};

// Simplified Stochastic Hard Sphere Collision (DSMC-like)
struct HardSphereCollision {
    static constexpr bool has_collision = true;
    
    // This kernel is launched over cells
    // One thread per cell (or launched differently)
};

template <typename Coll>
__global__ void collision_kernel(
    Particle* particles, int* grid_indices, int* grid_counts, 
    int num_cells, float sigma, curandState* states) 
{
    int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_idx >= num_cells) return;

    int count = grid_counts[cell_idx];
    if (count < 2) return;
    
    // Only process if it's a real collision model
    if constexpr (Coll::has_collision) {
        curandState local_state = states[cell_idx]; // One state per cell
        
        // Simplified DSMC pair selection: pick one random pair per cell per step
        // to exchange momentum isotropically (mocking Hard Sphere scattering).
        // A true DSMC would use N_c = N(N-1) * sigma_v * dt / (2V_c)
        int idx1 = grid_indices[cell_idx * MAX_PARTICLES_PER_CELL + (curand(&local_state) % count)];
        int idx2 = grid_indices[cell_idx * MAX_PARTICLES_PER_CELL + (curand(&local_state) % count)];
        
        if (idx1 != idx2 && particles[idx1].is_alive && particles[idx2].is_alive) {
            float3 p1 = particles[idx1].mom;
            float3 p2 = particles[idx2].mom;
            
            float3 p_cm = (p1 + p2) * 0.5f;
            float3 p_rel = p1 - p2;
            float p_rel_mag = length(p_rel);
            
            // Isotropic scattering in center of mass frame
            float z = curand_uniform(&local_state) * 2.0f - 1.0f; // cos(theta)
            float phi = curand_uniform(&local_state) * 2.0f * PI;
            float r = sqrtf(max(0.0f, 1.0f - z * z));
            
            float3 p_rel_new = make_float3(
                p_rel_mag * r * cosf(phi),
                p_rel_mag * r * sinf(phi),
                p_rel_mag * z
            );
            
            // Update momentums atomically or assume rare collision overlap
            particles[idx1].mom = p_cm + p_rel_new * 0.5f;
            particles[idx2].mom = p_cm - p_rel_new * 0.5f;
        }
        states[cell_idx] = local_state;
    }
}

} // namespace bamps
