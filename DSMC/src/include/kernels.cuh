#pragma once
#include "common.cuh"
#include "kinematics.cuh"
#include "potentials.cuh"
#include "thermostats.cuh"

namespace bamps {

// Boundary Policies
struct ReflectiveWall {
    static constexpr bool is_periodic = false;
    __device__ inline static void apply(float3& pos, float3& mom, float box_size, double* d_wall_mom) {
        float half = box_size * 0.5f;
        auto reflect = [&](float& p, float& m) {
            if (p > half) { 
                if (d_wall_mom) atomicAdd(d_wall_mom, (double)(2.0f * fabsf(m)));
                p = 2.0f * half - p; m = -m; 
            }
            else if (p < -half) { 
                if (d_wall_mom) atomicAdd(d_wall_mom, (double)(2.0f * fabsf(m)));
                p = -2.0f * half - p; m = -m; 
            }
        };
        reflect(pos.x, mom.x); reflect(pos.y, mom.y); reflect(pos.z, mom.z);
    }
};

struct PeriodicBoundary {
    static constexpr bool is_periodic = true;
    __device__ inline static void apply(float3& pos, float3& mom, float box_size, double* d_wall_mom) {
        auto wrap = [&](float& p) {
            while (p >  box_size * 0.5f) p -= box_size;
            while (p < -box_size * 0.5f) p += box_size;
        };
        wrap(pos.x); wrap(pos.y); wrap(pos.z);
    }
};

__global__ void init_particles_kernel(
    Particle* particles, curandState* states, int n, 
    float box_size, float target_T) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    curandState local_state = states[idx];

    particles[idx].pos = make_float3(
        (curand_uniform(&local_state) - 0.5f) * box_size,
        (curand_uniform(&local_state) - 0.5f) * box_size,
        (curand_uniform(&local_state) - 0.5f) * box_size
    );

    float sig = sqrtf(target_T / MASS);
    particles[idx].mom = make_float3(
        curand_normal(&local_state) * sig * MASS,
        curand_normal(&local_state) * sig * MASS,
        curand_normal(&local_state) * sig * MASS
    );

    particles[idx].is_alive = 1;
    particles[idx].E = dot(particles[idx].mom, particles[idx].mom) * (0.5f / MASS);
    states[idx] = local_state;
}

__global__ void build_grid_kernel(
    const Particle* particles, int* grid_indices, int* grid_counts, 
    int n, int IX, float box_size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n || !particles[idx].is_alive) return;

    float half = box_size * 0.5f;
    int ix = (int)((particles[idx].pos.x + half) / box_size * (float)IX);
    int iy = (int)((particles[idx].pos.y + half) / box_size * (float)IX);
    int iz = (int)((particles[idx].pos.z + half) / box_size * (float)IX);
    
    ix = max(0, min(ix, IX - 1));
    iy = max(0, min(iy, IX - 1));
    iz = max(0, min(iz, IX - 1));

    int cell_idx = ix + IX * iy + IX * (IX * iz);
    int offset = atomicAdd(&grid_counts[cell_idx], 1);
    if (offset < MAX_PARTICLES_PER_CELL) {
        grid_indices[cell_idx * MAX_PARTICLES_PER_CELL + offset] = idx;
    }
}

template <typename Kinematics, typename Boundary>
__global__ void kick_drift_kernel(
    Particle* particles, float3* forces, int n, 
    float dt, float box_size, double* d_wall_mom) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n || !particles[idx].is_alive) return;

    Particle& p = particles[idx];
    float3& f = forces[idx];

    // Kick 1: Half-step momentum
    p.mom = p.mom + f * (0.5f * dt);

    // Drift: Position
    float3 v = Kinematics::get_velocity(p.mom, p.E);
    p.pos = p.pos + v * dt;

    // Boundary
    Boundary::apply(p.pos, p.mom, box_size, d_wall_mom);
}

template <typename Potential, typename Boundary>
__global__ void compute_forces_kernel(
    Particle* particles, float3* forces, 
    int* grid_indices, int* grid_counts, double* d_virial,
    int n, int IX, float box_size, float sigma, float epsilon) 
{
    int idx_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_i >= n || !particles[idx_i].is_alive) return;

    float3 pos_i = particles[idx_i].pos;
    float3 force_i = make_float3(0, 0, 0);
    double local_virial = 0;

    float half = box_size * 0.5f;
    int ix = (int)((pos_i.x + half) / box_size * (float)IX);
    int iy = (int)((pos_i.y + half) / box_size * (float)IX);
    int iz = (int)((pos_i.z + half) / box_size * (float)IX);
    ix = max(0, min(ix, IX - 1)); iy = max(0, min(iy, IX - 1)); iz = max(0, min(iz, IX - 1));

    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                int nix, niy, niz;
                if (Boundary::is_periodic) {
                    nix = (ix + dx + IX) % IX;
                    niy = (iy + dy + IX) % IX;
                    niz = (iz + dz + IX) % IX;
                } else {
                    nix = ix + dx; niy = iy + dy; niz = iz + dz;
                    if (nix < 0 || nix >= IX || niy < 0 || niy >= IX || niz < 0 || niz >= IX) continue;
                }
                
                int cell_idx = nix + IX * niy + IX * (IX * niz);
                int count = min(grid_counts[cell_idx], MAX_PARTICLES_PER_CELL);
                for (int j = 0; j < count; j++) {
                    int idx_j = grid_indices[cell_idx * MAX_PARTICLES_PER_CELL + j];
                    if (idx_i == idx_j) continue;

                    float3 f = Potential::calculate_force(
                        pos_i, particles[idx_j].pos, sigma, epsilon, box_size, 
                        (idx_i < idx_j) ? &local_virial : nullptr
                    );
                    
                    // Force Capping for Stability at high density
                    float f_mag = length(f);
                    if (f_mag > 10000.0f) f = f * (10000.0f / f_mag);
                    
                    force_i = force_i + f;
                }
            }
        }
    }

    forces[idx_i] = force_i;
    if (local_virial != 0 && d_virial) atomicAdd(d_virial, local_virial);
}

template <typename Kinematics, typename Thermostat>
__global__ void kick_final_kernel(
    Particle* particles, float3* forces, curandState* states,
    int n, float dt, float target_T, float nu) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n || !particles[idx].is_alive) return;

    Particle& p = particles[idx];
    float3& f = forces[idx];

    // Kick 2: Final half-step momentum
    p.mom = p.mom + f * (0.5f * dt);

    // Apply Thermostat strategy
    Thermostat::apply(p, target_T, dt, nu, states[idx]);

    // Update Energy
    p.E = Kinematics::calculate_energy(p.mom);
}

__global__ void apply_global_scaling_kernel(Particle* particles, int n, float s) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && particles[idx].is_alive) {
        particles[idx].mom.x *= s;
        particles[idx].mom.y *= s;
        particles[idx].mom.z *= s;
    }
}

} // namespace bamps
