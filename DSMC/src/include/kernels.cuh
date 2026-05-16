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
    __device__ inline static void apply(float3& pos, float3& mom, float box_size, double* d_wall_mom, int3& image_flags) {
        auto wrap = [&](float& p, int& image) {
            float half = box_size * 0.5f;
            if (p >  half) { p -= box_size; image++; }
            else if (p < -half) { p += box_size; image--; }
        };
        wrap(pos.x, image_flags.x); wrap(pos.y, image_flags.y); wrap(pos.z, image_flags.z);
    }
};

__global__ void init_particles_kernel(
    Particle* particles, curandState* states, int n, 
    float box_size, float target_T) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int n_side = (int)ceilf(powf((float)n, 1.0f/3.0f));
    float spacing = box_size / n_side;
    
    int ix = idx % n_side;
    int iy = (idx / n_side) % n_side;
    int iz = idx / (n_side * n_side);
    
    float half = box_size * 0.5f;
    particles[idx].pos = make_float3(
        ix * spacing - half + spacing * 0.5f,
        iy * spacing - half + spacing * 0.5f,
        iz * spacing - half + spacing * 0.5f
    );
    
    particles[idx].initial_pos = particles[idx].pos;
    particles[idx].image_flags = make_int3(0, 0, 0);

    curandState local_state = states[idx];
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

__global__ void reset_msd_kernel(Particle* particles, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    particles[idx].initial_pos = particles[idx].pos;
    particles[idx].image_flags = make_int3(0, 0, 0);
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
    ix = max(0, min(ix, IX - 1)); iy = max(0, min(iy, IX - 1)); iz = max(0, min(iz, IX - 1));

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
    p.mom = p.mom + forces[idx] * (0.5f * dt);
    p.pos = p.pos + Kinematics::get_velocity(p.mom, p.E) * dt;

    if constexpr (Boundary::is_periodic) {
        Boundary::apply(p.pos, p.mom, box_size, d_wall_mom, p.image_flags);
    } else {
        Boundary::apply(p.pos, p.mom, box_size, d_wall_mom);
    }
}

template <typename Potential, typename Boundary>
__global__ void compute_forces_kernel(
    Particle* particles, float3* forces, 
    int* grid_indices, int* grid_counts, double* d_virial, double* d_pe,
    int n, int IX, float box_size, float sigma, float epsilon) 
{
    int idx_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_i >= n || !particles[idx_i].is_alive) return;

    float3 pos_i = particles[idx_i].pos;
    float3 force_i = make_float3(0, 0, 0);
    double local_virial = 0;
    double local_pe = 0;

    float half = box_size * 0.5f;
    int ix = (int)((pos_i.x + half) / box_size * (float)IX);
    int iy = (int)((pos_i.y + half) / box_size * (float)IX);
    int iz = (int)((pos_i.z + half) / box_size * (float)IX);
    ix = max(0, min(ix, IX - 1)); iy = max(0, min(iy, IX - 1)); iz = max(0, min(iz, IX - 1));

    for (int dx = -2; dx <= 2; dx++) { 
        for (int dy = -2; dy <= 2; dy++) {
            for (int dz = -2; dz <= 2; dz++) {
                int nix, niy, niz;
                if (Boundary::is_periodic) {
                    nix = (ix + dx + IX) % IX; niy = (iy + dy + IX) % IX; niz = (iz + dz + IX) % IX;
                } else {
                    nix = ix + dx; niy = iy + dy; niz = iz + dz;
                    if (nix < 0 || nix >= IX || niy < 0 || niy >= IX || niz < 0 || niz >= IX) continue;
                }
                
                int cell_idx = nix + IX * niy + IX * (IX * niz);
                int count = min(grid_counts[cell_idx], MAX_PARTICLES_PER_CELL);
                for (int j = 0; j < count; j++) {
                    int idx_j = grid_indices[cell_idx * MAX_PARTICLES_PER_CELL + j];
                    if (idx_i == idx_j) continue;

                    float3 r_vec = pos_i - particles[idx_j].pos;
                    if (Boundary::is_periodic) {
                        if (r_vec.x > half) r_vec.x -= box_size; else if (r_vec.x < -half) r_vec.x += box_size;
                        if (r_vec.y > half) r_vec.y -= box_size; else if (r_vec.y < -half) r_vec.y += box_size;
                        if (r_vec.z > half) r_vec.z -= box_size; else if (r_vec.z < -half) r_vec.z += box_size;
                    }

                    float3 f_uncapped = Potential::calculate_force(
                        pos_i, particles[idx_j].pos, sigma, epsilon, box_size, nullptr, (idx_i < idx_j) ? &local_pe : nullptr
                    );
                    
                    float f_mag = length(f_uncapped);
                    float3 f_capped = f_uncapped;
                    if (f_mag > 10000.0f) f_capped = f_uncapped * (10000.0f / f_mag);
                    
                    force_i = force_i + f_capped;
                    if (d_virial && idx_i < idx_j) {
                         local_virial += dot(f_capped, r_vec);
                    }
                }
            }
        }
    }

    forces[idx_i] = force_i;
    if (local_virial != 0 && d_virial) atomicAdd(d_virial, local_virial);
    if (local_pe != 0 && d_pe) atomicAdd(d_pe, local_pe);
}

template <typename Kinematics, typename Thermostat>
__global__ void kick_final_kernel(
    Particle* particles, float3* forces, curandState* states,
    int n, float dt, float target_T, float nu) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n || !particles[idx].is_alive) return;

    Particle& p = particles[idx];
    p.mom = p.mom + forces[idx] * (0.5f * dt);
    Thermostat::apply(p, target_T, dt, nu, states[idx]);
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

__global__ void calculate_msd_kernel(const Particle* particles, int n, float box_size, double* out_msd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n || !particles[idx].is_alive) return;

    const Particle& p = particles[idx];
    float3 unwrapped = make_float3(
        p.pos.x + p.image_flags.x * box_size,
        p.pos.y + p.image_flags.y * box_size,
        p.pos.z + p.image_flags.z * box_size
    );

    float3 diff = unwrapped - p.initial_pos;
    double d2 = (double)diff.x*diff.x + (double)diff.y*diff.y + (double)diff.z*diff.z;
    atomicAdd(out_msd, d2);
}

} // namespace bamps
