#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdarg.h>

/**
 * BAMPS GPU Implementation - "Ancient" Atomic Edition (Fixed & Robust)
 */

#ifndef MODE_RELATIVISTIC
#define MODE_RELATIVISTIC 1
#endif
#ifndef MODE_LJ
#define MODE_LJ 0
#endif
#ifndef ENSKOG_ORDER
#define ENSKOG_ORDER 1
#endif
#ifndef TOTAL_STEPS
#define TOTAL_STEPS 10000
#endif
#ifndef TESTPARTCL
#define TESTPARTCL 100
#endif
#ifndef DELTA_T
#define DELTA_T 0.01f
#endif
#ifndef N_PARTICLES_OVERRIDE
#define N_PARTICLES_OVERRIDE 100000
#endif

#define PI 3.14159265358979323846f
#define HBARC 0.197327f 
#define MAX_PARTICLES_PER_CELL 256 
#define MASS 1.0f 

// LJ Parameters
#ifndef SIGMA_OVERRIDE
#define LJ_SIGMA 0.4f
#else
#define LJ_SIGMA SIGMA_OVERRIDE
#endif

#ifndef EPSILON_OVERRIDE
#define LJ_EPSILON 0.01f
#else
#define LJ_EPSILON EPSILON_OVERRIDE
#endif

#define LJ_CUTOFF (2.5f * LJ_SIGMA)
#define THERMOSTAT_NU 0.01f // Reduced from 0.2f to prevent washing out epsilon physics

struct Particle { float4 pos; float4 mom; float4 force; };

void log_info(FILE* f, const char* format, ...) {
    va_list args;
    va_start(args, format); vprintf(format, args); va_end(args);
    va_start(args, format); if (f) { vfprintf(f, format, args); fflush(f); } va_end(args);
}

__global__ void init_rand_kernel(curandState* state, unsigned long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) curand_init(seed, idx, 0, &state[idx]);
}

__global__ void init_particles_kernel(float4* pos, float4* mom, int* is_alive, curandState* state, int n, float box_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandState local_state = state[idx % 1024];
        float p_mag = sqrtf(3.0f * 0.5f * MASS); // Target T=0.5
        float phi = curand_uniform(&local_state) * 2.0f * PI;
        float costheta = curand_uniform(&local_state) * 2.0f - 1.0f;
        float sintheta = sqrtf(fmaxf(0.0f, 1.0f - costheta * costheta));
        mom[idx] = make_float4(p_mag*sintheta*cosf(phi), p_mag*sintheta*sinf(phi), p_mag*costheta, (p_mag*p_mag)/(2.0f*MASS));
        
        int IX_init = (int)powf((float)n, 1.0f/3.0f) + 1;
        float dx_init = box_size / (float)IX_init;
        int iz = idx / (IX_init * IX_init); int iy = (idx / IX_init) % IX_init; int ix = idx % IX_init;
        pos[idx] = make_float4((float)ix * dx_init - box_size/2.0f + dx_init/2.0f,
                               (float)iy * dx_init - box_size/2.0f + dx_init/2.0f,
                               (float)iz * dx_init - box_size/2.0f + dx_init/2.0f, 0.0f);
        is_alive[idx] = 1;
    }
}

__global__ void build_grid_kernel(float4* pos, int* is_alive, int* grid_indices, int* grid_counts, int num_slots, float box_size, float dx, int IX) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_slots || is_alive[idx] != 1) return;
    float4 p = pos[idx]; float half = box_size / 2.0f;
    int ix = (int)((p.x + half) / box_size * (float)IX);
    int iy = (int)((p.y + half) / box_size * (float)IX);
    int iz = (int)((p.z + half) / box_size * (float)IX);
    ix = max(0, min(ix, IX - 1)); iy = max(0, min(iy, IX - 1)); iz = max(0, min(iz, IX - 1));
    int cell_id = ix + IX * iy + IX * IX * iz;
    int offset = atomicAdd(&grid_counts[cell_id], 1);
    if (offset < MAX_PARTICLES_PER_CELL) grid_indices[cell_id * MAX_PARTICLES_PER_CELL + offset] = idx;
}

__global__ void compute_lj_forces_kernel(
    float4* pos, float4* force, int* is_alive, int* grid_indices, int* grid_counts, double* d_virial,
    int num_slots, int IX, float box_size, float lj_sigma, float lj_epsilon) 
{
    int idx_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_i >= num_slots || is_alive[idx_i] != 1) return;

    float4 xi = pos[idx_i]; float3 fi = make_float3(0,0,0);
    float half = box_size / 2.0f;
    int ix = (int)((xi.x + half) / box_size * (float)IX);
    int iy = (int)((xi.y + half) / box_size * (float)IX);
    int iz = (int)((xi.z + half) / box_size * (float)IX);
    ix = max(0, min(ix, IX - 1)); iy = max(0, min(iy, IX - 1)); iz = max(0, min(iz, IX - 1));

    float cutoff = 2.5f * lj_sigma;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                int nix = ix + dx; int niy = iy + dy; int niz = iz + dz;
                if (nix < 0 || nix >= IX || niy < 0 || niy >= IX || niz < 0 || niz >= IX) continue;
                int n_cell_idx = nix + IX * niy + IX * IX * niz;
                int n_j = min(grid_counts[n_cell_idx], MAX_PARTICLES_PER_CELL);
                for (int j_off = 0; j_off < n_j; j_off++) {
                    int idx_j = grid_indices[n_cell_idx * MAX_PARTICLES_PER_CELL + j_off];
                    if (idx_i == idx_j) continue;
                    float4 xj = pos[idx_j];
                    float dx_ = xi.x - xj.x; float dy_ = xi.y - xj.y; float dz_ = xi.z - xj.z;
                    float r2 = dx_*dx_ + dy_*dy_ + dz_*dz_;
                    if (r2 < cutoff * cutoff && r2 > 1e-4f) {
                        float r2inv = 1.0f / r2;
                        float r6inv = r2inv * r2inv * r2inv;
                        float s6 = powf(lj_sigma, 6.0f);
                        float f_mag = 24.0f * lj_epsilon * r2inv * r6inv * s6 * (2.0f * r6inv * s6 - 1.0f);
                        // Force capping for numerical stability during overlaps
                        f_mag = fmaxf(-10000.0f, fminf(10000.0f, f_mag));
                        fi.x += f_mag * dx_; fi.y += f_mag * dy_; fi.z += f_mag * dz_;
                        if (idx_i < idx_j) atomicAdd(d_virial, (double)(f_mag * r2));
                    }
                }
            }
        }
    }
    force[idx_i] = make_float4(fi.x, fi.y, fi.z, 0);
}

__global__ void integrate_andersen_kernel(
    float4* pos, float4* mom, float4* force, int* is_alive,
    double* d_wall_mom, curandState* rand_states,
    int num_slots, float dt, float box_size, float target_T) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_slots || is_alive[idx] != 1) return;
    
    curandState local_state = rand_states[idx % 1024];
    float4 p = pos[idx]; float4 m = mom[idx]; float4 f = force[idx];

#if !MODE_RELATIVISTIC
    m.x += f.x * 0.5f * dt; m.y += f.y * 0.5f * dt; m.z += f.z * 0.5f * dt;
#endif

#if MODE_RELATIVISTIC
    float vx = m.x/m.w; float vy = m.y/m.w; float vz = m.z/m.w;
#else
    float vx = m.x/MASS; float vy = m.y/MASS; float vz = m.z/MASS;
#endif
    p.x += vx * dt; p.y += vy * dt; p.z += vz * dt;

    float half = box_size / 2.0f; double dp_sum = 0;
    if (p.x > half) { dp_sum += 2.0 * fabs(m.x); p.x = 2*half - p.x; m.x = -m.x; }
    else if (p.x < -half) { dp_sum += 2.0 * fabs(m.x); p.x = -2*half - p.x; m.x = -m.x; }
    if (p.y > half) { dp_sum += 2.0 * fabs(m.y); p.y = 2*half - p.y; m.y = -m.y; }
    else if (p.y < -half) { dp_sum += 2.0 * fabs(m.y); p.y = -2*half - p.y; m.y = -m.y; }
    if (p.z > half) { dp_sum += 2.0 * fabs(m.z); p.z = 2*half - p.z; m.z = -m.z; }
    else if (p.z < -half) { dp_sum += 2.0 * fabs(m.z); p.z = -2*half - p.z; m.z = -m.z; }
    if (dp_sum > 0) atomicAdd(d_wall_mom, dp_sum);

#if !MODE_RELATIVISTIC
    if (curand_uniform(&local_state) < THERMOSTAT_NU) {
        float sig = sqrtf(target_T / MASS);
        m.x = curand_normal(&local_state) * sig * MASS;
        m.y = curand_normal(&local_state) * sig * MASS;
        m.z = curand_normal(&local_state) * sig * MASS;
    } else {
        m.x += f.x * 0.5f * dt; m.y += f.y * 0.5f * dt; m.z += f.z * 0.5f * dt;
    }
    m.w = (m.x*m.x + m.y*m.y + m.z*m.z) / (2.0f * MASS);
#endif
    pos[idx] = p; mom[idx] = m;
    rand_states[idx % 1024] = local_state;
}

__device__ inline float4 boost_relativistic(float4 P, float3 beta) {
    float beta2 = beta.x * beta.x + beta.y * beta.y + beta.z * beta.z;
    if (beta2 < 1e-10f) return P;
    float gamma = 1.0f / sqrtf(fmaxf(1e-10f, 1.0f - beta2));
    float bp = beta.x * P.x + beta.y * P.y + beta.z * P.z;
    float factor = (gamma - 1.0f) / beta2 * bp - gamma * P.w;
    return make_float4(P.x + factor * beta.x, P.y + factor * beta.y, P.z + factor * beta.z, gamma * (P.w - bp));
}

__device__ void apply_collision_atomic(float4* mom, int i, int j, float4 P1n, float4 P2n, float4 P1o, float4 P2o) {
    atomicAdd(&mom[i].x, P1n.x - P1o.x); atomicAdd(&mom[i].y, P1n.y - P1o.y);
    atomicAdd(&mom[i].z, P1n.z - P1o.z); atomicAdd(&mom[i].w, P1n.w - P1o.w);
    atomicAdd(&mom[j].x, P2n.x - P2o.x); atomicAdd(&mom[j].y, P2n.y - P2o.y);
    atomicAdd(&mom[j].z, P2n.z - P2o.z); atomicAdd(&mom[j].w, P2n.w - P2o.w);
}

__global__ void collide_enskog_atomic_kernel(
    float4* mom, int* is_alive, int* grid_indices, int* grid_counts, curandState* rand_states,
    int num_cells, int IX, float box_size, float dt, float dv, int testpartcl, float as, float md2, float target_T,
    float lj_sigma, float lj_epsilon) 
{
    int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_idx >= num_cells) return;
    int n_i = min(grid_counts[cell_idx], MAX_PARTICLES_PER_CELL); if (n_i < 1) return;
    curandState local_state = rand_states[cell_idx % 1024];

    if (n_i >= 2) {
        int M = n_i / 2; float amp = (float)n_i * (n_i - 1) / 2.0f / (float)M;
        for (int k = 0; k < M; k++) {
            int i_off = (int)(curand_uniform(&local_state) * n_i);
            int j_off = (int)(curand_uniform(&local_state) * n_i);
            if (i_off == j_off) continue;
            int idx_i = grid_indices[cell_idx * MAX_PARTICLES_PER_CELL + i_off];
            int idx_j = grid_indices[cell_idx * MAX_PARTICLES_PER_CELL + j_off];
            float4 P1 = mom[idx_i]; float4 P2 = mom[idx_j];
#if MODE_RELATIVISTIC
            float E_sum = P1.w + P2.w; float3 beta = make_float3((P1.x+P2.x)/E_sum, (P1.y+P2.y)/E_sum, (P1.z+P2.z)/E_sum);
            float s = E_sum*E_sum - ((P1.x+P2.x)*(P1.x+P2.x) + (P1.y+P2.y)*(P1.y+P2.y) + (P1.z+P2.z)*(P1.z+P2.z));
            if (s <= md2) continue;
            float sigma = (9.0f * PI * as * as) / (md2 * (1.0f + md2 / s)) * HBARC * HBARC;
            float prob = amp * (s / (2.0f * P1.w * P2.w)) * sigma * dt / (dv * testpartcl);
            if (curand_uniform(&local_state) < prob) {
                float3 b = make_float3((P1.x+P2.x)/E_sum, (P1.y+P2.y)/E_sum, (P1.z+P2.z)/E_sum);
                float cost = curand_uniform(&local_state)*2-1; float sint = sqrtf(1-cost*cost); float phi = curand_uniform(&local_state)*2*PI;
                float p_cm = sqrtf(s)/2.0f; float4 P1n_cm = make_float4(p_cm*sint*cosf(phi), p_cm*sint*sinf(phi), p_cm*cost, p_cm);
                float4 P2n_cm = make_float4(-P1n_cm.x, -P1n_cm.y, -P1n_cm.z, P1n_cm.w);
                apply_collision_atomic(mom, idx_i, idx_j, boost_relativistic(P1n_cm, make_float3(-b.x, -b.y, -b.z)), boost_relativistic(P2n_cm, make_float3(-b.x, -b.y, -b.z)), P1, P2);
            }
#else
            float3 vr = make_float3(P1.x/MASS - P2.x/MASS, P1.y/MASS - P2.y/MASS, P1.z/MASS - P2.z/MASS);
            float v_rel = sqrtf(vr.x*vr.x + vr.y*vr.y + vr.z*vr.z);
            
            float sigma_coll = PI * lj_sigma * lj_sigma;
            float boltzmann_factor = expf(lj_epsilon / target_T); 
            float prob = fminf(1.0f, amp * v_rel * sigma_coll * boltzmann_factor * dt / (dv * testpartcl));
            
            if (curand_uniform(&local_state) < prob) {
                float3 v_cm = make_float3((P1.x+P2.x)/(2*MASS), (P1.y+P2.y)/(2*MASS), (P1.z+P2.z)/(2*MASS));
                float cost = curand_uniform(&local_state)*2-1; float sint = sqrtf(1-cost*cost); float phi = curand_uniform(&local_state)*2*PI;
                float3 v1n = make_float3((v_rel/2.0f)*sint*cosf(phi), (v_rel/2.0f)*sint*sinf(phi), (v_rel/2.0f)*cost);
                float4 P1n = make_float4((v_cm.x+v1n.x)*MASS, (v_cm.y+v1n.y)*MASS, (v_cm.z+v1n.z)*MASS, 0);
                P1n.w = (P1n.x*P1n.x + P1n.y*P1n.y + P1n.z*P1n.z)/(2.0f*MASS);
                float4 P2n = make_float4((v_cm.x-v1n.x)*MASS, (v_cm.y-v1n.y)*MASS, (v_cm.z-v1n.z)*MASS, 0);
                P2n.w = (P2n.x*P2n.x + P2n.y*P2n.y + P2n.z*P2n.z)/(2.0f*MASS);
                apply_collision_atomic(mom, idx_i, idx_j, P1n, P2n, P1, P2);
            }
#endif
        }
    }

#if ENSKOG_ORDER >= 1
    int ix = cell_idx % IX; int iy = (cell_idx / IX) % IX; int iz = cell_idx / (IX * IX);
    int n_offsets[][3] = { {-1, 0, 0}, {0, -1, 0}, {0, 0, -1}, {-1, -1, 0}, {-1, 0, -1}, {0, -1, -1}, {-2, 0, 0}, {0, -2, 0}, {0, 0, -2} };
    int num_n = (ENSKOG_ORDER == 1) ? 3 : 9;
    for (int n = 0; n < num_n; n++) {
        int nix = (ix + n_offsets[n][0] + IX) % IX; int niy = (iy + n_offsets[n][1] + IX) % IX; int niz = (iz + n_offsets[n][2] + IX) % IX;
        int n_cell_idx = nix + IX * niy + IX * IX * niz;
        int n_j = min(grid_counts[n_cell_idx], MAX_PARTICLES_PER_CELL); if (n_j < 1) continue;
        int idx_i = grid_indices[cell_idx * MAX_PARTICLES_PER_CELL + (int)(curand_uniform(&local_state)*n_i)];
        int idx_j = grid_indices[n_cell_idx * MAX_PARTICLES_PER_CELL + (int)(curand_uniform(&local_state)*n_j)];
        float4 P1 = mom[idx_i]; float4 P2 = mom[idx_j];
        float3 k_vec = make_float3((float)n_offsets[n][0], (float)n_offsets[n][1], (float)n_offsets[n][2]);
        float k_mag = sqrtf(k_vec.x*k_vec.x + k_vec.y*k_vec.y + k_vec.z*k_vec.z);
        k_vec.x /= k_mag; k_vec.y /= k_mag; k_vec.z /= k_mag;
#if MODE_RELATIVISTIC
        float3 v12 = make_float3(P1.x/P1.w - P2.x/P2.w, P1.y/P1.w - P2.y/P2.w, P1.z/P1.w - P2.z/P2.w);
#else
        float3 v12 = make_float3(P1.x/MASS - P2.x/MASS, P1.y/MASS - P2.y/MASS, P1.z/MASS - P2.z/MASS);
#endif
        float v_dot_k = v12.x*k_vec.x + v12.y*k_vec.y + v12.z*k_vec.z;
        if (v_dot_k < 0) { 
            float v_proj = -v_dot_k;
#if MODE_RELATIVISTIC
            float E_sum = P1.w + P2.w; float s = E_sum*E_sum - ((P1.x+P2.x)*(P1.x+P2.x) + (P1.y+P2.y)*(P1.y+P2.y) + (P1.z+P2.z)*(P1.z+P2.z));
            if (s <= md2) continue;
            float sigma = (9.0f * PI * as * as) / (md2 * (1.0f + md2 / s)) * HBARC * HBARC;
            float prob = (float)n_i * n_j * v_proj * sigma * dt / (dv * testpartcl);
            if (curand_uniform(&local_state) < prob) {
                float3 b = make_float3((P1.x+P2.x)/E_sum, (P1.y+P2.y)/E_sum, (P1.z+P2.z)/E_sum);
                float cost = curand_uniform(&local_state)*2-1; float sint = sqrtf(1-cost*cost); float phi = curand_uniform(&local_state)*2*PI;
                float p_cm = sqrtf(s)/2.0f; float4 P1n_cm = make_float4(p_cm*sint*cosf(phi), p_cm*sint*sinf(phi), p_cm*cost, p_cm);
                float4 P2n_cm = make_float4(-P1n_cm.x, -P1n_cm.y, -P1n_cm.z, P1n_cm.w);
                apply_collision_atomic(mom, idx_i, idx_j, boost_relativistic(P1n_cm, make_float3(-b.x, -b.y, -b.z)), boost_relativistic(P2n_cm, make_float3(-b.x, -b.y, -b.z)), P1, P2);
            }
#else
            float v_rel = sqrtf(v12.x*v12.x + v12.y*v12.y + v12.z*v12.z);
            float sigma_coll = PI * LJ_SIGMA * LJ_SIGMA * (1.0f + LJ_EPSILON / 0.5f); // Using target_T=0.5
            float prob = (float)n_i * n_j * v_proj * sigma_coll * dt / (dv * testpartcl);
            if (curand_uniform(&local_state) < prob) {
                float3 v_cm = make_float3((P1.x+P2.x)/(2*MASS), (P1.y+P2.y)/(2*MASS), (P1.z+P2.z)/(2*MASS));
                float cost = curand_uniform(&local_state)*2-1; float sint = sqrtf(1-cost*cost); float phi = curand_uniform(&local_state)*2*PI;
                float3 v1n = make_float3((v_rel/2.0f)*sint*cosf(phi), (v_rel/2.0f)*sint*sinf(phi), (v_rel/2.0f)*cost);
                float4 P1n = make_float4((v_cm.x+v1n.x)*MASS, (v_cm.y+v1n.y)*MASS, (v_cm.z+v1n.z)*MASS, 0);
                P1n.w = (P1n.x*P1n.x + P1n.y*P1n.y + P1n.z*P1n.z)/(2.0f*MASS);
                float4 P2n = make_float4((v_cm.x-v1n.x)*MASS, (v_cm.y-v1n.y)*MASS, (v_cm.z-v1n.z)*MASS, 0);
                P2n.w = (P2n.x*P2n.x + P2n.y*P2n.y + P2n.z*P2n.z)/(2.0f*MASS);
                apply_collision_atomic(mom, idx_i, idx_j, P1n, P2n, P1, P2);
            }
#endif
        }
    }
#endif
    rand_states[cell_idx % 1024] = local_state;
}

__global__ void diagnostics_kernel(float4* mom, int* is_alive, double* stats, int num_slots) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_slots && is_alive[idx] == 1) {
        float4 m = mom[idx]; atomicAdd(&stats[0], (double)m.w); atomicAdd(&stats[1], 1.0);
    }
}

class BAMPS_Ancient {
public:
    int max_slots, num_cells, IX; float box_size, dx, dv, dt;
    float lj_sigma, lj_epsilon;
    float4 *d_pos, *d_mom, *d_force; int *d_is_alive, *d_grid_indices, *d_grid_counts;
    double *d_stats, *d_wall_mom, *d_virial; curandState *d_rand_states;

    BAMPS_Ancient(int n, float box, int grid_res, float sig, float eps) : max_slots(n*2), box_size(box), IX(grid_res), lj_sigma(sig), lj_epsilon(eps) {
        num_cells = IX * IX * IX; dx = box_size / (float)IX; dv = dx * dx * dx; dt = 0.01f;
        cudaMalloc(&d_pos, max_slots * sizeof(float4)); cudaMalloc(&d_mom, max_slots * sizeof(float4));
        cudaMalloc(&d_force, max_slots * sizeof(float4));
        cudaMalloc(&d_is_alive, max_slots * sizeof(int)); cudaMalloc(&d_grid_indices, num_cells * MAX_PARTICLES_PER_CELL * sizeof(int));
        cudaMalloc(&d_grid_counts, num_cells * sizeof(int)); cudaMalloc(&d_stats, 10 * sizeof(double));
        cudaMalloc(&d_wall_mom, sizeof(double)); cudaMalloc(&d_virial, sizeof(double));
        cudaMalloc(&d_rand_states, (num_cells + max_slots) * sizeof(curandState));
        init_rand_kernel<<<((num_cells + max_slots) + 255)/256, 256>>>(d_rand_states, (unsigned long)time(NULL), num_cells + max_slots);
        cudaMemset(d_is_alive, 0, max_slots * sizeof(int));
        init_particles_kernel<<<(n + 255)/256, 256>>>(d_pos, d_mom, d_is_alive, d_rand_states + num_cells, n, box_size);
    }

    void evolve(float target_T) {
        cudaMemset(d_grid_counts, 0, num_cells * sizeof(int));
        build_grid_kernel<<<(max_slots+255)/256, 256>>>(d_pos, d_is_alive, d_grid_indices, d_grid_counts, max_slots, box_size, dx, IX);
#if MODE_LJ
        cudaMemset(d_force, 0, max_slots * sizeof(float4));
        cudaMemset(d_virial, 0, sizeof(double));
        compute_lj_forces_kernel<<<(max_slots+255)/256, 256>>>(d_pos, d_force, d_is_alive, d_grid_indices, d_grid_counts, d_virial, max_slots, IX, box_size, lj_sigma, lj_epsilon);
#endif
        integrate_andersen_kernel<<<(max_slots+255)/256, 256>>>(d_pos, d_mom, d_force, d_is_alive, d_wall_mom, d_rand_states + num_cells, max_slots, dt, box_size, target_T);
        collide_enskog_atomic_kernel<<<(num_cells+255)/256, 256>>>(d_mom, d_is_alive, d_grid_indices, d_grid_counts, d_rand_states, num_cells, IX, box_size, dt, dv, TESTPARTCL, 0.3f, 0.5f, target_T, lj_sigma, lj_epsilon);
        cudaDeviceSynchronize();
    }

    void get_diagnostics(double* h_stats, double* h_wall, double* h_virial) {
        cudaMemset(d_stats, 0, 10 * sizeof(double));
        diagnostics_kernel<<<(max_slots+255)/256, 256>>>(d_mom, d_is_alive, d_stats, max_slots);
        cudaMemcpy(h_stats, d_stats, 10 * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_wall, d_wall_mom, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_virial, d_virial, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemset(d_wall_mom, 0, sizeof(double));
    }
};

int main(int argc, char** argv) {
    float sig_in = LJ_SIGMA;
    float eps_in = LJ_EPSILON;
    int steps_in = TOTAL_STEPS;
    float dt_in = DELTA_T;
    int N_in = N_PARTICLES_OVERRIDE;

    if (argc >= 3) {
        sig_in = atof(argv[1]);
        eps_in = atof(argv[2]);
    }
    if (argc >= 4) steps_in = atoi(argv[3]);
    if (argc >= 5) dt_in = atof(argv[4]);
    if (argc >= 6) N_in = atoi(argv[5]);

    int N = N_in; float box_size = 10.0f; BAMPS_Ancient sim(N, box_size, 10, sig_in, eps_in);
    sim.dt = dt_in; float target_T = 0.5f;
    FILE* master_f = fopen("bamps_master.log", "a");
    log_info(master_f, "\n# RUN MODE: %s, LJ: %d, N: %d, DT: %f, T_target: %f, TESTPARTCL: %d, SIGMA: %f, EPS: %f\n", 
             MODE_RELATIVISTIC ? "REL" : "CLASS", MODE_LJ, N, sim.dt, target_T, TESTPARTCL, sig_in, eps_in);
    FILE* short_log = fopen("physics_log.txt", "w"); fprintf(short_log, "Step Time Temp Pressure_Wall Energy Count Pressure_Virial\n");
    cudaEvent_t start_bench, stop_bench; cudaEventCreate(&start_bench); cudaEventCreate(&stop_bench); cudaEventRecord(start_bench);
    for(int i=0; i<=steps_in; i++) {
        sim.evolve(target_T);
        if(i % 100 == 0) {
            double stats[10], wall_mom, virial; sim.get_diagnostics(stats, &wall_mom, &virial);
            double Area = 6.0 * box_size * box_size; double Vol = box_size*box_size*box_size;
            double P_wall = wall_mom / (100.0 * sim.dt * Area * TESTPARTCL); 
            double T = MODE_RELATIVISTIC ? stats[0] / (3.0 * stats[1]) : stats[0] / (1.5 * stats[1]);
            double P_virial = (stats[1]/TESTPARTCL * T / Vol) + (virial / (3.0 * Vol * TESTPARTCL * TESTPARTCL));
            fprintf(short_log, "%d %f %f %f %f %f %f\n", i, i*sim.dt, T, P_wall, stats[0], stats[1], P_virial);
        }
    }
    cudaEventRecord(stop_bench); cudaEventSynchronize(stop_bench);
    float ms_total = 0; cudaEventElapsedTime(&ms_total, start_bench, stop_bench);
    log_info(master_f, "Performance: %f ms/step\n", ms_total / (float)TOTAL_STEPS);
    fclose(short_log); fclose(master_f);
    float4* h_pos = new float4[sim.max_slots]; float4* h_mom = new float4[sim.max_slots]; int* h_alive = new int[sim.max_slots];
    cudaMemcpy(h_pos, sim.d_pos, sim.max_slots*sizeof(float4), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mom, sim.d_mom, sim.max_slots*sizeof(float4), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_alive, sim.d_is_alive, sim.max_slots*sizeof(int), cudaMemcpyDeviceToHost);
    FILE* ef = fopen("energies.txt", "w"); FILE* pf = fopen("positions.txt", "w");
    for(int i=0; i<sim.max_slots; i++) if(h_alive[i]==1) { fprintf(ef, "%f\n", h_mom[i].w); fprintf(pf, "%f %f %f\n", h_pos[i].x, h_pos[i].y, h_pos[i].z); }
    fclose(ef); fclose(pf);
    return 0;
}
