#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdarg.h>

/**
 * BAMPS GPU Implementation - "Ancient" Hybrid Edition
 * Support for both Classical DSMC and Relativistic BAMPS modes.
 */

// --- MODE SWITCH ---
#define MODE_RELATIVISTIC 1 // 1 for Relativistic (BAMPS), 0 for Classical (Newtonian)
// -------------------

#define PI 3.14159265358979323846f
#define HBARC 0.197327f 
#define MAX_PARTICLES_PER_CELL 128 
#define MASS 1.0f // GeV

struct Particle {
    float4 pos; 
    float4 mom; 
};

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
        float p_mag = 2.0f;
        float phi = curand_uniform(&local_state) * 2.0f * PI;
        float costheta = curand_uniform(&local_state) * 2.0f - 1.0f;
        float sintheta = sqrtf(fmaxf(0.0f, 1.0f - costheta * costheta));
        
        float px = p_mag * sintheta * cosf(phi);
        float py = p_mag * sintheta * sinf(phi);
        float pz = p_mag * costheta;
#if MODE_RELATIVISTIC
        float energy = p_mag; // m=0 limit
#else
        float energy = (px*px + py*py + pz*pz) / (2.0f * MASS);
#endif
        mom[idx] = make_float4(px, py, pz, energy);
        pos[idx] = make_float4(curand_uniform(&local_state)*box_size - box_size/2.0f,
                               curand_uniform(&local_state)*box_size - box_size/2.0f,
                               curand_uniform(&local_state)*box_size - box_size/2.0f, 0.0f);
        is_alive[idx] = 1;
    }
}

__global__ void update_pos_wall_kernel(
    float4* pos, float4* mom, int* is_alive,
    int* grid_indices, int* grid_counts, double* d_wall_mom,
    int num_slots, float dt, float box_size, float dx, int IX) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_slots || is_alive[idx] != 1) return;

    float4 p = pos[idx]; float4 m = mom[idx];
#if MODE_RELATIVISTIC
    float vx = m.x / m.w; float vy = m.y / m.w; float vz = m.z / m.w;
#else
    float vx = m.x / MASS; float vy = m.y / MASS; float vz = m.z / MASS;
#endif
    p.x += vx * dt; p.y += vy * dt; p.z += vz * dt; p.w += dt;

    float half = box_size / 2.0f;
    double dp = 0;
    if (p.x > half) { dp += 2.0 * fabs(m.x); p.x = 2*half - p.x; m.x = -m.x; }
    else if (p.x < -half) { dp += 2.0 * fabs(m.x); p.x = -2*half - p.x; m.x = -m.x; }
    if (p.y > half) { dp += 2.0 * fabs(m.y); p.y = 2*half - p.y; m.y = -m.y; }
    else if (p.y < -half) { dp += 2.0 * fabs(m.y); p.y = -2*half - p.y; m.y = -m.y; }
    if (p.z > half) { dp += 2.0 * fabs(m.z); p.z = 2*half - p.z; m.z = -m.z; }
    else if (p.z < -half) { dp += 2.0 * fabs(m.z); p.z = -2*half - p.z; m.z = -m.z; }

    if (dp > 0) atomicAdd(d_wall_mom, dp);
    pos[idx] = p; mom[idx] = m;

    int ix = (int)((p.x + half) / box_size * (float)IX);
    int iy = (int)((p.y + half) / box_size * (float)IX);
    int iz = (int)((p.z + half) / box_size * (float)IX);
    ix = max(0, min(ix, IX - 1)); iy = max(0, min(iy, IX - 1)); iz = max(0, min(iz, IX - 1));
    int cell_id = ix + IX * iy + IX * IX * iz;
    int offset = atomicAdd(&grid_counts[cell_id], 1);
    if (offset < MAX_PARTICLES_PER_CELL) grid_indices[cell_id * MAX_PARTICLES_PER_CELL + offset] = idx;
}

__device__ inline float4 boost_relativistic(float4 P, float3 beta) {
    float beta2 = beta.x * beta.x + beta.y * beta.y + beta.z * beta.z;
    if (beta2 < 1e-10f) return P;
    float gamma = 1.0f / sqrtf(fmaxf(1e-10f, 1.0f - beta2));
    float bp = beta.x * P.x + beta.y * P.y + beta.z * P.z;
    float factor = (gamma - 1.0f) / beta2 * bp - gamma * P.w;
    return make_float4(P.x + factor * beta.x, P.y + factor * beta.y, P.z + factor * beta.z, gamma * (P.w - bp));
}

__global__ void collide_enskog_kernel(
    float4* mom, int* is_alive, int* grid_indices, int* grid_counts, curandState* rand_states,
    int num_cells, int IX, float box_size, float dt, float dv, int testpartcl, float as, float md2) 
{
    int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_idx >= num_cells) return;
    int n_i = grid_counts[cell_idx]; if (n_i < 2) return;
    curandState local_state = rand_states[cell_idx];
    int M = n_i / 2; float amplification = (float)n_i * (n_i - 1) / 2.0f / (float)M;
    for (int k = 0; k < M; k++) {
        int i_off = (int)(curand_uniform(&local_state) * n_i);
        int j_off = (int)(curand_uniform(&local_state) * n_i);
        if (i_off == j_off) continue;
        int idx_i = grid_indices[cell_idx * MAX_PARTICLES_PER_CELL + i_off];
        int idx_j = grid_indices[cell_idx * MAX_PARTICLES_PER_CELL + j_off];
        float4 P1 = mom[idx_i]; float4 P2 = mom[idx_j];

#if MODE_RELATIVISTIC
        float E_sum = P1.w + P2.w;
        float3 beta = make_float3((P1.x + P2.x) / E_sum, (P1.y + P2.y) / E_sum, (P1.z + P2.z) / E_sum);
        float s = E_sum * E_sum - ((P1.x + P2.x) * (P1.x + P2.x) + (P1.y + P2.y) * (P1.y + P2.y) + (P1.z + P2.z) * (P1.z + P2.z));
        if (s <= md2) continue;
        float v_rel = s / (2.0f * P1.w * P2.w);
        float sigma = (9.0f * PI * as * as) / (md2 * (1.0f + md2 / s)) * HBARC * HBARC;
#else
        float3 v_rel_vec = make_float3(P1.x/MASS - P2.x/MASS, P1.y/MASS - P2.y/MASS, P1.z/MASS - P2.z/MASS);
        float v_rel = sqrtf(v_rel_vec.x*v_rel_vec.x + v_rel_vec.y*v_rel_vec.y + v_rel_vec.z*v_rel_vec.z);
        float sigma = 0.5f; // Classical hard sphere cross section
#endif
        float prob = amplification * v_rel * sigma * dt / (dv * testpartcl);

        if (curand_uniform(&local_state) < prob) {
#if MODE_RELATIVISTIC
            float p_cm = sqrtf(s) / 2.0f;
            float q2 = md2 * curand_uniform(&local_state) / (1.0f - curand_uniform(&local_state) + 4.0f * md2 / s); 
            float costheta = 1.0f - 2.0f * q2 / s;
            float sintheta = sqrtf(fmaxf(0.0f, 1.0f - costheta * costheta));
            float phi = curand_uniform(&local_state) * 2.0f * PI;
            float4 P1_cm = make_float4(p_cm * sintheta * cosf(phi), p_cm * sintheta * sinf(phi), p_cm * costheta, p_cm);
            float4 P2_cm = make_float4(-P1_cm.x, -P1_cm.y, -P1_cm.z, P1_cm.w);
            float3 minus_beta = make_float3(-beta.x, -beta.y, -beta.z);
            mom[idx_i] = boost_relativistic(P1_cm, minus_beta); mom[idx_j] = boost_relativistic(P2_cm, minus_beta);
#else
            // Classical Isotropic Elastic Collision
            float3 v_cm = make_float3((P1.x + P2.x)/(2*MASS), (P1.y + P2.y)/(2*MASS), (P1.z + P2.z)/(2*MASS));
            float phi = curand_uniform(&local_state) * 2.0f * PI;
            float costheta = curand_uniform(&local_state) * 2.0f - 1.0f;
            float sintheta = sqrtf(fmaxf(0.0f, 1.0f - costheta * costheta));
            float v_mag = v_rel / 2.0f;
            float3 v1_new = make_float3(v_mag*sintheta*cosf(phi), v_mag*sintheta*sinf(phi), v_mag*costheta);
            mom[idx_i].x = (v_cm.x + v1_new.x) * MASS; mom[idx_i].y = (v_cm.y + v1_new.y) * MASS; mom[idx_i].z = (v_cm.z + v1_new.z) * MASS;
            mom[idx_j].x = (v_cm.x - v1_new.x) * MASS; mom[idx_j].y = (v_cm.y - v1_new.y) * MASS; mom[idx_j].z = (v_cm.z - v1_new.z) * MASS;
            mom[idx_i].w = (mom[idx_i].x*mom[idx_i].x + mom[idx_i].y*mom[idx_i].y + mom[idx_i].z*mom[idx_i].z)/(2*MASS);
            mom[idx_j].w = (mom[idx_j].x*mom[idx_j].x + mom[idx_j].y*mom[idx_j].y + mom[idx_j].z*mom[idx_j].z)/(2*MASS);
#endif
        }
    }
    rand_states[cell_idx] = local_state;
}

__global__ void diagnostics_kernel(float4* mom, int* is_alive, double* stats, int num_slots) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_slots && is_alive[idx] == 1) {
        float4 m = mom[idx];
        atomicAdd(&stats[0], (double)m.w); // Total Energy
        atomicAdd(&stats[1], 1.0);        // Count
    }
}

class BAMPS_Ancient {
public:
    int max_slots, num_cells, IX; float box_size, dx, dv, dt;
    float4 *d_pos, *d_mom; int *d_is_alive, *d_grid_indices, *d_grid_counts;
    double *d_stats, *d_wall_mom; curandState *d_rand_states;

    BAMPS_Ancient(int n, float box, int grid_res) : max_slots(n*2), box_size(box), IX(grid_res) {
        num_cells = IX * IX * IX; dx = box_size / (float)IX; dv = dx * dx * dx; dt = 0.01f;
        cudaMalloc(&d_pos, max_slots * sizeof(float4)); cudaMalloc(&d_mom, max_slots * sizeof(float4));
        cudaMalloc(&d_is_alive, max_slots * sizeof(int)); cudaMalloc(&d_grid_indices, num_cells * MAX_PARTICLES_PER_CELL * sizeof(int));
        cudaMalloc(&d_grid_counts, num_cells * sizeof(int)); cudaMalloc(&d_stats, 10 * sizeof(double));
        cudaMalloc(&d_wall_mom, sizeof(double)); cudaMalloc(&d_rand_states, (num_cells + n) * sizeof(curandState));
        init_rand_kernel<<<((num_cells + n) + 255)/256, 256>>>(d_rand_states, (unsigned long)time(NULL), num_cells + n);
        cudaMemset(d_is_alive, 0, max_slots * sizeof(int));
        init_particles_kernel<<<(n + 255)/256, 256>>>(d_pos, d_mom, d_is_alive, d_rand_states + num_cells, n, box_size);
    }

    void evolve() {
        cudaMemset(d_grid_counts, 0, num_cells * sizeof(int));
        update_pos_wall_kernel<<<(max_slots+255)/256, 256>>>(d_pos, d_mom, d_is_alive, d_grid_indices, d_grid_counts, d_wall_mom, max_slots, dt, box_size, dx, IX);
        collide_enskog_kernel<<<(num_cells+255)/256, 256>>>(d_mom, d_is_alive, d_grid_indices, d_grid_counts, d_rand_states, num_cells, IX, box_size, dt, dv, 100, 0.3f, 0.5f);
    }

    void get_diagnostics(double* h_stats, double* h_wall) {
        cudaMemset(d_stats, 0, 10 * sizeof(double));
        diagnostics_kernel<<<(max_slots+255)/256, 256>>>(d_mom, d_is_alive, d_stats, max_slots);
        cudaMemcpy(h_stats, d_stats, 10 * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_wall, d_wall_mom, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemset(d_wall_mom, 0, sizeof(double));
    }
};

int main() {
    int N = 100000; float box_size = 10.0f;
    BAMPS_Ancient sim(N, box_size, 10);
    time_t now = time(0); char* timestamp = ctime(&now);
    FILE* master_f = fopen("bamps_master.log", "a");
    log_info(master_f, "\n# [%s] MODE: %s\n", timestamp, MODE_RELATIVISTIC ? "RELATIVISTIC" : "CLASSICAL");
    
    FILE* short_log = fopen("physics_log.txt", "w");
    fprintf(short_log, "Step Time Temp Pressure_Wall Energy Count\n");

    int total_steps = 10000;
    log_info(master_f, "Simulation Start: %d particles, %d steps\n", N, total_steps);
    
    cudaEvent_t start_bench, stop_bench;
    cudaEventCreate(&start_bench); cudaEventCreate(&stop_bench);
    cudaEventRecord(start_bench);

    for(int i=0; i<=total_steps; i++) {
        sim.evolve();
        if(i % 100 == 0) {
            double stats[10], wall_mom; sim.get_diagnostics(stats, &wall_mom);
            double Area = 6.0 * box_size * box_size;
            double P_wall = wall_mom / (1.0 * Area); // Avg over 100 steps
#if MODE_RELATIVISTIC
            double T = stats[0] / (3.0 * stats[1]);
#else
            double T = stats[0] / (1.5 * stats[1]); // E = 3/2 NkT
#endif
            fprintf(short_log, "%d %f %f %f %f %f\n", i, i*0.01, T, P_wall, stats[0], stats[1]);
            if(i % 2000 == 0) log_info(master_f, "Step %5d: T=%.4f, P=%.4f\n", i, T, P_wall);
        }
    }
    cudaEventRecord(stop_bench); cudaEventSynchronize(stop_bench);
    float ms_total = 0; cudaEventElapsedTime(&ms_total, start_bench, stop_bench);
    log_info(master_f, "Performance: %f ms/step\n", ms_total / (float)total_steps);
    
    fclose(short_log);

    float4* h_mom = new float4[sim.max_slots]; int* h_alive = new int[sim.max_slots];
    cudaMemcpy(h_mom, sim.d_mom, sim.max_slots * sizeof(float4), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_alive, sim.d_is_alive, sim.max_slots * sizeof(int), cudaMemcpyDeviceToHost);
    FILE* ef = fopen("energies.txt", "w");
    for(int i=0; i<sim.max_slots; i++) if(h_alive[i]==1) fprintf(ef, "%f\n", h_mom[i].w);
    fclose(ef);
    if (master_f) fclose(master_f);
    return 0;
}
