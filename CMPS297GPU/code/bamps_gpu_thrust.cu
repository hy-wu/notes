#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/binary_search.h>
#include <thrust/sequence.h>
#include <thrust/for_each.h>
#include <thrust/reduce.h>
#include <thrust/count.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/**
 * BAMPS GPU Implementation - Phase 3 (Performance Optimized)
 * Using Sampling Method for both 2->2 and 3->2 to ensure O(N) scaling.
 */

#define PI 3.14159265358979323846f
#define HBARC 0.197327f // GeV*fm

struct Particle {
    float4 pos; 
    float4 mom; 
    int is_alive;
};

__global__ void init_rand_kernel(curandState* state, unsigned long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) curand_init(seed, idx, 0, &state[idx]);
}

__device__ inline int get_cell_id(float4 p, float dx, int IX, float box_size) {
    float half = box_size / 2.0f;
    int ix = (int)((p.x + half) / box_size * IX);
    int iy = (int)((p.y + half) / box_size * IX);
    int iz = (int)((p.z + half) / box_size * IX);
    ix = max(0, min(ix, IX - 1)); iy = max(0, min(iy, IX - 1)); iz = max(0, min(iz, IX - 1));
    return ix + IX * iy + IX * IX * iz;
}

__global__ void compute_cell_ids_kernel(float4* pos, int* is_alive, int* cell_ids, int num_slots, float dx, int IX, float box_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_slots) {
        if (is_alive[idx]) cell_ids[idx] = get_cell_id(pos[idx], dx, IX, box_size);
        else cell_ids[idx] = 2000000000; 
    }
}

__global__ void propagate_kernel(float4* pos, float4* mom, int* is_alive, int num_slots, float dt, float box_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_slots && is_alive[idx]) {
        float4 p = pos[idx]; float4 m = mom[idx];
        p.x += (m.x / m.w) * dt; p.y += (m.y / m.w) * dt; p.z += (m.z / m.w) * dt; p.w += dt;
        float half = box_size / 2.0f;
        if (p.x > half) p.x -= box_size; else if (p.x < -half) p.x += box_size;
        if (p.y > half) p.y -= box_size; else if (p.y < -half) p.y += box_size;
        if (p.z > half) p.z -= box_size; else if (p.z < -half) p.z += box_size;
        pos[idx] = p;
    }
}

__device__ inline float4 boost(float4 P, float3 beta) {
    float beta2 = beta.x * beta.x + beta.y * beta.y + beta.z * beta.z;
    if (beta2 < 1e-10f) return P;
    float gamma = 1.0f / sqrtf(1.0f - beta2);
    float bp = beta.x * P.x + beta.y * P.y + beta.z * P.z;
    float factor = (gamma - 1.0f) / beta2 * bp - gamma * P.w;
    return make_float4(P.x + factor * beta.x, P.y + factor * beta.y, P.z + factor * beta.z, gamma * (P.w - bp));
}

// Optimized 2->2 and 2->3 with sampling
__global__ void collide_pairs_sampled_kernel(
    float4* sorted_pos, float4* sorted_mom, int* sorted_is_alive,
    int* cell_starts, int* cell_ends, float* cell_md2,
    curandState* rand_states, int* d_atomic_counter,
    int max_slots, int num_cells, float dt, float dv, int testpartcl, float as) 
{
    int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_idx >= num_cells) return;
    int start = cell_starts[cell_idx]; int end = cell_ends[cell_idx];
    int n = end - start;
    if (n < 2) return;

    float md2 = cell_md2[cell_idx]; if (md2 < 0.01f) md2 = 0.01f;
    curandState local_state = rand_states[cell_idx];

    // Sample M pairs per cell. M = n/2
    int M = n / 2;
    float amplification = (float)n * (n - 1) / 2.0f / (float)M;

    for (int k = 0; k < M; k++) {
        int i = start + (int)(curand_uniform(&local_state) * n);
        int j = start + (int)(curand_uniform(&local_state) * n);
        if (i == j) continue;

        float4 P1 = sorted_mom[i]; float4 P2 = sorted_mom[j];
        float E_sum = P1.w + P2.w;
        float3 beta = make_float3((P1.x + P2.x) / E_sum, (P1.y + P2.y) / E_sum, (P1.z + P2.z) / E_sum);
        float s = E_sum * E_sum - ((P1.x + P2.x) * (P1.x + P2.x) + (P1.y + P2.y) * (P1.y + P2.y) + (P1.z + P2.z) * (P1.z + P2.z));
        if (s <= md2) continue;

        float v_rel = s / (2.0f * P1.w * P2.w);
        float sigma22 = (9.0f * PI * as * as) / (md2 * (1.0f + md2 / s)) * HBARC * HBARC;
        float sigma23 = 0.5f * sigma22; 
        float prob = amplification * v_rel * (sigma22 + sigma23) * dt / (dv * testpartcl);

        if (curand_uniform(&local_state) < prob) {
            if (curand_uniform(&local_state) < (sigma22 / (sigma22 + sigma23))) {
                float p_cm = sqrtf(s) / 2.0f;
                float q2 = md2 * curand_uniform(&local_state) / (1.0f - curand_uniform(&local_state) + 4.0f * md2 / s); 
                float costheta = 1.0f - 2.0f * q2 / s;
                float sintheta = sqrtf(fmaxf(0.0f, 1.0f - costheta * costheta));
                float phi = curand_uniform(&local_state) * 2.0f * PI;
                float4 P1_cm = make_float4(p_cm * sintheta * cosf(phi), p_cm * sintheta * sinf(phi), p_cm * costheta, p_cm);
                float4 P2_cm = make_float4(-P1_cm.x, -P1_cm.y, -P1_cm.z, P1_cm.w);
                float3 minus_beta = make_float3(-beta.x, -beta.y, -beta.z);
                sorted_mom[i] = boost(P1_cm, minus_beta); sorted_mom[j] = boost(P2_cm, minus_beta);
            } else {
                float p_cm = sqrtf(s) / 3.0f;
                float4 P1_cm = make_float4(p_cm, 0, 0, p_cm); float4 P2_cm = make_float4(-p_cm, 0, 0, p_cm); float4 P3_cm = make_float4(0, 0, 0, p_cm);
                float3 minus_beta = make_float3(-beta.x, -beta.y, -beta.z);
                sorted_mom[i] = boost(P1_cm, minus_beta); sorted_mom[j] = boost(P2_cm, minus_beta);
                int new_idx = atomicAdd(d_atomic_counter, 1);
                if (new_idx < max_slots) {
                    sorted_pos[new_idx] = sorted_pos[i]; sorted_mom[new_idx] = boost(P3_cm, minus_beta); sorted_is_alive[new_idx] = 1;
                }
            }
        }
    }
    rand_states[cell_idx] = local_state;
}

__global__ void collide_triplets_sampled_kernel(
    float4* sorted_pos, float4* sorted_mom, int* sorted_is_alive,
    int* cell_starts, int* cell_ends, float* cell_md2,
    curandState* rand_states,
    int max_slots, int num_cells, float dt, float dv, int testpartcl, float as) 
{
    int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_idx >= num_cells) return;
    int start = cell_starts[cell_idx]; int end = cell_ends[cell_idx];
    int n = end - start;
    if (n < 3) return;
    curandState local_state = rand_states[cell_idx + num_cells];
    int M = 5; 
    float amplification = (float)n * (n - 1) * (n - 2) / 6.0f / (float)M;
    for (int k = 0; k < M; k++) {
        int i = start + (int)(curand_uniform(&local_state) * n);
        int j = start + (int)(curand_uniform(&local_state) * n);
        int l = start + (int)(curand_uniform(&local_state) * n);
        if (i == j || j == l || i == l) continue;
        float4 P1 = sorted_mom[i]; float4 P2 = sorted_mom[j]; float4 P3 = sorted_mom[l];
        float E_sum = P1.w + P2.w + P3.w;
        float s = E_sum * E_sum - ((P1.x+P2.x+P3.x)*(P1.x+P2.x+P3.x) + (P1.y+P2.y+P3.y)*(P1.y+P2.y+P3.y) + (P1.z+P2.z+P3.z)*(P1.z+P2.z+P3.z));
        float prob32 = amplification * (1.0f / (8.0f * P1.w * P2.w * P3.w)) * (50.0f) * dt / (dv * dv * testpartcl * testpartcl);
        if (curand_uniform(&local_state) < prob32) {
            float p_cm = sqrtf(s) / 2.0f;
            float3 beta = make_float3((P1.x+P2.x+P3.x)/E_sum, (P1.y+P2.y+P3.y)/E_sum, (P1.z+P2.z+P3.z)/E_sum);
            float4 P1_cm = make_float4(p_cm, 0, 0, p_cm); float4 P2_cm = make_float4(-p_cm, 0, 0, p_cm);
            float3 minus_beta = make_float3(-beta.x, -beta.y, -beta.z);
            sorted_mom[i] = boost(P1_cm, minus_beta); sorted_mom[j] = boost(P2_cm, minus_beta);
            sorted_is_alive[l] = 0; 
        }
    }
    rand_states[cell_idx + num_cells] = local_state;
}

struct InvP { __device__ float operator()(const float4& m) const { float p = sqrtf(m.x * m.x + m.y * m.y + m.z * m.z); return (p > 1e-6f) ? (1.0f / p) : 0.0f; } };
__global__ void update_cell_md2_kernel(int* keys, float* sums, float* cell_md2, int num_found, float factor, int num_cells) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_found) { int k = keys[idx]; if (k < num_cells) cell_md2[k] = sums[idx] * factor; }
}

class BAMPS_GPU {
public:
    int max_slots; int num_cells; float dx, dv, box_size; int IX; int testpartcl; float alpha_s;
    thrust::device_vector<float4> pos, mom; thrust::device_vector<int> is_alive, cell_ids, cell_starts, cell_ends, cell_id_range;
    thrust::device_vector<float> cell_md2; thrust::device_vector<curandState> rand_states;
    int* d_atomic_counter;

    BAMPS_GPU(int initial_n, float box_sz, float cell_sz, int tp) 
        : box_size(box_sz), dx(cell_sz), testpartcl(tp), alpha_s(0.3f) {
        max_slots = initial_n * 2; IX = (int)(box_size / dx); num_cells = IX * IX * IX; dv = dx * dx * dx;
        pos.resize(max_slots); mom.resize(max_slots); is_alive.resize(max_slots, 0); cell_ids.resize(max_slots);
        cell_starts.resize(num_cells); cell_ends.resize(num_cells); cell_id_range.resize(num_cells); cell_md2.resize(num_cells, 0.0f);
        thrust::sequence(cell_id_range.begin(), cell_id_range.end());
        rand_states.resize(num_cells * 2);
        init_rand_kernel<<<(num_cells * 2 + 255) / 256, 256>>>(thrust::raw_pointer_cast(rand_states.data()), 1234ULL, num_cells * 2);
        cudaMalloc(&d_atomic_counter, sizeof(int));
    }
    ~BAMPS_GPU() { cudaFree(d_atomic_counter); }

    void compute_local_md2() {
        thrust::device_vector<float> inv_p(max_slots);
        thrust::transform(mom.begin(), mom.end(), inv_p.begin(), InvP());
        thrust::fill(cell_md2.begin(), cell_md2.end(), 0.0f);
        thrust::device_vector<int> keys(num_cells); thrust::device_vector<float> sums(num_cells);
        auto end = thrust::reduce_by_key(cell_ids.begin(), cell_ids.end(), inv_p.begin(), keys.begin(), sums.begin());
        int num_found = end.first - keys.begin();
        float factor = (16.0f * PI * alpha_s * 3.0f / (dv * testpartcl)) * HBARC * HBARC;
        update_cell_md2_kernel<<<(num_found + 255) / 256, 256>>>(thrust::raw_pointer_cast(keys.data()), thrust::raw_pointer_cast(sums.data()), thrust::raw_pointer_cast(cell_md2.data()), num_found, factor, num_cells);
    }

    void evolve(float dt) {
        int threadsPerBlock = 256;
        int blocks_s = (max_slots + threadsPerBlock - 1) / threadsPerBlock;
        int blocks_c = (num_cells + threadsPerBlock - 1) / threadsPerBlock;
        propagate_kernel<<<blocks_s, threadsPerBlock>>>(thrust::raw_pointer_cast(pos.data()), thrust::raw_pointer_cast(mom.data()), thrust::raw_pointer_cast(is_alive.data()), max_slots, dt, box_size);
        compute_cell_ids_kernel<<<blocks_s, threadsPerBlock>>>(thrust::raw_pointer_cast(pos.data()), thrust::raw_pointer_cast(is_alive.data()), thrust::raw_pointer_cast(cell_ids.data()), max_slots, dx, IX, box_size);
        auto begin = thrust::make_zip_iterator(thrust::make_tuple(pos.begin(), mom.begin(), is_alive.begin()));
        thrust::sort_by_key(cell_ids.begin(), cell_ids.end(), begin);
        thrust::lower_bound(cell_ids.begin(), cell_ids.end(), cell_id_range.begin(), cell_id_range.end(), cell_starts.begin());
        thrust::upper_bound(cell_ids.begin(), cell_ids.end(), cell_id_range.begin(), cell_id_range.end(), cell_ends.begin());
        compute_local_md2();
        int active_count = thrust::count(is_alive.begin(), is_alive.end(), 1);
        cudaMemcpy(d_atomic_counter, &active_count, sizeof(int), cudaMemcpyHostToDevice);
        collide_pairs_sampled_kernel<<<blocks_c, threadsPerBlock>>>(thrust::raw_pointer_cast(pos.data()), thrust::raw_pointer_cast(mom.data()), thrust::raw_pointer_cast(is_alive.data()), thrust::raw_pointer_cast(cell_starts.data()), thrust::raw_pointer_cast(cell_ends.data()), thrust::raw_pointer_cast(cell_md2.data()), thrust::raw_pointer_cast(rand_states.data()), d_atomic_counter, max_slots, num_cells, dt, dv, testpartcl, alpha_s);
        collide_triplets_sampled_kernel<<<blocks_c, threadsPerBlock>>>(thrust::raw_pointer_cast(pos.data()), thrust::raw_pointer_cast(mom.data()), thrust::raw_pointer_cast(is_alive.data()), thrust::raw_pointer_cast(cell_starts.data()), thrust::raw_pointer_cast(cell_ends.data()), thrust::raw_pointer_cast(cell_md2.data()), thrust::raw_pointer_cast(rand_states.data()), max_slots, num_cells, dt, dv, testpartcl, alpha_s);
    }
};

int main() {
    int initial_n = 200000; float box_size = 10.0f; float cell_size = 1.0f; int testpartcl = 200; float dt = 0.01f;
    BAMPS_GPU sim(initial_n, box_size, cell_size, testpartcl);
    thrust::host_vector<float4> h_mom(sim.max_slots, make_float4(0,0,0,0));
    thrust::host_vector<float4> h_pos(sim.max_slots, make_float4(0,0,0,0));
    thrust::host_vector<int> h_alive(sim.max_slots, 0);
    for(int i=0; i<initial_n; ++i) {
        float p_mag = 2.0f; 
        float phi = (float)rand()/RAND_MAX * 2.0f * PI;
        float costheta = (float)rand()/RAND_MAX * 2.0f - 1.0f;
        float sintheta = sqrtf(fmaxf(0.0f, 1.0f - costheta * costheta));
        h_mom[i] = make_float4(p_mag * sintheta * cosf(phi), 
                               p_mag * sintheta * sinf(phi), 
                               p_mag * costheta, 
                               p_mag);
        h_pos[i] = make_float4((float)rand()/RAND_MAX * box_size - box_size/2.0f,
                               (float)rand()/RAND_MAX * box_size - box_size/2.0f,
                               (float)rand()/RAND_MAX * box_size - box_size/2.0f,
                               0.0f);
        h_alive[i] = 1;
    }
    sim.mom = h_mom; sim.pos = h_pos; sim.is_alive = h_alive;
    printf("Starting 200k Optimized Benchmark...\n");
    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start);
    for (int i = 0; i < 1000; i++) {
        sim.evolve(dt);
        if (i % 25 == 0) { int current_n = thrust::count(sim.is_alive.begin(), sim.is_alive.end(), 1); printf("Step %d: Particles = %d\n", i, current_n); }
    }
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms = 0; cudaEventElapsedTime(&ms, start, stop);
    printf("Benchmark finished: 100 steps in %f ms (Avg: %f ms/step)\n", ms, ms / 100.0f);
    thrust::host_vector<float4> final_mom = sim.mom;
    thrust::host_vector<int> final_alive = sim.is_alive;
    FILE* f = fopen("energies.txt", "w");
    for(int i=0; i<sim.max_slots; i++) { if(final_alive[i]) fprintf(f, "%f\n", final_mom[i].w); }
    fclose(f);
    return 0;
}
