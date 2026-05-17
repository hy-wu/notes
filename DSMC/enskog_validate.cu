#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "src/include/common.cuh"
#include "src/include/kernels.cuh"
#include "src/include/collisions.cuh"

using namespace bamps;

__global__ void setup_rng_kernel(curandState* states, int n, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) curand_init(seed, idx, 0, &states[idx]);
}

__global__ void init_enskog_particles_kernel(
    Particle* particles, curandState* states, int n, float box, float tx, float ty, float tz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int side = (int)ceilf(powf((float)n, 1.0f / 3.0f));
    float spacing = box / (float)side;
    float half = 0.5f * box;
    int ix = idx % side;
    int iy = (idx / side) % side;
    int iz = idx / (side * side);

    particles[idx].pos = make_float3(
        ix * spacing - half + 0.5f * spacing,
        iy * spacing - half + 0.5f * spacing,
        iz * spacing - half + 0.5f * spacing);
    particles[idx].initial_pos = particles[idx].pos;
    particles[idx].image_flags = make_int3(0, 0, 0);
    particles[idx].is_alive = 1;

    curandState local = states[idx];
    particles[idx].mom = make_float3(
        curand_normal(&local) * sqrtf(tx * MASS),
        curand_normal(&local) * sqrtf(ty * MASS),
        curand_normal(&local) * sqrtf(tz * MASS));
    particles[idx].E = 0.5f * dot(particles[idx].mom, particles[idx].mom) / MASS;
    states[idx] = local;
}

__global__ void drift_periodic_kernel(Particle* particles, int n, float dt, float box) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n || !particles[idx].is_alive) return;
    Particle& p = particles[idx];
    p.pos = p.pos + p.mom * (dt / MASS);
    PeriodicBoundary::apply(p.pos, p.mom, box, nullptr, p.image_flags);
}

__global__ void reduce_validation_stats_kernel(const Particle* particles, int n, double* stats) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n || !particles[idx].is_alive) return;
    float3 p = particles[idx].mom;
    atomicAdd(&stats[0], (double)p.x);
    atomicAdd(&stats[1], (double)p.y);
    atomicAdd(&stats[2], (double)p.z);
    atomicAdd(&stats[3], (double)(0.5f * dot(p, p) / MASS));
    atomicAdd(&stats[4], (double)(p.x * p.x / MASS));
    atomicAdd(&stats[5], (double)(p.y * p.y / MASS));
    atomicAdd(&stats[6], (double)(p.z * p.z / MASS));
}

static std::vector<double> read_stats(Particle* d_particles, int n, double* d_stats) {
    CUDA_CHECK(cudaMemset(d_stats, 0, 7 * sizeof(double)));
    reduce_validation_stats_kernel<<<(n + 255) / 256, 256>>>(d_particles, n, d_stats);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<double> h(7, 0.0);
    CUDA_CHECK(cudaMemcpy(h.data(), d_stats, 7 * sizeof(double), cudaMemcpyDeviceToHost));
    return h;
}

int main(int argc, char** argv) {
    if (argc < 10) {
        std::cerr << "Usage: enskog_validate <out_prefix> <order> <rho> <T> <diameter> <steps> <box> <IX> <dt> [anisotropy]\n";
        return 1;
    }

    const std::string prefix = argv[1];
    int order = std::atoi(argv[2]);
    float rho = std::atof(argv[3]);
    float target_T = std::atof(argv[4]);
    float diameter = std::atof(argv[5]);
    int steps = std::atoi(argv[6]);
    float box = std::atof(argv[7]);
    int IX = std::atoi(argv[8]);
    float dt = std::atof(argv[9]);
    float anisotropy = argc >= 11 ? std::atof(argv[10]) : 0.5f;

    int n = (int)std::round(rho * box * box * box);
    int num_cells = IX * IX * IX;
    float cell_volume = (box * box * box) / (float)num_cells;
    float cell_size = box / (float)IX;
    float actual_rho = (float)n / (box * box * box);
    float eta = HardSphereCollision::packing_fraction(actual_rho, diameter);

    float tx = target_T * (1.0f + 2.0f * anisotropy);
    float ty = target_T * fmaxf(0.05f, 1.0f - anisotropy);
    float tz = ty;

    Particle* d_particles = nullptr;
    int* d_grid_indices = nullptr;
    int* d_grid_counts = nullptr;
    int* d_overflow = nullptr;
    curandState* d_particle_states = nullptr;
    curandState* d_cell_states = nullptr;
    unsigned long long* d_collision_counts = nullptr;
    double* d_stats = nullptr;

    CUDA_CHECK(cudaMalloc(&d_particles, n * sizeof(Particle)));
    CUDA_CHECK(cudaMalloc(&d_grid_indices, num_cells * MAX_PARTICLES_PER_CELL * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_grid_counts, num_cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_overflow, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_particle_states, n * sizeof(curandState)));
    CUDA_CHECK(cudaMalloc(&d_cell_states, num_cells * sizeof(curandState)));
    CUDA_CHECK(cudaMalloc(&d_collision_counts, 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_stats, 7 * sizeof(double)));

    setup_rng_kernel<<<(n + 255) / 256, 256>>>(d_particle_states, n, 1234ULL);
    setup_rng_kernel<<<(num_cells + 255) / 256, 256>>>(d_cell_states, num_cells, 5678ULL);
    CUDA_CHECK(cudaDeviceSynchronize());

    init_enskog_particles_kernel<<<(n + 255) / 256, 256>>>(d_particles, d_particle_states, n, box, tx, ty, tz);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(d_collision_counts, 0, 4 * sizeof(unsigned long long)));

    auto initial = read_stats(d_particles, n, d_stats);
    int record_interval = std::max(1, steps / 100);
    std::ofstream ts(prefix + "_timeseries.csv");
    ts << "Step,Time,Tx,Ty,Tz,T,KE,Anisotropy,AcceptedCollisions,Trials,Capped\n";

    auto loop_start = std::chrono::steady_clock::now();
    for (int step = 0; step < steps; ++step) {
        drift_periodic_kernel<<<(n + 255) / 256, 256>>>(d_particles, n, dt, box);
        CUDA_CHECK(cudaMemset(d_grid_counts, 0, num_cells * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_overflow, 0, sizeof(int)));
        build_grid_kernel<<<(n + 255) / 256, 256>>>(d_particles, d_grid_indices, d_grid_counts, d_overflow, n, IX, box);
        enskog_collision_kernel<HardSphereCollision><<<(num_cells + 255) / 256, 256>>>(
            d_particles, d_grid_indices, d_grid_counts, num_cells, IX, diameter, dt, cell_volume,
            actual_rho, order, d_cell_states, d_collision_counts);
        CUDA_CHECK(cudaDeviceSynchronize());

        if ((step + 1) % record_interval == 0 || step == steps - 1) {
            auto s = read_stats(d_particles, n, d_stats);
            unsigned long long counts[4] = {0, 0, 0, 0};
            CUDA_CHECK(cudaMemcpy(counts, d_collision_counts, 4 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            double Tx = s[4] / n;
            double Ty = s[5] / n;
            double Tz = s[6] / n;
            double T = (Tx + Ty + Tz) / 3.0;
            double A = (Tx - 0.5 * (Ty + Tz)) / T;
            double KE = s[3];
            ts << (step + 1) << "," << ((step + 1) * dt) << "," << Tx << "," << Ty << "," << Tz << "," << T << ","
               << KE << "," << A << "," << counts[0] << "," << counts[2] << "," << counts[3] << "\n";
        }
    }
    auto loop_end = std::chrono::steady_clock::now();
    ts.close();

    auto final = read_stats(d_particles, n, d_stats);
    unsigned long long counts[4] = {0, 0, 0, 0};
    CUDA_CHECK(cudaMemcpy(counts, d_collision_counts, 4 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    double T_initial = (initial[4] + initial[5] + initial[6]) / (3.0 * n);
    double T_final = (final[4] + final[5] + final[6]) / (3.0 * n);
    double chi_trunc = HardSphereCollision::chi_truncated(actual_rho, diameter, order);
    double chi_inf_cs = HardSphereCollision::chi_resummed_cs(actual_rho, diameter);
    double chi_inf_ev = HardSphereCollision::chi_resummed_ev(actual_rho, diameter);
    double g_mean = 4.0 * std::sqrt(T_initial / PI);
    double theory_rate_0 = 0.5 * n * actual_rho * HardSphereCollision::cross_section(diameter) * g_mean;
    double theory_rate_trunc = theory_rate_0 * chi_trunc;
    double theory_rate_inf_cs = theory_rate_0 * chi_inf_cs;
    double theory_rate_inf_ev = theory_rate_0 * chi_inf_ev;
    double measured_rate = (double)counts[0] / (steps * dt);
    double runtime_sec = std::chrono::duration<double>(loop_end - loop_start).count();
    double mean_step_ms = 1000.0 * runtime_sec / (double)steps;
    double particle_steps_per_sec = ((double)n * (double)steps) / runtime_sec;

    std::ofstream summary(prefix + "_summary.csv");
    summary << std::setprecision(12);
    summary << "N,Box,IX,CellSize,Rho,Eta,T_initial,T_final,Diameter,Order,ChiTrunc,ChiResummedCS,ChiResummedEV,";
    summary << "TheoryRate0,TheoryRateTrunc,TheoryRateInfCS,TheoryRateInfEV,MeasuredRate,AcceptedCollisions,Trials,Capped,";
    summary << "RuntimeSec,MeanStepMs,ParticleStepsPerSec,PxInitial,PyInitial,PzInitial,PxFinal,PyFinal,PzFinal,KEInitial,KEFinal\n";
    summary << n << "," << box << "," << IX << "," << cell_size << "," << actual_rho << "," << eta << "," << T_initial << "," << T_final << "," << diameter << "," << order << ",";
    summary << chi_trunc << "," << chi_inf_cs << "," << chi_inf_ev << ",";
    summary << theory_rate_0 << "," << theory_rate_trunc << "," << theory_rate_inf_cs << "," << theory_rate_inf_ev << ",";
    summary << measured_rate << "," << counts[0] << "," << counts[2] << "," << counts[3] << ",";
    summary << runtime_sec << "," << mean_step_ms << "," << particle_steps_per_sec << ",";
    summary << initial[0] << "," << initial[1] << "," << initial[2] << "," << final[0] << "," << final[1] << "," << final[2] << ",";
    summary << initial[3] << "," << final[3] << "\n";
    summary.close();

    std::vector<Particle> h_particles(n);
    CUDA_CHECK(cudaMemcpy(h_particles.data(), d_particles, n * sizeof(Particle), cudaMemcpyDeviceToHost));
    std::ofstream energy_out(prefix + "_final_energy.csv");
    energy_out << std::setprecision(12);
    energy_out << "Particle,Energy\n";
    for (int i = 0; i < n; ++i) {
        if (!h_particles[i].is_alive) continue;
        energy_out << i << "," << h_particles[i].E << "\n";
    }
    energy_out.close();

    std::cout << "SUMMARY_FILE: " << prefix << "_summary.csv\n";
    std::cout << "TIMESERIES_FILE: " << prefix << "_timeseries.csv\n";
    std::cout << "FINAL_ENERGY_FILE: " << prefix << "_final_energy.csv\n";
    std::cout << "MEASURED_RATE: " << measured_rate << "\n";
    std::cout << "THEORY_RATE_TRUNC: " << theory_rate_trunc << "\n";
    std::cout << "THEORY_RATE_INF_CS: " << theory_rate_inf_cs << "\n";
    std::cout << "THEORY_RATE_INF_EV: " << theory_rate_inf_ev << "\n";
    std::cout << "MEAN_STEP_MS: " << mean_step_ms << "\n";
    std::cout << "PARTICLE_STEPS_PER_SEC: " << particle_steps_per_sec << "\n";

    cudaFree(d_particles);
    cudaFree(d_grid_indices);
    cudaFree(d_grid_counts);
    cudaFree(d_overflow);
    cudaFree(d_particle_states);
    cudaFree(d_cell_states);
    cudaFree(d_collision_counts);
    cudaFree(d_stats);
    return 0;
}
