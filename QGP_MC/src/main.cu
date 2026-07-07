#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <chrono>

#include "include/common.cuh"
#include "include/kinematics.cuh"
#include "include/gradients.cuh"
#include "include/forces.cuh"
#include "include/collisions.cuh"
#include "include/polarization.cuh"

using namespace qgp;

__global__ void setup_rng_kernel(curandState* states, int n, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void reduce_relativistic_temperature_kernel(const Particle* particles, int n, float* out_sums) {
    float t_sum = 0;
    float e_sum = 0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && particles[idx].is_alive) {
        float3 p = particles[idx].mom;
        float E = particles[idx].E;
        t_sum = dot(p, p) / E;
        e_sum = E;
    }
    t_sum = warpReduceSum(t_sum);
    e_sum = warpReduceSum(e_sum);
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd(&out_sums[0], t_sum);
        atomicAdd(&out_sums[1], e_sum);
    }
}

__global__ void scale_momenta_kernel(Particle* particles, int n, float s, float mass) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && particles[idx].is_alive) {
        particles[idx].mom.x *= s;
        particles[idx].mom.y *= s;
        particles[idx].mom.z *= s;
        particles[idx].E = sqrtf(dot(particles[idx].mom, particles[idx].mom) + mass * mass);
    }
}

__global__ void init_qgp_particles_kernel(
    Particle* particles, curandState* states, int n,
    float box_size, float target_T, float mass, float shear_v)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    curandState local_state = states[idx];
    float half = box_size * 0.5f;

    float3 pos;
    if (shear_v == 0.0f) {
        pos.x = (curand_uniform(&local_state) - 0.5f) * box_size;
        pos.y = (curand_uniform(&local_state) - 0.5f) * box_size;
        pos.z = (curand_uniform(&local_state) - 0.5f) * box_size;
    } else {
        float ax = 0.45f * box_size;
        float ay = 0.22f * box_size;
        float az = 0.45f * box_size;
        while (true) {
            float x = (curand_uniform(&local_state) * 2.0f - 1.0f) * ax;
            float y = (curand_uniform(&local_state) * 2.0f - 1.0f) * ay;
            float z = (curand_uniform(&local_state) * 2.0f - 1.0f) * az;
            if ((x*x)/(ax*ax) + (y*y)/(ay*ay) + (z*z)/(az*az) <= 1.0f) {
                pos = make_float3(x, y, z);
                break;
            }
        }
    }
    particles[idx].pos = pos;
    particles[idx].initial_pos = pos;
    particles[idx].image_flags = make_int3(0, 0, 0);

    // 2. Momentum Initialization: Relativistic Thermal + Shear Flow (Vorticity)
    float sig_p = sqrtf(target_T * mass);
    float3 p_thermal = make_float3(
        curand_normal(&local_state) * sig_p,
        curand_normal(&local_state) * sig_p,
        curand_normal(&local_state) * sig_p
    );

    // Initial energy estimate
    float E_est = sqrtf(dot(p_thermal, p_thermal) + mass * mass);

    // Add collective velocity shear vy = shear_v * (x / half)
    float vy = shear_v * (pos.x / half);
    p_thermal.y += E_est * vy;

    particles[idx].mom = p_thermal;
    particles[idx].E = sqrtf(dot(p_thermal, p_thermal) + mass * mass);
    particles[idx].pol = make_float3(0.0f, 0.0f, 0.0f);
    particles[idx].is_alive = 1;

    states[idx] = local_state;
}

__global__ void kick_drift_qgp_kernel(
    Particle* particles, const float3* forces, int n,
    float dt, float box_size, float mass)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n || !particles[idx].is_alive) return;

    Particle& p = particles[idx];

    // Momentum updated by force (kick)
    p.mom = p.mom + forces[idx] * dt;
    p.E = sqrtf(dot(p.mom, p.mom) + mass * mass);

    // Position updated by velocity (drift)
    float3 v = p.mom * (1.0f / p.E);
    p.pos = p.pos + v * dt;

    // Periodic boundaries
    float half = box_size * 0.5f;
    auto wrap = [&](float& coord, int& image) {
        while (coord > half) { coord -= box_size; image++; }
        while (coord < -half) { coord += box_size; image--; }
    };
    wrap(p.pos.x, p.image_flags.x);
    wrap(p.pos.y, p.image_flags.y);
    wrap(p.pos.z, p.image_flags.z);
}

__global__ void build_grid_kernel(
    const Particle* particles, int* grid_indices, int* grid_counts,
    int* overflow_count, int n, int IX, float box_size)
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
    } else if (overflow_count) {
        atomicAdd(overflow_count, 1);
    }
}

int main(int argc, char** argv) {
    int N = 100000;
    float box_size = 12.0f;
    int IX = 12;
    float dt = 0.02f;
    int steps = 200;

    float mass = 0.5f;       // GeV (thermal mass of quarks)
    float target_T = 0.3f;   // GeV (typical QGP temperature)
    float shear_v = 0.25f;    // shear velocity at box edge (rotation/vorticity)

    float a_vdw = 0.25f;      // Van der Waals attraction parameter (phase separation)
    float c2_surf = 0.08f;    // surface tension parameter
    float C_omega = 0.20f;    // vorticity coupling
    float C_shear = 0.08f;    // shear coupling
    float C_grad = 0.35f;     // density gradient coupling
    float lambda = 0.60f;     // second-level gradient coefficient

    if (argc >= 2) steps = atoi(argv[1]);
    if (argc >= 3) lambda = atof(argv[2]);
    if (argc >= 4) {
        a_vdw = (float)atof(argv[3]);
        if (a_vdw == 0.0f) c2_surf = 0.0f;
    }
    if (argc >= 5) shear_v = (float)atof(argv[4]);

    int num_cells = IX * IX * IX;
    float cell_volume = (box_size * box_size * box_size) / (float)num_cells;
    float h = box_size / (float)IX;
    float sigma_cross = 2.0f; // fm^2 (typical QGP cross section, 20 mb)

    std::cout << "Starting QGP transport simulation..." << std::endl;
    std::cout << "Steps: " << steps << ", Lambda: " << lambda << ", Grid: " << IX << "^3" << std::endl;

    // Device Memory Allocation
    Particle* d_particles;
    float3* d_forces;
    int* d_grid_indices;
    int* d_grid_counts;
    int* d_grid_overflow_count;
    curandState* d_states;
    curandState* d_cell_states;

    float* d_cell_counts;
    float3* d_cell_momenta;
    float* d_cell_energies;
    float* d_cell_densities;
    float* d_cell_densities_temp;
    float3* d_cell_velocities;
    float3* d_cell_grad_density;
    float* d_cell_laplacian_density;
    float3* d_cell_grad_laplacian_density;
    float3* d_cell_vorticity;
    float* d_cell_shear;
    float* d_temp_sum; // Array of size 2: [0] = temp_sum, [1] = energy_sum

    CUDA_CHECK(cudaMalloc(&d_particles, N * sizeof(Particle)));
    CUDA_CHECK(cudaMalloc(&d_forces, N * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_grid_indices, num_cells * MAX_PARTICLES_PER_CELL * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_grid_counts, num_cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_grid_overflow_count, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_states, N * sizeof(curandState)));
    CUDA_CHECK(cudaMalloc(&d_cell_states, num_cells * sizeof(curandState)));

    CUDA_CHECK(cudaMalloc(&d_cell_counts, num_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cell_momenta, num_cells * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_cell_energies, num_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cell_densities, num_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cell_densities_temp, num_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cell_velocities, num_cells * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_cell_grad_density, num_cells * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_cell_laplacian_density, num_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cell_grad_laplacian_density, num_cells * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_cell_vorticity, num_cells * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_cell_shear, num_cells * 9 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_temp_sum, 2 * sizeof(float)));

    // Initialize CUDA memory
    CUDA_CHECK(cudaMemset(d_forces, 0, N * sizeof(float3)));
    CUDA_CHECK(cudaMemset(d_grid_overflow_count, 0, sizeof(int)));

    setup_rng_kernel<<<(N + 255) / 256, 256>>>(d_states, N, 42ULL);
    setup_rng_kernel<<<(num_cells + 255) / 256, 256>>>(d_cell_states, num_cells, 84ULL);
    CUDA_CHECK(cudaDeviceSynchronize());

    init_qgp_particles_kernel<<<(N + 255) / 256, 256>>>(d_particles, d_states, N, box_size, target_T, mass, shear_v);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Main Integration Loop
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < steps; ++step) {
        // 1. Kick & Drift
        kick_drift_qgp_kernel<<<(N + 255) / 256, 256>>>(d_particles, d_forces, N, dt, box_size, mass);

        // 2. Spatial Grid Build
        CUDA_CHECK(cudaMemset(d_grid_counts, 0, num_cells * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_grid_overflow_count, 0, sizeof(int)));
        build_grid_kernel<<<(N + 255) / 256, 256>>>(d_particles, d_grid_indices, d_grid_counts, d_grid_overflow_count, N, IX, box_size);

        // 3. Accumulate Cell Properties
        CUDA_CHECK(cudaMemset(d_cell_counts, 0, num_cells * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_cell_momenta, 0, num_cells * sizeof(float3)));
        CUDA_CHECK(cudaMemset(d_cell_energies, 0, num_cells * sizeof(float)));
        accumulate_cell_vars_kernel<<<(N + 255) / 256, 256>>>(d_particles, N, IX, box_size, d_cell_counts, d_cell_momenta, d_cell_energies);

        // 4. Normalize and calculate Hydrodynamic Density and Velocity
        normalize_cell_vars_kernel<<<(num_cells + 255) / 256, 256>>>(num_cells, cell_volume, d_cell_counts, d_cell_momenta, d_cell_energies, d_cell_densities, d_cell_velocities);

        // 4b. Smooth density field 3 times to suppress high-frequency Poisson statistical noise
        for (int iter = 0; iter < 3; ++iter) {
            smooth_density_kernel<<<(num_cells + 255) / 256, 256>>>(d_cell_densities, d_cell_densities_temp, IX);
            CUDA_CHECK(cudaMemcpy(d_cell_densities, d_cell_densities_temp, num_cells * sizeof(float), cudaMemcpyDeviceToDevice));
        }

        // 5. Relativistic Stochastic Collisions
        relativistic_collision_kernel<<<(num_cells + 255) / 256, 256>>>(d_particles, d_grid_indices, d_grid_counts, num_cells, sigma_cross, dt, cell_volume, mass, d_cell_states);

        // 6. Compute Density Gradients (First order and Laplacian)
        compute_density_gradients_kernel<<<(num_cells + 255) / 256, 256>>>(d_cell_densities, IX, h, d_cell_grad_density, d_cell_laplacian_density);

        // 7. Compute Second-level Density Gradients, Vorticity, and Shear Tensor
        compute_second_level_gradients_kernel<<<(num_cells + 255) / 256, 256>>>(d_cell_laplacian_density, d_cell_velocities, IX, h, d_cell_grad_laplacian_density, d_cell_vorticity, d_cell_shear);

        // 8. Apply Non-local Attractive and Interfacial Forces
        apply_non_local_forces_kernel<<<(N + 255) / 256, 256>>>(d_particles, N, IX, box_size, d_cell_grad_density, d_cell_grad_laplacian_density, d_forces, a_vdw, c2_surf);

        // 9. Compute Particle Spin Polarization Vector
        compute_spin_polarization_kernel<<<(N + 255) / 256, 256>>>(d_particles, N, IX, box_size, d_cell_grad_density, d_cell_grad_laplacian_density, d_cell_vorticity, d_cell_shear, C_omega, C_shear, C_grad, lambda);

        // 10. Relativistic Global Scaling Thermostat to maintain target_T
        CUDA_CHECK(cudaMemset(d_temp_sum, 0, 2 * sizeof(float)));
        reduce_relativistic_temperature_kernel<<<(N + 255) / 256, 256>>>(d_particles, N, d_temp_sum);
        float h_sums[2] = {0.0f, 0.0f};
        CUDA_CHECK(cudaMemcpy(h_sums, d_temp_sum, 2 * sizeof(float), cudaMemcpyDeviceToHost));
        float T_meas = h_sums[0] / (3.0f * N);
        float E_mean = h_sums[1] / N;
        bool do_print = (step < 5) || (step % 20 == 0);
        if (do_print) {
            std::cout << "Step " << step << ": Measured T = " << T_meas << " GeV, E_mean = " << E_mean << " GeV";
        }
        if (T_meas > 1e-5f) {
            float p = 1.0f - 0.5f * (mass / E_mean);
            float s = powf(target_T / T_meas, p);
            if (do_print) {
                std::cout << ", Scaling factor s = " << s << " (p = " << p << ")" << std::endl;
            }
            scale_momenta_kernel<<<(N + 255) / 256, 256>>>(d_particles, N, s, mass);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        } else if (do_print) {
            std::cout << ", No scaling" << std::endl;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end_time = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "Simulation finished in " << duration << " seconds." << std::endl;

    // Check Grid Overflow
    int h_overflow = 0;
    CUDA_CHECK(cudaMemcpy(&h_overflow, d_grid_overflow_count, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_overflow > 0) {
        std::cout << "Warning: grid overflow detected in " << h_overflow << " instances." << std::endl;
    }

    // Copy data back and output to CSV files
    std::vector<Particle> h_particles(N);
    CUDA_CHECK(cudaMemcpy(h_particles.data(), d_particles, N * sizeof(Particle), cudaMemcpyDeviceToHost));

    double test_temp_sum = 0;
    for (int i = 0; i < N; ++i) {
        float3 p = h_particles[i].mom;
        test_temp_sum += (double)dot(p, p) / (double)h_particles[i].E;
    }
    std::cout << "Host measured final temperature: " << test_temp_sum / (3.0 * N) << " GeV" << std::endl;

    std::cout << "Saving final particle states..." << std::endl;
    std::ofstream p_file("particles_final.csv");
    p_file << "x,y,z,px,py,pz,vx,vy,vz,E,pol_x,pol_y,pol_z\n";
    for (int i = 0; i < N; ++i) {
        if (!h_particles[i].is_alive) continue;
        float3 v = h_particles[i].mom * (1.0f / h_particles[i].E);
        p_file << h_particles[i].pos.x << "," << h_particles[i].pos.y << "," << h_particles[i].pos.z << ","
               << h_particles[i].mom.x << "," << h_particles[i].mom.y << "," << h_particles[i].mom.z << ","
               << v.x << "," << v.y << "," << v.z << ","
               << h_particles[i].E << ","
               << h_particles[i].pol.x << "," << h_particles[i].pol.y << "," << h_particles[i].pol.z << "\n";
    }
    p_file.close();

    std::cout << "Saving final cell states..." << std::endl;
    std::vector<float> h_densities(num_cells);
    std::vector<float3> h_velocities(num_cells);
    std::vector<float3> h_vorticity(num_cells);
    std::vector<float3> h_grad_density(num_cells);
    std::vector<float3> h_grad_lap_density(num_cells);

    CUDA_CHECK(cudaMemcpy(h_densities.data(), d_cell_densities, num_cells * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_velocities.data(), d_cell_velocities, num_cells * sizeof(float3), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vorticity.data(), d_cell_vorticity, num_cells * sizeof(float3), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_grad_density.data(), d_cell_grad_density, num_cells * sizeof(float3), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_grad_lap_density.data(), d_cell_grad_laplacian_density, num_cells * sizeof(float3), cudaMemcpyDeviceToHost));

    std::ofstream c_file("grid_final.csv");
    c_file << "cell_idx,ix,iy,iz,n,ux,uy,uz,vort_x,vort_y,vort_z,gn_x,gn_y,gn_z,gl_x,gl_y,gl_z\n";
    for (int cell_idx = 0; cell_idx < num_cells; ++cell_idx) {
        int iz = cell_idx / (IX * IX);
        int iy = (cell_idx / IX) % IX;
        int ix = cell_idx % IX;
        c_file << cell_idx << "," << ix << "," << iy << "," << iz << ","
               << h_densities[cell_idx] << ","
               << h_velocities[cell_idx].x << "," << h_velocities[cell_idx].y << "," << h_velocities[cell_idx].z << ","
               << h_vorticity[cell_idx].x << "," << h_vorticity[cell_idx].y << "," << h_vorticity[cell_idx].z << ","
               << h_grad_density[cell_idx].x << "," << h_grad_density[cell_idx].y << "," << h_grad_density[cell_idx].z << ","
               << h_grad_lap_density[cell_idx].x << "," << h_grad_lap_density[cell_idx].y << "," << h_grad_lap_density[cell_idx].z << "\n";
    }
    c_file.close();

    std::cout << "Data output complete." << std::endl;

    // Free memory
    cudaFree(d_particles);
    cudaFree(d_forces);
    cudaFree(d_grid_indices);
    cudaFree(d_grid_counts);
    cudaFree(d_grid_overflow_count);
    cudaFree(d_states);
    cudaFree(d_cell_states);
    cudaFree(d_cell_counts);
    cudaFree(d_cell_momenta);
    cudaFree(d_cell_energies);
    cudaFree(d_cell_densities);
    cudaFree(d_cell_densities_temp);
    cudaFree(d_cell_velocities);
    cudaFree(d_cell_grad_density);
    cudaFree(d_cell_laplacian_density);
    cudaFree(d_cell_grad_laplacian_density);
    cudaFree(d_cell_vorticity);
    cudaFree(d_cell_shear);
    cudaFree(d_temp_sum);

    return 0;
}
