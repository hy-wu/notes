#include <iostream>
#include <vector>
#include <string>
#include <type_traits>
#include "include/common.cuh"
#include "include/kinematics.cuh"
#include "include/kernels.cuh"
#include "include/nhc.cuh"
#include "include/collisions.cuh"
#include "include/thermostats.cuh"

using namespace bamps;

__global__ void setup_rng_kernel(curandState* states, int n, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

template <typename Kin, typename Bound, typename Pot, typename Coll, typename Therm>
class Simulation {
public:
    int n, num_cells, IX;
    float box_size, dt, target_T, nu;
    float sigma, epsilon;

    Particle *d_particles;
    float3 *d_forces;
    int *d_grid_indices, *d_grid_counts;
    int *d_grid_overflow_count;
    curandState *d_states;
    curandState *d_cell_states;
    double *d_wall_mom, *d_virial, *d_ke_sum;

    NHC_State *nhc;

    Simulation(int n_in, float box, float sig, float eps, float temp) 
        : n(n_in), box_size(box), sigma(sig), epsilon(eps), target_T(temp) 
    {
        IX = 10;
        num_cells = IX * IX * IX;
        dt = 0.002f; nu = 0.01f;

        CUDA_CHECK(cudaMalloc(&d_particles, n * sizeof(Particle)));
        CUDA_CHECK(cudaMalloc(&d_forces, n * sizeof(float3)));
        CUDA_CHECK(cudaMalloc(&d_grid_indices, num_cells * MAX_PARTICLES_PER_CELL * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_grid_counts, num_cells * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_grid_overflow_count, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_states, n * sizeof(curandState)));
        CUDA_CHECK(cudaMalloc(&d_cell_states, num_cells * sizeof(curandState)));
        CUDA_CHECK(cudaMalloc(&d_wall_mom, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_virial, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_ke_sum, sizeof(double)));

        CUDA_CHECK(cudaMemset(d_forces, 0, n * sizeof(float3)));
        CUDA_CHECK(cudaMemset(d_wall_mom, 0, sizeof(double)));
        CUDA_CHECK(cudaMemset(d_virial, 0, sizeof(double)));
        CUDA_CHECK(cudaMemset(d_grid_overflow_count, 0, sizeof(int)));

        setup_rng_kernel<<<(n + 255) / 256, 256>>>(d_states, n, 1234ULL);
        setup_rng_kernel<<<(num_cells + 255) / 256, 256>>>(d_cell_states, num_cells, 5678ULL);
        CUDA_CHECK(cudaDeviceSynchronize());

        init_particles_kernel<<<(n + 255) / 256, 256>>>(d_particles, d_states, n, box_size, target_T);
        CUDA_CHECK(cudaDeviceSynchronize());

        int ndof = Bound::is_periodic && n > 1 ? (3 * n - 3) : (3 * n);
        nhc = new NHC_State(n, target_T, 0.1f, ndof);
    }

    ~Simulation() {
        cudaFree(d_particles);
        cudaFree(d_forces);
        cudaFree(d_grid_indices);
        cudaFree(d_grid_counts);
        cudaFree(d_grid_overflow_count);
        cudaFree(d_states);
        cudaFree(d_cell_states);
        cudaFree(d_wall_mom);
        cudaFree(d_virial);
        cudaFree(d_ke_sum);
        delete nhc;
    }

    void reset_accumulators() {
        CUDA_CHECK(cudaMemset(d_wall_mom, 0, sizeof(double)));
    }

    void apply_global_scaling_half_step(float dt_scale) {
        double h_ke;
        CUDA_CHECK(cudaMemset(d_ke_sum, 0, sizeof(double)));
        reduce_ke_kernel<Kin><<<(n + 255) / 256, 256>>>(d_particles, n, d_ke_sum);
        CUDA_CHECK(cudaMemcpy(&h_ke, d_ke_sum, sizeof(double), cudaMemcpyDeviceToHost));
        float s = nhc->propagate((float)h_ke, dt_scale);
        apply_global_scaling_kernel<<<(n + 255) / 256, 256>>>(d_particles, n, s);
    }

    void ensure_grid_capacity() {
        int h_overflow = 0;
        CUDA_CHECK(cudaMemcpy(&h_overflow, d_grid_overflow_count, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_overflow != 0) {
            std::cerr << "Grid overflow detected: " << h_overflow
                      << " particles exceeded MAX_PARTICLES_PER_CELL=" << MAX_PARTICLES_PER_CELL
                      << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    void step() {
        if constexpr (std::is_same_v<Therm, GlobalScalingThermostat>) {
            apply_global_scaling_half_step(0.5f * dt);
        }

        kick_drift_kernel<Kin, Bound><<<(n + 255) / 256, 256>>>(
            d_particles, d_forces, n, dt, box_size, d_wall_mom
        );

        CUDA_CHECK(cudaMemset(d_grid_counts, 0, num_cells * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_grid_overflow_count, 0, sizeof(int)));
        build_grid_kernel<<<(n + 255) / 256, 256>>>(
            d_particles, d_grid_indices, d_grid_counts, d_grid_overflow_count, n, IX, box_size
        );
        ensure_grid_capacity();

        if constexpr (Coll::has_collision) {
            collision_kernel<Coll><<<(num_cells + 255) / 256, 256>>>(
                d_particles, d_grid_indices, d_grid_counts, num_cells, sigma, d_cell_states
            );
        }

        CUDA_CHECK(cudaMemset(d_forces, 0, n * sizeof(float3)));
        CUDA_CHECK(cudaMemset(d_virial, 0, sizeof(double)));
        compute_forces_kernel<Pot, Bound><<<(n + 255) / 256, 256>>>(
            d_particles, d_forces, d_grid_indices, d_grid_counts, d_virial, nullptr,
            n, IX, box_size, sigma, epsilon
        );

        kick_final_kernel<Kin, Therm><<<(n + 255) / 256, 256>>>(
            d_particles, d_forces, d_states, n, dt, target_T, nu
        );

        if constexpr (std::is_same_v<Therm, GlobalScalingThermostat>) {
            apply_global_scaling_half_step(0.5f * dt);
        }
        
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void get_stats(double& wall_mom, double& virial_sum, double& ke_sum) {
        CUDA_CHECK(cudaMemcpy(&wall_mom, d_wall_mom, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&virial_sum, d_virial, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemset(d_ke_sum, 0, sizeof(double)));
        reduce_ke_kernel<Kin><<<(n + 255) / 256, 256>>>(d_particles, n, d_ke_sum);
        CUDA_CHECK(cudaMemcpy(&ke_sum, d_ke_sum, sizeof(double), cudaMemcpyDeviceToHost));
    }
};

int main(int argc, char** argv) {
    float sig = 0.1f;
    float eps = 0.4f;
    float rho = 6.0f;
    float box = 10.0f;
    float T_target = 1.2f;
    int steps = 1000;

    if (argc >= 2) sig = atof(argv[1]);
    if (argc >= 3) eps = atof(argv[2]);
    if (argc >= 4) rho = atof(argv[3]);
    if (argc >= 5) T_target = atof(argv[4]);
    if (argc >= 6) steps = atoi(argv[5]);

    int N = (int)(rho * box * box * box);
    Simulation<ClassicalKinematics, ReflectiveWall, LennardJones, NullCollision, GlobalScalingThermostat> sim(N, box, sig, eps, T_target);

    // 1. Equilibration
    int eq_steps = (int)(steps * 0.5);
    for (int i = 0; i < eq_steps; ++i) sim.step();
    sim.reset_accumulators();

    // 2. Production (Averaging over the rest)
    double acc_p_virial = 0;
    double acc_t = 0;
    int prod_steps = steps - eq_steps;

    for (int i = 0; i < prod_steps; ++i) {
        sim.step();
        double wall, vir, ke;
        sim.get_stats(wall, vir, ke);
        
        float Vol = box * box * box;
        float T_meas = (float)(2.0 * ke / (3.0 * N));
        float p_tail = LennardJones::calculate_p_tail(rho, eps, sig);
        
        acc_p_virial += (rho * T_meas) + (vir / (3.0 * Vol)) + p_tail;
        acc_t += T_meas;
    }
    
    // Independent measurement from Wall Momentum
    double final_wall_mom, dummy_v, dummy_k;
    sim.get_stats(final_wall_mom, dummy_v, dummy_k);
    
    float Area = 6.0f * box * box;
    double P_wall = final_wall_mom / (prod_steps * sim.dt * Area);
    
    std::cout << "FINAL_TEMPERATURE: " << acc_t / prod_steps << std::endl;
    std::cout << "FINAL_P_VIRIAL: " << acc_p_virial / prod_steps << std::endl;
    std::cout << "FINAL_P_WALL: " << P_wall << std::endl;

    return 0;
}
