#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <type_traits>
#include "include/common.cuh"
#include "include/kernels.cuh"
#include "include/nhc.cuh"
#include "include/collisions.cuh"

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
    curandState *d_states;
    curandState *d_cell_states;
    double *d_wall_mom, *d_virial, *d_ke_sum;

    NHC_State *nhc;

    Simulation(int n_in, float box, float sig, float eps, float temp) 
        : n(n_in), box_size(box), sigma(sig), epsilon(eps), target_T(temp) 
    {
        IX = 10;
        num_cells = IX * IX * IX;
        dt = 0.002f; nu = 2.0f; // Increased coupling from 0.01f to 2.0f

        CUDA_CHECK(cudaMalloc(&d_particles, n * sizeof(Particle)));
        CUDA_CHECK(cudaMalloc(&d_forces, n * sizeof(float3)));
        CUDA_CHECK(cudaMalloc(&d_grid_indices, num_cells * MAX_PARTICLES_PER_CELL * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_grid_counts, num_cells * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_states, n * sizeof(curandState)));
        CUDA_CHECK(cudaMalloc(&d_cell_states, num_cells * sizeof(curandState)));
        CUDA_CHECK(cudaMalloc(&d_wall_mom, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_virial, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_ke_sum, sizeof(double)));

        CUDA_CHECK(cudaMemset(d_forces, 0, n * sizeof(float3)));
        CUDA_CHECK(cudaMemset(d_wall_mom, 0, sizeof(double)));
        CUDA_CHECK(cudaMemset(d_virial, 0, sizeof(double)));

        setup_rng_kernel<<<(n + 255) / 256, 256>>>(d_states, n, 1234ULL);
        setup_rng_kernel<<<(num_cells + 255) / 256, 256>>>(d_cell_states, num_cells, 5678ULL);
        CUDA_CHECK(cudaDeviceSynchronize());

        init_particles_kernel<<<(n + 255) / 256, 256>>>(d_particles, d_states, n, box_size, target_T);
        CUDA_CHECK(cudaDeviceSynchronize());

        nhc = new NHC_State(n, target_T, 0.1f);
    }

    ~Simulation() {
        cudaFree(d_particles);
        cudaFree(d_forces);
        cudaFree(d_grid_indices);
        cudaFree(d_grid_counts);
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

    void step() {
        if constexpr (std::is_same_v<Therm, GlobalScalingThermostat>) {
            double h_ke;
            CUDA_CHECK(cudaMemset(d_ke_sum, 0, sizeof(double)));
            reduce_ke_kernel<Kin><<<(n + 255) / 256, 256>>>(d_particles, n, d_ke_sum);
            CUDA_CHECK(cudaMemcpy(&h_ke, d_ke_sum, sizeof(double), cudaMemcpyDeviceToHost));
            float s = nhc->propagate((float)h_ke, dt);
            apply_global_scaling_kernel<<<(n + 255) / 256, 256>>>(d_particles, n, s);
        }

        kick_drift_kernel<Kin, Bound><<<(n + 255) / 256, 256>>>(
            d_particles, d_forces, n, dt, box_size, d_wall_mom
        );

        CUDA_CHECK(cudaMemset(d_grid_counts, 0, num_cells * sizeof(int)));
        build_grid_kernel<<<(n + 255) / 256, 256>>>(
            d_particles, d_grid_indices, d_grid_counts, n, IX, box_size
        );

        if constexpr (Coll::has_collision) {
            collision_kernel<Coll><<<(num_cells + 255) / 256, 256>>>(
                d_particles, d_grid_indices, d_grid_counts, num_cells, sigma, d_cell_states
            );
        }

        CUDA_CHECK(cudaMemset(d_forces, 0, n * sizeof(float3)));
        CUDA_CHECK(cudaMemset(d_virial, 0, sizeof(double)));
        compute_forces_kernel<Pot, Bound><<<(n + 255) / 256, 256>>>(
            d_particles, d_forces, d_grid_indices, d_grid_counts, d_virial,
            n, IX, box_size, sigma, epsilon
        );

        kick_final_kernel<Kin, Therm><<<(n + 255) / 256, 256>>>(
            d_particles, d_forces, d_states, n, dt, target_T, nu
        );
        
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

template<typename Therm>
void run_sim(int N, float box, float sig, float eps, float T_target, int steps, const char* out_prefix) {
    Simulation<ClassicalKinematics, ReflectiveWall, LennardJones, NullCollision, Therm> sim(N, box, sig, eps, T_target);
    
    std::string ts_file = std::string(out_prefix) + "_timeseries.csv";
    std::ofstream ofs(ts_file);
    ofs << "Step,T,P_virial,P_wall\n";
    
    float Area = 6.0f * box * box;
    float Vol = box * box * box;
    
    sim.reset_accumulators();
    int record_interval = 10;
    
    for (int i = 0; i < steps; ++i) {
        sim.step();
        
        if ((i + 1) % record_interval == 0) {
            double wall, vir, ke;
            sim.get_stats(wall, vir, ke);
            
            float T_meas = (float)(2.0 * ke / (3.0 * N));
            float p_tail = LennardJones::calculate_p_tail(N/Vol, eps, sig);
            
            double p_wall_inst = wall / (record_interval * sim.dt * Area);
            float p_vir_inst = (N/Vol * T_meas) + (vir / (3.0 * Vol)) + p_tail;
            
            ofs << (i + 1) << "," << T_meas << "," << p_vir_inst << "," << p_wall_inst << "\n";
            sim.reset_accumulators();
        }
    }
    ofs.close();
    
    std::string mom_file = std::string(out_prefix) + "_momenta.csv";
    std::ofstream ofsm(mom_file);
    ofsm << "p_mag\n";
    
    Particle* h_particles = new Particle[N];
    cudaMemcpy(h_particles, sim.d_particles, N * sizeof(Particle), cudaMemcpyDeviceToHost);
    for(int i=0; i<N; ++i) {
        float px = h_particles[i].mom.x;
        float py = h_particles[i].mom.y;
        float pz = h_particles[i].mom.z;
        float p_mag = sqrtf(px*px + py*py + pz*pz);
        ofsm << p_mag << "\n";
    }
    delete[] h_particles;
    std::cout << "Completed " << out_prefix << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./bamps_compare <therm_type> <out_prefix>" << std::endl;
        return 1;
    }
    
    int therm_type = atoi(argv[1]);
    const char* prefix = argv[2];
    
    float sig = 0.1f;
    float eps = 0.4f;
    float rho = 6.0f;
    float box = 10.0f;
    float T_target = 1.2f;
    int steps = 2000;
    int N = (int)(rho * box * box * box);
    
    if (therm_type == 0) run_sim<NullThermostat>(N, box, sig, eps, T_target, steps, prefix);
    else if (therm_type == 1) run_sim<AndersenThermostat>(N, box, sig, eps, T_target, steps, prefix);
    else if (therm_type == 2) run_sim<LangevinThermostat>(N, box, sig, eps, T_target, steps, prefix);
    else if (therm_type == 3) run_sim<GlobalScalingThermostat>(N, box, sig, eps, T_target, steps, prefix);
    
    return 0;
}
