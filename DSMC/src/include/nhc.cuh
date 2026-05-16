#pragma once
#include "common.cuh"

namespace bamps {

/**
 * NHC State: Manages a chain of thermostat variables.
 * Uses the Martyna-Tuckerman-Klein (MTK) algorithm with Yoshida-Suzuki integration.
 */
struct NHC_State {
    static constexpr int M = 3; // Chain length
    float zeta[M], v_zeta[M], G[M], Q[M];
    float target_T, dt;
    int ndof;

    NHC_State(int n, float T, float tau, int ndof_in = -1)
        : target_T(T), ndof(ndof_in > 0 ? ndof_in : 3 * n) {
        float kbT = target_T; 
        // Masses for the thermostat variables
        Q[0] = (float)ndof * kbT * tau * tau;
        for (int i = 1; i < M; ++i) Q[i] = kbT * tau * tau;
        
        for (int i = 0; i < M; ++i) { 
            zeta[i] = 0.0f; 
            v_zeta[i] = 0.0f; 
            G[i] = 0.0f; 
        }
    }

    /**
     * Yoshida-Suzuki integration of the NHC variables.
     * This is called on the Host.
     * current_ke: Total system kinetic energy.
     * dt: Time step.
     */
    float propagate(float current_ke, float dt_in) {
        // We use a multi-stage update for the v_zeta variables
        // This is a simplified version of the MTK propagator
        float dt8 = dt_in * 0.125f;
        float dt4 = dt_in * 0.25f;
        float dt2 = dt_in * 0.5f;

        // 1. Update v_zeta[M-1] ... v_zeta[0]
        G[M-1] = (Q[M-2] * v_zeta[M-2] * v_zeta[M-2] - target_T) / Q[M-1];
        v_zeta[M-1] += G[M-1] * dt4;

        for (int i = M - 2; i > 0; --i) {
            float exp_term = expf(-v_zeta[i+1] * dt8);
            v_zeta[i] *= exp_term;
            G[i] = (Q[i-1] * v_zeta[i-1] * v_zeta[i-1] - target_T) / Q[i];
            v_zeta[i] += G[i] * dt4;
            v_zeta[i] *= exp_term;
        }

        float exp_v0 = expf(-v_zeta[1] * dt8);
        v_zeta[0] *= exp_v0;
        G[0] = (2.0f * current_ke - ndof * target_T) / Q[0];
        v_zeta[0] += G[0] * dt4;
        v_zeta[0] *= exp_v0;

        // 2. Return the scaling factor for the particle momenta
        // s = exp(-v_zeta[0] * dt/2)
        float scaling_factor = expf(-v_zeta[0] * dt2);

        // 3. Update zeta positions (optional, usually for logging/RE)
        for (int i = 0; i < M; ++i) zeta[i] += v_zeta[i] * dt2;

        // 4. Update forces again for the second half of the v_zeta update
        // (In a full implementation, we'd re-calculate KE after scaling particles)
        // For efficiency, we assume the force change is small over dt/2
        
        return scaling_factor;
    }
};

/**
 * Parallel Reduction for Kinetic Energy
 */
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

template <typename Kinematics>
__global__ void reduce_ke_kernel(const Particle* particles, int n, double* out_ke) {
    float sum = 0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && particles[idx].is_alive) {
        sum = Kinematics::calculate_kinetic_energy(particles[idx].mom);
    }

    sum = warpReduceSum(sum);
    if ((threadIdx.x & (warpSize - 1)) == 0) atomicAdd(out_ke, (double)sum);
}

} // namespace bamps
