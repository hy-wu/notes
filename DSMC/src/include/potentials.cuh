#pragma once
#include "common.cuh"

namespace bamps {

struct NullPotential {
    __device__ inline static float3 calculate_force(
        float3 ri, float3 rj, float sigma, float epsilon, 
        float box_size, double* out_virial) 
    {
        return make_float3(0, 0, 0);
    }
};

struct LennardJones {
    __device__ inline static float3 calculate_force(
        float3 ri, float3 rj, float sigma, float epsilon, 
        float box_size, double* out_virial) 
    {
        float dx = ri.x - rj.x;
        float dy = ri.y - rj.y;
        float dz = ri.z - rj.z;

        // Minimum Image Convention (MIC)
        float half = box_size * 0.5f;
        if (dx >  half) dx -= box_size; else if (dx < -half) dx += box_size;
        if (dy >  half) dy -= box_size; else if (dy < -half) dy += box_size;
        if (dz >  half) dz -= box_size; else if (dz < -half) dz += box_size;

        float r2 = dx*dx + dy*dy + dz*dz;
        float cutoff = 2.5f * sigma;
        
        if (r2 < cutoff * cutoff && r2 > 1e-6f) {
            float r2inv = 1.0f / r2;
            float r6inv = r2inv * r2inv * r2inv;
            float s2 = sigma * sigma;
            float s6 = s2 * s2 * s2;
            
            // F = 24 * eps * [ 2*(s/r)^12 - (s/r)^6 ] / r^2 * vec(r)
            float f_mag = 24.0f * epsilon * r2inv * r6inv * s6 * (2.0f * r6inv * s6 - 1.0f);
            
            if (out_virial) *out_virial += (double)(f_mag * r2);
            
            return make_float3(f_mag * dx, f_mag * dy, f_mag * dz);
        }
        return make_float3(0, 0, 0);
    }

    /**
     * First Principles Correction: Effective attraction strength a(T)
     * based on B2 virial integral fit for LJ potential.
     */
    __host__ __device__ inline static float calculate_a_eff(float epsilon, float sigma, float T) {
        // Simple 16/9 * PI * eps * sigma^3 is the mean-field constant.
        // A more advanced fit could be added here.
        return (16.0f / 9.0f) * PI * epsilon * (sigma * sigma * sigma);
    }
    
    __host__ __device__ inline static float calculate_p_tail(float rho, float epsilon, float sigma) {
        float rc = 2.5f * sigma;
        float s_rc = sigma / rc;
        float s_rc3 = s_rc * s_rc * s_rc;
        float s_rc9 = s_rc3 * s_rc3 * s_rc3;
        // P_tail = 16/3 * PI * rho^2 * eps * sigma^3 * [ 2/3 * (sigma/rc)^9 - (sigma/rc)^3 ]
        return (16.0f / 3.0f) * PI * rho * rho * epsilon * (sigma * sigma * sigma) * 
               ((2.0f / 3.0f) * s_rc9 - s_rc3);
    }
};

} // namespace bamps
