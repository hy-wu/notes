#pragma once
#include "common.cuh"

namespace bamps {

struct NullPotential {
    __device__ inline static float3 calculate_force(
        float3 ri, float3 rj, float sigma, float epsilon, 
        float box_size, double* out_virial, double* out_pe) 
    {
        return make_float3(0, 0, 0);
    }

    __host__ __device__ inline static float cutoff_radius(float sigma) {
        return 0.0f;
    }
};

struct LennardJones {
    __device__ inline static float3 calculate_force(
        float3 ri, float3 rj, float sigma, float epsilon, 
        float box_size, double* out_virial, double* out_pe) 
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
            float s2 = sigma * sigma;
            float r6inv = s2 * r2inv;
            r6inv = r6inv * r6inv * r6inv;
            
            // F = 24 * eps * [ 2*(s/r)^12 - (s/r)^6 ] / r^2 * vec(r)
            float f_mag = 24.0f * epsilon * r2inv * r6inv * (2.0f * r6inv - 1.0f);
            
            if (out_virial) *out_virial += (double)(f_mag * r2);
            if (out_pe) {
                // V(r) = 4 * eps * [ (s/r)^12 - (s/r)^6 ]
                *out_pe += (double)(4.0f * epsilon * r6inv * (r6inv - 1.0f));
            }
            
            return make_float3(f_mag * dx, f_mag * dy, f_mag * dz);
        }
        return make_float3(0, 0, 0);
    }

    __host__ __device__ inline static float cutoff_radius(float sigma) {
        return 2.5f * sigma;
    }

    __host__ __device__ inline static float calculate_p_tail(float rho, float epsilon, float sigma) {
        float rc = 2.5f * sigma;
        float s_rc = sigma / rc;
        float s_rc3 = s_rc * s_rc * s_rc;
        float s_rc9 = s_rc3 * s_rc3 * s_rc3;
        return (16.0f / 3.0f) * PI * rho * rho * epsilon * (sigma * sigma * sigma) * 
               ((2.0f / 3.0f) * s_rc9 - s_rc3);
    }

    __host__ __device__ inline static float calculate_u_tail(float rho, float epsilon, float sigma) {
        float rc = 2.5f * sigma;
        float s_rc = sigma / rc;
        float s_rc3 = s_rc * s_rc * s_rc;
        float s_rc9 = s_rc3 * s_rc3 * s_rc3;
        // Analytical tail correction for potential energy per particle
        return (8.0f / 3.0f) * PI * rho * epsilon * (sigma * sigma * sigma) *
               ((1.0f / 3.0f) * s_rc9 - s_rc3);
    }
};

} // namespace bamps
