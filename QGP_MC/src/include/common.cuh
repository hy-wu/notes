#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <cstdio>
#include <cstdlib>

namespace qgp {

// Physics Constants
constexpr float PI = 3.14159265358979323846f;
constexpr float HBARC = 0.197327f; // GeV*fm

// Limits
constexpr int MAX_PARTICLES_PER_CELL = 2048;

// CUDA Error Checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Core Data Structures
struct Particle {
    float3 pos;
    float3 mom;
    float E; // Relativistic energy
    float3 pol; // Spin polarization vector
    int is_alive;

    // Unwrapped coordinate tracking
    int3 image_flags;
    float3 initial_pos;
};

// Math Helpers
__host__ __device__ inline float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__host__ __device__ inline float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__host__ __device__ inline float3 operator*(float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
__host__ __device__ inline float3 operator*(float s, float3 a) { return a * s; }
__host__ __device__ inline void operator+=(float3& a, float3 b) { a.x += b.x; a.y += b.y; a.z += b.z; }
__host__ __device__ inline float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__host__ __device__ inline float3 cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__host__ __device__ inline float length2(float3 a) { return dot(a, a); }
__host__ __device__ inline float length(float3 a) { return sqrtf(length2(a)); }

} // namespace qgp
