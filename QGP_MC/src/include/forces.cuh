#pragma once
#include "common.cuh"

namespace qgp {

__global__ void apply_non_local_forces_kernel(
    const Particle* particles, int n, int IX, float box_size,
    const float3* cell_grad_density, const float3* cell_grad_laplacian_density,
    float3* forces, float a, float c2)
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

    int cell_idx = ix + IX * iy + IX * IX * iz;

    float3 grad_n = cell_grad_density[cell_idx];
    float3 grad_lap_n = cell_grad_laplacian_density[cell_idx];

    // F_NL = 2a * \nabla n - c2 * \nabla(\nabla^2 n)
    float3 f_nl;
    f_nl.x = 2.0f * a * grad_n.x - c2 * grad_lap_n.x;
    f_nl.y = 2.0f * a * grad_n.y - c2 * grad_lap_n.y;
    f_nl.z = 2.0f * a * grad_n.z - c2 * grad_lap_n.z;

    // Force Capping to prevent numerical instability at steep boundaries
    float f_mag = length(f_nl);
    if (f_mag > 1000.0f) {
        f_nl = f_nl * (1000.0f / f_mag);
    }

    forces[idx] = f_nl;
}

} // namespace qgp
