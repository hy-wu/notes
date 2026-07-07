#pragma once
#include "common.cuh"

namespace qgp {

__global__ void compute_spin_polarization_kernel(
    Particle* particles, int n, int IX, float box_size,
    const float3* cell_grad_density, const float3* cell_grad_laplacian_density,
    const float3* cell_vorticity, const float* cell_shear,
    float C_omega, float C_shear, float C_grad, float lambda)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n || !particles[idx].is_alive) return;

    Particle& p = particles[idx];

    // Find particle's cell
    float half = box_size * 0.5f;
    int ix = (int)((p.pos.x + half) / box_size * (float)IX);
    int iy = (int)((p.pos.y + half) / box_size * (float)IX);
    int iz = (int)((p.pos.z + half) / box_size * (float)IX);
    ix = max(0, min(ix, IX - 1));
    iy = max(0, min(iy, IX - 1));
    iz = max(0, min(iz, IX - 1));

    int cell_idx = ix + IX * iy + IX * IX * iz;

    // Velocity v = p / E
    float3 v = p.mom * (1.0f / p.E);

    // 1. Vorticity contribution
    float3 omega = cell_vorticity[cell_idx];
    float3 pol_vort = omega * C_omega;

    // 2. Shear contribution
    // S contains flat 9 components: xx, xy, xz, yx, yy, yz, zx, zy, zz
    const float* S = &cell_shear[cell_idx * 9];
    float3 pol_shear;
    pol_shear.x = C_shear * (S[0] * v.x + S[1] * v.y + S[2] * v.z);
    pol_shear.y = C_shear * (S[3] * v.x + S[4] * v.y + S[5] * v.z);
    pol_shear.z = C_shear * (S[6] * v.x + S[7] * v.y + S[8] * v.z);

    // 3. Density gradient contribution (Spin Hall-like effect with two-level expansion)
    float3 grad_n = cell_grad_density[cell_idx];
    float3 grad_lap_n = cell_grad_laplacian_density[cell_idx];

    // g_eff = \nabla n - \lambda \nabla(\nabla^2 n)
    float3 g_eff;
    g_eff.x = grad_n.x - lambda * grad_lap_n.x;
    g_eff.y = grad_n.y - lambda * grad_lap_n.y;
    g_eff.z = grad_n.z - lambda * grad_lap_n.z;

    // S_gradient \propto v \times g_eff
    float3 pol_grad = cross(v, g_eff) * C_grad;

    // Total polarization vector
    p.pol = pol_vort + pol_shear + pol_grad;
}

} // namespace qgp
