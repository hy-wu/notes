#pragma once
#include "common.cuh"

namespace qgp {

__global__ void accumulate_cell_vars_kernel(
    const Particle* particles, int n, int IX, float box_size,
    float* cell_counts, float3* cell_momenta, float* cell_energies)
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

    atomicAdd(&cell_counts[cell_idx], 1.0f);
    atomicAdd(&cell_momenta[cell_idx].x, particles[idx].mom.x);
    atomicAdd(&cell_momenta[cell_idx].y, particles[idx].mom.y);
    atomicAdd(&cell_momenta[cell_idx].z, particles[idx].mom.z);
    atomicAdd(&cell_energies[cell_idx], particles[idx].E);
}

__global__ void normalize_cell_vars_kernel(
    int num_cells, float cell_volume,
    const float* cell_counts, const float3* cell_momenta, const float* cell_energies,
    float* cell_densities, float3* cell_velocities)
{
    int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_idx >= num_cells) return;

    float count = cell_counts[cell_idx];
    cell_densities[cell_idx] = count / cell_volume;

    float E_tot = cell_energies[cell_idx];
    if (E_tot > 1e-6f) {
        cell_velocities[cell_idx] = cell_momenta[cell_idx] * (1.0f / E_tot);
    } else {
        cell_velocities[cell_idx] = make_float3(0.0f, 0.0f, 0.0f);
    }
}

__global__ void compute_density_gradients_kernel(
    const float* cell_densities, int IX, float h,
    float3* cell_grad_density, float* cell_laplacian_density)
{
    int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_cells = IX * IX * IX;
    if (cell_idx >= num_cells) return;

    int iz = cell_idx / (IX * IX);
    int iy = (cell_idx / IX) % IX;
    int ix = cell_idx % IX;

    auto get_idx = [IX](int x, int y, int z) {
        return ((x + IX) % IX) + IX * ((y + IX) % IX) + IX * IX * ((z + IX) % IX);
    };

    float n_xp = cell_densities[get_idx(ix + 1, iy, iz)];
    float n_xm = cell_densities[get_idx(ix - 1, iy, iz)];
    float n_yp = cell_densities[get_idx(ix, iy + 1, iz)];
    float n_ym = cell_densities[get_idx(ix, iy - 1, iz)];
    float n_zp = cell_densities[get_idx(ix, iy, iz + 1)];
    float n_zm = cell_densities[get_idx(ix, iy, iz - 1)];
    float n_c = cell_densities[cell_idx];

    float3 grad;
    grad.x = (n_xp - n_xm) / (2.0f * h);
    grad.y = (n_yp - n_ym) / (2.0f * h);
    grad.z = (n_zp - n_zm) / (2.0f * h);
    cell_grad_density[cell_idx] = grad;

    float lap = (n_xp + n_xm - 2.0f * n_c) / (h * h) +
                (n_yp + n_ym - 2.0f * n_c) / (h * h) +
                (n_zp + n_zm - 2.0f * n_c) / (h * h);
    cell_laplacian_density[cell_idx] = lap;
}

__global__ void compute_second_level_gradients_kernel(
    const float* cell_laplacian_density, const float3* cell_velocities,
    int IX, float h,
    float3* cell_grad_laplacian_density, float3* cell_vorticity, float* cell_shear)
{
    int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_cells = IX * IX * IX;
    if (cell_idx >= num_cells) return;

    int iz = cell_idx / (IX * IX);
    int iy = (cell_idx / IX) % IX;
    int ix = cell_idx % IX;

    auto get_idx = [IX](int x, int y, int z) {
        return ((x + IX) % IX) + IX * ((y + IX) % IX) + IX * IX * ((z + IX) % IX);
    };

    // Density Laplacian Gradients: \nabla(\nabla^2 n)
    float l_xp = cell_laplacian_density[get_idx(ix + 1, iy, iz)];
    float l_xm = cell_laplacian_density[get_idx(ix - 1, iy, iz)];
    float l_yp = cell_laplacian_density[get_idx(ix, iy + 1, iz)];
    float l_ym = cell_laplacian_density[get_idx(ix, iy - 1, iz)];
    float l_zp = cell_laplacian_density[get_idx(ix, iy, iz + 1)];
    float l_zm = cell_laplacian_density[get_idx(ix, iy, iz - 1)];

    float3 grad_lap;
    grad_lap.x = (l_xp - l_xm) / (2.0f * h);
    grad_lap.y = (l_yp - l_ym) / (2.0f * h);
    grad_lap.z = (l_zp - l_zm) / (2.0f * h);
    cell_grad_laplacian_density[cell_idx] = grad_lap;

    // Velocity Gradients: du_i / dx_j
    float3 u_xp = cell_velocities[get_idx(ix + 1, iy, iz)];
    float3 u_xm = cell_velocities[get_idx(ix - 1, iy, iz)];
    float3 u_yp = cell_velocities[get_idx(ix, iy + 1, iz)];
    float3 u_ym = cell_velocities[get_idx(ix, iy - 1, iz)];
    float3 u_zp = cell_velocities[get_idx(ix, iy, iz + 1)];
    float3 u_zm = cell_velocities[get_idx(ix, iy, iz - 1)];

    float3 du_dx = (u_xp - u_xm) * (0.5f / h);
    float3 du_dy = (u_yp - u_ym) * (0.5f / h);
    float3 du_dz = (u_zp - u_zm) * (0.5f / h);

    // Vorticity: curl(u) = (du_z/dy - du_y/dz, du_x/dz - du_z/dx, du_y/dx - du_x/dy)
    float3 vort;
    vort.x = du_dy.z - du_dz.y;
    vort.y = du_dz.x - du_dx.z;
    vort.z = du_dx.y - du_dy.x;
    cell_vorticity[cell_idx] = vort;

    // Shear tensor: Sigma_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
    // Flat 9 components: xx, xy, xz, yx, yy, yz, zx, zy, zz
    float* S = &cell_shear[cell_idx * 9];
    S[0] = du_dx.x; // Sigma_xx
    S[1] = 0.5f * (du_dy.x + du_dx.y); // Sigma_xy
    S[2] = 0.5f * (du_dz.x + du_dx.z); // Sigma_xz
    S[3] = S[1]; // Sigma_yx
    S[4] = du_dy.y; // Sigma_yy
    S[5] = 0.5f * (du_dz.y + du_dy.z); // Sigma_yz
    S[6] = S[2]; // Sigma_zx
    S[7] = S[5]; // Sigma_zy
    S[8] = du_dz.z; // Sigma_zz
}

__global__ void smooth_density_kernel(
    const float* cell_densities_in, float* cell_densities_out, int IX)
{
    int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_cells = IX * IX * IX;
    if (cell_idx >= num_cells) return;

    int iz = cell_idx / (IX * IX);
    int iy = (cell_idx / IX) % IX;
    int ix = cell_idx % IX;

    auto get_idx = [IX](int x, int y, int z) {
        return ((x + IX) % IX) + IX * ((y + IX) % IX) + IX * IX * ((z + IX) % IX);
    };

    float n_xp = cell_densities_in[get_idx(ix + 1, iy, iz)];
    float n_xm = cell_densities_in[get_idx(ix - 1, iy, iz)];
    float n_yp = cell_densities_in[get_idx(ix, iy + 1, iz)];
    float n_ym = cell_densities_in[get_idx(ix, iy - 1, iz)];
    float n_zp = cell_densities_in[get_idx(ix, iy, iz + 1)];
    float n_zm = cell_densities_in[get_idx(ix, iy, iz - 1)];
    float n_c = cell_densities_in[cell_idx];

    // 3D 7-point stencil smoothing (sum of weights is 1.0)
    cell_densities_out[cell_idx] = 0.4f * n_c + 0.1f * (n_xp + n_xm + n_yp + n_ym + n_zp + n_zm);
}

} // namespace qgp
