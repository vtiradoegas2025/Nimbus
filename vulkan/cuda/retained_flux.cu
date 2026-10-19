extern "C" __global__ void nimbus_retained_flux(
    const float* rho, const float* u, const float* v, const float* w,
    float* flux_x, float* flux_y, float* flux_z,
    int nx, int ny, int nz, float frame_u, float frame_v, double weight)
{
    unsigned cell = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (cell >= unsigned(nx) * unsigned(ny) * unsigned(nz)) return;
    unsigned k = cell % unsigned(nz);
    unsigned j = (cell / unsigned(nz)) % unsigned(ny);
    unsigned i = cell / unsigned(ny * nz);
    if (i + 1 < unsigned(nx))
    {
        double density = __dmul_rn(0.5, __dadd_rn(double(rho[cell]), double(rho[cell + unsigned(ny * nz)])));
        double velocity = __dsub_rn(double(u[cell]), double(frame_u));
        flux_x[cell] = __double2float_rn(__fma_rn(weight, __dmul_rn(density, velocity), double(flux_x[cell])));
    }
    if (j + 1 < unsigned(ny))
    {
        double density = __dmul_rn(0.5, __dadd_rn(double(rho[cell]), double(rho[cell + unsigned(nz)])));
        double velocity = __dsub_rn(double(v[cell]), double(frame_v));
        flux_y[cell] = __double2float_rn(__fma_rn(weight, __dmul_rn(density, velocity), double(flux_y[cell])));
    }
    double density = k + 1 < unsigned(nz)
        ? __dmul_rn(0.5, __dadd_rn(double(rho[cell]), double(rho[cell + 1]))) : double(rho[cell]);
    flux_z[cell] = __double2float_rn(__fma_rn(weight, __dmul_rn(density, double(w[cell])), double(flux_z[cell])));
}
