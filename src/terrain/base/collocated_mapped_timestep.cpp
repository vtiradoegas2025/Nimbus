#include "terrain/collocated_mapped_timestep.hpp"

#include "core/infra/physical_constants.hpp"
#include "dynamics/eos_pressure.hpp"
#include "terrain/conservative_transport.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace tmv::terrain_collocated
{
namespace
{

struct CandidateFields
{
    Field3D u;
    Field3D v;
    Field3D w;
    Field3D rho;
    Field3D p;
    Field3D theta;
    Field3D qv;
    Field3D tracer;
    Field3D tke;
};

struct Sources
{
    Field3D x;
    Field3D y;
    Field3D z;
};

void copy_field(Field3D& destination, const Field3D& source)
{
    destination.resize(source.size_r(), source.size_th(), source.size_z(), 0.0f);
    std::copy_n(source.data(), source.size(), destination.data());
}

CollocatedMappedStateView view(const CandidateFields& fields)
{
    return {fields.u, fields.v, fields.w, fields.rho, fields.p,
            fields.theta, fields.qv, fields.tracer, fields.tke};
}

std::array<Field3D*, 9> mutable_fields(CollocatedMappedState state)
{
    return {&state.u, &state.v, &state.w, &state.rho, &state.p,
            &state.theta, &state.qv, &state.tracer, &state.tke};
}

std::array<const Field3D*, 9> const_fields(CollocatedMappedState state)
{
    return {&state.u, &state.v, &state.w, &state.rho, &state.p,
            &state.theta, &state.qv, &state.tracer, &state.tke};
}

void validate_distinct(const std::array<const Field3D*, 9>& fields)
{
    const std::size_t bytes = fields.front()->size() * sizeof(float);
    for (std::size_t first = 0; first < fields.size(); ++first)
        for (std::size_t second = first + 1; second < fields.size(); ++second)
        {
            const auto a = reinterpret_cast<std::uintptr_t>(fields[first]->data());
            const auto b = reinterpret_cast<std::uintptr_t>(fields[second]->data());
            const auto distance = a < b ? b - a : a - b;
            if (distance < bytes)
                throw std::invalid_argument(
                    "mapped A-grid state fields must have distinct storage");
        }
}

void validate_state(CollocatedMappedState state,
                    const TerrainMetrics3D& metrics,
                    const std::vector<double>& reference_pressure,
                    const std::vector<double>& reference_theta,
                    double dt_s)
{
    terrain_transport::validate_geometry(metrics);
    const int nx = metrics.J.size_r();
    const int ny = metrics.J.size_th();
    const int nz = metrics.J.size_z();
    if (nx < 5 || ny < 5 || nz < 5)
        throw std::invalid_argument(
            "mapped A-grid stage requires at least 5x5x5 storage");
    if (!std::isfinite(dt_s) || dt_s <= 0.0)
        throw std::invalid_argument("mapped A-grid stage requires finite dt > 0");
    if (reference_pressure.size() != static_cast<std::size_t>(nz)
        || reference_theta.size() != static_cast<std::size_t>(nz))
    {
        throw std::invalid_argument(
            "mapped A-grid stage requires full pressure and theta reference profiles");
    }
    for (std::size_t k = 0; k < reference_pressure.size(); ++k)
    {
        if (!std::isfinite(reference_pressure[k]) || reference_pressure[k] <= 0.0
            || !std::isfinite(reference_theta[k]) || reference_theta[k] <= 0.0)
        {
            throw std::invalid_argument(
                "mapped A-grid reference profiles must be positive and finite");
        }
    }
    const auto fields = const_fields(state);
    for (const Field3D* field : fields)
        if (field->size_r() != nx || field->size_th() != ny
            || field->size_z() != nz)
            throw std::invalid_argument(
                "mapped A-grid state fields must match the terrain geometry");
    validate_distinct(fields);
    for (std::size_t n = 0; n < state.rho.size(); ++n)
    {
        const float values[] = {
            state.u.data()[n], state.v.data()[n], state.w.data()[n],
            state.rho.data()[n], state.p.data()[n], state.theta.data()[n],
            state.qv.data()[n], state.tracer.data()[n], state.tke.data()[n]};
        if (!std::all_of(std::begin(values), std::end(values),
                         [](float value) { return std::isfinite(value); })
            || values[3] <= 0.0f || values[4] <= 0.0f || values[5] <= 0.0f
            || values[6] < 0.0f || values[7] < 0.0f || values[8] < 0.0f)
        {
            throw std::invalid_argument(
                "mapped A-grid state requires finite velocity, positive rho/p/theta, and nonnegative scalars");
        }
    }
}

double reference_perturbation(const Field3D& pressure,
                              const TerrainMetrics3D& metrics,
                              const std::vector<double>& reference_pressure,
                              int i, int j, int k)
{
    return static_cast<double>(pressure[i][j][k])
        - terrain_transport::reference_profile_value_fp32(
            reference_pressure, metrics.z(i, j, k), metrics.dzeta);
}

terrain_transport::PhysicalVector3 pressure_gradient(
    const Field3D& pressure,
    const TerrainMetrics3D& metrics,
    const std::vector<double>& reference_pressure,
    int i, int j, int k)
{
    const double dx = (reference_perturbation(
                           pressure, metrics, reference_pressure, i + 1, j, k)
                     - reference_perturbation(
                           pressure, metrics, reference_pressure, i - 1, j, k))
                    / (2.0 * metrics.dx);
    const double dy = (reference_perturbation(
                           pressure, metrics, reference_pressure, i, j + 1, k)
                     - reference_perturbation(
                           pressure, metrics, reference_pressure, i, j - 1, k))
                    / (2.0 * metrics.dy);
    const double dzeta = (reference_perturbation(
                              pressure, metrics, reference_pressure, i, j, k + 1)
                        - reference_perturbation(
                              pressure, metrics, reference_pressure, i, j, k - 1))
                       / (2.0 * metrics.dzeta);
    const double jacobian = metrics.J(i, j, k);
    return terrain_transport::physical_gradient_from_computational(
        dx, dy, dzeta, jacobian,
        jacobian * metrics.mx(i, j, k),
        jacobian * metrics.my(i, j, k));
}

TerrainContravariantFluxes build_mass_flux(
    const CollocatedMappedStateView& state,
    const TerrainMetrics3D& metrics,
    std::uint64_t& internal_faces)
{
    const int nx = metrics.J.size_r();
    const int ny = metrics.J.size_th();
    const int nz = metrics.J.size_z();
    TerrainContravariantFluxes flux;
    flux.x.resize(nx + 1, ny, nz, 0.0);
    flux.y.resize(nx, ny + 1, nz, 0.0);
    flux.z.resize(nx, ny, nz + 1, 0.0);
    internal_faces = 0;

    #pragma omp parallel for collapse(2)
    for (int face = 2; face <= nx - 2; ++face)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 1; k < nz - 1; ++k)
            {
                const int left = face - 1;
                const int right = face;
                const double density = 0.5 * (
                    static_cast<double>(state.rho[left][j][k])
                  + static_cast<double>(state.rho[right][j][k]));
                const double velocity = 0.5 * (
                    static_cast<double>(state.u[left][j][k])
                  + static_cast<double>(state.u[right][j][k]));
                flux.x(face, j, k) = metrics.Jx_face(face, j, k)
                    * density * velocity;
            }

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < nx - 1; ++i)
        for (int face = 2; face <= ny - 2; ++face)
            for (int k = 1; k < nz - 1; ++k)
            {
                const int south = face - 1;
                const int north = face;
                const double density = 0.5 * (
                    static_cast<double>(state.rho[i][south][k])
                  + static_cast<double>(state.rho[i][north][k]));
                const double velocity = 0.5 * (
                    static_cast<double>(state.v[i][south][k])
                  + static_cast<double>(state.v[i][north][k]));
                flux.y(i, face, k) = metrics.Jy_face(i, face, k)
                    * density * velocity;
            }

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int face = 2; face <= nz - 2; ++face)
            {
                const int lower = face - 1;
                const int upper = face;
                const double density = 0.5 * (
                    static_cast<double>(state.rho[i][j][lower])
                  + static_cast<double>(state.rho[i][j][upper]));
                const double u = 0.5 * (
                    static_cast<double>(state.u[i][j][lower])
                  + static_cast<double>(state.u[i][j][upper]));
                const double v = 0.5 * (
                    static_cast<double>(state.v[i][j][lower])
                  + static_cast<double>(state.v[i][j][upper]));
                const double w = 0.5 * (
                    static_cast<double>(state.w[i][j][lower])
                  + static_cast<double>(state.w[i][j][upper]));
                flux.z(i, j, face) = density * (
                    w + metrics.Gx_face(i, j, face) * u
                      + metrics.Gy_face(i, j, face) * v);
            }
    internal_faces = static_cast<std::uint64_t>(nx - 3)
                       * static_cast<std::uint64_t>(ny - 2)
                       * static_cast<std::uint64_t>(nz - 2)
        + static_cast<std::uint64_t>(nx - 2)
                       * static_cast<std::uint64_t>(ny - 3)
                       * static_cast<std::uint64_t>(nz - 2)
        + static_cast<std::uint64_t>(nx - 2)
                       * static_cast<std::uint64_t>(ny - 2)
                       * static_cast<std::uint64_t>(nz - 3);
    return flux;
}

double outgoing_courant(const TerrainContravariantFluxes& flux,
                        const TerrainMetrics3D& metrics,
                        const Field3D& rho,
                        int i, int j, int k,
                        double dt_s)
{
    const double outward =
        (std::max(0.0, flux.x(i + 1, j, k))
         - std::min(0.0, flux.x(i, j, k))) / metrics.dx
      + (std::max(0.0, flux.y(i, j + 1, k))
         - std::min(0.0, flux.y(i, j, k))) / metrics.dy
      + (std::max(0.0, flux.z(i, j, k + 1))
         - std::min(0.0, flux.z(i, j, k))) / metrics.dzeta;
    return dt_s * outward /
        (metrics.J(i, j, k) * static_cast<double>(rho[i][j][k]));
}

double upwind_value(double flux, const Field3D& field,
                    int low_i, int low_j, int low_k,
                    int high_i, int high_j, int high_k)
{
    return flux >= 0.0
        ? static_cast<double>(field[low_i][low_j][low_k])
        : static_cast<double>(field[high_i][high_j][high_k]);
}

double flux_divergence(const TerrainContravariantFluxes& mass_flux,
                       const TerrainMetrics3D& metrics,
                       const Field3D* transported,
                       int i, int j, int k)
{
    const auto weighted_x = [&](int face)
    {
        const double mass = mass_flux.x(face, j, k);
        if (transported == nullptr || mass == 0.0) return mass;
        return mass * upwind_value(
            mass, *transported, face - 1, j, k, face, j, k);
    };
    const auto weighted_y = [&](int face)
    {
        const double mass = mass_flux.y(i, face, k);
        if (transported == nullptr || mass == 0.0) return mass;
        return mass * upwind_value(
            mass, *transported, i, face - 1, k, i, face, k);
    };
    const auto weighted_z = [&](int face)
    {
        const double mass = mass_flux.z(i, j, face);
        if (transported == nullptr || mass == 0.0) return mass;
        return mass * upwind_value(
            mass, *transported, i, j, face - 1, i, j, face);
    };
    return (weighted_x(i + 1) - weighted_x(i)) / metrics.dx
         + (weighted_y(j + 1) - weighted_y(j)) / metrics.dy
         + (weighted_z(k + 1) - weighted_z(k)) / metrics.dzeta;
}

void apply_closure_boundaries(CandidateFields& state,
                              const TerrainMetrics3D& metrics,
                              std::uint64_t& closure_cells)
{
    const int nx = metrics.J.size_r();
    const int ny = metrics.J.size_th();
    const int nz = metrics.J.size_z();
    const std::array<Field3D*, 6> scalars = {
        &state.rho, &state.p, &state.theta, &state.qv, &state.tracer, &state.tke};

    for (int j = 0; j < ny; ++j)
        for (int k = 0; k < nz; ++k)
        {
            for (Field3D* scalar : scalars)
            {
                (*scalar)[0][j][k] = (*scalar)[1][j][k];
                (*scalar)[nx - 1][j][k] = (*scalar)[nx - 2][j][k];
            }
            state.u[0][j][k] = -state.u[1][j][k];
            state.u[nx - 1][j][k] = -state.u[nx - 2][j][k];
            state.v[0][j][k] = state.v[1][j][k];
            state.v[nx - 1][j][k] = state.v[nx - 2][j][k];
            state.w[0][j][k] = state.w[1][j][k];
            state.w[nx - 1][j][k] = state.w[nx - 2][j][k];
        }
    for (int i = 0; i < nx; ++i)
        for (int k = 0; k < nz; ++k)
        {
            for (Field3D* scalar : scalars)
            {
                (*scalar)[i][0][k] = (*scalar)[i][1][k];
                (*scalar)[i][ny - 1][k] = (*scalar)[i][ny - 2][k];
            }
            state.u[i][0][k] = state.u[i][1][k];
            state.u[i][ny - 1][k] = state.u[i][ny - 2][k];
            state.v[i][0][k] = -state.v[i][1][k];
            state.v[i][ny - 1][k] = -state.v[i][ny - 2][k];
            state.w[i][0][k] = state.w[i][1][k];
            state.w[i][ny - 1][k] = state.w[i][ny - 2][k];
        }
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
        {
            for (Field3D* scalar : scalars)
            {
                (*scalar)[i][j][0] = (*scalar)[i][j][1];
                (*scalar)[i][j][nz - 1] = (*scalar)[i][j][nz - 2];
            }
            state.u[i][j][0] = state.u[i][j][1];
            state.v[i][j][0] = state.v[i][j][1];
            const auto ground = terrain_transport::ground_surface_frame(metrics, i, j);
            const double tangent_w = ground.slope_x
                    * static_cast<double>(state.u[i][j][1])
                + ground.slope_y * static_cast<double>(state.v[i][j][1]);
            state.w[i][j][0] = static_cast<float>(
                2.0 * tangent_w - static_cast<double>(state.w[i][j][1]));
            state.u[i][j][nz - 1] = state.u[i][j][nz - 2];
            state.v[i][j][nz - 1] = state.v[i][j][nz - 2];
            state.w[i][j][nz - 1] = -state.w[i][j][nz - 2];
        }
    const std::uint64_t total = static_cast<std::uint64_t>(nx) * ny * nz;
    const std::uint64_t physical = static_cast<std::uint64_t>(nx - 2)
        * static_cast<std::uint64_t>(ny - 2)
        * static_cast<std::uint64_t>(nz - 2);
    closure_cells = total - physical;
}

long double inventory(const Field3D& rho,
                      const Field3D* scalar,
                      const TerrainMetrics3D& metrics)
{
    const int nx = metrics.J.size_r();
    const int ny = metrics.J.size_th();
    const int nz = metrics.J.size_z();
    const long double volume = static_cast<long double>(metrics.dx)
        * metrics.dy * metrics.dzeta;
    long double result = 0.0L;
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 1; k < nz - 1; ++k)
            {
                long double value = static_cast<long double>(rho[i][j][k]);
                if (scalar != nullptr)
                    value *= static_cast<long double>((*scalar)[i][j][k]);
                result += value * metrics.J(i, j, k) * volume;
            }
    return result;
}

ConservationClosure closure(double before, double after, double source = 0.0)
{
    ConservationClosure result;
    result.before = before;
    result.after = after;
    result.source = source;
    result.residual = after - before - source;
    return result;
}

void require_candidate(const CandidateFields& fields,
                       int nx, int ny, int nz)
{
    const auto candidate = view(fields);
    const std::array<const Field3D*, 9> all = {
        &candidate.u, &candidate.v, &candidate.w, &candidate.rho,
        &candidate.p, &candidate.theta, &candidate.qv, &candidate.tracer,
        &candidate.tke};
    const std::size_t cells = static_cast<std::size_t>(nx) * ny * nz;
    for (const Field3D* field : all)
        for (std::size_t n = 0; n < cells; ++n)
            if (!std::isfinite(field->data()[n]))
                throw std::runtime_error(
                    "mapped A-grid stage produced non-finite candidate state");
    for (std::size_t n = 0; n < cells; ++n)
        if (candidate.rho.data()[n] <= 0.0f
            || candidate.p.data()[n] <= 0.0f
            || candidate.theta.data()[n] <= 0.0f
            || candidate.qv.data()[n] < 0.0f
            || candidate.tracer.data()[n] < 0.0f
            || candidate.tke.data()[n] < 0.0f)
            throw std::runtime_error(
                "mapped A-grid stage produced an inadmissible candidate state");
}

} // namespace

void advance_collocated_mapped_stage(
    CollocatedMappedState state,
    const TerrainMetrics3D& metrics,
    const std::vector<double>& reference_pressure,
    const std::vector<double>& reference_theta,
    double dt_s,
    CollocatedMappedStepReport& report,
    CollocatedMappedCompletedObserver completed_observer,
    void* completed_observer_context)
{
    validate_state(state, metrics, reference_pressure, reference_theta, dt_s);
    const int nx = metrics.J.size_r();
    const int ny = metrics.J.size_th();
    const int nz = metrics.J.size_z();
    const auto live = CollocatedMappedStateView{
        state.u, state.v, state.w, state.rho, state.p,
        state.theta, state.qv, state.tracer, state.tke};

    CandidateFields candidate;
    copy_field(candidate.u, state.u);
    copy_field(candidate.v, state.v);
    copy_field(candidate.w, state.w);
    copy_field(candidate.rho, state.rho);
    copy_field(candidate.p, state.p);
    copy_field(candidate.theta, state.theta);
    copy_field(candidate.qv, state.qv);
    copy_field(candidate.tracer, state.tracer);
    copy_field(candidate.tke, state.tke);

    CollocatedMappedStepReport staged;
    staged.dt_s = dt_s;
    staged.physical_cells = static_cast<std::uint64_t>(nx - 2)
        * static_cast<std::uint64_t>(ny - 2)
        * static_cast<std::uint64_t>(nz - 2);
    TerrainContravariantFluxes mass_flux = build_mass_flux(
        live, metrics, staged.internal_mass_faces);

    double max_courant = 0.0;
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 1; k < nz - 1; ++k)
            {
                const double courant = outgoing_courant(
                    mass_flux, metrics, state.rho, i, j, k, dt_s);
                if (!std::isfinite(courant) || courant > 1.0 + 1e-12)
                    throw std::invalid_argument(
                        "mapped A-grid stage violates the conservative outgoing CFL bound");
                max_courant = std::max(max_courant, courant);
            }
    staged.max_outgoing_courant = max_courant;

    Sources source;
    source.x.resize(nx, ny, nz, 0.0f);
    source.y.resize(nx, ny, nz, 0.0f);
    source.z.resize(nx, ny, nz, 0.0f);

    int invalid = 0;
    #pragma omp parallel for collapse(2) reduction(+:invalid)
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 1; k < nz - 1; ++k)
            {
                const double jacobian = metrics.J(i, j, k);
                const double rho_old = state.rho[i][j][k];
                const double conserved_rho = jacobian * rho_old
                    - dt_s * flux_divergence(
                        mass_flux, metrics, nullptr, i, j, k);
                const double rho_new = conserved_rho / jacobian;
                if (!std::isfinite(rho_new) || rho_new <= 0.0)
                {
                    ++invalid;
                    continue;
                }
                candidate.rho[i][j][k] = static_cast<float>(rho_new);

                const std::array<std::pair<const Field3D*, Field3D*>, 4> scalars = {{
                    {&state.theta, &candidate.theta},
                    {&state.qv, &candidate.qv},
                    {&state.tracer, &candidate.tracer},
                    {&state.tke, &candidate.tke}}};
                for (const auto& scalar : scalars)
                {
                    const double conserved = jacobian * rho_old
                            * static_cast<double>((*scalar.first)[i][j][k])
                        - dt_s * flux_divergence(
                            mass_flux, metrics, scalar.first, i, j, k);
                    const double specific = conserved / (jacobian * rho_new);
                    if (!std::isfinite(specific) || specific < 0.0)
                        ++invalid;
                    else
                        (*scalar.second)[i][j][k] = static_cast<float>(specific);
                }

                const auto gradient = pressure_gradient(
                    state.p, metrics, reference_pressure, i, j, k);
                const double rho_reference =
                    terrain_transport::reference_density_fp32(
                        reference_pressure, reference_theta,
                        metrics.z(i, j, k), metrics.dzeta);
                source.x[i][j][k] = static_cast<float>(-gradient.x);
                source.y[i][j][k] = static_cast<float>(-gradient.y);
                source.z[i][j][k] = static_cast<float>(
                    -gradient.z - physical_constants::gravity_ms2
                        * (rho_old - rho_reference));

                const std::array<std::pair<const Field3D*, Field3D*>, 3> momenta = {{
                    {&state.u, &candidate.u},
                    {&state.v, &candidate.v},
                    {&state.w, &candidate.w}}};
                const float source_values[] = {
                    source.x[i][j][k], source.y[i][j][k], source.z[i][j][k]};
                for (std::size_t component = 0; component < momenta.size(); ++component)
                {
                    const double conserved = jacobian * rho_old
                            * static_cast<double>((*momenta[component].first)[i][j][k])
                        - dt_s * flux_divergence(
                            mass_flux, metrics, momenta[component].first, i, j, k)
                        + dt_s * jacobian
                            * static_cast<double>(source_values[component]);
                    const double velocity = conserved / (jacobian * rho_new);
                    if (!std::isfinite(velocity))
                        ++invalid;
                    else
                        (*momenta[component].second)[i][j][k] =
                            static_cast<float>(velocity);
                }
            }
    if (invalid != 0)
        throw std::runtime_error(
            "mapped A-grid stage produced an unrepresentable conservative update");

    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 1; k < nz - 1; ++k)
            {
                const double pressure = dynamics_eos::pressure_fp64(
                    candidate.rho[i][j][k], candidate.theta[i][j][k]);
                const float stored = static_cast<float>(pressure);
                if (!std::isfinite(stored) || stored <= 0.0f)
                    throw std::runtime_error(
                        "mapped A-grid EOS projection produced invalid pressure");
                candidate.p[i][j][k] = stored;
                ++staged.eos_projected_cells;
            }
    apply_closure_boundaries(candidate, metrics, staged.closure_cells_written);
    require_candidate(candidate, nx, ny, nz);

    staged.dry_mass = closure(
        static_cast<double>(inventory(state.rho, nullptr, metrics)),
        static_cast<double>(inventory(candidate.rho, nullptr, metrics)));
    staged.rho_theta = closure(
        static_cast<double>(inventory(state.rho, &state.theta, metrics)),
        static_cast<double>(inventory(candidate.rho, &candidate.theta, metrics)));
    staged.water_vapor = closure(
        static_cast<double>(inventory(state.rho, &state.qv, metrics)),
        static_cast<double>(inventory(candidate.rho, &candidate.qv, metrics)));
    staged.tracer = closure(
        static_cast<double>(inventory(state.rho, &state.tracer, metrics)),
        static_cast<double>(inventory(candidate.rho, &candidate.tracer, metrics)));
    staged.subgrid_tke = closure(
        static_cast<double>(inventory(state.rho, &state.tke, metrics)),
        static_cast<double>(inventory(candidate.rho, &candidate.tke, metrics)));

    const double cell_volume = metrics.dx * metrics.dy * metrics.dzeta;
    std::array<long double, 3> source_impulse{};
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 1; k < nz - 1; ++k)
            {
                const long double scale = dt_s * metrics.J(i, j, k)
                    * cell_volume;
                source_impulse[0] += source.x[i][j][k] * scale;
                source_impulse[1] += source.y[i][j][k] * scale;
                source_impulse[2] += source.z[i][j][k] * scale;
            }
    staged.x_momentum = closure(
        static_cast<double>(inventory(state.rho, &state.u, metrics)),
        static_cast<double>(inventory(candidate.rho, &candidate.u, metrics)),
        static_cast<double>(source_impulse[0]));
    staged.y_momentum = closure(
        static_cast<double>(inventory(state.rho, &state.v, metrics)),
        static_cast<double>(inventory(candidate.rho, &candidate.v, metrics)),
        static_cast<double>(source_impulse[1]));
    staged.z_momentum = closure(
        static_cast<double>(inventory(state.rho, &state.w, metrics)),
        static_cast<double>(inventory(candidate.rho, &candidate.w, metrics)),
        static_cast<double>(source_impulse[2]));
    staged.rigid_closed_boundary = true;
    staged.final_pressure_uses_canonical_fp64 = true;
    staged.complete = true;

    if (completed_observer != nullptr)
        completed_observer(
            completed_observer_context, view(candidate), mass_flux, staged);

    const std::array<const Field3D*, 9> candidate_fields = {
        &candidate.u, &candidate.v, &candidate.w, &candidate.rho, &candidate.p,
        &candidate.theta, &candidate.qv, &candidate.tracer, &candidate.tke};
    const auto destinations = mutable_fields(state);
    const std::size_t bytes = state.rho.size() * sizeof(float);
    for (std::size_t field = 0; field < destinations.size(); ++field)
        std::memcpy(destinations[field]->data(), candidate_fields[field]->data(), bytes);
    report = staged;
}

} // namespace tmv::terrain_collocated
