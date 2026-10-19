#include "core/runtime/boundary_layer_checkpoint.hpp"
#include "core/runtime/checkpoint_format.hpp"
#include "boundary_layer/config_validation.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace tmv::checkpoint
{
namespace
{

constexpr std::array<std::uint8_t, 8> k_magic = {
    'N', 'I', 'M', 'B', 'L', 'P', '0', '1'};
constexpr std::uint64_t k_max_elements = wire::max_elements;
using wire::Decoder;
using wire::Encoder;

void encode_config(Encoder& out, const BoundaryLayerConfig& value)
{
    out.string(value.scheme_id);
    out.floating(value.dt_pbl);
    out.boolean(value.apply_surface_fluxes);
    out.string(value.surface_layer_id);
    out.boolean(value.clear_sky_only);
    out.boolean(value.enable_tke);
    out.string(value.pbl_top_method);
    out.floating(value.pbl_max_height);
    out.floating(value.min_ustar);
}

bool decode_config(Decoder& in, BoundaryLayerConfig& value)
{
    return in.string(value.scheme_id) && in.floating(value.dt_pbl) && in.boolean(value.apply_surface_fluxes) &&
           in.string(value.surface_layer_id) && in.boolean(value.clear_sky_only) && in.boolean(value.enable_tke) &&
           in.string(value.pbl_top_method) && in.floating(value.pbl_max_height) && in.floating(value.min_ustar);
}

void encode_surface(Encoder& out, const SurfaceConfig& value)
{
    out.floating(value.z0m);
    out.floating(value.z0h);
    out.floating(value.albedo);
    out.floating(value.emissivity);
    out.floating(value.Tsfc);
    out.floating(value.qsfc);
}

bool decode_surface(Decoder& in, SurfaceConfig& value)
{
    return in.floating(value.z0m) && in.floating(value.z0h) && in.floating(value.albedo) &&
           in.floating(value.emissivity) && in.floating(value.Tsfc) && in.floating(value.qsfc);
}

void encode_float_vector(Encoder& out, const std::vector<float>& values)
{
    out.unsigned_value<std::uint64_t>(values.size());
    for (float value : values)
        out.floating(value);
}

void encode_field(Encoder& out, const Field3D& field)
{
    out.integer(static_cast<std::int32_t>(field.size_r()));
    out.integer(static_cast<std::int32_t>(field.size_th()));
    out.integer(static_cast<std::int32_t>(field.size_z()));
    for (std::size_t index = 0; index < field.size(); ++index)
        out.floating(field.data()[index]);
}

bool decode_field(Decoder& in, Field3D& field)
{
    std::int32_t nr = 0;
    std::int32_t nth = 0;
    std::int32_t nz = 0;
    if (!in.integer(nr) || !in.integer(nth) || !in.integer(nz) ||
        nr < 0 || nth < 0 || nz < 0)
        return false;
    const std::uint64_t count = static_cast<std::uint64_t>(nr) *
        static_cast<std::uint64_t>(nth) * static_cast<std::uint64_t>(nz);
    if (count > k_max_elements || count > in.remaining() / sizeof(float))
        return false;
    field.resize(nr, nth, nz, 0.0f);
    for (std::size_t index = 0; index < field.size(); ++index)
    {
        if (!in.floating(field.data()[index])) return false;
    }
    return true;
}

bool decode_float_vector(Decoder& in, std::vector<float>& values)
{
    std::uint64_t size = 0;
    if (!in.unsigned_value(size) || size > k_max_elements ||
        size > in.remaining() / sizeof(float))
        return false;
    values.resize(static_cast<std::size_t>(size));
    for (float& value : values)
    {
        if (!in.floating(value))
            return false;
    }
    return true;
}

bool valid_shape(int nr, int nth, std::size_t size) noexcept
{
    if (nr == -1 && nth == -1)
        return size == 0;
    if (nr < 0 || nth < 0)
        return false;
    const std::uint64_t count = static_cast<std::uint64_t>(nr) * static_cast<std::uint64_t>(nth);
    return count == size;
}

bool valid_held_sources(const BoundaryLayerRuntimeState& state) noexcept
{
    const std::array<const Field3D*, 6> fields{{
        &state.held_u_force_n, &state.held_v_force_n,
        &state.held_theta_inventory_rate_kgk_s,
        &state.held_vapor_mass_rate_kg_s,
        &state.held_cloud_liquid_mass_rate_kg_s,
        &state.held_cloud_ice_mass_rate_kg_s}};
    const Field3D& first = *fields.front();
    for (const Field3D* field : fields)
    {
        if (field->size_r() != first.size_r() ||
            field->size_th() != first.size_th() ||
            field->size_z() != first.size_z())
            return false;
        for (std::size_t index = 0; index < field->size(); ++index)
        {
            if (!std::isfinite(field->data()[index])) return false;
        }
    }
    return true;
}

bool valid_momentum_work(const BoundaryLayerRuntimeState& state) noexcept
{
    const auto& work = state.last_momentum_work;
    return std::isfinite(work.requested_linear_work_j) &&
           std::isfinite(work.finite_step_correction_j) &&
           std::isfinite(work.storage_rounding_j) &&
           std::isfinite(work.committed_work_j) &&
           std::isfinite(state.cumulative_requested_momentum_work_j) &&
           std::isfinite(
               state.cumulative_finite_step_momentum_correction_j) &&
           std::isfinite(state.cumulative_momentum_storage_rounding_j) &&
           std::isfinite(state.cumulative_committed_momentum_work_j);
}

bool unwrap_boundary_layer_checkpoint(
    const std::vector<std::uint8_t>& bytes,
    std::uint32_t& version,
    std::vector<std::uint8_t>& payload,
    std::string& error)
{
    if (bytes.size() < k_magic.size() + sizeof(std::uint32_t) ||
        !std::equal(k_magic.begin(), k_magic.end(), bytes.begin()))
    {
        error = "invalid boundary-layer checkpoint magic";
        return false;
    }
    Decoder header(bytes.data() + k_magic.size(),
                   bytes.size() - k_magic.size());
    if (!header.unsigned_value(version))
    {
        error = header.error();
        return false;
    }
    if (version < 1 || version > boundary_layer_checkpoint_version)
    {
        error = "unsupported boundary-layer checkpoint version";
        return false;
    }
    return wire::unwrap(bytes, k_magic, version, payload, error,
                        "boundary-layer");
}

} // namespace

bool serialize_boundary_layer_runtime(const BoundaryLayerRuntimeState& state,
                                      std::vector<std::uint8_t>&       bytes,
                                      std::string&                     error)
{
    error.clear();
    try
    {
        validate_boundary_layer_configuration(
            state.config, state.surface_config, state.config.scheme_id);
    }
    catch (const std::exception& exception)
    {
        error = std::string("invalid boundary-layer checkpoint configuration: ")
            + exception.what();
        return false;
    }
    if (!valid_shape(state.skin_nr, state.skin_nth, state.skin_theta_2d.size()) ||
        state.skin_theta_2d.size() != state.skin_qv_2d.size())
    {
        error = "boundary-layer skin shape mismatch";
        return false;
    }
    if (!valid_shape(state.slab_nr, state.slab_nth, state.slab_columns.size()))
    {
        error = "boundary-layer slab shape mismatch";
        return false;
    }
    if (!valid_held_sources(state) || !valid_momentum_work(state))
    {
        error = "boundary-layer held-source shape or value mismatch";
        return false;
    }

    Encoder payload;
    encode_config(payload, state.config);
    encode_surface(payload, state.surface_config);
    payload.floating(state.last_update_time_s);
    payload.floating(state.next_update_time_s);
    payload.boolean(state.updated_this_step);
    encode_float_vector(payload, state.skin_theta_2d);
    encode_float_vector(payload, state.skin_qv_2d);
    payload.boolean(state.skin_captured);
    payload.integer(state.skin_nr);
    payload.integer(state.skin_nth);
    payload.unsigned_value<std::uint64_t>(state.slab_columns.size());
    for (const SlabColumnState& column : state.slab_columns)
    {
        payload.floating(column.h_m);
        payload.floating(column.theta_m_k);
        payload.floating(column.qv_m_kgkg);
        payload.boolean(column.initialized);
    }
    payload.integer(state.slab_nr);
    payload.integer(state.slab_nth);
    encode_field(payload, state.held_u_force_n);
    encode_field(payload, state.held_v_force_n);
    encode_field(payload, state.held_theta_inventory_rate_kgk_s);
    encode_field(payload, state.held_vapor_mass_rate_kg_s);
    encode_field(payload, state.held_cloud_liquid_mass_rate_kg_s);
    encode_field(payload, state.held_cloud_ice_mass_rate_kg_s);
    payload.floating(state.last_momentum_work.requested_linear_work_j);
    payload.floating(state.last_momentum_work.finite_step_correction_j);
    payload.floating(state.last_momentum_work.storage_rounding_j);
    payload.floating(state.last_momentum_work.committed_work_j);
    payload.floating(state.cumulative_requested_momentum_work_j);
    payload.floating(state.cumulative_finite_step_momentum_correction_j);
    payload.floating(state.cumulative_momentum_storage_rounding_j);
    payload.floating(state.cumulative_committed_momentum_work_j);
    payload.unsigned_value(state.momentum_work_step_count);
    if (!payload.good())
    {
        error = payload.error();
        return false;
    }
    wire::wrap(k_magic, boundary_layer_checkpoint_version,
               payload.bytes, bytes);
    return true;
}

bool deserialize_boundary_layer_runtime(const std::vector<std::uint8_t>& bytes,
                                        BoundaryLayerRuntimeState&       state,
                                        std::string&                     error)
{
    error.clear();
    std::vector<std::uint8_t> payload;
    std::uint32_t version = 0;
    if (!unwrap_boundary_layer_checkpoint(bytes, version, payload, error))
        return false;

    BoundaryLayerRuntimeState restored;
    Decoder in(payload.data(), payload.size());
    if (!decode_config(in, restored.config) || !decode_surface(in, restored.surface_config) ||
        !in.floating(restored.last_update_time_s) ||
        (version >= 2 && !in.floating(restored.next_update_time_s)) ||
        !in.boolean(restored.updated_this_step) ||
        !decode_float_vector(in, restored.skin_theta_2d) || !decode_float_vector(in, restored.skin_qv_2d) ||
        !in.boolean(restored.skin_captured))
    {
        error = in.error().empty() ? "invalid boundary-layer checkpoint payload" : in.error();
        return false;
    }
    if (version == 1)
    {
        restored.next_update_time_s =
            std::isfinite(restored.last_update_time_s)
                ? restored.last_update_time_s + restored.config.dt_pbl
                : 0.0;
    }
    std::int32_t  skin_nr    = -1;
    std::int32_t  skin_nth   = -1;
    std::uint64_t slab_count = 0;
    constexpr std::size_t encoded_slab_bytes = 3 * sizeof(double) + sizeof(std::uint8_t);
    if (!in.integer(skin_nr) || !in.integer(skin_nth) ||
        !in.unsigned_value(slab_count) || slab_count > k_max_elements ||
        slab_count > in.remaining() / encoded_slab_bytes)
    {
        error = in.error().empty() ? "invalid boundary-layer checkpoint shape" : in.error();
        return false;
    }
    restored.skin_nr  = skin_nr;
    restored.skin_nth = skin_nth;
    restored.slab_columns.resize(static_cast<std::size_t>(slab_count));
    for (SlabColumnState& column : restored.slab_columns)
    {
        if (!in.floating(column.h_m) || !in.floating(column.theta_m_k) || !in.floating(column.qv_m_kgkg) ||
            !in.boolean(column.initialized))
        {
            error = in.error();
            return false;
        }
    }
    std::int32_t slab_nr  = -1;
    std::int32_t slab_nth = -1;
    if (!in.integer(slab_nr) || !in.integer(slab_nth))
    {
        error = in.error().empty() ? "invalid boundary-layer slab shape" : in.error();
        return false;
    }
    restored.slab_nr  = slab_nr;
    restored.slab_nth = slab_nth;

    if (version >= 3)
    {
        if (!decode_field(in, restored.held_u_force_n) ||
            !decode_field(in, restored.held_v_force_n) ||
            !decode_field(in, restored.held_theta_inventory_rate_kgk_s) ||
            !decode_field(in, restored.held_vapor_mass_rate_kg_s))
        {
            error = in.error().empty()
                ? "invalid boundary-layer held-source fields" : in.error();
            return false;
        }
    }
    else if (restored.config.scheme_id != "none")
    {
        // Legacy checkpoints retained only specific tendencies in the parent
        // simulation section. They cannot define density-invariant held rates;
        // force an immediate safe resample rather than silently continuing the
        // old, state-dependent forcing.
        restored.last_update_time_s =
            std::numeric_limits<double>::quiet_NaN();
        restored.next_update_time_s = 0.0;
        restored.updated_this_step = false;
    }
    if (version >= 5)
    {
        if (!decode_field(in, restored.held_cloud_liquid_mass_rate_kg_s) ||
            !decode_field(in, restored.held_cloud_ice_mass_rate_kg_s))
        {
            error = in.error().empty()
                ? "invalid boundary-layer cloud held-source fields"
                : in.error();
            return false;
        }
    }
    else
    {
        const auto nr = restored.held_u_force_n.size_r();
        const auto nth = restored.held_u_force_n.size_th();
        const auto nz = restored.held_u_force_n.size_z();
        restored.held_cloud_liquid_mass_rate_kg_s.resize(nr, nth, nz, 0.0f);
        restored.held_cloud_ice_mass_rate_kg_s.resize(nr, nth, nz, 0.0f);
        if (!restored.config.clear_sky_only && restored.config.scheme_id != "none")
        {
            restored.last_update_time_s =
                std::numeric_limits<double>::quiet_NaN();
            restored.next_update_time_s = 0.0;
            restored.updated_this_step = false;
        }
    }
    if (version >= 4)
    {
        if (!in.floating(
                restored.last_momentum_work.requested_linear_work_j) ||
            !in.floating(
                restored.last_momentum_work.finite_step_correction_j) ||
            !in.floating(restored.last_momentum_work.storage_rounding_j) ||
            !in.floating(restored.last_momentum_work.committed_work_j) ||
            !in.floating(restored.cumulative_requested_momentum_work_j) ||
            !in.floating(
                restored.cumulative_finite_step_momentum_correction_j) ||
            !in.floating(restored.cumulative_momentum_storage_rounding_j) ||
            !in.floating(restored.cumulative_committed_momentum_work_j) ||
            !in.unsigned_value(restored.momentum_work_step_count))
        {
            error = in.error().empty()
                ? "invalid boundary-layer momentum-work ledger" : in.error();
            return false;
        }
    }
    if (!in.finished())
    {
        error = in.error().empty() ? "trailing boundary-layer checkpoint data" : in.error();
        return false;
    }

    if (!valid_shape(restored.skin_nr, restored.skin_nth, restored.skin_theta_2d.size()) ||
        restored.skin_theta_2d.size() != restored.skin_qv_2d.size() ||
        !valid_shape(restored.slab_nr, restored.slab_nth, restored.slab_columns.size()))
    {
        error = "boundary-layer checkpoint shape mismatch";
        return false;
    }
    if (!valid_held_sources(restored) || !valid_momentum_work(restored))
    {
        error = "boundary-layer held-source shape or value mismatch";
        return false;
    }

    try
    {
        validate_boundary_layer_configuration(
            restored.config, restored.surface_config,
            restored.config.scheme_id);
        restored.scheme = create_boundary_layer_scheme(restored.config.scheme_id);
        restored.scheme->initialize(restored.config);
    }
    catch (const std::exception& exception)
    {
        error = std::string("cannot restore boundary-layer scheme: ") + exception.what();
        return false;
    }
    state = std::move(restored);
    return true;
}

bool write_boundary_layer_checkpoint(const std::filesystem::path&     path,
                                     const BoundaryLayerRuntimeState& state,
                                     std::string&                     error)
{
    std::vector<std::uint8_t> bytes;
    if (!serialize_boundary_layer_runtime(state, bytes, error))
        return false;
    return wire::write_atomic(path, bytes, error, "boundary-layer");
}

bool read_boundary_layer_checkpoint(const std::filesystem::path& path,
                                    BoundaryLayerRuntimeState&   state,
                                    std::string&                 error)
{
    std::vector<std::uint8_t> bytes;
    if (!wire::read_file(path, bytes, error, "boundary-layer")) return false;
    return deserialize_boundary_layer_runtime(bytes, state, error);
}

} // namespace tmv::checkpoint
