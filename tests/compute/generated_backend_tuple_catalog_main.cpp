#include "boundary_layer/scheme_descriptor.hpp"
#include "chaos/scheme_descriptor.hpp"
#include "core/infra/sha256.hpp"
#include "core/runtime/science_profile.hpp"
#include "dynamics/scheme_descriptor.hpp"
#include "microphysics/scheme_descriptor.hpp"
#include "numerics/advection/scheme_descriptor.hpp"
#include "numerics/diffusion/scheme_descriptor.hpp"
#include "numerics/time_stepping/scheme_descriptor.hpp"
#include "radiation/scheme_descriptor.hpp"
#include "soundings/scheme_descriptor.hpp"
#include "terrain/scheme_descriptor.hpp"
#include "turbulence/scheme_descriptor.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using tmv::runtime::ComputeCommitOwner;
using tmv::runtime::ScienceInteractionSelection;
using tmv::runtime::SelectedScheme;

struct Group
{
    std::size_t count = 0;
    std::string representative;
};

bool supports(const std::vector<std::string>& values, const std::string& requested)
{
    return std::find(values.begin(), values.end(), requested) != values.end() ||
           std::find(values.begin(), values.end(), "any") != values.end() ||
           std::find(values.begin(), values.end(), "not_applicable") != values.end();
}

bool descriptor_accepts(const tmv::SchemeDescriptor& descriptor,
                        bool                         qualified,
                        const std::string&           backend,
                        const std::string&           staggering,
                        bool                         host_preflight_input = false)
{
    if (!descriptor.constructible || !descriptor.runtime_selectable)
        return false;
    if (qualified && descriptor.maturity != tmv::SchemeMaturity::Qualified)
        return false;
    if (!supports(descriptor.capabilities.coordinate_systems, "cartesian") ||
        !supports(descriptor.capabilities.staggerings, staggering))
        return false;
    const bool active = !descriptor.capabilities.required_state_fields.empty() ||
                        !descriptor.capabilities.produced_state_fields.empty();
    return !active || host_preflight_input || supports(descriptor.capabilities.backends, backend);
}

std::string owner_signature(const std::string&                  grid,
                            const std::string&                  backend,
                            bool                                qualified,
                            const std::vector<SelectedScheme>&  selections,
                            const ScienceInteractionSelection& interactions)
{
    std::string result = "grid=" + grid + "|backend=" + backend +
                         "|qualified=" + (qualified ? "true" : "false");
    for (const auto& selection : selections)
    {
        const auto* descriptor = tmv::find_scheme_descriptor(*selection.descriptors, selection.id);
        if (descriptor == nullptr)
            throw std::runtime_error("ownership signature contains an unknown descriptor");
        result += "|" + selection.component + "=" +
                  tmv::runtime::compute_commit_owner_name(
                      tmv::runtime::resolve_compute_process_owner(*descriptor, backend));
    }
    const auto commits = tmv::runtime::resolve_compute_ownership(interactions);
    if (!commits.errors.empty())
        throw std::runtime_error("accepted tuple has unsupported state-commit ownership");
    result += "|commit.microphysics=" +
              std::string(tmv::runtime::compute_commit_owner_name(commits.microphysics_state_commit));
    result += "|commit.slow=" +
              std::string(tmv::runtime::compute_commit_owner_name(commits.slow_state_commit));
    result += "|commit.sgs=" +
              std::string(tmv::runtime::compute_commit_owner_name(commits.sgs_state_commit));
    return result;
}

void append_accepted(const std::string& tuple,
                     const std::string& signature,
                     bool               emit,
                     bool               emit_signatures,
                     nimbus::crypto::Sha256& digest,
                     std::map<std::string, Group>& groups,
                     std::map<std::string, std::size_t>& partitions,
                     const std::string& partition)
{
    const std::string line = tuple + '\n';
    digest.update(line.data(), line.size());
    auto& group = groups[signature];
    ++group.count;
    if (group.representative.empty())
        group.representative = tuple;
    ++partitions[partition];
    if (emit)
    {
        if (emit_signatures)
            std::cout << signature << '\t';
        std::cout << tuple << '\n';
    }
}

void json_string(const std::string& value)
{
    std::cout << '"';
    for (const char c : value)
    {
        if (c == '"' || c == '\\')
            std::cout << '\\';
        std::cout << c;
    }
    std::cout << '"';
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const bool emit = argc == 2 &&
                          (std::string(argv[1]) == "--tuples" ||
                           std::string(argv[1]) == "--tuples-with-signatures");
        const bool emit_signatures = argc == 2 && std::string(argv[1]) == "--tuples-with-signatures";
        if (argc > 2 || (argc == 2 && !emit))
            throw std::runtime_error(
                "usage: nimbus_generated_backend_tuples [--tuples|--tuples-with-signatures]");

        const auto& microphysics = tmv::get_microphysics_scheme_descriptors();
        const auto& radiation = tmv::get_radiation_scheme_descriptors();
        const auto& boundary_layer = tmv::get_boundary_layer_scheme_descriptors();
        const auto& turbulence = tmv::get_turbulence_scheme_descriptors();
        const auto& chaos = tmv::chaos::get_chaos_scheme_descriptors();
        const auto& terrain = tmv::get_terrain_scheme_descriptors();
        const auto& advection = tmv::get_advection_scheme_descriptors();
        const auto& diffusion = tmv::get_diffusion_scheme_descriptors();
        const auto& dynamics = tmv::get_dynamics_scheme_descriptors();
        const auto& time_stepping = tmv::get_time_stepping_scheme_descriptors();
        const auto& soundings = tmv::get_sounding_scheme_descriptors();
        const auto* agrid_dynamics = tmv::find_scheme_descriptor(dynamics, tmv::kCartesianAGridDynamicsId);
        const auto* cgrid_dynamics = tmv::find_scheme_descriptor(dynamics, tmv::kCartesianCGridDynamicsId);
        const auto* explicit_diffusion = tmv::find_scheme_descriptor(diffusion, "explicit");
        const auto* tvd = tmv::find_scheme_descriptor(advection, "tvd");
        const auto* split_explicit = tmv::find_scheme_descriptor(time_stepping, "split_explicit");
        const auto* automatic = tmv::find_scheme_descriptor(soundings, "auto");
        const auto* cm1 = tmv::find_scheme_descriptor(soundings, "cm1");
        const auto* sharpy = tmv::find_scheme_descriptor(soundings, "sharpy");
        if (!agrid_dynamics || !cgrid_dynamics || !explicit_diffusion || !tvd ||
            !split_explicit || !automatic || !cm1 || !sharpy)
            throw std::runtime_error("generated tuple fixed selections are incomplete");

        const std::array<std::string, 2> backends = {"cpu", "gpu"};
        const std::array<std::string, 3> collocated_turbulence_modes = {"2d", "vertical_only", "3d"};
        nimbus::crypto::Sha256 accepted_digest;
        std::map<std::string, Group> groups;
        std::map<std::string, std::size_t> partitions;
        std::size_t collocated_evaluated = 0;
        std::size_t cgrid_evaluated = 0;

        for (const auto& mp : microphysics)
            for (const auto& rad : radiation)
                for (const auto& pbl : boundary_layer)
                    for (const auto& sgs : turbulence)
                        for (const auto& stochastic : chaos)
                            for (const auto& terrain_scheme : terrain)
                                for (const auto& transport : advection)
                                    for (const auto& backend : backends)
                                        for (const auto& mode : collocated_turbulence_modes)
                                    {
                                        ++collocated_evaluated;
                                        const std::vector<SelectedScheme> selections = {
                                            {"microphysics.scheme", mp.id, &microphysics},
                                            {"radiation.scheme", rad.id, &radiation},
                                            {"boundary_layer.scheme", pbl.id, &boundary_layer},
                                            {"turbulence.scheme", sgs.id, &turbulence},
                                            {"chaos.scheme", stochastic.id, &chaos},
                                            {"terrain.scheme", terrain_scheme.id, &terrain},
                                            {"advection.scheme", transport.id, &advection},
                                            {"dynamics.scheme", agrid_dynamics->id, &dynamics},
                                            {"diffusion.scheme", explicit_diffusion->id, &diffusion},
                                            {"time.integrator", split_explicit->id, &time_stepping},
                                        };
                                        auto errors = tmv::runtime::validate_science_profile(
                                            selections, false, "cartesian", "collocated", backend);
                                        ScienceInteractionSelection interaction;
                                        interaction.boundary_layer_id = pbl.id;
                                        interaction.turbulence_id = sgs.id;
                                        interaction.turbulence_mode = mode;
                                        interaction.surface_id = "none";
                                        interaction.terrain_id = terrain_scheme.id;
                                        interaction.radiation_id = rad.id;
                                        interaction.chaos_id = stochastic.id;
                                        interaction.diffusion_id = explicit_diffusion->id;
                                        interaction.diffusion_target = "scalars";
                                        interaction.output_format = "npy_3d";
                                        interaction.staggering = "collocated";
                                        interaction.lateral_west = "rigid";
                                        interaction.lateral_east = "rigid";
                                        interaction.lateral_south = "rigid";
                                        interaction.lateral_north = "rigid";
                                        interaction.bottom_boundary = "rigid_slip";
                                        interaction.top_boundary = "rigid";
                                        interaction.backend = backend;
                                        interaction.dynamics_id = agrid_dynamics->id;
                                        interaction.microphysics_id = mp.id;
                                        const auto interaction_errors =
                                            tmv::runtime::validate_science_interactions(interaction);
                                        errors.insert(errors.end(), interaction_errors.begin(),
                                                      interaction_errors.end());
                                        const bool flat_supported =
                                            (mp.id == "none" || mp.id == "kessler" || mp.id == "lin" ||
                                             mp.id == "thompson" || mp.id == "milbrandt") &&
                                            (sgs.id == "none" || sgs.id == "smagorinsky" || sgs.id == "tke") &&
                                            terrain_scheme.id == "none" &&
                                            (rad.id == "none" || rad.id == "simple_grey" || rad.id == "rrtmg");
                                        const bool mapped_dry_supported =
                                            mp.id == "none" && rad.id == "none" && pbl.id == "none" &&
                                            sgs.id == "none" && stochastic.id == "none" &&
                                            (terrain_scheme.id == "bell" || terrain_scheme.id == "schar");
                                        const bool descriptors_accept = std::all_of(
                                            selections.begin(), selections.end(), [&](const SelectedScheme& selection)
                                            {
                                                const auto* descriptor = tmv::find_scheme_descriptor(
                                                    *selection.descriptors, selection.id);
                                                return descriptor && descriptor_accepts(
                                                    *descriptor, false, backend, "collocated");
                                            });
                                        const bool accepted = (flat_supported || mapped_dry_supported) &&
                                                              descriptors_accept && interaction_errors.empty();
                                        if (errors.empty() != accepted)
                                            throw std::runtime_error("collocated tuple oracle differs from admission");
                                        if (!accepted)
                                            continue;
                                        const std::string tuple =
                                            "grid=collocated|backend=" + backend + "|qualified=false|microphysics=" +
                                            mp.id + "|radiation=" + rad.id + "|boundary_layer=" + pbl.id +
                                            "|turbulence=" + sgs.id + "|chaos=" + stochastic.id + "|terrain=" +
                                            terrain_scheme.id + "|advection=" + transport.id + "|dynamics=" +
                                            agrid_dynamics->id + "|diffusion=explicit|time_stepping=split_explicit" +
                                            "|turbulence_mode=" + mode;
                                        append_accepted(tuple, owner_signature("collocated", backend, false,
                                                                                selections, interaction), emit,
                                                        emit_signatures,
                                                        accepted_digest, groups, partitions,
                                                        "collocated." + backend);
                                    }

        const std::array<std::string, 3> turbulence_modes = {"2d", "vertical_only", "3d"};
        const std::array<std::pair<std::string, bool>, 4> surface_profiles = {{{"none", false},
                                                                                {"none", true},
                                                                                {"slab", false},
                                                                                {"slab", true}}};
        const std::array<std::string, 2> output_formats = {"npy_3d", "zfp_high"};
        const std::array<const tmv::SchemeDescriptor*, 4> initialization_profiles = {
            nullptr, automatic, cm1, sharpy};

        for (const auto& mp : microphysics)
            for (const auto& rad : radiation)
                for (const auto& pbl : boundary_layer)
                    for (const auto& sgs : turbulence)
                        for (const auto& stochastic : chaos)
                            for (const auto& diff : diffusion)
                                for (const auto& terrain_scheme : terrain)
                                    for (const auto& backend : backends)
                                        for (const bool qualified : {false, true})
                                            for (const auto& mode : turbulence_modes)
                                                for (const auto& surface : surface_profiles)
                                                    for (const auto* sounding : initialization_profiles)
                                                        for (const auto& output_format : output_formats)
                                                        {
                                                            ++cgrid_evaluated;
                                                            std::vector<SelectedScheme> selections = {
                                                                {"microphysics.scheme", mp.id, &microphysics},
                                                                {"radiation.scheme", rad.id, &radiation},
                                                                {"boundary_layer.scheme", pbl.id, &boundary_layer},
                                                                {"turbulence.scheme", sgs.id, &turbulence},
                                                                {"chaos.scheme", stochastic.id, &chaos},
                                                                {"numerics.diffusion.scheme", diff.id, &diffusion},
                                                                {"terrain.scheme", terrain_scheme.id, &terrain},
                                                                {"dynamics.scheme", cgrid_dynamics->id, &dynamics},
                                                                {"advection.scheme", tvd->id, &advection},
                                                                {"time.integrator", split_explicit->id, &time_stepping},
                                                            };
                                                            if (sounding)
                                                                selections.push_back({
                                                                    "environment.sounding.file.scheme_id",
                                                                    sounding->id,
                                                                    &soundings});
                                                            auto errors = tmv::runtime::validate_science_profile(
                                                                selections, qualified, "cartesian", "c_grid", backend);
                                                            ScienceInteractionSelection interaction{
                                                                pbl.id, sgs.id, mode, surface.first, surface.second,
                                                                terrain_scheme.id, rad.id, false, stochastic.id, diff.id,
                                                                "scalars", true, output_format, qualified};
                                                            interaction.backend = backend;
                                                            interaction.dynamics_id = cgrid_dynamics->id;
                                                            interaction.microphysics_id = mp.id;
                                                            const auto interaction_errors =
                                                                tmv::runtime::validate_science_interactions(interaction);
                                                            errors.insert(errors.end(), interaction_errors.begin(),
                                                                          interaction_errors.end());
                                                            const bool descriptors_accept = std::all_of(
                                                                selections.begin(), selections.end(),
                                                                [&](const SelectedScheme& selection)
                                                                {
                                                                    const auto* descriptor = tmv::find_scheme_descriptor(
                                                                        *selection.descriptors, selection.id);
                                                                    return descriptor && descriptor_accepts(
                                                                        *descriptor, qualified, backend, "c_grid",
                                                                        selection.component ==
                                                                            "environment.sounding.file.scheme_id");
                                                                });
                                                            const bool accepted =
                                                                descriptors_accept && interaction_errors.empty();
                                                            if (errors.empty() != accepted)
                                                                throw std::runtime_error(
                                                                    "C-grid tuple oracle differs from admission");
                                                            if (!accepted)
                                                                continue;
                                                            const std::vector<SelectedScheme> engine_selections(
                                                                selections.begin(), selections.begin() + 10);
                                                            const std::string sounding_id = sounding ? sounding->id : "parametric";
                                                            const std::string tuple =
                                                                "grid=c_grid|backend=" + backend + "|qualified=" +
                                                                (qualified ? "true" : "false") + "|microphysics=" + mp.id +
                                                                "|radiation=" + rad.id + "|boundary_layer=" + pbl.id +
                                                                "|turbulence=" + sgs.id + "|chaos=" + stochastic.id +
                                                                "|diffusion=" + diff.id + "|terrain=" + terrain_scheme.id +
                                                                "|dynamics=" + cgrid_dynamics->id + "|advection=tvd" +
                                                                "|time_stepping=split_explicit|turbulence_mode=" + mode +
                                                                "|surface=" + surface.first + "|surface_fluxes=" +
                                                                (surface.second ? "true" : "false") + "|sounding=" + sounding_id +
                                                                "|output=" + output_format;
                                                            append_accepted(
                                                                tuple,
                                                                owner_signature("c_grid", backend, qualified,
                                                                                engine_selections, interaction),
                                                                emit, emit_signatures, accepted_digest, groups, partitions,
                                                                "c_grid." + backend + "." +
                                                                    (qualified ? "qualified" : "exploratory"));
                                                        }

        std::size_t collocated_accepted = 0;
        std::size_t cgrid_accepted      = 0;
        for (const auto& [partition, count] : partitions)
        {
            if (partition.rfind("collocated.", 0) == 0)
            {
                collocated_accepted += count;
            }
            else if (partition.rfind("c_grid.", 0) == 0)
            {
                cgrid_accepted += count;
            }
            else
            {
                throw std::runtime_error("generated backend tuple has an unknown grid partition");
            }
        }
        const std::size_t accepted = collocated_accepted + cgrid_accepted;
        std::size_t       grouped  = 0;
        for (const auto& [_, group] : groups)
        {
            if (group.count == 0 || group.representative.empty())
            {
                throw std::runtime_error("generated backend tuple has an empty ownership group");
            }
            grouped += group.count;
        }
        if (accepted == 0 || grouped != accepted || collocated_accepted > collocated_evaluated ||
            cgrid_accepted > cgrid_evaluated)
        {
            throw std::runtime_error("generated backend tuple accounting is inconsistent");
        }
        if (emit)
            return 0;

        std::cout << "{\"schema\":\"nimbus.generated-backend-tuples.v1\",\"version\":1,";
        std::cout << "\"summary\":{\"evaluated\":" << (collocated_evaluated + cgrid_evaluated)
                  << ",\"accepted\":" << accepted << ",\"collocatedEvaluated\":" << collocated_evaluated
                  << ",\"collocatedAccepted\":" << collocated_accepted << ",\"cgridEvaluated\":" << cgrid_evaluated
                  << ",\"cgridAccepted\":" << cgrid_accepted << ",\"ownershipSignatures\":" << groups.size()
                  << "},\"acceptedTupleSha256\":\"" << accepted_digest.hex_digest() << "\",\"partitions\":{";
        bool first = true;
        for (const auto& [name, count] : partitions)
        {
            if (!first)
                std::cout << ',';
            first = false;
            json_string(name);
            std::cout << ':' << count;
        }
        std::cout << "},\"ownershipGroups\":[";
        first = true;
        for (const auto& [signature, group] : groups)
        {
            if (!first)
                std::cout << ',';
            first = false;
            std::cout << "{\"signature\":";
            json_string(signature);
            std::cout << ",\"count\":" << group.count << ",\"representative\":";
            json_string(group.representative);
            std::cout << '}';
        }
        std::cout << "]}\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "generated backend tuple catalog failed: " << error.what() << '\n';
        return 1;
    }
}
