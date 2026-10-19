#include "core/runtime/simulation_checkpoint.hpp"
#include "core/output/npy_writer.hpp"
#include "core/infra/nested_scalar_feedback.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <sstream>

int main(int argc, char** argv)
{
    try
    {
        if (argc < 3 || argc > 4 || (argc == 4 && std::string(argv[3]) != "--support-fields"))
            throw std::invalid_argument(
                "usage: nimbus_private_checkpoint_probe CHECKPOINT OUTPUT_DIRECTORY [--support-fields]");
        const bool                                       support_fields = argc == 4;
        SimulationStorageState                           state;
        tmv::checkpoint::SimulationCheckpointAttachments attachments;
        tmv::checkpoint::SimulationCheckpointIntegrity   integrity;
        std::string                                      error;
        if (!tmv::checkpoint::read_simulation_checkpoint(argv[1], state, attachments, integrity, error))
            throw std::runtime_error(error);
        const std::filesystem::path output(argv[2]);
        if (std::filesystem::exists(output) && !std::filesystem::is_empty(output))
            throw std::invalid_argument("probe output directory must be empty");
        std::filesystem::create_directories(output);
        std::ostringstream manifest;
        manifest.exceptions(std::ios::badbit | std::ios::failbit);
        manifest << std::setprecision(17) << "{\"time_s\":" << state.time_s << ",\"levels\":{";
        const auto publish = [&](const char* name, const SimulationStateView& view)
        {
            if (!view.microphysics_runtime || !view.microphysics_runtime->scheme)
                throw std::runtime_error("missing private physics owner");
            const auto fields    = view.microphysics_runtime->scheme->prognostic_scalars_for_transport();
            const auto inventory = nested_private_inventory(view);
            const auto directory = output / name;
            std::filesystem::create_directories(directory);
            manifest << std::quoted(name) << ":{";
            for (std::size_t f = 0; f < fields.size(); ++f)
            {
                const std::string scalar_name(fields[f].name);
                if (scalar_name.find_first_not_of(
                        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-") != std::string::npos)
                    throw std::runtime_error("private scalar name is not a safe artifact name");
                if (!npy::write_field3d(*fields[f].field, (directory / (scalar_name + ".npy")).string()))
                    throw std::runtime_error("private field export failed");
                if (f)
                    manifest << ',';
                manifest << std::quoted(scalar_name) << ':' << inventory[f];
            }
            if (support_fields)
            {
                const auto support = output / "support" / name;
                std::filesystem::create_directories(support);
                const std::array<const Field3D*, 8> values{&view.fields.rho,
                                                           &view.fields.qv,
                                                           &view.fields.qc,
                                                           &view.fields.qr,
                                                           &view.fields.qi,
                                                           &view.fields.qs,
                                                           &view.fields.qg,
                                                           &view.fields.qh};
                const std::array<const char*, 8>    names{"rho", "qv", "qc", "qr", "qi", "qs", "qg", "qh"};
                for (std::size_t f = 0; f < values.size(); ++f)
                    if (!npy::write_field3d(*values[f], (support / (std::string(names[f]) + ".npy")).string()))
                        throw std::runtime_error("private support field export failed");
            }
            manifest << '}';
        };
        publish("parent", make_simulation_state_view(state));
        manifest << ',';
        if (!state.nested_execution.relationship_ready)
            throw std::runtime_error("checkpoint does not contain a child");
        publish("child", make_simulation_state_view(state.nested_execution.child, state.configuration));
        manifest << "}}\n";
        const auto temporary_manifest = output / "manifest.json.tmp";
        {
            std::ofstream file(temporary_manifest);
            file.exceptions(std::ios::badbit | std::ios::failbit);
            file << manifest.str();
            file.close();
        }
        std::filesystem::rename(temporary_manifest, output / "manifest.json");
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
