#include "microphysics/schemes/thompson/required_packaged_data.hpp"

#include <stdexcept>
#include <string>

namespace tmv::thompson
{
namespace
{

[[noreturn]] void throw_artifact_error(const std::string& error)
{
    throw std::runtime_error(
        error + "; run `make thompson-data` to prepare the pinned tables");
}

}

std::shared_ptr<const ActivationTable> require_packaged_activation_table(
    const std::filesystem::path& path)
{
    ActivationTableStore store;
    std::string error;
    if (!store.publish_file(
            path, error,
            ActivationTableMetadata::element_count * sizeof(float)))
        throw_artifact_error(error);
    return store.snapshot();
}

std::shared_ptr<const CollectionTables> require_packaged_collection_tables(
    const std::filesystem::path& path)
{
    CollectionTableStore store;
    std::string error;
    if (!store.publish_packaged_file(
            path, error, checked_payload_bytes()))
        throw_artifact_error(error);
    return store.snapshot();
}

std::shared_ptr<const ColdProcessTables> require_packaged_cold_process_tables(
    const std::filesystem::path& path)
{
    ColdProcessTableStore store;
    std::string error;
    if (!store.publish_packaged_file(
            path, error, checked_cold_payload_bytes()))
        throw_artifact_error(error);
    return store.snapshot();
}

}
