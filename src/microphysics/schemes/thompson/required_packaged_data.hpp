#pragma once

#include "microphysics/schemes/thompson/activation_table.hpp"
#include "microphysics/schemes/thompson/collection_tables.hpp"
#include "microphysics/schemes/thompson/cold_process_tables.hpp"

#include <filesystem>
#include <memory>

namespace tmv::thompson
{

std::shared_ptr<const ActivationTable> require_packaged_activation_table(
    const std::filesystem::path& path = ActivationTableMetadata::packaged_path);
std::shared_ptr<const CollectionTables> require_packaged_collection_tables(
    const std::filesystem::path& path = CollectionTableMetadata::packaged_path);
std::shared_ptr<const ColdProcessTables> require_packaged_cold_process_tables(
    const std::filesystem::path& path = ColdProcessTableMetadata::packaged_path);

}
