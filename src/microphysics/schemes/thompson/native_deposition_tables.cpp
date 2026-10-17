#include "microphysics/schemes/thompson/native_deposition_tables.hpp"
#include "microphysics/schemes/thompson/required_packaged_data.hpp"
#include <stdexcept>
namespace tmv::thompson
{
std::shared_ptr<const nimbus::compute::ThompsonDepositionTables> native_deposition_tables()
{
    static const auto tables = []()
    {
        const auto          source = require_packaged_cold_process_tables();
        std::vector<double> fractions;
        fractions.reserve(source->ice_snow.size());
        for (const auto& cell : source->ice_snow)
            fractions.push_back(cell.ice_deposition_fraction);
        auto result = nimbus::compute::ThompsonDepositionTables::create(fractions);
        if (!result)
            throw std::runtime_error("Thompson deposition fractions are not representable");
        return result;
    }();
    return tables;
}
} // namespace tmv::thompson
