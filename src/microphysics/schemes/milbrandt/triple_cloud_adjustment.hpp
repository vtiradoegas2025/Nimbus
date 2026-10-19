#pragma once

#include <string>

namespace tmv::milbrandt_detail
{

struct TripleCloudAdjustmentCell
{
    double temperature_k = 0.0;
    double pressure_pa = 0.0;
    double dry_density_kg_m3 = 0.0;
    double vapor_per_kg = 0.0;
    double cloud_per_kg = 0.0;
    double cloud_number_per_kg = 0.0;
    double vertical_velocity_ms = 0.0;
};

struct TripleCloudAdjustmentEndpoints
{
    double vapor_per_kg = 0.0;
    double cloud_per_kg = 0.0;
    double cloud_number_per_kg = 0.0;
    double temperature_increment_k = 0.0;
    double temperature_after_k = 0.0;
    double admitted_cloud_increment_per_kg = 0.0;
    double activated_number_per_m3 = 0.0;
    bool cloud_donor_exhausted = false;
};

class TripleCloudAdjustmentResult
{
public:
    const TripleCloudAdjustmentCell& source_cell() const noexcept
    {
        return source_;
    }

    const TripleCloudAdjustmentEndpoints& endpoints() const noexcept
    {
        return endpoints_;
    }

    double duration_s() const noexcept
    {
        return duration_s_;
    }

private:
    friend bool produce_exact_triple_cloud_adjustment(
        const TripleCloudAdjustmentCell&, double,
        TripleCloudAdjustmentResult&, std::string&);

    TripleCloudAdjustmentCell source_{};
    TripleCloudAdjustmentEndpoints endpoints_{};
    double duration_s_ = 0.0;
};

bool produce_exact_triple_cloud_adjustment(
    const TripleCloudAdjustmentCell& cell,
    double duration_s,
    TripleCloudAdjustmentResult& producer,
    std::string& error);

bool apply_exact_triple_cloud_adjustment(
    const TripleCloudAdjustmentCell& cell,
    double duration_s,
    const TripleCloudAdjustmentResult& producer,
    TripleCloudAdjustmentResult& result,
    std::string& error);

} // namespace tmv::milbrandt_detail
