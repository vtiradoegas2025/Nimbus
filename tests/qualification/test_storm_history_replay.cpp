#include "catch2/catch.hpp"
#include "core/infra/domain_motion.hpp"
#include "diagnostics/storm_tracker.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

struct HistoryRow
{
    std::string case_id;
    double      time_s = 0.0;
    std::string detection_status;
    std::size_t candidate_count = 0;
    bool        overflow = false;
    double      x0_m = 0.0;
    double      y0_m = 0.0;
    double      x1_m = 0.0;
    double      y1_m = 0.0;
    std::string expected_continuity;
    std::string expected_continuity_reason;
    std::string expected_movement;
    std::string expected_movement_reason;
    int         expected_i_cells = 0;
    int         expected_j_cells = 0;
};

std::vector<std::string> split_csv(const std::string& line)
{
    std::vector<std::string> values;
    std::stringstream        stream(line);
    std::string              value;
    while (std::getline(stream, value, ','))
        values.push_back(value);
    return values;
}

std::vector<HistoryRow> load_history()
{
    const std::filesystem::path path =
        std::filesystem::path(__FILE__).parent_path() / "fixtures" / "storm_history_synthetic.csv";
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("cannot open stored storm history: " + path.string());

    constexpr const char* expected_header =
        "case_id,time_s,detection_status,candidate_count,overflow,x0_m,y0_m,x1_m,y1_m,"
        "expected_continuity,expected_continuity_reason,expected_movement,expected_movement_reason,"
        "expected_i_cells,expected_j_cells";
    bool                    header_seen = false;
    std::vector<HistoryRow> rows;
    std::string             line;
    while (std::getline(input, line))
    {
        if (line.empty() || line.front() == '#')
            continue;
        if (!header_seen)
        {
            if (line != expected_header)
                throw std::runtime_error("stored storm history has an unsupported header");
            header_seen = true;
            continue;
        }
        const auto values = split_csv(line);
        if (values.size() != 15)
            throw std::runtime_error("stored storm history row does not contain 15 fields");
        HistoryRow row;
        row.case_id                    = values[0];
        row.time_s                     = std::stod(values[1]);
        row.detection_status           = values[2];
        row.candidate_count            = static_cast<std::size_t>(std::stoul(values[3]));
        row.overflow                   = std::stoi(values[4]) != 0;
        row.x0_m                       = std::stod(values[5]);
        row.y0_m                       = std::stod(values[6]);
        row.x1_m                       = std::stod(values[7]);
        row.y1_m                       = std::stod(values[8]);
        row.expected_continuity        = values[9];
        row.expected_continuity_reason = values[10];
        row.expected_movement          = values[11];
        row.expected_movement_reason   = values[12];
        row.expected_i_cells           = std::stoi(values[13]);
        row.expected_j_cells           = std::stoi(values[14]);
        rows.push_back(row);
    }
    if (!header_seen || rows.empty())
        throw std::runtime_error("stored storm history is empty");
    return rows;
}

tmv::diagnostics::StormDetectionStatus detection_status(const std::string& name)
{
    if (name == "weak")
        return tmv::diagnostics::StormDetectionStatus::Weak;
    if (name == "detected")
        return tmv::diagnostics::StormDetectionStatus::Detected;
    if (name == "ambiguous")
        return tmv::diagnostics::StormDetectionStatus::Ambiguous;
    throw std::runtime_error("unsupported stored detection status: " + name);
}

tmv::diagnostics::StormFeatureDetection detection_from(const HistoryRow& row, std::size_t row_index)
{
    using namespace tmv::diagnostics;
    if (row.candidate_count > 2)
        throw std::runtime_error("stored storm history exceeds its two-candidate schema");
    StormFeatureDetection detection;
    detection.status                       = detection_status(row.detection_status);
    detection.candidate_component_count    = row.candidate_count;
    detection.reported_branch_candidate_count = row.candidate_count;
    detection.branch_candidate_overflow    = row.overflow;
    detection.confidence = row.candidate_count == 0 ? 0.0 : 1.0 / static_cast<double>(row.candidate_count);
    constexpr double west_face_m           = 0.0;
    constexpr double east_face_m           = 30000.0;
    constexpr double south_face_m          = 0.0;
    constexpr double north_face_m          = 20000.0;
    constexpr double envelope_half_width_m = 500.0;
    const double x[2] = {row.x0_m, row.x1_m};
    const double y[2] = {row.y0_m, row.y1_m};
    for (std::size_t rank = 0; rank < row.candidate_count; ++rank)
    {
        auto& candidate             = detection.branch_candidates[rank];
        candidate.component_label   = 1000 + row_index * 2 + rank;
        candidate.component_cells   = 16;
        candidate.core_cells        = 8;
        candidate.score_fraction    = detection.confidence;
        candidate.maximum_updraft_ms = 25.0;
        candidate.condensate_mass_kg = 1.0e9;
        candidate.centroid_x_m       = x[rank];
        candidate.centroid_y_m       = y[rank];
        candidate.core_centroid_x_m  = x[rank];
        candidate.core_centroid_y_m  = y[rank];
        candidate.envelope_west_m    = x[rank] - envelope_half_width_m;
        candidate.envelope_east_m    = x[rank] + envelope_half_width_m;
        candidate.envelope_south_m   = y[rank] - envelope_half_width_m;
        candidate.envelope_north_m   = y[rank] + envelope_half_width_m;
        candidate.distance_west_m    = candidate.envelope_west_m - west_face_m;
        candidate.distance_east_m    = east_face_m - candidate.envelope_east_m;
        candidate.distance_south_m   = candidate.envelope_south_m - south_face_m;
        candidate.distance_north_m   = north_face_m - candidate.envelope_north_m;
    }
    if (row.candidate_count > 0)
    {
        const auto& primary            = detection.branch_candidates[0];
        detection.core_centroid_x_m    = primary.core_centroid_x_m;
        detection.core_centroid_y_m    = primary.core_centroid_y_m;
        detection.envelope_west_m      = primary.envelope_west_m;
        detection.envelope_east_m      = primary.envelope_east_m;
        detection.envelope_south_m     = primary.envelope_south_m;
        detection.envelope_north_m     = primary.envelope_north_m;
        detection.distance_west_m      = primary.distance_west_m;
        detection.distance_east_m      = primary.distance_east_m;
        detection.distance_south_m     = primary.distance_south_m;
        detection.distance_north_m     = primary.distance_north_m;
    }
    return detection;
}

} // namespace

TEST_CASE("stored synthetic storm history drives continuity and movement authority",
          "[qualification][storm_history][domain_motion]")
{
    using namespace tmv::diagnostics;
    const auto rows = load_history();
    REQUIRE(rows.size() == 9);

    StormContinuityThresholds continuity_policy;
    continuity_policy.minimum_samples          = 3;
    continuity_policy.lost_after_misses        = 2;
    continuity_policy.maximum_speed_ms         = 40.0;
    continuity_policy.maximum_acceleration_ms2 = 0.25;
    continuity_policy.position_uncertainty_m   = 100.0;

    tmv::domain::StormWindowMovementPolicy movement_policy;
    movement_policy.inner_margin_m       = 6500.0;
    movement_policy.release_margin_m     = 7000.0;
    movement_policy.dx_m                 = 500.0;
    movement_policy.dy_m                 = 500.0;
    movement_policy.maximum_shift_cells  = 12;
    movement_policy.cooldown_s           = 120.0;
    movement_policy.minimum_confidence   = 0.8;
    movement_policy.confirmation_samples = 2;

    StormContinuityState                 continuity;
    tmv::domain::StormWindowMovementState movement;
    for (std::size_t index = 0; index < rows.size(); ++index)
    {
        const auto& row       = rows[index];
        const auto  detection = detection_from(row, index);
        update_storm_continuity(continuity, detection, row.time_s, continuity_policy);
        INFO("stored history row " << index << " at time " << row.time_s
                                    << ", continuity_confidence=" << continuity.continuity_confidence);
        REQUIRE(std::string(storm_continuity_status_name(continuity.status)) == row.expected_continuity);
        REQUIRE(std::string(storm_continuity_reason_name(continuity.reason)) == row.expected_continuity_reason);

        const bool selected = continuity.selected_candidate_available &&
                              continuity.selected_candidate_rank < detection.reported_branch_candidate_count;
        const auto* candidate = selected ? &detection.branch_candidates[continuity.selected_candidate_rank] : nullptr;
        const tmv::domain::StormWindowMovementEvidence evidence{
            row.time_s,
            selected,
            continuity.status == StormContinuityStatus::Tracking,
            continuity.motion_available,
            continuity.status == StormContinuityStatus::Ambiguous ||
                continuity.reason == StormContinuityReason::SpatialAmbiguity,
            detection.branch_candidate_overflow ||
                continuity.reason == StormContinuityReason::BranchCandidateOverflow,
            continuity.topology_reorganization,
            continuity.continuity_confidence,
            candidate != nullptr ? candidate->distance_west_m : 0.0,
            candidate != nullptr ? candidate->distance_east_m : 0.0,
            candidate != nullptr ? candidate->distance_south_m : 0.0,
            candidate != nullptr ? candidate->distance_north_m : 0.0,
            continuity.velocity_x_ms,
            continuity.velocity_y_ms,
        };
        const auto decision = tmv::domain::evaluate_storm_window_movement(movement, evidence, movement_policy);
        INFO("movement status=" << tmv::domain::storm_window_movement_status_name(decision.status)
                                 << ", reason=" << tmv::domain::storm_window_movement_reason_name(decision.reason)
                                 << ", i=" << decision.requested_i_shift_cells
                                 << ", j=" << decision.requested_j_shift_cells);
        REQUIRE(std::string(tmv::domain::storm_window_movement_status_name(decision.status)) ==
                row.expected_movement);
        REQUIRE(std::string(tmv::domain::storm_window_movement_reason_name(decision.reason)) ==
                row.expected_movement_reason);
        REQUIRE(decision.requested_i_shift_cells == row.expected_i_cells);
        REQUIRE(decision.requested_j_shift_cells == row.expected_j_cells);
    }
}
