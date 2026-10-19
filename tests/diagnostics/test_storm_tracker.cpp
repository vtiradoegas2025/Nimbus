#include "catch2/catch.hpp"
#include "diagnostics/storm_rotation.hpp"
#include "diagnostics/storm_tracker.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

using namespace tmv::diagnostics;

namespace
{

StormColumnSignalView
view(std::size_t nx, std::size_t ny, const std::vector<double>& updraft, const std::vector<double>& condensate)
{
    static std::vector<double> helicity;
    helicity.assign(nx * ny, 0.0);
    return {nx, ny, 1000.0, 2000.0, 500.0, 1000.0, &updraft, &condensate, &helicity};
}

StormFeatureDetection detected_at(double x_m, double y_m, double confidence = 0.9)
{
    StormFeatureDetection result;
    result.status            = StormDetectionStatus::Detected;
    result.confidence        = confidence;
    result.core_centroid_x_m = x_m;
    result.core_centroid_y_m = y_m;
    return result;
}

StormFeatureDetection branches_at(std::initializer_list<std::pair<double, double>> positions)
{
    StormFeatureDetection result;
    result.status = positions.size() == 1 ? StormDetectionStatus::Detected : StormDetectionStatus::Ambiguous;
    result.candidate_component_count        = positions.size();
    result.reported_branch_candidate_count  = positions.size();
    result.confidence                       = 1.0 / static_cast<double>(positions.size());
    std::size_t rank = 0;
    for (const auto& position : positions)
    {
        auto& candidate             = result.branch_candidates[rank];
        candidate.component_label   = 100 + rank;
        candidate.component_cells   = 4;
        candidate.core_cells        = 4;
        candidate.score_fraction    = result.confidence;
        candidate.core_centroid_x_m = position.first;
        candidate.core_centroid_y_m = position.second;
        ++rank;
    }
    result.core_centroid_x_m = result.branch_candidates[0].core_centroid_x_m;
    result.core_centroid_y_m = result.branch_candidates[0].core_centroid_y_m;
    return result;
}

void set_primary_envelope(StormFeatureDetection& detection,
                          double west_m,
                          double east_m,
                          double south_m,
                          double north_m)
{
    auto& candidate = detection.branch_candidates[0];
    candidate.envelope_west_m = west_m;
    candidate.envelope_east_m = east_m;
    candidate.envelope_south_m = south_m;
    candidate.envelope_north_m = north_m;
}

} // namespace

TEST_CASE("connected detector returns physical envelope and boundary distances", "[diagnostics][storm_tracker]")
{
    constexpr std::size_t nx = 6;
    constexpr std::size_t ny = 5;
    std::vector<double>   updraft(nx * ny, 0.0);
    std::vector<double>   condensate(nx * ny, 0.0);
    for (std::size_t i = 2; i <= 3; ++i)
        for (std::size_t j = 1; j <= 2; ++j)
        {
            const std::size_t index = i * ny + j;
            updraft[index]          = 10.0;
            condensate[index]       = 1.0;
        }

    StormFeatureDetector   detector;
    StormFeatureThresholds thresholds;
    thresholds.minimum_component_cells = 4;
    const auto result                  = detector.detect(view(nx, ny, updraft, condensate), thresholds);

    REQUIRE(result.status == StormDetectionStatus::Detected);
    REQUIRE(result.confidence == Approx(1.0));
    REQUIRE(result.selected_component_cells == 4);
    REQUIRE(result.selected_core_cells == 4);
    REQUIRE(result.selected_rotating_core_cells == 0);
    REQUIRE(result.reported_branch_candidate_count == 1);
    REQUIRE_FALSE(result.branch_candidate_overflow);
    REQUIRE(result.branch_candidates[0].component_cells == 4);
    REQUIRE(result.branch_candidates[0].score_fraction == Approx(1.0));
    REQUIRE(result.branch_candidates[0].separation_from_primary_m == Approx(0.0));
    REQUIRE(result.branch_candidates[0].rotation_class == StormRotationClass::Nonrotating);
    REQUIRE(result.centroid_x_m == Approx(2500.0));
    REQUIRE(result.centroid_y_m == Approx(4000.0));
    REQUIRE(result.envelope_west_m == Approx(2000.0));
    REQUIRE(result.envelope_east_m == Approx(3000.0));
    REQUIRE(result.envelope_south_m == Approx(3000.0));
    REQUIRE(result.envelope_north_m == Approx(5000.0));
    REQUIRE(result.distance_west_m == Approx(1000.0));
    REQUIRE(result.distance_east_m == Approx(1000.0));
    REQUIRE(result.distance_south_m == Approx(1000.0));
    REQUIRE(result.distance_north_m == Approx(2000.0));
    REQUIRE(result.condensate_mass_kg == Approx(2.0e6));
}

TEST_CASE("shared condensate envelope preserves distinct updraft core branches",
          "[diagnostics][storm_tracker][watershed][merger]")
{
    constexpr std::size_t nx = 9;
    constexpr std::size_t ny = 5;
    std::vector<double> updraft(nx * ny, 0.0);
    std::vector<double> condensate(nx * ny, 1.0);
    for (std::size_t i = 1; i <= 2; ++i)
        for (std::size_t j = 1; j <= 2; ++j)
            updraft[i * ny + j] = 10.0;
    for (std::size_t i = 6; i <= 7; ++i)
        for (std::size_t j = 1; j <= 2; ++j)
            updraft[i * ny + j] = 12.0;

    StormFeatureDetector detector;
    StormFeatureThresholds thresholds;
    thresholds.minimum_component_cells = 4;
    thresholds.minimum_primary_score_fraction = 0.5;
    const auto result = detector.detect(view(nx, ny, updraft, condensate), thresholds);

    REQUIRE(result.envelope_component_count == 1);
    REQUIRE(result.candidate_component_count == 2);
    REQUIRE(result.reported_branch_candidate_count == 2);
    REQUIRE_FALSE(result.branch_candidate_overflow);
    REQUIRE(result.status == StormDetectionStatus::Detected);
    REQUIRE(result.branch_candidates[0].core_cells == 4);
    REQUIRE(result.branch_candidates[1].core_cells == 4);
    REQUIRE(result.branch_candidates[0].core_centroid_x_m == Approx(4500.0));
    REQUIRE(result.branch_candidates[1].core_centroid_x_m == Approx(2000.0));
    REQUIRE(result.branch_candidates[0].score_fraction == Approx(48.0 / 88.0));
    REQUIRE(result.branch_candidates[1].score_fraction == Approx(40.0 / 88.0));
    REQUIRE(result.branch_candidates[0].component_cells < nx * ny);
    REQUIRE(result.branch_candidates[1].component_cells < nx * ny);
}

TEST_CASE("detector reports exact disjoint overlap with the prior watershed partition",
          "[diagnostics][storm_tracker][watershed][overlap]")
{
    constexpr std::size_t nx = 9;
    constexpr std::size_t ny = 5;
    std::vector<double> condensate(nx * ny, 1.0);
    std::vector<double> one_core(nx * ny, 0.0);
    for (std::size_t i = 3; i <= 5; ++i)
        for (std::size_t j = 1; j <= 3; ++j)
            one_core[i * ny + j] = 12.0;
    StormFeatureThresholds thresholds;
    thresholds.minimum_component_cells = 4;
    thresholds.minimum_primary_score_fraction = 0.5;
    StormFeatureDetector detector;
    const auto original = detector.detect(view(nx, ny, one_core, condensate), thresholds);
    REQUIRE(original.reported_branch_candidate_count == 1);
    std::vector<std::uint8_t> previous;
    detector.copy_partition(original.branch_candidates[0].component_label, previous);
    REQUIRE(previous.size() == nx * ny);
    REQUIRE(std::count(previous.begin(), previous.end(), 1U) == nx * ny);

    std::vector<double> split(nx * ny, 0.0);
    for (std::size_t i = 1; i <= 2; ++i)
        for (std::size_t j = 1; j <= 2; ++j)
            split[i * ny + j] = 10.0;
    for (std::size_t i = 6; i <= 7; ++i)
        for (std::size_t j = 1; j <= 2; ++j)
            split[i * ny + j] = 12.0;
    const auto descendants = detector.detect(view(nx, ny, split, condensate), thresholds, &previous);
    REQUIRE(descendants.previous_partition_available);
    REQUIRE(descendants.reported_branch_candidate_count == 2);
    const double overlap_sum =
        descendants.branch_candidates[0].previous_partition_overlap_fraction +
        descendants.branch_candidates[1].previous_partition_overlap_fraction;
    REQUIRE(overlap_sum == Approx(1.0));
    REQUIRE((descendants.branch_candidates[0].previous_partition_overlap_fraction >= 0.5) !=
            (descendants.branch_candidates[1].previous_partition_overlap_fraction >= 0.5));
}

TEST_CASE("physical core area is invariant under horizontal refinement",
          "[diagnostics][storm_tracker][resolution][area]")
{
    for (const auto& representation :
         std::vector<std::pair<std::size_t, double>>{{2, 500.0}, {4, 250.0}})
    {
        const std::size_t side = representation.first;
        const double spacing_m = representation.second;
        const std::size_t nx = side + 2;
        const std::size_t ny = side + 2;
        std::vector<double> updraft(nx * ny, 0.0);
        std::vector<double> condensate(nx * ny, 0.0);
        for (std::size_t i = 1; i <= side; ++i)
            for (std::size_t j = 1; j <= side; ++j)
            {
                updraft[i * ny + j] = 12.0;
                condensate[i * ny + j] = 1.0;
            }
        std::vector<double> helicity(nx * ny, 0.0);
        StormColumnSignalView signals{
            nx, ny, 0.0, 0.0, spacing_m, spacing_m, &updraft, &condensate, &helicity};
        StormFeatureThresholds thresholds;
        thresholds.minimum_component_cells = 1;
        thresholds.minimum_core_area_m2 = 1.0e6;
        StormFeatureDetector detector;
        const auto result = detector.detect(signals, thresholds);
        REQUIRE(result.status == StormDetectionStatus::Detected);
        REQUIRE(result.candidate_component_count == 1);
    }
}

TEST_CASE("detector retains signed rotation evidence without excluding nonrotating convection",
          "[diagnostics][storm_tracker][rotation]")
{
    constexpr std::size_t nx = 4;
    constexpr std::size_t ny = 4;
    std::vector<double> updraft(nx * ny, 10.0);
    std::vector<double> condensate(nx * ny, 1.0);
    std::vector<double> helicity(nx * ny, 0.0);
    helicity[1 * ny + 1] = -50.0;
    helicity[1 * ny + 2] = -100.0;

    StormColumnSignalView signals{
        nx, ny, 0.0, 0.0, 1000.0, 1000.0, &updraft, &condensate, &helicity};
    StormFeatureDetector detector;
    const auto result = detector.detect(signals, {});

    REQUIRE(result.status == StormDetectionStatus::Detected);
    REQUIRE(result.selected_rotating_core_cells == 2);
    REQUIRE(result.maximum_absolute_updraft_helicity_m2s2 == Approx(100.0));
    REQUIRE(result.signed_core_mean_updraft_helicity_m2s2 == Approx(-75.0));
    REQUIRE(result.branch_candidates[0].rotation_class == StormRotationClass::Negative);
    REQUIRE(result.branch_candidates[0].signed_rotating_core_mean_updraft_helicity_m2s2 == Approx(-75.0));
}

TEST_CASE("opposite rotating cores are not mislabeled as negative rotation",
          "[diagnostics][storm_tracker][rotation][branches]")
{
    constexpr std::size_t nx = 2;
    constexpr std::size_t ny = 2;
    std::vector<double> updraft(nx * ny, 10.0);
    std::vector<double> condensate(nx * ny, 1.0);
    std::vector<double> helicity{50.0, -50.0, 0.0, 0.0};
    StormColumnSignalView signals{
        nx, ny, 0.0, 0.0, 1000.0, 1000.0, &updraft, &condensate, &helicity};
    StormFeatureDetector detector;
    const auto result = detector.detect(signals, {});

    REQUIRE(result.status == StormDetectionStatus::Detected);
    REQUIRE(result.branch_candidates[0].rotating_core_cells == 2);
    REQUIRE(result.branch_candidates[0].signed_rotating_core_mean_updraft_helicity_m2s2 == Approx(0.0));
    REQUIRE(result.branch_candidates[0].rotation_class == StormRotationClass::MixedZeroMean);
}

TEST_CASE("2-5 km AGL updraft helicity recovers solid-body rotation on both Cartesian staggerings",
          "[diagnostics][storm_tracker][rotation][manufactured]")
{
    constexpr int n = 9;
    constexpr int nz = 14;
    constexpr double dx = 500.0;
    constexpr double dz = 500.0;
    constexpr double zeta = 0.02;
    constexpr double updraft = 10.0;
    const PhysicalCellDomain horizontal_and_c_vertical{1, n - 1, 1, n - 1, 0, nz};
    const PhysicalCellDomain a_domain{1, n - 1, 1, n - 1, 1, nz - 1};

    for (StaggerType staggering : {StaggerType::Collocated, StaggerType::CGrid})
    {
        Field3D u(n, n, nz, 0.0f);
        Field3D v(n, n, nz, 0.0f);
        Field3D w(n, n, nz, static_cast<float>(updraft));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    u(i, j, k) = static_cast<float>(-0.5 * zeta * static_cast<double>(j) * dx);
                    v(i, j, k) = static_cast<float>(0.5 * zeta * static_cast<double>(i) * dx);
                }

        const PhysicalCellDomain domain =
            staggering == StaggerType::CGrid ? horizontal_and_c_vertical : a_domain;
        std::vector<double> output(
            static_cast<std::size_t>(domain.nx()) * static_cast<std::size_t>(domain.ny()));
        compute_storm_updraft_helicity_columns(
            StormRotationGridView{
                &u, &v, &w, domain, staggering, dx, dx, dz, nullptr, nullptr, nullptr, nullptr},
            StormRotationLayer{},
            output);
        for (double value : output)
            CHECK(value == Approx(updraft * zeta * 3000.0).epsilon(2.0e-6));
    }
}

TEST_CASE("mapped-grid helicity differentiates at constant physical height",
          "[diagnostics][storm_tracker][rotation][terrain][manufactured]")
{
    constexpr int n = 9;
    constexpr int nz = 14;
    constexpr double dx = 500.0;
    constexpr double dz = 500.0;
    constexpr double zeta = -0.015;
    constexpr double updraft = 8.0;
    TerrainMetrics3D terrain;
    terrain.z.resize(n, n, nz, 0.0);
    terrain.z_face.resize(n, n, nz + 1, 0.0);
    terrain.J.resize(n, n, nz, 1.0);
    terrain.mx.resize(n, n, nz, 0.0);
    terrain.my.resize(n, n, nz, 0.0);
    terrain.zeta.resize(n, n, nz, 0.0);
    terrain.Jx_face.resize(n, n, nz, 1.0);
    terrain.Jy_face.resize(n, n, nz, 1.0);
    terrain.Gx_face.resize(n, n, nz + 1, 0.0);
    terrain.Gy_face.resize(n, n, nz + 1, 0.0);
    terrain.dx = dx;
    terrain.dy = dx;
    terrain.dzeta = dz;

    Field3D u(n, n, nz, 0.0f);
    Field3D v(n, n, nz, 0.0f);
    Field3D w(n, n, nz, static_cast<float>(updraft));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            const double ground = 0.08 * static_cast<double>(i) * dx -
                                  0.05 * static_cast<double>(j) * dx;
            for (int k = 0; k <= nz; ++k)
                terrain.z_face(i, j, k) = ground + static_cast<double>(k) * dz;
            for (int k = 0; k < nz; ++k)
            {
                const double height = ground + (static_cast<double>(k) + 0.5) * dz;
                terrain.z(i, j, k) = height;
                // Vertical shear terms would contaminate zeta without the
                // constant-physical-height metric correction.
                u(i, j, k) = static_cast<float>(
                    -0.5 * zeta * static_cast<double>(j) * dx + 0.003 * height);
                v(i, j, k) = static_cast<float>(
                    0.5 * zeta * static_cast<double>(i) * dx - 0.004 * height);
            }
        }

    const PhysicalCellDomain domain{1, n - 1, 1, n - 1, 1, nz - 1};
    std::vector<double> output(
        static_cast<std::size_t>(domain.nx()) * static_cast<std::size_t>(domain.ny()));
    compute_storm_updraft_helicity_columns(
        StormRotationGridView{
            &u, &v, &w, domain, StaggerType::Collocated, dx, dx, dz, &terrain, nullptr, nullptr, nullptr},
        StormRotationLayer{},
        output);
    for (double value : output)
        CHECK(value == Approx(updraft * zeta * 3000.0).epsilon(3.0e-5));
}

TEST_CASE("mapped C-grid helicity collocates velocity faces before its metric correction",
          "[diagnostics][storm_tracker][rotation][terrain][cgrid][manufactured]")
{
    constexpr int n = 9;
    constexpr int nz = 14;
    constexpr double dx = 500.0;
    constexpr double dz = 500.0;
    constexpr double zeta = 0.012;
    constexpr double updraft = 9.0;
    TerrainMetrics3D terrain;
    terrain.z.resize(n, n, nz, 0.0);
    terrain.z_face.resize(n, n, nz + 1, 0.0);
    terrain.J.resize(n, n, nz, 1.0);
    terrain.mx.resize(n, n, nz, 0.0);
    terrain.my.resize(n, n, nz, 0.0);
    terrain.zeta.resize(n, n, nz, 0.0);
    terrain.Jx_face.resize(n, n, nz, 1.0);
    terrain.Jy_face.resize(n, n, nz, 1.0);
    terrain.Gx_face.resize(n, n, nz + 1, 0.0);
    terrain.Gy_face.resize(n, n, nz + 1, 0.0);
    terrain.dx = dx;
    terrain.dy = dx;
    terrain.dzeta = dz;

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            const double ground = 0.06 * static_cast<double>(i) * dx;
            for (int k = 0; k <= nz; ++k)
                terrain.z_face(i, j, k) = ground + static_cast<double>(k) * dz;
            for (int k = 0; k < nz; ++k)
                terrain.z(i, j, k) = ground + (static_cast<double>(k) + 0.5) * dz;
        }

    Field3D u(n, n, nz, 0.0f);
    Field3D v(n, n, nz, 0.0f);
    Field3D w(n, n, nz, static_cast<float>(updraft));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const int right = std::min(i + 1, n - 1);
                const double z_u_face = 0.5 * (terrain.z(i, j, k) + terrain.z(right, j, k));
                u(i, j, k) = static_cast<float>(
                    -0.5 * zeta * static_cast<double>(j) * dx + 0.002 * z_u_face);
                v(i, j, k) = static_cast<float>(
                    0.5 * zeta * static_cast<double>(i) * dx - 0.003 * terrain.z(i, j, k));
            }

    const PhysicalCellDomain domain{1, n - 1, 1, n - 1, 0, nz};
    std::vector<double> ground_w(static_cast<std::size_t>(n * n), 0.0);
    std::vector<double> output(
        static_cast<std::size_t>(domain.nx()) * static_cast<std::size_t>(domain.ny()));
    compute_storm_updraft_helicity_columns(
        StormRotationGridView{
            &u, &v, &w, domain, StaggerType::CGrid, dx, dx, dz, &terrain, nullptr, nullptr, &ground_w},
        StormRotationLayer{},
        output);
    for (double value : output)
        CHECK(value == Approx(updraft * zeta * 3000.0).epsilon(4.0e-5));
}

TEST_CASE("updraft-helicity layer coverage fails closed", "[diagnostics][storm_tracker][rotation][validation]")
{
    Field3D u(5, 5, 6, 0.0f);
    Field3D v(5, 5, 6, 0.0f);
    Field3D w(5, 5, 6, 1.0f);
    const PhysicalCellDomain domain{1, 4, 1, 4, 0, 6};
    std::vector<double> output(9);
    REQUIRE_THROWS_WITH(
        compute_storm_updraft_helicity_columns(
            StormRotationGridView{
                &u, &v, &w, domain, StaggerType::CGrid, 1000.0, 1000.0, 500.0, nullptr, nullptr, nullptr, nullptr},
            StormRotationLayer{},
            output),
        Catch::Matchers::Contains("incomplete requested AGL coverage"));
}

TEST_CASE("two comparable coherent components are explicitly ambiguous", "[diagnostics][storm_tracker][ambiguity]")
{
    constexpr std::size_t nx = 8;
    constexpr std::size_t ny = 4;
    std::vector<double>   updraft(nx * ny, 0.0);
    std::vector<double>   condensate(nx * ny, 0.0);
    for (std::size_t i : {1u, 2u, 5u, 6u})
        for (std::size_t j : {1u, 2u})
        {
            updraft[i * ny + j]    = 10.0;
            condensate[i * ny + j] = 1.0;
        }

    StormFeatureDetector   detector;
    StormFeatureThresholds thresholds;
    thresholds.minimum_component_cells        = 4;
    thresholds.minimum_primary_score_fraction = 0.8;
    const auto result                         = detector.detect(view(nx, ny, updraft, condensate), thresholds);

    REQUIRE(result.status == StormDetectionStatus::Ambiguous);
    REQUIRE(result.candidate_component_count == 2);
    REQUIRE(result.reported_branch_candidate_count == 2);
    REQUIRE_FALSE(result.branch_candidate_overflow);
    REQUIRE(result.confidence == Approx(0.5));
    REQUIRE(result.branch_candidates[0].score_fraction == Approx(0.5));
    REQUIRE(result.branch_candidates[1].score_fraction == Approx(0.5));
    REQUIRE(result.branch_candidates[0].component_label < result.branch_candidates[1].component_label);
    REQUIRE(result.branch_candidates[0].separation_from_primary_m == Approx(0.0));
    REQUIRE(result.branch_candidates[1].separation_from_primary_m == Approx(2000.0));
}

TEST_CASE("branch evidence is fixed-capacity and retains the strongest candidates",
          "[diagnostics][storm_tracker][branches][capacity]")
{
    constexpr std::size_t branch_count = storm_branch_candidate_capacity + 1;
    constexpr std::size_t nx = 2 * branch_count + 3;
    constexpr std::size_t ny = 3;
    std::vector<double> updraft(nx * ny, 0.0);
    std::vector<double> condensate(nx * ny, 0.0);
    for (std::size_t branch = 0; branch < branch_count; ++branch)
    {
        const std::size_t i = 1 + 2 * branch;
        for (std::size_t j = 0; j < 2; ++j)
        {
            updraft[i * ny + j] = 6.0 + static_cast<double>(branch);
            condensate[i * ny + j] = 1.0;
        }
    }

    StormFeatureThresholds thresholds;
    thresholds.minimum_component_cells = 2;
    thresholds.minimum_primary_score_fraction = 0.8;
    StormFeatureDetector detector;
    const auto result = detector.detect(view(nx, ny, updraft, condensate), thresholds);

    REQUIRE(result.status == StormDetectionStatus::Ambiguous);
    REQUIRE(result.candidate_component_count == branch_count);
    REQUIRE(result.reported_branch_candidate_count == storm_branch_candidate_capacity);
    REQUIRE(result.branch_candidate_overflow);
    REQUIRE(result.branch_candidates.front().component_label >
            result.branch_candidates.back().component_label);
    REQUIRE(result.branch_candidates.front().score_fraction >
            result.branch_candidates.back().score_fraction);
    REQUIRE(result.branch_candidates.front().core_centroid_x_m == Approx(65750.0));
    REQUIRE(result.branch_candidates.back().core_centroid_x_m == Approx(2750.0));
}

TEST_CASE("detector reports weak instead of inventing a center", "[diagnostics][storm_tracker][weak]")
{
    constexpr std::size_t nx = 4;
    constexpr std::size_t ny = 4;
    std::vector<double>   updraft(nx * ny, 20.0);
    std::vector<double>   condensate(nx * ny, 0.0);

    StormFeatureDetector detector;
    const auto           result = detector.detect(view(nx, ny, updraft, condensate), {});
    REQUIRE(result.status == StormDetectionStatus::Weak);
    REQUIRE(result.candidate_component_count == 0);
}

TEST_CASE("detector preallocates the exact 8-connected fragmentation ceiling",
          "[diagnostics][storm_tracker][workspace]")
{
    constexpr std::size_t nx = 5;
    constexpr std::size_t ny = 5;
    std::vector<double>   updraft(nx * ny, 0.0);
    std::vector<double>   condensate(nx * ny, 0.0);
    for (std::size_t i = 0; i < nx; i += 2)
        for (std::size_t j = 0; j < ny; j += 2)
            condensate[i * ny + j] = 1.0;

    REQUIRE(storm_feature_component_capacity(nx, ny) == 9);
    StormFeatureDetector detector;
    detector.reserve(nx, ny);
    const auto result = detector.detect(view(nx, ny, updraft, condensate), {});
    REQUIRE(result.envelope_component_count == 9);
    REQUIRE(result.status == StormDetectionStatus::Weak);
}

TEST_CASE("detector rejects non-finite scientific signals", "[diagnostics][storm_tracker][validation]")
{
    std::vector<double> updraft(4, 0.0);
    std::vector<double> condensate(4, 1.0);
    condensate[2] = -1.0;
    StormFeatureDetector detector;
    REQUIRE_THROWS_WITH(detector.detect(view(2, 2, updraft, condensate), {}),
                        Catch::Matchers::Contains("non-negative"));

    condensate[2] = 1.0;
    std::vector<double> helicity(4, 0.0);
    helicity[1] = std::numeric_limits<double>::quiet_NaN();
    const StormColumnSignalView signals{
        2, 2, 0.0, 0.0, 1000.0, 1000.0, &updraft, &condensate, &helicity};
    REQUIRE_THROWS_WITH(detector.detect(signals, {}), Catch::Matchers::Contains("finite"));
}

TEST_CASE("continuity regression recovers deterministic physical storm motion",
          "[diagnostics][storm_tracker][continuity]")
{
    StormContinuityState      state;
    StormContinuityThresholds thresholds;
    thresholds.minimum_samples          = 3;
    thresholds.lost_after_misses        = 2;
    thresholds.maximum_speed_ms         = 50.0;
    thresholds.maximum_acceleration_ms2 = 0.1;
    thresholds.position_uncertainty_m   = 100.0;

    update_storm_continuity(state, detected_at(1000.0, 5000.0), 0.0, thresholds);
    REQUIRE(state.status == StormContinuityStatus::Tentative);
    REQUIRE_FALSE(state.motion_available);
    REQUIRE(state.branch_id == 1);

    update_storm_continuity(state, detected_at(1600.0, 4700.0), 60.0, thresholds);
    REQUIRE(state.status == StormContinuityStatus::Tentative);
    REQUIRE_FALSE(state.motion_available);

    update_storm_continuity(state, detected_at(2200.0, 4400.0), 120.0, thresholds);
    REQUIRE(state.status == StormContinuityStatus::Tracking);
    REQUIRE(state.motion_available);
    REQUIRE(state.velocity_x_ms == Approx(10.0));
    REQUIRE(state.velocity_y_ms == Approx(-5.0));
    REQUIRE(state.speed_ms == Approx(std::hypot(10.0, 5.0)));
    REQUIRE(state.velocity_uncertainty_ms > 0.0);
    REQUIRE(state.continuity_confidence == Approx(0.9));
    REQUIRE(valid_storm_continuity_state(state));
}

TEST_CASE("selected envelope face motion retains signed coordinate velocities",
          "[diagnostics][storm_tracker][continuity][face_motion]")
{
    StormContinuityThresholds thresholds;
    thresholds.minimum_samples = 2;
    thresholds.maximum_speed_ms = 50.0;
    thresholds.maximum_acceleration_ms2 = 1.0;
    thresholds.position_uncertainty_m = 100.0;

    StormContinuityState state;
    auto first = branches_at({{1000.0, 2000.0}});
    set_primary_envelope(first, 400.0, 1600.0, 1200.0, 2800.0);
    update_storm_continuity(state, first, 0.0, thresholds);
    REQUIRE_FALSE(derive_storm_envelope_face_motion(
                      StormContinuityState{}, first, state, 0.0).available);
    const auto prior = state;

    auto second = branches_at({{1600.0, 2000.0}});
    set_primary_envelope(second, -200.0, 2500.0, 1400.0, 2500.0);
    update_storm_continuity(state, second, 60.0, thresholds);
    const auto motion = derive_storm_envelope_face_motion(
        prior, second, state, 60.0);

    REQUIRE(motion.available);
    REQUIRE(motion.interval_s == Approx(60.0));
    REQUIRE(motion.west_coordinate_velocity_ms == Approx(-10.0));
    REQUIRE(motion.east_coordinate_velocity_ms == Approx(15.0));
    REQUIRE(motion.south_coordinate_velocity_ms == Approx(10.0 / 3.0));
    REQUIRE(motion.north_coordinate_velocity_ms == Approx(-5.0));
}

TEST_CASE("selected envelope face motion is restart-derivable and zeroes ULP noise",
          "[diagnostics][storm_tracker][continuity][face_motion][restart]")
{
    StormContinuityThresholds thresholds;
    thresholds.minimum_samples = 2;
    thresholds.maximum_speed_ms = 50.0;
    thresholds.maximum_acceleration_ms2 = 1.0;
    thresholds.position_uncertainty_m = 100.0;

    StormContinuityState continuous;
    auto first = branches_at({{1.0e9, 1.0e9}});
    set_primary_envelope(
        first, 1.0e9, 1.0e9 + 1200.0,
        1.0e9 + 2400.0, 1.0e9 + 3600.0);
    update_storm_continuity(continuous, first, 30.0, thresholds);
    const StormContinuityState restored = continuous;

    auto second = branches_at({{1.0e9 + 300.0, 1.0e9}});
    const double ulp_noise = 1.0e-7;
    set_primary_envelope(
        second, 1.0e9 + ulp_noise, 1.0e9 + 1500.0,
        1.0e9 + 2400.0 - ulp_noise, 1.0e9 + 3900.0);
    auto continuous_current = continuous;
    auto restored_current = restored;
    update_storm_continuity(
        continuous_current, second, 60.0, thresholds);
    update_storm_continuity(
        restored_current, second, 60.0, thresholds);

    const auto continuous_motion = derive_storm_envelope_face_motion(
        continuous, second, continuous_current, 60.0);
    const auto restored_motion = derive_storm_envelope_face_motion(
        restored, second, restored_current, 60.0);
    REQUIRE(continuous_motion.available);
    REQUIRE(restored_motion.available);
    REQUIRE(continuous_motion.interval_s == restored_motion.interval_s);
    REQUIRE(continuous_motion.west_coordinate_velocity_ms == 0.0);
    REQUIRE(continuous_motion.south_coordinate_velocity_ms == 0.0);
    REQUIRE(continuous_motion.east_coordinate_velocity_ms == Approx(10.0));
    REQUIRE(continuous_motion.north_coordinate_velocity_ms == Approx(10.0));
    REQUIRE(restored_motion.west_coordinate_velocity_ms ==
            continuous_motion.west_coordinate_velocity_ms);
    REQUIRE(restored_motion.east_coordinate_velocity_ms ==
            continuous_motion.east_coordinate_velocity_ms);
    REQUIRE(restored_motion.south_coordinate_velocity_ms ==
            continuous_motion.south_coordinate_velocity_ms);
    REQUIRE(restored_motion.north_coordinate_velocity_ms ==
            continuous_motion.north_coordinate_velocity_ms);
}

TEST_CASE("selected envelope face motion fails closed across ambiguity and topology reorganization",
          "[diagnostics][storm_tracker][continuity][face_motion][authority]")
{
    StormContinuityThresholds thresholds;
    thresholds.minimum_samples = 2;
    thresholds.lost_after_misses = 3;
    thresholds.maximum_speed_ms = 40.0;
    thresholds.maximum_acceleration_ms2 = 0.05;
    thresholds.position_uncertainty_m = 50.0;

    SECTION("ambiguity does not bridge the last accepted envelope")
    {
        StormContinuityState state;
        auto first = branches_at({{0.0, 0.0}});
        set_primary_envelope(first, -500.0, 500.0, -500.0, 500.0);
        update_storm_continuity(state, first, 0.0, thresholds);
        const auto prior = state;

        auto jumped = branches_at({{5000.0, 0.0}});
        set_primary_envelope(jumped, 4500.0, 5500.0, -500.0, 500.0);
        update_storm_continuity(state, jumped, 60.0, thresholds);
        REQUIRE(state.status == StormContinuityStatus::Ambiguous);
        const auto motion = derive_storm_envelope_face_motion(
            prior, jumped, state, 60.0);
        REQUIRE_FALSE(motion.available);
        REQUIRE(motion.interval_s == 0.0);
        REQUIRE(motion.west_coordinate_velocity_ms == 0.0);
        REQUIRE(motion.east_coordinate_velocity_ms == 0.0);
        REQUIRE(motion.south_coordinate_velocity_ms == 0.0);
        REQUIRE(motion.north_coordinate_velocity_ms == 0.0);
    }

    SECTION("a majority-overlap reorganization starts a new face baseline")
    {
        StormContinuityState state;
        auto first = branches_at({{0.0, 0.0}});
        set_primary_envelope(first, -500.0, 500.0, -500.0, 500.0);
        update_storm_continuity(state, first, 0.0, thresholds);
        auto second = branches_at({{600.0, 0.0}});
        set_primary_envelope(second, 100.0, 1100.0, -500.0, 500.0);
        update_storm_continuity(state, second, 60.0, thresholds);
        const auto prior = state;

        auto merged = branches_at({{6000.0, 0.0}});
        set_primary_envelope(merged, 100.0, 7000.0, -500.0, 500.0);
        update_storm_continuity(state, merged, 120.0, thresholds);
        REQUIRE(state.topology_reorganization);
        const auto motion = derive_storm_envelope_face_motion(
            prior, merged, state, 120.0);
        REQUIRE_FALSE(motion.available);
    }

    SECTION("branch changes and incomplete candidate sets do not provide face motion")
    {
        StormContinuityState state;
        auto first = branches_at({{0.0, 0.0}});
        set_primary_envelope(first, -500.0, 500.0, -500.0, 500.0);
        update_storm_continuity(state, first, 0.0, thresholds);
        const auto prior = state;

        auto second = branches_at({{600.0, 0.0}});
        set_primary_envelope(second, 100.0, 1100.0, -500.0, 500.0);
        update_storm_continuity(state, second, 60.0, thresholds);
        auto changed_branch = state;
        ++changed_branch.branch_id;
        REQUIRE(valid_storm_continuity_state(changed_branch));
        REQUIRE_FALSE(derive_storm_envelope_face_motion(
                          prior, second, changed_branch, 60.0).available);

        second.branch_candidate_overflow = true;
        REQUIRE_FALSE(derive_storm_envelope_face_motion(
                          prior, second, state, 60.0).available);
        second.branch_candidate_overflow = false;
        REQUIRE_THROWS_WITH(
            derive_storm_envelope_face_motion(
                prior, second, state, 61.0),
            Catch::Matchers::Contains("sample time"));
    }
}

TEST_CASE("continuity rejects a branch jump and becomes explicitly lost",
          "[diagnostics][storm_tracker][continuity][lost]")
{
    StormContinuityState      state;
    StormContinuityThresholds thresholds;
    thresholds.minimum_samples          = 2;
    thresholds.lost_after_misses        = 2;
    thresholds.maximum_speed_ms         = 40.0;
    thresholds.maximum_acceleration_ms2 = 0.05;
    thresholds.position_uncertainty_m   = 50.0;

    update_storm_continuity(state, detected_at(0.0, 0.0), 0.0, thresholds);
    update_storm_continuity(state, detected_at(600.0, 0.0), 60.0, thresholds);
    REQUIRE(state.status == StormContinuityStatus::Tracking);

    update_storm_continuity(state, detected_at(5000.0, 0.0), 120.0, thresholds);
    REQUIRE(state.status == StormContinuityStatus::Ambiguous);
    REQUIRE(state.reason == StormContinuityReason::SpeedGate);
    REQUIRE_FALSE(state.motion_available);
    REQUIRE(state.sample_count == 2);

    StormFeatureDetection weak;
    weak.status = StormDetectionStatus::Weak;
    update_storm_continuity(state, weak, 180.0, thresholds);
    REQUIRE(state.status == StormContinuityStatus::Lost);
    REQUIRE(state.sample_count == 0);
    REQUIRE_FALSE(state.motion_available);

    update_storm_continuity(state, detected_at(5200.0, 0.0), 240.0, thresholds);
    REQUIRE(state.status == StormContinuityStatus::Tentative);
    REQUIRE(state.branch_id == 2);
}

TEST_CASE("continuity distinguishes acceleration inconsistency from impossible speed",
          "[diagnostics][storm_tracker][continuity][association]")
{
    StormContinuityState      state;
    StormContinuityThresholds thresholds;
    thresholds.minimum_samples          = 2;
    thresholds.lost_after_misses        = 3;
    thresholds.maximum_speed_ms         = 40.0;
    thresholds.maximum_acceleration_ms2 = 0.05;
    thresholds.position_uncertainty_m   = 50.0;

    update_storm_continuity(state, detected_at(0.0, 0.0), 0.0, thresholds);
    update_storm_continuity(state, detected_at(600.0, 0.0), 60.0, thresholds);
    REQUIRE(state.status == StormContinuityStatus::Tracking);

    // Twenty metres per second is below the speed gate, but the 600 m
    // innovation from the established 10 m/s track exceeds the explicit
    // acceleration-plus-position-uncertainty gate.
    update_storm_continuity(state, detected_at(1800.0, 0.0), 120.0, thresholds);
    REQUIRE(state.status == StormContinuityStatus::Ambiguous);
    REQUIRE(state.reason == StormContinuityReason::AccelerationGate);
    REQUIRE(state.innovation_distance_m == Approx(600.0));
    REQUIRE(state.association_gate_radius_m == Approx(190.0));
    REQUIRE(state.sample_count == 2);
}

TEST_CASE("majority envelope overlap preserves identity across a core merger without inventing motion",
          "[diagnostics][storm_tracker][continuity][reorganization]")
{
    StormContinuityThresholds thresholds;
    thresholds.minimum_samples          = 3;
    thresholds.lost_after_misses        = 3;
    thresholds.maximum_speed_ms         = 40.0;
    thresholds.maximum_acceleration_ms2 = 0.05;
    thresholds.position_uncertainty_m   = 50.0;
    StormContinuityState state;

    auto first = branches_at({{0.0, 0.0}});
    first.branch_candidates[0].envelope_west_m = -500.0;
    first.branch_candidates[0].envelope_east_m = 500.0;
    first.branch_candidates[0].envelope_south_m = -500.0;
    first.branch_candidates[0].envelope_north_m = 500.0;
    update_storm_continuity(state, first, 0.0, thresholds);

    auto second = branches_at({{600.0, 0.0}});
    second.branch_candidates[0].envelope_west_m = 100.0;
    second.branch_candidates[0].envelope_east_m = 1100.0;
    second.branch_candidates[0].envelope_south_m = -500.0;
    second.branch_candidates[0].envelope_north_m = 500.0;
    update_storm_continuity(state, second, 60.0, thresholds);
    REQUIRE(state.branch_id == 1);

    // A merger moves the aggregate core centroid 5.4 km in one interval, but
    // the new envelope contains the entire previously accepted partition.
    auto merged = branches_at({{6000.0, 0.0}});
    merged.branch_candidates[0].envelope_west_m = 100.0;
    merged.branch_candidates[0].envelope_east_m = 7000.0;
    merged.branch_candidates[0].envelope_south_m = -500.0;
    merged.branch_candidates[0].envelope_north_m = 500.0;
    update_storm_continuity(state, merged, 120.0, thresholds);
    REQUIRE(state.status == StormContinuityStatus::Tentative);
    REQUIRE(state.branch_id == 1);
    REQUIRE(state.selected_candidate_available);
    REQUIRE(state.topology_reorganization);
    REQUIRE(state.reorganization_overlap_fraction == Approx(1.0));
    REQUIRE(state.sample_count == 1);
    REQUIRE_FALSE(state.motion_available);
    REQUIRE(state.velocity_x_ms == Approx(0.0));

    auto stable = merged;
    stable.branch_candidates[0].core_centroid_x_m = 6300.0;
    update_storm_continuity(state, stable, 180.0, thresholds);
    REQUIRE_FALSE(state.topology_reorganization);
    REQUIRE(state.status == StormContinuityStatus::Tentative);
    stable.branch_candidates[0].core_centroid_x_m = 6600.0;
    update_storm_continuity(state, stable, 240.0, thresholds);
    REQUIRE(state.status == StormContinuityStatus::Tracking);
    REQUIRE(state.velocity_x_ms == Approx(5.0));
    REQUIRE(state.branch_id == 1);
    REQUIRE(valid_storm_continuity_state(state));
}

TEST_CASE("overlapping split successors cannot claim a unique reorganized branch",
          "[diagnostics][storm_tracker][continuity][reorganization][ambiguity]")
{
    StormContinuityThresholds thresholds;
    thresholds.minimum_samples          = 2;
    thresholds.maximum_speed_ms         = 40.0;
    thresholds.maximum_acceleration_ms2 = 0.05;
    thresholds.position_uncertainty_m   = 50.0;
    StormContinuityState state;

    auto initial = branches_at({{0.0, 0.0}});
    initial.branch_candidates[0].envelope_west_m = -1000.0;
    initial.branch_candidates[0].envelope_east_m = 1000.0;
    initial.branch_candidates[0].envelope_south_m = -500.0;
    initial.branch_candidates[0].envelope_north_m = 500.0;
    update_storm_continuity(state, initial, 0.0, thresholds);

    auto established = initial;
    established.branch_candidates[0].core_centroid_x_m = 600.0;
    established.branch_candidates[0].envelope_west_m = -400.0;
    established.branch_candidates[0].envelope_east_m = 1600.0;
    update_storm_continuity(state, established, 60.0, thresholds);

    auto split = branches_at({{6000.0, -100.0}, {6000.0, 100.0}});
    for (std::size_t rank = 0; rank < 2; ++rank)
    {
        split.branch_candidates[rank].envelope_west_m = -400.0;
        split.branch_candidates[rank].envelope_east_m = 1600.0;
        split.branch_candidates[rank].envelope_south_m = -500.0;
        split.branch_candidates[rank].envelope_north_m = 500.0;
    }
    update_storm_continuity(state, split, 120.0, thresholds);
    REQUIRE(state.status == StormContinuityStatus::Ambiguous);
    REQUIRE(state.reason == StormContinuityReason::SpatialAmbiguity);
    REQUIRE_FALSE(state.selected_candidate_available);
    REQUIRE_FALSE(state.topology_reorganization);
    REQUIRE(state.branch_id == 1);
}

TEST_CASE("exact partition majority resolves descendants whose bounding boxes overlap",
          "[diagnostics][storm_tracker][continuity][reorganization][partition]")
{
    StormContinuityThresholds thresholds;
    thresholds.minimum_samples          = 2;
    thresholds.maximum_speed_ms         = 40.0;
    thresholds.maximum_acceleration_ms2 = 0.05;
    thresholds.position_uncertainty_m   = 50.0;
    StormContinuityState state;

    auto initial = branches_at({{0.0, 0.0}});
    initial.branch_candidates[0].envelope_west_m = -1000.0;
    initial.branch_candidates[0].envelope_east_m = 1000.0;
    initial.branch_candidates[0].envelope_south_m = -500.0;
    initial.branch_candidates[0].envelope_north_m = 500.0;
    update_storm_continuity(state, initial, 0.0, thresholds);
    auto established = initial;
    established.branch_candidates[0].core_centroid_x_m = 600.0;
    update_storm_continuity(state, established, 60.0, thresholds);

    auto split = branches_at({{6000.0, -100.0}, {6000.0, 100.0}});
    split.previous_partition_available = true;
    split.branch_candidates[0].previous_partition_overlap_fraction = 0.7;
    split.branch_candidates[1].previous_partition_overlap_fraction = 0.3;
    // Both summary boxes deliberately cover the whole predecessor. Exact
    // watershed cells, not bbox overlap, must select the majority descendant.
    for (std::size_t rank = 0; rank < 2; ++rank)
    {
        split.branch_candidates[rank].envelope_west_m = -1000.0;
        split.branch_candidates[rank].envelope_east_m = 1000.0;
        split.branch_candidates[rank].envelope_south_m = -500.0;
        split.branch_candidates[rank].envelope_north_m = 500.0;
    }
    update_storm_continuity(state, split, 120.0, thresholds);
    REQUIRE(state.status == StormContinuityStatus::Tentative);
    REQUIRE(state.selected_candidate_rank == 0);
    REQUIRE(state.branch_id == 1);
    REQUIRE(state.topology_reorganization);
    REQUIRE(state.reorganization_overlap_fraction == Approx(0.7));
    REQUIRE_FALSE(state.motion_available);
}

TEST_CASE("continuity regression retains a bounded newest-sample window",
          "[diagnostics][storm_tracker][continuity][capacity]")
{
    StormContinuityState      state;
    StormContinuityThresholds thresholds;
    thresholds.minimum_samples        = 3;
    thresholds.position_uncertainty_m = 100.0;

    for (std::size_t sample = 0; sample < 7; ++sample)
    {
        const double time_s = 60.0 * static_cast<double>(sample);
        update_storm_continuity(state, detected_at(1000.0 + 12.0 * time_s, 2000.0 - 3.0 * time_s), time_s, thresholds);
    }

    REQUIRE(state.status == StormContinuityStatus::Tracking);
    REQUIRE(state.sample_count == storm_continuity_history_capacity);
    REQUIRE(state.samples.front().time_s == Approx(120.0));
    REQUIRE(state.samples.back().time_s == Approx(360.0));
    REQUIRE(state.velocity_x_ms == Approx(12.0));
    REQUIRE(state.velocity_y_ms == Approx(-3.0));
    REQUIRE(state.branch_id == 1);
}

TEST_CASE("continuity state copy produces exact resumed history", "[diagnostics][storm_tracker][continuity][restart]")
{
    StormContinuityThresholds thresholds;
    thresholds.position_uncertainty_m = 75.0;
    StormContinuityState continuous;
    update_storm_continuity(continuous, detected_at(100.0, 200.0), 0.0, thresholds);
    update_storm_continuity(continuous, detected_at(700.0, 500.0), 60.0, thresholds);

    StormContinuityState resumed           = continuous;
    const auto           third             = detected_at(1300.0, 800.0);
    const auto           continuous_result = update_storm_continuity(continuous, third, 120.0, thresholds);
    const auto           resumed_result    = update_storm_continuity(resumed, third, 120.0, thresholds);

    REQUIRE(continuous_result.status == resumed_result.status);
    REQUIRE(continuous_result.sample_count == resumed_result.sample_count);
    REQUIRE(continuous_result.velocity_x_ms == resumed_result.velocity_x_ms);
    REQUIRE(continuous_result.velocity_y_ms == resumed_result.velocity_y_ms);
    REQUIRE(continuous_result.velocity_uncertainty_ms == resumed_result.velocity_uncertainty_ms);
    REQUIRE(continuous_result.continuity_confidence == resumed_result.continuity_confidence);
}

TEST_CASE("split and merger topology require consecutive complete snapshots",
          "[diagnostics][storm_tracker][continuity][branches][hysteresis]")
{
    StormContinuityState state;
    StormContinuityThresholds thresholds;
    thresholds.minimum_samples              = 2;
    thresholds.topology_confirmation_samples = 2;
    thresholds.position_uncertainty_m       = 100.0;
    thresholds.maximum_acceleration_ms2     = 0.5;

    update_storm_continuity(state, branches_at({{0.0, 0.0}}), 0.0, thresholds);
    REQUIRE(state.topology == StormBranchTopology::Unknown);
    update_storm_continuity(state, branches_at({{600.0, 0.0}}), 60.0, thresholds);
    REQUIRE(state.topology == StormBranchTopology::Single);
    REQUIRE(state.topology_event == StormBranchTopologyEvent::None);
    REQUIRE(state.branch_id == 1);

    update_storm_continuity(
        state, branches_at({{1200.0, 0.0}, {1200.0, 900.0}}), 120.0, thresholds);
    REQUIRE(state.topology == StormBranchTopology::Single);
    REQUIRE(state.topology_event == StormBranchTopologyEvent::SplitCandidate);
    REQUIRE(state.pending_topology_samples == 1);
    REQUIRE(state.selected_candidate_rank == 0);

    update_storm_continuity(
        state, branches_at({{1800.0, 0.0}, {1800.0, 900.0}}), 180.0, thresholds);
    REQUIRE(state.topology == StormBranchTopology::Multiple);
    REQUIRE(state.topology_event == StormBranchTopologyEvent::SplitConfirmed);
    REQUIRE(state.topology_event_id == 1);
    REQUIRE(state.branch_id == 1);

    update_storm_continuity(state, branches_at({{2400.0, 0.0}}), 240.0, thresholds);
    REQUIRE(state.topology == StormBranchTopology::Multiple);
    REQUIRE(state.topology_event == StormBranchTopologyEvent::MergerCandidate);
    update_storm_continuity(state, branches_at({{3000.0, 0.0}}), 300.0, thresholds);
    REQUIRE(state.topology == StormBranchTopology::Single);
    REQUIRE(state.topology_event == StormBranchTopologyEvent::MergerConfirmed);
    REQUIRE(state.topology_event_id == 2);
    REQUIRE(state.branch_id == 1);
    REQUIRE(valid_storm_continuity_state(state));
}

TEST_CASE("Bunkers branch preference discriminates motion rather than helicity sign",
          "[diagnostics][storm_tracker][continuity][branches][bunkers]")
{
    StormContinuityThresholds thresholds;
    thresholds.minimum_samples          = 2;
    thresholds.position_uncertainty_m   = 700.0;
    thresholds.maximum_acceleration_ms2 = 0.5;

    StormContinuityState state;
    update_storm_continuity(state, branches_at({{0.0, 0.0}}), 0.0, thresholds);
    update_storm_continuity(state, branches_at({{600.0, 0.0}}), 60.0, thresholds);

    auto split = branches_at({{1200.0, 600.0}, {1200.0, -600.0}});
    // Deliberately make the preferred right-moving branch negatively rotating
    // and the other branch positively rotating. Selection must not consult it.
    split.branch_candidates[0].rotation_class = StormRotationClass::Positive;
    split.branch_candidates[1].rotation_class = StormRotationClass::Negative;
    const StormBranchAssociationContext right_preference{
        true,
        StormBranchPreference::BunkersRight,
        10.0,
        -10.0,
        10.0,
        10.0,
        2.0,
    };
    update_storm_continuity(state, split, 120.0, thresholds, right_preference);

    REQUIRE(state.status == StormContinuityStatus::Tracking);
    REQUIRE(state.selected_candidate_rank == 1);
    REQUIRE(state.selected_component_label == 101);
    REQUIRE(state.selected_motion_class == StormBranchMotionClass::BunkersRight);
    REQUIRE(state.samples[state.sample_count - 1].y_m == Approx(-600.0));
    REQUIRE(state.selected_bunkers_right_mismatch_ms == Approx(0.0));
    REQUIRE(state.selected_bunkers_left_mismatch_ms == Approx(20.0));
}

TEST_CASE("distant independent convection cannot manufacture a tracked split",
          "[diagnostics][storm_tracker][continuity][branches][topology]")
{
    StormContinuityThresholds thresholds;
    thresholds.minimum_samples              = 2;
    thresholds.topology_confirmation_samples = 1;
    thresholds.position_uncertainty_m       = 100.0;
    thresholds.maximum_speed_ms             = 40.0;
    thresholds.maximum_acceleration_ms2     = 0.1;
    StormContinuityState state;
    update_storm_continuity(state, branches_at({{0.0, 0.0}}), 0.0, thresholds);
    update_storm_continuity(state, branches_at({{600.0, 0.0}}), 60.0, thresholds);
    REQUIRE(state.topology == StormBranchTopology::Single);

    update_storm_continuity(
        state, branches_at({{1200.0, 0.0}, {10000.0, 10000.0}}), 120.0, thresholds);
    REQUIRE(state.status == StormContinuityStatus::Tracking);
    REQUIRE(state.selected_candidate_rank == 0);
    REQUIRE(state.topology == StormBranchTopology::Single);
    REQUIRE(state.topology_event == StormBranchTopologyEvent::None);
    REQUIRE(state.topology_event_id == 0);
}

TEST_CASE("unresolved equivalent branches remain ambiguous and overflow never selects",
          "[diagnostics][storm_tracker][continuity][branches][ambiguity]")
{
    StormContinuityThresholds thresholds;
    thresholds.minimum_samples          = 2;
    thresholds.lost_after_misses        = 4;
    thresholds.position_uncertainty_m   = 700.0;
    thresholds.maximum_acceleration_ms2 = 0.5;
    StormContinuityState state;
    update_storm_continuity(state, branches_at({{0.0, 0.0}}), 0.0, thresholds);
    update_storm_continuity(state, branches_at({{600.0, 0.0}}), 60.0, thresholds);

    const std::size_t before = state.sample_count;
    update_storm_continuity(
        state, branches_at({{1200.0, 600.0}, {1200.0, -600.0}}), 120.0, thresholds);
    REQUIRE(state.status == StormContinuityStatus::Ambiguous);
    REQUIRE(state.reason == StormContinuityReason::SpatialAmbiguity);
    REQUIRE(state.sample_count == before);
    REQUIRE_FALSE(state.selected_candidate_available);

    auto overflow = branches_at({{1800.0, 0.0}, {1800.0, 900.0}});
    overflow.branch_candidate_overflow = true;
    overflow.candidate_component_count = storm_branch_candidate_capacity + 1;
    update_storm_continuity(state, overflow, 180.0, thresholds);
    REQUIRE(state.reason == StormContinuityReason::BranchCandidateOverflow);
    REQUIRE_FALSE(state.selected_candidate_available);
    REQUIRE(state.sample_count == before);
}

TEST_CASE("resolved dominant primary survives lower-score branch report overflow",
          "[diagnostics][storm_tracker][continuity][branches][overflow]")
{
    StormContinuityThresholds thresholds;
    thresholds.minimum_samples          = 2;
    thresholds.lost_after_misses        = 4;
    thresholds.position_uncertainty_m   = 700.0;
    thresholds.maximum_acceleration_ms2 = 0.5;
    StormContinuityState state;
    update_storm_continuity(state, branches_at({{0.0, 0.0}}), 0.0, thresholds);
    update_storm_continuity(state, branches_at({{600.0, 0.0}}), 60.0, thresholds);

    auto overflow = branches_at({{1200.0, 0.0}, {1200.0, 5000.0}, {1200.0, -5000.0}, {9000.0, 9000.0}});
    overflow.status                    = StormDetectionStatus::Detected;
    overflow.confidence                = 0.98;
    overflow.branch_candidate_overflow = true;
    overflow.candidate_component_count = storm_branch_candidate_capacity + 7;
    overflow.branch_candidates[0].score_fraction = 0.98;
    for (std::size_t rank = 1; rank < storm_branch_candidate_capacity; ++rank)
        overflow.branch_candidates[rank].score_fraction = 0.005;

    update_storm_continuity(state, overflow, 120.0, thresholds);
    REQUIRE(state.status == StormContinuityStatus::Tracking);
    REQUIRE(state.reason == StormContinuityReason::None);
    REQUIRE(state.selected_candidate_available);
    REQUIRE(state.selected_candidate_rank == 0);
    REQUIRE(state.selected_component_label == overflow.branch_candidates[0].component_label);
    REQUIRE(state.motion_available);
    REQUIRE(state.topology == StormBranchTopology::Single);
    REQUIRE(state.topology_event == StormBranchTopologyEvent::None);
    REQUIRE(state.velocity_x_ms == Approx(10.0));
    REQUIRE(state.velocity_y_ms == Approx(0.0));
}

TEST_CASE("continuity state validation rejects impossible persisted claims",
          "[diagnostics][storm_tracker][continuity][validation]")
{
    StormContinuityState invalid_enum;
    invalid_enum.status = static_cast<StormContinuityStatus>(999);
    REQUIRE_FALSE(valid_storm_continuity_state(invalid_enum));

    StormContinuityState inconsistent_motion;
    inconsistent_motion.status           = StormContinuityStatus::Tracking;
    inconsistent_motion.motion_available = true;
    inconsistent_motion.sample_count     = 1;
    inconsistent_motion.samples[0]       = {0.0, 0.0, 0.0, 0.9};
    REQUIRE_FALSE(valid_storm_continuity_state(inconsistent_motion));

    StormContinuityState inconsistent_speed;
    inconsistent_speed.velocity_x_ms = 3.0;
    inconsistent_speed.velocity_y_ms = 4.0;
    inconsistent_speed.speed_ms      = 4.0;
    REQUIRE_FALSE(valid_storm_continuity_state(inconsistent_speed));

    StormContinuityState invalid_topology;
    invalid_topology.topology = static_cast<StormBranchTopology>(999);
    REQUIRE_FALSE(valid_storm_continuity_state(invalid_topology));

    StormContinuityState inconsistent_pending;
    inconsistent_pending.pending_topology = StormBranchTopology::Multiple;
    REQUIRE_FALSE(valid_storm_continuity_state(inconsistent_pending));

    StormContinuityState invalid_selected_rank;
    invalid_selected_rank.selected_candidate_available = true;
    invalid_selected_rank.selected_candidate_rank      = storm_branch_candidate_capacity;
    REQUIRE_FALSE(valid_storm_continuity_state(invalid_selected_rank));

    StormFeatureDetection invalid_detection;
    invalid_detection.status = static_cast<StormDetectionStatus>(999);
    StormContinuityState valid_state;
    REQUIRE_THROWS_WITH(update_storm_continuity(valid_state, invalid_detection, 0.0, {}),
                        Catch::Matchers::Contains("are invalid"));
}
