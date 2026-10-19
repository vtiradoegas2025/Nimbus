/**
 * @file test_radar_zdr.cpp
 * @brief Dimensional checks for the synthetic bulk ZDR rain-shape estimate.
 */

#include "catch2/catch.hpp"
#include "radar/radar_base.hpp"
#include "radar/polarimetric/rain_mp.hpp"
#include "radar/schemes/zdr/zdr.hpp"

#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

struct ZdrSample
{
    float zdr_db = 0.0f;
    float reflectivity_dbz = 0.0f;
    float reflectivity_linear = 0.0f;
};

ZdrSample compute_simple_zdr(float q_rain, const Field3D* rho_dry)
{
    Field3D qr(1, 1, 1, q_rain);
    RadarStateView state;
    state.NR = 1;
    state.NTH = 1;
    state.NZ = 1;
    state.qr = &qr;
    state.rho = rho_dry;

    RadarConfig config;
    config.operator_tier = "simple";
    ZDRScheme scheme;
    scheme.initialize(config, 1, 1, 1);
    RadarOut output;
    scheme.compute(config, state, output);
    return {
        output.ZDR_dB(0, 0, 0),
        output.Z_dBZ(0, 0, 0),
        output.Ze_linear(0, 0, 0),
    };
}

TEST_CASE("Zhang 2021 rain operator reproduces published polynomial coefficients",
          "[radar][polarimetric][reference][units]")
{
    constexpr double qr_for_dm_one_mm = 9.817477042468104e-5;
    const auto value = tmv::radar::polarimetric::zhang2021_rain_mp(
        qr_for_dm_one_mm, 1.0);
    REQUIRE(value.dm_mm == Approx(1.0).margin(1.0e-12));
    REQUIRE(value.zh_linear == Approx(358.4419553155874).margin(1.0e-9));
    REQUIRE(value.zdr_db == Approx(0.5372479201940424).margin(1.0e-12));
    REQUIRE(value.kdp_deg_km == Approx(0.009345452746266237).margin(1.0e-14));
    REQUIRE(value.rhohv == Approx(0.9985833).margin(1.0e-12));

    const auto dry = tmv::radar::polarimetric::zhang2021_rain_mp(0.0, 1.0);
    REQUIRE(dry.zh_linear == 0.0);
    REQUIRE(dry.zdr_db == 0.0);
    REQUIRE(dry.kdp_deg_km == 0.0);
    REQUIRE(dry.rhohv == 1.0);
    const auto subresolution =
        tmv::radar::polarimetric::zhang2021_rain_mp(
            6.135923151542565e-10, 1.0);
    REQUIRE(subresolution.zh_linear == 0.0);
    REQUIRE(subresolution.zdr_db == 0.0);
    REQUIRE(subresolution.kdp_deg_km == 0.0);
    REQUIRE(subresolution.rhohv == 1.0);
    const auto weak_rain =
        tmv::radar::polarimetric::zhang2021_rain_mp(
            1.5707963267948966e-7, 1.0);
    REQUIRE(weak_rain.dm_mm == Approx(0.2).margin(1.0e-12));
    REQUIRE(weak_rain.kdp_deg_km == 0.0);
    const auto light_rain =
        tmv::radar::polarimetric::zhang2021_rain_mp(
            6.135923151542565e-6, 1.0);
    REQUIRE(light_rain.dm_mm == Approx(0.5).margin(1.0e-12));
    REQUIRE(light_rain.rhohv == 1.0);
    REQUIRE_THROWS_AS(
        tmv::radar::polarimetric::zhang2021_rain_mp(-1.0e-6, 1.0),
        std::invalid_argument);
}

TEST_CASE("maintained rain polarimetry fails closed on frozen species",
          "[radar][polarimetric][atomicity][scope]")
{
    Field3D rho(2, 1, 1, 1.0f), qr(2, 1, 1, 1.0e-4f), qs(2, 1, 1, 0.0f);
    RadarStateView state;
    state.NR = 2;
    state.NTH = state.NZ = 1;
    state.rho = &rho;
    state.qr = &qr;
    state.qs = &qs;
    RadarConfig config;
    config.operator_tier = "zhang2021_rain_mp";
    ZDRScheme scheme;
    scheme.initialize(config, 2, 1, 1);
    RadarOut output;
    output.initialize(2, 1, 1);
    output.Ze_linear.fill(11.0f);
    output.Z_dBZ.fill(12.0f);
    output.ZH_dBZ.fill(13.0f);
    output.ZV_dBZ.fill(14.0f);
    output.ZDR_dB.fill(7.0f);
    output.KDP_deg_km.fill(3.0f);
    output.rhohv.fill(0.9f);
    const RadarOut original = output;
    const Field3D original_rho = rho;
    const Field3D original_qr = qr;
    qs(1, 0, 0) = 1.0e-8f;
    const Field3D failing_qs = qs;
    REQUIRE_THROWS_AS(scheme.compute(config, state, output), std::domain_error);
    for (const auto& fields : {
             std::pair<const Field3D*, const Field3D*>{
                 &output.Ze_linear, &original.Ze_linear},
             {&output.Z_dBZ, &original.Z_dBZ},
             {&output.ZH_dBZ, &original.ZH_dBZ},
             {&output.ZV_dBZ, &original.ZV_dBZ},
             {&output.ZDR_dB, &original.ZDR_dB},
             {&output.KDP_deg_km, &original.KDP_deg_km},
             {&output.rhohv, &original.rhohv}})
    {
        for (std::size_t cell = 0; cell < fields.first->size(); ++cell)
            REQUIRE(fields.first->data()[cell] == fields.second->data()[cell]);
    }
    for (std::size_t cell = 0; cell < rho.size(); ++cell)
    {
        REQUIRE(rho.data()[cell] == original_rho.data()[cell]);
        REQUIRE(qr.data()[cell] == original_qr.data()[cell]);
        REQUIRE(qs.data()[cell] == failing_qs.data()[cell]);
    }
}

TEST_CASE("maintained rain polarimetry is bitwise deterministic across threads",
          "[radar][polarimetric][openmp][determinism]")
{
    constexpr int nr = 40, nth = 30, nz = 32;
    Field3D rho(nr, nth, nz, 1.0f), qr(nr, nth, nz, 0.0f);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
                qr(i, j, k) = 5.0e-5f +
                    static_cast<float>((i + 3 * j + 5 * k) % 100) * 1.0e-6f;
    RadarStateView state;
    state.NR = nr;
    state.NTH = nth;
    state.NZ = nz;
    state.staggering = StaggerType::CGrid;
    state.rho = &rho;
    state.qr = &qr;
    RadarConfig config;
    config.operator_tier = "zhang2021_rain_mp";
    auto compute = [&](int threads)
    {
#ifdef _OPENMP
        omp_set_dynamic(0);
        omp_set_num_threads(threads);
#else
        (void)threads;
#endif
        ZDRScheme scheme;
        scheme.initialize(config, nr, nth, nz);
        RadarOut output;
        scheme.compute(config, state, output);
        return output;
    };
    const RadarOut one = compute(1);
    const RadarOut four = compute(4);
    REQUIRE(one.ZDR_dB.size_r() == nr);
    REQUIRE(one.ZDR_dB.size_th() == nth);
    REQUIRE(one.ZDR_dB.size_z() == nz);
    for (std::size_t index = 0; index < one.ZDR_dB.size(); ++index)
    {
        REQUIRE(one.ZDR_dB.data()[index] == four.ZDR_dB.data()[index]);
        REQUIRE(one.KDP_deg_km.data()[index] == four.KDP_deg_km.data()[index]);
        REQUIRE(one.rhohv.data()[index] == four.rhohv.data()[index]);
    }
}

ZdrSample compute_snow_zdr(float q_snow, const Field3D* rho_dry)
{
    Field3D qs(1, 1, 1, q_snow);
    RadarStateView state;
    state.NR = 1;
    state.NTH = 1;
    state.NZ = 1;
    state.qs = &qs;
    state.rho = rho_dry;

    RadarConfig config;
    config.operator_tier = "polarimetric_fo";
    config.has_qr = false;
    config.has_qs = true;
    ZDRScheme scheme;
    scheme.initialize(config, 1, 1, 1);
    RadarOut output;
    scheme.compute(config, state, output);
    return {
        output.ZDR_dB(0, 0, 0),
        output.Z_dBZ(0, 0, 0),
        output.Ze_linear(0, 0, 0),
    };
}

}  // namespace

TEST_CASE("bulk radar rain rate has precipitation-flux units",
          "[radar][zdr][units]")
{
    // qr=1e-3 kg_water/kg_dry gives the shared bulk Vt=9 m/s. At
    // rho_d=1 kg_dry/m3, rho_d*qr*Vt*3600 = 32.4 kg/m2/h = 32.4 mm/h.
    REQUIRE(radar_bulk_rain_fall_speed_ms(1.0e-3f) ==
            Approx(9.0f).margin(1.0e-6));
    REQUIRE(radar_bulk_rain_rate_mmph(1.0e-3f, 1.0f) ==
            Approx(32.4f).margin(1.0e-5));
    REQUIRE_THROWS_AS(
        radar_bulk_rain_rate_mmph(1.0e-3f, 0.0f), std::invalid_argument);
}

TEST_CASE("ZDR rain-shape heuristic responds to density through rain rate",
          "[radar][zdr][units][synthetic]")
{
    constexpr float q_rain = 5.0e-5f;
    Field3D low_density(1, 1, 1, 0.8f);
    Field3D high_density(1, 1, 1, 1.4f);

    const float low_rate = radar_bulk_rain_rate_mmph(q_rain, 0.8f);
    const float high_rate = radar_bulk_rain_rate_mmph(q_rain, 1.4f);
    REQUIRE(low_rate < 1.0f);
    REQUIRE(high_rate > 1.0f);

    const float low_zdr = compute_simple_zdr(q_rain, &low_density).zdr_db;
    const float high_zdr = compute_simple_zdr(q_rain, &high_density).zdr_db;
    REQUIRE(low_zdr == Approx(-40.0 * std::log10(0.95)).margin(1.0e-5));
    REQUIRE(high_zdr == Approx(-40.0 * std::log10(0.85)).margin(1.0e-5));
    REQUIRE(high_zdr > low_zdr);

    // No density means no defensible mass concentration or rain-rate estimate.
    REQUIRE_THROWS_AS(
        compute_simple_zdr(q_rain, nullptr), std::invalid_argument);
}

TEST_CASE("ZDR reflectivity converts dry-air mixing ratios with live dry density",
          "[radar][zdr][density][SCI-024]")
{
    constexpr float q = 1.0e-5f;
    Field3D thin(1, 1, 1, 0.6f);
    Field3D dense(1, 1, 1, 1.2f);

    // At this small rain content both samples retain the same axis ratio, so
    // the synthetic q_mass^2 law must scale exactly with (rho_d*q)^2.
    const ZdrSample thin_rain = compute_simple_zdr(q, &thin);
    const ZdrSample dense_rain = compute_simple_zdr(q, &dense);
    REQUIRE(dense_rain.reflectivity_linear / thin_rain.reflectivity_linear ==
            Approx(4.0f).margin(2.0e-5));
    REQUIRE(dense_rain.reflectivity_dbz - thin_rain.reflectivity_dbz ==
            Approx(20.0 * std::log10(2.0)).margin(1.0e-5));

    // Frozen species use the same dry-density conversion in the
    // polarimetric-forward path; this guards the formerly unweighted q^2 path.
    const ZdrSample thin_snow = compute_snow_zdr(1.0e-3f, &thin);
    const ZdrSample dense_snow = compute_snow_zdr(1.0e-3f, &dense);
    REQUIRE(dense_snow.reflectivity_linear / thin_snow.reflectivity_linear ==
            Approx(4.0f).margin(2.0e-5));
    REQUIRE(dense_snow.reflectivity_dbz - thin_snow.reflectivity_dbz ==
            Approx(20.0 * std::log10(2.0)).margin(1.0e-5));
}
