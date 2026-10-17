#pragma once

#include "core/field/field3d.hpp"
#include "core/runtime/portable_scheme_state.hpp"
#include "microphysics/microphysics_base.hpp"

#include <array>
#include <string>
#include <vector>

namespace tmv::thompson
{

struct AerosolProfile
{
    double water_surface_total_m3 = 350.0e6;
    double water_free_troposphere_m3 = 50.0e6;
    double ice_surface_total_m3 = 2.0e6;
    double ice_free_troposphere_m3 = 0.5e6;
};

std::vector<double> flat_profile_sampling_heights(
    int nr,
    int nth,
    int nz,
    const std::vector<double>& layer_thickness_m);

class PrognosticState
{
public:
    static constexpr std::uint32_t portable_version = 1;
    static constexpr double surface_water_friendly_aerosol_m3 = 350.0e6;
    static constexpr double free_troposphere_water_friendly_aerosol_m3 = 50.0e6;
    static constexpr double surface_ice_friendly_aerosol_m3 = 2.0e6;
    static constexpr double free_troposphere_ice_friendly_aerosol_m3 = 0.5e6;

    void prepare(const Field3D& rho_d,
                 const Field3D& qc,
                 const Field3D& qr,
                 const Field3D& qi,
                 const std::vector<double>& profile_sample_height_m,
                 AerosolProfile aerosol_profile = {});
    void prepare(const Field3D& rho_d,
                 const Field3D& qc,
                 const Field3D& qr,
                 const Field3D& qi,
                 const std::vector<double>& profile_sample_height_m,
                 AerosolProfile aerosol_profile,
                 const PhysicalCellDomain& domain);

    MicrophysicsPrognosticScalarList transport_bindings();
    bool validate_process_state(const Field3D& reference,
                                std::string& error) const;
    bool validate_process_state(const Field3D& reference, const PhysicalCellDomain& domain, std::string& error) const;
    bool validate_update(double dt, std::string& error) const;
    bool validate_update(double dt, const PhysicalCellDomain& domain, std::string& error) const;
    bool apply_column_multipliers(double                    dt,
                                  const Field3D&            density,
                                  const PhysicalCellDomain& domain,
                                  std::vector<float>&       column_multipliers,
                                  std::string&              error);
    void commit_update(double dt) noexcept;
    void commit_update(double dt, const PhysicalCellDomain& domain) noexcept;
    void invalidate_transient() noexcept;

    std::vector<Field3D> portable_fields() const;
    bool restore_fields(const std::vector<Field3D>& fields, std::string& error);
    bool bind_aerosol_profile(AerosolProfile aerosol_profile,
                              std::string& error);

    Field3D& cloud_number_tendency() noexcept { return tendencies_[0]; }
    Field3D& rain_number_tendency() noexcept { return tendencies_[1]; }
    Field3D& ice_number_tendency() noexcept { return tendencies_[2]; }
    Field3D& water_aerosol_tendency() noexcept { return tendencies_[3]; }
    Field3D& ice_aerosol_tendency() noexcept { return tendencies_[4]; }

    const Field3D& cloud_number() const noexcept { return moments_[0]; }
    const Field3D& rain_number() const noexcept { return moments_[1]; }
    const Field3D& ice_number() const noexcept { return moments_[2]; }
    const Field3D& water_aerosol() const noexcept { return moments_[3]; }
    const Field3D& ice_aerosol() const noexcept { return moments_[4]; }

private:
    std::array<Field3D, 5> moments_;
    std::array<Field3D, 5> tendencies_;
    AerosolProfile aerosol_profile_{};
    bool aerosol_profile_bound_ = false;
};

} // namespace tmv::thompson
