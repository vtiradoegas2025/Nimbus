#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace tmv::thompson
{

struct ActivationTableMetadata
{
    static constexpr std::uint32_t format_version = 1;
    static constexpr std::size_t aerosol_counts = 7;
    static constexpr std::size_t updraft_speeds = 9;
    static constexpr std::size_t temperatures = 7;
    static constexpr std::size_t mean_radii = 5;
    static constexpr std::size_t hygroscopicities = 4;
    static constexpr std::size_t element_count = aerosol_counts * updraft_speeds
        * temperatures * mean_radii * hygroscopicities;
    static constexpr const char* upstream_release = "WRF v4.7.1";
    static constexpr const char* upstream_commit = "f52c197";
    static constexpr const char* upstream_artifact = "run/CCN_ACTIVATE.BIN";
    static constexpr const char* upstream_module_sha256 =
        "fabf19e2a9073cff886e882b187080bfdf089d3fd40c0fce1d19bc93b1e5e802";
    static constexpr const char* upstream_license_sha256 =
        "b30611d35ad1922bc741e86d21e1a897234c1a9c9e2af28076e021f565dfbec6";
    static constexpr const char* upstream_sha256 =
        "f2b8d3916560f9046f89f8ac5f32c5292a1800498fd75301e422f147c82a3dbd";
    static constexpr const char* payload_sha256 =
        "547e2592d01bae74e096f3e280e4eacd8ce22562a5a412039b28bed2c2c5d2be";
    static constexpr const char* packaged_sha256 =
        "6f72d69fcedb89b3d9e978e3f5b2a0eda204c5652114a4f4e7e0d033a73f66bf";
    static constexpr const char* packaged_path =
        "data/thompson/ccn_activation.ntact";
    static constexpr std::size_t selected_mean_radius_index = 2;
    static constexpr std::size_t selected_hygroscopicity_index = 1;
    static constexpr double selected_mean_radius_um = 0.04;
    static constexpr double selected_hygroscopicity = 0.4;
};

struct ActivationTable
{
    std::vector<float> fractions;
    std::string payload_sha256;
};

std::size_t activation_flatten(std::size_t aerosol_count,
                               std::size_t updraft_speed,
                               std::size_t temperature,
                               std::size_t mean_radius,
                               std::size_t hygroscopicity);

bool import_wrf_activation_table(const std::filesystem::path& source,
                                 ActivationTable& table,
                                 std::string& error);
bool write_activation_table(const std::filesystem::path& path,
                            const ActivationTable& table,
                            std::string& error);
bool load_activation_table(const std::filesystem::path& path,
                           ActivationTable& table,
                           std::string& error,
                           std::size_t maximum_resident_bytes =
                               std::numeric_limits<std::size_t>::max());
bool validate_activation_table(const ActivationTable& table,
                               std::string& error);

double activated_cloud_number_m3(const ActivationTable& table,
                                 double temperature_k,
                                 double updraft_m_s,
                                 double available_aerosol_m3);

class ActivationTableStore
{
public:
    bool publish_wrf_file(const std::filesystem::path& path,
                          std::string& error);
    bool publish_file(const std::filesystem::path& path,
                      std::string& error,
                      std::size_t maximum_resident_bytes =
                          std::numeric_limits<std::size_t>::max());
    std::shared_ptr<const ActivationTable> snapshot() const noexcept;

private:
    std::shared_ptr<const ActivationTable> table_;
};

} // namespace tmv::thompson
