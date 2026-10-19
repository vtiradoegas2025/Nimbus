#include "auto_sounding.hpp"

#include "soundings/schemes/cm1/cm1_sounding.hpp"
#include "soundings/schemes/sharpy/sharpy_sounding.hpp"
#include "util/log.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

std::string lower_extension(const std::string& path)
{
    std::string value = std::filesystem::path(path).extension().string();
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::string first_substantive_line(const std::string& path)
{
    std::ifstream input(path);
    std::array<char, 4097> buffer{};
    while (input.getline(buffer.data(), static_cast<std::streamsize>(buffer.size())))
    {
        std::string line(buffer.data());
        const auto bang = line.find('!');
        const auto hash = line.find('#');
        const auto cut = std::min(
            bang == std::string::npos ? line.size() : bang,
            hash == std::string::npos ? line.size() : hash);
        line.resize(cut);
        std::istringstream stream(line);
        std::vector<std::string> tokens;
        std::string token;
        while (stream >> token)
        {
            tokens.push_back(token);
        }
        if (tokens.empty())
        {
            continue;
        }
        if (tokens.size() == 3)
        {
            bool numeric = true;
            for (std::string value : tokens)
            {
                std::replace(value.begin(), value.end(), 'd', 'e');
                std::replace(value.begin(), value.end(), 'D', 'E');
                std::size_t consumed = 0;
                try
                {
                    (void)std::stod(value, &consumed);
                    numeric = numeric && consumed == value.size();
                }
                catch (const std::exception&)
                {
                    numeric = false;
                }
            }
            if (numeric)
            {
                return "cm1";
            }
        }
        return line.rfind("TMV_SHARPY_PROFILE_V1", 0) == 0 ? "sharpy" : "";
    }
    if (input.fail() && !input.eof())
    {
        throw std::runtime_error(
            "sounding format detection encountered a line longer than 4096 bytes");
    }
    return "";
}

std::string detect_scheme(const std::string& path)
{
    if (!std::filesystem::is_regular_file(path))
    {
        throw std::runtime_error("sounding file is not a regular file: " + path);
    }

    std::ifstream input(path, std::ios::binary);
    std::array<char, 22> magic{};
    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    const auto magic_bytes = input.gcount();
    if ((magic_bytes >= 8
         && ((magic[0] == '\211' && magic[1] == 'H' && magic[2] == 'D'
             && magic[3] == 'F')
             || (magic[0] == 'C' && magic[1] == 'D' && magic[2] == 'F')))
        || (magic_bytes >= 21
            && std::string(magic.data(), 21) == "TMV_SHARPY_PROFILE_V1"))
    {
        return "sharpy";
    }

    const std::string extension = lower_extension(path);
    if (extension == ".h5" || extension == ".hdf5"
        || extension == ".nc" || extension == ".nc4"
        || extension == ".netcdf" || extension == ".tmv")
    {
        return "sharpy";
    }

    const std::string content_hint = first_substantive_line(path);
    if (!content_hint.empty())
    {
        return content_hint;
    }
    throw std::runtime_error(
        "cannot auto-detect sounding format for '" + path
        + "'; expected native CM1 input_sounding, TMV_SHARPY_PROFILE_V1, "
          "HDF5, or netCDF");
}

} // namespace

void AutoSoundingScheme::initialize(const SoundingConfig& config)
{
    config_ = config;
    resolved_scheme_id_ = detect_scheme(config.file_path);
    SoundingConfig resolved = config;
    resolved.scheme_id = resolved_scheme_id_;
    if (resolved_scheme_id_ == "cm1")
    {
        delegate_ = std::make_unique<Cm1SoundingScheme>();
    }
    else
    {
        delegate_ = std::make_unique<SharpySoundingScheme>();
    }
    delegate_->initialize(resolved);
    tmv::log_info("Auto-detected sounding reader: ", resolved_scheme_id_);
}

SoundingData AutoSoundingScheme::load_sounding(const std::string& file_path)
{
    if (!delegate_)
    {
        throw std::runtime_error("automatic sounding reader is not initialized");
    }
    return delegate_->load_sounding(file_path);
}

SoundingData AutoSoundingScheme::interpolate_to_heights(
    const SoundingData& sounding,
    const std::vector<double>& target_heights_m)
{
    if (!delegate_)
    {
        throw std::runtime_error("automatic sounding reader is not initialized");
    }
    return delegate_->interpolate_to_heights(sounding, target_heights_m);
}
