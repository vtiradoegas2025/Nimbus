#include "core/output/retention_policy.hpp"
#include "core/output/history_index.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iterator>
#include <regex>
#include <set>
#include <unordered_map>
#include <vector>

namespace nimbus::output
{
namespace
{

struct StepCandidate
{
    std::uint64_t         index = 0;
    std::filesystem::path path;
    std::set<std::uint64_t> parents;
    std::string terrain_geometry_version;
};

struct CheckpointCandidate
{
    std::uint64_t         step = 0;
    std::filesystem::path binary;
    std::filesystem::path sidecar;
};

bool regular_file_without_symlink(const std::filesystem::path& path)
{
    std::error_code ec;
    const auto status = std::filesystem::symlink_status(path, ec);
    return !ec && std::filesystem::is_regular_file(status);
}

bool directory_without_symlink(const std::filesystem::path& path)
{
    std::error_code ec;
    const auto status = std::filesystem::symlink_status(path, ec);
    return !ec && std::filesystem::is_directory(status);
}

bool read_manifest_parents(const std::filesystem::path& manifest,
                           std::set<std::uint64_t>&     parents,
                           std::string&                 terrain_geometry_version,
                           std::string&                 error)
{
    std::ifstream input(manifest, std::ios::binary);
    if (!input)
    {
        error = "cannot read completed output manifest '" + manifest.string() + "'";
        return false;
    }
    const std::string json((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    if (!input.good() && !input.eof())
    {
        error = "failed while reading completed output manifest '" + manifest.string() + "'";
        return false;
    }

    static const std::regex arrays(R"("parent_step_indices"\s*:\s*\[([^\]]*)\])");
    static const std::regex integer(R"(([0-9]+))");
    for (std::sregex_iterator array(json.begin(), json.end(), arrays), end; array != end; ++array)
    {
        const std::string body = (*array)[1].str();
        for (std::sregex_iterator value(body.begin(), body.end(), integer); value != end; ++value)
        {
            try
            {
                parents.insert(std::stoull((*value)[1].str()));
            }
            catch (const std::exception&)
            {
                error = "invalid parent step index in completed output manifest '" + manifest.string() + "'";
                return false;
            }
        }
    }
    static const std::regex geometry(
        R"GEOMETRY("storage"\s*:\s*"versioned_npy_geometry"[\s\S]*?"content_sha256"\s*:\s*"([0-9a-fA-F]{64})"[\s\S]*?"publication_generation"\s*:\s*"([0-9a-fA-F]+)")GEOMETRY");
    std::smatch geometry_match;
    if (std::regex_search(json, geometry_match, geometry))
        terrain_geometry_version = geometry_match[2].str() + "/" + geometry_match[1].str();
    return true;
}

bool tree_size_without_links(const std::filesystem::path& path,
                             std::uintmax_t&               bytes,
                             std::string&                 error)
{
    std::error_code ec;
    const auto status = std::filesystem::symlink_status(path, ec);
    if (ec)
    {
        error = "cannot inspect retention target '" + path.string() + "': " + ec.message();
        return false;
    }
    if (std::filesystem::is_symlink(status))
    {
        error = "refusing to prune retention target containing symbolic link '" + path.string() + "'";
        return false;
    }
    if (std::filesystem::is_regular_file(status))
    {
        const auto size = std::filesystem::file_size(path, ec);
        if (ec)
        {
            error = "cannot size retention target '" + path.string() + "': " + ec.message();
            return false;
        }
        bytes += size;
        return true;
    }
    if (!std::filesystem::is_directory(status))
    {
        error = "refusing to prune unexpected filesystem object '" + path.string() + "'";
        return false;
    }
    for (std::filesystem::directory_iterator it(path, ec), end; !ec && it != end; it.increment(ec))
    {
        if (!tree_size_without_links(it->path(), bytes, error))
            return false;
    }
    if (ec)
    {
        error = "cannot enumerate retention target '" + path.string() + "': " + ec.message();
        return false;
    }
    return true;
}

bool remove_validated_tree(const std::filesystem::path& path, std::string& error)
{
    std::error_code ec;
    std::vector<std::filesystem::path> children;
    for (std::filesystem::directory_iterator it(path, ec), end; !ec && it != end; it.increment(ec))
        children.push_back(it->path());
    if (ec)
    {
        error = "cannot enumerate retention target '" + path.string() + "': " + ec.message();
        return false;
    }
    for (const auto& child : children)
    {
        const auto status = std::filesystem::symlink_status(child, ec);
        if (ec || std::filesystem::is_symlink(status))
        {
            error = "retention target changed during removal at '" + child.string() + "'";
            return false;
        }
        if (std::filesystem::is_directory(status))
        {
            if (!remove_validated_tree(child, error))
                return false;
        }
        else if (std::filesystem::is_regular_file(status))
        {
            if (!std::filesystem::remove(child, ec) || ec)
            {
                error = "cannot remove retained output file '" + child.string() + "': " + ec.message();
                return false;
            }
        }
        else
        {
            error = "retention target changed to an unexpected object at '" + child.string() + "'";
            return false;
        }
    }
    if (!std::filesystem::remove(path, ec) || ec)
    {
        error = "cannot remove retained output directory '" + path.string() + "': " + ec.message();
        return false;
    }
    return true;
}

std::filesystem::path normalized_path(const std::filesystem::path& path)
{
    std::error_code ec;
    const auto normalized = std::filesystem::weakly_canonical(path, ec);
    return ec ? path.lexically_normal() : normalized;
}

} // namespace

bool enforce_step_retention(const std::filesystem::path& series_directory,
                            std::size_t                  keep_last_frames,
                            RetentionResult&             result,
                            std::string&                 error)
{
    result = {};
    error.clear();
    if (keep_last_frames == 0 || !std::filesystem::exists(series_directory))
        return true;
    if (!directory_without_symlink(series_directory))
    {
        error = "output retention root is not a real directory: '" + series_directory.string() + "'";
        return false;
    }

    static const std::regex pattern(R"(^step_([0-9]{6,})$)");
    std::vector<StepCandidate> candidates;
    std::error_code ec;
    for (std::filesystem::directory_iterator it(series_directory, ec), end; !ec && it != end; it.increment(ec))
    {
        std::smatch match;
        const auto name = it->path().filename().string();
        if (!std::regex_match(name, match, pattern) || !directory_without_symlink(it->path()))
            continue;
        const auto manifest = it->path() / "manifest.json";
        if (!regular_file_without_symlink(manifest))
            continue;
        StepCandidate candidate;
        try
        {
            candidate.index = std::stoull(match[1].str());
        }
        catch (const std::exception&)
        {
            error = "output step index is outside the supported range: '" + name + "'";
            return false;
        }
        candidate.path = it->path();
        if (!read_manifest_parents(
                manifest, candidate.parents, candidate.terrain_geometry_version, error))
            return false;
        candidates.push_back(std::move(candidate));
    }
    if (ec)
    {
        error = "cannot enumerate output retention root '" + series_directory.string() + "': " + ec.message();
        return false;
    }
    if (candidates.size() <= keep_last_frames)
        return true;

    std::sort(candidates.begin(), candidates.end(), [](const auto& left, const auto& right) {
        return left.index < right.index;
    });
    std::unordered_map<std::uint64_t, const StepCandidate*> by_index;
    for (const auto& candidate : candidates)
        by_index[candidate.index] = &candidate;

    std::set<std::uint64_t> retained;
    for (auto it = candidates.end() - static_cast<std::ptrdiff_t>(keep_last_frames); it != candidates.end(); ++it)
        retained.insert(it->index);
    std::vector<std::uint64_t> pending(retained.begin(), retained.end());
    while (!pending.empty())
    {
        const auto index = pending.back();
        pending.pop_back();
        const auto found = by_index.find(index);
        if (found == by_index.end())
            continue;
        for (const auto parent : found->second->parents)
            if (by_index.count(parent) != 0 && retained.insert(parent).second)
                pending.push_back(parent);
    }
    result.dependency_frames = retained.size() > keep_last_frames ? retained.size() - keep_last_frames : 0;

    for (const auto& candidate : candidates)
    {
        if (retained.count(candidate.index) != 0)
            continue;
        std::uintmax_t bytes = 0;
        if (!tree_size_without_links(candidate.path, bytes, error) || !remove_validated_tree(candidate.path, error))
            return false;
        ++result.removed_items;
        result.reclaimed_bytes += bytes;
    }
    std::set<std::string> retained_geometry;
    for (const auto& candidate : candidates)
        if (retained.count(candidate.index) != 0 && !candidate.terrain_geometry_version.empty())
            retained_geometry.insert(candidate.terrain_geometry_version);
    const auto geometry_root = series_directory / "geometry";
    if (directory_without_symlink(geometry_root))
    {
        std::vector<std::filesystem::path> empty_generations;
        for (std::filesystem::directory_iterator generation(geometry_root, ec), end;
             !ec && generation != end; generation.increment(ec))
        {
            const auto generation_name = generation->path().filename().string();
            if (!directory_without_symlink(generation->path()) || generation_name.empty())
                continue;
            for (std::filesystem::directory_iterator version(generation->path(), ec), version_end;
                 !ec && version != version_end; version.increment(ec))
            {
                const auto digest = version->path().filename().string();
                if (digest.size() != 64 ||
                    !std::all_of(digest.begin(), digest.end(), [](unsigned char c)
                    { return std::isxdigit(c) != 0; }) ||
                    !directory_without_symlink(version->path()) ||
                    retained_geometry.count(generation_name + "/" + digest) != 0)
                    continue;
                std::uintmax_t bytes = 0;
                if (!tree_size_without_links(version->path(), bytes, error) ||
                    !remove_validated_tree(version->path(), error))
                    return false;
                ++result.removed_geometry_versions;
                result.reclaimed_bytes += bytes;
            }
            if (ec)
            {
                error = "cannot enumerate terrain geometry retention root: " + ec.message();
                return false;
            }
            if (std::filesystem::is_empty(generation->path(), ec) && !ec)
                empty_generations.push_back(generation->path());
        }
        if (ec)
        {
            error = "cannot enumerate terrain geometry root: " + ec.message();
            return false;
        }
        for (const auto& generation : empty_generations)
            std::filesystem::remove(generation, ec);
    }
    return rebuild_history_index_if_present(series_directory, error);
}

bool enforce_checkpoint_retention(const std::filesystem::path& checkpoint_directory,
                                  std::size_t                  keep_last_checkpoints,
                                  const std::filesystem::path& protected_checkpoint,
                                  RetentionResult&             result,
                                  std::string&                 error)
{
    result = {};
    error.clear();
    if (keep_last_checkpoints == 0 || !std::filesystem::exists(checkpoint_directory))
        return true;
    if (!directory_without_symlink(checkpoint_directory))
    {
        error = "checkpoint retention root is not a real directory: '" + checkpoint_directory.string() + "'";
        return false;
    }

    static const std::regex pattern(R"(^checkpoint_step_([0-9]{9,})\.nimbus$)");
    std::vector<CheckpointCandidate> candidates;
    std::error_code ec;
    for (std::filesystem::directory_iterator it(checkpoint_directory, ec), end; !ec && it != end; it.increment(ec))
    {
        std::smatch match;
        const auto name = it->path().filename().string();
        if (!std::regex_match(name, match, pattern) || !regular_file_without_symlink(it->path()))
            continue;
        const auto sidecar = std::filesystem::path(it->path().string() + ".json");
        if (!regular_file_without_symlink(sidecar))
            continue;
        CheckpointCandidate candidate;
        try
        {
            candidate.step = std::stoull(match[1].str());
        }
        catch (const std::exception&)
        {
            error = "checkpoint step is outside the supported range: '" + name + "'";
            return false;
        }
        candidate.binary  = it->path();
        candidate.sidecar = sidecar;
        candidates.push_back(std::move(candidate));
    }
    if (ec)
    {
        error = "cannot enumerate checkpoint retention root '" + checkpoint_directory.string() + "': " + ec.message();
        return false;
    }
    if (candidates.size() <= keep_last_checkpoints)
        return true;

    std::sort(candidates.begin(), candidates.end(), [](const auto& left, const auto& right) {
        return left.step < right.step;
    });
    std::set<std::filesystem::path> retained;
    for (auto it = candidates.end() - static_cast<std::ptrdiff_t>(keep_last_checkpoints); it != candidates.end(); ++it)
        retained.insert(normalized_path(it->binary));
    if (!protected_checkpoint.empty())
        retained.insert(normalized_path(protected_checkpoint));

    for (const auto& candidate : candidates)
    {
        if (retained.count(normalized_path(candidate.binary)) != 0)
            continue;
        const auto binary_bytes  = std::filesystem::file_size(candidate.binary, ec);
        if (ec)
        {
            error = "cannot size checkpoint retention target '" + candidate.binary.string() + "': " + ec.message();
            return false;
        }
        const auto sidecar_bytes = std::filesystem::file_size(candidate.sidecar, ec);
        if (ec)
        {
            error = "cannot size checkpoint retention target '" + candidate.sidecar.string() + "': " + ec.message();
            return false;
        }
        if (!std::filesystem::remove(candidate.binary, ec) || ec)
        {
            error = "cannot remove checkpoint '" + candidate.binary.string() + "': " + ec.message();
            return false;
        }
        if (!std::filesystem::remove(candidate.sidecar, ec) || ec)
        {
            error = "cannot remove checkpoint sidecar '" + candidate.sidecar.string() + "': " + ec.message();
            return false;
        }
        ++result.removed_items;
        result.reclaimed_bytes += binary_bytes + sidecar_bytes;
    }
    return true;
}

bool require_free_space(const std::filesystem::path& destination,
                        std::uintmax_t               required_bytes,
                        std::string&                 error)
{
    error.clear();
    if (required_bytes == 0)
        return true;
    std::filesystem::path probe = destination;
    std::error_code ec;
    while (!probe.empty() && !std::filesystem::exists(probe, ec))
        probe = probe.parent_path();
    if (ec || probe.empty())
    {
        error = "cannot resolve filesystem capacity for '" + destination.string() + "'";
        return false;
    }
    const auto space = std::filesystem::space(probe, ec);
    if (ec)
    {
        error = "cannot query free space for '" + probe.string() + "': " + ec.message();
        return false;
    }
    if (space.available < required_bytes)
    {
        error = "insufficient free space for '" + destination.string() + "': available=" +
                std::to_string(space.available) + " bytes, required=" + std::to_string(required_bytes) + " bytes";
        return false;
    }
    return true;
}

} // namespace nimbus::output
