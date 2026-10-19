#include "core/output/history_index.hpp"

#include "core/infra/sha256.hpp"
#include "core/output/step_manifest.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <regex>
#include <sstream>
#include <vector>

namespace nimbus::output
{
namespace
{
constexpr const char* kSegmentDirectory    = "history_index";
constexpr const char* kPendingFilename     = "history_index.pending.json";
constexpr std::size_t kSegmentBatchRecords = 256;

struct Entry
{
    std::uint64_t index;
    std::string   directory;
    std::string   manifest_sha256;
};
struct Head
{
    std::string   generation;
    std::uint64_t sequence       = 0;
    std::uint64_t reset_sequence = 0;
    std::string   segment;
    std::string   segment_sha256;
};
struct ParsedSegment
{
    std::string        operation;
    std::string        previous_segment;
    std::string        previous_sha256;
    std::vector<Entry> entries;
};

bool read_text(const std::filesystem::path& path, std::string& text, std::string& error)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
    {
        error = "cannot read history index artifact '" + path.string() + "'";
        return false;
    }
    constexpr std::size_t maximum_bytes = 1024U * 1024U * 1024U;
    input.seekg(0, std::ios::end);
    const auto size = input.tellg();
    if (size < 0 || static_cast<std::uintmax_t>(size) > maximum_bytes)
    {
        error = "invalid history index artifact '" + path.string() + "'";
        return false;
    }
    input.seekg(0);
    std::string candidate(static_cast<std::size_t>(size), '\0');
    input.read(candidate.data(), static_cast<std::streamsize>(candidate.size()));
    if (!input || input.peek() != std::char_traits<char>::eof())
    {
        error = "history index artifact changed or could not be read: '" + path.string() + "'";
        return false;
    }
    text.swap(candidate);
    return true;
}

bool extract(const std::string& json, const std::regex& pattern, std::string& value)
{
    std::smatch match;
    if (!std::regex_search(json, match, pattern))
        return false;
    value = match[1].str();
    return true;
}

bool manifest_identity(const std::string& json, std::string& generation, std::uint64_t& index, std::string& directory)
{
    static const std::regex generation_pattern(R"I("publication_generation"\s*:\s*"([0-9a-fA-F]{32})")I");
    static const std::regex index_pattern(R"("step_index"\s*:\s*([0-9]+))");
    static const std::regex directory_pattern(R"I("step_dir"\s*:\s*"(step_[0-9]{6,})")I");
    std::string             raw_index;
    if (!extract(json, generation_pattern, generation) || !extract(json, index_pattern, raw_index) ||
        !extract(json, directory_pattern, directory))
        return false;
    try
    {
        index = std::stoull(raw_index);
    }
    catch (...)
    {
        return false;
    }
    std::ostringstream expected;
    expected << "step_" << std::setfill('0') << std::setw(6) << index;
    return directory == expected.str();
}

std::string head_payload(const Head& head)
{
    std::ostringstream out;
    out << "{\n  \"schema\": \"nimbus.history.index.head\",\n  \"version\": 1,\n"
        << "  \"publication_generation\": \"" << head.generation << "\",\n"
        << "  \"sequence\": " << head.sequence << ",\n"
        << "  \"reset_sequence\": " << head.reset_sequence << ",\n"
        << "  \"segment\": \"" << head.segment << "\",\n"
        << "  \"segment_sha256\": \"" << head.segment_sha256 << "\"\n}";
    return out.str();
}

std::string checked_json(const std::string& payload)
{
    return payload.substr(0, payload.size() - 1) + ",\n  \"checksum\": \"" + nimbus::crypto::sha256_hex(payload) +
           "\"\n}\n";
}

std::string segment_payload(const Head*               previous,
                            const std::string&        generation,
                            std::uint64_t             sequence,
                            const char*               operation,
                            const std::vector<Entry>& entries)
{
    std::ostringstream out;
    out << "{\n  \"schema\": \"nimbus.history.index.segment\",\n  \"version\": 1,\n"
        << "  \"publication_generation\": \"" << generation << "\",\n"
        << "  \"sequence\": " << sequence << ",\n  \"operation\": \"" << operation << "\",\n  \"previous_segment\": ";
    if (previous)
        out << "\"" << previous->segment << "\",\n  \"previous_segment_sha256\": \"" << previous->segment_sha256
            << "\",\n";
    else
        out << "null,\n  \"previous_segment_sha256\": null,\n";
    out << "  \"entries\": [";
    for (std::size_t i = 0; i < entries.size(); ++i)
    {
        if (i)
            out << ',';
        out << "\n    {\"step_index\": " << entries[i].index << ", \"step_dir\": \"" << entries[i].directory
            << "\", \"manifest_sha256\": \"" << entries[i].manifest_sha256 << "\"}";
    }
    if (!entries.empty())
        out << "\n  ";
    out << "]\n}";
    return out.str();
}

bool parse_segment(const std::string& json, ParsedSegment& parsed)
{
    static const std::regex operation_pattern(R"I("operation"\s*:\s*"(append|reset)")I");
    static const std::regex previous_pattern(R"I("previous_segment"\s*:\s*"([^"]+)")I");
    static const std::regex previous_hash_pattern(R"I("previous_segment_sha256"\s*:\s*"([0-9a-f]{64})")I");
    static const std::regex entry_pattern(
        R"I(\{"step_index": ([0-9]+), "step_dir": "(step_[0-9]{6,})", "manifest_sha256": "([0-9a-f]{64})"\})I");
    if (!extract(json, operation_pattern, parsed.operation))
        return false;
    const bool has_previous = extract(json, previous_pattern, parsed.previous_segment);
    const bool has_hash     = extract(json, previous_hash_pattern, parsed.previous_sha256);
    if (has_previous != has_hash)
        return false;
    parsed.entries.clear();
    for (std::sregex_iterator it(json.begin(), json.end(), entry_pattern), end; it != end; ++it)
        parsed.entries.push_back({std::stoull((*it)[1].str()), (*it)[2].str(), (*it)[3].str()});
    return parsed.entries.size() <= kSegmentBatchRecords;
}

bool load_head(
    const std::filesystem::path& series, const std::string& generation, Head& head, bool& exists, std::string& error)
{
    const auto path = series / kHistoryIndexFilename;
    exists          = std::filesystem::exists(path);
    if (!exists)
        return true;
    std::string json, raw_sequence, raw_reset, checksum;
    if (!read_text(path, json, error))
        return false;
    static const std::regex generation_pattern(R"I("publication_generation"\s*:\s*"([0-9a-fA-F]{32})")I");
    static const std::regex sequence_pattern(R"("sequence"\s*:\s*([0-9]+))");
    static const std::regex reset_pattern(R"("reset_sequence"\s*:\s*([0-9]+))");
    static const std::regex segment_pattern(
        R"I("segment"\s*:\s*"(history_index/(?:active_[ab]|segment_[0-9]{20}_[0-9a-f]{16})\.json)")I");
    static const std::regex hash_pattern(R"I("segment_sha256"\s*:\s*"([0-9a-f]{64})")I");
    static const std::regex checksum_pattern(R"I("checksum"\s*:\s*"([0-9a-f]{64})")I");
    if (json.find("\"schema\": \"nimbus.history.index.head\"") == std::string::npos ||
        !extract(json, generation_pattern, head.generation) || head.generation != generation ||
        !extract(json, sequence_pattern, raw_sequence) || !extract(json, reset_pattern, raw_reset) ||
        !extract(json, segment_pattern, head.segment) || !extract(json, hash_pattern, head.segment_sha256) ||
        !extract(json, checksum_pattern, checksum))
    {
        error = "current history index head is invalid";
        return false;
    }
    try
    {
        head.sequence       = std::stoull(raw_sequence);
        head.reset_sequence = std::stoull(raw_reset);
    }
    catch (...)
    {
        error = "current history index sequence is invalid";
        return false;
    }
    if (nimbus::crypto::sha256_hex(head_payload(head)) != checksum)
    {
        error = "current history index head checksum is invalid";
        return false;
    }
    std::string segment;
    if (!read_text(series / head.segment, segment, error) || nimbus::crypto::sha256_hex(segment) != head.segment_sha256)
    {
        error = "current history index segment checksum is invalid";
        return false;
    }
    ParsedSegment parsed;
    if (!parse_segment(segment, parsed))
    {
        error = "current history index segment contract is invalid";
        return false;
    }
    return true;
}

bool publish_segment(const std::filesystem::path& series,
                     const std::string&           generation,
                     const Head*                  previous,
                     const char*                  operation,
                     const std::vector<Entry>&    entries,
                     std::string&                 error)
{
    const std::uint64_t sequence = previous ? previous->sequence + 1 : 1;
    Head                base;
    const Head*         segment_previous = previous;
    std::vector<Entry>  segment_entries  = entries;
    std::string         active_name      = std::string(kSegmentDirectory) + "/active_a.json";
    if (previous && std::string(operation) == "append")
    {
        std::string   current_json;
        ParsedSegment current;
        if (!read_text(series / previous->segment, current_json, error) || !parse_segment(current_json, current))
            return false;
        if (current.operation == "append" && current.entries.size() + entries.size() <= kSegmentBatchRecords)
        {
            current.entries.insert(current.entries.end(), entries.begin(), entries.end());
            segment_entries = std::move(current.entries);
            if (!current.previous_segment.empty())
            {
                base             = {generation,
                                    previous->sequence,
                                    previous->reset_sequence,
                                    current.previous_segment,
                                    current.previous_sha256};
                segment_previous = &base;
            }
            else
                segment_previous = nullptr;
            active_name = previous->segment.find("active_a.json") != std::string::npos
                              ? std::string(kSegmentDirectory) + "/active_b.json"
                              : std::string(kSegmentDirectory) + "/active_a.json";
        }
        else
        {
            std::ostringstream sealed;
            sealed << kSegmentDirectory << "/segment_" << std::setfill('0') << std::setw(20) << previous->sequence
                   << '_' << previous->segment_sha256.substr(0, 16) << ".json";
            if (!publish_output_json_atomically(series / sealed.str(), current_json, error))
                return false;
            base = {generation, previous->sequence, previous->reset_sequence, sealed.str(), previous->segment_sha256};
            segment_previous = &base;
            active_name      = std::string(kSegmentDirectory) + "/active_a.json";
        }
    }
    const std::string payload = segment_payload(segment_previous, generation, sequence, operation, segment_entries);
    const std::string segment_json = checked_json(payload);
    const std::string segment_hash = nimbus::crypto::sha256_hex(segment_json);
    std::error_code   ec;
    std::filesystem::create_directories(series / kSegmentDirectory, ec);
    if (ec)
    {
        error = "cannot create history index segment directory: " + ec.message();
        return false;
    }
    if (!publish_output_json_atomically(series / active_name, segment_json, error))
        return false;
    const Head head{generation, sequence, previous ? previous->reset_sequence : 0, active_name, segment_hash};
    return publish_output_json_atomically(series / kHistoryIndexFilename, checked_json(head_payload(head)), error);
}

bool rebuild(const std::filesystem::path& series,
             const std::string&           generation,
             const Head*                  previous,
             std::string&                 error)
{
    static const std::regex pattern(R"(^step_([0-9]{6,})$)");
    std::vector<Entry>      entries;
    std::error_code         ec;
    for (std::filesystem::directory_iterator it(series, ec), end; !ec && it != end; it.increment(ec))
    {
        const std::string name = it->path().filename().string();
        std::smatch       match;
        if (!it->is_directory() || !std::regex_match(name, match, pattern))
            continue;
        const auto path = it->path() / "manifest.json";
        if (!std::filesystem::is_regular_file(path))
            continue;
        std::string   manifest, manifest_generation, directory;
        std::uint64_t index = 0;
        if (!read_text(path, manifest, error))
            return false;
        if (!manifest_identity(manifest, manifest_generation, index, directory) || manifest_generation != generation ||
            manifest.find("\"publication_state\": \"complete\"") == std::string::npos)
            continue;
        entries.push_back({index, directory, nimbus::crypto::sha256_hex(manifest)});
    }
    if (ec)
    {
        error = "cannot enumerate history series: " + ec.message();
        return false;
    }
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) { return a.index < b.index; });
    const std::uint64_t                sequence = previous ? previous->sequence + 1 : 1;
    Head                               base;
    const Head*                        chain  = nullptr;
    std::size_t                        offset = 0;
    bool                               first  = true;
    std::vector<std::filesystem::path> reachable;
    do
    {
        const std::size_t        count = std::min(kSegmentBatchRecords, entries.size() - offset);
        const std::vector<Entry> batch(entries.begin() + static_cast<std::ptrdiff_t>(offset),
                                       entries.begin() + static_cast<std::ptrdiff_t>(offset + count));
        const std::string  payload = segment_payload(chain, generation, sequence, first ? "reset" : "append", batch);
        const std::string  json    = checked_json(payload);
        const std::string  hash    = nimbus::crypto::sha256_hex(json);
        std::ostringstream name;
        name << kSegmentDirectory << "/segment_" << std::setfill('0') << std::setw(20) << sequence << '_'
             << hash.substr(0, 16) << ".json";
        std::error_code create_error;
        std::filesystem::create_directories(series / kSegmentDirectory, create_error);
        if (create_error || !publish_output_json_atomically(series / name.str(), json, error))
            return false;
        base = {generation, sequence, sequence, name.str(), hash};
        reachable.push_back((series / name.str()).lexically_normal());
        chain = &base;
        offset += count;
        first = false;
    } while (offset < entries.size());
    if (!publish_output_json_atomically(series / kHistoryIndexFilename, checked_json(head_payload(base)), error))
        return false;
    std::error_code gc_error;
    for (std::filesystem::directory_iterator it(series / kSegmentDirectory, gc_error), end; !gc_error && it != end;
         it.increment(gc_error))
    {
        const auto candidate = it->path().lexically_normal();
        if (std::find(reachable.begin(), reachable.end(), candidate) == reachable.end() && it->is_regular_file())
            std::filesystem::remove(candidate, gc_error);
    }
    if (gc_error)
    {
        error = "cannot collect unreachable history index segments: " + gc_error.message();
        return false;
    }
    return sync_directory_to_disk(series / kSegmentDirectory, error);
}

bool recover_pending(const std::filesystem::path& series, const std::string& generation, std::string& error)
{
    const auto pending = series / kPendingFilename;
    if (!std::filesystem::exists(pending))
        return true;
    Head head;
    bool exists = false;
    if (!load_head(series, generation, head, exists, error) ||
        !rebuild(series, generation, exists ? &head : nullptr, error))
        return false;
    std::error_code ec;
    std::filesystem::remove(pending, ec);
    return !ec && sync_directory_to_disk(series, error);
}
} // namespace

bool prepare_history_index_update(const std::filesystem::path& series, const std::string& manifest, std::string& error)
{
    std::string   generation, directory;
    std::uint64_t index = 0;
    if (!manifest_identity(manifest, generation, index, directory))
        return true;
    if (!recover_pending(series, generation, error))
        return false;
    const std::string pending =
        "{\"publication_generation\":\"" + generation + "\",\"step_index\":" + std::to_string(index) + "}\n";
    return publish_output_json_atomically(series / kPendingFilename, pending, error);
}

bool publish_history_index_after_manifest(const std::filesystem::path& series,
                                          const std::string&           manifest,
                                          std::string&                 error)
{
    std::string   generation, directory;
    std::uint64_t index = 0;
    if (!manifest_identity(manifest, generation, index, directory))
        return true;
    if (const char* inject = std::getenv(
            "NIMBUS_TEST_HISTORY_INDEX_FAIL_AFTER_MANIFEST");
        inject != nullptr && std::string(inject) == "1")
    {
        error = "injected post-manifest history-index failure";
        return false;
    }
    Head head;
    bool exists = false;
    if (!load_head(series, generation, head, exists, error))
        return false;
    const Entry entry{index, directory, nimbus::crypto::sha256_hex(manifest)};
    if (!publish_segment(series, generation, exists ? &head : nullptr, "append", {entry}, error))
        return false;
    std::error_code ec;
    std::filesystem::remove(series / kPendingFilename, ec);
    return !ec && sync_directory_to_disk(series, error);
}

bool rebuild_history_index_if_present(const std::filesystem::path& series, std::string& error)
{
    const auto path = series / kHistoryIndexFilename;
    if (!std::filesystem::exists(path))
        return true;
    std::string json, generation;
    if (!read_text(path, json, error))
        return false;
    static const std::regex generation_pattern(R"I("publication_generation"\s*:\s*"([0-9a-fA-F]{32})")I");
    if (!extract(json, generation_pattern, generation))
        return false;
    Head head;
    bool exists = false;
    if (!load_head(series, generation, head, exists, error))
        return false;
    return rebuild(series, generation, &head, error);
}
} // namespace nimbus::output
