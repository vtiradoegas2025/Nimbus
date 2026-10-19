#include "catch2/catch.hpp"
#include "core/runtime/checkpoint_format.hpp"

#include <chrono>
#include <iostream>
#include <iterator>
#if !defined(_WIN32)
#include <csignal>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace
{
namespace wire = tmv::checkpoint::wire;
constexpr std::array<std::uint8_t, 8> magic{{'N', 'I', 'M', 'B', 'T', 'E', 'S', 'T'}};
struct Directory
{
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("nimbus_wire_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    Directory() { std::filesystem::create_directories(path); }
    ~Directory() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
};
std::string contents(const std::filesystem::path& path)
{
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

bool publish(int kind, const std::filesystem::path& path, const std::vector<std::uint8_t>& data,
             std::string& error, wire::PublishedFileIntegrity* integrity = nullptr)
{
    if (kind == 0) return wire::write_atomic(path, data, error, "wire-test");
    if (kind == 1) return wire::write_sections_atomic(path, magic, 1,
        {{1, 1, wire::section_required, data}}, error, "wire-test", integrity);
    return wire::write_streaming_sections_atomic(path, magic, 1,
        {{1, 1, wire::section_required, [&](wire::Encoder& out, std::string&) {
            out.raw(data); return out.good();
        }}}, error, "wire-test", integrity);
}
}

TEST_CASE("checkpoint bulk floats preserve scalar wire bytes and special values", "[checkpoint][wire][bulk]")
{
    std::vector<float> values(16387);
    const std::array<std::uint32_t, 12> special{{0, 0x80000000U, 1, 0x007fffffU, 0x00800000U,
        0x7f7fffffU, 0xff7fffffU, 0x7f800000U, 0xff800000U, 0x7fc01234U, 0x7f801234U, 0xffc04321U}};
    std::uint32_t bits = 0x91334567U;
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        bits = bits * 1664525U + 1013904223U;
        const auto value = i < special.size() ? special[i] : bits;
        std::memcpy(&values[i], &value, sizeof(value));
    }
    wire::Encoder scalar, bulk;
    for (float value : values) scalar.floating(value);
    bulk.floating_array(values.data(), values.size());
    REQUIRE(bulk.good());
    CHECK(bulk.bytes == scalar.bytes);
    CHECK(bulk.size() == scalar.size());
    const auto original = values;
    std::vector<std::uint8_t> streamed;
    wire::Encoder sink([&](const std::uint8_t* p, std::size_t n) {
        streamed.insert(streamed.end(), p, p+n); return true;
    });
    sink.floating_array(values.data(), 3);
    sink.floating_array(values.data()+3, values.size()-3);
    CHECK(sink.good());
    CHECK(streamed == scalar.bytes);
    CHECK(std::memcmp(original.data(), values.data(), values.size()*sizeof(float)) == 0);
    wire::Encoder failed([](const std::uint8_t*, std::size_t) { return false; });
    failed.floating_array(values.data(), values.size());
    CHECK_FALSE(failed.good());
    CHECK(failed.size() == 0);
    CHECK(failed.error() == "checkpoint encoder sink failed");
    wire::Encoder empty;
    empty.floating_array(nullptr, 0);
    CHECK(empty.good());
    empty.floating_array(nullptr, 1);
    CHECK_FALSE(empty.good());
    wire::Encoder oversized;
    oversized.floating_array(values.data(), std::numeric_limits<std::size_t>::max());
    CHECK_FALSE(oversized.good());
    CHECK(oversized.bytes.empty());
}

TEST_CASE("checkpoint streaming publication matches envelope and integrity references", "[checkpoint][wire][integrity]")
{
    Directory dir;
    std::vector<std::uint8_t> data(65539);
    for (std::size_t i=0; i<data.size(); ++i) data[i]=static_cast<std::uint8_t>(i*37U);
    std::vector<std::uint8_t> expected;
    std::string error;
    REQUIRE(wire::wrap_sections(magic, 1, {{1,1,wire::section_required,data}}, expected, error));
    for (int kind : {1, 2})
    {
        const auto path=dir.path/std::to_string(kind);
        wire::PublishedFileIntegrity result, read_result;
        REQUIRE(publish(kind,path,data,error,&result));
        std::vector<std::uint8_t> actual;
        REQUIRE(wire::read_file(path,actual,error,"test",&read_result));
        CHECK(actual == expected);
        CHECK(result.bytes == actual.size());
        CHECK(result.fnv1a64 == read_result.fnv1a64);
        CHECK(result.sha256 == read_result.sha256);
        CHECK_FALSE(std::filesystem::exists(path.string()+".tmp"));
    }
}

TEST_CASE("streaming checkpoint hashes preserve section and sink boundaries",
          "[checkpoint][wire][integrity][streaming]")
{
    Directory dir;
    for (std::size_t size : {0u, 1u, 63u, 64u, 65u, 65535u, 65536u, 65537u, 1048579u})
    {
        const std::vector<wire::Section> sections = [&]() {
            std::vector<wire::Section> result;
            for (unsigned n = 0; n < 3; ++n)
            {
                std::vector<std::uint8_t> bytes(n == 1 ? 0 : size + n);
                for (std::size_t i = 0; i < bytes.size(); ++i)
                    bytes[i] = static_cast<std::uint8_t>((i * 37u + n * 113u) ^ (i >> 8));
                result.push_back({3u + n * 4u, 2u + n, n % 2, std::move(bytes)});
            }
            return result;
        }();
        std::string error;
        std::vector<std::uint8_t> expected;
        REQUIRE(wire::wrap_sections(magic, 7, sections, expected, error));
        std::uint64_t expected_fnv = 14695981039346656037ULL;
        for (auto byte : expected)
            expected_fnv = (expected_fnv ^ byte) * 1099511628211ULL;
        nimbus::crypto::Sha256 digest;
        digest.update(expected.data(), expected.size());
        for (std::size_t chunk : {1u, 63u, 65536u, 1048583u})
        {
            std::array<unsigned, 3> calls{};
            std::vector<wire::StreamingSection> streamed;
            for (std::size_t n = 0; n < sections.size(); ++n)
            {
                const auto& section = sections[n];
                streamed.push_back({section.id, section.version, section.flags,
                    [&, n](wire::Encoder& out, std::string&) {
                        ++calls[n];
                        out.raw(nullptr, 0);
                        const auto& bytes = sections[n].payload;
                        for (std::size_t offset = 0; offset < bytes.size(); offset += chunk)
                            out.raw(bytes.data() + offset, std::min(chunk, bytes.size() - offset));
                        return out.good();
                    }});
            }
            const auto path = dir.path / "checkpoint";
            { std::ofstream previous(path); previous << "previous checkpoint"; }
            wire::PublishedFileIntegrity integrity{17, 19, "previous"};
            REQUIRE(wire::write_streaming_sections_atomic(path, magic, 7, streamed, error, "test", &integrity));
            CHECK((calls == std::array<unsigned, 3>{{3, 3, 3}}));
            const auto actual = contents(path);
            REQUIRE(actual.size() == expected.size());
            CHECK(std::memcmp(actual.data(), expected.data(), expected.size()) == 0);
            CHECK(integrity.bytes == expected.size());
            CHECK(integrity.fnv1a64 == expected_fnv);
            CHECK(integrity.sha256 == digest.hex_digest());
            CHECK_FALSE(std::filesystem::exists(path.string() + ".tmp"));
        }
    }
    const auto path = dir.path / "empty-sections";
    std::string error;
    std::vector<std::uint8_t> expected;
    REQUIRE(wire::wrap_sections(magic, 7, {}, expected, error));
    REQUIRE(wire::write_streaming_sections_atomic(path, magic, 7, {}, error, "test"));
    const auto actual = contents(path);
    REQUIRE(actual.size() == expected.size());
    CHECK(std::memcmp(actual.data(), expected.data(), expected.size()) == 0);
}

TEST_CASE("streaming checkpoint prepublication failures preserve bytes and integrity",
          "[checkpoint][wire][failure][streaming]")
{
    Directory dir;
    for (unsigned defect = 0; defect < 8; ++defect)
    {
        const auto path = dir.path / std::to_string(defect);
        { std::ofstream previous(path); previous << "previous checkpoint"; }
        wire::PublishedFileIntegrity integrity{17, 19, "previous"};
        unsigned calls = 0;
        std::vector<wire::StreamingSection> sections{{1, 1, wire::section_required,
            [&](wire::Encoder& out, std::string& error) {
                ++calls;
                out.integer(17);
                if (defect == 4 && calls == 3)
                {
                    error = "encoder refused publication";
                    return false;
                }
                if (defect == 5 && calls == 3)
                    out.floating_array(nullptr, 1);
                return out.good();
            }}};
        if (defect == 0) sections[0].id = 0;
        if (defect == 1) sections.push_back(sections.front());
        if (defect == 2) sections[0].flags = 2;
        if (defect == 3) sections[0].encode = {};
        const std::uint64_t reserve = defect == 6 ? wire::max_file_bytes :
            defect == 7 ? std::filesystem::space(dir.path).available + (1ULL << 30) : 0;
        std::string error;
        REQUIRE_FALSE(wire::write_streaming_sections_atomic(path, magic, 1, sections, error, "test", &integrity, reserve));
        CHECK_FALSE(error.empty());
        CHECK(contents(path) == "previous checkpoint");
        CHECK(integrity.bytes == 17);
        CHECK(integrity.fnv1a64 == 19);
        CHECK(integrity.sha256 == "previous");
        CHECK_FALSE(std::filesystem::exists(path.string() + ".tmp"));
    }
}

TEST_CASE("streaming checkpoint sync and rename failures do not publish integrity",
          "[checkpoint][wire][failure][streaming]")
{
    Directory dir;
    for (bool missing_temporary : {false, true})
    {
#if defined(_WIN32)
        if (missing_temporary) continue;
#endif
        const auto path = dir.path / (missing_temporary ? "sync" : "rename");
        if (missing_temporary)
        {
            std::ofstream previous(path);
            previous << "previous checkpoint";
        }
        else
        {
            std::filesystem::create_directory(path);
            std::ofstream previous(path / "sentinel");
            previous << "previous checkpoint";
        }
        unsigned calls = 0;
        const std::vector<wire::StreamingSection> sections{{1, 1, wire::section_required,
            [&](wire::Encoder& out, std::string&) {
                ++calls;
                out.integer(23);
                if (missing_temporary && calls == 3)
                    std::filesystem::remove(path.string() + ".tmp");
                return out.good();
            }}};
        wire::PublishedFileIntegrity integrity{17, 19, "previous"};
        std::string error;
        REQUIRE_FALSE(wire::write_streaming_sections_atomic(path, magic, 1, sections, error, "test", &integrity));
        CHECK(calls == 3);
        CHECK(error.find(missing_temporary ? "fsync" : "atomically replace") != std::string::npos);
        CHECK(contents(missing_temporary ? path : path / "sentinel") == "previous checkpoint");
        CHECK(integrity.bytes == 17);
        CHECK(integrity.fnv1a64 == 19);
        CHECK(integrity.sha256 == "previous");
        CHECK_FALSE(std::filesystem::exists(path.string() + ".tmp"));
    }
}

TEST_CASE("checkpoint open errors identify destination and preserve existing checkpoint", "[checkpoint][wire][failure]")
{
    Directory dir;
    for (int kind : {0,1,2})
    {
        const auto path=dir.path/std::to_string(kind);
        {std::ofstream out(path);out << "previous checkpoint";}
        std::filesystem::create_directory(path.string()+".tmp");
        std::string error;
        CHECK_FALSE(publish(kind,path,{1,2,3},error));
        CHECK(error.find("checkpoint open failed") != std::string::npos);
        CHECK(error.find(path.string()+".tmp") != std::string::npos);
        CHECK(error.find("errno=0 ") == std::string::npos);
        CHECK(error.find("stream_bytes_accepted=0") != std::string::npos);
        CHECK(contents(path) == "previous checkpoint");
    }
}

TEST_CASE("checkpoint encoder mutation cannot replace published data or integrity", "[checkpoint][wire][failure]")
{
    Directory dir;
    for (int failing_pass : {2,3})
    {
        const auto path=dir.path/std::to_string(failing_pass);
        {std::ofstream out(path);out << "previous checkpoint";}
        int pass=0;
        wire::PublishedFileIntegrity integrity{42,43,"old-integrity"};
        const std::vector<wire::StreamingSection> sections{{1,1,wire::section_required,
            [&](wire::Encoder& out,std::string&) {++pass;out.integer(pass==failing_pass?9:8);return true;}}};
        std::string error;
        REQUIRE_FALSE(wire::write_streaming_sections_atomic(path,magic,1,sections,error,"test",&integrity));
        CHECK(error.find(failing_pass==2?"not deterministic":"changed during publication")!=std::string::npos);
        CHECK(contents(path)=="previous checkpoint");
        CHECK(integrity.bytes==42);
        CHECK(integrity.fnv1a64==43);
        CHECK(integrity.sha256=="old-integrity");
        CHECK_FALSE(std::filesystem::exists(path.string()+".tmp"));
    }
}

#if !defined(_WIN32)
TEST_CASE("checkpoint OS write and close errors survive cleanup atomically", "[checkpoint][wire][failure][posix]")
{
    Directory dir;
    for (int kind : {0,1,2}) for (bool large : {false,true})
    {
        const auto path=dir.path/(std::to_string(kind)+(large?"-write":"-close"));
        {std::ofstream out(path);out << "previous checkpoint";}
        const pid_t child=::fork();
        REQUIRE(child>=0);
        if (child==0)
        {
            ::signal(SIGXFSZ,SIG_IGN);
            struct rlimit limit{};
            if (::getrlimit(RLIMIT_FSIZE,&limit)!=0) ::_exit(20);
            limit.rlim_cur=0;
            if (::setrlimit(RLIMIT_FSIZE,&limit)!=0) ::_exit(21);
            wire::PublishedFileIntegrity integrity{42,43,"old-integrity"};
            std::string error;
            const bool accepted=publish(kind,path,std::vector<std::uint8_t>(large?65536:3,42),error,&integrity);
            const bool ok=!accepted && error.find(large?"checkpoint write failed":"checkpoint close failed")!=std::string::npos &&
                error.find("errno="+std::to_string(EFBIG)+" ")!=std::string::npos &&
                error.find(path.string()+".tmp")!=std::string::npos && integrity.bytes==42 &&
                integrity.fnv1a64==43 && integrity.sha256=="old-integrity";
            std::cout << error << std::endl;
            ::_exit(ok?0:22);
        }
        int status=0;
        REQUIRE(::waitpid(child,&status,0)==child);
        REQUIRE(WIFEXITED(status));
        CHECK(WEXITSTATUS(status)==0);
        CHECK(contents(path)=="previous checkpoint");
        CHECK_FALSE(std::filesystem::exists(path.string()+".tmp"));
    }
}
#endif

TEST_CASE("bulk checkpoint decoding preserves bits and rejects truncated arrays atomically", "[checkpoint][wire][bulk]")
{
    std::vector<std::uint8_t> payload{0x42};
    std::vector<float> expected(16387);
    std::uint32_t bits=0x81818283U;
    for (std::size_t i=0;i<expected.size();++i)
    {
        bits=bits*1664525U+1013904223U;
        const auto value=i==0?0x80000000U:i==1?0x7f801234U:bits;
        std::memcpy(&expected[i],&value,sizeof(value));
        for (unsigned byte=0;byte<4;++byte) payload.push_back(static_cast<std::uint8_t>(value>>(8*byte)));
    }
    const auto original=payload;
    wire::Decoder decoder(payload.data()+1,payload.size()-1);
    std::vector<float> restored(expected.size()+2,42.0f);
    REQUIRE(decoder.floating_array(restored.data()+1,3));
    REQUIRE(decoder.floating_array(restored.data()+4,expected.size()-3));
    CHECK(decoder.finished());
    CHECK(std::memcmp(expected.data(),restored.data()+1,expected.size()*sizeof(float))==0);
    CHECK(restored.front()==42.0f);
    CHECK(restored.back()==42.0f);
    CHECK(payload==original);
    for (std::size_t tail=1;tail<=4;++tail)
    {
        wire::Decoder truncated(payload.data()+1,payload.size()-1-tail);
        const auto prior=restored;
        REQUIRE_FALSE(truncated.floating_array(restored.data()+1,expected.size()));
        CHECK(truncated.remaining()==payload.size()-1-tail);
        CHECK(std::memcmp(prior.data(),restored.data(),restored.size()*sizeof(float))==0);
    }
    wire::Decoder invalid(payload.data(),payload.size());
    CHECK_FALSE(invalid.floating_array(nullptr,1));
    CHECK(invalid.remaining()==payload.size());
    wire::Decoder oversized(payload.data(),payload.size());
    CHECK_FALSE(oversized.floating_array(restored.data(),std::numeric_limits<std::size_t>::max()));
    CHECK(oversized.remaining()==payload.size());
    wire::Decoder empty(nullptr,0);
    CHECK(empty.floating_array(nullptr,0));
    CHECK(empty.finished());
}

TEST_CASE("combined checkpoint validation matches established wire and publication hashes", "[checkpoint][wire][validation]")
{
    std::uint32_t random=0x4863bac7U;
    for (unsigned trial=0;trial<48;++trial)
    {
        std::vector<wire::Section> original;
        for (unsigned n=0;n<trial%7;++n)
        {
            std::vector<std::uint8_t> payload((trial*137+n*89)%2049);
            if (trial==47 && n==0) payload.resize(1024*1024+3);
            for (auto& byte:payload) {random=random*1664525U+1013904223U;byte=static_cast<std::uint8_t>(random>>24);}
            original.push_back({n+1,trial,static_cast<std::uint32_t>(n%2),std::move(payload)});
        }
        std::vector<std::uint8_t> bytes;
        std::string error;
        REQUIRE(wire::wrap_sections(magic,1,original,bytes,error));
        wire::ByteView payload;
        std::vector<wire::SectionView> reference;
        REQUIRE(wire::unwrap(bytes,magic,1,payload,error,"test"));
        REQUIRE(wire::decode_sections(payload,reference,error));
        for (bool with_integrity:{false,true})
        {
            std::vector<wire::SectionView> actual;
            wire::PublishedFileIntegrity integrity{17,19,"prior"};
            REQUIRE(wire::unwrap_section_views(bytes,magic,1,actual,error,"test",with_integrity?&integrity:nullptr));
            REQUIRE(error.empty());
            REQUIRE(actual.size()==reference.size());
            for (std::size_t n=0;n<actual.size();++n)
            {
                CHECK(actual[n].id==reference[n].id);
                CHECK(actual[n].version==reference[n].version);
                CHECK(actual[n].flags==reference[n].flags);
                CHECK(actual[n].payload.size()==reference[n].payload.size());
                CHECK(actual[n].payload.data()==reference[n].payload.data());
            }
            if (with_integrity)
            {
                std::uint64_t fnv=14695981039346656037ULL;
                for (const auto byte:bytes) {fnv^=byte;fnv*=1099511628211ULL;}
                nimbus::crypto::Sha256 digest;digest.update(bytes.data(),bytes.size());
                CHECK(integrity.bytes==bytes.size());
                CHECK(integrity.fnv1a64==fnv);
                CHECK(integrity.sha256==digest.hex_digest());
            }
            else
            {
                CHECK(integrity.bytes==17);
                CHECK(integrity.fnv1a64==19);
                CHECK(integrity.sha256=="prior");
            }
        }
    }
}

TEST_CASE("combined checkpoint validation rejects damage without publishing views or integrity", "[checkpoint][wire][validation][failure]")
{
    std::string error;
    std::vector<std::uint8_t> valid;
    REQUIRE(wire::wrap_sections(magic,1,{{1,3,wire::section_required,{1,2,3,4,5}},{7,2,0,{9,8,7}}},valid,error));
    const auto reject=[&](const std::vector<std::uint8_t>& bytes) {
        wire::ByteView payload;
        std::vector<wire::SectionView> reference;
        const bool accepted=wire::unwrap(bytes,magic,1,payload,error,"test") && wire::decode_sections(payload,reference,error);
        REQUIRE_FALSE(accepted);
        for (bool with_integrity:{false,true})
        {
            std::vector<wire::SectionView> output{{91,92,0,{valid.data(),1}}};
            wire::PublishedFileIntegrity integrity{17,19,"prior"};
            REQUIRE_FALSE(wire::unwrap_section_views(bytes,magic,1,output,error,"test",with_integrity?&integrity:nullptr));
            CHECK_FALSE(error.empty());
            REQUIRE(output.size()==1);
            CHECK(output.front().id==91);
            CHECK(output.front().payload.data()==valid.data());
            CHECK(integrity.bytes==17);
            CHECK(integrity.fnv1a64==19);
            CHECK(integrity.sha256=="prior");
        }
    };
    for (std::size_t length=0;length<valid.size();++length)
        reject({valid.begin(),valid.begin()+static_cast<std::ptrdiff_t>(length)});
    for (std::size_t index=0;index<valid.size();++index)
        for (unsigned bit=0;bit<8;++bit)
        {
            auto damaged=valid;damaged[index]^=static_cast<std::uint8_t>(1U<<bit);reject(damaged);
        }
    std::vector<std::uint8_t> payload;
    REQUIRE(wire::unwrap(valid,magic,1,payload,error,"test"));
    for (unsigned defect=0;defect<6;++defect)
    {
        auto damaged=payload;
        switch (defect)
        {
            case 0: damaged[12]=2;break;
            case 1: damaged[4]=0;break;
            case 2: damaged[4+28+5]=1;break;
            case 3: for (std::size_t n=16;n<24;++n) damaged[n]=255;break;
            case 4: damaged.push_back(0);break;
            case 5: damaged[0]=1;damaged[1]=16;break;
        }
        std::vector<std::uint8_t> rebuilt;
        wire::wrap(magic,1,damaged,rebuilt);
        reject(rebuilt);
    }
}
