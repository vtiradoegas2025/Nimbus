#include "catch2/catch.hpp"
#include "core/io/npy_reader.hpp"

#include <filesystem>
#include <fstream>
#include <cstdint>
#include <vector>

TEST_CASE("NPY reader validates the complete shape tuple", "[core][npy][input]")
{
    const auto path = std::filesystem::temp_directory_path() / "nimbus-strict-npy-shape.npy";
    const auto write_header = [&](const std::string& header) {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write("\x93NUMPY\x01\x00", 8);
        const auto size = static_cast<std::uint16_t>(header.size());
        const char length[2] = {static_cast<char>(size & 255), static_cast<char>(size >> 8)};
        out.write(length, 2);
        out << header;
        const std::vector<float> payload(32, 1.25f);
        out.write(reinterpret_cast<const char*>(payload.data()), payload.size() * sizeof(float));
    };
    const auto write = [&](const std::string& shape) {
        write_header("{'descr': '<f4', 'fortran_order': False, 'shape': " + shape + ", }\n");
    };
    for (const std::string shape : {"(-4, 8)", "(+4, 8)", "(4 8)", "(4.0, 8)", "(4,,8)",
            "(4x,8)", "(04,8)", "(2147483648,8)", "(0,8)", "(4)", "()"})
    {
        CAPTURE(shape);
        write(shape);
        oglcpp::NpyArray2D array;
        std::string error;
        REQUIRE_FALSE(oglcpp::load_npy_float32_2d(path, array, error));
        REQUIRE_FALSE(error.empty());
        REQUIRE(oglcpp::npy_ndim(path) == 0);
    }
    for (const std::string shape : {"(4,8)", "(4,8,)", "( 4 , 8 )"})
    {
        write(shape);
        oglcpp::NpyArray2D array;
        std::string error;
        REQUIRE(oglcpp::load_npy_float32_2d(path, array, error));
        REQUIRE(array.rows == 4);
        REQUIRE(array.cols == 8);
        REQUIRE(array.data == std::vector<float>(32, 1.25f));
    }
    for (const std::string header : {
            "{'descr':'<f4','fortran_order':Falsejunk,'shape':(4,8)}",
            "{'descr':'<f4','fortran_order':False,'shape':(4,8),'shape':(8,4)}",
            "{'descr':'<f4','fortran_order':False,'shape':(4,8),'extra':0}",
            "{'descr':'<f4','fortran_order':False,'shape':(4,8)} trailing"})
    {
        CAPTURE(header);
        write_header(header);
        REQUIRE(oglcpp::npy_ndim(path) == 0);
    }
    for (const std::string shape : {"(4,9)", "(2147483647,2147483647)"})
    {
        write(shape);
        oglcpp::NpyArray2D array;
        array.rows = 1;
        array.cols = 1;
        array.data = {7.0f};
        std::string error;
        REQUIRE_FALSE(oglcpp::load_npy_float32_2d(path, array, error));
        REQUIRE(array.rows == 1);
        REQUIRE(array.cols == 1);
        REQUIRE(array.data == std::vector<float>{7.0f});
    }
    write("(4,8,2)");
    oglcpp::NpyArray3D volume;
    volume.data = {7.0f};
    std::string error;
    REQUIRE_FALSE(oglcpp::load_npy_float32_3d(path, volume, error));
    REQUIRE(volume.data == std::vector<float>{7.0f});
    std::filesystem::remove(path);
}
