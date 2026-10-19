// Strict C-order float32 NPY input shared by tools and presentation clients.

#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace oglcpp
{

struct NpyArray2D
{
    int rows = 0;
    int cols = 0;
    std::vector<float> data;
};

struct NpyArray2DShape
{
    int rows = 0;
    int cols = 0;
};

bool load_npy_float32_2d(const std::filesystem::path& path, NpyArray2D& out, std::string& error);

bool load_npy_float32_2d_shape(const std::filesystem::path& path, NpyArray2DShape& out, std::string& error);

struct NpyArray3D
{
    int dim0 = 0;
    int dim1 = 0;
    int dim2 = 0;
    std::vector<float> data;
};

bool load_npy_float32_3d(const std::filesystem::path& path, NpyArray3D& out, std::string& error);

// Returns 2 or 3 for supported arrays and zero for invalid input.
int npy_ndim(const std::filesystem::path& path);

} // namespace oglcpp
