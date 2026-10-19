#include "core/output/zfp_reader.hpp"

#include <iostream>
#include <string>

int main(int argc, char** argv)
{
    if (argc != 2)
        return 2;
    Zfp3dHeader header;
    std::string error;
    if (read_zfp_3d_header(argv[1], header, error) ||
        error.find("unsupported version 4") == std::string::npos)
    {
        std::cerr << "current native reader did not reject draft v4: " << error << '\n';
        return 1;
    }
    std::cout << error << '\n';
    return 0;
}
