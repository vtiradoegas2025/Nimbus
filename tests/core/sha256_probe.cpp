#include "core/infra/sha256.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main()
{
    try
    {
        std::size_t chunk=0, offset=0;
        std::string hex;
        while (std::cin>>chunk>>offset>>hex)
        {
            if (hex=="-") hex.clear();
            if (chunk==0 || offset>63 || hex.size()>2*1048576 || hex.size()%2)
                throw std::runtime_error("invalid bounded hash case");
            std::vector<unsigned char> storage(offset+hex.size()/2);
            for (std::size_t i=0;i<hex.size();i+=2)
                storage[offset+i/2]=static_cast<unsigned char>(std::stoul(hex.substr(i,2),nullptr,16));
            nimbus::crypto::Sha256 hash;
            hash.update(nullptr,0);
            for (std::size_t i=0;i<hex.size()/2;i+=chunk)
                hash.update(storage.data()+offset+i,std::min(chunk,hex.size()/2-i));
            const auto original=hash.hex_digest();
            auto branch=hash;
            hash.update("x",1);
            branch.update("y",1);
            std::cout << original << ' ' << hash.hex_digest() << ' ' << branch.hex_digest() << ' '
#ifdef NIMBUS_SHA256_BASELINE
                      << "baseline"
#else
                      << nimbus::crypto::sha256_implementation()
#endif
                      << '\n';
        }
        return 0;
    }
    catch (const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
