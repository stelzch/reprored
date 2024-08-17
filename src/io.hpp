#pragma once
#include <string>
#include <vector>

namespace IO {
std::vector<double> read_psllh(const std::string path);
std::vector<double> read_binpsllh(const std::string path);
} // namespace IO
