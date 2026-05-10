#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace pc {

struct Options {
    std::string algorithm;
    std::string comm_mode;
    std::string input_path;
    std::string output_path;
    std::size_t rows = 32;
    std::size_t cols = 32;
    std::size_t iterations = 20;
    std::size_t vector_size = 128;
};

[[noreturn]] void fail(const std::string& message, int rank = -1);
void require(bool condition, const std::string& message, int rank = -1);
std::string usage();
Options parse_args(int argc, char** argv, int rank);
int to_int(std::size_t value, const std::string& label);
std::pair<std::vector<int>, std::vector<int>> balanced_counts(std::size_t total, int world_size);

std::vector<double> load_matrix_file(const std::string& path, std::size_t& rows, std::size_t& cols);
void write_matrix_file(const std::string& path, std::size_t rows, std::size_t cols, const std::vector<double>& values);
std::vector<long long> load_vector_file(const std::string& path, std::size_t& count);
void write_vector_file(const std::string& path, const std::vector<long long>& values);

std::vector<double> make_heat_grid(std::size_t rows, std::size_t cols);
std::vector<long long> make_vector(std::size_t count);

}  // namespace pc
