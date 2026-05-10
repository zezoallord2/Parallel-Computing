#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace pc {

struct Options {
    string algorithm;
    string comm_mode;
    string input_path;
    string output_path;
    size_t rows = 32;
    size_t cols = 32;
    size_t iterations = 20;
    size_t vector_size = 128;
};

[[noreturn]] void fail(const string& message, int rank = -1);
void require(bool condition, const string& message, int rank = -1);
string usage();
Options parse_args(int argc, char** argv, int rank);
int to_int(size_t value, const string& label);
pair<vector<int>, vector<int>> balanced_counts(size_t total, int world_size);

vector<double> load_matrix_file(const string& path, size_t& rows, size_t& cols);
void write_matrix_file(const string& path, size_t rows, size_t cols, const vector<double>& values);
vector<long long> load_vector_file(const string& path, size_t& count);
void write_vector_file(const string& path, const vector<long long>& values);

vector<double> make_heat_grid(size_t rows, size_t cols);
vector<long long> make_vector(size_t count);

}  // namespace pc
