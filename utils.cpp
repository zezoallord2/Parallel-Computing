#include "utils.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>

using namespace std;

namespace pc {

[[noreturn]] void fail(const string& message, int rank) {
    if (rank < 0 || rank == 0) {
        cerr << message << '\n';
    }
    throw runtime_error(message);
}

void require(bool condition, const string& message, int rank) {
    if (!condition) {
        fail(message, rank);
    }
}

string usage() {
    return
        "Usage:\n"
        "  mpirun -np <p> ./parallel_mpi heat [--rows N --cols N --iterations N] [--input file] [--output file] [--comm blocking|nonblocking]\n"
        "  mpirun -np <p> ./parallel_mpi matrix [--size N] [--input file] [--output file] [--comm pipeline|collective]\n"
        "    (Category B vector accumulation workflow implemented in algorithms/matrix_mult.cpp)\n\n"
        "Matrix file format:\n"
        "  <rows> <cols> followed by rows*cols floating-point values\n\n"
        "Vector file format:\n"
        "  <count> followed by <count> integer values\n";
}

namespace {

optional<string> next_value(int& index, int argc, char** argv) {
    if (index + 1 >= argc) {
        return nullopt;
    }
    ++index;
    return string(argv[index]);
}

size_t parse_size(const string& value, const string& flag) {
    size_t consumed = 0;
    unsigned long long parsed = 0;
    try {
        parsed = stoull(value, &consumed);
    } catch (const exception&) {
        fail("Invalid numeric value for " + flag + ": " + value);
    }

    if (consumed != value.size()) {
        fail("Invalid numeric value for " + flag + ": " + value);
    }

    if (parsed > static_cast<unsigned long long>(numeric_limits<size_t>::max())) {
        fail("Numeric value for " + flag + " is too large: " + value);
    }

    return static_cast<size_t>(parsed);
}

}  // namespace

Options parse_args(int argc, char** argv, int rank) {
    require(argc >= 2, usage(), rank);

    Options options;
    options.algorithm = argv[1];

    if (options.algorithm == "heat") {
        options.comm_mode = "blocking";
    } else if (options.algorithm == "matrix") {
        options.comm_mode = "pipeline";
    } else {
        fail("Unsupported algorithm: " + options.algorithm + "\n\n" + usage(), rank);
    }

    for (int index = 2; index < argc; ++index) {
        const string arg = argv[index];
        if (arg == "--rows") {
            const auto value = next_value(index, argc, argv);
            require(value.has_value(), "Missing value for --rows", rank);
            options.rows = parse_size(*value, arg);
        } else if (arg == "--cols") {
            const auto value = next_value(index, argc, argv);
            require(value.has_value(), "Missing value for --cols", rank);
            options.cols = parse_size(*value, arg);
        } else if (arg == "--iterations") {
            const auto value = next_value(index, argc, argv);
            require(value.has_value(), "Missing value for --iterations", rank);
            options.iterations = parse_size(*value, arg);
        } else if (arg == "--size") {
            const auto value = next_value(index, argc, argv);
            require(value.has_value(), "Missing value for --size", rank);
            options.vector_size = parse_size(*value, arg);
        } else if (arg == "--comm") {
            const auto value = next_value(index, argc, argv);
            require(value.has_value(), "Missing value for --comm", rank);
            options.comm_mode = *value;
        } else if (arg == "--input") {
            const auto value = next_value(index, argc, argv);
            require(value.has_value(), "Missing value for --input", rank);
            options.input_path = *value;
        } else if (arg == "--output") {
            const auto value = next_value(index, argc, argv);
            require(value.has_value(), "Missing value for --output", rank);
            options.output_path = *value;
        } else if (arg == "--help" || arg == "-h") {
            fail(usage(), rank);
        } else {
            fail("Unknown argument: " + arg + "\n\n" + usage(), rank);
        }
    }

    require(options.rows > 0, "--rows must be greater than zero", rank);
    require(options.cols > 0, "--cols must be greater than zero", rank);
    require(options.vector_size > 0, "--size must be greater than zero", rank);
    require(options.iterations > 0, "--iterations must be greater than zero", rank);
    return options;
}

int to_int(size_t value, const string& label) {
    if (value > static_cast<size_t>(numeric_limits<int>::max())) {
        fail(label + " is too large for MPI integer counts");
    }
    return static_cast<int>(value);
}

pair<vector<int>, vector<int>> balanced_counts(size_t total, int world_size) {
    vector<int> counts(world_size, 0);
    vector<int> displacements(world_size, 0);
    const size_t base = total / static_cast<size_t>(world_size);
    const size_t remainder = total % static_cast<size_t>(world_size);
    int offset = 0;

    for (int rank = 0; rank < world_size; ++rank) {
        const size_t count = base + (static_cast<size_t>(rank) < remainder ? 1U : 0U);
        counts[rank] = to_int(count, "Chunk size");
        displacements[rank] = offset;
        offset += counts[rank];
    }
    return {counts, displacements};
}

vector<double> load_matrix_file(const string& path, size_t& rows, size_t& cols) {
    ifstream input(path);
    require(input.good(), "Failed to open matrix input file: " + path);

    input >> rows >> cols;
    require(rows > 0 && cols > 0, "Matrix input must declare positive rows and columns");

    vector<double> values(rows * cols, 0.0);
    for (double& value : values) {
        require(static_cast<bool>(input >> value), "Matrix input file ended before all values were read");
    }
    return values;
}

void write_matrix_file(const string& path, size_t rows, size_t cols, const vector<double>& values) {
    ofstream output(path);
    require(output.good(), "Failed to open matrix output file: " + path);
    output << rows << ' ' << cols << '\n';
    output << fixed << setprecision(6);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            if (col > 0) {
                output << ' ';
            }
            output << values[row * cols + col];
        }
        output << '\n';
    }
}

vector<long long> load_vector_file(const string& path, size_t& count) {
    ifstream input(path);
    require(input.good(), "Failed to open vector input file: " + path);

    input >> count;
    require(count > 0, "Vector input must declare a positive element count");

    vector<long long> values(count, 0);
    for (long long& value : values) {
        require(static_cast<bool>(input >> value), "Vector input file ended before all values were read");
    }
    return values;
}

void write_vector_file(const string& path, const vector<long long>& values) {
    ofstream output(path);
    require(output.good(), "Failed to open vector output file: " + path);
    output << values.size() << '\n';
    for (size_t index = 0; index < values.size(); ++index) {
        if (index > 0) {
            output << ' ';
        }
        output << values[index];
    }
    output << '\n';
}

vector<double> make_heat_grid(size_t rows, size_t cols) {
    vector<double> grid(rows * cols, 0.0);
    for (size_t row = 1; row + 1 < rows; ++row) {
        for (size_t col = 1; col + 1 < cols; ++col) {
            const bool hot_spot = abs(static_cast<long long>(row) - static_cast<long long>(rows / 2)) <= 1 &&
                                  abs(static_cast<long long>(col) - static_cast<long long>(cols / 2)) <= 1;
            grid[row * cols + col] = hot_spot ? 100.0 : static_cast<double>((row + col) % 9);
        }
    }
    return grid;
}

vector<long long> make_vector(size_t count) {
    vector<long long> values(count, 0);
    for (size_t index = 0; index < count; ++index) {
        values[index] = static_cast<long long>((index % 17U) + 1U);
    }
    return values;
}

}  // namespace pc
