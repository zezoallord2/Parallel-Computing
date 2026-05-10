#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace {

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

[[noreturn]] void fail(const string& message, int rank = -1) {
    if (rank < 0 || rank == 0) {
        cerr << message << '\n';
    }
    throw runtime_error(message);
}

void require(bool condition, const string& message, int rank = -1) {
    if (!condition) {
        fail(message, rank);
    }
}

string usage() {
    return
        "Usage:\n"
        "  mpirun -np <p> ./parallel_mpi heat [--rows N --cols N --iterations N] [--input file] [--output file] [--comm blocking|nonblocking]\n"
        "  mpirun -np <p> ./parallel_mpi prefix [--size N] [--input file] [--output file] [--comm pipeline|collective]\n\n"
        "Matrix file format:\n"
        "  <rows> <cols> followed by rows*cols floating-point values\n\n"
        "Vector file format:\n"
        "  <count> followed by <count> integer values\n";
}

optional<string> next_value(int& index, int argc, char** argv) {
    if (index + 1 >= argc) {
        return nullopt;
    }
    return argv[++index];
}

size_t parse_size(const string& value, const string& flag) {
    try {
        size_t consumed = 0;
        const auto parsed = stoull(value, &consumed);
        if (consumed != value.size()) {
            fail("Invalid numeric value for " + flag + ": " + value);
        }
        return static_cast<size_t>(parsed);
    } catch (const exception&) {
        fail("Invalid numeric value for " + flag + ": " + value);
    }
}

Options parse_args(int argc, char** argv, int rank) {
    require(argc >= 2, usage(), rank);

    Options options;
    options.algorithm = argv[1];

    if (options.algorithm == "heat") {
        options.comm_mode = "blocking";
    } else if (options.algorithm == "prefix") {
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

    // Distribute the remainder to the first ranks so the program still works
    // when rows/elements are not evenly divisible by the process count.
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

void exchange_with_neighbor_blocking(MPI_Comm comm, int self_rank, int neighbor_rank,
                                     double* send_buffer, double* recv_buffer, int width) {
    if (neighbor_rank == MPI_PROC_NULL) {
        return;
    }

    // Rank ordering prevents the classic deadlock where both neighbors call
    // MPI_Send first and neither side has posted a receive yet.
    if (self_rank < neighbor_rank) {
        MPI_Send(send_buffer, width, MPI_DOUBLE, neighbor_rank, 0, comm);
        MPI_Recv(recv_buffer, width, MPI_DOUBLE, neighbor_rank, 1, comm, MPI_STATUS_IGNORE);
    } else {
        MPI_Recv(recv_buffer, width, MPI_DOUBLE, neighbor_rank, 0, comm, MPI_STATUS_IGNORE);
        MPI_Send(send_buffer, width, MPI_DOUBLE, neighbor_rank, 1, comm);
    }
}

void exchange_halos_blocking(MPI_Comm active_comm, int active_rank, int active_size,
                             vector<double>& current, int local_rows, int cols) {
    const int top = active_rank > 0 ? active_rank - 1 : MPI_PROC_NULL;
    const int bottom = active_rank + 1 < active_size ? active_rank + 1 : MPI_PROC_NULL;

    exchange_with_neighbor_blocking(active_comm, active_rank, top,
                                    current.data() + cols,
                                    current.data(),
                                    cols);
    exchange_with_neighbor_blocking(active_comm, active_rank, bottom,
                                    current.data() + static_cast<size_t>(local_rows) * cols,
                                    current.data() + static_cast<size_t>(local_rows + 1) * cols,
                                    cols);
}

void exchange_halos_nonblocking(MPI_Comm active_comm, int active_rank, int active_size,
                                vector<double>& current, int local_rows, int cols) {
    const int top = active_rank > 0 ? active_rank - 1 : MPI_PROC_NULL;
    const int bottom = active_rank + 1 < active_size ? active_rank + 1 : MPI_PROC_NULL;

    // Post all receives/sends first, then wait once, so communication can
    // progress without forcing a strict send-then-receive order.
    vector<MPI_Request> requests;
    requests.reserve(4);

    if (top != MPI_PROC_NULL) {
        MPI_Request recv_request{};
        MPI_Request send_request{};
        MPI_Irecv(current.data(), cols, MPI_DOUBLE, top, 0, active_comm, &recv_request);
        MPI_Isend(current.data() + cols, cols, MPI_DOUBLE, top, 1, active_comm, &send_request);
        requests.push_back(recv_request);
        requests.push_back(send_request);
    }

    if (bottom != MPI_PROC_NULL) {
        MPI_Request recv_request{};
        MPI_Request send_request{};
        MPI_Irecv(current.data() + static_cast<size_t>(local_rows + 1) * cols,
                  cols, MPI_DOUBLE, bottom, 1, active_comm, &recv_request);
        MPI_Isend(current.data() + static_cast<size_t>(local_rows) * cols,
                  cols, MPI_DOUBLE, bottom, 0, active_comm, &send_request);
        requests.push_back(recv_request);
        requests.push_back(send_request);
    }

    if (!requests.empty()) {
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
    }
}

int run_heat(const Options& options, int world_rank, int world_size) {
    size_t rows = options.rows;
    size_t cols = options.cols;
    vector<double> global_grid;

    if (world_rank == 0) {
        global_grid = options.input_path.empty() ? make_heat_grid(rows, cols)
                                                 : load_matrix_file(options.input_path, rows, cols);
    }

    const auto rows_as_ull = static_cast<unsigned long long>(rows);
    const auto cols_as_ull = static_cast<unsigned long long>(cols);
    unsigned long long dimensions[2] = {rows_as_ull, cols_as_ull};
    MPI_Bcast(dimensions, 2, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    rows = static_cast<size_t>(dimensions[0]);
    cols = static_cast<size_t>(dimensions[1]);
    require(cols >= 2 && rows >= 2, "Heat diffusion requires at least a 2x2 grid", world_rank);

    const auto [row_counts, row_displacements] = balanced_counts(rows, world_size);
    const int local_rows = row_counts[world_rank];
    const int local_elements = local_rows * to_int(cols, "Column count");

    vector<int> element_counts(world_size, 0);
    vector<int> element_displacements(world_size, 0);
    for (int rank = 0; rank < world_size; ++rank) {
        element_counts[rank] = row_counts[rank] * to_int(cols, "Column count");
        element_displacements[rank] = row_displacements[rank] * to_int(cols, "Column count");
    }

    vector<double> local_data(static_cast<size_t>(max(local_elements, 0)), 0.0);
    MPI_Scatterv(world_rank == 0 ? global_grid.data() : nullptr,
                 element_counts.data(),
                 element_displacements.data(),
                 MPI_DOUBLE,
                 local_data.data(),
                 local_elements,
                 MPI_DOUBLE,
                 0,
                 MPI_COMM_WORLD);

    MPI_Comm active_comm = MPI_COMM_NULL;
    // Ranks with zero rows are removed from stencil communication, which keeps
    // neighbor exchange logic simple even when processes outnumber rows.
    MPI_Comm_split(MPI_COMM_WORLD, local_rows > 0 ? 1 : MPI_UNDEFINED, world_rank, &active_comm);

    int active_rank = -1;
    int active_size = 0;
    if (active_comm != MPI_COMM_NULL) {
        MPI_Comm_rank(active_comm, &active_rank);
        MPI_Comm_size(active_comm, &active_size);
    }

    vector<double> current((static_cast<size_t>(local_rows) + 2U) * cols, 0.0);
    vector<double> next = current;
    for (int row = 0; row < local_rows; ++row) {
        copy_n(local_data.data() + static_cast<size_t>(row) * cols,
                    cols,
                    current.data() + static_cast<size_t>(row + 1) * cols);
    }

    double final_delta = 0.0;
    double local_elapsed = 0.0;

    if (active_comm != MPI_COMM_NULL) {
        MPI_Barrier(active_comm);
        const double start = MPI_Wtime();

        for (size_t iteration = 0; iteration < options.iterations; ++iteration) {
            if (active_size > 1) {
                if (options.comm_mode == "blocking") {
                    exchange_halos_blocking(active_comm, active_rank, active_size, current, local_rows, to_int(cols, "Column count"));
                } else if (options.comm_mode == "nonblocking") {
                    exchange_halos_nonblocking(active_comm, active_rank, active_size, current, local_rows, to_int(cols, "Column count"));
                } else {
                    fail("Heat diffusion supports only --comm blocking|nonblocking", world_rank);
                }
            }

            const int global_start_row = row_displacements[world_rank];
            double local_delta = 0.0;
            for (int local_row = 1; local_row <= local_rows; ++local_row) {
                const int global_row = global_start_row + local_row - 1;
                for (size_t col = 0; col < cols; ++col) {
                    const size_t index = static_cast<size_t>(local_row) * cols + col;
                    const bool boundary = global_row == 0 || global_row == static_cast<int>(rows) - 1 || col == 0 || col + 1 == cols;
                    if (boundary) {
                        next[index] = current[index];
                        continue;
                    }

                    // Five-point stencil: center + left + right + top + bottom.
                    const double updated = (current[index] +
                                            current[index - 1] +
                                            current[index + 1] +
                                            current[index - cols] +
                                            current[index + cols]) / 5.0;
                    local_delta = max(local_delta, abs(updated - current[index]));
                    next[index] = updated;
                }
            }

            swap(current, next);
            MPI_Allreduce(&local_delta, &final_delta, 1, MPI_DOUBLE, MPI_MAX, active_comm);
        }

        local_elapsed = MPI_Wtime() - start;
    }

    for (int row = 0; row < local_rows; ++row) {
        copy_n(current.data() + static_cast<size_t>(row + 1) * cols,
                    cols,
                    local_data.data() + static_cast<size_t>(row) * cols);
    }

    vector<double> result;
    if (world_rank == 0) {
        result.resize(rows * cols, 0.0);
    }

    MPI_Gatherv(local_data.data(),
                local_elements,
                MPI_DOUBLE,
                world_rank == 0 ? result.data() : nullptr,
                element_counts.data(),
                element_displacements.data(),
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);

    double max_elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        if (!options.output_path.empty()) {
            write_matrix_file(options.output_path, rows, cols, result);
        }

        const double checksum = accumulate(result.begin(), result.end(), 0.0);
        const int active_processes = static_cast<int>(count_if(row_counts.begin(), row_counts.end(), [](int count) { return count > 0; }));
        cout << fixed << setprecision(6)
                  << "algorithm=heat comm=" << options.comm_mode
                  << " rows=" << rows
                  << " cols=" << cols
                  << " iterations=" << options.iterations
                  << " active_processes=" << active_processes
                  << " checksum=" << checksum
                  << " final_max_delta=" << final_delta
                  << " elapsed_seconds=" << max_elapsed << '\n';
    }

    if (active_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&active_comm);
    }
    return 0;
}

int run_prefix(const Options& options, int world_rank, int world_size) {
    size_t count = options.vector_size;
    vector<long long> global_values;

    if (world_rank == 0) {
        global_values = options.input_path.empty() ? make_vector(count)
                                                   : load_vector_file(options.input_path, count);
    }

    unsigned long long count_as_ull = static_cast<unsigned long long>(count);
    MPI_Bcast(&count_as_ull, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    count = static_cast<size_t>(count_as_ull);

    const auto [counts, displacements] = balanced_counts(count, world_size);
    const int local_count = counts[world_rank];
    vector<long long> local_values(static_cast<size_t>(max(local_count, 0)), 0);

    MPI_Scatterv(world_rank == 0 ? global_values.data() : nullptr,
                 counts.data(),
                 displacements.data(),
                 MPI_LONG_LONG,
                 local_values.data(),
                 local_count,
                 MPI_LONG_LONG,
                 0,
                 MPI_COMM_WORLD);

    MPI_Comm active_comm = MPI_COMM_NULL;
    // Prefix computation only involves ranks that received elements.
    MPI_Comm_split(MPI_COMM_WORLD, local_count > 0 ? 1 : MPI_UNDEFINED, world_rank, &active_comm);

    int active_rank = -1;
    int active_size = 0;
    if (active_comm != MPI_COMM_NULL) {
        MPI_Comm_rank(active_comm, &active_rank);
        MPI_Comm_size(active_comm, &active_size);
    }

    double local_elapsed = 0.0;
    if (active_comm != MPI_COMM_NULL) {
        MPI_Barrier(active_comm);
        const double start = MPI_Wtime();

        for (int index = 1; index < local_count; ++index) {
            local_values[index] += local_values[index - 1];
        }

        const long long local_total = local_count > 0 ? local_values.back() : 0LL;
        long long offset = 0;

        if (options.comm_mode == "pipeline") {
            // Each rank receives the cumulative total of all previous ranks,
            // shifts its local prefix values, then forwards the new total.
            if (active_rank > 0) {
                MPI_Recv(&offset, 1, MPI_LONG_LONG, active_rank - 1, 77, active_comm, MPI_STATUS_IGNORE);
            }
            for (long long& value : local_values) {
                value += offset;
            }
            const long long outgoing = local_count > 0 ? local_values.back() : offset;
            if (active_rank + 1 < active_size) {
                MPI_Send(&outgoing, 1, MPI_LONG_LONG, active_rank + 1, 77, active_comm);
            }
        } else if (options.comm_mode == "collective") {
            // MPI_Exscan gives each rank the sum of all earlier ranks without
            // explicitly stepping through the communicator one process at a time.
            MPI_Exscan(&local_total, &offset, 1, MPI_LONG_LONG, MPI_SUM, active_comm);
            if (active_rank == 0) {
                offset = 0;
            }
            for (long long& value : local_values) {
                value += offset;
            }
        } else {
            fail("Prefix sum supports only --comm pipeline|collective", world_rank);
        }

        local_elapsed = MPI_Wtime() - start;
    }

    vector<long long> result;
    if (world_rank == 0) {
        result.resize(count, 0);
    }

    MPI_Gatherv(local_values.data(),
                local_count,
                MPI_LONG_LONG,
                world_rank == 0 ? result.data() : nullptr,
                counts.data(),
                displacements.data(),
                MPI_LONG_LONG,
                0,
                MPI_COMM_WORLD);

    double max_elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        vector<long long> expected = global_values;
        for (size_t index = 1; index < expected.size(); ++index) {
            expected[index] += expected[index - 1];
        }
        const bool valid = result == expected;
        if (!options.output_path.empty()) {
            write_vector_file(options.output_path, result);
        }
        const auto checksum = accumulate(result.begin(), result.end(), 0LL);
        const int active_processes = static_cast<int>(count_if(counts.begin(), counts.end(), [](int chunk) { return chunk > 0; }));
        cout << "algorithm=prefix comm=" << options.comm_mode
                  << " size=" << count
                  << " active_processes=" << active_processes
                  << " checksum=" << checksum
                  << " verification=" << (valid ? "PASSED" : "FAILED")
                  << " elapsed_seconds=" << fixed << setprecision(6) << max_elapsed
                  << '\n';
        if (!valid) {
            return 2;
        }
    }

    if (active_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&active_comm);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int exit_code = 0;
    try {
        require(world_size >= 2,
                "Run this program with at least 2 MPI processes (currently running with " + to_string(world_size) + ")",
                world_rank);
        const Options options = parse_args(argc, argv, world_rank);
        if (options.algorithm == "heat") {
            exit_code = run_heat(options, world_rank, world_size);
        } else if (options.algorithm == "prefix") {
            exit_code = run_prefix(options, world_rank, world_size);
        }
    } catch (const exception&) {
        exit_code = 1;
    }

    int global_exit_code = 0;
    // If any rank fails, return a non-zero code for the whole MPI job.
    MPI_Allreduce(&exit_code, &global_exit_code, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return global_exit_code;
}
