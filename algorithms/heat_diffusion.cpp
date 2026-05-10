#include "communication.hpp"
#include "utils.hpp"

#include <mpi.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>

namespace pc {

int run_heat_diffusion(const Options& options, int world_rank, int world_size) {
    std::size_t rows = options.rows;
    std::size_t cols = options.cols;
    std::vector<double> global_grid;

    if (world_rank == 0) {
        global_grid = options.input_path.empty() ? make_heat_grid(rows, cols)
                                                 : load_matrix_file(options.input_path, rows, cols);
    }

    const auto rows_as_ull = static_cast<unsigned long long>(rows);
    const auto cols_as_ull = static_cast<unsigned long long>(cols);
    unsigned long long dimensions[2] = {rows_as_ull, cols_as_ull};
    MPI_Bcast(dimensions, 2, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    rows = static_cast<std::size_t>(dimensions[0]);
    cols = static_cast<std::size_t>(dimensions[1]);
    require(cols >= 2 && rows >= 2, "Heat diffusion requires at least a 2x2 grid", world_rank);

    const auto [row_counts, row_displacements] = balanced_counts(rows, world_size);
    const int local_rows = row_counts[world_rank];
    const int local_elements = local_rows * to_int(cols, "Column count");

    std::vector<int> element_counts(world_size, 0);
    std::vector<int> element_displacements(world_size, 0);
    for (int rank = 0; rank < world_size; ++rank) {
        element_counts[rank] = row_counts[rank] * to_int(cols, "Column count");
        element_displacements[rank] = row_displacements[rank] * to_int(cols, "Column count");
    }

    std::vector<double> local_data(static_cast<std::size_t>(std::max(local_elements, 0)), 0.0);
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
    MPI_Comm_split(MPI_COMM_WORLD, local_rows > 0 ? 1 : MPI_UNDEFINED, world_rank, &active_comm);

    int active_rank = -1;
    int active_size = 0;
    if (active_comm != MPI_COMM_NULL) {
        MPI_Comm_rank(active_comm, &active_rank);
        MPI_Comm_size(active_comm, &active_size);
    }

    std::vector<double> current((static_cast<std::size_t>(local_rows) + 2U) * cols, 0.0);
    std::vector<double> next = current;
    for (int row = 0; row < local_rows; ++row) {
        std::copy_n(local_data.data() + static_cast<std::size_t>(row) * cols,
                    cols,
                    current.data() + static_cast<std::size_t>(row + 1) * cols);
    }

    double final_delta = 0.0;
    double local_elapsed = 0.0;

    if (active_comm != MPI_COMM_NULL) {
        MPI_Barrier(active_comm);
        const double start = MPI_Wtime();

        for (std::size_t iteration = 0; iteration < options.iterations; ++iteration) {
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
                for (std::size_t col = 0; col < cols; ++col) {
                    const std::size_t index = static_cast<std::size_t>(local_row) * cols + col;
                    const bool boundary = global_row == 0 || global_row == static_cast<int>(rows) - 1 || col == 0 || col + 1 == cols;
                    if (boundary) {
                        next[index] = current[index];
                        continue;
                    }

                    const double updated = (current[index] +
                                            current[index - 1] +
                                            current[index + 1] +
                                            current[index - cols] +
                                            current[index + cols]) / 5.0;
                    local_delta = std::max(local_delta, std::abs(updated - current[index]));
                    next[index] = updated;
                }
            }

            std::swap(current, next);
            MPI_Allreduce(&local_delta, &final_delta, 1, MPI_DOUBLE, MPI_MAX, active_comm);
        }

        local_elapsed = MPI_Wtime() - start;
    }

    for (int row = 0; row < local_rows; ++row) {
        std::copy_n(current.data() + static_cast<std::size_t>(row + 1) * cols,
                    cols,
                    local_data.data() + static_cast<std::size_t>(row) * cols);
    }

    std::vector<double> result;
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

        const double checksum = std::accumulate(result.begin(), result.end(), 0.0);
        const int active_processes = static_cast<int>(std::count_if(row_counts.begin(), row_counts.end(), [](int count) { return count > 0; }));
        std::cout << std::fixed << std::setprecision(6)
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

}  // namespace pc
