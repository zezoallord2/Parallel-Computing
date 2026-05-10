#include "communication.hpp"
#include "utils.hpp"

#include <mpi.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

namespace pc {

int run_category_b_data_compute(const Options& options, int world_rank, int world_size) {
    std::size_t count = options.vector_size;
    std::vector<long long> global_values;

    if (world_rank == 0) {
        global_values = options.input_path.empty() ? make_vector(count)
                                                   : load_vector_file(options.input_path, count);
    }

    unsigned long long count_as_ull = static_cast<unsigned long long>(count);
    MPI_Bcast(&count_as_ull, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    count = static_cast<std::size_t>(count_as_ull);

    const auto [counts, displacements] = balanced_counts(count, world_size);
    const int local_count = counts[world_rank];
    std::vector<long long> local_values(static_cast<std::size_t>(std::max(local_count, 0)), 0);

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

        long long offset = 0;
        if (options.comm_mode == "pipeline") {
            offset = ring_prefix_offset(active_comm, active_rank, active_size, local_values);
        } else if (options.comm_mode == "collective") {
            const long long local_total = local_count > 0 ? local_values.back() : 0LL;
            MPI_Exscan(&local_total, &offset, 1, MPI_LONG_LONG, MPI_SUM, active_comm);
            if (active_rank == 0) {
                offset = 0;
            }
        } else {
            fail("Category B data computation supports only --comm pipeline|collective", world_rank);
        }

        for (long long& value : local_values) {
            value += offset;
        }

        local_elapsed = MPI_Wtime() - start;
    }

    std::vector<long long> result;
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
        std::vector<long long> expected = global_values;
        for (std::size_t index = 1; index < expected.size(); ++index) {
            expected[index] += expected[index - 1];
        }
        const bool valid = result == expected;
        if (!options.output_path.empty()) {
            write_vector_file(options.output_path, result);
        }
        const auto checksum = std::accumulate(result.begin(), result.end(), 0LL);
        const int active_processes = static_cast<int>(std::count_if(counts.begin(), counts.end(), [](int chunk) { return chunk > 0; }));
        std::cout << "algorithm=matrix category=data_split_compute comm=" << options.comm_mode
                  << " size=" << count
                  << " active_processes=" << active_processes
                  << " checksum=" << checksum
                  << " verification=" << (valid ? "PASSED" : "FAILED")
                  << " elapsed_seconds=" << std::fixed << std::setprecision(6) << max_elapsed
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

}  // namespace pc
