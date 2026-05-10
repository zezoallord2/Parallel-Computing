#include <mpi.h>

#include <exception>
#include <string>

#include "utils.hpp"

namespace pc {
int run_heat_diffusion(const Options& options, int world_rank, int world_size);
int run_vector_accumulation(const Options& options, int world_rank, int world_size);
}  // namespace pc

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int exit_code = 0;
    try {
        pc::require(world_size >= 2,
                    "Run this program with at least 2 MPI processes (currently running with " + std::to_string(world_size) + ")",
                    world_rank);
        const pc::Options options = pc::parse_args(argc, argv, world_rank);
        if (options.algorithm == "heat") {
            exit_code = pc::run_heat_diffusion(options, world_rank, world_size);
        } else if (options.algorithm == "matrix") {
            exit_code = pc::run_vector_accumulation(options, world_rank, world_size);
        }
    } catch (const std::exception&) {
        exit_code = 1;
    }

    int global_exit_code = 0;
    MPI_Allreduce(&exit_code, &global_exit_code, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return global_exit_code;
}
