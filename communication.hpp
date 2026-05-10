#pragma once

#include <mpi.h>

#include <vector>

using namespace std;

namespace pc {

void exchange_halos_blocking(MPI_Comm active_comm, int active_rank, int active_size,
                             vector<double>& current, int local_rows, int cols);

void exchange_halos_nonblocking(MPI_Comm active_comm, int active_rank, int active_size,
                                vector<double>& current, int local_rows, int cols);

long long ring_prefix_offset(MPI_Comm active_comm, int active_rank, int active_size,
                             const vector<long long>& local_values);

}  // namespace pc
