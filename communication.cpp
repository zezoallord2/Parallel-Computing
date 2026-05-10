#include "communication.hpp"

namespace pc {

namespace {

void exchange_with_neighbor_blocking(MPI_Comm comm, int self_rank, int neighbor_rank,
                                     double* send_buffer, double* recv_buffer, int width) {
    if (neighbor_rank == MPI_PROC_NULL) {
        return;
    }

    if (self_rank < neighbor_rank) {
        MPI_Send(send_buffer, width, MPI_DOUBLE, neighbor_rank, 0, comm);
        MPI_Recv(recv_buffer, width, MPI_DOUBLE, neighbor_rank, 1, comm, MPI_STATUS_IGNORE);
    } else {
        MPI_Recv(recv_buffer, width, MPI_DOUBLE, neighbor_rank, 0, comm, MPI_STATUS_IGNORE);
        MPI_Send(send_buffer, width, MPI_DOUBLE, neighbor_rank, 1, comm);
    }
}

}  // namespace

void exchange_halos_blocking(MPI_Comm active_comm, int active_rank, int active_size,
                             std::vector<double>& current, int local_rows, int cols) {
    const int top = active_rank > 0 ? active_rank - 1 : MPI_PROC_NULL;
    const int bottom = active_rank + 1 < active_size ? active_rank + 1 : MPI_PROC_NULL;

    exchange_with_neighbor_blocking(active_comm, active_rank, top,
                                    current.data() + cols,
                                    current.data(),
                                    cols);
    exchange_with_neighbor_blocking(active_comm, active_rank, bottom,
                                    current.data() + static_cast<std::size_t>(local_rows) * cols,
                                    current.data() + static_cast<std::size_t>(local_rows + 1) * cols,
                                    cols);
}

void exchange_halos_nonblocking(MPI_Comm active_comm, int active_rank, int active_size,
                                std::vector<double>& current, int local_rows, int cols) {
    const int top = active_rank > 0 ? active_rank - 1 : MPI_PROC_NULL;
    const int bottom = active_rank + 1 < active_size ? active_rank + 1 : MPI_PROC_NULL;

    std::vector<MPI_Request> requests;
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
        MPI_Irecv(current.data() + static_cast<std::size_t>(local_rows + 1) * cols,
                  cols, MPI_DOUBLE, bottom, 1, active_comm, &recv_request);
        MPI_Isend(current.data() + static_cast<std::size_t>(local_rows) * cols,
                  cols, MPI_DOUBLE, bottom, 0, active_comm, &send_request);
        requests.push_back(recv_request);
        requests.push_back(send_request);
    }

    if (!requests.empty()) {
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
    }
}

long long ring_prefix_offset(MPI_Comm active_comm, int active_rank, int active_size,
                             const std::vector<long long>& local_values) {
    long long offset = 0;
    if (active_rank > 0) {
        MPI_Recv(&offset, 1, MPI_LONG_LONG, active_rank - 1, 77, active_comm, MPI_STATUS_IGNORE);
    }

    const long long outgoing = local_values.empty() ? offset : local_values.back();
    if (active_rank + 1 < active_size) {
        MPI_Send(&outgoing, 1, MPI_LONG_LONG, active_rank + 1, 77, active_comm);
    }
    return offset;
}

}  // namespace pc
