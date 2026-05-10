# Parallel Computing

Advanced Parallel Grid Processing with MPI in C++.

## Implemented algorithms

The project exposes two runtime-selectable distributed algorithms from different categories:

- **Category A – Grid / Spatial:** `heat`
  - 2D heat diffusion stencil
  - Row-wise grid decomposition with halo exchange
  - Supports uneven row counts and more ranks than rows through active-worker communicators
- **Category B – Data / Computation:** `prefix`
  - Distributed prefix sum on a large vector
  - Supports uneven element counts and more ranks than data chunks

## Communication strategies

The system demonstrates multiple communication patterns:

- **Blocking point-to-point:** `heat --comm blocking`
  - Uses `MPI_Send` / `MPI_Recv` for halo exchange
- **Non-blocking point-to-point:** `heat --comm nonblocking`
  - Uses `MPI_Isend` / `MPI_Irecv` for halo exchange
- **Pipeline pattern:** `prefix --comm pipeline`
  - Passes partial totals rank-by-rank
- **Collective-based communication:**
  - `MPI_Scatterv` / `MPI_Gatherv` for uneven distribution
  - `MPI_Allreduce` for heat-diffusion convergence tracking
  - `MPI_Exscan` via `prefix --comm collective`

## Process organization

The implementation uses `MPI_Comm_split` to create an **active-worker communicator** that excludes ranks with zero assigned rows/elements. This keeps the program correct for any `N >= 2`, even when the data size is smaller than the process count.

## Build

```bash
cmake -S /home/runner/work/Parallel-Computing/Parallel-Computing -B /home/runner/work/Parallel-Computing/Parallel-Computing/build
cmake --build /home/runner/work/Parallel-Computing/Parallel-Computing/build
```

## Run

### Heat diffusion

```bash
mpirun --allow-run-as-root --oversubscribe -np 4 /home/runner/work/Parallel-Computing/Parallel-Computing/build/parallel_mpi heat --rows 200 --cols 200 --iterations 100 --comm blocking
mpirun --allow-run-as-root --oversubscribe -np 4 /home/runner/work/Parallel-Computing/Parallel-Computing/build/parallel_mpi heat --rows 200 --cols 200 --iterations 100 --comm nonblocking
```

### Prefix sum

```bash
mpirun --allow-run-as-root --oversubscribe -np 4 /home/runner/work/Parallel-Computing/Parallel-Computing/build/parallel_mpi prefix --size 1000000 --comm pipeline
mpirun --allow-run-as-root --oversubscribe -np 4 /home/runner/work/Parallel-Computing/Parallel-Computing/build/parallel_mpi prefix --size 1000000 --comm collective
```

## Input file formats

### Matrix file

```text
<rows> <cols>
<r0c0> <r0c1> ...
<r1c0> <r1c1> ...
...
```

### Vector file

```text
<count>
<v0> <v1> <v2> ...
```

If `--input` is omitted, the program generates deterministic data so you can still test scalability quickly. If `--output` is provided, rank 0 writes the final matrix/vector to disk.

## Deadlock scenario and correction

### Real deadlock scenario

A naive blocking halo exchange for heat diffusion can deadlock when every rank calls `MPI_Send` to a neighbor before posting the corresponding `MPI_Recv`.

Example failure pattern:

1. rank 0 sends to rank 1 and waits
2. rank 1 sends to rank 2 and waits
3. rank 2 sends to rank 3 and waits
4. rank 3 sends to rank 2 and waits

All processes can end up waiting on sends that have no posted receive yet.

### Corrected solution

This implementation fixes the problem in two ways:

- **Blocking mode:** neighbor exchanges are ordered by rank so one side receives first and the other sends first
- **Non-blocking mode:** both receives and sends are posted with `MPI_Irecv` / `MPI_Isend`, followed by `MPI_Waitall`

## Performance observations

Representative observations from local runs on this repository setup. These values were captured on an oversubscribed CI-style runner and should be treated as examples, not fixed expectations:

- `heat --rows 32 --cols 32 --iterations 20` on 4 ranks completed in about `0.00043s` with blocking exchange and `0.00040s` with non-blocking exchange.
- `prefix --size 1000` on 4 ranks completed in about `0.000026s` with pipeline mode and `0.000023s` with collective mode on this small test.
- The heat stencil shows visible communication impact because every iteration requires neighbor exchange.
- The prefix pipeline is simple but serialized across ranks, so collective scan is the better scaling direction for larger runs.

## Notes

- The program requires at least 2 MPI processes.
- The active-worker communicator keeps inactive ranks safe when `rows < processes` or `size < processes`.
- The root process validates the final distributed prefix sum against a serial reference result.
