# Solver builds

`udarinv` calls two Fortran programs, `fwd2D` and `fwd3D`, built from the
LogSimulator3D source with an added discrete adjoint. The source is not part of
this repository. This folder holds what turns it into the builds `udarinv` needs.

## What a build is

A folder with three files:

| File | Purpose |
|---|---|
| `fwd2D` or `fwd3D` | the solver |
| `adjoint-spec.json` | written by the build; tells `udarinv` the dimension and how to pull the adjoint back onto the model cells |
| `libHDFql.so` | HDF5 reader the solver links against |

Point `[solver] binary` in a config at the program. `udarinv` reads the spec next
to it and puts the folder on `LD_LIBRARY_PATH`.

## Requirements

- The source tree `LogSimulator3D_v4.2fullfortran_jacobi_nrx_2D_3D` (it must
  contain `src/*.f90` and `src/hdfql_linux/{hdfql.mod, libHDFql.so}`).
- Intel oneAPI: `mpiifx` (ifx and Intel MPI) and MKL. The builds used so far
  were made with ifx 2026.1 and Intel MPI 2021.18.
- Python 3 (standard library only) for the build script.

## Building

```bash
source /path/to/oneapi/oneapi-vars.sh
python build_discrete_adjoint.py /path/to/LogSimulator3D_v4.2fullfortran_jacobi_nrx_2D_3D /path/to/build-2d --fidelity 2D
python build_discrete_adjoint.py /path/to/LogSimulator3D_v4.2fullfortran_jacobi_nrx_2D_3D /path/to/build-3d --fidelity 3D
```

The destination folder must not exist. The script copies the sources there,
patches the copies and compiles them; the source tree is not changed.

## What the build changes in the solver

| Change | Files | Why |
|---|---|---|
| Discrete adjoint: the material assembly is recorded and the adjoint tapes are written instead of the solver's own Jacobian | `bai_em_matrix_mod*.f90`, `bai_em_main_mod.f90`, `adjoint_capture.f90` | exact derivatives; the in-solver Jacobian is not usable. 8 or 16 components, kernel loop in OpenMP |
| Phase derivatives | `jacobianprocessing.f90` | the original formulas did not match the forward definition of the phases |
| `orientation.inp` | `buildgrid.f90` | lets `udarinv` keep the anisotropy axis vertical (bedding) instead of following resistivity gradients, which flips axes between cells on smooth models |
| Block_ROM breakdown guard | `bai_em_atuzi_mod.f90` | a solve that stalls just above its tolerance can reach a NaN error bound; the progress print then crashed the program (exit 174). It now stops with a convergence error |
| No `MPI_BARRIER` in the convergence-error branch | `fwd2D_parallel.f90`, `fwd3D_parallel.f90` | with two ranks, every convergence error hung the solve until the caller's timeout |

Converged solves are unaffected by the last two: predictions and Jacobians are
identical with and without them (checked at one point in 2D and in 3D).

Not included, on purpose:

- **Apparent dip (`apparent_dip.inp`).** `udarinv` has no dip variable. The build
  script in `Dashboard/scripts` adds it.
- **Refined solver grid.** `udarinv` never writes `customgrids.inp`, so the solver
  uses its built-in grid, the one the benchmark data were generated with.
- **LAPACK status checks in the block solver** (the third part of the earlier
  Block_ROM patch): never part of a build that was used.

## Running

- **Workstation** (`launcher = "local"`): one process with `threads` OpenMP and
  MKL threads. If the oneAPI runtime libraries are not on `LD_LIBRARY_PATH`, set
  `env_script` to the `oneapi-vars.sh` file. Use 8 threads to reproduce the
  cluster results: the solver's convergence depends on the MKL thread count.
- **Slurm** (`launcher = "srun"`): each solve is a step of the current
  allocation with `ranks` MPI ranks of `cpus` CPUs. Set `step_memory_gb` when
  `concurrency` is above 1; a step without a memory request takes all of the
  job's memory and the solves then run one after another.

## Checking a build

```bash
udarinv check-jacobian experiments/benchmark3_2d/config.toml --index 120
udarinv check-jacobian experiments/benchmark2_3d/config.toml --index 421
```

compares the adjoint Jacobian with central differences on the benchmark model.
With the builds of November 2026 every direction agrees to 0.3 % or better.

## Known limits

- A solve can still end in a segmentation fault after `EXIT ROM: MAX Iterations`
  (on extreme models); `udarinv` treats any crash as a failed solve.
- The earlier 3D builds did not have the Block_ROM guard, and the 3D solver
  crashed at tolerance 1e-6 on some models. The guard has not yet been seen to
  act in a 3D solve.
