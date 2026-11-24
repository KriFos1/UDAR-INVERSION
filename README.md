# udar-inversion

Sequential look-around inversion of ultra-deep azimuthal resistivity (UDAR) data
in 2D or 3D.

At every logging point the code finds the most probable resistivity model on a
window around the bit, computes its uncertainty in closed form, and carries the
result to the next point in closed form. There is no ensemble, no sampling and no
lower-fidelity step.

## What it does at each logging point

1. **Prior.** The cellwise mean and standard deviation carried from the previous
   point, relaxed towards a background field with a memory along the well, are
   lifted onto the solve grid and given a Matérn correlation.
2. **MAP search.** A trust-region Levenberg–Marquardt search with an adjoint
   Jacobian minimises data misfit plus prior term. The solver tolerance follows
   the misfit (loose while the fit is poor, tight near the end).
3. **Posterior.** The Gauss–Newton Laplace covariance at the MAP gives the
   standard deviation of every cell and of the carried cells exactly, as a
   low-rank correction to the prior.
4. **Handoff.** The mean and standard deviation in the well plane go to the next
   point; the MAP supplies its starting model and, in 3D, its off-plane structure.

The unknowns are `ln Rh` and the latent `ln(Rv/Rh)` of every cell. Nothing is
bounded or clipped.

| | 2D | 3D |
|---|---|---|
| Solve grid (default) | 65 × 120 cells, x ±320 ft, z ±200 ft | 65 × 89 × 120 cells, y ±300 ft in addition |
| Cell size | 2 ft near the tool, growing outwards; z edges on a fixed 2 ft TVD lattice | the same |
| Unknowns | 15,600 | 1.39 million |
| Data per tool | 8 in-plane geosignals | 16 (in-plane and cross-plane) |
| Carried state | mean and std on 33 × 80 cells of 5 ft in the well plane | the same |

## Install

Python 3.11 or newer.

```bash
pip install .            # numpy, scipy, h5py
pip install ".[plot]"    # adds matplotlib and pillow for udarinv-plot
pip install ".[dev]"     # adds pytest
```

The forward solver is a separate Fortran program; see [`solver/README.md`](solver/README.md).

## Data

The benchmark data are in <https://gitlab.norceresearch.no/krfo/udar-benchmarks>.

```bash
experiments/fetch_data.sh                 # Benchmark-2 and Benchmark-3 into data/udar-benchmarks
experiments/fetch_data.sh Benchmark-3     # only one of them (Benchmark-2 needs about 2.5 GB)
```

## Run

```bash
cd experiments/benchmark3_2d
export UDARINV_FWD2D=/path/to/build-2d/fwd2D
udarinv run config.toml                   # all 281 points
udarinv run config.toml --start 80 --stop 141 --output runs/x400-700
```

```bash
cd experiments/benchmark2_3d
export UDARINV_FWD3D=/path/to/build-3d/fwd3D
udarinv run config.toml                   # points 332-481
```

- Running the same command again continues after the last finished point.
- `touch <output>/PAUSE` stops the chain between points.
- `--seconds N` stops between points after N seconds with exit code 75; the
  `run.sbatch` files use this to resubmit themselves on a cluster.
- Each experiment has two configs. `config.toml` lets the solver tolerance follow
  the misfit (3e-3 above chi2/n 12, 1e-3 below). `config-tight.toml` solves
  everything at 1e-6, the setting of the earlier chains; it is slower and less
  robust (see Validation).

Afterwards:

```bash
udarinv summarize runs/chain              # fit, starts, solver record, error against the true model
udarinv-plot curtain runs/chain           # also: sections, residuals, signals, progress, animate
```

## Output

In the output folder:

| File | Content |
|---|---|
| `config.json` | the settings of the chain |
| `summary.json` | one row per finished point: misfit, start, solver tolerance, failures, time |
| `progress.json` | state of the chain and the points that need attention |
| `point-NNNN/report.json` | every search of the point, with its history |
| `point-NNNN/map.npz` | MAP, prior mean, prior and posterior std and the cellwise correlation of the two variables on the solve grid, with the cell edges |
| `point-NNNN/handoff.npz` | mean and std on the carry lattice (before and after the data), data, prediction and channel mask |

Solver work folders are deleted after each solve; the end of the solver log of a
failed solve is kept under `point-NNNN/failed-logs/`. A 3D point takes about
15 MB, a 2D point well under 1 MB.

## Settings

A config is a TOML file; every key has a default (`src/udarinv/config.py`). The
ones that matter most:

| Key | Default | Meaning |
|---|---|---|
| `[run] dimension` | 3 | 2 or 3 |
| `[data] floors` | none | smallest standard deviation per component, e.g. `{ USDP = 0.5, USDA = 0.25 }` |
| `[data] noise_fraction` | 0.025 | std of a component = this fraction of its mean absolute value over the tools |
| `[grid] *_half`, `*_grade` | Benchmark-2 window | extent and grading of the solve grid |
| `[prior] rh_mean`, `rh_std`, `latent_mean`, `latent_std` | 0.855, 1.094, 0.5, 0.5 | background field in ln units |
| `[prior] rh_ranges`, `latent_ranges` | (100, 100, 50), (160, 160, 35) ft | correlation ranges (x, y, z) |
| `[carry] memory_ft` | 50 | memory of the carried state along the well |
| `[search] target_misfit` | 5 | reduced chi-square at which a search stops |
| `[search] tolerance_schedule` | `[[12, 3e-3], [0, 1e-3]]` | solver tolerance by misfit of the current model; one pair is a fixed tolerance |
| `[search] tolerance_retries` | 3 | times a tolerance that failed is tried again after later steps |
| `[search] candidates`, `concurrency` | 2, 1 | trial steps per round, and how many solves run at once |
| `[start] warm_streak` | 5 | points in a row below the target before a search starts from the previous MAP |

Start rule: a point is searched from the prior mean until the last `warm_streak`
points all ended below the target, and from the previous MAP while that holds. A
warm search that ends above the target, or cannot start, is repeated from the
prior mean. If no search can start, the point is restarted from the background
prior (`[start] background_reset`).

## Code

| Module | Role |
|---|---|
| `grid` | graded solve grid, carry lattice |
| `data` | benchmark loader and error model |
| `prior` | separable Matérn prior of a solve; background field on the lattice |
| `forward`, `solver`, `adjoint` | solver inputs, launch, predictions and adjoint Jacobian |
| `lm` | trust-region Levenberg–Marquardt with the tolerance schedule |
| `laplace` | closed-form posterior moments |
| `carry` | moment carry, lift to the solve grid, average back |
| `point`, `chain` | one logging point; the resumable chain |
| `scoring`, `check` | summary and benchmark scores; Jacobian check |
| `udarinv_plot` | figures |

## Tests

```bash
pytest
```

The tests run without the Fortran solver: a small synthetic earth and a linear
stand-in solver exercise the whole chain in 2D and 3D. With a solver build,
`udarinv check-jacobian` verifies the adjoint against central differences.

## Validation (November 2026)

Checks at single points, all on a workstation:

- **Adjoint Jacobian against central differences on the graded grid.** 0.3 % or
  better in 2D (Benchmark-3, points 120 and 200) and 0.13 % or better in 3D
  (Benchmark-2, points 421 and 500, including the cross-plane components). The
  2D check at point 200 and the 3D check at point 500 used builds made with
  `solver/build_discrete_adjoint.py`.
- **A stored point reproduced.** Benchmark-2 point 332 with the carried prior of
  the earlier exact-carry run: misfit 2.7 (earlier code 3.3); the update of
  `ln Rh` in the well plane correlates 0.994 with the earlier one, and the
  posterior standard deviations agree to a median ratio of 1.00.
- **Solver tolerance.** For five MAP models (2D points 81, 95, 110; 3D points
  333, 340) the response at tolerance 1e-3 differs from the one at 1e-6 by
  0.01 standard deviations rms (0.07 at most); at 1e-2 by 0.1 to 0.2.

Short chains run during development, compared with the earlier runs on the
same points and the same scoring (layered truth, carry lattice):

| Chain | Result | Earlier code |
|---|---|---|
| 3D, Benchmark-2 points 332-341, tolerance 1e-6 | 10 of 10 at the target, no failed solve, Rh error 0.289 | 0.281 |
| 2D, Benchmark-3 points 80-112, tolerance 1e-6 | 33 of 33 at the target, Rh error 0.204 | 0.239 |
| 2D, Benchmark-3 points 113-121 (fault), tolerance 1e-6 | 1 of 9 at the target: the solver stalls just above 1e-6 on trial models | 9 of 9 |
| 2D, same fault, tolerance by misfit ending at 1e-3 | points 112-115 at the target; stopped there | |

Not yet run: a whole section with the default tolerance schedule of
`config.toml`, in 2D or 3D.

Results are not identical to the earlier runs: the variables are unbounded, and
the 2D solve uses the graded grid with the separable prior.
