"""Settings of a chain, read from a TOML file.

Every table and key is optional except ``[data] path``, ``[data] outputs`` and
``[solver] binary``; the defaults below are the settings of the validated runs
(see the README). Relative paths are taken from the folder of the config file;
``$NAME`` is replaced by the environment variable.
"""

from dataclasses import asdict, dataclass, field, fields
import os
from pathlib import Path
import tomllib

from .grid import SolveGrid
from .lm import validate_tolerance_schedule


@dataclass
class Run:
    dimension: int = 3
    start: int = 0                    # first logging point
    stop: int | None = None           # one past the last logging point (None: the end of the well)
    output: str = 'runs/chain'
    seconds: float | None = None      # stop between points after this long (exit code 75: run again to continue)


@dataclass
class Data:
    path: str = ''
    outputs: str = ''
    noise_fraction: float = .025
    noise_floor: float = 1e-3
    floors: dict = field(default_factory=dict)        # minimum std per component, e.g. {USDP = 0.5, USDA = 0.25}


@dataclass
class Lattice:
    nx: int = 33
    nz: int = 80
    cell: float = 5.


@dataclass
class Prior:
    """Background (climatological) field. Ranges are (x, y, z) in ft; a 2D run uses x and z."""
    rh_mean: float = .855             # ln ohm m
    rh_std: float = 1.094
    latent_mean: float = .5           # ln(Rv / Rh)
    latent_std: float = .5
    rh_ranges: tuple = (100., 100., 50.)
    latent_ranges: tuple = (160., 160., 35.)
    std_floors: tuple = (.23, .1)     # smallest carried std a solve starts from (ln Rh, latent)

    def ranges(self, dimension):
        pick = (0, 2) if dimension == 2 else (0, 1, 2)
        return [[float(self.rh_ranges[i]) for i in pick], [float(self.latent_ranges[i]) for i in pick]]

    def lattice_ranges(self):
        return self.ranges(2)


@dataclass
class Carry:
    memory_ft: float = 50.            # relaxation of the carried moments towards the background along the well
    offplane_relax_ft: float = 100.   # return of the prior std to the background away from the lattice
    anomaly_memory_ft: float = 100.   # decay of the previous MAP's off-plane structure (3D)


@dataclass
class SolverSettings:
    binary: str = ''
    launcher: str = 'local'
    threads: int = 8
    mkl_threads: int | None = None
    ranks: int = 2
    cpus: int = 16
    step_memory_gb: float | None = None
    timeout: float = 600.
    env_script: str | None = None
    keep_scratch: bool = False


@dataclass
class Search:
    candidates: int = 2               # damped steps tried per round
    concurrency: int = 1              # solves run at once
    initial_trust: float = 2.
    max_iterations: int = 30
    max_rounds: int = 4
    max_seconds: float = 2400.
    stagnation_tolerance: float = 1e-3
    stagnation_iterations: int = 2
    target_misfit: float = 5.         # reduced chi-square at which a search stops
    # Solver tolerance by misfit of the current model: [[misfit above, tolerance], ...]; one pair = fixed.
    tolerance_schedule: list = field(default_factory=lambda: [[12., 3e-3], [0., 1e-3]])
    tolerance_retries: int = 3
    failure_limit: int = 12           # a search ends after more failures of one kind than this


@dataclass
class Start:
    warm_streak: int = 5              # points in a row below the target before the previous MAP is the start
    background_reset: bool = True     # restart a point from the background prior when every search fails


@dataclass
class Config:
    run: Run = field(default_factory=Run)
    data: Data = field(default_factory=Data)
    grid: SolveGrid = field(default_factory=SolveGrid)
    lattice: Lattice = field(default_factory=Lattice)
    prior: Prior = field(default_factory=Prior)
    carry: Carry = field(default_factory=Carry)
    solver: SolverSettings = field(default_factory=SolverSettings)
    search: Search = field(default_factory=Search)
    start: Start = field(default_factory=Start)

    def validate(self):
        if self.run.dimension not in (2, 3):
            raise ValueError('[run] dimension is 2 or 3')
        if self.run.start < 0 or (self.run.stop is not None and self.run.stop <= self.run.start):
            raise ValueError('[run] requires 0 <= start < stop')
        if not self.data.path or not self.data.outputs or not self.solver.binary:
            raise ValueError('[data] path, [data] outputs and [solver] binary are required')
        self.search.tolerance_schedule = [list(pair) for pair in
                                          validate_tolerance_schedule(self.search.tolerance_schedule)]
        if (self.search.candidates < 1 or self.search.concurrency < 1 or self.search.failure_limit < 1
                or self.search.target_misfit <= 0 or self.start.warm_streak < 1):
            raise ValueError('[search] and [start] counts and the target misfit must be positive')
        if min(self.carry.memory_ft, self.carry.offplane_relax_ft, self.carry.anomaly_memory_ft) <= 0:
            raise ValueError('[carry] lengths must be positive')
        if (len(self.prior.rh_ranges) != 3 or len(self.prior.latent_ranges) != 3 or len(self.prior.std_floors) != 2
                or min(*self.prior.rh_ranges, *self.prior.latent_ranges, *self.prior.std_floors,
                       self.prior.rh_std, self.prior.latent_std) <= 0):
            raise ValueError('[prior] needs three positive ranges per variable and positive stds')
        return self

    def as_dict(self):
        return asdict(self)


def _build(cls, values, table):
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f'Unknown key(s) in [{table}]: {sorted(unknown)}')
    return cls(**values)


def load(path, **overrides):
    """Read a config file. ``overrides`` replace keys of ``[run]`` (e.g. ``stop=40``) when not None."""
    path = Path(path).resolve()
    raw = tomllib.loads(path.read_text())
    tables = {f.name: f.type for f in fields(Config)}
    unknown = set(raw) - set(tables)
    if unknown:
        raise ValueError(f'Unknown table(s) in {path.name}: {sorted(unknown)}')
    config = Config(**{name: _build(tables[name], raw.get(name, {}), name) for name in tables})
    for key, value in overrides.items():
        if value is not None:
            setattr(config.run, key, value)

    def located(text):
        expanded = Path(os.path.expandvars(os.path.expanduser(str(text))))
        return str(expanded if expanded.is_absolute() else (path.parent / expanded).resolve())
    for section, key in ((config.data, 'path'), (config.solver, 'binary'), (config.run, 'output'),
                         (config.solver, 'env_script')):
        if getattr(section, key):
            setattr(section, key, located(getattr(section, key)))
    return config.validate()
