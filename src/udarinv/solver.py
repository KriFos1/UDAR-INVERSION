"""Launching the Fortran solver (fwd2D or fwd3D) for one prepared work directory.

A build is a directory with the binary, ``adjoint-spec.json`` and
``libHDFql.so`` (see ``solver/README.md``). The spec tells whether it is the 2D
or the 3D solver.
"""

from dataclasses import dataclass, field
import functools
import json
import os
from pathlib import Path
import re
import subprocess


class SolverError(RuntimeError):
    """The solver ran but did not produce a usable solution (convergence, orientation)."""


@functools.lru_cache(maxsize=None)
def _script_environment(script):
    """Environment after sourcing a shell script (e.g. the oneAPI variables), read once."""
    output = subprocess.run(['bash', '-c', f'source "{script}" >/dev/null 2>&1 && env -0'], check=True,
                            capture_output=True).stdout.decode(errors='replace')
    return dict(item.split('=', 1) for item in output.split('\0') if '=' in item)


@dataclass
class Solver:
    """One solver build and how to run it.

    ``launcher`` 'local' runs the binary as a single process with ``threads``
    OpenMP/MKL threads. 'srun' runs it as a Slurm step inside the current
    allocation with ``ranks`` MPI ranks of ``cpus`` CPUs; ``step_memory_gb``
    gives the step its own memory request, without which concurrent steps of
    one job run one after another.
    """
    binary: Path
    launcher: str = 'local'
    threads: int = 8
    mkl_threads: int | None = None
    ranks: int = 2
    cpus: int = 16
    step_memory_gb: float | None = None
    srun_options: tuple = ('--mpi=pmi2', '--exact', '--nodes=1')
    timeout: float = 600.
    env_script: str | None = None
    spec: dict = field(init=False, repr=False)
    dimension: int = field(init=False)

    def __post_init__(self):
        self.binary = Path(self.binary).expanduser().resolve()
        if not self.binary.is_file():
            raise FileNotFoundError(f'Solver binary not found: {self.binary}')
        spec_path = self.binary.parent / 'adjoint-spec.json'
        if not spec_path.is_file():
            raise FileNotFoundError(f'No adjoint-spec.json next to {self.binary}')
        self.spec = json.loads(spec_path.read_text())
        if self.spec.get('fidelity') not in ('2D', '3D'):
            raise ValueError(f'{spec_path} does not say whether the build is 2D or 3D')
        if not self.spec.get('orientation_input'):
            raise ValueError(f'{self.binary} was built without the bedding-orientation input')
        self.dimension = int(self.spec['fidelity'][0])
        if self.launcher not in ('local', 'srun') or self.threads < 1 or self.timeout <= 0:
            raise ValueError('Launcher is local or srun; threads and timeout are positive')

    def _environment(self):
        base = dict(_script_environment(self.env_script)) if self.env_script else dict(os.environ)
        if self.launcher == 'local':
            # Variables of an enclosing Slurm/MPI job break the solver's single-process MPI start.
            base = {key: value for key, value in base.items()
                    if not key.startswith(('SLURM_', 'PMIX_', 'PMI_', 'I_MPI_HYDRA'))}
            threads = self.threads
        else:
            threads = max(1, self.cpus // 2)
        base.update(OMP_NUM_THREADS=str(threads),
                    MKL_NUM_THREADS=str(self.mkl_threads or (self.threads if self.launcher == 'local' else 1)),
                    MKL_CBWR='AVX2,STRICT', OMP_STACKSIZE='64M')
        base['LD_LIBRARY_PATH'] = f"{self.binary.parent}:{base.get('LD_LIBRARY_PATH', '')}".rstrip(':')
        return base

    def command(self):
        if self.launcher == 'local':
            # The solver keeps large arrays on the stack.
            return ['bash', '-c', 'ulimit -s unlimited 2>/dev/null || ulimit -s "$(ulimit -H -s)"; exec "$0"',
                    str(self.binary)]
        memory = [f'--mem={self.step_memory_gb:g}G'] if self.step_memory_gb else []
        return ['srun', *self.srun_options, f'--ntasks={self.ranks}', f'--cpus-per-task={self.cpus}', *memory,
                str(self.binary)]

    def run(self, directory):
        """Run in ``directory``; raises on a crash, a timeout, a convergence error or a wrong orientation."""
        directory = Path(directory)
        with (directory / 'solver.log').open('wb') as log:
            subprocess.run(self.command(), cwd=directory, env=self._environment(), stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=self.timeout)
        report = directory / 'convergence_error.txt'
        if report.exists() and re.search(r'\berror\b', report.read_text(errors='replace'), re.IGNORECASE):
            raise SolverError('convergence error')
        used = re.findall(r'material orientation ifdip\s*=\s*(-?\d+)',
                          (directory / 'solver.log').read_text(errors='replace'))
        if not used or any(int(value) != 0 for value in used):
            raise SolverError(f'the bedding orientation was not applied: {used}')


def log_tail(directory, lines=120):
    """The last lines of a solver log (it carries a few non-text bytes); '' without a log."""
    path = Path(directory) / 'solver.log'
    if not path.is_file():
        return ''
    return '\n'.join(path.read_text(errors='replace').splitlines()[-lines:]) + '\n'
