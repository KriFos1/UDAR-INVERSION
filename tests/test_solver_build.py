"""The source patches of solver/build_discrete_adjoint.py, on snippets shaped like the Fortran they patch."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'solver' / 'build_discrete_adjoint.py'
spec = importlib.util.spec_from_file_location('build_discrete_adjoint', SCRIPT)
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)

BLOCK_ROM = """\tsubroutine Block_ROM()
\tinteger :: i,j,l
\t\tdo i=1,M
\t\t\tWrite (*,130) ,i,bnd,' |', repeat('-', INT(2*DMAX1(0d0,-DLOG10(bnd)))),repeat('/', ABS(26-INT(2*DMAX1(0d0,-DLOG10(bnd))))) ,'|'
\t\t\tif (bnd < tol) then
\t\t\t\tWrite (*,'(A,I5)')  "Solver converged at iteration:",i
\t\t\t\tNotConverged=.FALSE.;
\t\t\tend if
\t\tend do
\tend subroutine
"""

DRIVER = """\t\tif (convergenceflag==0) then
\t\t\topen(unit=999, file="convergence_error.txt", status="unknown", position="append", action="write")
\t\t\twrite(999,'(A,I0,A,I0)') 'error at log-pt=',logpt,' rx-pt=',rxpt
\t\t\tclose(999)
\t\t\tcall MPI_BARRIER(MPI_COMM_WORLD,ierror)
\t\tend if
\tend do
\tcall MPI_GATHER(sendbuf, ndat, MPI_COMPLEX, recvbuf, ndat, MPI_COMPLEX, 0, MPI_COMM_WORLD, ierror)
\tcall MPI_BARRIER(MPI_COMM_WORLD,ierror)
"""


def test_block_rom_guard_turns_a_bad_bound_into_a_clean_stop():
    patched = build.block_rom_source(BLOCK_ROM)
    assert 'integer :: i,j,l,bar' in patched
    assert "IF (.NOT.(bnd == bnd) .OR. bnd <= 0d0 .OR. bnd >= 1d30) THEN" in patched
    assert 'convergenceflag=0;' in patched and 'EXIT ROM: numerical breakdown' in patched
    assert "repeat('-', bar),repeat('/', 26-bar)" in patched and 'DLOG10(bnd)))),repeat' not in patched
    # The convergence test now sits in the ELSE branch, which is closed after it.
    lines = [line.strip() for line in patched.splitlines()]
    assert lines.index('ELSE') < lines.index('if (bnd < tol) then') < lines.index('END IF')
    assert lines[lines.index('END IF') - 1] == 'end if'
    for broken in (BLOCK_ROM.replace('integer :: i,j,l', 'integer :: i,j'),
                   BLOCK_ROM.replace('Write (*,130) ,i,bnd,', 'Write (*,131) ,i,bnd,'),
                   BLOCK_ROM.replace('Solver converged at iteration:', 'Done at')):
        with pytest.raises(ValueError):
            build.block_rom_source(broken)


def test_only_the_error_branch_barrier_is_removed():
    patched = build.barrier_source(DRIVER)
    assert patched.count('MPI_BARRIER') == 1 and patched.index('MPI_GATHER') < patched.index('MPI_BARRIER')
    assert "close(999)\n\t\tend if" in patched
    with pytest.raises(ValueError):
        build.barrier_source(patched)                       # already patched or another source: fail loudly


def test_capture_module_supports_both_component_sets_and_has_the_parallel_loop():
    capture = (SCRIPT.parent / 'adjoint_capture.f90').read_text()
    assert 'if(datflag==3) then' in capture and 'else if(datflag==5) then' in capture
    assert capture.count('!$omp parallel do') == 1 and capture.count('!$omp end parallel do') == 1
    assert len(build.PATCHES) == 5
