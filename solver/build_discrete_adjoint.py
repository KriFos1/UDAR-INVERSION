"""Build a candidate discrete adjoint without replacing installed simulators.

The candidate captures the exact material assembly and exports a tensor adjoint.
The NumPy pullback in udarinv.adjoint finishes the model-space chain rule.

    python build_discrete_adjoint.py <LogSimulator3D source> <new build folder> --fidelity 2D|3D

See README.md in this folder for what is needed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


def material_source(text):
    start = text.index("SUBROUTINE SQTENS")
    end = text.index("END SUBROUTINE SQTENS", start)
    original = text[start:end]
    transformed = original.replace("use buildgrid", "use buildgrid\nuse adjoint_capture")
    transformed = transformed.replace("integer :: cc", "integer :: cc\nlogical :: ad_uses_gradient\ninteger :: ad_gi,ad_gj,ad_gk")
    transformed = transformed.replace("PI=3.141592653589793D0",
                                      "PI=3.141592653589793D0\ncall begin_material(LGX1,LGY1,LGZ1)")
    specs = []
    # Each of the four staggered tensor blocks has eight sample groups.
    # Split at the actual SQSIG assignment, independent of upstream comments.
    previous = 0
    rebuilt = []
    for match in re.finditer(r"SQSIG\(I1,J1,K1,:,([1-4])\)=SQS\(:\)\*SQAMU", transformed, re.I):
        block = transformed[previous:match.end()]
        kind = int(match[1])
        samples = {}
        for sample in re.finditer(r"sigH6=sigmaH6\(([^\n]+?)\).*?\bs(\d{3})=s\2\+sigH6", block, re.S):
            samples[sample[2]] = [value.strip().lower() for value in sample[1].split(",")]
        sig = re.search(r"SIG\(1\)=\((.*?)\)/\s*&\s*\((.*?)\)", block, re.S)
        if len(samples) != 8 or not sig:
            raise ValueError(f"Unrecognized material assembly block {kind}")
        # Pull coefficients from the arithmetic average; harmonic uses the same weights.
        weights = {m[1]: m[2].replace("&", "").strip().lower()
                   for m in re.finditer(r"s(\d{3})\*(.*?)(?=\+s\d|$)", re.sub(r"\s+", "", sig[1]))}
        denominator = sig[2].replace("&", "").strip().lower()
        # Regex above stops at an inner parenthesis; the denominator is exactly three interval sums.
        den = re.search(r"\(\s*(\(HLX[^\n]+)\s*\)\s*\n\s*SIG\(2\)", block, re.I)
        if not den:
            raise ValueError("Unrecognized averaging denominator")
        denominator = den[1].strip().lower()
        while denominator.count(")") > denominator.count("("):
            denominator = denominator[:-1].rstrip()
        if len(weights) != 8:
            raise ValueError("Unrecognized averaging weights")
        specs.append({"samples": samples, "weights": weights, "denominator": denominator})
        grad = re.search(r"gradx=gradxI\(([^\n]+)\)", block, re.I)
        if not grad:
            raise ValueError("Missing material orientation indices")
        sub = [value.strip() for value in grad[1].split(",")]
        block = re.sub(r"(if\(ifdip==0)", "ad_uses_gradient=.false.\nad_gi=0; ad_gj=0; ad_gk=0\n    \\1", block, flags=re.I)
        block = re.sub(r"CALL gradan\(([^\n]+)\)",
                       "CALL gradan(\\1)\nad_uses_gradient=.true.\n" +
                       f"ad_gi=2*({sub[3]}-1)+{sub[0]}\n" +
                       f"ad_gj=2*({sub[4]}-1)+{sub[1]}\n" +
                       f"ad_gk=2*({sub[5]}-1)+{sub[2]}", block, flags=re.I)
        block = block.replace(match[0], match[0] + "\ncall record_material(I1,J1,K1," + str(kind) +
                              ",SIG(1),SIG(3),coY,siY,coZ,siZ,ad_uses_gradient,ad_gi,ad_gj,ad_gk)")
        rebuilt.append(block)
        previous = match.end()
    if len(specs) != 4:
        raise ValueError("Expected four staggered material tensors")
    rebuilt.append(transformed[previous:] + "\ncall save_material(f_rxpoint)\n")
    text = text[:start] + "".join(rebuilt) + text[end:]
    first = text.lower().index("subroutine computejacsparse(")
    last = text.lower().index("end subroutine computejacsparse", first)
    kernel = """subroutine computeJacSparse(SQSIG,SQSIGH,SQSIGV,field,ome,LGX1,LGY1,LGZ1,blsz,nfr,Jacc)
    use ATUZI_BAAI_ROM, only: SolMatFm
    implicit none
    integer, intent(in) :: LGX1,LGY1,LGZ1,blsz,nfr
    real(8), intent(in) :: ome(nfr),SQSIG(0:LGX1,0:LGY1,0:LGZ1,6,4)
    real(8), intent(in) :: SQSIGH(*),SQSIGV(*)
    complex(8), intent(in) :: field(-1:LGX1+1,-1:LGY1+1,-1:LGZ1+1,12,blsz,nfr)
    complex(8), intent(out) :: Jacc(0:LGX1,0:LGY1,0:LGZ1,blsz,blsz,nfr,2)
    Jacc=cmplx(0d0,0d0,8)
    call export_discrete_kernel(SQSIG,field,ome,LGX1,LGY1,LGZ1,blsz,nfr,SolMatFm)
"""
    return text[:first] + kernel + text[last:], specs


ORIENTATION_INPUT = "orientation.inp"


def orientation_source(text):
    """Let an optional orientation.inp set the material-orientation flag.

    buildgrid hard-codes ifdip = -1: inner solver cells take their TI axis from
    the dominant gradient of -2 ln Rh - ln Rv, whatever its size. A first integer
    in orientation.inp overrides it (0: the well-dip/bedding axis everywhere).
    Without the file the build behaves exactly as before; the value used is
    printed so the worker can verify it.
    """
    declaration = re.search(r"\n(\s*)integer :: it,jt,kt,NS,idmaxabsvals,vecsign, inder\n", text)
    if not declaration or text.count("f_ifdip=-1") != 1:
        raise ValueError("Unrecognized buildgrid material-orientation flag")
    indent = declaration[1]
    text = text.replace(declaration[0], declaration[0] + f"{indent}logical :: ad_orientation_input\n"
                        f"{indent}integer :: ad_orientation_unit\n", 1)
    reader = (f"f_ifdip=-1\n"
              f"{indent}inquire(file='{ORIENTATION_INPUT}', exist=ad_orientation_input)\n"
              f"{indent}if (ad_orientation_input) then\n"
              f"{indent}\topen(newunit=ad_orientation_unit, file='{ORIENTATION_INPUT}', status='old', action='read')\n"
              f"{indent}\tread(ad_orientation_unit,*) f_ifdip\n"
              f"{indent}\tclose(ad_orientation_unit)\n"
              f"{indent}end if\n"
              f"{indent}print *,'buildgrid: material orientation ifdip =',f_ifdip")
    return text.replace("f_ifdip=-1", reader, 1)


def phase_source(text):
    for name, terms in (("calcPha", (1, 2, 3, 4)), ("calcPha2", (1, 3))):
        start = text.index("\treal*8 :: m,n", text.index("subroutine " + name + "("))
        end = text.index("end subroutine " + name, start)
        signed = [f"{'+' if i in (1,2) else '-'}(a{i}*db{i}-b{i}*da{i})/(a{i}**2+b{i}**2)" for i in terms]
        text = text[:start] + "\tV=-180d0/pi*( &\n" + " &\n".join(signed) + ")\n" + text[end:]
    return text


def block_rom_source(text):
    """Turn a broken-down Block_ROM recurrence into a clean convergence error.

    When the solver stalls just above its tolerance the Gauss-Radau error bound
    can become NaN. The progress print then calls repeat() with a garbage count,
    which corrupts the heap and ends in a segmentation fault (exit 174). Here a
    non-finite or nonpositive bound stops the recurrence with convergenceflag = 0,
    so the caller writes its convergence_error.txt receipt, and the bar length of
    the print is clamped. Converged solves are unchanged.
    """
    lines = text.splitlines(keepends=True)
    declared = [i for i, line in enumerate(lines) if line.strip() == "integer :: i,j,l"]
    printed = [i for i, line in enumerate(lines) if line.lstrip().startswith("Write (*,130) ,i,bnd,")]
    if not declared or len(printed) != 1:
        raise ValueError("Unrecognized Block_ROM progress print")
    lines[declared[0]] = lines[declared[0]].replace("integer :: i,j,l", "integer :: i,j,l,bar")
    at = printed[0]
    indent = lines[at][:len(lines[at]) - len(lines[at].lstrip())]
    lines[at] = (f"{indent}IF (.NOT.(bnd == bnd) .OR. bnd <= 0d0 .OR. bnd >= 1d30) THEN\n"
                 f"{indent}\tWrite (*,130) ,i,bnd,' |',' BREAKDOWN   ','|'\n"
                 f"{indent}\tprint *, 'EXIT ROM: numerical breakdown (non-finite or nonpositive error bound)'\n"
                 f"{indent}\tsz_conv=i;\n"
                 f"{indent}\tconvergenceflag=0;\n"
                 f"{indent}\tNotConverged=.FALSE.;\n"
                 f"{indent}ELSE\n"
                 f"{indent}\tbar=INT(2*DMAX1(0d0,-DLOG10(bnd)))\n"
                 f"{indent}\tIF (bar < 0) bar = 0\n"
                 f"{indent}\tIF (bar > 26) bar = 26\n"
                 f"{indent}\tWrite (*,130) ,i,bnd,' |', repeat('-', bar),repeat('/', 26-bar) ,'|'\n")
    converged = next((i for i in range(at, len(lines)) if "Solver converged at iteration:" in lines[i]), None)
    closing = None if converged is None else next(
        (i for i in range(converged + 1, len(lines)) if "end if" in lines[i].lower()), None)
    if closing is None:
        raise ValueError("Unrecognized Block_ROM convergence test")
    lines[closing] = lines[closing].rstrip("\n") + f"\n{indent}END IF\n"
    return "".join(lines)


def barrier_source(text):
    """Drop the MPI_BARRIER that only the rank reporting a convergence error calls.

    The other rank never reaches a matching barrier, so with two ranks every
    convergence error hung the solve until the caller's timeout.
    """
    pattern = re.compile(r"(write\(999,'\(A,I0,A,I0\)'\) 'error at log-pt=',logpt,' rx-pt=',rxpt\s*\n\s*close\(999\)\s*\n)"
                         r"\s*call MPI_BARRIER\(MPI_COMM_WORLD,ierror\)\s*\n")
    text, count = pattern.subn(r"\1", text)
    if count != 1:
        raise ValueError("Unrecognized convergence-error branch in the parallel driver")
    return text


PATCHES = ["discrete adjoint capture (8 or 16 components, OpenMP)", "phase derivatives",
           "orientation.inp", "Block_ROM breakdown guard", "no MPI_BARRIER in the convergence-error branch"]


def build(source, destination, fidelity):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Use a fresh build directory: {destination}")
    destination.mkdir(parents=True)
    for path in (source / "src").glob("*.f90"):
        shutil.copy2(path, destination / path.name)
    for name in ("hdfql.mod", "libHDFql.so"):
        shutil.copy2(source / "src/hdfql_linux" / name, destination / name)
    shutil.copy2(Path(__file__).resolve().with_name("adjoint_capture.f90"), destination)
    matrix = "bai_em_matrix_mod2D.f90" if fidelity == "2D" else "bai_em_matrix_mod.f90"
    path = destination / matrix
    original = path.read_text()
    changed, specs = material_source(original)
    path.write_text(changed)
    path = destination / "buildgrid.f90"
    grid_original = path.read_text()
    path.write_text(orientation_source(grid_original))
    (destination / "adjoint-spec.json").write_text(json.dumps({"version": 2, "fidelity": fidelity,
        "source_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "buildgrid_source_sha256": hashlib.sha256(grid_original.encode()).hexdigest(),
        "orientation_input": ORIENTATION_INPUT, "patches": PATCHES, "stencils": specs}, indent=2))
    path = destination / "bai_em_atuzi_mod.f90"
    path.write_text(block_rom_source(path.read_text()))
    path = destination / f"fwd{fidelity}_parallel.f90"
    path.write_text(barrier_source(path.read_text()))
    path = destination / "jacobianprocessing.f90"
    path.write_text(phase_source(path.read_text()))
    path = destination / "bai_em_main_mod.f90"
    text = path.read_text().replace("CALL SQTENS(logpoint,sqsigh)", "sqsigh=sqsig")
    text = text.replace("CALL SQTENS(logpoint,sqsigv)", "sqsigv=sqsig")
    text = text.replace("call process_jacobian(LGX1,LGY1,LGZ1,nfr,SolMatOut,JaccOut)",
                        "! Model-space pullback is applied to outputs/adjoint by the worker.")
    path.write_text(text)
    sources = ["input.f90", "signalconversion.f90", "linear_interpolation_module.f90",
               "gridinterpolation.f90", "write_read_h5.f90", "buildgrid.f90", "output.f90",
               "bai_em_atuzi_mod.f90", "jacobianprocessing.f90", "adjoint_capture.f90", matrix,
               "bai_em_main_mod.f90", f"fwd{fidelity}_parallel.f90"]
    subprocess.run(["mpiifx", "-c", "-O3", "-qopenmp", "-qmkl", *sources], cwd=destination, check=True)
    subprocess.run(["mpiifx", "-o", f"fwd{fidelity}", *[str(p.name) for p in destination.glob("*.o")],
                    "-O3", "-qopenmp", "-qmkl", "-static-libgcc", "-static-intel", f"-L{destination}",
                    "-lHDFql", "-heap-arrays", "-Wl,--allow-multiple-definition"], cwd=destination, check=True)
    print(destination / f"fwd{fidelity}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--fidelity", choices=["2D", "3D"], required=True)
    args = parser.parse_args()
    build(args.source, args.destination, args.fidelity)
