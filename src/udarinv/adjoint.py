"""Exact reverse material map for the instrumented Fortran discrete adjoint.

Inputs are derivatives with respect to the six independent conductivity tensor
entries at each staggered node. Outputs differentiate independent natural-log
Rh and Rv at the input model cells. No finite-difference forward solves occur.
"""

import ast
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

import h5py
import numpy as np


def read_array(path, with_receiver=False):
    with Path(path).open("rb") as stream:
        shape = tuple(np.fromfile(stream, dtype="<i8", count=5))
        if len(shape) != 5 or any(n < 1 for n in shape):
            raise ValueError(f"Invalid adjoint tape: {path}")
        receiver = int(np.fromfile(stream, dtype="<i8", count=1)[0]) if with_receiver else None
        values = np.fromfile(stream, dtype="<f8")
    if values.size != np.prod(shape):
        raise ValueError(f"Incomplete adjoint tape: {path}")
    return values.reshape(shape, order="F"), receiver


def expression(text, variables):
    """Evaluate the small arithmetic-only stencil language captured at build time."""
    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name):
            return variables[node.id]
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in ("hlx", "hly", "hlz") and len(node.args) == 1:
                return variables[node.func.id][visit(node.args[0])]
        raise ValueError(f"Invalid material stencil expression: {text}")
    return visit(ast.parse(text, mode="eval"))


def nearest_indices(axis, points):
    upper = np.clip(np.searchsorted(axis, points), 1, len(axis) - 1)
    lower = upper - 1
    return np.where(points - axis[lower] <= axis[upper] - points, lower, upper)


def gradient_matrix(axis, difference_axis=None):
    """builddiffop followed by the forward's staggered-to-centre interpolation."""
    difference_axis = axis if difference_axis is None else difference_axis
    n = len(axis)
    differences = np.zeros((n - 1, n))
    rows = np.arange(n - 1)
    differences[rows, rows] = -1 / np.diff(difference_axis[:n])
    differences[rows, rows + 1] = 1 / np.diff(difference_axis[:n])
    midpoints = (axis[:-1] + axis[1:]) / 2
    interpolation = np.zeros((n, n - 1))
    interpolation[0, 0] = interpolation[-1, -1] = 1
    for i in range(1, n - 1):
        t = (axis[i] - midpoints[i - 1]) / (midpoints[i] - midpoints[i - 1])
        interpolation[i, i - 1:i + 1] = [1 - t, t]
    return interpolation @ differences


def apply_axis(matrix, array, axis):
    return np.moveaxis(np.tensordot(matrix, array, axes=(1, axis)), 0, axis)


def tridiagonal(matrix):
    """(sub, main, super) bands of a square tridiagonal operator; refuses wider ones."""
    matrix = np.asarray(matrix, dtype=float)
    n = matrix.shape[0]
    if matrix.shape != (n, n) or np.any(np.triu(matrix, 2)) or np.any(np.tril(matrix, -2)):
        raise ValueError("Expected a square tridiagonal operator")
    return (np.r_[0., np.diagonal(matrix, -1)], np.diagonal(matrix).copy(),
            np.r_[np.diagonal(matrix, 1), 0.])


def apply_tridiagonal(bands, array, axis, transpose=False):
    """apply_axis for a banded operator (or its transpose) in O(n) per line."""
    lower, main, upper = bands
    if transpose:
        lower, upper = np.r_[0., upper[:-1]], np.r_[lower[1:], 0.]
    values = np.moveaxis(array, axis, 0)
    shape = (-1,) + (1,) * (values.ndim - 1)
    result = main.reshape(shape) * values
    result[1:] += lower[1:].reshape(shape) * values[:-1]
    result[:-1] += upper[:-1].reshape(shape) * values[1:]
    return np.moveaxis(result, 0, axis)


def scatter_plan(flat_index):
    """Precomputed, duplicate-safe replacement for np.add.at on fixed row targets."""
    flat_index = np.asarray(flat_index, dtype=np.intp).ravel()
    order = np.argsort(flat_index, kind="stable")
    ordered = flat_index[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]]) if len(ordered) else ordered
    if len(starts) == len(flat_index):
        return {"targets": flat_index, "order": None, "starts": None}
    return {"targets": ordered[starts], "order": order, "starts": starts}


def scatter_add(target, plan, values):
    """target[rows] += values, summing rows that share a target (as np.add.at does)."""
    if not len(plan["targets"]):
        return
    if plan["order"] is None:
        target[plan["targets"]] += values
    else:
        target[plan["targets"]] += np.add.reduceat(values[plan["order"]], plan["starts"], axis=0)


def orientation_pullback(log_fields, axes, subdivisions, qbar):
    """Transpose through the principal left-singular-vector and gradient map."""
    # Match the legacy forward exactly: its y difference uses Zout spacings.
    operators = [tridiagonal(gradient_matrix(axes[0])), tridiagonal(gradient_matrix(axes[1], axes[2])),
                 tridiagonal(gradient_matrix(axes[2]))]
    trace = -2 * log_fields[0] - log_fields[1]
    grad = np.stack([apply_tridiagonal(op, trace, axis) for axis, op in enumerate(operators)])
    nx, ny, nz = [len(axis) // n for axis, n in zip(axes, subdivisions)]
    sx, sy, sz = subdivisions
    patches = grad.reshape(3, nx, sx, ny, sy, nz, sz).transpose(1, 3, 5, 0, 2, 4, 6)
    patches = patches.reshape(nx, ny, nz, 3, sx * sy * sz)
    covariance = patches @ patches.swapaxes(-1, -2)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    q = vectors[..., -1]
    sign_index = np.argmax(np.abs(q), axis=-1)
    signs = np.take_along_axis(q, sign_index[..., None], axis=-1)[..., 0]
    q = q * np.where(signs < 0, -1, 1)[..., None]
    # A repeated largest singular value has no unique orientation derivative.
    gap = eigenvalues[..., -1] - eigenvalues[..., -2]
    active = np.linalg.norm(qbar, axis=-2).max(axis=-1) > 1e-14
    if np.any(active & (gap <= np.maximum(eigenvalues[..., -1], 1e-30) * 1e-10)):
        raise ValueError("Material orientation is not differentiable (repeated dominant singular value)")
    # pbar = sum_i (o_i (q^T P) + q (o_i^T P)) c_i, gathered into two outer products.
    qta = np.einsum("...i,...ij->...j", q, patches)
    direction_sum = np.zeros((*q.shape, qbar.shape[-1]))
    patch_sum = np.zeros((*qta.shape, qbar.shape[-1]))
    for index in (0, 1):
        other = vectors[..., index]
        denom = eigenvalues[..., -1] - eigenvalues[..., index]
        coeff = np.einsum("...i,...ic->...c", other, qbar)
        coeff = np.divide(coeff, denom[..., None], out=np.zeros_like(coeff), where=denom[..., None] > 1e-30)
        ota = np.einsum("...i,...ij->...j", other, patches)
        direction_sum += other[..., :, None] * coeff[..., None, :]
        patch_sum += ota[..., :, None] * coeff[..., None, :]
    response = qbar.shape[-1]
    pbar = (direction_sum[..., :, None, :] * qta[..., None, :, None]
            + q[..., :, None, None] * patch_sum[..., None, :, :])
    gbar = pbar.reshape(nx, ny, nz, 3, sx, sy, sz, response)
    gbar = gbar.transpose(3, 0, 4, 1, 5, 2, 6, 7).reshape(3, *trace.shape, response)
    tbar = sum(apply_tridiagonal(op, gbar[axis], axis, transpose=True) for axis, op in enumerate(operators))
    return np.stack([-2 * tbar, -tbar])


def _prepare(directory, spec, receiver):
    """Everything the pullback needs that does not depend on the tool kernel."""
    directory = Path(directory)
    material, _ = read_array(directory / f"outputs/adjoint/material_rx{receiver}.bin")
    tokens = (directory / f"outputs/adjoint/grid_rx{receiver}.txt").read_text().split()
    subdivisions = tuple(map(int, tokens[:3]))
    lengths = list(map(int, tokens[3:6]))
    pos = 6
    axes = []
    for length in lengths:
        axes.append(np.asarray(tokens[pos:pos + length], dtype=float))
        pos += length
    grid_lengths = list(map(int, tokens[pos:pos + 3]))
    pos += 3
    grid = []
    for length in grid_lengths:
        grid.append(np.asarray(tokens[pos:pos + length], dtype=float))
        pos += length
    with h5py.File(directory / "globalmodel.h5") as model:
        source_axes = [model[f"formation/{name}"][:] for name in "xyz"]
        source = np.stack([np.log(model[f"formation/{name}"][:].transpose(2, 1, 0)) for name in ("Rh", "Rv")])
        well = {key: model[f"wellpath/{key}"][0, 0] for key in ("X", "Y", "Z", "Inclination", "Azimuth")}
    x, y, z = np.meshgrid(*axes, indexing="ij")
    dip, azimuth = np.deg2rad([well["Inclination"], well["Azimuth"]])
    xr = x * np.cos(dip) + z * np.sin(dip)
    zr = -x * np.sin(dip) + z * np.cos(dip)
    points = [xr * np.cos(azimuth) - y * np.sin(azimuth) + well["X"],
              xr * np.sin(azimuth) + y * np.cos(azimuth) + well["Y"], zr + well["Z"]]
    ids = tuple(nearest_indices(axis, point) for axis, point in zip(source_axes, points))
    log_fields = source[(slice(None), *ids)]
    indices = np.indices(material.shape[:3])
    i2, j2, k2 = indices + 1
    variables = {"i2": i2, "j2": j2, "k2": k2, "j21": np.maximum(1, j2 - 1), "j2cent": 1,
                 "i": indices[0] * 2, "j": indices[1] * 2, "k": indices[2] * 2}
    for name, values in zip(("hlx", "hly", "hlz"), grid):
        variables[name] = np.r_[0., np.diff(values)]
    variables.update(dict(zip(("nsub1", "nsub2", "nsub3"), subdivisions)))
    # Pre-resolve the material stencil to flat sample indices and weights once;
    # these depend only on geometry and the tape, never on the tool kernel.
    plans = []
    qshape = tuple(n // s for n, s in zip(lengths, subdivisions))
    for stencil in spec["stencils"]:
        kind = len(plans)
        sh, sv, cy, sy, cz, sz, uses, gi, gj, gk = np.moveaxis(material[..., kind], -1, 0)
        denominator = expression(stencil["denominator"], variables)
        entries = []
        for corner, subexpressions in stencil["samples"].items():
            weight = expression(stencil["weights"][corner], variables) / denominator / np.prod(subdivisions)
            for sub in np.ndindex(subdivisions):
                variables.update(dict(zip(("isub1", "isub2", "isub3"), np.array(sub) + 1)))
                sample = [expression(value, variables) for value in subexpressions]
                fine = np.broadcast_arrays(*[2 * n * (block - 1) + local - 1 for n, block, local in
                                             zip(subdivisions, sample[3:], sample[:3])])
                valid = weight > 0
                for axis, length in zip(fine, lengths):
                    valid = valid & (axis >= 0) & (axis < length)
                selected = tuple(axis[valid].astype(int) for axis in fine)
                active = np.broadcast_to(weight, valid.shape)[valid]
                # Store the reverse map as flat gathers/scatters with the
                # sigma and stencil weights folded into one coefficient.
                entries.append({
                    "node": np.flatnonzero(valid),
                    "scatter": scatter_plan(np.ravel_multi_index(selected, lengths)),
                    "coef_h": -np.exp(-log_fields[(0, *selected)]) * active,
                    "coef_v": -np.exp(log_fields[(1, *selected)]) * active * sv[valid] ** 2,
                })
        normal = np.stack([-sy * cz, -sy * sz, cy], axis=-1)
        used = uses > 0
        gradient_node = (gi.astype(int) - 1, gj.astype(int) - 1, gk.astype(int) - 1)
        plans.append({"entries": entries, "normal": normal, "used": np.flatnonzero(used),
                      "gradient_scatter": scatter_plan(np.ravel_multi_index(
                          tuple(node[used] for node in gradient_node), qshape)),
                      "sh_minus_sv": (sv - sh)})
    source_scatter = scatter_plan(np.ravel_multi_index(tuple(axis.ravel() for axis in ids),
                                                       source.shape[1:]))
    return {"spec": spec, "lengths": lengths, "axes": axes, "subdivisions": subdivisions,
            "ids": ids, "log_fields": log_fields, "source_shape": source.shape, "plans": plans,
            "qshape": qshape, "source_scatter": source_scatter}


def _pullback_prepared(pre, kernels):
    """Apply the reverse material map to stacked kernels (..., 24, ncomp, tools)."""
    kernels = np.asarray(kernels, dtype=float)
    tools = kernels.shape[-1]
    ncomp = kernels.shape[-2]
    # The geosignal components form the response axis; merge (ncomp, tools).
    kernels = kernels.reshape(*kernels.shape[:-2], ncomp * tools)
    response = ncomp * tools
    lengths = pre["lengths"]
    finebar = np.zeros((2, *lengths, response))
    fine_rows = finebar.reshape(2, -1, response)
    qshape = pre["qshape"]
    qbar = np.zeros((*qshape, 3, response))
    for kind, plan in enumerate(pre["plans"]):
        partial = kernels[..., 6 * kind:6 * (kind + 1), :]
        normal = plan["normal"]
        nn = np.stack([normal[..., 0]**2, normal[..., 1]**2, normal[..., 2]**2,
                       normal[..., 0]*normal[..., 1], normal[..., 1]*normal[..., 2],
                       normal[..., 0]*normal[..., 2]], axis=-1)
        svbar = np.einsum("...i,...ic->...c", nn, partial)
        shbar = partial[..., :3, :].sum(axis=-2) - svbar
        nbar = np.empty((*normal.shape, response))
        nbar[..., 0, :] = (2 * normal[..., 0, None] * partial[..., 0, :]
                           + normal[..., 1, None] * partial[..., 3, :]
                           + normal[..., 2, None] * partial[..., 5, :])
        nbar[..., 1, :] = (2 * normal[..., 1, None] * partial[..., 1, :]
                           + normal[..., 0, None] * partial[..., 3, :]
                           + normal[..., 2, None] * partial[..., 4, :])
        nbar[..., 2, :] = (2 * normal[..., 2, None] * partial[..., 2, :]
                           + normal[..., 1, None] * partial[..., 4, :]
                           + normal[..., 0, None] * partial[..., 5, :])
        nbar *= plan["sh_minus_sv"][..., None, None]
        if len(plan["used"]):
            used = nbar.reshape(-1, 3, response)[plan["used"]] * np.array([-1., 1., -1.])[None, :, None]
            scatter_add(qbar.reshape(-1, 3 * response), plan["gradient_scatter"],
                        used.reshape(len(used), 3 * response))
        shbar_rows, svbar_rows = shbar.reshape(-1, response), svbar.reshape(-1, response)
        for entry in plan["entries"]:
            scatter_add(fine_rows[0], entry["scatter"], entry["coef_h"][:, None] * shbar_rows[entry["node"]])
            scatter_add(fine_rows[1], entry["scatter"], entry["coef_v"][:, None] * svbar_rows[entry["node"]])
    if np.any(qbar):
        finebar += orientation_pullback(pre["log_fields"], pre["axes"], pre["subdivisions"], qbar)
    result = np.zeros((*pre["source_shape"], response))
    for field in (0, 1):
        scatter_add(result[field].reshape(-1, response), pre["source_scatter"],
                    finebar[field].reshape(-1, response))
    # (2, nx, ny, nz, response) -> (2, response, nz, ny, nx); response is
    # component-major so it splits into (ncomp, tools).
    transposed = result.transpose(0, 4, 3, 2, 1)
    return transposed.reshape(2, ncomp, tools, transposed.shape[2],
                              transposed.shape[3], transposed.shape[4])


def pullback(directory, spec_path, tool_index):
    kernel, receiver = read_array(Path(directory) / f"outputs/adjoint/kernel_tool{tool_index}.bin", True)
    return _pullback_prepared(
        _prepare(directory, json.loads(Path(spec_path).read_text()), receiver),
        kernel[..., None])[:, :, 0]


def finish(directory, spec_path, tools, workers=None):
    directory = Path(directory)
    output = directory / "outputs/jacobian"
    output.mkdir(exist_ok=True)
    spec = json.loads(Path(spec_path).read_text())
    groups = {}
    for index in range(1, len(tools) + 1):
        with (directory / f"outputs/adjoint/kernel_tool{index}.bin").open("rb") as stream:
            np.fromfile(stream, dtype="<i8", count=5)
            receiver = int(np.fromfile(stream, dtype="<i8", count=1)[0])
        groups.setdefault(receiver, []).append(index)
    # Frequencies sharing a receiver spacing share a material grid. Different
    # spacings do not: vectorizing them together would silently mix derivatives.
    # The pullback is linear and separable per tool, and NumPy releases the
    # GIL in its heavy array kernels: prepare each receiver grid once, then
    # pull every tool back on its own thread. HDF5 writes stay on this thread.
    tasks = [(receiver, index) for receiver, indices in groups.items() for index in indices]
    threads = max(1, min(len(tasks), workers or len(os.sched_getaffinity(0))))
    with ThreadPoolExecutor(max_workers=threads) as pool:
        prepared = dict(zip(groups, pool.map(lambda receiver: _prepare(directory, spec, receiver), groups)))

        def tool_jacobian(task):
            receiver, index = task
            kernel = read_array(directory / f"outputs/adjoint/kernel_tool{index}.bin", True)[0]
            return index, _pullback_prepared(prepared[receiver], kernel[..., None])[:, :, 0]
        results = list(pool.map(tool_jacobian, tasks))
    for index, jacobian in results:
        with h5py.File(output / f"jac_logpt1_tool{index}.h5", "w") as handle:
            handle["jac"] = jacobian
            handle.attrs["parameterization"] = "ln(Rh),ln(Rv)"
            handle.attrs["derivative"] = "discrete-adjoint-v1"
