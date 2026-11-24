"""Summary of a finished chain and, for a benchmark, its error against the true model.

The true model is finer than the carry lattice. A lattice cell that holds thin
horizontal layers has a known effective medium: Rh is the harmonic and Rv the
arithmetic average of the layers (layered upscaling), which is what the data
respond to. All scores are in log10 Rh on the lattice of each logging point.
"""

import json
from pathlib import Path

import h5py
import numpy as np

from .grid import FT, overlap_weights

LN10 = np.log(10.)


class Truth:
    """The benchmark model in the well plane, upscaled onto lattice cells on demand."""

    def __init__(self, benchmark):
        self.path = Path(benchmark) / 'globalmodel.h5'
        with h5py.File(self.path, 'r') as handle:
            self.x, self.y, self.z = (handle[f'formation/{name}'][:].ravel() / FT for name in 'xyz')
        self._plane = None

    def plane(self, well_y=0.):
        """Rh and Rv as (x, z) arrays of the model layer nearest the well plane."""
        layer = int(np.argmin(np.abs(self.y - well_y)))
        if self._plane is None or self._plane[0] != layer:
            with h5py.File(self.path, 'r') as handle:
                self._plane = (layer, handle['formation/Rh'][:, layer, :].T, handle['formation/Rv'][:, layer, :].T)
        return self._plane[1], self._plane[2]

    def on_cells(self, edges_x, edges_z, well_y=0.):
        """ln Rh (harmonic) and ln Rv (arithmetic) of the cells with these edges."""
        rh, rv = self.plane(well_y)
        wx, wz = overlap_weights(self.x, edges_x), overlap_weights(self.z, edges_z)
        return -np.log(wx @ (1. / rh) @ wz.T), np.log(wx @ rv @ wz.T)


def point_scores(handoff, truth, near=(25., 50.)):
    """Scores of one logging point on its lattice; ``near`` = (along, vertical) half-widths [ft] of the bit zone."""
    edges_x, edges_z = handoff['edges_x'], handoff['edges_z']
    nx, nz = len(edges_x) - 1, len(edges_z) - 1
    bit = handoff['bit_ft']
    true_rh, _ = truth.on_cells(edges_x, edges_z, float(bit[1]))
    mean = handoff['mean'][:nx * nz].reshape(nx, nz)
    std = handoff['std'][:nx * nz].reshape(nx, nz)
    prior = handoff['prior_mean'][:nx * nz].reshape(nx, nz)
    error = (mean - true_rh) / LN10
    x, z = .5 * (edges_x[1:] + edges_x[:-1]), .5 * (edges_z[1:] + edges_z[:-1])
    zone = (np.abs(x - bit[0]) <= near[0])[:, None] & (np.abs(z - bit[2]) <= near[1])[None, :]
    standardized = np.abs(mean - true_rh) / std
    return {'rh_rmse': float(np.sqrt(np.mean(error ** 2))),
            'prior_rh_rmse': float(np.sqrt(np.mean(((prior - true_rh) / LN10) ** 2))),
            'rh_rmse_near_bit': float(np.sqrt(np.mean(error[zone] ** 2))),
            'bias_ln': float(np.mean(mean - true_rh)), 'mean_std_ln_rh': float(std.mean()),
            'within_1_2_3_std': [float(np.mean(standardized <= k)) for k in (1, 2, 3)],
            'z_rms_near_bit': float(np.sqrt(np.mean(standardized[zone] ** 2)))}


def summarize(run, truth=True):
    """Fit, starts and solver record of a chain; with ``truth`` also the benchmark scores."""
    run = Path(run)
    rows = json.loads((run / 'summary.json').read_text())
    config = json.loads((run / 'config.json').read_text())
    if not rows:
        raise ValueError(f'{run} has no finished point')
    misfit = np.array([r['misfit'] for r in rows])
    target = config['search']['target_misfit']
    summary = {'run': run.name, 'dimension': config['run']['dimension'], 'points': len(rows),
               'indices': [rows[0]['index'], rows[-1]['index']], 'target_misfit': target,
               'at_target': int(np.sum(misfit <= target)),
               'misfit': {'median': float(np.median(misfit)), 'p90': float(np.percentile(misfit, 90)),
                          'max': float(misfit.max())},
               'starts': {name: sum(r['start'] == name for r in rows) for name in sorted({r['start'] for r in rows})},
               'prior_resets': [r['index'] for r in rows if r['prior_reset']],
               'not_at_final_tolerance': [r['index'] for r in rows if not r['at_final_tolerance']],
               'solver_failures': int(sum(r['solver_failures'] for r in rows)),
               'solver_evaluations': int(sum(r['solver_evaluations'] or 0 for r in rows)),
               'seconds': {'total': float(sum(r['seconds'] for r in rows)),
                           'median': float(np.median([r['seconds'] for r in rows]))}}
    if truth:
        model = Truth(config['data']['path'])
        scores = []
        for r in rows:
            with np.load(run / f"point-{r['index']:04d}" / 'handoff.npz') as handoff:
                scores.append(point_scores(handoff, model))
        summary['truth'] = {key: float(np.mean([s[key] for s in scores]))
                            for key in ('rh_rmse', 'prior_rh_rmse', 'rh_rmse_near_bit', 'bias_ln', 'mean_std_ln_rh',
                                        'z_rms_near_bit')}
        summary['truth']['within_1_2_3_std'] = np.mean([s['within_1_2_3_std'] for s in scores], axis=0).tolist()
        summary['truth']['upscaling'] = 'layered (Rh harmonic, Rv arithmetic)'
        summary['per_point'] = [{'index': r['index'], **s} for r, s in zip(rows, scores)]
    return summary


def format_summary(summary):
    lines = [f"{summary['run']}: {summary['dimension']}D, points {summary['indices'][0]}-{summary['indices'][1]} "
             f"({summary['points']})",
             f"  at target chi2/n <= {summary['target_misfit']:g}: {summary['at_target']} of {summary['points']}; "
             f"misfit median {summary['misfit']['median']:.2f}, p90 {summary['misfit']['p90']:.2f}, "
             f"max {summary['misfit']['max']:.2f}",
             '  starts: ' + ', '.join(f'{name} {count}' for name, count in summary['starts'].items())
             + f"; background resets {len(summary['prior_resets'])}; "
             f"not at the final tolerance {len(summary['not_at_final_tolerance'])}",
             f"  solver: {summary['solver_evaluations']} solves, {summary['solver_failures']} failed; "
             f"{summary['seconds']['total'] / 3600.:.2f} h in total, median {summary['seconds']['median']:.0f} s per point"]
    if 'truth' in summary:
        t = summary['truth']
        lines.append(f"  against the true model ({t['upscaling']}): Rh error {t['rh_rmse']:.3f} log10 "
                     f"(prior {t['prior_rh_rmse']:.3f}, near the bit {t['rh_rmse_near_bit']:.3f}); "
                     f"within 1/2/3 std {t['within_1_2_3_std'][0]:.0%} / {t['within_1_2_3_std'][1]:.0%} / "
                     f"{t['within_1_2_3_std'][2]:.0%}; z-rms near the bit {t['z_rms_near_bit']:.2f}")
    return '\n'.join(lines)
