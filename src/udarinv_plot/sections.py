"""Sections of one logging point's result on the graded solve grid."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def sections(results, index, limits=None):
    """MAP Rh and posterior std of ln Rh through the bit: the well plane and, in 3D, the cross-well
    and horizontal sections. Cells are drawn with their graded edges. Returns the figure."""
    saved = results.map(index)
    bit = saved['bit_ft']
    rh, std = np.exp(saved['map'][0].astype(float)), saved['posterior_std'][0].astype(float)
    prior_std = saved['prior_std'][0].astype(float)
    if results.dimension == 2:
        views = [('Well plane', 'edges_x', 'edges_z', rh, std, prior_std, bit[0], bit[2], 'X [ft]', 'TVD [ft]', True)]
    else:
        iy = int(np.argmin(np.abs(saved['centres_y'] - bit[1])))
        ix = int(np.argmin(np.abs(saved['centres_x'] - bit[0])))
        iz = int(np.argmin(np.abs(saved['centres_z'] - bit[2])))
        views = [('Well plane', 'edges_x', 'edges_z', rh[:, iy], std[:, iy], prior_std[:, iy], bit[0], bit[2],
                  'X [ft]', 'TVD [ft]', True),
                 ('Across the well', 'edges_y', 'edges_z', rh[ix], std[ix], prior_std[ix], bit[1], bit[2],
                  'Y [ft]', 'TVD [ft]', True),
                 ('Horizontal, at the bit', 'edges_x', 'edges_y', rh[:, :, iz], std[:, :, iz], prior_std[:, :, iz],
                  bit[0], bit[1], 'X [ft]', 'Y [ft]', False)]
    if limits is None:
        limits = np.percentile(rh, [2, 98])
    figure, axes = plt.subplots(2, len(views), figsize=(5.2 * len(views), 8), squeeze=False)
    for column, (title, first, second, values, spread, prior, mark_a, mark_b, label_a, label_b, depth) in enumerate(views):
        a, b = saved[first], saved[second]
        top = axes[0, column].pcolormesh(a, b, values.T, norm=LogNorm(*limits), cmap='viridis', shading='flat')
        bottom = axes[1, column].pcolormesh(a, b, (spread / prior).T, vmin=0., vmax=1., cmap='magma_r', shading='flat')
        for axis in axes[:, column]:
            axis.plot(mark_a, mark_b, marker='x', color='red', ms=7)
            axis.set_xlabel(label_a)
            axis.set_ylabel(label_b)
            if depth:
                axis.invert_yaxis()
        axes[0, column].set_title(f'{title}: MAP Rh', fontsize=10)
        axes[1, column].set_title(f'{title}: posterior / prior std of ln Rh', fontsize=10)
        figure.colorbar(top, ax=axes[0, column], label='ohm m')
        figure.colorbar(bottom, ax=axes[1, column])
    row = next(r for r in results.rows if r['index'] == index)
    figure.suptitle(f"{results.run.name}, point {index}: chi2/n {row['misfit']:.2f}, start {row['start']}", fontsize=11)
    figure.tight_layout()
    return figure
