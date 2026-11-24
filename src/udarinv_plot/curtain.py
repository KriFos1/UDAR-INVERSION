"""The at-bit curtain: Rh and Rv along the well as they were estimated when the bit was there."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np

from .results import painted


def washed(values, weight, norm, cmap):
    """RGBA image of ``values`` faded to white by ``weight`` in [0, 1]; NaN cells are grey."""
    rgba = cmap(norm(np.ma.masked_invalid(values)))
    weight = np.clip(np.nan_to_num(weight, nan=0.), 0., 1.)[..., None]
    rgba[..., :3] = rgba[..., :3] * (1. - weight) + weight
    rgba[np.isnan(values)] = (.75, .75, .75, 1.)
    return rgba


def draw_panel(axis, canvas, values, weight, norm, cmap, title):
    image = washed(values.T, None if weight is None else weight.T, norm, cmap) if weight is not None \
        else washed(values.T, np.zeros_like(values.T), norm, cmap)
    extent = (canvas['edges_x'][0], canvas['edges_x'][-1], canvas['edges_z'][-1], canvas['edges_z'][0])
    axis.imshow(image, extent=extent, aspect='auto', interpolation='nearest')
    axis.plot(canvas['bits'][:, 0], canvas['bits'][:, 2], color='k', lw=.8)
    axis.plot(*canvas['bits'][-1, [0, 2]], marker='x', color='red', ms=6)
    axis.set_title(title, fontsize=10)
    axis.set_ylabel('TVD [ft]')


def curtain(results, upto=None, limits=None, linear=False, wash=True, figure=None):
    """Truth (if known), Rh and Rv curtains and the misfit per point. Returns the figure.

    The wash fades a cell to white as its posterior std approaches the background
    std (ratio 1): full colour where the data determined the cell.
    """
    canvas = painted(results, upto)
    prior = results.config['prior']
    rh, rv = np.exp(canvas['rh']), np.exp(canvas['rh'] + canvas['latent'])
    rv_std = np.sqrt(canvas['rh_std'] ** 2 + canvas['latent_std'] ** 2)        # block correlation neglected
    rv_background = np.hypot(prior['rh_std'], prior['latent_std'])
    truth = canvas['truth']
    if limits is None:
        reference = np.exp(truth[0]) if truth is not None else rh
        limits = np.nanpercentile(reference, [2, 98])
    norm = Normalize(*limits) if linear else LogNorm(*limits)
    cmap = plt.get_cmap('viridis')
    panels = ([('True Rh (layered upscaling)', np.exp(truth[0]), None)] if truth is not None else []) + [
        ('Rh at the bit' + (' (white: no information)' if wash else ''), rh,
         canvas['rh_std'] / prior['rh_std'] if wash else None),
        ('Rv at the bit', rv, rv_std / rv_background if wash else None)]
    figure = figure or plt.figure(figsize=(12, 2.6 * len(panels) + 2.2))
    grid = figure.add_gridspec(len(panels) + 1, 2, width_ratios=[1, .015], height_ratios=[1] * len(panels) + [.7],
                               hspace=.35, wspace=.02)
    axes = [figure.add_subplot(grid[row, 0]) for row in range(len(panels))]
    for axis, (title, values, weight) in zip(axes, panels):
        draw_panel(axis, canvas, values, weight, norm, cmap, title)
        axis.tick_params(labelbottom=False)
    axes[-1].tick_params(labelbottom=True)
    axes[-1].set_xlabel('X along the well [ft]')
    figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=figure.add_subplot(grid[:len(panels), 1]),
                    label='Resistivity [ohm m]')
    misfit = figure.add_subplot(grid[len(panels), 0])
    rows = [row for row in results.rows if row['index'] in canvas['indices']]
    x = canvas['bits'][:, 0]
    warm = np.array([row['start'] == 'previous_map' for row in rows])
    values = np.array([row['misfit'] for row in rows])
    misfit.semilogy(x, values, color='.6', lw=.8)
    misfit.scatter(x[warm], values[warm], s=10, color='tab:blue', label='start: previous MAP')
    misfit.scatter(x[~warm], values[~warm], s=10, color='tab:orange', label='start: prior mean')
    misfit.axhline(results.config['search']['target_misfit'], color='red', ls='--', lw=.8, label='target')
    misfit.set_xlim(canvas['edges_x'][0], canvas['edges_x'][-1])
    misfit.set_xlabel('Bit X [ft]')
    misfit.set_ylabel('chi2 / n')
    misfit.legend(fontsize=7, ncol=3, loc='upper right')
    misfit.grid(alpha=.3)
    figure.suptitle(f"{results.run.name}: {results.dimension}D inversion, points {canvas['indices'][0]}-"
                    f"{canvas['indices'][-1]}", fontsize=11)
    return figure
