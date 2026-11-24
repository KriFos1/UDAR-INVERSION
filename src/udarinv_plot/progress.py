"""The course of a chain: fit, starts, solver tolerance and cost per logging point."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def progress(results):
    rows = results.rows
    x = np.array([row['index'] for row in rows])
    warm = np.array([row['start'] == 'previous_map' for row in rows])
    reset = np.array([row['prior_reset'] for row in rows])
    figure, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    final = np.array([row['misfit'] for row in rows])
    axes[0].semilogy(x, [row['start_misfit'] for row in rows], color='.6', lw=.8, label='at the start of the kept search')
    axes[0].semilogy(x, final, color='k', lw=.8)
    axes[0].scatter(x[warm], final[warm], s=12, color='tab:blue', label='final, start: previous MAP', zorder=3)
    axes[0].scatter(x[~warm], final[~warm], s=12, color='tab:orange', label='final, start: prior mean', zorder=3)
    if reset.any():
        axes[0].scatter(x[reset], final[reset], s=60, facecolors='none', edgecolors='red', label='background reset')
    axes[0].axhline(results.config['search']['target_misfit'], color='red', ls='--', lw=.8)
    axes[0].set_ylabel('chi2 / n')
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].semilogy(x, [row['solver_tolerance'] for row in rows], drawstyle='steps-mid', color='tab:purple')
    axes[1].set_ylabel('final solver tolerance')
    axes[2].bar(x, [row['iterations'] for row in rows], color='tab:green', label='accepted steps')
    axes[2].bar(x, [row['solver_failures'] for row in rows], color='tab:red', width=.4, label='failed solves')
    axes[2].set_ylabel('count')
    axes[2].legend(fontsize=7)
    axes[3].bar(x, [row['seconds'] for row in rows], color='.4')
    axes[3].set_ylabel('seconds per point')
    axes[3].set_xlabel('Logging point')
    for axis in axes:
        axis.grid(alpha=.3)
    total = sum(row['seconds'] for row in rows)
    figure.suptitle(f'{results.run.name}: {len(rows)} points, {int((final <= results.config["search"]["target_misfit"]).sum())} '
                    f'at the target, {total / 3600.:.2f} h', fontsize=11)
    figure.tight_layout()
    return figure
