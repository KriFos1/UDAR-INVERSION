"""How well the MAP models reproduce the data."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def _tables(results):
    """Observed, predicted and std as (points, channels) arrays; unused channels are NaN."""
    observed, predicted, std = [], [], []
    for index in results.indices:
        h = results.handoff(index)
        rows = [np.full(h['mask'].shape, np.nan) for _ in range(3)]
        for row, values in zip(rows, (h['observations'], h['prediction'], np.sqrt(h['variance']))):
            row[h['mask']] = values
        observed.append(rows[0])
        predicted.append(rows[1])
        std.append(rows[2])
    return np.array(observed), np.array(predicted), np.array(std)


def residuals(results):
    """Map of the standardised residuals (prediction - data) / std: channels against logging points."""
    observed, predicted, std = _tables(results)
    names, tools = results.component_names(), results.tools()
    figure, axis = plt.subplots(figsize=(12, .16 * observed.shape[1] + 2.))
    image = axis.imshow(((predicted - observed) / std).T, aspect='auto', cmap='RdBu_r', vmin=-4, vmax=4,
                        interpolation='nearest', extent=(results.indices[0] - .5, results.indices[-1] + .5,
                                                         observed.shape[1], 0))
    axis.set_facecolor('.8')
    axis.set_yticks([len(names) * (tool + .5) for tool in range(len(tools))])
    axis.set_yticklabels([f'{frequency:g} kHz\n{spacing:g} ft' for frequency, spacing in tools], fontsize=7)
    for tool in range(1, len(tools)):
        axis.axhline(len(names) * tool, color='k', lw=.5)
    axis.set_xlabel('Logging point')
    axis.set_title(f'{results.run.name}: (prediction - data) / std; grey = channel not used; '
                   f"components per tool: {', '.join(names)}", fontsize=8)
    figure.colorbar(image, ax=axis, label='standard deviations')
    figure.tight_layout()
    return figure


def signals(results, tool):
    """Observed and predicted geosignals of one tool along the well, one panel per component."""
    observed, predicted, std = _tables(results)
    names, tools = results.component_names(), results.tools()
    if not 0 <= tool < len(tools):
        raise ValueError(f'The tool number is 0..{len(tools) - 1}')
    columns = 4
    figure, axes = plt.subplots(len(names) // columns, columns, figsize=(15, 2.3 * len(names) // columns), sharex=True)
    x = np.array(results.indices)
    for number, (name, axis) in enumerate(zip(names, axes.ravel())):
        channel = len(names) * tool + number
        axis.fill_between(x, observed[:, channel] - 2 * std[:, channel], observed[:, channel] + 2 * std[:, channel],
                          color='.85', label='data +/- 2 std')
        axis.plot(x, observed[:, channel], color='k', lw=.8, label='data')
        axis.plot(x, predicted[:, channel], color='tab:red', lw=.8, label='MAP prediction')
        axis.set_title(f"{name} [{'deg' if number % 2 == 0 else 'dB'}]", fontsize=9)
        axis.grid(alpha=.3)
    axes.ravel()[0].legend(fontsize=7)
    for axis in axes[-1]:
        axis.set_xlabel('Logging point')
    frequency, spacing = tools[tool]
    figure.suptitle(f'{results.run.name}: {frequency:g} kHz, {spacing:g} ft', fontsize=11)
    figure.tight_layout()
    return figure
