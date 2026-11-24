"""The curtain growing with the bit, one frame per logging point."""

import io

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .curtain import curtain
from .results import painted


def animate(results, output, every=1, fps=4., dpi=80, **options):
    """Write a GIF of the at-bit curtain; ``every`` skips logging points. Colours are those of the final curtain."""
    from PIL import Image
    final = painted(results)
    if 'limits' not in options or options['limits'] is None:
        reference = np.exp(final['truth'][0]) if final['truth'] is not None else np.exp(final['rh'])
        options['limits'] = tuple(np.nanpercentile(reference, [2, 98]))
    frames = []
    for index in results.indices[::every]:
        figure = curtain(results, upto=index, **options)
        buffer = io.BytesIO()
        figure.savefig(buffer, format='png', dpi=dpi)
        plt.close(figure)
        buffer.seek(0)
        frames.append(Image.open(buffer).convert('P', palette=Image.ADAPTIVE))
    frames[0].save(output, save_all=True, append_images=frames[1:], loop=0, duration=int(1000. / fps))
    return len(frames)
