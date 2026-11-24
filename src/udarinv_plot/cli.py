"""Command line: ``udarinv-plot curtain | sections | residuals | signals | progress | animate`` RUN."""

import argparse
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog='udarinv-plot', description=__doc__)
    parser.add_argument('command', choices=['curtain', 'sections', 'residuals', 'signals', 'progress', 'animate'])
    parser.add_argument('run', type=Path, help='output folder of a chain')
    parser.add_argument('-o', '--output', type=Path, help='file to write (default: in the run folder)')
    parser.add_argument('--index', type=int, help='sections: logging point (default: the last finished one)')
    parser.add_argument('--tool', type=int, default=0, help='signals: tool number, in tool.inp order')
    parser.add_argument('--no-truth', action='store_true', help='do not draw the benchmark model')
    parser.add_argument('--no-wash', action='store_true', help='curtain: no fading with the posterior std')
    parser.add_argument('--linear', action='store_true', help='linear colour scale (default: logarithmic)')
    parser.add_argument('--limits', type=float, nargs=2, metavar=('LOW', 'HIGH'), help='colour limits [ohm m]')
    parser.add_argument('--every', type=int, default=1, help='animate: use every n-th logging point')
    parser.add_argument('--fps', type=float, default=4.)
    parser.add_argument('--dpi', type=int, default=150)
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .results import Results
    results = Results(args.run, truth=not args.no_truth)
    style = {'limits': args.limits, 'linear': args.linear, 'wash': not args.no_wash}
    if args.command == 'animate':
        output = args.output or args.run / 'curtain.gif'
        from .animate import animate
        frames = animate(results, output, every=args.every, fps=args.fps, dpi=min(args.dpi, 100), **style)
        print(f'Wrote {output} ({frames} frames)')
        return 0
    if args.command == 'curtain':
        from .curtain import curtain
        figure = curtain(results, **style)
    elif args.command == 'sections':
        from .sections import sections
        index = results.indices[-1] if args.index is None else args.index
        if index not in results.indices:
            parser.error(f'point {index} is not finished')
        figure, args.command = sections(results, index, args.limits), f'sections-{index:04d}'
    elif args.command == 'residuals':
        from .datafit import residuals
        figure = residuals(results)
    elif args.command == 'signals':
        from .datafit import signals
        figure, args.command = signals(results, args.tool), f'signals-tool{args.tool}'
    else:
        from .progress import progress
        figure = progress(results)
    output = args.output or args.run / f'{args.command}.png'
    figure.savefig(output, dpi=args.dpi)
    plt.close(figure)
    print(f'Wrote {output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
