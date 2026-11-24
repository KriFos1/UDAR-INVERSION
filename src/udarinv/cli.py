"""Command line: ``udarinv run | check-jacobian | summarize``."""

import argparse
import json
from pathlib import Path
import sys


def _run(args):
    from . import chain, config
    settings = config.load(args.config, start=args.start, stop=args.stop, seconds=args.seconds, output=args.output)
    return chain.run(settings, log=lambda message: print(message, flush=True))


def _check_jacobian(args):
    from . import check, config
    settings = config.load(args.config)
    rows = check.jacobian(settings, args.index, args.tolerance, args.step, log=lambda m: print(m, flush=True))
    worst = max(row['relative_error'] for row in rows)
    print(f'largest relative error {worst:.4f}')
    return 0 if worst < args.accept else 1


def _summarize(args):
    from . import scoring
    summary = scoring.summarize(Path(args.run), truth=not args.no_truth)
    if args.json:
        print(json.dumps(summary, indent=1))
    else:
        print(scoring.format_summary(summary))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog='udarinv', description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run', help='run or continue a chain')
    run.add_argument('config', type=Path)
    run.add_argument('--start', type=int, help='first logging point (overrides the config)')
    run.add_argument('--stop', type=int, help='one past the last logging point (overrides the config)')
    run.add_argument('--seconds', type=float, help='stop between points after this long; exit code 75')
    run.add_argument('--output', help='output folder (overrides the config)')
    run.set_defaults(function=_run)
    check = commands.add_parser('check-jacobian', help='adjoint Jacobian against central differences at one point')
    check.add_argument('config', type=Path)
    check.add_argument('--index', type=int, required=True)
    check.add_argument('--tolerance', type=float, default=1e-6, help='solver tolerance of every solve')
    check.add_argument('--step', type=float, default=.02, help='finite-difference step in the log fields')
    check.add_argument('--accept', type=float, default=.02, help='largest relative error that passes')
    check.set_defaults(function=_check_jacobian)
    summarize = commands.add_parser('summarize', help='fit, starts and (benchmarks) error of a finished chain')
    summarize.add_argument('run', type=Path)
    summarize.add_argument('--no-truth', action='store_true', help='do not score against the benchmark model')
    summarize.add_argument('--json', action='store_true')
    summarize.set_defaults(function=_summarize)
    args = parser.parse_args(argv)
    return args.function(args)


if __name__ == '__main__':
    sys.exit(main())
