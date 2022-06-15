#!/usr/bin/python3
"""Data plotting program.

Implements the tile plotting tools to plot data from the command line.
"""
from typing import Sequence
import argparse
import sys

from toolkit.argparse_tools import actions, parents

from tile_plotter.multi_plotter import MultiPlotter

#from loaders import *
#from parsers import global_parser
## Here import all plotting argparsers in the directory
#from plot_channel_maps import plot_channel_maps_parser
#from plot_data import plot_data_parser
#from plot_from_cube import plot_from_cube_parser
#from plot_maps import plot_maps_parser
#from plot_moments import plot_moments_parser
#from plot_multi import plot_multi_parser
#from plot_pvmap import plot_pvmaps_parser

def multiplot(args):
    plot = MultiPlotter(args.config[0])
    plot.plot_all()
    plot.savefig(args.plotname[0])

def main(args: Sequence):
    """Main program.

    Args:
      args: list of command line inputs.
    """
    # Evaluate parser functions
    #subpar = {}
    #subpar.update(plot_channel_maps_parser())
    #subpar.update(plot_maps_parser())
    #subpar.update(plot_moments_parser())
    #subpar.update(plot_from_cube_parser())
    #subpar.update(plot_data_parser())
    #subpar.update(plot_pvmaps_parser())
    #subpar.update(plot_multi_parser())

    # Command line options
    args_parents = [
        parents.logger('debug_plotter.log'),
        parents.verify_files('config',
                             config={'help': 'Configuration file name',
                                     'nargs': 1})
    ]
    parser = argparse.ArgumentParser(
        description='Data plotting tools.',
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=args_parents,
        conflict_handler='resolve',
    )
    parser.add_argument('plotname', nargs=1, action=actions.NormalizePath,
                        help='Plot file name')

    # Processing pipe
    pipe = [multiplot]
    # Subparsers
    #subparsers = parser.add_subparsers()
    # Add subparsers
    #for key,(p,h) in subpar.items():
    #    subparser = subparsers.add_parser(key, parents=[p], help=h)

    # Process arguments
    args = parser.parse_args(args)
    for step in pipe:
        step(args)

    # Parse arguments
    #args.logger(__name__, args)
    #for loader in args.loaders:
    #    args = loader(args)
    #args = load_overplot(args)
    #figure = args.func(args)
    #if figure is not None:
    #    args.post(figure, args)

if __name__=='__main__':
    main(sys.argv[1:])
