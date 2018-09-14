from itertools import product

import numpy as np
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from .base_plotter import BasePlotter, SinglePlotter
from myutils.fits import get_ref

class FieldPlotter(SinglePlotter):
    q = None

    def plot_field(self, x, y, vx, vy, scale=10000., scale_units='dots'):
        self.q = self.ax.quiver(x, y, vx, vy, scale_units=scale_units, scale=scale)
        self.ax.set_aspect('equal', 'datalim')

    def set_key(self, x, y, length, label, **kwargs):
        self.ax.quiverkey(self.q, x, y, length, label, **kwargs)



class FieldsPlotter(BasePlotter):

    def __init__(self, styles=[], rows=1, cols=1, xsize=5, ysize=5, left=.7, 
            right=0.15, bottom=0.6, top=0.15, wspace=0.2, hspace=0.2, 
            sharex=False, sharey=False):

        super(FieldsPlotter, self).__init__(styles=styles, rows=rows, cols=cols, 
                xsize=xsize, ysize=ysize, left=left, right=right, bottom=bottom,
                top=top, wspace=wspace, hspace=hspace, sharex=sharex,
                sharey=sharey)
        
    def get_plotter(self, n):
        axis, cbax = self.get_axis(n, include_cbar=False)

        return FieldPlotter(axis)

    def init_axis(self, n):
        super(FieldsPlotter, self).init_axis(n)
