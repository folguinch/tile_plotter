from itertools import product

import numpy as np
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from .base_plotter import BasePlotter, SinglePlotter
from myutils.fits import get_ref
from myutils.math import center_of_mass

def get_lines_props(colors=['r', 'g', 'b', 'm', 'c'], lines=['-', '--', ':', '-.']):
    return product(colors, lines)

class SlicePlotter(SinglePlotter):

    def plot_hvslices(self, *data, **kwargs):
        # Lines
	lntypes = kwargs.get('lntypes', None)
        if lntypes is None:
            lntypes = get_lines_props()

        # Labels
        labels = kwargs.get('labels', ['Data %i' % i for i in range(len(data))])

        for img, lb, ln in zip(data,labels,lntypes):
            ra, dec = kwargs.get('position', (None,None))
            x0, y0 = get_ref(img, kwargs.get('ref_pos', 'max'), ra=ra, dec=dec)

            if kwargs.setdefault('direction','horizontal') == 'horizontal':
                x = np.arange(img.data.shape[1]) - x0 + 0.5
                y = img.data[int(y0)]
            else:
                x = np.arange(img.data.shape[0]) - y0 + 0.5
                y = img.data[:,int(x0)]

            if kwargs.get('recenter', False):
                aux = ~np.isnan(y)
                if kwargs.get('recenter_lim'):
                    aux = aux & (x>kwargs['recenter_lim'][0]) & \
                            (x<kwargs['recenter_lim'][1])
                shift = center_of_mass(x[aux],y[aux])
                x = x - shift

            wcs = WCS(img.header).sub(['latitude','longitude'])
            pixsize = np.mean(np.abs(proj_plane_pixel_scales(wcs)))

            #pixsize = np.sqrt(np.abs(img.header['CDELT1']*img.header['CDELT2']))
            x = x*pixsize*3600.

            super(SlicePlotter, self).plot(x, y, ''.join(ln), label=lb)

    def plot_slices(self, *slices, **kwargs):
        # Lines
	lntypes = kwargs.get('lntypes', None)
        if lntypes is None:
            lntypes = get_lines_props()

        # Labels
        labels = kwargs.get('labels', ['Data %i' % i for i in range(len(slices))])

        for slc, lb, ln in zip(slices,labels,lntypes):
            nvals = 2
            if hasattr(slc, 'dtype'):
                x = slc[slc.dtype.names[0]] * kwargs.get('xfactor',1.)
                y = slc[slc.dtype.names[1]]
                try:
                    yerr = slc[slc.dtype.names[2]]
                except IndexError:
                    yerr = None
            else:
                x, y  = slc[0] * kwargs.get('xfactor',1.), slc[1]
                try:
                    yerr = slc[2]
                except IndexError:
                    yerr = None

            if yerr is None:
                super(SlicePlotter, self).plot(x, y, ''.join(ln), label=lb)
            else:
                super(SlicePlotter, self).errorbar(x, y, yerr=yerr,
                        fmt=kwargs.get('ptype',ln[1]), ecolor=ln[0], color=ln[0])


class SlicesPlotter(BasePlotter):

    def __init__(self, styles=[], rows=1, cols=1, xsize=5, ysize=3, left=.7, 
            right=0.15, bottom=0.6, top=0.15, wspace=0.2, hspace=0.2, 
            sharex=False, sharey=False):

        super(SlicesPlotter, self).__init__(styles=styles, rows=rows, cols=cols, 
                xsize=xsize, ysize=ysize, left=left, right=right, bottom=bottom,
                top=top, wspace=wspace, hspace=hspace, sharex=sharex,
                sharey=sharey)

    def get_axis(self, nax=0):
        axis, cbaxis = super(SlicesPlotter,self).get_axis(nax, include_cbar=False)
        return SlicePlotter(axis)

    def init_axis(self, n):
        super(SlicesPlotter, self).init_axis(n)
