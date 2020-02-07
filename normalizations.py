import astropy.visualization as visualization
import matplotlib.colors as colors
import numpy as np

# Based on https://matplotlib.org/tutorials/colors/colormapnorms.html
# Replace in future versions with DivergingNorm
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=0., clip=True):
        self.vcenter = vcenter
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # Masked data
        if isinstance(value, np.ma.MaskedArray):
            mask = value.mask
            value = value.filled(self.vmax)
        else:
            mask=False
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), mask=mask)

#class MidpointStretch(visualization.)
