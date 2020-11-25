import copy as cp
from itertools import product
from builtins import map, range
from collections import OrderedDict

import numpy as np

from .geometry import FigGeometry

log = logger.get_logger(__name__)

def get_geometry(opts, section='DEFAULT'):
    # Key list
    int_keys = ['nrows', 'ncols']
    margin_keys = ['left', 'right', 'top', 'bottom']
    space_keys = ['xsize', 'ysize', 'hspace', 'vspace']

    # Geometry
    ncols = opts[section].getint('ncols')
    nrows = opts[section].getint('nrows')
    left = opts[section].getfloat('left')
    right = opts[section].getfloat('right')
    top = opts[section].getfloat('top')
    bottom = opts[section].getfloat('bottom')
    xsize = opts[section].getfloat('xsize')
    ysize = opts[section].getfloat('ysize')
    hspace = opts[section].getfloat('hspace')
    vspace = opts[section].getfloat('vspace')
    sharex = opts[section].getboolean('sharex')
    sharey = opts[section].getboolean('sharey')
    # Axes ratios
    xratio = opts[section].get('xsize_ratio')
    yratio = opts[section].get('ysize_ratio')
    xratio_col = opts[section].get('xsize_ratio_col')
    yratio_row = opts[section].get('ysize_ratio_row')
    # Colobar
    vcbar = opts[section].getboolean('vcbar')
    hcbar = opts[section].getboolean('hcbar')
    cbar_width = opts[section].getfloat('cbar_width')
    cbar_spacing = opts[section].getfloat('cbar_spacing')
    vcbarpos = list(map(int, 
        opts[section]['vcbarpos'].replace(',',' ').split()))
    hcbarpos = list(map(int, 
        opts[section]['hcbarpos'].replace(',',' ').split()))
    # Validate options
    if ncols<=0 or nrows<=0:
        raise ValueError('ncols and nrows must be > 0')

    # Validate ratios
    xratio = get_ratio(xratio, ncols, 'x')
    yratio = get_ratio(yratio, nrows, 'y')
    xratio_col = get_ratio(xratio_col, nrows, 'x', shape=(nrows,ncols))
    yratio_row = get_ratio(yratio_row, ncols, 'y', shape=(nrows,ncols))

    # Validate cbar
    # only allow cbar in 1 axis
    if vcbar and hcbar:
        raise ValueError('Colorbars not allowed in both axes')

    # Basic geometries
    empty = FigGeometry(0, 0)
    general = FigGeometry(xsize, ysize, left=left, right=right, bottom=bottom,
            top=top)

    # Determine geometry
    cumx, cumy = 0, 0
    xdim = None
    axes = {}
    cbaxes = {}
    for i,j in product(range(nrows)[::-1], range(ncols)):
        
        # Geometry
        axis = cp.copy(general)
        cbax = cp.copy(empty)

        # Multiply by ratios
        if hasattr(xratio_col[0], 'index'):
            aux_xratio_col = xratio_col[j]
        else:
            aux_xratio_col = xratio_col
        if hasattr(yratio_row[0], 'index'):
            aux_yratio_row = yratio_row[i]
        else:
            aux_yratio_row = yratio_row
        axis.xsize = axis.xsize * xratio[j] * aux_xratio_col[i]
        axis.ysize = axis.ysize * yratio[i] * aux_yratio_row[j]

        # Recenter
        if aux_xratio_col[i] != max(aux_xratio_col):
            # Search the maximum
            xsizemax = general.xsize * xratio[j] * max(aux_xratio_col)
            dleft = abs(axis.xsize - xsizemax) / 2.
        else:
            dleft = 0
        if aux_yratio_row[j] != max(aux_yratio_row):
            # Search the maximum
            ysizemax = general.ysize * yratio[i] * aux_yratio_row[j]
            dbottom = abs(axis.ysize - ysizemax) / 2.
        else:
            dbottom = 0

        # Left and bottom borders
        if i==nrows-1:
            axis.bottom = axis.bottom + dbottom
        elif not sharex:
            axis.bottom = bottom + vspace + dbottom
        else:
            axis.bottom = vspace
        if j==0:
            axis.left = axis.left + dleft
        elif not sharey:
            axis.left = left + hspace + dleft
        else:
            axis.left = hspace

        # Top and right
        if i!=0 and sharex:
            axis.top = 0
        if j!=ncols-1 and sharey:
            axis.right = 0

        # Color bar
        vcbar_width = 0
        hcbar_height = 0
        if vcbar and ( j in vcbarpos or j-ncols in vcbarpos ):
            # Bar geometry
            # this superceeds sharey
            cbax.xsize = cbar_width
            cbax.ysize = axis.ysize
            cbax.left = cbar_spacing
            cbax.right = right
            cbax.bottom = axis.bottom
            cbax.top = axis.top
            vcbar_width = cbax.width

            # Set axis right
            axis.right = 0
            cbax.location = [cumx+axis.width, cumy]
        elif hcbar and ( i in hcbarpos or i-nrows in hcbarpos ):
            # Bar geometry
            # this superceeds sharex
            cbax.xsize = axis.xsize
            cbax.ysize = cbar_width
            cbax.left = axis.left
            cbax.right = axis.right
            cbax.bottom = cbar_spacing
            cbax.top = top
            hcbar_height = cbax.height

            # Set axis top
            axis.top = 0
            cbax.location = [cumx, cumy+axis.height]

        # Set location
        axis.location = [cumx, cumy]
        
        # Cumulative sums
        if j==ncols-1:
            if xdim is None:
                xdim = cumx + axis.width + vcbar_width
            cumx = 0
            if i!=0:
                cumy += axis.height + vspace + hcbar_height
            else:
                ydim = cumy + axis.height + hcbar_height
        else:
            cumx += axis.width + dleft + hspace + vcbar_width

        axes[(i,j)] = cp.copy(axis)
        cbaxes[(i,j)] = cp.copy(cbax)

    return (xdim, ydim), OrderedDict(sorted(axes.items())), OrderedDict(sorted(cbaxes.items()))

def get_ratio(val, length, axis, shape=None):
    ratio = list(map(float, val.split()))
    if shape is not None and len(shape)==2:
        size = shape[0]*shape[1]
        if length not in shape:
            raise ValueError('Length not in shape')
    else:
        size = None
    if len(ratio) == 1:
        if ratio[0] != 1:
            print('WARNING: using ratio to change %ssize' % axis)
        if length>1:
            ratio = ratio * length
    elif size and len(ratio)==size:
        aux = []
        for i in range(size):
            if i%length == 0:
                aux += [[]]
            aux[-1] += [ratio[i]]
        ratio = aux
    elif len(ratio)!=length:
        raise ValueError('ratio values must have %i values' % axis)
    else:
        pass
    return ratio

def get_ticks(vmin, vmax, a=1000, n=5, stretch='log'):
    if stretch=='log':
        x = lambda y: (10.**(y*np.log10(a+1.))-1.)/a
        y = np.linspace(0.,1.,n)
        x = x(y)
        return x*(vmax-vmin) + vmin
    else:
        return np.linspace(vmin, vmax, n)

def map_to_dict(x, sep=' '):
    aux = x.split(sep)
    res = {}
    for item in aux:
        key, val = item.split(':')
        try:
            res[key] = float(val)
        except ValueError:
            res[key] = val
    return res

