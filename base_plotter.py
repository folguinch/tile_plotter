from configparser import ConfigParser
from abc import ABCMeta, abstractmethod

import matplotlib as mpl
import matplotlib.pyplot as plt

from functions import *

class BasePlotter(object):

    """Figure axes collection.

    Keeps track of the Figure axes, whether they have been initialized or not.
    Attributes:
        axes (list): list of the axes.
        cbaxes (list): list of the colorbar axes.
    """

    __metaclass__ = ABCMeta

    def __init__(self, styles=[], rows=1, cols=1, nxcbar=0, nycbar=0,
            xsize=4.5, ysize=4.5, left=1.0, right=0.15, bottom=0.6, top=0.15,
            wspace=0.2, hspace=0.2, cbar_width=0.2, cbar_spacing=0.1,
            sharex=False, sharey=False, projection='rectilinear',
            share_cbar=False):
        try:
            plt.close()
        except:
            pass
        plt.style.use(styles)

        # Projection
        self.projection = projection

        # Determine the number of axes
        self.nx = cols 
        self.nxcbar = nxcbar
        self.ny = rows 
        self.nycbar = nycbar
        
        # Determine figure size
        #if rows==nycbar or cols==nxcbar:
        #    share_cbar = False
        #else:
        #    share_cbar = True
        figgeom, cbargeom = get_geometry(rows, cols, nxcbar, nycbar, xsize, 
                ysize, left, right, bottom, top, wspace, hspace, cbar_width, 
                cbar_spacing, sharex, sharey, share_cbar=share_cbar)
        width = 0
        height = 0
        for geom in figgeom[:cols] + cbargeom[:nxcbar]:
            width += geom.width
        for geom in figgeom[::cols] + cbargeom[:nycbar]:
            height += geom.height
        print 'Figure size: width=%.1f in height=%.1f in' % (width, height)

        # Get axes
        if nxcbar!=0:
            cbar_orientation = 'vertical'
        elif nycbar!=0:
            cbar_orientation = 'horizontal'
        else:
            cbar_orientation = None
        self.axes, self.cbaxes = get_axes(figgeom, cbargeom, rows, cols, width,
                height, cbar_orientation)

        # Create figure
        self.fig = plt.figure(figsize=(width, height))

    def __iter__(self):
        for ax in self.axes:
            yield self.fig.add_axes(ax, projection=self.projection)

    def savefig(self, fname, **kwargs):
        self.fig.savefig(fname, **kwargs)

    def is_init(self, n, cbaxis=False):
        """Check if an axis has been initialized.

        Args:
            n (int): number of the axis.
            cbaxis (bool, optional): whether to check for a colorbar axis 
                instead. Default: False.
        """
        if not cbaxis:
            return hasattr(self.axes[n], 'plot')
        else:
            return hasattr(self.cbaxes[n], 'plot')

    def get_axis(self, n, projection=None, include_cbar=True):
        if projection is None:
            projection = self.projection

        axis = self.fig.add_axes(self.axes[n].axis, projection=projection)
        if not include_cbar:
            cbax = None
        elif len(self.cbaxes)==len(self.axes):
            cbax = self.fig.add_axes(self.cbaxes[n].axis)
        elif self.nxcbar and n%self.nx==self.nx-1:
            cbax = self.fig.add_axes(self.cbaxes[n/self.nx].axis)
        elif self.nycbar and n/self.nx==0:
            cbax = self.fig.add_axes(self.cbaxes[n%self.nx].axis)
        else:
            cbax = None

        return axis, cbax

    @abstractmethod
    def init_axis(self, n, projection=None, include_cbar=True):
        if projection is None:
            projection = self.projection

        if self.is_init(n):
            pass
        else:
            assert hasattr(self.axes[n], 'axis')
            self.axes[n] = self.fig.add_axes(self.axes[n].axis, 
                    projection=projection)

        if not include_cbar:
            pass
        elif len(self.cbaxes)==len(self.axes):
            self.cbaxes[n] = self.fig.add_axes(self.cbaxes[n].axis)
        elif self.nxcbar and n%self.nx==self.nx-1:
            self.cbaxes[n/self.nx] = self.fig.add_axes(self.cbaxes[n/self.nx].axis)
        elif self.nycbar and n/self.nx==0:
            self.cbaxes[n%self.nx] = self.fig.add_axes(self.cbaxes[n%self.nx].axis)
        else:
            pass

        return 

class SinglePlotter(object):

    """Container for axes of a single plot.

    Class for easy and quick configuration of a single axes object. This plot
    may be included in a grid of several plots.

    Attributes:
        ax (matplotlib axes): axes of the plot.
        xscale (str): x axis scale (linear or log).
        yscale (str): y axis scale (linear or log).
    """

    def __init__(self, ax, xscale='linear', yscale='linear'):
        """ Create a single plot container.

        Args:
            ax (matplotlib axes): axes of the plot.
            xscale (str, optional): x axis scale (linear or log, default
                linear).
            yscale (str, optional): y axis scale (linear or log, default
                linear).
        """
        self.ax = ax
        self.xscale = xscale
        self.yscale = yscale

    def config_plot(self, config=ConfigParser(), **kwargs):
        """Configure the plot.

        Configures several aspects of the axes: limits, scales, labels and
        ticks. Keyword parameters overrride the values in the config variable.
        If a parameter has no value the option is not configured.

        Args:
            config (dict like, optional): dictionary like structure with
                configuration values (default empty dict).
        Keyword args:
            xlim (float iterable): x axis limits.
            ylim (float iterable): y axis limits.
            xscale (str): x axis scale (linear or log, default initialized
                value).
            yscale (str): y axis scale (linear or log, default initialized
                value).
            xlabel (str): x axis label (default '').
            ylabel (str): y axis label (default '').
            ticks_fmt (str): ticks string format (default %.3f).
            xticks (float list like): x ticks positions. The ticks labels are
                formated with xticks_fmt.
            yticks (float list like): y ticks positions. The ticks labels are
                formated with yticks_fmt.
            unset_xticks (bool): whether to unset the x ticks labels (default
                False).
            unset_yticks (bool): whether to unset the y ticks labels (default
                False.
        """
        # Limits
        if 'xlim' in kwargs or 'xlim' in config:
            xlim = kwargs.get('xlim') or map(float,config.get('xlim').split(','))
            self.ax.set_xlim(*xlim)
        if 'ylim' in kwargs or 'ylim' in config:
            ylim = kwargs.get('ylim') or map(float,config.get('ylim').split(','))
            self.ax.set_ylim(*ylim)

        # Coordinate scales
        if kwargs.get('xscale', self.xscale)=='log':
            self.ax.set_xscale('log')
            self.ax.xaxis.set_major_formatter(formatter('log'))
        if kwargs.get('yscale', self.yscale)=='log':
            self.ax.set_yscale('log')
            self.ax.yaxis.set_major_formatter(formatter('log'))

        # Labels
        self.ax.set_xlabel(kwargs.get('xlabel', ''))
        self.ax.set_ylabel(kwargs.get('ylabel', ''))

        # Ticks
        fmt = kwargs.setdefault('ticks_fmt', '%.3f')
        if 'xticks' in kwargs:
            self.ax.set_xticks(kwargs['xticks'])
            self.ax.set_xticklabels([fmt % nu for nu in kwargs['xticks']])
        elif kwargs.get('unset_xticks', False):
            self.ax.set_xticklabels(['']*len(self.ax.get_xticks()))
        if 'minor_xticks' in kwargs:
            self.ax.set_xticks(kwargs['minor_xticks'], minor=True)
        if 'yticks' in kwargs:
            self.ax.set_yticks(kwargs['yticks'])
            self.ax.set_yticklabels([fmt % nu for nu in kwargs['yticks']])
        elif kwargs.get('unset_yticks', False):
            self.ax.set_yticklabels(['']*len(self.ax.get_yticks()))
        if 'minor_yticks' in kwargs:
            self.ax.set_yticks(kwargs['minor_yticks'], minor=True)

    def set_xlim(self, xmin=None, xmax=None):
        """Set the axis x limits"""
        self.ax.set_xlim(left=xmin, right=xmax)

    def set_ylim(self, ymin=None, ymax=None):
        """Set the axis y limits"""
        self.ax.set_ylim(bottom=ymin, top=ymax)

    def plot(self, *args, **kwargs):
        """Plot on the axis.

        Arguments are the same as for the matplotlib.pyplot.plot() finction.

        Args:
            *args: data to plot.
            **kwargs: arguments for matplotlib.pyplot.plot().
        """
        return self.ax.plot(*args, **kwargs)

    def axhline(self, *args, **kwargs):
        """Plot horizontal line

        Arguments are the same as for the matplotlib.pyplot.axhline function
        """
        self.ax.axhline(*args,**kwargs)

    def errorbar(self, *args, **kwargs):
        """Plot on the axis.

        Arguments are the same as for the matplotlib.pyplot.errorbar() finction.

        Args:
            *args: data to plot.
            **kwargs: arguments for matplotlib.pyplot.errorbar().
        """
        self.ax.errorbar(*args, **kwargs)
    
    def annotate(self, *args, **kwargs):
        """Annotate the axis.

        Arguments are the same as for the matplotlib.pyplot.annotate() finction.

        Args:
            *args: data to plot.
            **kwargs: arguments for matplotlib.pyplot.annotate().
        """
        self.ax.annotate(*args, **kwargs)

    def legend(self, loc=0, **kwargs):
        """Plot the legend.

        Args:
            loc (int, optional): legend position. 
                See matplotlib.pyplot.lengend() documentation for available
                values and positions (default 0, i.e. best location).
        """
        self.ax.legend(loc=loc, frameon=False, **kwargs)

    def scatter(self, *args, **kwargs):
        return self.ax.scatter(*args, **kwargs)

    def clabel(self, *args, **kwargs):
        self.ax.clabel(*args, **kwargs)

    def contour(self, *args, **kwargs):
        return self.ax.contour(*args, **kwargs)

    def tricontour(self, x, y, *args, **kwargs):
        triangulation = mpl.tri.Triangulation(x, y)
        return self.ax.tricontour(triangulation, *args, **kwargs)
