from configparseradv import ConfigParserAdv
from abc import ABCMeta, abstractmethod

import matplotlib as mpl
import matplotlib.pyplot as plt

from .functions import get_geometry

class BasePlotter(metaclass=ABCMeta):
    """Figure axes collection.

    Keeps track of the Figure axes, whether they have been initialized or not.
    Attributes:
        axes (list): list of the axes.
        cbaxes (list): list of the colorbar axes.
    """

    def __init__(self, styles: list = [], rows: int = 1, cols: int = 1, 
            nxcbar: int = 0, nycbar: int = 0,  xsize: float = 4.5, 
            ysize: float = 4.5, left: float = 1.0, right: float = 0.15, 
            bottom: float = 0.6, top: float = 0.15, wspace: float = 0.2, 
            hspace: float = 0.2, cbar_width: float = 0.2, 
            cbar_spacing: float = 0.1, sharex: bool = False, 
            sharey: bool = False, projection: str = 'rectilinear',
            share_cbar: bool = False):
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
        cbax (matplotlib colorbar): color bar axis.
        xscale (str): x axis scale (linear or log).
        yscale (str): y axis scale (linear or log).
    """

    def __init__(self, ax, cbaxis=None, xscale='linear', yscale='linear'):
        """ Create a single plot container.

        Args:
            ax (matplotlib axes): axes of the plot.
            cbaxis (matplotlib colorbar): color bar axis.
            xscale (str, optional): x axis scale (linear or log, default
                linear).
            yscale (str, optional): y axis scale (linear or log, default
                linear).
        """
        self.ax = ax
        self.cbax = cbaxis
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
        return self.ax.errorbar(*args, **kwargs)
    
    def annotate(self, *args, **kwargs):
        """Annotate the axis.

        Arguments are the same as for the matplotlib.pyplot.annotate() finction.

        Args:
            *args: data to plot.
            **kwargs: arguments for matplotlib.pyplot.annotate().
        """
        self.ax.annotate(*args, **kwargs)

    def legend(self, handles=None, labels=None, loc=0, **kwargs):
        """Plot the legend.

        Args:
            loc (int, optional): legend position. 
                See matplotlib.pyplot.lengend() documentation for available
                values and positions (default 0, i.e. best location).
        """
        if handles and labels:
            self.ax.legend(handles, labels, loc=loc, frameon=False, **kwargs)
        else:
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

    def plot_cbar(self, fig, cs, label=None, ticks=None, ticklabels=None, 
            orientation='vertical', labelpad=10, lines=None):
        # Ticks
        #if ticks is None:
        #    ticks = get_ticks(self.vmin, self.vmax, self.a, stretch=self.stretch)

        # Create bar
        cbar = fig.colorbar(cs, ax=self.ax, cax=self.cbax,
                    orientation=orientation, drawedges=False)
        if lines is not None:
            cbar.add_lines(lines)
            # Tick Font properties
            #print cbar.ax.get_yticklabels()[-1].get_fontsize()
            #for label in cbax.get_xticklabels()+cbax.get_yticklabels():
            #    label.set_fontsize(ax.xaxis.get_majorticklabels()[0].get_fontsize())
            #    label.set_family(ax.xaxis.get_majorticklabels()[0].get_family())
            #    label.set_fontname(ax.xaxis.get_majorticklabels()[0].get_fontname())

        # Bar position
        if orientation=='vertical':
            cbar.ax.yaxis.set_ticks_position('right')
            if ticklabels is not None:
                cbar.ax.yaxis.set_ticklabels(ticklabels)
        elif orientation=='horizontal':
            cbar.ax.xaxis.set_ticks_position('top')
            if ticklabels is not None:
                cbar.ax.xaxis.set_ticklabels(ticklabels)

        # Label
        if label is not None:
            if orientation=='vertical':
                cbar.ax.yaxis.set_label_position('right')
                cbar.set_label(label,
                        fontsize=self.ax.yaxis.get_label().get_fontsize(),
                        family=self.ax.yaxis.get_label().get_family(),
                        fontname=self.ax.yaxis.get_label().get_fontname(),
                        weight=self.ax.yaxis.get_label().get_weight(),
                        labelpad=labelpad, verticalalignment='top')
            elif orientation=='horizontal':
                cbar.ax.xaxis.set_label_position('top')
                cbar.set_label(label,
                        fontsize=self.ax.xaxis.get_label().get_fontsize(),
                        family=self.ax.xaxis.get_label().get_family(),
                        fontname=self.ax.xaxis.get_label().get_fontname(),
                        weight=self.ax.xaxis.get_label().get_weight())
        for tlabel in cbar.ax.xaxis.get_ticklabels(which='both')+cbar.ax.yaxis.get_ticklabels(which='both'):
            tlabel.set_fontsize(self.ax.xaxis.get_majorticklabels()[0].get_fontsize())
            tlabel.set_family(self.ax.xaxis.get_majorticklabels()[0].get_family())
            tlabel.set_fontname(self.ax.xaxis.get_majorticklabels()[0].get_fontname())
