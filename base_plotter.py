import os
from configparser import ConfigParser
from abc import ABCMeta, abstractmethod
from builtins import map, range
        
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
    defconfig = os.path.join(os.path.realpath(__file__), 'configs/default.cfg')

    def __init__(self, config=None, section='single', **kwargs):
        # Close plt if still open
        try:
            plt.close()
        except:
            pass

        # Update options
        opts = ConfigParser()
        opts.read(self.defconfig)
        if config is None:
            pass
        elif os.path.isfile(os.path.expanduser(config)):
            opts.read(config)
        else:
            raise IOError('File %s does not exist' % config)
        opts.read_dict({section: kwargs})
        self.config = opts[section]

        # Set plot styles
        plt.style.use(opts.get(section, 'styles').replace(',',' ').split())

        # Get axes
        self.figsize, self.axes, self.cbaxes = get_geometry(opts, section=section)
        print('Figure size: width=%.1f in height=%.1f in' % self.figsize)

        # Create figure
        self.fig = plt.figure(figsize=self.figsize)

    def __iter__(self):
        for ax in self.axes:
            yield self.get_axis(*ax)
    
    @property
    def shape(self):
        return self.config.getint('nrows'), self.config.getint('ncols')

    @property
    def projection(self):
        return self.config['projection']

    @property
    def sharex(self):
        return self.config.getboolean('sharex')

    @property
    def sharey(self):
        return self.config.getboolean('sharey')

    def _get_loc(self, loc):
        # Compatible with older versions
        try:
            row, col = loc
        except TypeError:
            row, col = self.axes.keys()[loc]
        return row, col 

    @abstractmethod
    def init_axis(self, loc, projection=None, include_cbar=None):
        ij = self._get_loc(loc)
        if projection is None:
            projection = self.projection

        if self.is_init(ij):
            pass
        else:
            self.axes[ij].scalex(1./self.figsize[0])
            self.axes[ij].scaley(1./self.figsize[1])
            self.axes[ij] = self.fig.add_axes(self.axes[ij].axis, 
                    projection=projection)

        if include_cbar:
            self.init_cbar(ij)

        return 

    def init_cbar(self, loc):
        ij = self._get_loc(loc)
        
        if self.cbaxes[ij].is_empty() or self.is_init(loc, cbaxis=True):
            pass
        else:
            self.cbaxes[ij].scalex(1./self.figsize[0])
            self.cbaxes[ij].scaley(1./self.figsize[1])
            self.cbaxes[ij] = self.fig.add_axes(self.cbaxes[ij].axis)

    def is_init(self, loc, cbaxis=False):
        """Check if an axis has been initialized.

        Args:
            row (int): row of the axis.
            col (int): column of the axis.
            cbaxis (bool, optional): whether to check for a colorbar axis 
                instead. Default: False.
        """
        ij = self._get_loc(loc)
        if not cbaxis:
            return hasattr(self.axes[ij], 'plot')
        else:
            return hasattr(self.cbaxes[ij], 'plot')

    def get_axis(self, loc, projection=None, include_cbar=None):
        # Set projection
        if projection is None:
            projection = self.projection

        # Convert to row-col if needed
        ij = self._get_loc(loc)

        # Verify include_cbar
        if include_cbar is None:
            if self.config.getboolean('vcbar'):
                vcbarpos = list(map(int, 
                    self.config['vcbarpos'].replace(',',' ').split()))
                include_cbar = ij[1] in vcbarpos or \
                        ij[1]-self.shape[1] in vcbarpos
            elif self.config.getboolean('hcbar'):
                hcbarpos = list(map(int, 
                    self.config['hcbarpos'].replace(',',' ').split()))
                include_cbar = ij[0] in hcbarpos or \
                        ij[0]-self.shape[0] in hcbarpos
            else:
                include_cbar = False

        # Initialize axis if needed
        if not self.is_init(ij):
            self.init_axis(ij, projection=projection,
                    include_cbar=include_cbar)

        return self.axes[ij], self.cbaxes[ij]

    def get_value(self, key, default=None, ax=None, n=None, sep=' '):
        if ax is not None and ax in self.axes:
            n = self.axes.keys().index(ax)

        if n and (key + str(n)) in self.config:
            newkey = key + str(n)
            value = self.config[newkey]
        elif key in self.config:
            value = self.config[key]
            if n:
                if sep!=' ':
                    value = value.split(sep)
                else:
                    # for back compatibility
                    value = value.replace(',',' ').split()
                if len(value)==1:
                    value = value[0]
                else:
                    try:
                        value = value[n]
                    except IndexError:
                        print('WARNING: %s does not exist in config, using default'\
                        % key)
                        value = default
        else:
            value = default

        return value

    def savefig(self, fname, **kwargs):
        self.fig.savefig(fname, **kwargs)

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
