import os
from abc import ABCMeta, abstractmethod
from pathlib import Path
        
from configparseradv import ConfigParserAdv
from logging_tools import logger
import matplotlib.pyplot as plt

from .geometry import GeometryHandler

class BasePlotter(metaclass=ABCMeta):
    """Figure axes collection.

    Keeps track of the Figure axes, whether they have been initialized or not.
    Attributes:
        defconfig (Path): default configuration path
        log (): logger
        _config (ConfigParserAdv): original configuration file
        config (ConfigParserAdv proxy): configuration current section proxy
        fig (matplotlib.Figure): figure
        figsize (list): figure size
        axes (GeometryHandler): dictionary with the axes
    """

    defconfig = Path(__file__).resolve().parent / Path('configs/default.cfg')
    log = logger.get_logger(__name__)

    def __init__(self, config: Path = None, section: str = 'DEFAULT', **kwargs):
        """Create a new base plotter"""
        # Close plt if still open
        try:
            plt.close()
        except:
            pass

        # Update options
        self._config = ConfigParserAdv()
        log.debug(f'Default config: {self.defconfig}')
        self._config.read(self.defconfig)
        if config is None:
            pass
        else:
            self._config.read(config.expanduser().resolve())
        self._config.read_dict({section: kwargs})
        self.config = self._config[section]

        # Set plot styles
        plt.style.use(self.config.get('styles').replace(',', ' ').split())

        # Get axes
        self.axes = GeometryHandler()
        self.figsize = self.axes.get_geometry(self.config)
        self.log.info(f'Figure size: w={self.figsize[0]}, h={self.figsize[1]}')

        # Create figure
        self.fig = plt.figure(figsize=self.figsize)

    def __iter__(self):
        for ax in self.axes:
            yield self.get_axis(*ax)
    
    @property
    def shape(self):
        return self.axes.nrows, self.axes.ncols

    @property
    def projection(self):
        return self.config['projection']

    @property
    def sharex(self):
        return self.axes.sharex

    @property
    def sharey(self):
        return self.axes.sharey

    def is_init(self, loc, cbaxis: bool = False):
        """Check if an axis has been initialized.

        Args:
            loc: location
            cbaxis: whether to check for a colorbar axis instead
        """
        if not cbaxis:
            return hasattr(self.axes[ij].axis, 'plot')
        else:
            return hasattr(self.axes[ij].cbaxis, 'plot')

    @abstractmethod
    def init_axis(self, loc, handler, projection: str = None, 
            include_cbar: bool = False, **kwargs):
        """Initialize axis"""
        # Check projection
        if projection is None:
            projection = self.projection

        # Verify include_cbar
        if include_cbar is None:
            include_cbar = self.has_cbar(ij)
        
        # Initialize axis if needed
        if self.is_init(ij):
            self.log.info(f'Axis {ij} already initialized')
        else:
            self.log.info(f'Initializing axis: {ij}')
            self.axes[ij].axis.scalex(1./self.figsize[0])
            self.axes[ij].axis.scaley(1./self.figsize[1])
            axis = self.fig.add_axes(
                    self.axes[ij].axis.pyplot_axis, 
                    projection=projection)
        
        # Color bar
        if include_cbar:
            self.log.info(f'Initializing color bar: {ij}')
            cbaxis = self.init_cbar(ij)
        
        # Create plotter object
        self.axes[ij] = handler(axis, cbaxis, **kwargs)

    @abstractmethod
    def auto_plot(self):
        pass

    def init_cbar(self, loc):
        """Initialize color bar"""
        if not self.has_cbar(ij) or self.is_init(loc, cbaxis=True):
            pass
        else:
            self.axes[ij].cbaxis.scalex(1./self.figsize[0])
            self.axes[ij].cbaxis.scaley(1./self.figsize[1])
            self.cbaxes[ij] = self.fig.add_axes(
                    self.axes[ij].cbaxis.pyplot_axis)

    def has_cbar(self, loc):
        """Shortcut for axes[loc].has_cbar()"""
        return self.axes[ij].has_cbar()

    def has_axlabels(self, loc) -> tuple:
        """Check if axis has label or is shared"""
        xlabel = not self.sharex or \
                (self.sharex and loc[1]==0 and loc[0]==self.shape[0]-1)
        ylabel = not self.sharey or \
                (self.sharey and loc[1]==0 and loc[0]==self.shape[0]-1)
        return xlabel, ylabel

    def has_ticks_labels(self, loc):
        """Check if axis has tick labels"""
        xticks = not self.sharex or \
                (self.sharex and loc[0]==self.shape[0]-1)
        yticks = not self.sharey or (self.sharey and loc[1]==0)
        return xticks, yticks

    def get_value(self, key: str, loc: tuple = None, **kwargs):
        """Get value from configuration
        
        See configparseradv documentation for a list of available kwargs.
        """
        # Convert location to index
        if loc is not None and loc in self.axes:
            kwargs['n'] = self.axes.keys().index(loc)

        return self.config.getvalue(key, **kwargs)

    def set_title(self, title, **kwargs):
        """Set figure title"""
        self.fig.suptitle(title, **kwargs)

    def savefig(self, fname, **kwargs):
        """Save figure"""
        self.fig.savefig(fname, **kwargs)

