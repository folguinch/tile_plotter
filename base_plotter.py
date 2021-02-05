from typing import Any, Optional, Tuple, TypeVar
import abc
import os
import pathlib
        
from logging_tools import logger
import configparseradv as cfgparser
import matplotlib.pyplot as plt

import geometry

# Type aliases
Location = Tuple[int, int]
Path = TypeVar('Path', pathlib.Path, str)

class BasePlotter(metaclass=abc.ABCMeta):
    """Figure axes collection base class.

    Keeps track of the Figure axes, whether they have been initialized or not.

    Attributes:
      config: configuration parser proxy of the current section.
      fig: matplotlib figure.
      figsize: figure size.
      axes: the figure axes.
    """

    _defconfig = (pathlib.Path(__file__).resolve().parent / 
                 pathlib.Path('configs/default.cfg'))
    _log = logger.get_logger(__name__)

    def __init__(self, config: Path = None, section: str = 'DEFAULT', **kwargs):
        """Create a new base plotter."""
        # Close plt if still open
        try:
            plt.close()
        except:
            pass

        # Update options
        self._config = cfgparser.ConfigParserAdv()
        self._log.debug(f'Default config: {self._defconfig}')
        self._config.read(self._defconfig)
        if config is None:
            pass
        else:
            self._config.read(config.expanduser().resolve())
        self._config.read_dict({section: kwargs})
        self.config = self._config[section]

        # Set plot styles
        plt.style.use(self.config.get('styles').replace(',', ' ').split())

        # Get axes
        self.axes = geometry.GeometryHandler()
        self.figsize = self.axes.get_geometry(self.config)
        self._log.info(f'Figure size: w={self.figsize[0]}, h={self.figsize[1]}')

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

    def is_init(self, loc: Location, cbaxis: bool = False) -> bool:
        """Check if an axis has been initialized.

        Args:
          loc: axis location.
          cbaxis: whether to check for a colorbar axis instead.
        """
        if not cbaxis:
            return hasattr(self.axes[ij].axis, 'plot')
        else:
            return hasattr(self.axes[ij].cbaxis, 'plot')

    @abc.abstractmethod
    def init_axis(self, 
                  loc: Location, 
                  handler: 'PlotHandler', 
                  projection: Optional[str] = None,
                  include_cbar: Optional[bool] = None, 
                  **kwargs) -> None:
        """Initialize axis by assigning a plot handler.

        If the axis is not initialized, it replaces the axis geometry
        (FigGeometry) with a plot handler.
        
        Args:
          loc: axis location.
          handler: plot handler to replace the geometry.
          projection: optional; update axis projection.
          include_cbar: optional; initialize the color bar.
          **kwargs: optional arguments for the handler.
        """
        # Check projection
        if projection is None:
            projection = self.projection

        # Verify include_cbar
        if include_cbar is None:
            include_cbar = self.has_cbar(loc)
        
        # Initialize axis if needed
        if self.is_init(ij):
            self._log.info(f'Axis {loc} already initialized')
        else:
            self._log.info(f'Initializing axis: {loc}')
            self.axes[loc].axis.scalex(1./self.figsize[0])
            self.axes[loc].axis.scaley(1./self.figsize[1])
            axis = self.fig.add_axes(self.axes[loc].axis.pyplot_axis, 
                                     projection=projection)
        
        # Color bar
        if include_cbar:
            self._log.info(f'Initializing color bar: {loc}')
            cbaxis = self.init_cbar(loc)
        
        # Create plotter object
        self.axes[loc] = handler(axis, cbaxis, **kwargs)

    @abc.abstractmethod
    def auto_plot(self):
        pass

    def init_cbar(self, loc: Location) -> None:
        """Initialize color bar."""
        if not self.has_cbar(loc) or self.is_init(loc, cbaxis=True):
            pass
        else:
            self.axes[loc].cbaxis.scalex(1./self.figsize[0])
            self.axes[loc].cbaxis.scaley(1./self.figsize[1])
            self.cbaxes[loc] = self.fig.add_axes(
                self.axes[loc].cbaxis.pyplot_axis)

    def has_cbar(self, loc: Location) -> bool:
        """Shortcut for axes[loc].has_cbar()."""
        return self.axes[loc].has_cbar()

    def has_axlabels(self, loc: Location) -> Tuple[bool, bool]:
        """Check if axis has label or is shared."""
        xlabel = (not self.sharex or
                  (self.sharex and loc[1] == 0 and loc[0] == self.shape[0]-1))
        ylabel = (not self.sharey or
                  (self.sharey and loc[1] == 0 and loc[0] == self.shape[0]-1))
        return xlabel, ylabel

    def has_ticks_labels(self, loc: Location) -> Tuple[bool, bool]:
        """Check if axis has tick labels."""
        xticks = (not self.sharex or 
                  (self.sharex and loc[0] == self.shape[0]-1))
        yticks = not self.sharey or (self.sharey and loc[1] == 0)
        return xticks, yticks

    def get_value(self, 
                  key: str, 
                  loc: Optional[Location] = None, 
                  **kwargs) -> Any:
        """Get value from configuration.
        
        Args:
          key: option in the configuration parser.
          loc: optional; axis location.
          **kwargs: optional arguments for configparseradv.getvalue. 
            See configparseradv documentation for a list of available kwargs.
        """
        # Convert location to index
        if loc is not None and loc in self.axes:
            kwargs['n'] = self.axes.keys().index(loc)

        return self.config.getvalue(key, **kwargs)

    def set_title(self, title: str, **kwargs) -> None:
        """Set figure title."""
        self.fig.suptitle(title, **kwargs)

    def savefig(self, filename: Path, **kwargs) ->None:
        """Save figure."""
        self.fig.savefig(filename, **kwargs)

