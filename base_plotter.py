"""Define the base plotter file for all the plotting tools."""
from typing import Any, Optional, Tuple
import abc
import pathlib

from logging_tools import get_logger
import configparseradv.configparser as cfgparser
import matplotlib.pyplot as plt

from .geometry import GeometryHandler

# Type aliases
Location = Tuple[int, int]

class BasePlotter(metaclass=abc.ABCMeta):
    """Figure axes collection base class.

    Keeps track of the Figure axes, whether they have been initialized or not.

    The `BasePlotter` opens and store the configuration file. The plot geometry
    can be described in 1 or 2 configurations files:

    - 1 file: share the plot geometry information with the plot data
      information in one configuration file.
    - 2 files: the plot data information file defines a
      `geometry_config` option with the file name of the geometry information
      file. In this case, after reading the configuration in the data file, the
      configuration is updated with the geometry file, i.e. any value in the
      default of the data file may be replaced if it is in the geometry file.

    The 2nd option is useful for cases where the same geometry is used to plot
    different data. All the options for geometry configuration are available in
    `configs/default.cfg`.

    Attributes:
      config: configuration parser proxy of the current section.
      fig: matplotlib figure.
      figsize: figure size.
      axes: the figure axes.
    """

    _defconfig = (pathlib.Path(__file__).resolve().parent /
                  pathlib.Path('configs/default.cfg'))
    _log = get_logger(__name__)

    def __init__(self,
                 config: Optional[pathlib.Path] = None,
                 section: str = 'DEFAULT',
                 **kwargs):
        """Create a new base plotter."""
        # Close plt if still open
        plt.close()

        # Update options
        self._config = cfgparser.ConfigParserAdv()
        self._log.debug(f'Default config: {self._defconfig}')
        self._config.read(self._defconfig)
        if config is None:
            pass
        else:
            self._config.read(config.expanduser().resolve())
        self._config.read_dict({section: kwargs})

        # Read geometry configuration
        if 'geometry_config' in self._config[section]:
            self._config.read(self._config[section]['geometry_config'])
        self.config = self._config[section]

        # Set plot styles
        plt.style.use(self.config.get('styles').replace(',', ' ').split())

        # Get axes
        self.axes = GeometryHandler()
        self.figsize = self.axes.fill_from_config(self.config)
        self._log.info(f'Figure size: w={self.figsize[0]}, h={self.figsize[1]}')

        # Set the configuration mapping loc and config keys
        self._config_mapping = self._group_config_sections()

        # Create figure
        self.fig = plt.figure(figsize=self.figsize)

    def __iter__(self):
        for loc in self._config_mapping.items():
            yield loc

    def _group_config_sections(self):
        """Group config sections based on location of each section."""
        mapping = {}
        for section in self._config.sections():
            loc = tuple(self._config.getintlist(section, 'loc'))
            if loc in mapping:
                mapping[loc] += [section]
            else:
                mapping[loc] = [section]
        return mapping

    @property
    def loc(self):
        return tuple(self.config.getintlist('loc'))

    @property
    def section(self):
        return self.config.name

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

    @abc.abstractmethod
    def plot_all(self):
        """Iterate over the configuration sections and plot each one."""
        pass

    @abc.abstractmethod
    def apply_config(self):
        """Apply the configuration of the axis."""
        pass

    def switch_to(self, section: str) -> None:
        """Change the current `config` proxy."""
        self.config = self._config[section]

    def is_init(self, loc: Location, cbaxis: bool = False) -> bool:
        """Check if an axis has been initialized.

        Args:
          loc: axis location.
          cbaxis: whether to check for a colorbar axis instead.
        """
        if not cbaxis:
            #return hasattr(self.axes[loc].axis, 'plot')
            return self.axes[loc].handler is not None
        else:
            #return hasattr(self.axes[loc].cbaxis, 'plot')
            return self.axes[loc].handler is not None

    @abc.abstractmethod
    def init_axis(self,
                  loc: Location,
                  handler: 'PlotHandler',
                  projection: Optional[str] = None,
                  include_cbar: Optional[bool] = None,
                  **kwargs) -> 'PlotHandler':
        """Initialize axis by assigning a plot handler.

        If the axis is not initialized, it replaces the axis geometry
        (`AxisHandler`) with a plot handler.

        Args:
          loc: axis location.
          handler: plot handler to replace the geometry.
          projection: optional; update axis projection.
          include_cbar: optional; initialize the color bar.
          kwargs: additional arguments for the handler.
        """
        # Check projection
        if projection is None:
            projection = self.projection or 'rectilinear'

        # Verify include_cbar
        if include_cbar is None:
            include_cbar = self.has_cbar(loc)

        # Initialize axis if needed
        if self.is_init(loc):
            self._log.info(f'Axis {loc} already initialized')
            if include_cbar:
                cbaxis = self.init_cbar(loc)
            return self.axes[loc]
        else:
            self._log.info(f'Initializing axis: {loc}')
            self.axes[loc].axis.scalex(1./self.figsize[0])
            self.axes[loc].axis.scaley(1./self.figsize[1])
            axis = self.fig.add_axes(self.axes[loc].axis.pyplot_axis,
                                     projection=projection)

        # Color bar
        if include_cbar:
            cbaxis = self.init_cbar(loc)

        # Create plotter object
        self.axes[loc].set_handler(handler.from_config(self.config, axis,
                                                       cbaxis, **kwargs))

        return self.axes[loc].handler

    def init_cbar(self, loc: Location) -> None:
        """Initialize color bar."""
        if self.is_init(loc, cbaxis=True) or not self.has_cbar(loc):
            self._log.info(f'Color bar axis {loc} already initialized')
        else:
            self._log.info(f'Initializing color bar: {loc}')
            self.axes[loc].cbaxis.scalex(1. / self.figsize[0])
            self.axes[loc].cbaxis.scaley(1. / self.figsize[1])
            self.axes[loc].cbaxis = self.fig.add_axes(
                self.axes[loc].cbaxis.pyplot_axis)

        return self.axes[loc].cbaxis

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
          kwargs: optional arguments for `configparseradv.getvalue`.
            See `configparseradv` documentation for a list of available kwargs.
        """
        # Convert location to index
        if loc is not None and loc in self.axes:
            kwargs['n'] = self.axes.keys().index(loc)

        return self.config.getvalue(key, **kwargs)

    def set_title(self, title: str, **kwargs) -> None:
        """Set figure title."""
        self.fig.suptitle(title, **kwargs)

    def savefig(self, filename: pathlib.Path, **kwargs) -> None:
        """Save figure."""
        self.fig.savefig(filename, **kwargs)

