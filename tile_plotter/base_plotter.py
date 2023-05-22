"""Define the base plotter file for all the plotting tools."""
from typing import Any, Optional, Tuple, Dict
import abc
import pathlib

from toolkit.logger import LoggedObject
import configparseradv.configparser as cfgparser
import matplotlib.pyplot as plt

from .geometry import GeometryHandler
from .common_types import PlotHandler, Location

class BasePlotter(LoggedObject, metaclass=abc.ABCMeta):
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
                  pathlib.Path('../configs/default.cfg'))

    def __init__(self,
                 config: Optional[pathlib.Path] = None,
                 section: str = 'DEFAULT',
                 verbose: str = 'v',
                 **kwargs):
        """Create a new base plotter."""
        # Initiate log
        super().__init__(__name__, filename='plotter.log', verbose=verbose)

        # Close plt if still open
        plt.close()

        # Update options
        self._config = cfgparser.ConfigParserAdv()
        self.log.debug('Default config: %s', self._defconfig)
        self._config.read(self._defconfig)
        if config is None:
            pass
        else:
            self.log.debug('Reading input config: %s', config)
            self._config.read(config.expanduser().resolve())
        self._config.read_dict({section: kwargs})

        # Read geometry configuration
        if 'geometry_config' in self._config[section]:
            self.log.debug('Reading geometry config: %s',
                            self._config[section]['geometry_config'])
            self._config.read(self._config[section]['geometry_config'])
        self.config = self._config[section]

        # Set plot styles
        styles = self.config.get('styles').replace(',', ' ').split()
        self.log.debug('Setting styles: %s', styles)
        plt.style.use(styles)

        # Get axes
        self.log.debug('Initializing geometry handler')
        self.axes = GeometryHandler(verbose=verbose)
        self.figsize = self.axes.fill_from_config(self.config)
        self.log.info('Figure size: w=%f, h=%f',
                       self.figsize[0], self.figsize[1])

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
            locs = self._config[section]['loc'].split(',')
            for loc in locs:
                loc_tuple = tuple(map(int, loc.split()))
                if loc_tuple in mapping:
                    mapping[loc_tuple] += [section]
                else:
                    mapping[loc_tuple] = [section]
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
    def init_axis(self,
                  loc: Location,
                  handler: PlotHandler,
                  projection: Optional[str] = None,
                  include_cbar: Optional[bool] = None,
                  **kwargs) -> PlotHandler:
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
            self.log.info('Axis %s already initialized', loc)
            if include_cbar:
                cbaxis = self.init_cbar(loc)
            return self.axes[loc].handler
        else:
            self.log.info('Initializing axis: %s', loc)
            self.axes[loc].axis.scalex(1./self.figsize[0])
            self.axes[loc].axis.scaley(1./self.figsize[1])
            axis = self.fig.add_axes(self.axes[loc].axis.pyplot_axis,
                                     projection=projection)

        # Color bar
        if include_cbar:
            cbaxis = self.init_cbar(loc)
        else:
            cbaxis = None

        # Create plotter object
        self.axes[loc].set_handler(handler.from_config(self.config, axis,
                                                       cbaxis, **kwargs))

        return self.axes[loc].handler

    @abc.abstractmethod
    def plot_all(self):
        """Iterate over the configuration sections and plot each one."""
        pass

    @abc.abstractmethod
    def apply_config(self):
        """Apply the configuration of the axis."""
        pass

    def insert_section(self, section: str, value: Dict,
                       switch: bool = False) -> None:
        """Insert a new section to the configuration."""
        if section not in self._config:
            self._config[section] = value
        else:
            raise KeyError(f'Section {section} already in config')
        self._config_mapping = self._group_config_sections()

        if switch:
            self.switch_to(section)

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

    def init_cbar(self, loc: Location) -> None:
        """Initialize color bar."""
        if self.is_init(loc, cbaxis=True) or not self.has_cbar(loc):
            self.log.info('Color bar axis %s already initialized', loc)
        else:
            self.log.info('Initializing color bar: %s', loc)
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
        override_xlabel = self.get_value('override_xlabel', loc=loc,
                                         dtype=bool, sep=',')
        override_ylabel = self.get_value('override_ylabel', loc=loc,
                                         dtype=bool, sep=',')
        if not override_xlabel:
            xlabel = (not self.sharex or
                      (self.sharex and loc[1] == 0 and
                       loc[0] == self.shape[0]-1))
        else:
            xlabel = None
        if not override_ylabel:
            ylabel = (not self.sharey or
                      (self.sharey and loc[1] == 0 and
                       loc[0] == self.shape[0]-1))
        else:
            ylabel = None
        return xlabel, ylabel

    def has_ticks_labels(self, loc: Location) -> Tuple[bool, bool]:
        """Check if axis has tick labels."""
        override_xticks = self.get_value('override_xticks', loc=loc, dtype=bool,
                                         sep=',')
        override_yticks = self.get_value('override_yticks', loc=loc, dtype=bool,
                                         sep=',')
        if not override_xticks:
            xticks = (not self.sharex or
                      (self.sharex and loc[0] == self.shape[0]-1))
        else:
            xticks = None
        if not override_yticks:
            yticks = not self.sharey or (self.sharey and loc[1] == 0)
        else:
            yticks = None
        return xticks, yticks

    def get_loc_index(self, loc: Location) -> int:
        """Get the axis index of the given location."""
        return sorted(self.axes.keys()).index(loc)

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
            kwargs['n'] = self.get_loc_index(loc)

        return self.config.getvalue(key, **kwargs)

    def set_title(self, title: str, **kwargs) -> None:
        """Set figure title."""
        self.fig.suptitle(title, **kwargs)

    def savefig(self, filename: pathlib.Path, **kwargs) -> None:
        """Save figure."""
        self.fig.savefig(filename, **kwargs)

