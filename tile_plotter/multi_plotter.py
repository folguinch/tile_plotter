"""Manage different types of plots."""
from typing import Optional, Sequence
from itertools import product

from .base_plotter import BasePlotter
from .data_loaders import data_loader
from .handlers import get_handler, HANDLERS
from .common_types import PlotHandler, Location

class MultiPlotter(BasePlotter):
    """Multiple plot manager.

    It determines the type of plot for each `section` of the input
    configuration and initiates the appropriate handler.

    Attributes:
      config: configuration parser proxy of the current section.
      fig: matplotlib figure.
      figsize: figure size.
      axes: the figure axes.
    """

    def __init__(self,
                 config: Optional['pathlib.Path'] = None,
                 verbose: str = 'v',
                 **kwargs):
        """Initialize plotter."""
        super().__init__(config=config, section='DEFAULT', verbose=verbose,
                         **kwargs)

    @property
    def plot_type(self):
        """Type of plot."""
        return self.config['type']

    # Implement abstract methods: init_axis, plot_all, apply_config
    def init_axis(self,
                  loc: Location,
                  projection: Optional[str] = None,
                  include_cbar: Optional[bool] = None,
                  **kwargs) -> PlotHandler:
        """Initialize axis by assigning a plot handler.

        If the axis is not initialized, it replaces the axis geometry
        (`FigGeometry`) with a plot handler. The plot handler is determined
        from the configuration.

        Args:
          loc: axis location.
          projection: optional; update axis projection.
          include_cbar: optional; initialize the color bar.
          kwargs: additional arguments for the handler.
        """
        handler = get_handler(self.config)
        axis = super().init_axis(loc, handler, projection=projection,
                                 include_cbar=include_cbar)

        return axis

    def plot_all(self):
        """Plot all the sections in config.

        Data is loaded based on the plot type.
        """
        for loc, sections in self:
            self.plot_sections(sections, loc)

    def apply_config(self, loc: Location, handler: PlotHandler, dtype: str):
        """Apply the plot configuration."""
        # Title
        title = self.config.get('title', fallback=None)
        if title is not None:
            self.log.debug('Plotting title: %s', title)
            handler.title(title)

        # Configuration
        set_xlabel, set_ylabel = self.has_axlabels(loc)
        set_xticks, set_yticks = self.has_ticks_labels(loc)
        self.log.debug('Label switches: %s, %s', set_xlabel, set_ylabel)
        self.log.debug('Tick switches: %s, %s', set_xticks, set_yticks)
        if dtype in ['image', 'contour', 'moment']:
            handler.config_map(set_xlabel=set_xlabel, set_ylabel=set_ylabel,
                               set_xticks=set_xticks, set_yticks=set_yticks)
        else:
            handler.config_plot(set_xlabel=set_xlabel, set_ylabel=set_ylabel,
                                set_xticks=set_xticks, set_yticks=set_yticks)

        # Axes label
        label = self.config.get('label', fallback='')
        label_type = self.config.get('label_type', fallback='normal')
        if label_type == 'normal':
            label_loc = self.config.getfloatlist('label_position',
                                                 fallback=(0.1, 0.9))
            label_ha = 'left'
        elif label_type in ['science', 'nature']:
            label_loc = self.config.getfloatlist('label_position',
                                                 fallback=(0.9, 0.9))
            label_ha = 'right'
        else:
            raise ValueError(f'Axis label type not available: {label_type}')
        label_bkgc = self.config.getfloatlist('label_backgroundcolor',
                                              fallback='w')
        enum = self.config.getboolean('enumerate')
        if len(self.axes) > 1 and enum:
            nrows, ncols = self.shape
            ind = list(product(range(nrows), range(ncols))).index(loc)
            enumerate_shift = self.config.getint('enumerate_shift', fallback=0)
            enumerate_modulus = self.config.getint('enumerate_modulus',
                                                   fallback=None)
            if label_type in ['science', 'nature']:
                number = ''
                axnum = chr(ord('A') + ind + enumerate_shift)
                handler.label_axes(axnum,
                                   loc=(0.1, 0.9),
                                   backgroundcolor=label_bkgc,
                                   fontsize=9,
                                   fontweight='bold')
            else:
                enum_fmt = self.config['enumerate_fmt']
                real_index = ind + enumerate_shift
                if enumerate_modulus is not None:
                    div = real_index//enumerate_modulus
                    number1 = ord('a') + div
                    number2 = ord('a') + real_index%enumerate_modulus
                    number = enum_fmt.format(chr(number1) + chr(number2)) + ' '
                else:
                    n = abs(ord('a') + ord('z')) + 1
                    div = real_index//n
                    if div == 0:
                        number = ord('a') + real_index
                        number = enum_fmt.format(chr(number))
                    else:
                        number1 = ord('a') + div - 1
                        number2 = ord('a') + real_index%n
                        number = enum_fmt.format(chr(number1) + chr(number2))
                    number += ' '
            label = f'{number}{label}'.strip()
        if len(label) >= 1:
            handler.label_axes(label, loc=label_loc,
                               backgroundcolor=label_bkgc,
                               horizontalalignment=label_ha)

    def plot_sections(self, sections: Sequence, loc: Location):
        """Plot the requested sections in the requested location.

        Args:
          sections: list of sections to plot.
          loc: location where the sections are plotted.
        """
        color_bars = {}
        for section in sections:
            print('-' * 50)
            # Reset config
            self.log.info('Plotting %s in (%i, %i)', section, *loc)
            self.switch_to(section)

            # Load data
            self.log.debug('Loading data')
            data, projection, dtype = data_loader(self.config,
                                                  log=self.log.info)

            # Get axis
            handler = self.init_axis(loc, projection=projection)

            # Plot
            self.log.info('Plotting data')
            handler.auto_plot(data, dtype, self.config)

            # Plot color bar
            if self.has_cbar(loc) and not color_bars.get(loc, False):
                self.log.info('Setting color bar')
                color_bars[loc] = True
                handler.plot_cbar(self.fig, self.axes[loc].cborientation)

            # Config plot: things that need to be plotted once
            if not handler.is_config:
                # Plot artists
                self.log.info('Plotting artists')
                handler.plot_artists()

                # Config
                self.log.info('Configuring plot')
                self.apply_config(loc, handler, dtype)

        # Plot legends:
        for axis in self.axes:
            if self.is_init(axis):
                self.axes[axis].legend()

class OTFMultiPlotter(BasePlotter):
    """On-the-fly multiple plot manager.

    It manages multiplotter without needing a configuration file. It
    can be initialized with the basic properties of the plot that replace the
    default values of a tile plot (e.g. properties like the number of rows and
    columns).

    Attributes:
      config: configuration parser proxy of the current section.
      fig: matplotlib figure.
      figsize: figure size.
      axes: the figure axes.
    """

    def __init__(self, **props):
        """Initialize plotter."""
        super().__init__(section='DEFAULT', **props)

    def init_axis(self,
                  loc: Location,
                  handler: str,
                  projection: Optional[str] = None,
                  include_cbar: Optional[bool] = None,
                  ) -> PlotHandler:
        """Initialize axis by assigning a plot handler.

        If the axis is not initialized, it replaces the axis geometry
        (`FigGeometry`) with a plot handler. The plot handler is determined
        from the configuration.

        Args:
          loc: axis location.
          handler: handler name.
          projection: optional; update axis projection.
          include_cbar: optional; initialize the color bar.
        """
        handler = HANDLERS[handler]
        axis = super().init_axis(loc, handler, projection=projection,
                                 include_cbar=include_cbar)

        return axis

    def gen_handler(self,
                    loc: Location,
                    handler: str,
                    projection: Optional[str] = None,
                    include_cbar: Optional[bool] = None,
                    **props) -> PlotHandler:
        """Generate a handler with the given properties.

        Args:
          loc: axis location.
          handler: handler name.
          projection: optional; update axis projection.
          include_cbar: optional; initialize the color bar.
          props: optional; properties for the handler.
        """
        # Generate a new section
        section = f'section{loc[0]}{loc[1]}'
        props['loc'] = f'{loc[0]} {loc[1]}'
        props['handler'] = handler
        self.log.info('Generating dummy %s for (%i, %i)', section, *loc)
        self.insert_section(section, value=props, switch=True)

        return self.init_axis(loc, handler, projection=projection,
                              include_cbar=include_cbar)

    def plot_all(self):
        pass

    def apply_config(self, loc: Location, handler: PlotHandler, dtype: str,
                     label: str = '', **kwargs):
        """Apply the plot configuration."""
        # Title
        title = self.config.get('title', fallback=None)
        if title is not None:
            self.log.debug('Plotting title: %s', title)
            handler.title(title)

        # Configuration
        set_xlabel, set_ylabel = self.has_axlabels(loc)
        set_xticks, set_yticks = self.has_ticks_labels(loc)
        if dtype in ['image', 'contour', 'moment']:
            self.log.debug('Label switches: %s, %s', set_xlabel, set_ylabel)
            self.log.debug('Tick switches: %s, %s', set_xticks, set_yticks)
            handler.config_map(set_xlabel=set_xlabel, set_ylabel=set_ylabel,
                               set_xticks=set_xticks, set_yticks=set_yticks)
        else:
            handler.config_plot(**kwargs)

        # Axes label
        label = self.config.get('label', fallback=label)
        if label:
            label_loc = self.config.getfloatlist('label_position',
                                                 fallback=(0.1, 0.9))
            label_bkgc = self.config.getfloatlist('label_backgroundcolor',
                                                  fallback='w')
            handler.label_axes(label, loc=label_loc, backgroundcolor=label_bkgc)

#    def get_plotter(self, axis, cbax, projection):
#        dtype = self.config['type']
#        if dtype.lower() in ['map', 'contour_map', 'moment']:
#            # Get color stretch values
#            stretch = self.config.get('stretch', fallback='linear')
#            try:
#                vmin = float(self.config.get('vmin'))
#            except:
#                vmin = None
#            try:
#                vmax = float(self.config.get('vmax'))
#            except:
#                vmax = None
#            try:
#                a = float(self.config.get('a'))
#            except:
#                a = 1000.
#
#            # Radesys
#            radesys = ''
#            if projection is not None:
#                aux = projection.to_header()
#                if 'RADESYS' in aux:
#                    radesys = aux['RADESYS']
#
#            # Get the axis
#            plotter = MapPlotter(axis, cbax, vmin=vmin, vmax=vmax, a=a,
#                    stretch=stretch, radesys=radesys)
#        elif dtype.lower() in ['data', 'spectrum']:
#            xscale = self.config.get('xscale', fallback='linear')
#            yscale = self.config.get('yscale', fallback='linear')
#            plotter = AdvancedPlotter(axis, cbaxis=cbax, xscale=xscale,
#                                        yscale=yscale)
#        else:
#            raise NotImplementedError('Plot type %s not implemented yet'
#                                       % dtype)
#
#        return plotter
#
#    def get_axes_config(self, loc):
#        xlabel, ylabel = self.has_axlabels(loc)
#        xticks, yticks = self.has_ticks(loc)
#
#        return xlabel, ylabel, xticks, yticks
#
