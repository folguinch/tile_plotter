"""Manage different types of plots."""
from typing import Optional, Sequence

import configparseradv.configparser as cfgparser

from .base_plotter import BasePlotter, Location
from .data_loaders import data_loader
from .handlers import get_handler

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
                 config: Optional['Path'] = None,
                 config_parser: Optional[cfgparser.ConfigParserAdv] = None,
                 **kwargs):
        """Initialize plotter."""
        super().__init__(config=config, config_parser=config_parser,
                         section='DEFAULT', **kwargs)

    @property
    def plot_type(self):
        """Type of plot."""
        return self.config['type']

    # Implement abstract methods: init_axis, plot_all, apply_config
    def init_axis(self,
                  loc: Location,
                  projection: Optional[str] = None,
                  include_cbar: Optional[bool] = None,
                  **kwargs) -> 'PlotHandler':
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

    def apply_config(self, loc: Location, handler: 'PlotHandler', dtype: str):
        """Apply the plot configuration."""
        # Configuration
        set_xlabel, set_ylabel = self.has_axlabels(loc)
        set_xticks, set_yticks = self.has_ticks_labels(loc)
        if dtype in ['image', 'contour_map']:
            handler.config_map(set_xlabel=set_xlabel, set_ylabel=set_ylabel,
                               set_xticks=set_xticks, set_yticks=set_yticks)
        else:
            handler.config_plot()

        # Axes label
        label = self.config.get('label', fallback='')
        label_loc = self.config.getfloatlist('label_position',
                                             fallback=(0.1, 0.9))
        if len(self.axes) > 1:
            ind = self.axes.keys().index(loc)
            label = f"({chr(ord('a') + ind)}) {label}"
        if label:
            handler.label_axes(label, loc=label_loc)

    def plot_sections(self, sections: Sequence, loc: Location):
        """Plot the requested sections in the requested location.

        Args:
          sections: list of sections to plot.
          loc: location where the sections are plotted.
        """
        color_bars = {}
        is_config = ()
        for section in sections:
            print('-' * 50)
            # Reset config
            self._log.info('Plotting %s in (%i, %i)', section, *loc)
            self.switch_to(section)

            # Load data
            data, projection, dtype = data_loader(self.config,
                                                  log=self._log.info)

            # Get axis
            handler = self.init_axis(loc, projection=projection)

            # Update label
            if len(self.axes) > 1 and 'label' in handler.artists:
                ind = self.axes.keys().index(loc)
                label = self.artists[artist]['properties'][0].get('text', '')
                label = f'({chr(ind+1)}) {label}'
                self.artists[artist]['properties'][0] = label

            # Plot
            self._log.info('Plotting data')
            handler.auto_plot(data, dtype, self.config)

            # Plot color bar
            if self.has_cbar(loc) and not color_bars.get(loc, False):
                self._log.info('Setting color bar')
                color_bars[loc] = True
                handler.plot_cbar(self.fig,
                                  orientation=self.axes[loc].cborientation)

            # Config plot
            if loc not in is_config:
                self.apply_config(loc, handler, dtype)
                is_config += (loc,)

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
