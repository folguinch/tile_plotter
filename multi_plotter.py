try:
    from myutils.myconfigparser import myConfigParser as ConfigParser
except ImportError:
    from configparser import ConfigParser

from base_plotter import BasePlotter
from map_plotter import MapPlotter
from plotter import AdvancedPlotter

__metaclass__ = type

class MultiPlotter(BasePlotter):

    def __init__(self, config=None, **kwargs):
        super(MultiPlotter, self).__init__(config=config, section='DEFAULT',
                **kwargs)

    def init_axis(self, loc, projection='rectilinear', include_cbar=None):
        super(MultiPlotter, self).init_axis(loc, projection, include_cbar)

    def get_plotter(self, axis, cbax, projection):
        dtype = self.config['type']
        if dtype.lower() in ['map', 'contour_map', 'moment']:
            # Get color stretch values
            stretch = self.config.get('stretch', fallback='linear')
            try:
                vmin = float(self.config.get('vmin'))
            except:
                vmin = None
            try:
                vmax = float(self.config.get('vmax'))
            except:
                vmax = None
            try:
                a = float(self.config.get('a'))
            except:
                a = 1000.

            # Radesys
            radesys = '' 
            if projection is not None:
                aux = projection.to_header()
                if 'RADESYS' in aux:
                    radesys = aux['RADESYS']

            # Get the axis
            plotter = MapPlotter(axis, cbax, vmin=vmin, vmax=vmax, a=a,
                    stretch=stretch, radesys=radesys)
        elif dtype.lower() in ['data', 'spectrum']:
            xscale = self.config.get('xscale', fallback='linear')
            yscale = self.config.get('yscale', fallback='linear')
            plotter = AdvancedPlotter(axis, cbaxis=cbax, xscale=xscale, yscale=yscale)
        else:
            raise NotImplementedError('Plot type %s not implemented yet' % dtype)

        return plotter

    def get_axes_config(self, loc):
        xlabel, ylabel = self.has_axlabels(loc)
        xticks, yticks = self.has_ticks(loc)

        return xlabel, ylabel, xticks, yticks

    def auto_plot(self, loaders):
        for section in self._config.sections():
            # Reset config
            self.log.info('Plotting: %s', section)
            self.config = self._config[section]

            # Load data
            self.log.info('Loading data')
            data, projection = loaders[section](self.config)

            # Axis location
            if 'loc' not in self.config:
                raise KeyError('Location for %s not found' % section)
            loc = tuple(self.config.getintlist('loc'))

            # Get axis
            include_cbar = self.config.getboolean('include_cbar',
                    fallback=False)
            axis, cbax = self.get_axis(tuple(loc), projection=projection,
                    include_cbar=include_cbar)

            # Plot
            if not hasattr(self.axes[loc], 'auto_plot'):
                self.axes[loc] = self.get_plotter(axis, cbax, projection)
            if include_cbar:
                fig = self.fig
            else:
                fig = None
            self.log.info('Plotting data')
            self.axes[loc].auto_plot(data, self.config, fig,
                    *self.get_axes_config(loc))


