import astropy.units as u

from .base_plotter import BasePlotter, SinglePlotter

class Plotter(SinglePlotter):
    pass

class AdvancedPlotter(SinglePlotter):
    def hvlines(self, hv, config=None, positions=None, unit=None,
            restfreq=None, **kwargs):
        # Find iterator
        if positions is not None:
            iterover = positions
        elif config is not None and hv in config:
            iterover = config.getvalueiter(hv, sep=',', dtype='quantity')
        else:
            return

        # For spectral units
        if restfreq is not None:
            pass
        elif 'restfreq' in config:
            restfreq = config.getquantity('restfreq')
        else:
            restfreq = None

        # Iterate over lines
        for i, val in enumerate(iterover):
            # Convert to unit
            try:
                val = val.to(unit)
            except u.UnitConversionError:
                val = val.to(unit, equivalencies=u.doppler_radio(restfreq))

            # Line configuration
            if config is not None:
                kwargs = {'c': config.getvalue('%s_color' % hv, n=i,
                        fallback='#6e6e6e'),
                    'ls': config.getvalue('%s_style' % hv, n=i, fallback='--'),
                    'zorder':kwargs.get('%s_zorder' % hv, 0)}

            # Plot
            if hv == 'hlines':
                self.axhline(val.value, **kwargs)
            else:
                self.axvline(val.value, **kwargs)

    def auto_plot(self, data, config, fig, hasxlabel, hasylabel, hasxticks, 
            hasyticks, **kwargs):
        """This function only works if myConfigParser is used and data is
        astropy quantity
        """
        # Axes labels
        xlabel = kwargs.setdefault('xlabel', config.get('xlabel', fallback='x'))
        ylabel = kwargs.setdefault('ylabel', config.get('ylabel', fallback='y'))

        # Units
        labelfmt = "{0} ({1.unit:latex_inline})"
        try:
            xunit = u.Unit(config.get('xunit'))
            kwargs['xlabel'] = labelfmt.format(xlabel, 1.*xunit)
        except:
            xunit = None
        try:
            yunit = u.Unit(config.get('yunit'))
            kwargs['ylabel'] = labelfmt.format(ylabel, 1.*yunit)
        except:
            yunit = None

        # Iterate over data
        xlim = None
        for i,dt in enumerate(data):
            # Default 
            opts = {}
            opts['color'] = config.getvalue('color', n=i, fallback='k')
            opts['zorder'] = config.getvalue('zorder', n=i, fallback=1,
                    dtype=int)
            
            # Plot data
            if len(dt) == 1:
                if yunit is None:
                    yunit = dt[0].unit
                    kwargs['ylabel'] = labelfmt.format(ylabel, 1*yunit)
                y = dt[0].to(yunit)
                self.plot(y, **opts)
            elif len(dt) == 2:
                if xunit is None:
                    xunit = dt[0].unit
                    kwargs['xlabel'] = labelfmt.format(xlabel, 1*xunit)
                if yunit is None:
                    yunit = dt[1].unit
                    kwargs['ylabel'] = labelfmt.format(ylabel, 1*yunit)
                x = dt[0].to(xunit)
                y = dt[1].to(yunit)
                if xlim is None and  'xlim_index' in config:
                    xlimind = config.getintlist('xlim_index')
                    xlim = kwargs.setdefault('xlim', 
                            (x[xlimind[0]].value, x[xlimind[1]].value))
                self.plot(x, y, **opts)
            elif len(dt) == 3:
                if xunit is None:
                    xunit = dt[0].unit
                    kwargs['xlabel'] = labelfmt.format(xlabel, 1*xunit)
                if yunit is None:
                    yunit = dt[1].unit
                    kwargs['ylabel'] = labelfmt.format(ylabel, 1*yunit)
                x = dt[0].to(xunit)
                y = dt[1].to(yunit)
                yerr = dt[2].to(yunit)
                if xlim is None and  'xlim_index' in config:
                    xlimind = config.getintlist('xlim_index')
                    xlim = kwargs.setdefault('xlim', 
                            (x[xlimind[0]].value, x[xlimind[1]].value))
                self.errorbar(x, y, yerr, **opts)

        # Limits
        if xlim is None and 'xlim' in config:
            kwargs['xlim'] = config.getfloatlist('xlim')
        if 'ylim' in config:
            kwargs['ylim'] = config.getfloatlist('ylim')

        # Axes labels
        if not hasxlabel:
            kwargs['xlabel'] = ''
        if not hasylabel:
            kwargs['ylabel'] = ''

        # Ticks
        kwargs['unset_xticks'] = not hasxticks
        kwargs['unset_yticks'] = not hasyticks
        kwargs.setdefault('tickscolor', config.get('tickscolor', fallback='k'))
        self.config_plot(**kwargs)

        # Plot vertical/horizontal lines
        self.hvlines('vlines', config=config, unit=xunit)
        self.hvlines('hlines', config=config, unit=yunit)

        # Label 
        if 'label' in config:
            kwargslab = {}
            if 'label_background' in config:
                kwargslab['backgroundcolor'] = config['label_background']
            if 'label_loc' in config:
                kwargslab['loc'] = config.getfloatlist('label_loc')
            if 'label_color' in config:
                kwargslab['color'] = config.get('label_color')
            self.label_axes(config['label'], **kwargslab)

class NPlotter(BasePlotter):

    def __init__(self, config=None, section='nplotter', **kwargs):

        super(NPlotter, self).__init__(config=config, section=section,
                kwargs=kwargs)

    def get_axis(self, loc, projection='rectilinear', include_cbar=False):
        axis, cbaxis = super(NPlotter,self).get_axis(loc,
                projection=projection, include_cbar=include_cbar)

        self.axes[loc] = Plotter(axis, cbaxis)
        return self.axes[loc]

    def init_axis(self, loc, projection='rectilinear', include_cbar=False):
        super(NPlotter, self).init_axis(loc, projection=projection,
                include_cbar=include_cbar)

    def auto_config(self, config=None, section='nplotter', legend=False,
            units=None, **kwargs):
        # Read new config if requested
        if config is not None:
            cfg = ConfigParser()
            cfg.read(os.path.expanduser(config))
            cfg = cfg[section]
        else:
            cfg = self.config

        for i,(loc,ax) in enumerate(self.axes.items()):
            if not self.is_init(loc):
                break

            # Limits
            try:
                xlim = map(float, 
                        self.get_value('xlim', ax=loc, sep=',').split())
            except AttributeError:
                xlim = ax.xlim
            try:
                ylim = map(float, 
                        self.get_value('ylim', ax=loc, sep=',').split())
            except AttributeError:
                ylim = ax.ylim

            # Scales
            xscale = self.get_value('xscale', default='linear', ax=loc)
            yscale = self.get_value('yscale', default='linear', ax=loc)

            # Ticks and labels
            xlabel, ylabel = self.has_axlabels(loc)
            xticks, yticks = self.has_ticks(loc)
            if xlabel:
                xlabel = self.get_value('xlabel', default='x', ax=loc, sep=',')
                if units is not None and units[0][i] is not None:
                    xlabel += ' ({0.unit:latex_inline})'.format(units[0][i])
            else:
                xlabel = ''
            if ylabel:
                ylabel = self.get_value('ylabel', default='y', ax=loc, sep=',')
                if units is not None and units[1][i] is not None:
                    ylabel += ' ({0.unit:latex_inline})'.format(units[1][i])
            else:
                ylabel = ''

            ax.config_plot(xlim=xlim, ylim=ylim, xscale=xscale,
                    yscale=yscale, xlabel=xlabel, ylabel=ylabel,
                    unset_xticks = not xticks, unset_yticks = not yticks)

            if (legend or self.config.getboolean('legend', fallback=False)) and loc==(0,0):
                ax.legend(auto=True, loc=4, match_colors=True,
                        fancybox=self.config.getboolean('fancybox', fallback=False),
                        framealpha=self.config.getfloat('framealpha', fallback=None),
                        facecolor=self.config.get('facecolor', fallback=None))

