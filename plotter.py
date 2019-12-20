from .base_plotter import BasePlotter, SinglePlotter

class Plotter(SinglePlotter):
    pass

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

