"""Plot handler for different types of spectra formats."""

from .plot_handler import PhysPlotHandler

def SpectrumHandler(PhysPlotHandler):
    """Handler for `line_little_helper.Spectrum` plots.
    
    Attributes:
      axis: axes of the plot (alias ax).
      axes_props: properties of axes.
      pltd: plotted objects tracker.
      is_config: set to `True` when configuration has been applied.
      skeleton: base configuration.
    """

    def auto_plot(self,
                  data: 'line_little_helper.spectrum.Spectrum',
                  dtype: str,
                  config: 'configparseradv.ConfigParserAdv') -> None:
        """Automatic plot of spectrum.

        Args:
          data: spectrum to plot.
          dtype: type of plot.
          config: config parser proxy.
        """
        ls = config.get('linestyle', fallback='-')
        ds = config.get('drawstyle', fallback='steps-mid')
        cl = config.get('color', fallback='k')
        self.plot(data.spectral_axis, data.intensity, ls=ls, ds=ds, c=cl)
    
