"""Objects for handling all type of plots."""
from typing import (Optional, Tuple, TypeVar, Callable, Dict, List,
                    Union, Sequence)
import pathlib

from configparseradv.configparser import ConfigParserAdv
from matplotlib import patches
from toolkit.logger import get_logger
import astropy.units as u
import matplotlib as mpl
#import matplotlib.lines as mlines
import numpy as np

from .utils import generate_label, get_colorbar_ticks, tick_formatter

# Type aliases
Axes = type(mpl.axes.Axes)
Artist = TypeVar('Artist')
Limits = Union[None, Tuple[float, float], Dict[str, float]]
Plot = TypeVar('Plot')

class PlotHandler:
    """Container for axes of a single plot.

    Class for easy and quick configuration of a single axes object. This plot
    may be included in a grid of several plots.

    Attributes:
      axis: axes of the plot (alias ax).
      cbaxis: color bar axes (alias cbax).
      xscale: x axis scale (linear or log).
      yscale: y axis scale (linear or log).
      xname: x axis name.
      yname: y axis name.
      xunit: x axis unit.
      yunit: y axis unit.
      pltd: plotted objects tracker.
    """
    # Common class attributes
    _log = get_logger(__name__, filename='plotter.log')
    _defconfig = (pathlib.Path(__file__).resolve().parent /
                  pathlib.Path('configs/plot_default.cfg'))

    # Read default skeleton
    skeleton = ConfigParserAdv()
    skeleton.read(_defconfig)

    def __init__(self,
                 axis: Axes,
                 cbaxis: Optional[Axes] = None,
                 xscale: str = 'linear',
                 yscale: str = 'linear',
                 xname: str = 'x',
                 yname: str = 'y',
                 xunit: Optional[str] = None,
                 yunit: Optional[str] = None) -> None:
        """ Create a single plot container."""
        self.axis = axis
        self.cbaxis = cbaxis
        self.xscale = xscale
        self.yscale = yscale
        self.xname = xname
        self.yname = yname
        self.xunit = xunit
        self.yunit = yunit
        self.pltd = {}

    @property
    def ax(self) -> Axes:
        return self.axis

    @property
    def cbax(self) -> Axes:
        return self.cbaxis

    @property
    def xlim(self) -> Tuple[float, float]:
        """Get x axis limits."""
        return self.ax.get_xlim()

    @property
    def ylim(self) -> Tuple[float, float]:
        """Get y axis limits."""
        return self.ax.get_ylim()

    def _simple_plt(self, fn: Callable[[], Plot],
                    *args, **kwargs) -> Plot:
        """Plot and assign label.

        Args:
          fn: matplotlib plotting function.
          *args: arguments for function.
          **kwargs: keyword arguments for function.

        Returns:
          The plotted object resulted from function.
        """
        return self.insert_plt(kwargs.get('label', None),
                               fn(*args, **kwargs))

    def insert_plt(self, key: str, val: Plot) -> Plot:
        """Store plotted object.

        Args:
          key: label of the plotted object.
          val: plotted object.

        Returns:
          The plotted object in val.
        """
        # Store contours handler separatedly
        #if isinstance(val, mpl.contour.ContourSet):
        #    print(val.collections[-1].get_cmap())
        #    color = tuple(val.collections[-1].get_cmap().tolist()[0])
        #    handler = mlines.Line2D([], [], color=color, label=key)
        #else:
        handler = val

        # Store plotted
        if key is not None:
            self.pltd[key] = handler
        else:
            pass

        return val

    def config_plot(self,
                    xlim: Limits = None,
                    ylim: Limits = None,
                    xscale: Optional[str] = None,
                    yscale: Optional[str] = None,
                    xlabel: Optional[str] = None,
                    ylabel: Optional[str] = None,
                    xticks: List[float] = None,
                    yticks: List[float] = None,
                    minor_xticks: List[float] = None,
                    minor_yticks: List[float] = None,
                    xticks_fmt: str = '{:.3f}',
                    yticks_fmt: str = '{:.3f}',
                    unset_xtick_labels: bool = False,
                    unset_ytick_labels: bool = False,
                    ticks_color: str = 'k') -> None:
        """Configure the plot.

        Configures several aspects of the axes: limits, scales, labels and
        ticks. It overrides and updates some of the default parameters.

        Args:
          xlim: optional; x axis limits.
          ylim: optional; y axis limits.
          xscale: optional; x axis scale (linear or log). Updates the stored
            scale.
          yscale: optional; y axis scale (linear or log). Updates the stored
            scale.
          xlabel: optional; x axis label.
          ylabel: optional; y axis label.
          xticks: optional; x ticks values.
          yticks: optional; y ticks values.
          minor_xticks: optional; minor x ticks values.
          minor_yticks: optional; minor y ticks values.
          xticks_fmt: optional; x axis ticks string format.
          yticks_fmt: optional; y axis ticks string format.
          unset_xtick_labels: optional; unset the x ticks labels?
          unset_ytick_labels: optional; unset the y ticks labels?
          ticks_color: optional; color of axis ticks.
        """
        # Limits
        if xlim is not None:
            if hasattr(xlim, 'index'):
                if len(xlim) != 2:
                    raise ValueError(f'Could not set xlim: {xlim}')
                xlim = dict(zip(['xmin', 'xmax'], xlim))
            self._log.info('Setting xlim: %s', xlim)
            self.set_xlim(**xlim)
        if ylim is not None:
            if hasattr(ylim, 'index'):
                if len(ylim) != 2:
                    raise ValueError(f'Could not set ylim: {ylim}')
                ylim = dict(zip(['ymin', 'ymax'], ylim))
            self._log.info('Setting ylim: %s', ylim)
            self.set_ylim(**ylim)

        # Axis scales
        self.xscale = xscale or self.xscale
        self.yscale = yscale or self.yscale
        if self.xscale == 'log':
            self.ax.set_xscale('log')
            self.ax.xaxis.set_major_formatter(tick_formatter('log'))
        if self.yscale == 'log':
            self.ax.set_yscale('log')
            self.ax.yaxis.set_major_formatter(tick_formatter('log'))

        # Labels
        self.set_axlabels(xlabel=xlabel, ylabel=ylabel)

        # Ticks
        if unset_xtick_labels:
            self.ax.set_xticklabels([''] * len(self.ax.get_xticks()))
        elif xticks:
            self.ax.set_xticks(xticks)
            self.ax.set_xticklabels([xticks_fmt.format(x) for x in xticks])
        if minor_xticks:
            self.ax.set_xticks(minor_xticks, minor=True)
        if unset_ytick_labels:
            self.ax.set_yticklabels([''] * len(self.ax.get_yticks()))
        if yticks:
            self.ax.set_yticks(yticks)
            self.ax.set_yticklabels([yticks_fmt.format(y) for y in yticks])
        if minor_yticks:
            self.ax.set_yticks(minor_yticks, minor=True)

        # Ticks colors
        self.ax.tick_params('both', color=ticks_color)

    # Getters
    def get_xlabel(self, unit_fmt: str = '({})') -> str:
        """Return the xlabel from stored values."""
        return generate_label(self.xname, unit=self.xunit, unit_fmt=unit_fmt)

    def get_ylabel(self, unit_fmt: str = '({})') -> str:
        """Return the ylabel from stored values."""
        return generate_label(self.yname, unit=self.yunit, unit_fmt=unit_fmt)

    # Setters
    def set_xlim(self, xmin: Optional[float] = None,
                 xmax: Optional[float] = None) -> None:
        """Set the axis x limits."""
        self.axis.set_xlim(left=xmin, right=xmax)

    def set_ylim(self, ymin: Optional[float] = None,
                 ymax: Optional[float] = None) -> None:
        """Set the axis y limits."""
        self.axis.set_ylim(bottom=ymin, top=ymax)

    def set_axlabels(self, xlabel: Optional[str] = None,
                     ylabel: Optional[str] = None) -> None:
        """Set the axis labels.

        Args:
          xlabel: optional; x axis label.
          ylabel: optional; y axis label.
        """
        self.ax.set_xlabel(xlabel or self.get_xlabel())
        self.ax.set_ylabel(ylabel or self.get_ylabel())

    # Matplotlib plotting functions
    def plot(self, *args, **kwargs) -> Plot:
        """Plot on the axis.

        Arguments are the same as for the matplotlib.pyplot.plot() finction.

        Args:
          args: data to plot.
          kwargs: arguments for matplotlib.pyplot.plot().
        """
        return self._simple_plt(self.axis.plot, *args, **kwargs)

    def axhline(self, *args, **kwargs) -> None:
        """Plot horizontal line.

        Arguments are the same as for the matplotlib.pyplot.axhline function.
        """
        self.axis.axhline(*args,**kwargs)

    def axvline(self, *args, **kwargs) -> None:
        """Plot vertical line.

        Arguments are the same as for the matplotlib.pyplot.axvline function.
        """
        self.axis.axvline(*args,**kwargs)

    def axvspan(self, *args, **kwargs) -> None:
        """Axis vertical span."""
        self.axis.axvspan(*args, **kwargs)

    def axhspan(self, *args, **kwargs) -> None:
        """Axis horizontal span."""
        self.axis.axhspan(*args, **kwargs)

    def errorbar(self, *args, **kwargs) -> Plot:
        """Plot on the axis.

        Arguments are the same as for the matplotlib.pyplot.errorbar() finction.

        Args:
          args: data to plot.
          kwargs: arguments for matplotlib.pyplot.errorbar().
        """
        return self._simple_plt(self.axis.errorbar, *args, **kwargs)

    def title(self, text: str, **kwargs) -> None:
        """Set plot title."""
        self.axis.set_title(text, **kwargs)

    def annotate(self, *args, **kwargs) -> None:
        """Annotate the axis.

        Arguments are the same as for the matplotlib.pyplot.annotate() finction.

        Args:
          args: data to plot.
          kwargs: arguments for matplotlib.pyplot.annotate().
        """
        self.axis.annotate(*args, **kwargs)

    def legend(self,
               handlers: Optional[Plot] = None,
               labels: Optional[List[str]] = None,
               loc: int = 0,
               auto: bool = False,
               match_colors: bool = False,
               **kwargs) -> None:
        """Plot the legend.

        Args:
          handlers: matplotlib plot handlers.
          labels: labels for each handler.
          loc: legend position. See matplotlib.pyplot.lengend() documentation
            for available values and positions (default 0, i.e. best location).
          auto: use stored handlers.
          match_colors: match legend artist and legend text colors.
        """
        # Get handlers from plotted data
        if auto:
            handlers = self.pltd.values()
            labels = self.pltd.keys()

        # Frame
        kwargs.setdefault('frameon', kwargs.get('fancybox'))

        # Plot legend
        if handlers and labels:
            leg = self.axis.legend(handlers, labels, loc=loc, **kwargs)
        else:
            leg = self.axis.legend(loc=loc, **kwargs)

        # Match text and artist colors
        if match_colors:
            for artist, text in zip(leg.legendHandles, leg.get_texts()):
                try:
                    col = artist.get_color()
                except AttributeError:
                    col = artist.get_facecolor()
                if isinstance(col, np.ndarray): col = col[0]
                text.set_color(col)

    def scatter(self, *args, **kwargs) -> Plot:
        """Scatter plot."""
        return self._simple_plt(self.axis.plot, *args, **kwargs)

    def text(self, *args, **kwargs) -> Plot:
        """Scatter plot."""
        return self._simple_plt(self.axis.text, *args, **kwargs)

    def clabel(self, *args, **kwargs) -> None:
        """Contour labels."""
        self.axis.clabel(*args, **kwargs)

    def contour(self, *args, **kwargs) -> Plot:
        """Contour map."""
        return self._simple_plt(self.axis.contour, *args, **kwargs)

    def tricontour(self, x, y, *args, **kwargs) -> Plot:
        """Triangulation contour map."""
        triangulation = mpl.tri.Triangulation(x, y)
        return self._simple_plt(self.axis.tricontour, triangulation,
                                *args, **kwargs)

    def plot_cbar(self,
                  fig: mpl.figure.Figure,
                  cs: mpl.colors.Colormap,
                  label: Optional[str] = None,
                  ticks: Optional[Sequence[float]] = None,
                  nticks: int = 5,
                  vmin: Optional[float] = None,
                  vmax: Optional[float] = None,
                  a: float = 1000,
                  ticklabels: Optional[Sequence[str]] = None,
                  tickstretch: Optional[str] = None,
                  orientation: str = 'vertical',
                  labelpad: float = 0,
                  lines: Optional[Plot] = None,
                  label_cbar2: Optional[str] = None,
                  ticks_cbar2: Optional[Sequence[float]] = None,
                  norm_cbar2: Optional['Normalization'] = None,
                  ) -> Optional[mpl.colorbar.Colorbar]:
        """Plot color bar.

        If ticks are not given, they will be determined from the other
        parameters (nticks, vmin, vmax, a, stretch, etc.) or use the defaults
        from matplotlib.

        Args:
          fig: figure object.
          cs: color map.
          label: optional; color bar label.
          ticks: optional; color bar ticks.
          nticks: optional; number of ticks for auto ticks.
          vmin: optional; lower limit for auto ticks.
          vmax: optional; upper limit for auto ticks.
          a: optional; scaling for log stretch for auto ticks.
          ticklabels: optional; tick labels.
          tickstretch: optional; stretch for the ticks.
          orientation: optional; color bar orientation.
          labelpad: optional; shift the color bar label.
          lines: optional; lines from contour plot to overplot.
          label_cbar2: optional; label of the second axis of the color bar.
          ticks_cbar2: optional; ticks of the second axis of the color bar.
        """
        # Check if cbax exists
        if self.cbaxis is None:
            self._log.warn('Skipping color bar')
            return None
        self._log.info('Plotting color bar:')

        # Ticks
        if ticks is None and vmin and vmax:
            ticks = get_colorbar_ticks(vmin, vmax, a=a, n=nticks,
                                       stretch=tickstretch)
        self._log.info('Tick values: %s', ticks)

        # Create bar
        cbar = fig.colorbar(cs, ax=self.axis, cax=self.cbaxis,
                            orientation=orientation, drawedges=False,
                            ticks=ticks)
        if lines is not None:
            cbar.add_lines(lines)

        # Bar position
        if orientation == 'vertical':
            cbar.ax.yaxis.set_ticks_position('right')
            if ticklabels is not None:
                cbar.ax.yaxis.set_ticklabels(ticklabels)
                self._log.info('Tick labels: %s', ticklabels)
        elif orientation == 'horizontal':
            cbar.ax.xaxis.set_ticks_position('top')
            if ticklabels is not None:
                cbar.ax.xaxis.set_ticklabels(ticklabels)
                self._log.info('Tick labels: %s', ticklabels)

        # Label
        if label is not None:
            if orientation == 'vertical':
                cbar.ax.yaxis.set_label_position('right')
                cbar.set_label(
                    label,
                    fontsize=self.ax.yaxis.get_label().get_fontsize(),
                    family=self.ax.yaxis.get_label().get_family(),
                    fontname=self.ax.yaxis.get_label().get_fontname(),
                    weight=self.ax.yaxis.get_label().get_weight(),
                    labelpad=labelpad,
                    verticalalignment='top')
            elif orientation == 'horizontal':
                cbar.ax.xaxis.set_label_position('top')
                cbar.set_label(
                    label,
                    fontsize=self.ax.xaxis.get_label().get_fontsize(),
                    family=self.ax.xaxis.get_label().get_family(),
                    fontname=self.ax.xaxis.get_label().get_fontname(),
                    weight=self.ax.xaxis.get_label().get_weight(),
                    labelpad=labelpad)

        # Secondary bar axis
        if ticks_cbar2 is not None:
            self._log.info('Secondary bar ticks: %s', ticks_cbar2)
            vmin_cbar2 = np.min(ticks_cbar2)
            vmax_cbar2 = np.max(ticks_cbar2)
            if orientation=='vertical':
                cbar2 = cbar.ax.twinx()
                cbar2.set_ylim([vmin_cbar2, vmax_cbar2])
                if label_cbar2 is not None:
                    cbar2.yaxis.set_label_position('left')
                    cbar2.set_ylabel(label_cbar2, labelpad=labelpad)
                cbar2.yaxis.set_ticks(ticks_cbar2)
            else:
                cbar2 = cbar.ax.twiny()
                cbar2.set_xscale(cbar.ax.get_xscale(),
                                 functions=(norm_cbar2, norm_cbar2.inverse))
                cbar2.set_xlim([vmin_cbar2, vmax_cbar2])
                if label_cbar2 is not None:
                    cbar.ax.xaxis.set_label_position('bottom')
                    cbar2.xaxis.set_label_position('top')
                    cbar2.set_xlabel(label_cbar2, labelpad=labelpad)
                cbar2.xaxis.set_ticks(ticks_cbar2)

        # Font
        tlabels = (cbar.ax.xaxis.get_ticklabels(which='both') +
                   cbar.ax.yaxis.get_ticklabels(which='both'))
        for tlabel in tlabels:
            tlabel.set_fontsize(
                self.ax.xaxis.get_majorticklabels()[0].get_fontsize())
            tlabel.set_family(
                self.ax.xaxis.get_majorticklabels()[0].get_family())
            tlabel.set_fontname(
                self.ax.xaxis.get_majorticklabels()[0].get_fontname())

        return cbar

    def arrow(self,
              arrow: Union[str, Tuple[float]],
              arrowprops: Optional[dict] = None,
              color: Optional[str] = None,
              **kwargs) -> None:
        """Draw an arrow.

        An arrow can be specified by the following combiantions:

          - (PA,)
          - (x, y, PA)
          - (PA, length)
          - (x, y, PA, length)

        with PA the position angle (from axis y>0 towards axis x<0). The
        positions (x, y) are in axes fraction and their default value is
        (0.5, 0.5). Similarly, the default legth is 0.5, i.e. half the axis
        length.

        Args:
          arrow: arrow position, angle and/or legth.
          arrowprops: optional; arrow properties.
          color: optional; use the default arrowprops but replace the color.
          kwargs: other arrow properties. See `matplotlib.pyplot.annotate`.
        """
        # Check input
        if arrowprops is None:
            arrowprops = {'arrowstyle':'->', 'fc':'k', 'ec':'k', 'ls':'-',
                          'lw':2}

        # Arrow specified as PA or (x y PA) or (PA len) or (x y PA len)
        try:
            arrow = tuple(map(float, arrow.split()))
        except AttributeError:
            pass
        x0, y0, l = 0.5, 0.5, 0.5
        if len(arrow) == 1:
            pa = arrow[0]
        elif len(arrow) == 2:
            pa, l = arrow
        elif len(arrow) == 3:
            x0, y0, pa = arrow
        elif len(arrow) == 4:
            x0, y0, pa, l = arrow
        else:
            raise ValueError(f'Cannot configure arrow: {arrow}')
        l = l * 0.5
        pa = pa + 90.

        # Locations
        dx = l * np.cos(np.radians(pa))
        dy = l * np.sin(np.radians(pa))
        xy = (x0 + dx, y0 + dy)
        xytext = (x0 - dx, y0 - dy)

        # Check colors
        if color and color != arrowprops['fc']:
            arrowprops['fc'] = color
            arrowprops['ec'] = color
        elif color is None:
            color = arrowprops['fc']

        # Draw
        self.annotate('', xy=xy, xytext=xytext, xycoords='axes fraction',
                      arrowprops=arrowprops, color=color, **kwargs)

    def arc(self, *args, **kwargs):
        """Draw an arc."""
        # Arc specified as patches.Arc
        arcpatch = patches.Arc(*args, **kwargs)
        self.axis.add_patch(arcpatch)

    # Other utilities
    def label_axes(self,
                   text: str,
                   loc: Sequence[float] = (0.1, 0.9),
                   **kwargs) -> None:
        """Add a label for the axes.

        Args:
          text: label text.
          loc: label location.
          **kwargs: additional options for the annotate functions.
        """
        kwargs.setdefault('xycoords', 'axes fraction')
        self.annotate(text, xy=loc, xytext=loc, **kwargs)

class PhysPlotHandler(PlotHandler):
    """Plot handler to manage plots with astropy.quantity objects.

    Attributes:
      xunit: `astropy.Unit` for x axis.
      yunit: `astropy.Unit` for y axis.
    """

    def __init__(self,
                 axis: Axes,
                 cbaxis: Optional[Axes] = None,
                 xscale: str = 'linear',
                 yscale: str = 'linear',
                 xname: str = 'x',
                 yname: str = 'y',
                 xunit: u.Unit = u.Unit(1),
                 yunit: u.Unit = u.Unit(1)) -> None:
        """Create a single plot container."""
        super().__init__(axis, cbaxis=cbaxis, xscale=xscale, yscale=yscale,
                         xname=xname, yname=yname)
        self.xunit = xunit
        self.yunit = yunit

    def _simple_plt(self,
                    fn: Callable[[], Plot],
                    *args,
                    nphys_args: Optional[int] = None,
                    is_image: bool = False,
                    **kwargs) -> Plot:
        """Plot and assign label.

        It verifies that args are in the correct units of the corresponding
        axis. The parameter `nphys_args` can be used to specify the number of
        `args` that are physical quantities. If `args` consist of an image,
        then it is assumed that it is in the correct units and the `is_image`
        keyword can be used to skip the unit checking.
        
        Args:
          fn: plotting function.
          args: arguments for the function.
          nphys_args: optional; number of physical quantities.
          is_image: optional; is the input args a 2-D image?
          kwargs: keyword arguments for the function.
        """
        # Verify nphys_args
        if nphys_args is None:
            nphys_args = len(args)

        # Cases
        if is_image:
            phys_args = (args[0].value,)
        elif nphys_args == 1:
            phys_args = (args[0].to(self.yunit).value,)
        elif nphys_args == 2:
            phys_args = (args[0].to(self.xunit).value,
                         args[1].to(self.yunit).value)
        elif nphys_args == 3:
            phys_args = (args[0].to(self.xunit).value,
                         args[1].to(self.yunit).value,
                         args[2].to(self.yunit).value)
        else:
            raise ValueError('Could not convert values')

        # Non physical arguments
        non_phys_args = args[nphys_args:]
        fn_args = phys_args + non_phys_args

        return self.insert_plt(kwargs.get('label', None),
                               fn(*fn_args, **kwargs))

    @staticmethod
    def _check_unit(value: Union[u.Quantity, None],
                    unit: u.Unit) -> Union[u.Quantity, None]:
        """Convert the value to unit.

        Args:
          value: quantity to check.
          unit: unit to convert to.

        Returns:
          The value of the quantity value converted to unit or `None` if value
          is `"None"`.
        """
        if value is None: return None
        return value.to(unit).value

    # Getters
    def get_xlabel(self, unit_fmt: str = '({:latex_inline})') -> str:
        """Return the xlabel from stored values."""
        return super().get_xlabel(unit_fmt=unit_fmt)

    def get_ylabel(self, unit_fmt: str = '({:latex_inline})') -> str:
        """Return the ylabel from stored values."""
        return super().get_ylabel(unit_fmt=unit_fmt)

    # Setters
    def set_xlim(self, xmin: Optional[u.Quantity] = None,
                 xmax: Optional[u.Quantity] = None) -> None:
        """Set the axis x limits."""
        super().set_xlim(xmin=self._check_unit(xmin, self.xunit),
                         xmax=self._check_unit(xmax, self.xunit))

    def set_ylim(self, ymin: Optional[u.Quantity] = None,
                 ymax: Optional[u.Quantity] = None) -> None:
        """Set the axis y limits."""
        super().set_ylim(ymin=self._check_unit(ymin, self.yunit),
                         ymax=self._check_unit(ymax, self.yunit))
