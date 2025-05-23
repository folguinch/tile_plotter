"""Objects for handling all type of plots."""
from typing import (Optional, Tuple, TypeVar, Callable, Dict, List,
                    Union, Sequence, Mapping)
import dataclasses
import pathlib

from configparseradv.configparser import ConfigParserAdv
from matplotlib import patches
from toolkit.logger import get_logger
import astropy.units as u
import matplotlib as mpl
import matplotlib.transforms as transforms
#import matplotlib.lines as mlines
import numpy as np
import numpy.typing as npt

from .utils import tick_formatter
from .axes_props import AxesProps, PhysAxesProps, VScaleProps, PhysVScaleProps

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
      axes_props: properties of axes.
      vscale: instensity scale and color bar properties.
      pltd: plotted objects tracker.
      is_config: set to `True` when configuration has been applied.
      skeleton: base configuration.
    """
    # Common class attributes
    _log = get_logger(__name__, filename='plotter.log')
    _defconfig = (pathlib.Path(__file__).resolve().parent /
                  pathlib.Path('../configs/plot_default.cfg'))

    # Read default skeleton
    skeleton = ConfigParserAdv()
    skeleton.read(_defconfig)

    def __init__(self,
                 axis: Axes,
                 cbaxis: Optional[Axes] = None,
                 vscale: Optional[Mapping] = None,
                 **axes_props) -> None:
        """ Create a single plot container."""
        self.pltd = {}
        self.axis = axis
        self.cbaxis = cbaxis
        if axes_props:
            self.axes_props = AxesProps(**axes_props)
        else:
            self.axes_props = None
        if vscale is not None:
            self.vscale = VScaleProps(**vscale)
        else:
            self.vscale = None
        self.is_config = False

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
                    xticks: List[float] = None,
                    yticks: List[float] = None,
                    minor_xticks: List[float] = None,
                    minor_yticks: List[float] = None,
                    **axes_props) -> None:
        """Configure the plot.

        Configures several aspects of the axes: limits, scales, labels and
        ticks. It overrides and updates some of the default parameters.

        Args:
          xticks: optional; x ticks values.
          yticks: optional; y ticks values.
          minor_xticks: optional; minor x ticks values.
          minor_yticks: optional; minor y ticks values.
          axes_props: optional; any other axes property.
        """
        # Check config
        if self.is_config:
            self._log.warning('Plot already configured, skipping')
            return self.is_config

        # Updated version of axes_props
        if axes_props:
            self._log.info('Replacing plot configs: %s', axes_props)
            props = dataclasses.replace(self.axes_props, **axes_props)
        else:
            props = self.axes_props

        # Limits
        if props.xlim is not None:
            self._log.info('Setting xlim: %s', props.xlim)
            self.ax.set_xlim(**props.xlim)
        if props.ylim is not None:
            self._log.info('Setting ylim: %s', props.ylim)
            self.ax.set_ylim(**props.ylim)

        # Axis scales
        if props.xscale == 'log':
            self.ax.set_xscale('log')
            self.ax.xaxis.set_major_formatter(tick_formatter('log'))
        if props.yscale == 'log':
            self.ax.set_yscale('log')
            self.ax.yaxis.set_major_formatter(tick_formatter('log'))

        # Labels
        if props.set_xlabel:
            xlabel = props.xlabel
        else:
            xlabel = ''
        if props.set_ylabel:
            ylabel = props.ylabel
        else:
            ylabel = ''
        self.set_axlabels(xlabel=xlabel, ylabel=ylabel)

        # Ticks
        if xticks:
            self.ax.set_xticks(xticks)
            fmt = props.xticks_fmt
            self.ax.set_xticklabels([fmt.format(x) for x in xticks])
        elif not props.set_xticks:
            self.ax.set_xticklabels([''] * len(self.ax.get_xticks()))
        if minor_xticks:
            self.ax.set_xticks(minor_xticks, minor=True)
        if yticks:
            self.ax.set_yticks(yticks)
            fmt = props.yticks_fmt
            self.ax.set_yticklabels([fmt.format(y) for y in yticks])
        elif not props.set_yticks:
            self.ax.set_yticklabels([''] * len(self.ax.get_yticks()))
        if minor_yticks:
            self.ax.set_yticks(minor_yticks, minor=True)

        # Ticks colors
        self.ax.tick_params('both', color=props.ticks_color)

        # Update is_config
        self.is_config = True
        return self.is_config

    def set_axlabels(self, xlabel: Optional[str] = None,
                     ylabel: Optional[str] = None) -> None:
        """Set the axis labels.

        Args:
          xlabel: optional; x axis label.
          ylabel: optional; y axis label.
        """
        if xlabel == '':
            pass
        elif xlabel is not None:
            self.ax.set_xlabel(xlabel, labelpad=self.axes_props.label_xpad)
        elif self.axes_props.set_xlabel:
            self.ax.set_xlabel(self.axes_props.xlabel,
                               labelpad=self.axes_props.label_xpad)
        if ylabel == '':
            pass
        elif ylabel is not None:
            self.ax.set_ylabel(ylabel, labelpad=self.axes_props.label_ypad)
        elif self.axes_props.set_ylabel:
            self.ax.set_ylabel(self.axes_props.ylabel,
                               labelpad=self.axes_props.label_ypad)

    # Matplotlib plotting functions
    def plot(self, *args, **kwargs) -> Plot:
        """Plot on the axis.

        Arguments are the same as for the `matplotlib.pyplot.plot` function.

        Args:
          args: data to plot.
          kwargs: arguments for `matplotlib.pyplot.plot`.
        """
        return self._simple_plt(self.axis.plot, *args, **kwargs)

    def axhline(self, *args, **kwargs) -> None:
        """Plot horizontal line.

        Arguments are the same as for the `matplotlib.pyplot.axhline` function.
        """
        self.axis.axhline(*args,**kwargs)

    def axvline(self, *args, **kwargs) -> None:
        """Plot vertical line.

        Arguments are the same as for the `matplotlib.pyplot.axvline` function.
        """
        self.axis.axvline(*args,**kwargs)

    def axline(self, *args, **kwargs) -> None:
        """Plot a line from position and slope.

        Arguments are the same as for the `matplotlib.pyplot.axline` function.
        """
        self.axis.axline(*args, **kwargs)

    def axvspan(self, *args, **kwargs) -> None:
        """Axis vertical span."""
        self.axis.axvspan(*args, **kwargs)

    def axhspan(self, *args, **kwargs) -> None:
        """Axis horizontal span."""
        self.axis.axhspan(*args, **kwargs)

    def errorbar(self, *args, **kwargs) -> Plot:
        """Plot on the axis.

        Arguments are the same as for the `matplotlib.pyplot.errorbar` function.

        Args:
          args: data to plot.
          kwargs: arguments for `matplotlib.pyplot.errorbar`.
        """
        return self._simple_plt(self.axis.errorbar, *args, **kwargs)

    def title(self, text: str, **kwargs) -> None:
        """Set plot title."""
        self.axis.set_title(text, **kwargs)

    def annotate(self, *args, **kwargs) -> None:
        """Annotate the axis.

        Arguments are the same as for the `matplotlib.pyplot.annotate` function.

        Args:
          args: data to plot.
          kwargs: arguments for `matplotlib.pyplot.annotate`.
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
          loc: legend position. See `matplotlib.pyplot.lengend` documentation
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
        return self._simple_plt(self.axis.scatter, *args, **kwargs)

    def marker(self, *args, **kwargs) -> Plot:
        """Simple scatter plot.
        
        Similar to `scatter` but uses the plot function.
        """
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
                  orientation: str,
                  lines: Optional[Plot] = None,
                  compute_ticks: Optional[bool] = None,
                  **cbar_props
                  ) -> Optional[mpl.colorbar.Colorbar]:
        """Plot color bar.

        If ticks are not given, they will be determined from the other
        parameters (nticks, vmin, vmax, a, stretch, etc.) or use the defaults
        from matplotlib.

        When a second (clone) color bar axis is requested, the stored
        `equivalency` can be used to convert the values of the color bar axis
        ticks.

        Args:
          fig: figure object.
          cs: color map.
          orientation: color bar orientation.
          lines: optional; lines from contour plot to overplot.
          compute_ticks: optional; compute ticks or use default?
          cbar_props: optional; additional color bar properties.
        """
        # Check if cbax exists
        if self.cbaxis is None:
            self._log.warning('Skipping color bar')
            return None

        # Update color bar properties
        if self.vscale is not None and cbar_props:
            props = dataclasses.replace(self.vscale, **cbar_props)
        elif self.vscale is not None:
            props = self.vscale
        else:
            props = VScaleProps(**cbar_props)

        # Compute ticks or use default?
        if compute_ticks is None:
            compute_ticks = self.vscale.compute_ticks
        if (compute_ticks or props.ticks is not None):
            ticks, ticks_cbar2 = props.get_ticks()
            self._log.info('Color bar ticks: %s', props.ticks)
            self._log.info('Color bar 2 ticks: %s', props.ticks_cbar2)
        else:
            ticks = ticks_cbar2 = None

        # Create bar
        cbar = fig.colorbar(cs, ax=self.axis, cax=self.cbaxis,
                            orientation=orientation, drawedges=False,
                            ticks=ticks)
        if lines is not None:
            cbar.add_lines(lines)

        # Bar position
        if orientation == 'vertical':
            cbar.ax.yaxis.set_ticks_position('right')
            if props.ticklabels is not None:
                cbar.ax.yaxis.set_ticklabels(props.ticklabels)
                self._log.info('Tick labels: %s', props.ticklabels)
        elif orientation == 'horizontal':
            cbar.ax.xaxis.set_ticks_position('top')
            if props.ticklabels is not None:
                cbar.ax.xaxis.set_ticklabels(props.ticklabels)
                self._log.info('Tick labels: %s', props.ticklabels)

        # Label
        if props.label is not None:
            if orientation == 'vertical':
                cbar.ax.yaxis.set_label_position('right')
                cbar.set_label(
                    props.label,
                    fontsize=self.ax.yaxis.get_label().get_fontsize(),
                    family=self.ax.yaxis.get_label().get_family(),
                    fontname=self.ax.yaxis.get_label().get_fontname(),
                    weight=self.ax.yaxis.get_label().get_weight(),
                    labelpad=props.labelpad,
                    verticalalignment='top')
            elif orientation == 'horizontal':
                cbar.ax.xaxis.set_label_position('top')
                cbar.set_label(
                    props.label,
                    fontsize=self.ax.xaxis.get_label().get_fontsize(),
                    family=self.ax.xaxis.get_label().get_family(),
                    fontname=self.ax.xaxis.get_label().get_fontname(),
                    weight=self.ax.xaxis.get_label().get_weight(),
                    labelpad=props.labelpad)

        # Secondary bar axis
        if ticks_cbar2 is not None:
            #vmin_cbar2 = np.min(props.ticks_cbar2)
            #vmax_cbar2 = np.max(props.ticks_cbar2)
            #vmin_cbar2 = props.vmin2.value
            #vmax_cbar2 = props.vmax2.value
            if orientation == 'vertical':
                cbar2 = cbar.ax.twinx()
                cbar2.set_yscale(cbar.ax.get_yscale(),
                                 functions=(props.norm_cbar2,
                                            props.norm_cbar2.inverse))
                cbar2.set_ylim(*props.get_vlim(axis=2))
                if props.label_cbar2 is not None:
                    cbar2.yaxis.set_label_position('left')
                    cbar2.set_ylabel(props.label_cbar2,
                                     labelpad=props.labelpad_cbar2)
                cbar2.yaxis.set_ticks(ticks_cbar2)
            else:
                cbar2 = cbar.ax.twiny()
                cbar2.set_xscale(cbar.ax.get_xscale(),
                                 functions=(props.norm_cbar2,
                                            props.norm_cbar2.inverse))
                cbar2.set_xlim(*props.get_vlim(axis=2))
                if props.label_cbar2 is not None:
                    cbar.ax.xaxis.set_label_position('bottom')
                    cbar2.xaxis.set_label_position('top')
                    cbar2.set_xlabel(props.label_cbar2,
                                     labelpad=props.labelpad_cbar2)
                cbar2.xaxis.set_ticks(ticks_cbar2)

        # Font
        #tlabels = (cbar.ax.xaxis.get_ticklabels(which='both') +
        #           cbar.ax.yaxis.get_ticklabels(which='both'))
        #for tlabel in tlabels:
        #    tlabel.set_fontsize(
        #        self.ax.xaxis.get_majorticklabels()[0].get_fontsize())
        #    tlabel.set_family(
        #        self.ax.xaxis.get_majorticklabels()[0].get_family())
        #    tlabel.set_fontname(
        #        self.ax.xaxis.get_majorticklabels()[0].get_fontname())

        return cbar

    def arrow(self,
              arrow: Union[str, Tuple[float]],
              arrowprops: Optional[dict] = None,
              origin: str = 'mid',
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
        (0.5, 0.5). Similarly, the default length is 0.5, i.e. half the axis
        length.

        Args:
          arrow: Arrow position, angle and/or legth.
          arrowprops: Optional. Arrow properties.
          origin: Optional. Arrow origin (`mid` or `origin`).
          color: Optional. Use the default arrowprops but replace the color.
          kwargs: Other arrow properties. See `matplotlib.pyplot.annotate`.
        """
        # Check input
        if arrowprops is None:
            lw = kwargs.pop('linewidth', kwargs.pop('lw', 2))
            arrowprops = {'arrowstyle':'->', 'fc':'k', 'ec':'k', 'ls':'-',
                          'lw':float(lw)}

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
        if origin == 'mid':
            xy = (x0 + dx, y0 + dy)
            xytext = (x0 - dx, y0 - dy)
        elif origin == 'origin':
            xy = (x0 + 2*dx, y0 + 2*dy)
            xytext = (x0, y0)
        else:
            raise NotImplementedError(f'Arrow origin {origin} not implemented')

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

    def ellipse(self, *args, **kwargs):
        """Draw an ellipse."""
        ellipse_patch = patches.Ellipse(*args, **kwargs)
        self.axis.add_patch(ellipse_patch)

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
      axis: axes of the plot (alias ax).
      cbaxis: color bar axes (alias cbax).
      axes_props: properties of axes.
      vscale: instensity scale and color bar properties.
      pltd: plotted objects tracker.
      skeleton: base configuration.
    """

    def __init__(self,
                 axis: Axes,
                 cbaxis: Optional[Axes] = None,
                 vscale: Optional[Mapping] = None,
                 **axes_props) -> None:
        """Create a single plot container."""
        super().__init__(axis, cbaxis=cbaxis)
        if axes_props:
            self.axes_props = PhysAxesProps(**axes_props)
        else:
            self.axes_props = None
        if vscale:
            self.vscale = PhysVScaleProps(**vscale)
        else:
            self.vscale = None

    @classmethod
    def from_config(cls,
                    config: ConfigParserAdv,
                    axis: Axes,
                    cbaxis: Axes,
                    **kwargs):
        """Create a new `PhysPlotHandler` from a config proxy.

        Args:
          config: configuration parser proxy.
          axis: matplotlib axis.
          cbaxis: matplotlib color bar axis.
          kwargs: replace values in the config.
        """
        # Get axes properties
        axes_props = {}
        for opt, val in cls.skeleton.items('axes_props'):
            if 'unit' in opt:
                value = config.get(opt, vars=kwargs, fallback=val)
                try:
                    value = u.Unit(value)
                except ValueError:
                    value = None
            elif opt.startswith('set_') or opt.startswith('invert'):
                fallback = cls.skeleton.getboolean('axes_props', opt)
                value = config.getboolean(opt, vars=kwargs, fallback=fallback)
            elif opt in ['label_xpad', 'label_ypad']:
                value = config.getfloat(opt, vars=kwargs, fallback=float(val))
            elif opt in ['xlim', 'ylim']:
                value = config.getquantity(opt, vars=kwargs, fallback=None)
            else:
                value = config.get(opt, vars=kwargs, fallback=val)
            axes_props[opt] = value

        # Get vscale
        vscale = {'stretch': cls.skeleton['vscale']['stretch']}
        for opt in cls.skeleton.options('vscale'):
            if opt not in config and opt not in kwargs:
                continue
            val = config.get(opt, vars=kwargs)
            if 'unit' in opt:
                try:
                    val = u.Unit(val)
                except ValueError:
                    val = None
            elif 'name' in opt or 'stretch' in opt:
                pass
            elif opt in ['ticks', 'ticklabels']:
                val = val.split()
                if opt == 'ticks':
                    val = np.array(list(map(float, val[:-1]))) * u.Unit(val[-1])
            elif opt in ['nticks']:
                val = int(val)
            elif opt in ['compute_ticks']:
                val = config.getboolean(opt, vars=kwargs)
            else:
                try:
                    split_val = val.split()
                    val = float(split_val[0]) * u.Unit(split_val[1])
                except ValueError:
                    pass
                except IndexError:
                    val = float(val)
            vscale[opt] = val

        return cls(axis, cbaxis=cbaxis, vscale=vscale, **axes_props)

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
        xunit = self.axes_props.xunit
        yunit = self.axes_props.yunit
        if is_image:
            phys_args = (args[0].value,)
        elif nphys_args == 1:
            phys_args = (args[0].to(yunit).value,)
        elif nphys_args == 2:
            phys_args = tuple()
            for xarg, yarg in zip(args[0::2], args[1::2]):
                try:
                    phys_args += (xarg.to(xunit).value,
                                  yarg.to(yunit).value)
                except AttributeError:
                    break
        elif nphys_args == 3:
            phys_args = (args[0].to(xunit).value,
                         args[1].to(yunit).value,
                         args[2].to(yunit).value)
        else:
            raise ValueError('Could not convert values')

        # Non physical arguments
        #non_phys_args = args[nphys_args:]
        non_phys_args = args[len(phys_args):]
        fn_args = phys_args + non_phys_args

        return self.insert_plt(kwargs.get('label', None),
                               fn(*fn_args, **kwargs))

    def plot(self, *args, **kwargs) -> Plot:
        return self._simple_plt(self.axis.plot, *args, nphys_args=2, **kwargs)

    def plot_array(self,
                   data: Union[npt.ArrayLike, Tuple[npt.ArrayLike, Dict]],
                   config: Optional[ConfigParserAdv] = None,
                   usecols: Optional[Tuple[int,int]] = None,
                   **kwargs) -> Plot:
        """Plot numpy arrays or structured arrays with units.

        Args:
          data: A Quantity array or a tuple with an structured array and units.
          config: Optional. Configuration parser proxy.
          usecols: Optional. Columns to plot.
          kwargs: Optional. Additional arguments for plot.
        """
        # Extract data
        if hasattr(data, 'shape'):
            if usecols is None:
                colx, coly = 0, 1
            x, y = data[:,colx], data[:,coly]
        else:
            if usecols is not None:
                xkey, ykey = usecols
            elif config is not None:
                keys = config.get('plot_keys').split(',')
                xkey = keys[0].strip()
                ykey = keys[1].strip()
            else:
                raise ValueError('Cannot determine columns')
            x, y = data[0][xkey] * data[1][xkey], data[0][ykey] * data[1][ykey]

        # Configure plot
        linestyle = config.get('linestyle', vars=kwargs, fallback='-')
        color = config.get('color', vars=kwargs, fallback='r')
        marker = config.get('marker', vars=kwargs, fallback='')
        kwargs.setdefault('linestyle', linestyle)
        kwargs.setdefault('marker', marker)
        kwargs.setdefault('color', color)

        return self.plot(x, y, **kwargs)

    def draw_scale(self,
                   x0: u.Quantity,
                   y0: u.Quantity,
                   size: u.Quantity,
                   axis: str = 'x',
                   unit: Optional[u.Unit] = None,
                   distance: Optional[u.Quantity] = None,
                   offset: u.Quantity = None,
                   **kwargs):
        """Draw scale along axis.

        For a scale along the `x` axis, the given `offset` should be in units
        of the `y` axis, and viceversa.

        Args:
          x0: X-axis origin point.
          y0: Y-axis origin point.
          size: Size of the scale.
          axis: Optional. Scale direction.
          unit: Optional. Unit of the scale label.
          distance: Optional. Source distance to convert arcsec to au.
          offset: Optional. Label offset in data units.
          kwargs: 
        """
        # Check units
        kwargs_label = {}
        xaxis_unit = self.axes_props.xunit
        yaxis_unit = self.axes_props.yunit
        if axis == 'x':
            xval = [x0.to(xaxis_unit).value, (x0 + size).to(xaxis_unit).value]
            yval = [y0.to(yaxis_unit).value] * 2
            kwargs_label['verticalalignment'] = kwargs.pop('verticalalignment',
                                                           'bottom')
            dx = 0
            if offset is not None:
                dy = offset.to(yaxis_unit).value
            else:
                dy = 0
        else:
            xval = [x0.to(xaxis_unit).value] * 2
            yval = [y0.to(yaxis_unit).value, (y0 + size).to(yaxis_unit).value]
            kwargs_label['horizontalalignment'] = kwargs.pop(
                'horizontalalignment',
                'right',
            )
            dy = 0
            if offset is not None:
                dx = offset.to(xaxis_unit).value
            else:
                dx = 0

        # Plotting scale
        color = kwargs.get('color', 'w')
        zorder = kwargs.get('zorder', 10)
        self._log.info('Scale length: %s', size)
        self.plot(xval, yval, color=color, ls='-', lw=1, marker='_',
                  zorder=zorder)

        # Annotate
        if unit is not None:
            if size.unit.is_equivalent(u.arcsec) and unit.is_equivalent(u.pc):
                if distance is None:
                    self._log.warn('Ignoring scale unit conversion')
                    pass
                else:
                    fsize = size.to(u.arcsec).value * distance.to(u.pc).value
                    fsize = fsize * u.au
                    fsize = fsize.to(unit)
            else:
                fsize = size.to(unit)
        else:
            fsize = size
        label = f'{fsize.value} {fsize.unit:latex_inline}  '.lower()
        xycoords = 'data'
        self.annotate(label,
                      xy=[xval[0]+dx, yval[0]+dy],
                      xytext=[xval[0]+dx, yval[0]+xy],
                      color=color,
                      xycoords=xycoords,
                      zorder=zorder,
                      transform=label_transform,
                      **kwargs_label)

