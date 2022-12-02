"""Objects for plotting images."""
from typing import (Sequence, Optional, TypeVar, Union, Tuple, Dict, Mapping,
                    List)
import pathlib

from configparseradv.configparser import ConfigParserAdv
from matplotlib.patches import Ellipse
from matplotlib import cm
from radio_beam import Beam
import astropy.units as u
import astropy.wcs as apy_wcs
import matplotlib as mpl
import numpy as np

from .base_plotter import BasePlotter
from .plot_handler import PhysPlotHandler
from .utils import (get_artist_properties, auto_vmin, auto_vmax,
                    auto_levels, get_extent)

# Type aliases
Axes = TypeVar('Axes')
Artist = TypeVar('Artist')
Limits = Union[None, Sequence[float], Dict[str, float]]
Location = Tuple[int, int]
Plot = TypeVar('Plot')
Projection = TypeVar('Projection', apy_wcs.WCS, str)
Map = TypeVar('Map', u.Quantity, 'astropy.io.PrimaryHDU')
Position = TypeVar('Position', 'astroppy.SkyCoord', u.Quantity,
                   Tuple[u.Quantity, u.Quantity])

def filter_config_data(config: ConfigParserAdv, keys: Sequence,
                       ignore: Sequence = ('image', 'contour')) -> Dict:
    """Filter the data options in `config` present in `skeleton`.

    Args:
      config: input configuration.
      keys: available keys.
      ignore: optional; keys to ignore.

    Returns:
      A dictionary with the filtered values.
    """
    options = {}
    for key in keys:
        if key not in config or key in ignore:
            continue
        if key in ['center']:
            options[key] = config.getskycoord(key)
        elif key in ['radius', 'rms', 'levels']:
            options[key] = config.getquantity(key)
        elif key in ['self_contours', 'use_extent']:
            options[key] = config.getboolean(key)
        elif key in ['nsigma', 'negative_nsigma', 'nsigma_level',
                     'contour_linewidth']:
            options[key] = config.getfloat(key)
        else:
            aux = config.get(key).split()
            if len(aux) == 1:
                options[key] = aux[0]
            else:
                options[key] = aux

    return options

class MapHandler(PhysPlotHandler):
    """Handler for a map single plot.

    Attributes:
      im: Object from `plt.imshow`.
      axes_props: axes properties.
      vscale: intensity scale parameters.
      artists: artists to plot.
      radesys: projection system.
      skeleton: skeleton for the configuration options.
    """
    # Common class attributes
    _defconfig = (pathlib.Path(__file__).resolve().parent /
                  pathlib.Path('configs/map_default.cfg'))

    # Read default skeleton
    skeleton = ConfigParserAdv()
    skeleton.read(_defconfig)

    def __init__(self,
                 axis: Axes,
                 cbaxis: Optional[Axes] = None,
                 radesys: Optional[str] = None,
                 axes_props: Optional[Mapping] = None,
                 vscale: Optional[Mapping] = None,
                 #cbar_props: Mapping = {},
                 artists: Optional[Mapping] = None):
        """Initiate a map plot handler."""
        # Initialize
        super().__init__(axis, cbaxis=cbaxis, vscale=vscale,
                         xname=axes_props.pop('xname', 'R.A.'),
                         yname=axes_props.pop('yname', 'Dec.'),
                         xunit=axes_props.pop('xunit', u.deg),
                         yunit=axes_props.pop('yunit', u.deg),
                         **axes_props)
        self._log.info('Setting bunit (%s): %s',
                       self.vscale.name, self.vscale.unit)

        # Units and names
        self._log.info('Creating map plot')
        self.im = None

        # Store artists
        self.artists = artists

        # Projection system
        if radesys is not None:
            self.radesys = radesys.lower()
            if radesys.upper() == 'J2000':
                self.radesys = 'fk5'
            self._log.info('Setting RADESYS: %s', self.radesys)
        else:
            self.radesys = None

    @classmethod
    def from_config(cls,
                    config: ConfigParserAdv,
                    axis: Axes,
                    cbaxis: Axes,
                    radesys: Optional[str] = None,
                    **kwargs):
        """Create a new MapHandler from a config proxy.

        Default `units` are taken from the class `skeleton`.

        Args:
          config: configuration parser proxy.
          axis: matplotlib axis.
          cbaxis: matplotlib color bar axis.
          radesys: optional; projection system.
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

        # Artists
        artists = {}
        for opt in cls.skeleton.options('artists'):
            if opt not in config and opt not in kwargs:
                continue
            artists[opt] = get_artist_properties(opt, config)

        return cls(axis, cbaxis, radesys=radesys, axes_props=axes_props,
                   vscale=vscale, artists=artists)

    @property
    def vmin(self):
        return self.vscale.vmin

    @property
    def vmax(self):
        return self.vscale.vmax

    @property
    def vcenter(self):
        return self.vscale.vcenter

    @property
    def stretch(self):
        return self.vscale.stretch

    def _validate_data(self, data: Map,
                       wcs: apy_wcs.WCS,
                       ignore_units: bool = False
                       ) -> Tuple[u.Quantity, apy_wcs.WCS]:
        """Validate input data.

        Convert input data to a quantity with the same unit as `self.bunit` or
        determines `self.bunit` from header. If the unit of the data cannot be
        determined it is assumed to be in the same units as `self.bunit` or is
        considered dimensionless.

        Args:
          data: input data.
          wcs: WCS of the data.
          ignore_units: optional; ignore data units?
        """
        if hasattr(data, 'unit'):
            # Check bunit
            if not ignore_units and self.vscale.unit is None:
                self.vscale.set_unit(data.unit)
                self._log.info('Setting bunit to data unit: %s', data.unit)
            elif not ignore_units:
                valdata = data.to(self.vscale.unit)
            else:
                valdata = data
        elif hasattr(data, 'header'):
            # Check bunit
            bunit = u.Unit(data.header.get('BUNIT', 1))
            valdata = np.squeeze(data.data) * bunit
            if not ignore_units and self.vscale.unit is None:
                self.vscale.set_unit(bunit)
                self._log.info('Setting bunit to header unit: %s', bunit)
            elif not ignore_units:
                self._log.info('Converting data unit: %s -> %s',
                               valdata.unit, self.vscale.unit)
                valdata = valdata.to(self.vscale.unit)
            else:
                pass

            # Check wcs
            if wcs is None:
                wcs = apy_wcs.WCS(data.header, naxis=2)

            # Set RADESYS
            if self.radesys is None:
                try:
                    self.radesys = data.header['RADESYS'].lower()
                    if self.radesys.upper() == 'J2000':
                        self.radesys = 'fk5'
                    self._log.info('Setting RADESYS: %s', self.radesys)
                except KeyError:
                    self._log.warning('Map does not have RADESYS')
                    self.radesys = ''
        elif not ignore_units:
            if self.vscale.unit is None:
                self._log.warning('Setting data as dimensionless')
                self.vscale.unit = u.Unit(1)
            else:
                self._log.warning('Assuming data in %s units', self.vscale.unit)
            valdata = data * self.vscale.unit
        else:
            self._log.warning('Setting data as dimensionless')
            valdata = data * u.Unit(1)

        # Check wcs
        if wcs is None:
            self._log.warning('WCS is None')

        return valdata, wcs

    def _validate_vscale(self,
                         data: Map,
                         rms: Optional[u.Quantity] = None) -> None:
        """Validate and set vscale.

        Args:
          data: to determine the intensity limits.
          rms: optional; use this rms to determine the intensity limits.
        """
        if self.vmin is None:
            self.vscale.vmin = auto_vmin(data, rms=rms, log=self._log.info)
            self._log.info('Setting vmin from data: %s', self.vscale.vmin)
        if self.vmax is None:
            self.vscale.vmax = auto_vmax(data)
            self._log.info('Setting vmax from data: %s', self.vscale.vmax)
        self.vscale.check_scale_units()

    def _validate_extent(self,
                         extent: Tuple[u.Quantity, u.Quantity,
                                       u.Quantity, u.Quantity],
                         ) -> Tuple[float, float, float, float]:
        """Validate extent values.

        Convert the values in a extent sequence to the respective axis units
        and returns the quantity values.
        """
        xextent = [ext.to(self.axes_props.xunit).value for ext in extent[:2]]
        yextent = [ext.to(self.axes_props.yunit).value for ext in extent[2:]]
        self._log.info('Axes extent: %s, %s', xextent, yextent)

        return tuple(xextent + yextent)

    # Plotters
    def plot_map(self,
                 data: Map,
                 wcs: Optional[apy_wcs.WCS] = None,
                 extent: Optional[Tuple[u.Quantity]] = None,
                 use_extent: bool = False,
                 rms: Optional[u.Quantity] = None,
                 mask_bad: bool = False,
                 mask_color: str = 'w',
                 position: Optional['astropy.SkyCoord'] = None,
                 radius: Optional[u.Quantity] = None,
                 shift_data: Optional[u.Quantity] = None,
                 self_contours: bool = False,
                 contour_levels: Optional[List[u.Quantity]] = None,
                 contour_colors: str = 'w',
                 contour_linewidths: Optional[float] = None,
                 contour_nsigma: float = 5.,
                 contour_negative_nsigma: Optional[float] = None,
                 contour_nsigmalevel: Optional[float] = None,
                 **kwargs) -> None:
        """Plot image data.

        The map can be recenter and zoomed using the `position` and `radius`
        parameters if the data `wcs` can be determined or is given.

        Contour keywords are used only if `self_contours` are requested.

        Args:
          data: input map.
          wcs: optional; WCS object of the map.
          extent: optional; `xy` range of the data (left, right, bottom, top).
          use_extent: optional; determine extent from data and ignore wcs.
          rms: optional; map noise level.
          mask_bad: optional; mask bad/null pixels?
          mask_color: optional; color for bad/null pixels.
          position: optional; center of the map.
          radius: optional; radius of the region shown.
          shift_data: optional; an additive constant to apply to the data.
          self_contours: optional; plot contours of the data too?
          contour_levels: optional; self contour levels.
          contour_colors: optional; self contour colors.
          contour_linewidths: optional; self contours line width.
          contour_nsigma: optional; level of the lowest contour.
          contour_negative_nsigma: optional; level of the highest negative
            contour.
          contour_nsigmalevel: optional; plot only one contour at this level
            times the rms value.
          **kwargs: optional; additional arguments for `pyplot.imshow`.
        """
        # Validate data: valdata is a quantity with self.bunit units
        valdata, valwcs = self._validate_data(data, wcs)

        # Shift data
        if shift_data is not None:
            valdata = valdata + shift_data.to(valdata.unit)

        # Check vscale and get normalization
        self._validate_vscale(valdata, rms=rms)
        norm = self.vscale.get_normalization()

        # Check extent
        if extent is not None:
            extent_val = self._validate_extent(extent)
            self._log.info('Validated extent: %s', extent_val)
            valwcs = None
        elif extent is None and use_extent:
            extent = get_extent(data, wcs=wcs)
            self._log.info('Extent from data: %s', extent)
            extent_val = self._validate_extent(extent)
            self._log.info('Validated extent: %s', extent_val)
            valwcs = None
        else:
            extent_val = None

        # Check wcs and re-center the image
        if valwcs is not None and radius is not None and position is not None:
            self.recenter(radius, position, valwcs)
        elif extent_val is not None:
            self.ax.set_xlim(*extent_val[:2])
            self.ax.set_ylim(*extent_val[2:])

        # Plot data
        zorder = kwargs.setdefault('zorder', 1)
        if mask_bad:
            cmap = cm.get_cmap()
            cmap.set_bad(mask_color, 1.0)
            maskdata = np.ma.array(valdata.value, mask=np.isnan(data))
            self.im = self.ax.imshow(maskdata,
                                     norm=norm,
                                     extent=extent_val,
                                     **kwargs)
        else:
            self.im = self.ax.imshow(valdata.value,
                                     norm=norm,
                                     extent=extent_val,
                                     **kwargs)

        # Plot self contours
        if self_contours:
            self.plot_contours(valdata,
                               wcs=valwcs,
                               extent=extent,
                               rms=rms,
                               levels=contour_levels,
                               colors=contour_colors,
                               nsigma=contour_nsigma,
                               negative_nsigma=contour_negative_nsigma,
                               nsigmalevel=contour_nsigmalevel,
                               linewidths=contour_linewidths,
                               zorder=zorder+1)

    def plot_contours(self,
                      data: Map,
                      wcs: Optional[apy_wcs.WCS] = None,
                      extent: Optional[Tuple[u.Quantity]] = None,
                      use_extent: bool = False,
                      rms: Optional[u.Quantity] = None,
                      levels: Optional[List[u.Quantity]] = None,
                      colors: Optional[Sequence[str]] = None,
                      nsigma: float = 5.,
                      negative_nsigma: Optional[float] = None,
                      nsigmalevel: Optional[float] = None,
                      ignore_units: bool = False,
                      **kwargs):
        """Plot a contour map.

        Args:
          data: input map.
          wcs: optional; WCS object of the map.
          extent: optional; `xy` range of the data (left, right, bottom, top).
          use_extent: optional; determine extent from data and ignore wcs.
          rms: optional; map noise level.
          levels: optional; contour levels.
          colors: optional; contour colors.
          nsigma: optional; level of the lowest contour over rms.
          negative_nsigma: optional; level of the highest negative contour.
          nsigmalevel: optional; plot only one contour at this level times the
            rms value.
          ignore_units: optional; ignore data units?
          **kwargs: optional; additional arguments for `pyplot.contours`.
        """
        # Validate data
        valdata, valwcs = self._validate_data(data, wcs,
                                              ignore_units=ignore_units)

        # Check extent
        if extent is not None:
            extent_val = self._validate_extent(extent)
            self._log.info('Contour validated extent: %s', extent_val)
        elif extent is None and use_extent:
            extent = get_extent(data, wcs=wcs)
            self._log.info('Contour extent from data: %s', extent)
            extent_val = self._validate_extent(extent)
            self._log.info('Contour validated extent: %s', extent_val)
        else:
            extent_val = None

        # Levels
        if levels is None:
            try:
                if nsigmalevel is not None:
                    nlevels = 1
                else:
                    nlevels = None
                levels = auto_levels(data=valdata,
                                     rms=rms,
                                     nsigma=nsigma,
                                     nsigmalevel=nsigmalevel,
                                     nlevels=nlevels,
                                     negative_nsigma=negative_nsigma,
                                     log=self._log.info)
                levels_val = levels.to(valdata.unit).value
            except ValueError:
                return None
        else:
            levels_val = levels.to(valdata.unit).value

        # Color map
        if 'cmap' not in kwargs:
            kwargs['colors'] = colors or self.skeleton.get('data',
                                                           'contour_colors')
        elif 'norm' not in kwargs:
            kwargs['norm'] = self.vscale.get_normalization()
        else:
            pass

        # Plot
        kwargs.setdefault('zorder', 0)
        if valwcs is not None and extent_val is None:
            return super().contour(valdata,
                                   is_image=True,
                                   levels=levels_val,
                                   transform=self.ax.get_transform(valwcs),
                                   **kwargs)
        else:
            return super().contour(valdata,
                                   is_image=True,
                                   levels=levels_val,
                                   extent=extent_val,
                                   **kwargs)

    def _plot_artist(self, artist: str) -> None:
        """Plot each value of the artist"""
        for position, props in zip(self.artists[artist]['positions'],
                                   self.artists[artist]['properties']):
            if self.radesys:
                pos = position.transform_to(self.radesys)
            if artist == 'scatters':
                self.scatter(pos.ra, pos.dec, **props)
            elif artist == 'texts':
                text = props.pop('text')
                self.text(pos.ra, pos.dec, text, nphys_args=2, **props)
            elif artist == 'arrows':
                self.arrow(pos.ra, pos.dec, **props)
            elif artist == 'hlines':
                pos = position.to(self.axes_props.yunit)
                self.axhline(pos.value, **props)
            elif artist == 'vlines':
                pos = position.to(self.axes_props.xunit)
                self.axvline(pos.value, **props)
            elif artist == 'axlines':
                pos = (position.ra.to(self.axes_props.xunit).value,
                       position.dec.to(self.axes_props.yunit).value)
                slope = props.pop('slope')
                slope = slope.to(self.axes_props.yunit/self.axes_props.xunit)
                self.axvline(pos, slope=slope.value, **props)

    def plot_artists(self) -> None:
        """Plot all the stored artists."""
        for artist in self.artists:
            self._plot_artist(artist)
        #        elif artist=='arcs':
        #            width = cfg.getvalue('%s_width' % artist, n=i, dtype=float)
        #            height = cfg.getvalue('%s_height' % artist, n=i,
        #                    dtype=float)
        #            if width is None or height is None:
        #                print('Arc artist requires width and height')
        #                continue
        #            kwargs = {'angle':0.0, 'theta1':0.0, 'theta2':360.0,
        #                    'linewidth':1, 'linestyle':'-'}
        #            textopts = ['linestyle']
        #            for opt in kwargs:
        #                if opt in textopts:
        #                    kwargs[opt] = cfg.getvalue('%s_%s' % (artist, opt),
        #                            n=i, fallback=kwargs[opt])
        #                    continue
        #                kwargs[opt] = cfg.getvalue('%s_%s' % (artist, opt),
        #                        n=i, fallback=kwargs[opt], dtype=float)
        #            mk = self.arc((art.ra.degree, art.dec.degree), width,
        #                    height, color=color, zorder=zorder,
        #                    transform=self.ax.get_transform('world'), **kwargs)

    def plot_cbar(self,
                  fig: 'Figure',
                  orientation: str,
                  lines: Optional[Plot] = None,
                  ) -> Optional[mpl.colorbar.Colorbar]:
        """Plot the color bar.

        If ticks are not given, they will be determined from the other
        parameters (nticks, vmin, vmax, a, stretch, etc.) or use the defaults
        from matplotlib.

        When a second (clone) color bar axis is requested, the `equivalency`
        argument can be used to convert the values of the color bar axis ticks.

        Args:
          fig: figure object.
          orientation: orientation of the color bar.
          lines: optional; lines from contour plot to overplot.
        """
        return super().plot_cbar(fig, self.im, orientation, lines=lines)

    def plot_beam(self,
                  header: Mapping,
                  beam: Optional[Beam] = None,
                  color: str = 'k',
                  dx: float = 1.,
                  dy: float = 1.,
                  pad: float = 2.,
                  **kwargs) -> None:
        """Plot radio beam.

        Args:
          header: data header to read pixel size and beam.
          beam: optional; replaces the values from header.
          color: optional; beam color.
          dx: optional; x-axis shift.
          dy: optional; y-axis shift.
          pad: optional; shift for both axes.
          kwargs: optional keywords for `matplotlib.patches.Ellipse`.
        """
        # Get wcs
        wcs = apy_wcs.WCS(header, naxis=2)

        # Beam
        if beam is None:
            beam = Beam.from_fits_header(header)
        self._log.info('Plotting %s', beam)

        # Store beam equivalency
        if self.vscale.unit_equiv is None:
            try:
                self.brightness_temperature(header, beam)
            except KeyError:
                pass
        else:
            self._log.warning('Bunit equivalency already stored')

        # Convert to pixel
        pixsize = np.sqrt(wcs.proj_plane_pixel_area())
        bmaj = beam.major.cgs / pixsize.cgs
        bmin = beam.minor.cgs / pixsize.cgs

        # Define position
        xmin, _ = self.ax.get_xlim()
        ymin, _ = self.ax.get_ylim()
        xmin += dx
        ymin += dy
        size = bmaj + pad

        # Plot beam
        #rect = Rectangle((xmin,ymin), size, size, fill=True, fc='w', zorder=3)
        kwargs.setdefault('zorder', 4)
        beam = Ellipse((xmin + size/2., ymin + size/2.), bmin.value, bmaj.value,
                       angle=beam.pa.to(u.deg).value, fc=color,
                       transform=self.ax.get_transform(wcs), **kwargs)
        #ax.add_patch(rect)
        self.ax.add_patch(beam)

    # Configurations
    def recenter(self,
                 r: u.Quantity,
                 position: 'astroppy.SkyCoord',
                 wcs: apy_wcs.WCS) -> None:
        """Recenter and zoom the plot.

        Args:
          r: radius of the region shown.
          position: center of the map.
          wcs: WCS object of the map.
        """
        self._log.info('Recentering to: %s', position)
        x, y = wcs.world_to_pixel(position)
        cdelt = np.sqrt(wcs.proj_plane_pixel_area())
        self._log.info('Pixel increment: %s', cdelt.to(u.arcsec))
        if hasattr(r, 'unit'):
            radius = r.to(u.deg) / cdelt.to(u.deg)
        else:
            radius = r*u.deg / cdelt.to(u.deg)
        self._log.info('Zoom radius: %.1f pixels', radius)
        self.ax.set_xlim(x-radius.value, x+radius.value)
        self.ax.set_ylim(y-radius.value, y+radius.value)

    def config_map(self,
                   set_xlabel: Optional[bool] = None,
                   set_ylabel: Optional[bool] = None,
                   set_xticks: Optional[bool] = None,
                   set_yticks: Optional[bool] = None,
                   xcoord: str = 'ra',
                   ycoord: str = 'dec') -> None:
        """Apply configuration to map axes.

        Args:
          set_xlabel: optional; display `x` axis label?
          set_ylabel: optional; display `y` axis label?
          set_xticks: optional; display `x` axis tick labels?
          set_yticks: optional; display `y` axis tick labels?
          xcoord: optional; name of the `x` coordinate axis.
          ycoord: optional; name of the `y` coordinate axis.
        """
        # Update stored axes properties
        if set_xlabel is not None:
            self.axes_props.set_xlabel = set_xlabel
        else:
            set_xlabel = self.axes_props.set_xlabel
        if set_ylabel is not None:
            self.axes_props.set_ylabel = set_ylabel
        else:
            set_ylabel = self.axes_props.set_ylabel
        if set_xticks is not None:
            self.axes_props.set_xticks = set_xticks
        else:
            set_xticks = self.axes_props.set_xticks
        if set_yticks is not None:
            self.axes_props.set_yticks = set_yticks
        else:
            set_yticks = self.axes_props.set_yticks

        # Get axes
        ra, dec = self.ax.coords[xcoord], self.ax.coords[ycoord]

        # Axes labels
        if self.radesys == 'icrs':
            system = f'({self.radesys.upper()})'
        elif self.radesys == 'fk5':
            system = '(J2000)'
        else:
            system = ''
        if set_xlabel:
            xname = f'{self.axes_props.xname} {system}'
        else:
            xname = ' '
        if set_ylabel:
            yname = f'{self.axes_props.yname} {system}'
        else:
            yname = ' '
        ra.set_axislabel(xname,
                         size=self.ax.xaxis.get_label().get_fontsize(),
                         family=self.ax.xaxis.get_label().get_family(),
                         fontname=self.ax.xaxis.get_label().get_fontname(),
                         minpad=self.axes_props.label_xpad)
        dec.set_axislabel(yname,
                          size=self.ax.xaxis.get_label().get_fontsize(),
                          family=self.ax.xaxis.get_label().get_family(),
                          fontname=self.ax.xaxis.get_label().get_fontname(),
                          minpad=self.axes_props.label_ypad)

        # Ticks labels
        ra.set_major_formatter(self.axes_props.xticks_fmt)
        ra.set_ticklabel_visible(set_xticks)
        dec.set_major_formatter(self.axes_props.yticks_fmt)
        dec.set_ticklabel_visible(set_yticks)
        ra.set_ticks(color=self.axes_props.ticks_color)
        dec.set_ticks(color=self.axes_props.ticks_color)

        # Ticks fonts
        ra.set_ticklabel(
            size=self.ax.xaxis.get_majorticklabels()[0].get_fontsize(),
            family=self.ax.xaxis.get_majorticklabels()[0].get_family(),
            fontname=self.ax.xaxis.get_majorticklabels()[0].get_fontname(),
            exclude_overlapping=True,
        )
        dec.set_ticklabel(
            size=self.ax.yaxis.get_majorticklabels()[0].get_fontsize(),
            family=self.ax.yaxis.get_majorticklabels()[0].get_family(),
            fontname=self.ax.yaxis.get_majorticklabels()[0].get_fontname(),
        )

        # Minor ticks
        ra.display_minor_ticks(True)
        dec.display_minor_ticks(True)

    def config_plot(self, **kwargs) -> None:
        # Docs is inherited
        # Change axes
        try:
            self.ax.coords[0].set_major_formatter(self.axes_props.xticks_fmt)
            self.ax.coords[0].set_format_unit(self.xunit)
        except AttributeError:
            self.ax.xaxis.set_major_formatter(self.axes_props.xticks_fmt)
        except ValueError:
            pass
        try:
            self.ax.coords[1].set_major_formatter(self.axes_props.yticks_fmt)
            self.ax.coords[1].set_format_unit(self.yunit)
        except AttributeError:
            self.ax.yaxis.set_major_formatter(self.axes_props.yticks_fmt)
        except ValueError:
            pass

        if self.axes_props.invertx:
            self.ax.invert_xaxis()
        if self.axes_props.inverty:
            self.ax.invert_yaxis()

        super().config_plot(**kwargs)

    # Artist functions
    def scatter(self,
                x: Union[float, u.Quantity],
                y: Union[float, u.Quantity],
                **kwargs):
        return super().scatter(x, y,
                               transform=self.ax.get_transform(self.radesys),
                               **kwargs)

    def text(self,
             x: Union[float, u.Quantity],
             y: Union[float, u.Quantity],
             text: str,
             **kwargs):
        return super().text(x, y, text,
                            transform=self.ax.get_transform(self.radesys),
                            **kwargs)

    def arrow(self,
              x: Union[u.Quantity, float],
              y: Union[u.Quantity, float],
              pa: u.Quantity,
              length: float = 0.5,
              **kwargs):
        """Draw an arrow.

        An arrow can be specified by the following combiantions:

          - (PA,)
          - (x, y, PA)
          - (PA, length)
          - (x, y, PA, length)

        with PA the position angle (from axis `y>0` towards axis `x<0`). The
        keyword argument `xycoords` can be used to specify the coordinate
        system of the `(x, y)` position. The default is to use `data`
        coordinates. The `length` is specified in axes fraction.

        Args:
          x, y: position of the center of the arrow.
          pa: position angle of the arrow.
          length: optional; length of the arrow.
          kwargs: optional; arrow properties for `matplotlib.pyplot.annotate`.
        """
        xycoords = kwargs.pop('xycoords', 'data')
        if xycoords == 'data':
            xy = (x.value, y.value)
            vals = (pa.to(u.deg).value, length)

            # Transform to axes coordinates
            xy_disp = self.ax.get_transform(self.radesys).transform(xy)
            xy_axes = tuple(self.ax.transAxes.inverted().transform(xy_disp))
        else:
            xy_axes = (x, y)
            vals = (pa.to(u.deg).value, length)


        return super().arrow(xy_axes + vals, **kwargs)

    #def circle(self, x, y, r, color='g', facecolor='none', zorder=0):
    #    cir = SphericalCircle((x, y), r, edgecolor=color, facecolor=facecolor,
    #            transform=self.ax.get_transform('world'), zorder=zorder)
    #    self.ax.add_patch(cir)

    #def rectangle(self, blc, width, height, edgecolor='green',
    #        facecolor='none', **kwargs):
    #    r = Rectangle(blc, width, height, edgecolor=edgecolor,
    #            facecolor=facecolor, **kwargs)
    #    self.ax.add_patch(r)

    #def plot(self, *args, **kwargs):
    #    """Plot data."""
    #    try:
    #        kwargs['transform'] = self.ax.get_transform(self.radesys)
    #    except TypeError:
    #        pass
    #    self.ax.plot(*args, **kwargs)

    def phys_scale(self, x0: u.Quantity, y0: u.Quantity, length: u.Quantity,
                   label: str, color: str = 'w', zorder: int = 10) -> None:
        """Plot physical scale.
        
        Args:
          x0, y0: position of the origin of the scale.
          length: length of the scale.
          label: scale label.
          color: optional; scale color.
          zorder: optional; plotting order.
        """
        # Plot bar
        self._log.info('Plotting physical scale')
        self._log.info('Scale length: %s', length)
        xval = np.array([x0.value, x0.value]) * x0.unit
        yval = np.array([y0.value, (y0 + length).value]) * y0.unit
        self.plot(xval, yval, color=color, ls='-', lw=1, marker='_',
                  zorder=zorder, transform=self.ax.get_transform(self.radesys))

        # Plot label
        try:
            xycoords = self.ax.get_transform('world')
        except TypeError:
            xycoords = 'data'
        xval = xval.to(self.axes_props.xunit).value
        yval = yval.to(self.axes_props.yunit).value
        self.annotate(label,
                      xy=[xval[0], yval[0]],
                      xytext=[xval[0], yval[0]],
                      color=color,
                      horizontalalignment='right',
                      xycoords=xycoords,
                      zorder=zorder)

    #def set_aspect(self, *args):
    #    """Set plot aspect ratio."""
    #    self.ax.set_aspect(*args)

    # Misc
    def auto_plot(self,
                  data: Map,
                  dtype: str,
                  config: ConfigParserAdv) -> None:
        """Plot the input data and the stored artists.

        Args:
          data: data to plot.
          dtype: type of plot.
          config: config parser proxy.
        """
        # Common options from config
        position = config.getskycoord('center', fallback=None)
        radius = config.getquantity('radius', fallback=None)
        self_contours = config.getboolean('self_contours', fallback=None)
        use_extent = config.getboolean('use_extent', fallback=False)
        contour_colors = config.get('contour_colors', fallback=None)
        contour_linewidth = config.getfloat('contour_linewidth', fallback=None)
        ignore_units = config.getboolean('ignore_units', fallback=False)
        rms = config.getquantity('rms', fallback=None)
        levels = config.getquantity('levels', fallback=None)
        nsigma = self.skeleton.getfloat('data', 'nsigma')
        nsigma = config.getfloat('nsigma', fallback=nsigma)
        negative_nsigma = config.getfloat('negative_nsigma', fallback=None)
        nsigma_level = config.getfloat('nsigma_level', fallback=None)
        shift_data = config.getquantity('shift_data', fallback=None)

        # Special cases
        if dtype == 'pvmap':
            use_extent = True

        # Plot contours or map
        if dtype == 'contour':
            self.plot_contours(data, use_extent=use_extent, rms=rms,
                               levels=levels, colors=contour_colors,
                               nsigma=nsigma, negative_nsigma=negative_nsigma,
                               ignore_units=ignore_units,
                               linewidths=contour_linewidth, zorder=2)
        elif 'with_style' in config:
            self._log.info('Changing style: %s', config['with_style'])
            with mpl.pyplot.style.context(config['with_style']):
                self.plot_map(data,
                              use_extent=use_extent,
                              rms=rms,
                              position=position,
                              radius=radius,
                              shift_data=shift_data,
                              self_contours=self_contours,
                              contour_levels=levels,
                              contour_colors=contour_colors,
                              contour_linewidths=contour_linewidth,
                              contour_nsigma=nsigma,
                              contour_negative_nsigma=negative_nsigma,
                              contour_nsigmalevel=nsigma_level)
        else:
            self.plot_map(data,
                          use_extent=use_extent,
                          rms=rms,
                          position=position,
                          radius=radius,
                          shift_data=shift_data,
                          self_contours=self_contours,
                          contour_levels=levels,
                          contour_colors=contour_colors,
                          contour_linewidths=contour_linewidth,
                          contour_nsigma=nsigma,
                          contour_negative_nsigma=negative_nsigma,
                          contour_nsigmalevel=nsigma_level)

        # Plot artists
        self.plot_artists()

        # Plot beam
        plot_beam = self.skeleton.getboolean('data', 'plot_beam')
        if (config.getboolean('plot_beam', fallback=plot_beam) and
            hasattr(data, 'header')):
            self._log.info('Plotting beam')
            # Configuration keys
            if 'bmin' in config and 'bmaj' in config and 'bpa' in config:
                beam = Beam(major=config.getquantity('bmaj'),
                            minor=config.getquantity('bmin'),
                            pa=config.getquantity('bpa'))
            else:
                beam = None
            beam_color = self.skeleton.get('data', 'beam_color')
            beam_color = config.get('beam_color', fallback=beam_color)
            beam_pad = self.skeleton.getfloat('data', 'beam_pad')
            beam_pad = config.getfloat('beam_pad', fallback=beam_pad)

            # Plot
            self.plot_beam(data.header, beam=beam, color=beam_color,
                           pad=beam_pad)

        ## Title
        #if 'title' in config:
        #    self.title(config['title'])

        # Scale
        if 'scale' in config:
            # From config
            scale_pos = config.getskycoord('scale')
            scale_pos = scale_pos.transform_to(self.radesys)
            distance = config.getquantity('source_distance').to(u.pc)
            if 'scale_size' in config:
                size = config.getquantity('scale_size').to(u.au)
                length = size.value / distance.value * u.arcsec
            elif 'scale_length' in config:
                length = config.getquantity('scale_length').to(u.arcsec)
                size = distance.value * length.value * u.au
            scalecolor = config.get('scale_color', fallback='w')

            # Check size unit
            size = size.to(u.Unit(config.get('scale_unit', fallback='au')))

            # Scale label
            label = f'{size.value:.0f} {size.unit:latex_inline}'
        
            # Plot scale
            self.phys_scale(scale_pos.ra, scale_pos.dec, length, label,
                            color=scalecolor)

    def brightness_temperature(self,
                               header: Mapping,
                               beam: Optional[Beam] = None):
        """Store brightness temperature equivalency function.

        Args:
          header: FITS file header.
          beam: optional; beam area.
        """
        # Frequency from header
        freq = header['CRVAL3'] * u.Hz

        # Beam
        if beam is None:
            beam = Beam.from_fits_header(header)

        # Store
        self._log.info('Storing brightness temperature equivalency:')
        self._log.info('Frequency: %s', freq.to(u.GHz))
        self._log.info('Beam: %s', beam)
        self.vscale.unit_equiv = u.brightness_temperature(freq, beam)

class MapsPlotter(BasePlotter):
    """Plotter for managing 2-D maps."""

    def __init__(self,
                 config: Optional[pathlib.Path] = None,
                 section: str = 'map_plot',
                 projection: Projection = None,
                 **kwargs):
        super().__init__(config=config, section=section, **kwargs)
        self._projection = projection

    @property
    def projection(self):
        return self._projection

    def init_axis(self,
                  loc: Location,
                  projection: Optional[Projection] = None,
                  include_cbar: Optional[bool] = None) -> MapHandler:
        """Initialize the axis.

        Args:
          loc: axis location (row, column).
          projection: optional; map projection.
          include_cbar: optional; include color bar?
          kwargs: additional parameters for `MapHandler.from_config`.

        Returns:
          A `MapHandler`.
        """
        if self.is_init(loc):
            self._log.info('Axis %s already initialized', loc)
            return self.axes[loc]

        # Projection system
        radesys = ''
        if projection is None:
            projection = self.projection
            radesys = 'world'
        if projection is not None:
            try:
                aux = projection.to_header()
                if 'RADESYS' in aux:
                    radesys = aux['RADESYS']
            except AttributeError:
                pass

        # Get the axis
        ax = super().init_axis(loc,
                               MapHandler,
                               projection=projection,
                               include_cbar=include_cbar,
                               radesys=radesys,
                               **kwargs)

        return ax

    def plot(self, projection: Projection = None) -> None:
        """Plot the current configuration section.

        Args:
          projection: optional; map projection.
        """
        # Load data
        if 'image' in self.config:
            image = pathlib.Path(self.config['filename'])
            image = fits.open(image)[0]
        else:
            image = None
        if 'contour' in self.config:
            contour = pathlib.Path(self.config['contour'])
            contour = fits.open(contour)[0]
        else:
            contour = None

        # Check data
        if projection is None:
            try:
                wcs = WCS(image.header).sub(['longitude', 'latitude'])
            except:
                wcs = WCS(contour.header).sub(['longitude', 'latitude'])
            projection = wcs

        # Get axis
        ax = self.init_axis(self.loc, projection=projection)

        # Plot data
        options = filter_config_data(self.config,
                                     ax.skeleton.options('data'))
        ax.auto_plot(image=image, contour=contour, options=options)

    def plot_loc(self, loc: Location,
                 projection: Optional[Projection] = None) -> None:
        """Plot everythnig at a given location.

        Args:
          loc: axis location.
          projection: optional; map projection.
        """
        for section in self._config_mapping[loc]:
            # Switch to section
            self.switch_to(section)
            self.plot(projection=projection)

#    def plot_all(self,
#                 skip_loc: Sequence[Location] = (),
#                 projections: Mapping = {},
#                 ) -> None:
#        """
#        """
#        for loc in self:
#            # Skip locations
#            if loc in skip_loc:
#                continue
#
#            # Projection
#            projection = projections.get(loc, self.projection)
#
#            # Plot location
#            self.plot_loc(loc, projection=projection)
#
#    def apply_config(self, config=None, section='map_plot', legend=False,
#            dtype='intensity', **kwargs):
#        # Read new config if requested
#        if config is not None:
#            cfg = ConfigParser()
#            cfg.read(os.path.expanduser(config))
#            cfg = cfg['map_plot']
#        else:
#            cfg = self.config
#
#        # Config map options
#        xformat = kwargs.get('xformat',
#                cfg.get('xformat', fallback="hh:mm:ss.s"))
#        yformat = kwargs.get('yformat',
#                cfg.get('yformat', fallback="dd:mm:ss"))
#        tickscolors = kwargs.get('tickscolor',
#                cfg.get('tickscolor', fallback="k"))
#
#        # Config
#        for i,(loc,ax) in enumerate(self.axes.items()):
#            if not self.is_init(loc):
#                break
#
#            # Labels and ticks
#            xlabel, ylabel = self.has_axlabels(loc)
#            xticks, yticks = self.has_ticks(loc)
#
#            # Ticks color
#            if len(tickscolors) == 1:
#                tickscolor = tickscolors
#            else:
#                try:
#                    tickscolor = tickscolors.replace(',',' ').split()[i]
#                except IndexError:
#                    tickscolor = tickscolors[0]
#
#            if self.config.getfloat('xsize')==self.config.getfloat('ysize'):
#                self.log.info('Setting equal axis aspect ratio')
#                ax.set_aspect(1./ax.ax.get_data_ratio())
#
#            if dtype=='pvmap':
#                try:
#                    xlim = map(float, self.get_value('xlim', (None,None), loc,
#                        sep=',').split())
#                except (TypeError, AttributeError):
#                    xlim = (None, None)
#                try:
#                    ylim = map(float, self.get_value('ylim', (None,None), loc,
#                        sep=',').split())
#                except (TypeError, AttributeError):
#                    ylim = (None, None)
#                #xlabel = 'Offset (arcsec)' if xlabel else ''
#                #ylabel = 'Velocity (km s$^{-1}$)' if ylabel else ''
#                ylabel = 'Offset (arcsec)' if xlabel else ''
#                xlabel = 'Velocity (km s$^{-1}$)' if ylabel else ''
#                ax.config_plot(xlim=tuple(xlim), xlabel=xlabel,
#                        unset_xticks=not xticks,
#                        ylim=tuple(ylim), ylabel=ylabel,
#                        unset_yticks=not yticks,
#                        tickscolor=tickscolor)
#            else:
#                ax.config_map(xformat=xformat, yformat=yformat, xlabel=xlabel,
#                        ylabel=ylabel, xticks=xticks, yticks=yticks,
#                        xpad=1., ypad=-0.7, tickscolor=tickscolor, xcoord='ra', ycoord='dec')
#
#            # Scale
#            if 'scale_position' in self.config:
#                # From config
#                scale_pos = self.config.getskycoord('scale_position')
#                distance = self.config.getquantity('distance').to(u.pc)
#                length = self.config.getquantity('scale_length', 
#                        fallback=1*u.arcsec).to(u.arcsec)
#                labeldy = self.config.getquantity('scale_label_dy',
#                        fallback=1*u.arcsec).to(u.deg)
#                scalecolor = self.config.get('scale_color', fallback='w')
#
#                # Scale label
#                label = distance * length
#                label = label.value * u.au
#                label = '{0.value:.0f} {0.unit:latex_inline}  '.format(label)
#                
#                # Plot scale
#                ax.phys_scale(scale_pos.ra.degree, scale_pos.dec.degree,
#                        length.to(u.degree).value, label, 
#                        color=scalecolor)
#
#            # Legend
#            if (legend or self.config.getboolean('legend', fallback=False)) and loc==(0,0):
#                ax.legend(auto=True, loc=4, match_colors=True,
#                        fancybox=self.config.getboolean('fancybox', fallback=False),
#                        framealpha=self.config.getfloat('framealpha', fallback=None),
#                        facecolor=self.config.get('facecolor', fallback=None))
#
