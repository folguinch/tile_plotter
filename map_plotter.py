"""Objects for plotting images."""
from typing import (Sequence, Optional, TypeVar, Union, Tuple, Dict, Mapping,
                    List, Callable)
import pathlib

from astropy.visualization import LogStretch, LinearStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization.wcsaxes import SphericalCircle
from configparseradv.configparser import ConfigParserAdv
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle, Ellipse
from matplotlib import cm
from radio_beam import Beam
import astropy.units as u
import astropy.visualization as vis
import astropy.wcs as apy_wcs
import matplotlib as mpl
import numpy as np

from .base_plotter import BasePlotter
from .plot_handler import PhysPlotHandler
from .utils import (get_artist_properties, auto_vmin, auto_vmax,
                    generate_label, auto_levels, get_colorbar_ticks)

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
                       ignore: Sequence = ['image', 'contour']) -> Dict:
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
        elif key in ['self_contours']:
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
      bname
      bunit
      bname_cbar2
      bunit_cbar2
      vscale: intensity scale parameters.
      artists: artists to plot.
      colors: replace style colors.
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
                 units: Mapping = {},
                 vscale: Mapping = {},
                 colors: Mapping = {},
                 artists: Mapping = {}):
        """Initiate a map plot handler."""
        super().__init__(axis, cbaxis=cbaxis,
                         xname=units.get('xname', 'RA'),
                         yname=units.get('yname', 'Dec'),
                         xunit=units.get('xunit', u.deg),
                         yunit=units.get('yunit', u.deg))

        # Units and names
        self._log.info('Creating map plot')
        self.im = None
        self.bname = units.get('bname', 'Intensity')
        self.bunit = units.get('bunit', u.Unit(1))
        self._log.info(f'Setting bunit ({self.bname}): {self.bunit}')
        self.bname_cbar2 = units.get('bname_cbar2')
        self.bunit_cbar2 = units.get('bunit_cbar2')

        # Store other options
        self.vscale = vscale
        self.colors = colors
        self.artists = artists

        # Projection system
        if radesys is not None:
            self.radesys = radesys.lower()
            if radesys.upper() == 'J2000':
                self.radesys = 'fk5'
            self._log.info(f'Setting RADESYS: {self.radesys}')
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
        # Get units
        units = {}
        for opt in cls.skeleton.options('units'):
            default = cls.skeleton.get('units', opt)
            val = config.get(opt, vars=kwargs, fallback=default)
            if 'unit' in opt:
                try:
                    units[opt] = u.Unit(val)
                except ValueError:
                    units[opt] = None
            else:
                units[opt] = val

        # Get vscale
        vscale = {}
        for opt in cls.skeleton.options('vscale'):
            if opt not in config and opt not in kwargs:
                continue
            val = config.get(opt, vars=kwargs)
            try:
                split_val = val.split()
                vscale[opt] = float(split_val[0]) * u.Unit(split_val[1])
            except ValueError:
                vscale[opt] = val
            except IndexError:
                vscale[opt] = float(val)

        # Colors
        colors = {}
        for opt, val in cls.skeleton.items('colors'):
            if opt not in config and opt not in kwargs:
                continue
            colors[opt] = config.get(opt, vars=kwargs)

        # Artists
        artists = {}
        for opt, val in cls.skeleton.items('artists'):
            if opt not in config and opt not in kwargs:
                continue
            artists[opt] = get_artist_properties(opt, config)

        return cls(axis, cbaxis, radesys=radesys, units=units, vscale=vscale,
                   colors=colors, artists=artists)

    @property
    def vmin(self):
        if 'vmin' not in self.vscale:
            return None
        return self.vscale['vmin'].to(self.bunit)

    @property
    def vmax(self):
        if 'vmax' not in self.vscale:
            return None
        return self.vscale['vmax'].to(self.bunit)

    @property
    def vcenter(self):
        if 'vcenter' not in self.vscale:
            return None
        return self.vscale['vcenter'].to(self.bunit)

    @property
    def stretch(self):
        return self.vscale['stretch']

    def _validate_data(self, data: Map,
                       wcs: apy_wcs.WCS) -> Tuple[u.Quantity, apy_wcs.WCS]:
        """Validate input data.

        Convert input data to a quantity with the same unit as `self.bunit` or
        determines `self.bunit` from header. If the unit of the data cannot be
        determined it is assumed to be in the same units as `self.bunit` or is
        considered dimensionless.

        Args:
          data: input data.
          wcs: WCS of the data.
        """
        if hasattr(data, 'unit'):
            # Check bunit
            if self.bunit is None:
                self.bunit = data.bunit
                self._log.info(f'Setting bunit to data unit: {self.bunit}')
            else:
                valdata = data.to(self.bunit)
        elif hasattr(data, 'header'):
            # Check bunit
            bunit = u.Unit(data.header.get('BUNIT', 1))
            valdata = np.squeeze(data.data) * bunit
            if self.bunit is None:
                self.bunit = bunit
                self._log.info(f'Setting bunit to header unit: {bunit}')
            else:
                valdata = valdata.to(self.bunit)

            # Check wcs
            if wcs is None:
                wcs = apy_wcs.WCS(data.header, naxis=['longitude', 'latitude'])

            # Set RADESYS
            if self.radesys is None:
                self.radesys = data.header['RADESYS'].lower()
                if self.radesys.upper() == 'J2000':
                    self.radesys = 'fk5'
                self._log.info(f'Setting RADESYS: {self.radesys}')
        else:
            if self.bunit is None:
                self._log.warning('Setting data as dimensionless')
                self.bunit = u.Unit(1)
            else:
                self._log.warning(f'Assuming data in {self.bunit} units')
            valdata = data * self.bunit

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
            self.vscale['vmin'] = auto_vmin(data, rms=rms, log=self._log.info)
            self._log.info(f"Setting vmin from data: {self.vscale['vmin']}")
        if self.vmax is None:
            self.vscale['vmax'] = auto_vmax(data)
            self._log.info(f"Setting vmax from data: {self.vscale['vmax']}")

    def _validate_extent(self,
                         extent: Tuple[u.Quantity, u.Quantity,
                                       u.Quantity, u.Quantity],
                         ) -> Tuple[float, float, float, float]:
        """Validate extent values.

        Convert the values in a extent sequence to the respective axis units
        and returns the quantity values.
        """
        xextent = [ext.to(self.xunit).value for ext in extent[:2]]
        yextent = [ext.to(self.yunit).value for ext in extent[2:]]

        return tuple(xextent + yextent)

    # Getters
    def get_normalization(self,
                          vmin: Optional[u.Quantity] = None,
                          vmax: Optional[u.Quantity] = None) -> cm:
        """Determine the normalization of the color stretch.

        Args:
          vmin: optional; scale minimum.
          vmax: optional; scale maximum.
        """
        if vmin is None:
            vmin = self.vmin
        if vmax is None:
            vmax = self.vmax
        if self.stretch == 'log':
            return ImageNormalize(vmin=vmin.value,
                                  vmax=vmax.value,
                                  stretch=LogStretch(a=self.vscale.get('a',
                                                                       1000)))
        elif self.stretch == 'midnorm':
            return TwoSlopeNorm(self.vcenter,
                                vmin=vmin.value,
                                vmax=vmax.value)
        else:
            return ImageNormalize(vmin=vmin.value,
                                  vmax=vmax.value,
                                  stretch=LinearStretch())

    def get_blabel(self, unit_fmt: str = '({:latex_inline})') -> str:
        """Generate a string for the color bar label."""
        return generate_label(self.bname, unit=self.bunit, unit_fmt=unit_fmt)

    # Plotters
    def plot_map(self,
                 data: Map,
                 wcs: Optional[apy_wcs.WCS] = None,
                 extent: Optional[Tuple[u.Quantity]] = None,
                 rms: Optional[u.Quantity] = None,
                 mask_bad: bool = False,
                 mask_color: str = 'w',
                 position: Optional['astroppy.SkyCoord'] = None,
                 radius: Optional[u.Quantity] = None,
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
          rms: optional; map noise level.
          mask_bad: optional; mask bad/null pixels?
          mask_color: optional; color for bad/null pixels.
          position: optional; center of the map.
          radius: optional; radius of the region shown.
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

        # Check vscale and get normalization
        self._validate_vscale(valdata, rms=rms)
        norm = self.get_normalization()

        # Check extent
        if extent is not None:
            extent_val = _validate_extent(extent)
        else:
            extent_val = None

        # Check wcs and re-center the image
        if valwcs is not None and radius is not None and position is not None:
            self.recenter(radius, position, valwcs)

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
                      rms: Optional[u.Quantity] = None,
                      levels: Optional[List[u.Quantity]] = None,
                      colors: Optional[Sequence[str]] = None,
                      nsigma: float = 5.,
                      negative_nsigma: Optional[float] = None,
                      nsigmalevel: Optional[float] = None,
                      **kwargs):
        """Plot a contour map.

        Args:
          data: input map.
          wcs: optional; WCS object of the map.
          extent: optional; `xy` range of the data (left, right, bottom, top).
          rms: optional; map noise level.
          levels: optional; contour levels.
          colors: optional; contour colors.
          nsigma: optional; level of the lowest contour over rms.
          negative_nsigma: optional; level of the highest negative contour.
          nsigmalevel: optional; plot only one contour at this level times the
            rms value.
          **kwargs: optional; additional arguments for `pyplot.contours`.
        """
        # Validate data: valdata is a quantity with self.bunit units
        valdata, valwcs = self._validate_data(data, wcs)

        # Check extent
        if extent is not None:
            extent_val = _validate_extent(extent)

        # Levels
        if levels is None:
            try:
                if nsigmalevel is not None:
                    nlevels = 1
                else:
                    nlevels = None
                levels = auto_levels(valdata,
                                     rms=rms,
                                     nsigma=nsigma,
                                     nsigmalevel=nsigmalevel,
                                     nlevels=nlevels,
                                     negative_nsigma=negative_nsigma,
                                     log=self._log.info)
                levels_val = levels.value
            except ValueError:
                return None
        else:
            levels_val = levels.to(self.bunit).value

        # Color map
        if 'cmap' not in kwargs:
            kwargs['colors'] = colors or self.colors.get('contours', 'g')
        elif 'norm' not in kwargs:
            kwargs['norm'] = self.get_normalization()
        else:
            pass

        # Plot
        zorder = kwargs.setdefault('zorder', 0)
        if wcs is not None:
            return super().contour(valdata,
                                   is_image=True,
                                   levels=levels_val,
                                   transform=self.ax.get_transform(wcs),
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
            if artist == 'scatters':
                pos = position.transform_to(self.radesys)
                art = self.scatter(pos.ra, pos.dec, **props)
            elif artist == 'texts':
                pos = position.transform_to(self.radesys)
                text = props.pop('text')
                art = self.text(pos.ra, pos.dec, text, nphys_args=2, **props)

    def plot_artists(self) -> None:
        """Plot all the stored artists."""
        for artist in self.artists:
            self._plot_artist(artist)
        #for artist in artists:
        #    if artist not in cfg:
        #        continue
        #    if artist=='texts' or artist=='arrows':
        #        iterover = cfg.getvalueiter(artist, sep=',')
        #    else:
        #        iterover = cfg.getvalueiter(artist, sep=',', dtype='skycoord')
        #    for i, art in enumerate(iterover):
        #        color = cfg.getvalue('%s_color' % artist, n=i, fallback='g')
        #        facecolor = cfg.getvalue('%s_facecolor' % artist, n=i,
        #                fallback=color)
        #        edgecolor = cfg.getvalue('%s_edgecolor' % artist, n=i,
        #                fallback=color)
        #        zorder = cfg.getvalue('%s_zorder' % artist, n=i, fallback=2,
        #                dtype=int)
        #        if artist=='markers':
        #            fmt = cfg.getvalue('%s_fmt' % artist, n=i, fallback='+')
        #            size = cfg.getvalue('%s_size' % artist, n=i, fallback=100,
        #                    dtype=float)
        #            if self.radesys == 'icrs':
        #                markra = art.icrs.ra.degree
        #                markdec = art.icrs.dec.degree
        #            elif self.radesys == 'fk5':
        #                markra = art.fk5.ra.degree
        #                markdec = art.fk5.dec.degree
        #            else:
        #                markra = art.ra.degree
        #                markdec = art.dec.degree
        #            mk = self.scatter(markra, markdec,
        #                    edgecolors=edgecolor, facecolors=facecolor,
        #                    marker=fmt, s=size, zorder=zorder)
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
        #        elif artist=='texts':
        #            loc = cfg.getvalue('%s_loc' % artist, n=i, sep=',')
        #            if r'\n' in art:
        #                art = art.replace(r'\n', '\n')
        #            kwargs = {'weight':'normal'}
        #            for opt in kwargs:
        #                kwargs[opt] = cfg.getvalue('%s_%s' % (artist, opt),
        #                        n=i, fallback=kwargs[opt])
        #            loc = map(float, loc.split())
        #            self.label_axes(str(art), loc=loc, color=color,
        #                    zorder=zorder, **kwargs)
        #        elif artist=='arrows':
        #            self.arrow(art, color=color)

    def plot_cbar(self,
                  fig: 'Figure',
                  label: Optional[str] = None,
                  ticks: Optional[List[u.Quantity]] = None,
                  nticks: int = 5,
                  ticklabels: Optional[List[str]] = None,
                  tickstretch: Optional[str] = None,
                  orientation: str = 'vertical',
                  labelpad: float = 10,
                  lines: Optional[Plot] = None,
                  equivalency: Optional[Callable] = None,
                  ) -> Optional[mpl.colorbar.Colorbar]:
        """Plot the color bar.

        If ticks are not given, they will be determined from the other
        parameters (nticks, vmin, vmax, a, stretch, etc.) or use the defaults
        from matplotlib.

        When a second (clone) color bar axis is requested, the `equivalency`
        argument can be used to convert the values of the color bar axis ticks.

        Args:
          fig: figure object.
          label: optional; color bar label.
          ticks: optional; color bar ticks.
          nticks: optional; number of ticks for auto ticks.
          ticklabels: optional; tick labels.
          tickstretch: optional; stretch for the ticks.
          orientation: optional; color bar orientation.
          labelpad: optional; shift the color bar label.
          lines: optional; lines from contour plot to overplot.
          equivalency: optional; function to convert between intensity units.
        """
        # Verify input
        kwargs = {
            'a': self.vscale.get('a', 1000),
            'label': label or self.get_blabel(),
            'orientation': orientation,
            'labelpad': labelpad,
            'lines': lines,
            'ticklabels': ticklabels,
        }

        # Get ticks
        if ticks is None:
            aux = get_colorbar_ticks(self.vmin, self.vmax,
                                     a=kwargs['a'],
                                     n=nticks,
                                     stretch=tickstretch or self.stretch)
        else:
            aux = ticks.to(self.bunit)
        kwargs['ticks'] = aux

        # Ticks of cbar2
        self._log.info('Color bar ticks: %s', kwargs['ticks'])
        if self.bunit_cbar2 is not None:
            if equivalency is None:
                equivalency = self.vscale['equivalency']
            ticks_cbar2 = kwargs['ticks'].to(self.bunit_cbar2,
                                             equivalencies=equivalency)
            label_cbar2 = generate_label(self.bname_cbar2,
                                         unit=self.bunit_cbar2,
                                         unit_fmt='({:latex_inline})')
            kwargs['ticks_cbar2'] = ticks_cbar2
            kwargs['label_cbar2'] = label_cbar2
            kwargs['norm_cbar2'] = self.get_normalization(
                vmin=np.min(kwargs['ticks_cbar2']),
                vmax=np.max(kwargs['ticks_cbar2']))
            self._log.info('Color bar 2 ticks: %s', kwargs['ticks_cbar2'])
            kwargs['ticks_cbar2'] = kwargs['ticks_cbar2'].value

        # Get value
        kwargs['ticks'] = kwargs['ticks'].value

        return super().plot_cbar(fig, self.im, **kwargs)

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
        wcs = apy_wcs.WCS(header, naxis=['longitude', 'latitude'])

        # Beam
        if beam is None:
            beam = Beam.from_fits_header(header)
        self._log.info('Plotting %s', beam)

        # Store beam equivalency
        self.brightness_temperature(header, beam)

        # Convert to pixel
        pixsize = np.sqrt(wcs.proj_plane_pixel_area())
        bmaj = beam.major.cgs / pixsize.cgs
        bmin = beam.minor.cgs / pixsize.cgs

        # Define position
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()
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
        #if hasattr(position, 'ra'):
        #    ra = position.ra.degree
        #    dec = position.dec.degree
        #elif len(position)==2:
        #    ra, dec = position
        #else:
        #    print('WARN: Could not recenter plot')
        #    return
        #x, y = wcs.all_world2pix([[ra,dec]], 0)[0]
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
                   xformat: str = "hh:mm:ss.s",
                   yformat: str = "dd:mm:ss",
                   xlabel: bool = True,
                   ylabel: bool = True,
                   xticks: bool = True,
                   yticks: bool = True,
                   xpad: float = 1.,
                   ypad: float = 1.,
                   tickscolor: Optional[str] = None,
                   xcoord: str = 'ra',
                   ycoord: str = 'dec') -> None:
        """Apply configuration to map axes.

        Args:
          xformat: optional; format of the `x` axis tick labels.
          yformat: optional; format of the `y` axis tick labels.
          xlabel: optional; display `x` axis label?
          ylabel: optional; display `y` axis label?
          xticks: optional; display `x` axis tick labels?
          yticks: optional; display `y` axis tick labels?
          xpad: optional; shift of the `x` axis label.
          ypad: optional; shift of the `y` axis label.
          tickscolor: optional; ticks color.
          xcoord: optional; name of the `x` coordinate axis.
          ycoord: optional; name of the `y` coordinate axis.
        """
        # Get axes
        ra, dec = self.ax.coords[xcoord], self.ax.coords[ycoord]

        # Axes labels
        if self.radesys == 'icrs':
            system = '(%s)' % self.radesys.upper()
        elif self.radesys == 'fk5':
            system = '(J2000)'
        else:
            system = ''
        ra.set_axislabel('RA %s' % system if xlabel else '',
                         size=self.ax.xaxis.get_label().get_fontsize(),
                         family=self.ax.xaxis.get_label().get_family(),
                         fontname=self.ax.xaxis.get_label().get_fontname(),
                         minpad=xpad)
        dec.set_axislabel('Dec %s' % system if ylabel else '',
                          size=self.ax.xaxis.get_label().get_fontsize(),
                          family=self.ax.xaxis.get_label().get_family(),
                          fontname=self.ax.xaxis.get_label().get_fontname(),
                          minpad=ypad)

        # Ticks labels
        tickscolor = tickscolor or self.colors.get('tickscolor', 'w')
        ra.set_major_formatter(xformat)
        ra.set_ticklabel_visible(xticks)
        dec.set_major_formatter(yformat)
        dec.set_ticklabel_visible(yticks)
        ra.set_ticks(color=tickscolor, exclude_overlapping=True)
        dec.set_ticks(color=tickscolor)

        # Ticks
        ra.set_ticklabel(
            size=self.ax.xaxis.get_majorticklabels()[0].get_fontsize(),
            family=self.ax.xaxis.get_majorticklabels()[0].get_family(),
            fontname=self.ax.xaxis.get_majorticklabels()[0].get_fontname(),
        )
        dec.set_ticklabel(
            size=self.ax.yaxis.get_majorticklabels()[0].get_fontsize(),
            family=self.ax.yaxis.get_majorticklabels()[0].get_family(),
            fontname=self.ax.yaxis.get_majorticklabels()[0].get_fontname(),
        )
        ra.display_minor_ticks(True)
        dec.display_minor_ticks(True)

    def set_title(self, title):
        self.ax.set_title(title)

    def scatter(self,
                x: Union[float, u.Quantity],
                y: Union[float, u.Quantity],
                **kwargs):
        """Scatter plot."""
        return super().scatter(x, y,
                               transform=self.ax.get_transform(self.radesys),
                               **kwargs)

    def text(self,
             x: Union[float, u.Quantity],
             y: Union[float, u.Quantity],
             text: str,
             **kwargs):
        """Plot text."""
        return super().text(x, y, text,
                            transform=self.ax.get_transform(self.radesys),
                            **kwargs)

    def circle(self, x, y, r, color='g', facecolor='none', zorder=0):
        cir = SphericalCircle((x, y), r, edgecolor=color, facecolor=facecolor,
                transform=self.ax.get_transform('world'), zorder=zorder)
        self.ax.add_patch(cir)

    def rectangle(self, blc, width, height, edgecolor='green',
            facecolor='none', **kwargs):
        r = Rectangle(blc, width, height, edgecolor=edgecolor,
                facecolor=facecolor, **kwargs)
        self.ax.add_patch(r)

    def plot(self, *args, **kwargs):
        try:
            kwargs['transform'] = self.ax.get_transform('world')
        except TypeError:
            pass
        self.ax.plot(*args, **kwargs)

    def plot_scale(self, size, r, distance, x=0.1, y=0.1, dy=0.01, color='g',
                   zorder=10, unit=u.au, loc=3):
        length = size.to(u.arcsec) / (2*r.to(u.arcsec))
        label = distance.to(u.pc) * size.to(u.arcsec)
        label = label.value * u.au
        label = "%s" % label.to(unit)
        label = label.lower()
        self.annotate('', xy=(x,y), xytext=(x+length.value, y),
                xycoords='axes fraction', arrowprops=dict(arrowstyle="|-|",
                    facecolor=color),
                color=color)
        xmid = x + length.value/2.
        self.annotate(label, xy=(xmid,y+dy), xytext=(xmid, y+dy),
                xycoords='axes fraction', color=color)
        #bar = AnchoredSizeBar(self.ax.transData, size.to(u.deg), label, loc,
        #        color=color)
        #self.ax.add_artist(bar)

    def phys_scale(self, xstart, ystart, dy, label, color='w', zorder=10):
        self.log.info('Plotting physical scale')
        self.plot([xstart, xstart], [ystart, ystart+dy], color=color, ls='-',
                lw=1, marker='_', zorder=zorder)
        try:
            xycoords = self.ax.get_transform('world')
        except TypeError:
            xycoords = 'data'
        self.annotate(label, xy=[xstart, ystart],
                xytext=[xstart, ystart], color=color,
                horizontalalignment='right',
                xycoords=xycoords, zorder=zorder)

    def plot_markers(self, markers, skip_label=False, zorder=3, **kwargs):
        self.log.info('Plotting markers')
        self.log.info('Converting markers to %s system', self.radesys)
        for m in markers:
            # Plot marker
            if self.radesys == 'icrs':
                mark = m['loc'].icrs
            elif self.radesys == 'fk5':
                mark = m['loc'].fk5
            else:
                mark = m['loc']
            mp = self.scatter(mark.ra.degree, mark.dec.degree, c=m['color'],
                    marker=m['style'], label=m.get('legend'),
                    s=m.get('size'), zorder=zorder,
                    **kwargs)

            # Marker label
            if m.get('label') and not skip_label:
                labloc = m['labloc'].ra.degree, m['labloc'].dec.degree
                self.annotate(m['label'].strip(), xy=labloc, xytext=labloc,
                        xycoords=self.ax.get_transform('world'),
                        color=m['color'], zorder=zorder,
                        fontweight=m['font_weight'])

    def set_aspect(self, *args):
        self.ax.set_aspect(*args)

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
        contour_colors = config.get('contour_colors', fallback=None)
        contour_linewidth = config.getfloat('contour_linewidth', fallback=None)
        rms = config.getquantity('rms', fallback=None)
        levels = config.getquantity('levels', fallback=None)
        nsigma = self.skeleton.getfloat('data', 'nsigma')
        nsigma = config.getfloat('nsigma', fallback=nsigma)
        negative_nsigma = config.getfloat('negative_nsigma', fallback=None)
        nsigma_level = config.getfloat('nsigma_level', fallback=None)

        ## Plot contours or map
        if dtype == 'contour':
            self.plot_contours(data, rms=rms, levels=levels,
                               colors=contours_colors, nsigma=nsigma,
                               negative_nsigma=negative_nsigma,
                               linewidths=contour_linewidth, zorder=2)
        elif 'with_style' in config:
            self._log.info('Changing style: %s', config['with_style'])
            with matplotlib.pyplot.style.context(config['with_style']):
                self.plot_map(data, rms=rms, position=position, radius=radius,
                              self_contours=self_contours,
                              contour_levels=levels,
                              contour_colors=contour_colors,
                              contour_linewidths=contour_linewidth,
                              contour_nsigma=nsigma,
                              contour_negative_nsigma=negative_nsigma,
                              contour_nsigmalevel=nsigma_level)
        else:
            self.plot_map(data, rms=rms, position=position, radius=radius,
                          self_contours=self_contours, contour_levels=levels,
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

        ## Colorbar
        #if fig is not None and dtype != 'contour_map':
        #    # Color bar label
        #    cbarlabel = config.get('cbarlabel', fallback='Intensity')
        #    if dtype == 'moment' and 'cbarlabel' not in config:
        #        moment = config.getint('moment', fallback=0)
        #        if moment==1:
        #            cbarlabel = 'Velocity'
        #        elif moment==2:
        #            cbarlabel = 'Velocity dispersion'
        #        else:
        #            cbarlabel = 'Intensity'
        #    else:
        #        cbarlabel = config.get('cbarlabel', fallback='Intensity')

        #    # Color bar unit
        #    if bunit:
        #        cbarlabel += ' (%s)' % (bunit.to_string('latex_inline'),)
        #    if config.getboolean('vcbar', fallback=False):
        #        orientation = 'vertical'
        #    else:
        #        orientation = 'horizontal'

        #    self.plot_cbar(fig, orientation=orientation, label=cbarlabel,
        #            labelpad=config.getfloat('labelpad', fallback=10))

        ## Title
        #if 'title' in config:
        #    self.title(config['title'])

        ## Scale
        #if 'scale_position' in config:
        #    # From config
        #    scale_pos = config.getskycoord('scale_position')
        #    distance = config.getquantity('distance').to(u.pc)
        #    length = config.getquantity('scale_length',
        #            fallback=1*u.arcsec).to(u.arcsec)
        #    scalecolor = config.get('scale_color', fallback='w')

        #    # Scale label
        #    label = distance * length
        #    label = label.value * u.au
        #    label = '{0.value:.0f} {0.unit:latex_inline}  '.format(label)
        #    label = label.lower()
        #
        #    # Plot scale
        #    self.phys_scale(scale_pos.ra.degree, scale_pos.dec.degree,
        #            length.to(u.degree).value, label,
        #            color=scalecolor)

        ## Label
        #if 'label' in config:
        #    self.label_axes(config['label'],
        #            backgroundcolor=config.get('label_background', None))


    def auto_config(self, cfg, xlabel, ylabel, xticks, yticks, **kwargs):
        # Config map options
        xformat = kwargs.get('xformat',
                cfg.get('xformat', fallback="hh:mm:ss.s"))
        yformat = kwargs.get('yformat',
                cfg.get('yformat', fallback="dd:mm:ss"))

        # Ticks color
        tickscolor = kwargs.get('tickscolor',
                cfg.get('tickscolor', fallback="k"))

        # Apect ratio
        self.set_aspect(1./self.ax.get_data_ratio())

        # Config
        if cfg['type']=='pvmap':
            xlim = cfg.getfloatlist('xlim', fallback=(None,None))
            ylim = cfg.getfloatlist('ylim', fallback=(None,None))
            xlabel = 'Offset (arcsec)' if xlabel else ''
            ylabel = 'Velocity (km s$^{-1}$)' if ylabel else ''
            self.config_plot(xlim=tuple(xlim), xlabel=xlabel,
                    unset_xticks=not xticks,
                    ylim=tuple(ylim), ylabel=ylabel,
                    unset_yticks=not yticks,
                    tickscolor=tickscolor)
        else:
            self.config_map(xformat=xformat, yformat=yformat, xlabel=xlabel,
                    ylabel=ylabel, xticks=xticks, yticks=yticks,
                    xpad=1., ypad=-0.7, tickscolor=tickscolor, xcoord='ra',
                            ycoord='dec')

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
        self.vscale['equivalency'] = u.brightness_temperature(freq, beam)

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
            self._log.info(f'Axis {loc} already initialized')
            return self.axes[loc]

        # Projection system
        radesys = ''
        if projection is None:
            projection = self.projection
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

    def plot_all(self,
                 skip_loc: Sequence[Location] = (),
                 projections: Mapping = {},
                 ) -> None:
        """
        """
        for loc in self:
            # Skip locations
            if loc in skip_loc:
                continue

            # Projection
            projection = projections.get(loc, self.projection)

            # Plot location
            self.plot_loc(loc, projection=projection)

    def apply_config(self, config=None, section='map_plot', legend=False,
            dtype='intensity', **kwargs):
        # Read new config if requested
        if config is not None:
            cfg = ConfigParser()
            cfg.read(os.path.expanduser(config))
            cfg = cfg['map_plot']
        else:
            cfg = self.config

        # Config map options
        xformat = kwargs.get('xformat',
                cfg.get('xformat', fallback="hh:mm:ss.s"))
        yformat = kwargs.get('yformat',
                cfg.get('yformat', fallback="dd:mm:ss"))
        tickscolors = kwargs.get('tickscolor',
                cfg.get('tickscolor', fallback="k"))

        # Config
        for i,(loc,ax) in enumerate(self.axes.items()):
            if not self.is_init(loc):
                break

            # Labels and ticks
            xlabel, ylabel = self.has_axlabels(loc)
            xticks, yticks = self.has_ticks(loc)
            
            # Ticks color
            if len(tickscolors) == 1:
                tickscolor = tickscolors
            else:
                try:
                    tickscolor = tickscolors.replace(',',' ').split()[i]
                except IndexError:
                    tickscolor = tickscolors[0]

            if self.config.getfloat('xsize')==self.config.getfloat('ysize'):
                self.log.info('Setting equal axis aspect ratio')
                ax.set_aspect(1./ax.ax.get_data_ratio())

            if dtype=='pvmap':
                try:
                    xlim = map(float, self.get_value('xlim', (None,None), loc,
                        sep=',').split())
                except (TypeError, AttributeError):
                    xlim = (None, None)
                try:
                    ylim = map(float, self.get_value('ylim', (None,None), loc,
                        sep=',').split())
                except (TypeError, AttributeError):
                    ylim = (None, None)
                #xlabel = 'Offset (arcsec)' if xlabel else ''
                #ylabel = 'Velocity (km s$^{-1}$)' if ylabel else ''
                ylabel = 'Offset (arcsec)' if xlabel else ''
                xlabel = 'Velocity (km s$^{-1}$)' if ylabel else ''
                ax.config_plot(xlim=tuple(xlim), xlabel=xlabel, 
                        unset_xticks=not xticks,
                        ylim=tuple(ylim), ylabel=ylabel, 
                        unset_yticks=not yticks, 
                        tickscolor=tickscolor)
            else:
                ax.config_map(xformat=xformat, yformat=yformat, xlabel=xlabel,
                        ylabel=ylabel, xticks=xticks, yticks=yticks, 
                        xpad=1., ypad=-0.7, tickscolor=tickscolor, xcoord='ra', ycoord='dec')

            # Scale
            if 'scale_position' in self.config:
                # From config
                scale_pos = self.config.getskycoord('scale_position')
                distance = self.config.getquantity('distance').to(u.pc)
                length = self.config.getquantity('scale_length', 
                        fallback=1*u.arcsec).to(u.arcsec)
                labeldy = self.config.getquantity('scale_label_dy',
                        fallback=1*u.arcsec).to(u.deg)
                scalecolor = self.config.get('scale_color', fallback='w')

                # Scale label
                label = distance * length
                label = label.value * u.au
                label = '{0.value:.0f} {0.unit:latex_inline}  '.format(label)
                
                # Plot scale
                ax.phys_scale(scale_pos.ra.degree, scale_pos.dec.degree,
                        length.to(u.degree).value, label, 
                        color=scalecolor)

            # Legend
            if (legend or self.config.getboolean('legend', fallback=False)) and loc==(0,0):
                ax.legend(auto=True, loc=4, match_colors=True,
                        fancybox=self.config.getboolean('fancybox', fallback=False),
                        framealpha=self.config.getfloat('framealpha', fallback=None),
                        facecolor=self.config.get('facecolor', fallback=None))

