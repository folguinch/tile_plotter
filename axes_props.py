"""Objects for storing axes properties."""
from typing import Union, Optional, Tuple, Callable, Sequence, TypeVar
import dataclasses

from astropy.visualization import LogStretch, LinearStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from matplotlib.colors import TwoSlopeNorm
import astropy.units as u

from .utils import generate_label, get_colorbar_ticks

Colormap = TypeVar('Colormap')

@dataclasses.dataclass
class AxesProps:
    """Store Axes properties."""
    xname: str = 'x'
    """X-axis name."""
    yname: str = 'y'
    """Y-axis name."""
    xlim: Optional[Union[Tuple[float, float], Tuple[None, float]]] = None
    """X-axis limits."""
    ylim: Optional[Union[Tuple[float, float], Tuple[None, float]]] = None
    """Y-axis limits."""
    xscale: str = 'linear'
    """X-axis scale (linear or log)."""
    yscale: str = 'linear'
    """Y-axis scale (linear or log)."""
    xunit: Optional[str] = None
    """X-axis unit."""
    yunit: Optional[str] = None
    """Y-axis unit."""
    xlabel: Optional[str] = None
    """X-axis label."""
    ylabel: Optional[str] = None
    """Y-axis label."""
    label_xpad: float = 0
    """Padding for the x-axis label."""
    label_ypad: float = 0
    """Padding for the y-axis label."""
    set_xlabel: bool = True
    """Draw x-axis label?"""
    set_ylabel: bool = True
    """Draw y-axis label?"""
    set_xticks: bool = True
    """Draw x-axis tick labels?"""
    set_yticks: bool = True
    """Draw y-axis tick labels?"""
    invertx: bool = False
    """Invert x axis direction?"""
    inverty: bool = False
    """Invert y axis direction?"""
    ticks_color: str = 'k'
    """Ticks color."""
    xticks_fmt: Union[None, str, Callable] = '{:.3f}'
    """Ticks labe format for x-axis."""
    yticks_fmt: Union[None, str, Callable] = '{:.3f}'
    """Ticks labe format for y-axis."""
    unit_fmt: str = '({})'
    """Format for axes unit label."""

    def __post_init__(self):
        # Generate labels
        if self.set_xlabel and self.xlabel is None:
            self.xlabel = generate_label(self.xname, self.xunit,
                                         unit_fmt=self.unit_fmt)
        if self.set_ylabel and self.ylabel is None:
            self.ylabel = generate_label(self.yname, self.yunit,
                                         unit_fmt=self.unit_fmt)

        # Limits
        if self.xlim is not None:
            self.xlim = dict(zip(('left', 'right'), self.xlim))
        if self.ylim is not None:
            self.ylim = dict(zip(('bottom', 'top'), self.ylim))

@dataclasses.dataclass
class PhysAxesProps(AxesProps):
    """Store Axes properties."""
    xlim: Union[None, Tuple[u.Quantity, u.Quantity], Tuple[None, None]] = None
    ylim: Union[None, Tuple[u.Quantity, u.Quantity], Tuple[None, None]] = None
    xunit: Union[None, u.Unit] = u.Unit(1)
    yunit: Union[None, u.Unit] = u.Unit(1)
    unit_fmt: str = '({:latex_inline})'

    def __post_init__(self):
        # Check units
        if self.xlim is not None:
            self.xlim = (self._check_unit(self.xlim[0], self.xunit),
                         self._check_unit(self.xlim[1], self.xunit))
        if self.ylim is not None:
            self.ylim = (self._check_unit(self.ylim[0], self.yunit),
                         self._check_unit(self.ylim[1], self.yunit))

        super().__post_init__()

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

@dataclasses.dataclass
class VScaleProps:
    """Store intensity scale and color bar properties."""
    name: str = 'Intensity'
    """Color bar name."""
    name_cbar2: Optional[str] = None
    """Color bar name of second axis."""
    vmin: Optional[float] = None
    """Lower limit of color intensity."""
    vmin_cbar2: Optional[float] = None
    """Lower limit of color intensity of bar second axis."""
    vmax: Optional[float] = None
    """Upper limit of color intensity."""
    vmax_cbar2: Optional[float] = None
    """Upper limit of color intensity of bar second axis."""
    vcenter: Optional[float] = None
    """Mid value for midnorm stretch."""
    a: float = 1000
    """Scaling for log stretch."""
    stretch: str = 'linear'
    """Stretch of color intensity."""
    compute_ticks: bool = False
    """Compute the ticks or use default?"""
    nticks: int = 5
    """Number of ticks for auto ticks."""
    ticks: Optional[Sequence[float]] = None
    """Color bar ticks."""
    ticklabels: Optional[Sequence[str]] = None
    """Tick labels."""
    tickstretch: Optional[str] = None
    """Stretch of the ticks."""
    label: Optional[str] = None
    """Color bar label."""
    labelpad: float = 0
    """Shift the color bar label."""
    labelpad_cbar2: float = 0
    """Shift the second color bar label."""
    label_cbar2: Optional[str] = None
    """Label of the second axis of the color bar."""
    ticks_cbar2: Optional[Sequence[float]] = None
    """Ticks of the second axis of the color bar."""
    norm_cbar2: Optional['Normalization'] = None
    """Normalization function of the second axis of the color bar."""

    def __post_init__(self):
        # Define ticks
        if self.tickstretch is None:
            self.tickstretch = self.stretch
        #self.generate_ticks()

    def get_ticks(self, generate_cbar2: bool = False) -> Tuple:
        """Generate the colorbar ticks."""
        if (self.ticks is None and self.vmin is not None and
            self.vmax is not None):
            self.ticks = get_colorbar_ticks(self.vmin, self.vmax, a=self.a,
                                            n=self.nticks,
                                            stretch=self.tickstretch)
            ticks = self.ticks

        if self.ticks_cbar2 is None and generate_cbar2:
            self.generate_cbar2()

        return self.ticks, self.ticks_cbar2

    def generate_cbar2(self, unit_fmt: str = '({:latex_inline})'):
        """Generate second colorbar."""
        pass

    def get_normalization(self, vmin: Optional[float] = None,
                          vmax: Optional[float] = None,
                          vcenter: Optional[float] = None) -> Colormap:
        """Determine the normalization of the color stretch.

        Args:
          vmin: optional; scale minimum.
          vmax: optional; scale maximum.
          vcenter: optional; `midnorm` scale center.
        """
        if vmin is None:
            vmin = self.vmin
        if vmax is None:
            vmax = self.vmax
        if vcenter is None:
            vcenter = self.vcenter
        if self.stretch == 'log':
            return ImageNormalize(vmin=vmin, vmax=vmax,
                                  stretch=LogStretch(a=self.a))
        elif self.stretch == 'midnorm':
            return TwoSlopeNorm(vcenter, vmin=vmin, vmax=vmax)
        else:
            return ImageNormalize(vmin=vmin, vmax=vmax, stretch=LinearStretch())

@dataclasses.dataclass
class PhysVScaleProps(VScaleProps):
    """Store intensity scale and color bar properties."""
    unit: Optional[u.Unit] = None
    """Intensity axis unit."""
    unit_cbar2: Optional[u.Unit] = None
    """Intensity axis unit for second color bar."""
    unit_equiv: Optional[Callable] = None
    """Equivalency between units of both color bars."""
    vmin: Optional[u.Quantity] = None
    """Lower limit of colot intensity."""
    vmax: Optional[u.Quantity] = None
    """Upper limit of color intensity."""
    vcenter: Optional[u.Quantity] = None
    """Mid value for midnorm stretch."""
    ticks: Optional[u.Quantity] = None
    """Color bar ticks."""
    ticks_cbar2: Optional[u.Quantity] = None
    """Ticks of the second axis of the color bar."""

    def __post_init__(self):
        # Check limits units
        self.check_scale_units()

        # Generate ticks
        super().__post_init__()
        if self.ticks is not None:
            self.ticks = self.ticks.to(self.unit)

        # Generate label
        if self.label is None and self.unit is not None:
            self.label = self.generate_cblabel()

    def set_unit(self, value: u.Unit) -> None:
        """Safe unit setter.

        It ensures that set values are all in the same units.
        """
        self.unit = value
        self.check_scale_units()

    def get_vlim(self, axis: int = 1) -> Tuple[float, float]:
        """Get limits for the intensity axis.

        Args:
          axis: optional; 1 for the main axis, 2 for second axis.
        """
        if axis == 1:
            return self.vmin.value, self.vmax.value
        elif axis == 2:
            return self.vmin_cbar2.value, self.vmax_cbar2.value
        else:
            raise ValueError(f'Axis {axis} not recognized')

    def get_normalization(self, vmin: Optional[u.Quantity] = None,
                          vmax: Optional[u.Quantity] = None) -> Colormap:
        """Determine the normalization of the color stretch.

        Args:
          vmin: optional; scale minimum.
          vmax: optional; scale maximum.
        """
        if vmin is None:
            vmin = self.vmin
        if vmax is None:
            vmax = self.vmax
        if self.vcenter is not None:
            vcenter = self.vcenter.value
        else:
            vcenter = None

        return super().get_normalization(vmin=vmin.value, vmax=vmax.value,
                                         vcenter=vcenter)

    def get_ticks(self, generate_cbar2: bool = False) -> Tuple:
        """Get the color bar ticks."""
        generate_cbar2 = self.unit_cbar2 is not None
        super().get_ticks(generate_cbar2=generate_cbar2)
        self.ticks = self.ticks.to(self.unit)

        return self.ticks.to(self.unit).value, self.ticks_cbar2.value

    def check_scale_units(self):
        """Check units of the values defining the intensity scale."""
        if self.unit is not None:
            if self.vmin is not None:
                self.vmin = self.vmin.to(self.unit)
            if self.vmax is not None:
                self.vmax = self.vmax.to(self.unit)
            if self.vcenter is not None:
                self.vcenter = self.vcenter.to(self.unit)
            if self.ticks is not None:
                self.ticks = self.ticks.to(self.unit)

    def generate_cbar2(self, unit_fmt: str = '({:latex_inline})'):
        """Fill the properties of second colorbar."""
        self.vmin_cbar2 = self.vmin.to(self.unit_cbar2,
                                       equivalencies=self.unit_equiv)
        self.vmax_cbar2 = self.vmax.to(self.unit_cbar2,
                                  equivalencies=self.unit_equiv)
        self.ticks_cbar2 = self.ticks.to(self.unit_cbar2,
                                         equivalencies=self.unit_equiv)
        self.label_cbar2 = generate_label(self.name_cbar2, unit=self.unit_cbar2,
                                          unit_fmt=unit_fmt)
        self.norm_cbar2 = self.get_normalization(vmin=self.vmin_cbar2,
                                                 vmax=self.vmax_cbar2)
        #self.norm_cbar2 = self.get_normalization(
        #    vmin=np.min(self.ticks_cbar2),
        #    vmax=np.max(self.ticks_cbar2),
        #)

    def generate_cblabel(self, unit_fmt: str = '({:latex_inline})') -> str:
        """Generate color bar label."""
        return generate_label(self.name, unit=self.unit, unit_fmt=unit_fmt)

