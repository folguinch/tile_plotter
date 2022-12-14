"""Objects to configure and manage plot geometries."""
from typing import List, Optional, Sequence, Union, Tuple
import collections
import itertools
from dataclasses import dataclass, field

from toolkit.logger import LoggedObject
import configparseradv.utils as cfgutils

from .handlers import PlotHandler

# Type Aliases
Position = Tuple[float, float]
Location = Tuple[int, int]

@dataclass
class BaseGeometry:
    """Figure geometry class.

    Stores the dimesions and position of an axis.

    Attributes:
      xsize: x axis size.
      ysize: y axis size.
      left: left margin.
      right: right margin.
      bottom: bottom margin.
      top: top margin.
      position: axis position within figure.
    """
    xsize: float
    ysize: float
    left: float = 0
    right: float = 0
    top: float = 0
    bottom: float = 0
    position: Position = field(default_factory=tuple)

    # To be remove in Python 3.7+
    #def __init__(self,
    #             xsize: float,
    #             ysize: float,
    #             left: float = 0,
    #             right: float = 0,
    #             bottom: float = 0,
    #             top: float = 0,
    #             position: Position = (0., 0.)) -> None:
    #    """Initiate geometry with input parameters."""
    #    self.xsize = xsize
    #    self.ysize = ysize
    #    self.left = left
    #    self.right = right
    #    self.bottom = bottom
    #    self.top = top
    #    self.position = position

    # To be remove in Python 3.7+
    #def __repr__(self) -> str:
    #    text = [f'x size: {self.xsize}']
    #    text.append(f'y size: {self.ysize}')
    #    text.append(f'left margin: {self.left}')
    #    text.append(f'right margin: {self.rigth}')
    #    text.append(f'bottom margin: {self.bottom}')
    #    text.append(f'top margin: {self.top}')
    #    text.append(f'axis position: {self.position}')

    #    return '\n'.join(text)

    def __post_init__(self):
        if len(self.position) == 0:
            self.position = (0., 0.)

    def __setitem__(self,
                    key: str,
                    val: Union[float, Location]) -> None:
        if key == 'xsize':
            self.xsize = val
        elif key == 'ysize':
            self.ysize = val
        elif key == 'left':
            self.left = val
        elif key == 'right':
            self.right = val
        elif key == 'bottom':
            self.bottom = val
        elif key == 'top':
            self.top = val
        elif key == 'position':
            self.position = val
        else:
            raise KeyError(f'Cannot set {key}')

    @property
    def height(self) -> float:
        return self.bottom + self.ysize + self.top

    @property
    def width(self) -> float:
        return self.left + self.xsize + self.right

    @property
    def pyplot_axis(self) -> List[float]:
        return [self.position[0]+self.left, self.position[1]+self.bottom,
                self.xsize, self.ysize]

    def scale_xaxis(self,
                    factor: float,
                    original: Optional[float] = None) -> float:
        """Scale the x axis and keep the axis centered.

        Args:
          factor: scaling factor.
          original: optional; original size for recentering. If None then the
            current size is used.

        Returns:
          Amount of shift applied to the axis in order to center in relation to
          the other axes in the horizontal direction.
        """
        # Shifts
        newsize = factor * self.xsize
        if original is None:
            original = self.xsize
        dleft = abs(newsize - original) / 2.

        # Apply
        self.xsize = newsize
        self.left = self.left + dleft
        self.right = self.right + dleft

        return dleft

    def scale_yaxis(self,
                    factor: float,
                    original: Optional[float] = None) -> float:
        """Scale the y axis and keep the axis centered.

        Args:
          factor: scaling factor.
          original: optional; original size for recentering. If None then the
            current size is used.

        Returns:
          Amount of shift applied to the axis in order to center in relation to
          the other axes in the vertical direction.
        """
        # Shifts
        newsize = factor * self.ysize
        if original is None:
            original = self.ysize
        dbottom = abs(newsize - original) / 2.

        # Apply
        self.ysize = newsize
        self.bottom = self.bottom + dbottom
        self.top = self.top + dbottom

        return dbottom

    def scalex(self, factor: float) -> None:
        """Scale all the horizontal variables by factor."""
        self.xsize = self.xsize * factor
        self.left = self.left * factor
        self.right = self.right * factor
        xposition = self.position[0]*factor
        self.position = (xposition, self.position[1])

    def scaley(self, factor: float) -> None:
        """Scale all the vertical variables by factor."""
        self.ysize = self.ysize*factor
        self.bottom = self.bottom*factor
        self.top = self.top*factor
        yposition = self.position[1]*factor
        self.position = (self.position[0], yposition)

    def is_empty(self) -> bool:
        """Is the geometry empty?"""
        condition1 = self.xsize == self.ysize == self.left == self.right
        condition2 = self.bottom == self.top == 0
        return condition1 == condition2

class AxisHandler:
    """Stores the axes of a plot.

    Attributes:
      axis: main axis.
      cbaxis: colorbar axis.
      cborientation: orientation of the colorbar.
    """

    def __init__(self,
                 axis: Optional[BaseGeometry] = None,
                 cbaxis: Optional[BaseGeometry] = None,
                 cbar_orientation: Optional[str] = None,
                 ) -> None:
        """Initiate the figure geometry."""
        self.axis = axis
        self.cbaxis = cbaxis
        self.cborientation = None
        if cbar_orientation is not None:
            self.set_cbar(cbar_orientation)
        self._handler = None

    def __str__(self):
        lines = [f'Axis: {self.axis}',
                 f'Color bar:{self.cbaxis}',
                 f'Color bar orientation: {self.cborientation}']
        return '\n'.join(lines)

    @property
    def width(self) -> float:
        """Returns the total width."""
        if self.axis is not None:
            width = self.axis.width
        else:
            raise ValueError('Axis not initialized')
        if self.has_vertical_cbar():
            width += self.cbaxis.width
        return width

    @property
    def height(self) -> float:
        """Return the total height."""
        if self.axis is not None:
            height = self.axis.height
        else:
            raise ValueError('Axis not initialized')
        if self.has_horizontal_cbar():
            height += self.cbaxis.height
        return height

    @property
    def dimensions(self) -> Tuple[float, float]:
        """The total width and height of the geometry."""
        return self.width, self.height

    @property
    def handler(self) -> PlotHandler:
        return self._handler

    def set_cbar(self, orientation: Union[None, str]) -> None:
        """Validate color bar value.

        Args:
          orientation: orientation of the color bar.

        Raises:
            ValueError: if orientation not in [None, vertical, horizontal].
        """
        if orientation is not None:
            if orientation.lower() not in ['vertical', 'horizontal']:
                raise ValueError(('Colorbar orientation not recognized: '
                                  f'{orientation}'))
            self.cborientation = orientation.lower()
        else:
            self.cborientation = orientation

    def unset_cbar(self,
                   sharex: bool = False,
                   sharey: bool = False) -> None:
        """Delete color bar axis.

        Args:
          sharex: optional; is the x-axis shared?
          sharey: optional; is the y-axis shared?
        """
        if self.has_vertical_cbar():
            if sharey:
                self.axis.right = 0
            else:
                self.axis.right = self.cbaxis.right
        elif self.has_horizontal_cbar():
            if sharex:
                self.axis.top = 0
            else:
                self.axis.top = self.cbaxis.top
        self.cborientation = None
        self.cbaxis = None

    def has_cbar(self) -> bool:
        return self.cborientation is not None

    def has_vertical_cbar(self) -> bool:
        return self.has_cbar() and self.cborientation.lower() == 'vertical'

    def has_horizontal_cbar(self) -> bool:
        return self.has_cbar() and self.cborientation.lower() == 'horizontal'

    def set_handler(self, handler: PlotHandler) -> None:
        """Set the handler value."""
        self._handler = handler

    def init_geometry(self,
                      xsize: float,
                      ysize: float,
                      left: float,
                      right: float,
                      top: float,
                      bottom: float,
                      cbar_orientation: Optional[str] = None,
                      cbar_width: Optional[float] = None,
                      cbar_spacing: Optional[float] = None,
                      position: Position = (0., 0.)) -> None:
        """Initialize axes with a BaseGeometry from input values.

        Args:
          xsize: x axis size.
          ysize: y axis size.
          left: left margin.
          right: right margin.
          top: top margin.
          bottom: margin.
          cbar_orientation: optional; color bar orientation.
          cbar_width: optional; color bar width.
          cbar_spacing: optional; separation between main axis and color bar.
          position: optional; position of the axis.
        """

        # Main axis
        self.axis = BaseGeometry(xsize, ysize, left=left, right=right,
                                 bottom=bottom, top=top, position=position)

        # Color bar
        self.set_cbar(cbar_orientation)
        if self.has_vertical_cbar():
            self.cbaxis = BaseGeometry(cbar_width, ysize, left=cbar_spacing,
                                       right=right, bottom=bottom, top=top,
                                       position=[self.axis.width, position[1]])
            self.axis.right = 0
        elif self.has_horizontal_cbar():
            self.cbaxis = BaseGeometry(xsize, cbar_width, left=left,
                                       right=right, bottom=cbar_spacing,
                                       top=top,
                                       position=[position[0], self.axis.height])
            self.axis.top = 0

    def geometry_from_config(self, config: 'configparseradv') -> None:
        """Initialize axes from config file.

        Args:
          config: configuration parser proxy.
        """
        # Keys
        space_keys = ['xsize', 'ysize']
        axis_keys = ['left', 'right', 'top', 'bottom']
        cbar_keys = ['cbar_width', 'cbar_spacing']

        # Values
        xsize, ysize = cfgutils.get_floatkeys(config, space_keys)
        left, right, top, bottom = cfgutils.get_floatkeys(config, axis_keys)
        cbar_width, cbar_spacing = cfgutils.get_floatkeys(config, cbar_keys)

        # Color bar
        vcbar = config.getboolean('vertical_cbar')
        hcbar = config.getboolean('horizontal_cbar')
        if vcbar and hcbar:
            raise ValueError('Colorbars not allowed in both axes')
        elif vcbar:
            cbar_orientation = 'vertical'
        elif hcbar:
            cbar_orientation = 'horizontal'
        else:
            cbar_width = cbar_spacing = None
            cbar_orientation = None

        # Generate geometries
        self.init_geometry(xsize, ysize, left, right, top,
                           bottom, cbar_orientation=cbar_orientation,
                           cbar_width=cbar_width, cbar_spacing=cbar_spacing)

    def set_spacing(self,
                    left_spacing: float = 0,
                    bottom_spacing: float = 0,
                    sharex: bool = False,
                    sharey: bool = False,
                    is_top: bool = False,
                    is_bottom: bool = False,
                    is_left: bool = False,
                    is_right: bool = False) -> None:
        """Update geometries for a shared axis.

        Left and bottom spacing are not added if the axis in the left or bottom
        margins, respectively. If axes are shared, then the respective values
        replace the current `left` and `bottom` values, while `top` and `right`
        values are set to zero if the position is not in the figure top and
        right margins.

        Args:
          left_spacing: optional; space at the left of the axis.
          bottom_spacing: optional; space at the bottom of the axis.
          sharex: is the x axis shared?
          sharey: is the y axis shared?
          is_top: is the axis at the topmost row of the figure?
          is_bottom: is the axis at the bottom row of the figure?
          is_left: is the axis at the leftmost column of the figure?
          is_right: is the axis at the rightmost column of the figure?
        """
        # Check shared axes
        if sharex:
            if not is_bottom:
                self.axis.bottom = bottom_spacing
            if not is_top:
                self.axis.top = 0
        elif not is_bottom:
            self.axis.bottom += bottom_spacing
        if sharey:
            if not is_left:
                self.axis.left = left_spacing
            if not is_right:
                self.axis.right = 0
        elif not is_left:
            self.axis.left += left_spacing

        # Update color bar
        if self.has_vertical_cbar():
            self.cbaxis.bottom = self.axis.bottom
            self.cbaxis.top = self.axis.top
            self.cbaxis.position[0] = self.axis.width
        elif self.has_horizontal_cbar():
            self.cbaxis.left = self.axis.left
            self.cbaxis.right = self.axis.right
            self.cbaxis.position[1] = self.axis.height

    def shift_position(self, xshift: float = 0., yshift: float = 0.) -> None:
        """Shift the positions of the axes.

        Args:
          xshift: shift for the x position.
          yshift: shift for the y position.
        """
        self.axis.position = [self.axis.position[0] + xshift,
                              self.axis.position[1] + yshift]
        if self.has_cbar():
            self.cbaxis.position = [self.cbaxis.position[0] + xshift,
                                    self.cbaxis.position[1] + yshift]

    def scale_axes(self,
                   xfactor: float = 1.,
                   yfactor: float = 1.,
                   xoriginal: Optional[float] = None,
                   yoriginal: Optional[float] = None) -> None:
        """Scale the axes.

        Args:
          xfactor: x axis scaling factor.
          yfactor: y axis scaling factor.
          xoriginal: optional; original x axis width for recentering.
          yoriginal: optional; original y axis height for recentering.
        """
        # Scale axis
        dleft = self.axis.scale_xaxis(xfactor, original=xoriginal)
        dbottom = self.axis.scale_yaxis(yfactor, original=yoriginal)

        # Scale color bar
        if self.has_vertical_cbar():
            self.cbax.ysize = self.axis.ysize
            self.cbax.top = self.axis.top
            self.cbax.bottom = self.axis.bottom
            self.cbax.right += dleft
            self.cbax.position[0] += dleft
            self.axis.right = 0
        elif self.has_horizontal_cbar():
            self.cbax.xsize = self.axis.xsize
            self.cbax.left = self.axis.left
            self.cbax.right = self.axis.right
            self.cbax.top += dbottom
            self.cbax.position[1] += dbottom
            self.axis.top = 0
        else:
            pass

@dataclass
class GeometryHandler(LoggedObject, collections.OrderedDict):
    """Manage the geometries in a figure.

    Attributes:
      nrows: number of rows.
      ncols: number of columns.
      sharex: are x axes shared?
      sharey: are y axes shared?
      vspace: vertical spacing between axes.
      hspace: horizontal spacing between axes.
      vcbarpos: vertical color bar positions.
      hcbarpos: horizontal color bar positions.
    """
    nrows: int = 0
    ncols: int = 0
    sharex: bool = False
    sharey: bool = False
    vspace: int = 0
    hspace: int = 0
    vcbarpos: Optional[Tuple[int]] = None
    hcbarpos: Optional[Tuple[int]] = None
    verbose: str = 'v'
    #single_dimensions = None

    def __post_init__(self):
        super().__init__(__name__, filename='plotter.log', verbose=self.verbose)

    def __setitiem__(self, key: Sequence[int], value: 'axis') -> None:
        if len(key) != 2:
            raise KeyError(f'Key length {len(key)} != 2')
        elif key[0] >= self.nrows:
            raise KeyError(f'Row {key[0]} >= {self.nrows}')
        elif key[1] >= self.nrows:
            raise KeyError(f'Column {key[1]} >= {self.ncols}')
        super().__setitiem__(key, value)

    def keys_from_shape(self, rows: int, cols: int) -> None:
        """Fill the handler with an empty geometry."""
        self.nrows = rows
        self.ncols = cols
        for loc in itertools.product(range(rows, 0, -1), range(cols)):
            self[loc] = AxisHandler(0, 0)

    def from_config(self, config: 'configparseradv') -> None:
        """Initiate spatial values from configuration parser."""
        # Useful values
        self.nrows, self.ncols = cfgutils.get_intkeys(config,
                                                      ['nrows', 'ncols'])
        self.vspace, self.hspace = cfgutils.get_floatkeys(config,
                                                          ['vspace', 'hspace'])
        # pylint: disable=unbalanced-tuple-unpacking
        self.sharex, self.sharey = cfgutils.get_boolkeys(config,
                                                         ['sharex', 'sharey'])
        vcbarpos, hcbarpos = cfgutils.get_keys(config,
                                               ['vcbarpos', 'hcbarpos'])

        # Colorbar positions
        if vcbarpos == '*':
            self.vcbarpos = tuple(range(self.ncols))
        else:
            self.vcbarpos = tuple(map(int, vcbarpos.replace(',',' ').split()))
        if hcbarpos == '*':
            self.hcbarpos = tuple(range(self.nrows))
        else:
            self.hcbarpos = tuple(map(int, hcbarpos.replace(',',' ').split()))

    def fill_from_config(self,
                         config: 'conifgparseradv.configarser.ConfigParserAdv'
                         ) -> None:
        """Fill the dictionary with `AxisHandler` from configuration parser."""
        # Update stored values
        self.from_config(config)

        # Fill the geometry
        cumx, cumy = 0, 0
        xdim = None
        # Star from the last row to accumulate the value of bottom
        ranges = range(self.nrows - 1, -1, -1),  range(self.ncols)
        for loc in itertools.product(*ranges):
            # Initialize and get dimensions
            self._log.debug('Defining axis at location: %s', loc)
            self.init_loc(loc, config, cumx, cumy)

            # Cumulative sums
            if loc[1] == self.ncols - 1:
                if xdim is None:
                    xdim = cumx + self[loc].width
                cumx = 0
                if loc[0] != 0:
                    cumy += self[loc].height
                else:
                    ydim = cumy + self[loc].height
            else:
                cumx += self[loc].width
            self._log.debug('=====')
            self._log.debug('Cumulative axis: %f, %f', cumx, cumy)
            self._log.debug('=====')
        return xdim, ydim

    def init_loc(self,
                 loc: Location,
                 config: 'ConfigParserAdv',
                 xshift: float = 0,
                 yshift: float = 0) -> Tuple[float, float]:
        """Initiate a single `AxisHandler` at given location from configuration.

        Args:
          loc: location of the axis.
          config: configuration parser proxy.
          xshift: optional; shift of the x axis location.
          yshift: optional; shift of the y axis location.

        Returns:
          The dimensions of the axis.
        """
        # Initial value
        self[loc] = AxisHandler()
        self[loc].geometry_from_config(config)
        self._log.debug('Initial new axes:\n%s', self[loc])

        # Scale axis
        options = [f'{loc[0]}*', f'*{loc[1]}', ''.join(map(str,loc))]
        xfactor, yfactor = None, None
        for opt in options:
            key1 = f'xsize_factor{opt}'
            key2 = f'ysize_factor{opt}'
            xfactor = config.getfloat(key1, fallback=None)
            yfactor = config.getfloat(key2, fallback=None)
        if xfactor or yfactor:
            self._log.debug('Scaling axes by %f x %f', xfactor, yfactor)
            xfactor = xfactor or 1.0
            yfactor = yfactor or 1.0
            self[loc].scale_axes(xfactor, yfactor)
            self._log.debug('Scaled axis:\n%s', self[loc])

        # Set spacing
        self.set_spacing(loc)
        self._log.debug('Set axes spacing:\n%s', self[loc])

        # Unset cbar
        self.remove_cbar(loc)
        self._log.debug('Updated color bar:\n%s', self[loc])

        # Shift position
        self[loc].shift_position(xshift=xshift, yshift=yshift)
        self._log.debug('Shifted positions:\n%s', self[loc])

        # Return dimensions
        self._log.debug('New axis dimensions: %s', self[loc].dimensions)
        return self[loc].dimensions

    def set_spacing(self, loc: Location) -> None:
        """Set the space between axes."""
        is_top = loc[0] == 0
        is_bottom = loc[0] == self.nrows-1
        is_left = loc[1] == 0
        is_right = loc[1] == self.ncols-1

        self[loc].set_spacing(left_spacing=self.hspace,
                              bottom_spacing=self.vspace,
                              sharex=self.sharex,
                              sharey=self.sharey,
                              is_top=is_top,
                              is_bottom=is_bottom,
                              is_left=is_left,
                              is_right=is_right)

    def remove_cbar(self, loc: Location) -> None:
        """Remove color bar and adjust spacing."""
        # Base case
        if not self[loc].has_cbar():
            return

        # Remove cbar and apply corrections
        if self[loc].has_vertical_cbar():
            has_cbar = (loc[1] in self.vcbarpos or
                        loc[1]-self.ncols in self.vcbarpos)
        elif self[loc].has_horizontal_cbar():
            has_cbar = (loc[0] in self.hcbarpos or
                        loc[0]-self.nrows in self.hcbarpos)
        else:
            has_cbar = False
        if not has_cbar:
            self._log.debug('Unsetting cbar')
            self[loc].unset_cbar(self.sharex, self.sharey)

