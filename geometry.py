from typing import List
from itertools import product
from collections import OrderedDict
# Future update for Python 3.7+
# from dataclasses import dataclass

#@dataclass
class BaseGeometry(object):
    """Figure geometry class"""
    # Future update for Python 3.7+
    #xsize: float
    #ysize: float
    #left: float = 0
    #right: float = 0
    #top: float = 0
    #bottom: float = 0
    #location: List[float] = [0, 0]

    # To be remove in Python 3.7+
    def __init__(self, xsize: float, ysize: float, left: float = 0, 
            right: float = 0, bottom: float = 0, top: float = 0, 
            location: List[float] = [0, 0]):
        self.xsize = xsize
        self.ysize = ysize
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.location = location

    # To be remove in Python 3.7+
    def __repr__(self):
        text = f'x size: {self.xsize}\n'
        text += f'y size: {self.ysize}\n'
        text += f'left margin: {self.left}\n'
        text += f'right margin: {self.rigth}\n'
        text += f'bottom margin: {self.bottom}\n'
        text += f'top margin: {self.top}\n'
        text += f'axis location: {self.location}\n'

        return text

    def __setitem__(self, key, val):
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
        elif key == 'location':
            self.location = val
        else:
            raise KeyError(f'Cannot set {key}')

    @property
    def height(self):
        return self.bottom + self.ysize + self.top

    @property
    def width(self):
        return self.left + self.xsize + self.right

    @property
    def pyplot_axis(self):
        return [self.location[0]+self.left, self.location[1]+self.bottom, self.xsize, self.ysize]

    def scale_xaxis(self, factor: float, original: float = None) -> float:
        """Scale the x axis and keep the axis centered
        
        Parameters:
            factor: scaling factor
            original: original size for recentering. If None then the current
                size is used
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

    def scale_yaxis(self, factor: float, original: float = None) -> float:
        """Scale the y axis and keep the axis centered
        
        Parameters:
            factor: scaling factor
            original: original size for recentering. If None then the current
                size is used
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

    def scalex(self, factor: float):
        """Scale all the horizontal variables by factor"""
        self.xsize = self.xsize * factor
        self.left = self.left * factor
        self.right = self.right * factor
        self.location[0] = self.location[0]*factor

    def scaley(self, factor: float):
        """Scale all the vertical variables by factor"""
        self.ysize = self.ysize*factor
        self.bottom = self.bottom*factor
        self.top = self.top*factor
        self.location[1] = self.location[1]*factor

    def is_empty(self):
        return self.xsize==self.ysize==self.left==self.right==self.bottom==self.top==0

class FigGeometry(object):

    def __init__(self, axis=None, cbaxis=None, cbar_orientation=None):
        self.axis = axis
        self.cbaxis = cbaxis
        self.cborientation = None
        if cbar_orientation is not None:
            self.set_cbar(cbar_orientation)
    
    @property
    def width(self):
        if self.axis is not None:
            width = self.axis.width
        else:
            raise ValueError('Axis not initialized')
        if self.has_vertical_cbar():
            width += self.cbaxis.width
        return width

    @property
    def height(self):
        if self.axis is not None:
            height = self.axis.height
        else:
            raise ValueError('Axis not initialized')
        if self.has_horizontal_cbar():
            height += self.cbaxis.height
        return height

    @property
    def dimensions(self):
        return self.width, self.height

    def set_cbar(self, orientation):
        """Validate color bar value"""
        if orientation is not None:
            if orientation.lower() not in ['vertical', 'horizontal']:
                raise ValueError('Colorbar orientation not recognized')
            self.cborientation = orientation.lower()
        else:
            self.cborientation = orientation

    def unset_cbar(self, sharex: bool = False, sharey: bool = False):
        """Delete color bar axis"""
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

    def init_geometry(self, xsize: float, ysize: float, left: float, 
            right: float, top: float, bottom: float, 
            cbar_orientation: str = None, cbar_width: float = None, 
            cbar_spacing: float = None, location: List[float] = [0, 0]) -> None:
        """Initialize axes from input values"""

        # Main axis
        self.axis = BaseGeometry(xsize, ysize, left, right, bottom, top,
                location)

        # Color bar
        self.set_cbar(cbar_orientation)
        if self.has_vertical_cbar():
            self.cbaxis = BaseGeometry(cbar_width, ysize, cbar_spacing, right,
                    bottom, top, [self.axis.width, location[1]])
            self.axis.right = 0
        elif self.has_horizontal_cbar():
            self.cbaxis = BaseGeometry(xsize, cbar_width, left, right,
                    cbar_spacing, top, [location[0], self.axis.height])
            self.axis.top = 0

    def geometry_from_config(self, config: 'configparseradv proxy') -> None:
        """Initialize axes from config file"""
        # Keys
        space_keys = ['xsize', 'ysize']
        axis_keys = ['left', 'right', 'top', 'bottom']
        cbar_keys = ['cbar_width', 'cbar_spacing']

        # Values
        xsize, ysize = config.getfloatkeys(space_keys)
        left, right, top, bottom = config.getfloatkeys(axis_keys)
        cbar_width, cbar_spacing = config.getbooleankeys(cbar_keys)

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

    def set_spacing(self, left_spacing: float = 0, bottom_spacing: float = 0,
            sharex: bool = False, sharey: bool = False, is_top: bool = False,
            is_right: bool = False) -> None:
        """Update geometries for a shared axis"""
        # Check shared axes
        if sharex:
            self.axis.bottom = bottom_spacing
            if not is_top:
                self.axis.top = 0
        else:
            self.axis.bottom += bottom_spacing
        if sharey:
            self.axis.left = left_spacing
            if not is_right:
                self.axis.right = 0
        else:
            self.axis.left += left_spacing

        # Update color bar
        if self.has_vertical_cbar():
            self.cbaxis.bottom = self.axis.bottom
            self.cbaxis.top = self.axis.top
            cbax.location[0] = self.axis.width
        elif self.has_horizontal_cbar():
            self.cbaxis.left = self.axis.left
            self.cbaxis.right = self.axis.right
            cbax.location[1] = self.axis.height

    def shift_location(self, xshift: float = 0, yshift: float = 0) -> None:
        """Shift the locations of the axes"""
        self.axis.location = [self.axis.location[0] + xshift,
                self.axis.location[1] + yshift]
        if self.has_cbar():
            self.cbaxis.location = [self.cbaxis.location[0] + xshift,
                    self.cbaxis.location[1] + yshift]

    def scale_axes(self, xfactor: float = 1, yfactor: float = 1, 
            xoriginal: float = None, yoriginal: float = None) -> None:
        """Scale the axes"""
        # Scale axis
        dleft = self.axis.scale_xaxis(xfactor, original=xoriginal)
        dbottom = self.axis.scale_yaxis(yfactor, original=yoriginal)

        # Scale color bar
        if self.has_vertical_cbar():
            self.cbax.ysize = self.axis.ysize
            self.cbax.top = self.axis.top
            self.cbax.bottom = self.axis.bottom
            self.cbax.right += dleft
            self.cbax.location[0] += dleft
            self.axis.right = 0
        elif self.has_horizontal_cbar():
            self.cbax.xsize = self.axis.xsize
            self.cbax.left = self.axis.left
            self.cbax.right = self.axis.right
            self.cbax.top += dbottom
            self.cbax.location[1] += dbottom
            self.axis.top = 0
        else:
            pass

class GeometryHandler(OrderedDict):
    nrows: int = 0
    ncols: int = 0
    sharex: bool = False
    sharey: bool = False
    vspace: int = 0
    hspace: int = 0
    vcbarpos = None
    hcbarpos = None
    single_dimensions = None

    def __setitiem__(self, key, value):
        if len(key) != 2:
            raise KeyError(f'Key length {len(key)} != 2')
        elif key[0] >= self.nrows:
            raise KeyError(f'Row {key[0]} >= {self.nrows}')
        elif key[1] >= self.nrows:
            raise KeyError(f'Column {key[1]} >= {self.ncols}')
        super().__setitiem__(key, value)

    def keys_from_shape(self, rows: int, cols: int) -> None:
        """Fill the handler with an empty geometry"""
        self.nrows = rows
        self.ncols = cols
        for loc in product(range(rows, 0, -1), range(cols)):
            self[loc] = FigGeometry(0, 0)

    def from_config(self, config: 'configparseradv') -> None:
        """Initiate spatial values"""
        # Useful values
        self.nrows, self.ncols = config.getintkeys(['nrows', 'ncols'])
        self.vspace, self.hspace = config.getfloatkeys(['vspace', 'hspace'])
        self.sharex, self.sharey = config.getbollkeys(['sharex', 'sharey'])
        vcbarpos, hcbarpos = config.getkeys(['vcbarpos', 'hcbarpos'])

        # Colorbar positions
        if vcbarpos == '*':
            self.vcbarpos = list(range(self.ncols))
        else:
            self.vcbarpos = list(map(int, vcbarpos.replace(',',' ').split()))
        if hcbarpos == '*':
            self.hcbarpos = list(range(self.nrows))
        else:
            self.hcbarpos = list(map(int, hcbarpos.replace(',',' ').split()))

    def fill_from_config(self, config: 'configparseradv') -> None:
        """Fill the dictionary with FigGeometry from config"""
        # Update stored values
        self.from_config(config)

        # Fill the geometry
        cumx, cumy = 0, 0
        xdim = None
        # Star from the last row to accumulate the value of bottom
        for loc in product(range(self.nrows-1,0,-1), range(self.ncols)):
            # Initialize and get dimensions
            width, height = self.init_loc(loc, config, cumx, cumy)

            # Cumulative sums
            if loc[1] == self.ncols-1:
                if xdim is None:
                    xdim = cumx + self[loc].width
                cumx = 0
                if loc[0] != 0:
                    cumy += self[loc].height
                else:
                    ydim = cumy + self[loc].height
            else:
                cumx += self[loc].width
        return xdim, ydim

    def init_loc(self, loc: tuple, config: 'configparseradv', 
            xshift: float = 0, yshift: float = 0) -> None:
        """Initiate a single FigGeometry at given location"""
        # Initial value
        self[loc] = FigGeometry()
        self[loc].geometry_from_config(config)

        # Scale axis
        options = [f'{loc[0]}*', f'*{loc[1]}', '{}{}'.format(*loc)]
        xfactor, yfactor = None, None
        for opt in options:
            key1 = f'xsize_factor{opt}'
            key2 = f'ysize_factor{opt}'
            xfactor = config.getfloat(key1, fallback=None)
            yfactor = config.getfloat(key2, fallback=None)
        if xfactor or yfactor:
            xfactor = xfactor or 1.0
            yfactor = yfactor or 1.0
            self[loc].scale_axes(xfactor, yfactor)

        # Set spacing
        self.set_spacing(loc)

        # Unset cbar
        self.remove_cbar(loc)

        # Shift location
        self[loc].shift_location(xshift=xshift, yshift=yshift)

        # Return dimensions
        return self[loc].dimensions

    def set_spacing(self, loc):
        """Set the space between axes"""
        if loc[0] == self.nrows-1:
            bottom_spacing = 0
            sharex = False
        else:
            bottom_spacing = self.vspace
            sharex = self.sharex
        if loc[1] == 0:
            left_spacing = 0
            sharey = False
        else:
            left_spacing = self.hspace
            sharey = self.sharey

        self[loc].set_spacing(left_spacing=left_spacing, 
                bottom_spacing=bottom_spacing, sharex=self.sharex, 
                sharey=self.sharey, is_top=loc[0]==0,
                is_right=loc[1]==self.ncols-1)

    def remove_cbar(self, loc):
        """Remove color bar and adjust spacing"""
        # Base case
        if not self[loc].has_cbar():
            return
        
        # Remove cbar and apply corrections
        new_right = 0
        new_top = 0
        if self[loc].has_vertical_cbar():
            has_cbar = loc[1] in self.vcbarpos or \
                    loc[1]-self.ncols in self.vcbarpos
        elif self[loc].has_horizontal_cbar():
            has_cbar = loc[0] in self.hcbarpos or \
                    loc[0]-self.nrows in self.hcbarpos
        else:
            has_cbar = False
        if not has_cbar:
            self[loc].unset_cbar(self.sharex, self.sharey)

