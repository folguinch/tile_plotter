"""Plotting utilities."""
from typing import Union, Optional, Callable, Tuple

from astropy.coordinates import SkyCoord
import astropy.stats as apystats
import astropy.units as u
import numpy as np

# Type aliases
Quantity = Union[u.Quantity, float, np.array]

def quick_rms(data: Quantity) -> Quantity:
    """Estimate the rms of the data using the MAD."""
    return 1.482602219 * apystats.median_absolute_deviation(data, 
                                                            ignore_nan=True)

def to_sigfig(x: Quantity, sigfig: int = 2) -> Quantity:
    """Round value to sigfig significant figures."""
    try:
        val = x.value
    except AttributeError:
        val = x
    ndigits = -int(np.floor(np.log10(val))) + (sigfig - 1)
    return np.round(x, ndigits)

def get_colorbar_ticks(vmin: u.Quantity, 
                       vmax: u.Quantity, 
                       a: float = 1000.,
                       n: int = 5, 
                       stretch: str = 'linear', 
                       ndigits: Optional[int] = None,
                       sigfig: Optional[int] = None) -> Quantity:
    """Calculate the tick values for the colorbar given the intensity stretch.

    Args:
      vmin: the minimum of the stretch.
      vmax: the maximum of the stretch.
      a: optional; scaling for logarithmic stretch.
      n: optional; number of ticks.
      stretch: optional; stretch type.
      ndigits: optional; value of digits for rounding.
      sigfig: optional; number of significant figures.
    
    Returns:
      A numpy array quantity with the ticks.
    """
    if stretch == 'log':
        x = lambda y: (10.**(y * np.log10(a + 1.)) - 1.) / a
        y = np.linspace(0., 1., n)
        x = x(y)
        ticks = x*(vmax - vmin) + vmin
    elif stretch == 'symlog':
        n = n//2 + 1
        linthresh = min(abs(vmin.value), abs(vmax.value))
        linthresh = linthresh / 10.
        if abs(vmin) < vmax:
            npos = n + 1
            nneg = n
        elif abs(vmin) > vmax:
            npos = n
            nneg = n + 1
        else:
            npos = nneg = n
        ypos = np.logspace(np.log10(linthresh), np.log10(vmax.value), npos)
        yneg = np.logspace(np.log10(linthresh), np.log10(abs(vmin.value)), nneg, 
                           endpoint=False)
        ticks = np.append(-yneg, [0])
        ticks = np.unique(np.append(ticks, ypos)) * vmin.unit
    elif stretch == 'linear':
        ticks = np.linspace(vmin, vmax, n)
    else:
        raise ValueError(f'Stretch {stretch} not recognized')

    # Round if needed
    if ndigits: 
        ticks = np.unique(np.round(ticks, ndigits))
    if sigfig:
        ticks = np.unique(to_sigfig(ticks, sigfig=ndigits))
    
    return ticks

def auto_vmin(data: u.Quantity, 
              rms: Optional[u.Quantity] = None, 
              nrms: float = 0.8, 
              velocity_fraction: float = 0.98, 
              map_type: str = 'intensity',
              log: Callable = print) -> u.Quantity:
    """Determine vmin from the data.

    If data_type=intensity, it calculates the rms from the data if not given.
    The value of vmin is rms*nrms in this case. If data_type=velocity, vmin is 
    a fraction of the minimum value of the data.

    Args:
      data: data to calculate vmin from.
      rms: optional; rms of the data.
      nrms: optional; number of rms values for vmin.
      velocity_fraction: optional; fraction of the minimum value of the data to
        use if data_type=velocity.
      map_type: optional; type of map.
      log: optional; logging function.
    """
    if map_type.lower() in ['intensity', 'pvmap']:
        if rms is None:
            rms = apystats.mad_std(data, ignore_nan=True)
            log(f'MAD std = {rms:.3e}')
        else:
            log(f'Input rms = {rms:.3e}')
        vmin = nrms * rms
    elif map_type.lower() == 'velocity':
        vmin = np.nanmin(data) * vfrac
    else:
        raise KeyError(f'Map type {map_type} not recognized')

    return to_sigfig(vmin.to(data.unit))

def auto_vmax(data: u.Quantity, 
              maximum: u.Quantity = None,
              fraction: float = 1.02, 
              map_type: str = 'intensity') -> u.Quantity:
    """Determine vmax from the data maximum.
    
    If maximum is given, then this value is used instead of the data. At the
    moment there is not any difference based on the map_type.
    
    Args:
      data: data to calculate vmax from.
      maximum: optional; maximum value.
      fraction: optional; fraction of the maximum for vmax.
      map_type: optional; type of map.
    """
    if maximum is not None:
        vmax = fraction * maximum
    else:
        vmax = np.nanmax(data) * fraction
    return to_sigfig(vmax.to(data.unit))

def auto_vminmax(data: u.Quantity, 
                 map_type: str = 'intensity', 
                 **kwargs) -> Tuple[u.Quantity, u.Quantity]:
    """Calculate vmin and vmax from data."""
    return (auto_vmin(data, map_type=map_type, **kwargs),
            auto_vmax(data, map_type=map_type))

def get_nlevels(max_val: u.Quantity, min_val: u.Quantity, stretch: str, 
                base: Optional[float] = 2.) -> int:
    """Determine the number of levels based on the stretch.

    Levels are assumed to be a multiple of min_val.
    
    Args: 
      max_val: maximum value.
      min_val: minimum value.
      stretch: level stretch.
      base: optional; base of the log levels.
    """
    if stretch == 'log':
        ratio = max_val.value / min_val.to(max_val.unit).value
        nlevels = int(np.floor(np.log(ratio) / np.log(base))) + 1
    elif stretch == 'linear':
        nlevels = int(max_val.value / min_val.to(max_val.unit).value)
    else:
        raise NotImplementedError

    return nlevels

def auto_levels(data: Optional[u.Quantity] = None,
                rms: Optional[u.Quantity] = None,
                max_val: Optional[u.Quantity] = None,
                min_val: Optional[u.Quantity] = None,
                nsigma: float = 5.,
                base: float = 2.,
                nlevels: Optional[int] = None,
                min_nlevels: int = 1,
                negative_nsigma: Optional[float] = None,
                min_base: float = 1.1,
                nsigmalevel: Optional = None,
                log: Callable = print) -> u.Quantity:
    """Determine the levels for contour plots.

    Args:
      data: optional; data to extract the levels from.
      rms: optional; rms of the data.
      max_val: optional; maximum contour.
      min_val: optional; minumum contour.
      nsigma: optional; minimum contour over rms value.
      negative_nsigma: optional; maximum contour for negative levels.
      base: optional; logarithmic base.
      nlevels: optional; number of levels.
      min_nlevels: optional; minimum number of levels.
      min_base: optional; minimum base for refining.
      nsigmalevel: optional; only one level at this nsigma value.
      log: optional; logging function.
    """
    # Value checks
    log('Calculating levels')
    if min_base > base:
        log('Minimum base > base, setting min_base = base')
        min_base = base
    if nlevels and nlevels < min_nlevels:
        log('Minimum nlevels > nlevels, setting nlevels = min_nlevels')
        nlevels = min_nlevels
    if data is not None and max_val is None:
        max_val = np.nanmax(data)

    # Determine rms
    log('Determining levels from data:')
    if rms is None and data is not None:
        log('Using automatic rms for levels')
        rms = apystats.mad_std(data, ignore_nan=True)
    elif ((rms is None and data is None) or
          (rms is not None and nlevels is None)):
        raise ValueError('Could not determine levels')
    rms = to_sigfig(rms)
    log(f'Getting levels from rms = {rms}')

    # One level at nsigmalevel
    if nlevels and nlevels == 1:
        log(f'Setting only one level at {nsigmalevel} sigma')
        levels = np.array([nsigma * rms.value]) * rms.unit
        log(f'Levels = {levels}')
        return levels

    # Limits
    if max_val is not None:
        base_level = nsigma * rms
        max_val = max_val.to(rms.unit)
        if nlevels is None:
            nlevels = get_nlevels(max_val, base_level, 'log', base=base)
        if min_nlevels and nlevels < min_nlevels and base == min_base:
            log('Minimum number of levels not achieved with min_base')
            log(f'Setting min_nlevels value = {nlevels}')
            min_nlevels = nlevels
        elif min_nlevels and nlevels < min_nlevels and base > min_base:
            log('Minimum number of levels not achieved, refining base')
            ratio = max_val.value / base_level.value
            new_base = to_sigfig(np.exp(np.log(ratio) / (min_nlevels-1)))
            if new_base < min_base:
                log(f'Estimated base {new_base} smaller than min_base')
                log(f'Changing base to {min_base}')
                base = min_base
            else:
                log(f'New base = {new_base}')
                base = new_base
            nlevels = get_nlevels(max_val, base_level, 'log', base=base)
        log(f'Number of levels = {nlevels}')
    elif nlevels:
        log(f'Using nlevels = {nlevels}')
    else:
        raise ValueError('Could not determine nlevels')

    # Geometric progression
    aux_levels = [nsigma * base**i for i in range(nlevels)]

    # Negative contours
    if negative_nsigma is not None:
        aux = -negative_nsigma * rms
        if min_val is None and data is not None:
            min_val = np.nanmin(data)
            log(f'Minimum data: {min_val}')
            log(f'Start negative level (-{negative_nsigma}sigma): {aux}')
            loop = True
        elif min_val is None and data is None:
            log('Could not determine minimum value')
            log('Skipping negative levels')
            loop = False
        else:
            log(f'Minimum data: {min_val}')
            log(f'Start negative level (-{negative_nsigma}sigma): {aux}')
            loop = True

        # Get negative levels
        i = -1
        while loop:
            factor = i * abs(negative_nsigma)
            newlevel = factor * rms
            if newlevel <= min_val:
                break
            elif factor not in aux_levels:
                aux_levels = [factor] + aux_levels
            i = i * base

    # Levels    
    levels = np.array(aux_levels) * rms
    log(f'Levels/rms = {aux_levels}')
    log(f'Levels = {levels}')

    return levels

def generate_label(name: str, unit: Optional[u.Unit] = None,
                   unit_fmt: str = '({})') -> str:
    """Generate a standard label from the input values.

    Args:
      name: name in the label.
      unit: optional; unit in the label.
      unit_fmt: optional; format of the unit string.
    """
    label = [f'{name.title()}']
    if unit is not None:
        label.append(unit_fmt.format(unit))
    return ' '.join(label)

def get_artist_positions(values: str, artist: str,
                         separator: str = ',') -> tuple:
    """Extract the position values for the input `artist`.

    It uses `artist` to determine the type of the output.

    Args:
      values: string where the positions are obtained.
      artist: artist type.
      separator: optional; separator between values.
    """
    vals = values.split(',')
    positions = []
    for val in vals:
        if artist in ['scatters', 'arcs']:
            try:
                ra, dec, frame = val.split()
            except ValueError:
                ra, dec = val.split()
                frame = 'icrs'
            positions.append(SkyCoord(ra, dec, frame=frame))
        else:
            positions.append(tuple(map(float, val.split())))

    return positions

def get_artist_properties(artist: str, config: 'ConfigParserAdv',
                          separator: str = ',' ) -> dict:
    """Extract the properties of the artists from `config`.

    This function creates a dictionary that contain the markers positions as
    keys and a dictionary of the artist properties as its value. The keywords
    are extracted from the option name in the `config`.

    Example:
    A configuration file with the following options:

    ```init
    scatter = pos1x pos1y, pos2x pos2y
    scatter_color = r, w
    ```

    will be converted to:

    ```python
    properties = {'positions': ((pos1x, pos1y), (pos2x, pos2y)),
                  'properties': ({'color': 'r'}, {'color': 'w'})}
    ```
    
    Each dictionary can be used as keyword input for the corresponding
    `matplotlib` function.

    Args:
      artist: name of the artist.
      config: configuration parser proxy.
      separator: optional; separator between values of an config option.
    """
    # Length
    positions = get_artist_positions(config[artist], artist,
                                     separator=separator)
    nprops = len(positions)

    # Iterate over properties
    props = ()
    for opt in config:
        # Filter
        if not opt.startswith(artist) or opt == artist:
            continue
        
        # Iterate over positions
        prop = opt.split('_')[-1]
        for i in range(nprops):
            if prop in ['size', 'width', 'height', 'angle', 's', 'alpha']:
                val = config.getvalue(opt, n=i, dtype=float, allow_global=True,
                                      sep=separator)
            elif prop in ['zorder']:
                val = config.getvalue(opt, n=i, dtype=int, allow_global=True,
                                      sep=separator)
            else:
                val = config.getvalue(opt, n=i, allow_global=True,
                                      sep=separator)

            # Special value
            if val == 'none' and prop != 'fillstyle':
                val = None

            # Fill the properties tuple
            try:
                props[i].update({prop: val})
            except IndexError:
                props += ({prop: val},)

    return {'positions': positions, 'properties': props}
