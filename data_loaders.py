"""Implement loaders for different types of data."""
from typing import Callable, Tuple

from astropy.io import fits
from astropy import wcs

# Loader functions
def load_image(filename: 'Path') -> Tuple[fits.PrimaryHDU, wcs.WCS]:
    """Load FITS file."""
    image = fits.open(filename)[0]
    proj = wcs.WCS(image.header, naxis=2)

    return image, proj

def load_pvimage(filename: 'Path') -> Tuple[fits.PrimaryHDU, str]:
    """Load FITS file."""
    image = fits.open(filename)[0]
    proj = 'rectilinear'

    return image, proj

# Available loaders
LOADERS = {
    'image': load_image,
    'contour': load_image,
    'pvmap': load_pvimage,
    'pvmap_contour': load_pvimage,
    'moment': load_image,
}

# General purpose loader
def data_loader(config: 'ConfigParserAdv',
                log: Callable = print) -> Tuple['Data', 'Projection', str]:
    """Find a loader and loads the data from a config parser proxy.

    If the `loader` option is given in the configuration, then this is used to
    determine the loader. The value of the `loader` option specifies option
    where the data file name is stored. Otherwise, it will iterate over the
    loaders available and use the one matching the options.

    Args:
      config: config parser proxy.
      log: optional; logging function.

    Returns:
      The loaded data.
      The projection of the data.
      The data type.
    """
    if 'loader' in config:
        log('Using loader option')
        key = config['loader']
        filename = config.getpath(key)
        loader = LOADERS[key]
    else:
        for key, loader in LOADERS.items():
            if key in config:
                log(f'Data option found: {key}')
                filename = config.getpath(key)
                break
        else:
            raise KeyError('Cannot find loader for data')

    log(f'Loading data: {filename}')
    data, proj = loader(filename)

    return data, proj, key
