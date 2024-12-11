"""Implement loaders for different types of data."""
from typing import Callable, Tuple, TypeVar, Sequence, Dict, List

from astropy.io import fits
from astropy import wcs
from line_little_helper.spectrum import Spectrum, CassisModelSpectra
from toolkit.array_utils import load_struct_array
import astropy.units as u
import numpy as np
import numpy.typing as npt

Data = TypeVar('Data')
Projection = TypeVar('Projection')

# Loader functions
def load_image(filename: 'pathlib.Path') -> Tuple[fits.PrimaryHDU, wcs.WCS]:
    """Load FITS file."""
    image = fits.open(filename)[0]
    proj = wcs.WCS(image.header, naxis=2)

    return image, proj

def load_composite(args: Sequence[Tuple[str, 'pathlib.Path']]):
    """Load FITS for composite images."""
    data = []
    proj = None
    for cmap, filename in args:
        # Open data
        img, aux = load_image(filename)
        if proj is None:
            proj = aux

        # Normalize data
        img.data = np.squeeze(img.data) - np.nanmin(img.data)
        img.data = img.data / np.nanmax(img.data)

        # Store
        data.append((cmap, img))

    return dict(data), proj

def load_pvimage(filename: 'pathlib.Path') -> Tuple[fits.PrimaryHDU, str]:
    """Load FITS file."""
    image = fits.open(filename)[0]
    proj = 'rectilinear'

    return image, proj

def load_spectrum_cassis(filename: 'pathlib.Path') -> Tuple[Spectrum, str]:
    """Load spectrum in CASSIS format."""
    spectrum = Spectrum.from_cassis(filename)
    proj = 'rectilinear'

    return spectrum, proj

def load_spectra_cassis_model(
    filename: 'pathlib.Path') -> Tuple[CassisModelSpectra, str]:
    """Load spectra from a CASSIS model."""
    spectrum = CassisModelSpectra.read(filename)
    proj = 'rectilinear'

    return spectrum, proj

def load_structured_array(
    filename: 'pathlib.Path') -> Tuple[Tuple[npt.ArrayLike, Dict], str]:
    """Load a structured array."""
    data = load_struct_array(filename)
    proj = 'rectilinear'

    return data, proj

def eval_function(function: str,
                  coeficients: List[float],
                  xval: u.Quantity,
                  rotation: u.Quantity):
    """Evaluate a function.

    Functions are:
    
    - `poly`: polynomial specified as in `np.poly1d`.
    """
    # Convert to functional form
    if function == 'poly':
        funct = np.poly1d(coeficients)
    else:
        raise NotImplementedError(f'Function {function} not implemented')

    # Evaluate and rotate
    yval = funct(xval.value) * xval.unit
    xrot = xval * np.cos(rotation) - yval * np.sin(rotation)
    yrot = xval * np.sin(rotation) + yval * np.cos(rotation)

    return (xrot, yrot), 'rectilinear'

# Available loaders
LOADERS = {
    'image': load_image,
    'contour': load_image,
    'pvmap': load_pvimage,
    'pvmap_contour': load_pvimage,
    'moment': load_image,
    'composite': load_composite,
    'spectrum_cassis': load_spectrum_cassis,
    'spectra_cassis_model': load_spectra_cassis_model,
    'structured_array': load_structured_array,
    'function': eval_function,
}

# General purpose loader
def data_loader(config: 'configparseradv.configparser.ConfigParserAdv',
                log: Callable = print) -> Tuple[Data, Projection, str]:
    """Find a loader and loads the data from a config parser proxy.

    If the `loader` option is given in the configuration, then this is used to
    determine the loader. The value of the `loader` option specifies option
    where the data file name is stored. Otherwise, it will iterate over the
    loaders available and use the one matching the options.

    Args:
      config: Config parser proxy.
      log: Optional. Logging function.

    Returns:
      The loaded data.
      The projection of the data.
      The data type.
    """
    if 'loader' in config:
        log('Using loader option')
        key = config['loader']
        if key == 'composite':
            loader_args = get_composite_args(config)
        elif key == 'function':
            loader_args = get_function_args(config)
        else:
            loader_args = config.getpath(key)
        loader = LOADERS[key]
    else:
        for key, loader in LOADERS.items():
            if key in config:
                log(f'Data option found: {key}')
                if key == 'composite':
                    loader_args = (get_composite_args(config),)
                elif key == 'function':
                    loader_args = get_function_args(config)
                else:
                    loader_args = (config.getpath(key),)
                break
        else:
            raise KeyError('Cannot find loader for data')

    log(f'Loading data: {loader_args}')
    data, proj = loader(*loader_args)

    return data, proj, key

def get_composite_args(config: 'configparseradv.configparser.ConfigParserAdv'):
    """Obtain inputs for composite maps."""
    # Get cmaps
    cmap_names = config['composite'].split()

    # Get filenames
    args = []
    for cmap in cmap_names:
        args.append((cmap, config.getpath(cmap)))

    return args

def get_function_args(config: 'configparseradv.configparser.ConfigParserAdv'):
    """Get axis poisitions for function."""
    # Common parameters
    function = config['function']
    x_low, x_high = config.getquantity('xrange')
    stretch = config.get('stretch', fallback='linear')
    rotation = config.getquantity('rotate', fallback=0)
    sampling = config.getint('sampling', fallback=100)
    coef = config.getfloatlist('coeficients')
    
    # Calculate the x-axis values
    if stretch == 'linear':
        xval = np.linspace(x_low, x_high, sampling)
    elif stretch == 'log':
        xval = np.logspace(np.log10(x_low), np.log10(x_high), sampling)
    else:
        raise NotImplementedError(f'Stretch {stretch} not implemented')

    return function, coef, xval, rotation

    
