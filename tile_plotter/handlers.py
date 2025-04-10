"""Manage the available plot handlers."""
from .map_plotter import MapHandler
from .plot_handler import PhysPlotHandler
from .common_types import PlotHandler

HANDLERS = {
    'image': MapHandler,
    'contour': MapHandler,
    'moment': MapHandler,
    'pvmap': MapHandler,
    'pvmap_contour': MapHandler,
    'composite': MapHandler,
    'spectrum_cassis': None,
    'spectra_cassis_model': None,
    'structured_array': PhysPlotHandler,
    'function': PhysPlotHandler,
    'region_patch': MapHandler,
}

def get_handler(config: 'configparser.ConfigParser') -> PlotHandler:
    """Find a plotting handler based on the `config` proxy options.

    If a `handler` option is given in the configuration, then this is used to
    obtain the respective handler. Otherwise, it will iterate over the stored
    handlers and find a key matching the options.

    Args:
      config: a config parser proxy.
    """
    # Check for specific option
    if 'handler' in config:
        return HANDLERS[config['handler']]

    # Iterate over keys
    for key, val in HANDLERS.items():
        if key in config:
            return val

    raise KeyError('Could not find a suitable plot handler')
