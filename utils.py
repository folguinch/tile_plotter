import numpy as np

from logger import get_logger

logger = get_logger(__name__)

def auto_vminmax(data, dtype='intensity', **kwargs):
    return auto_vmin(data, dtype=dtype, **kwargs), auto_vmax(data, dtype=dtype)

def auto_vmin(data, perc=20, nrms=2., vfrac=1.02, dtype='intensity', rms=None):
    if dtype=='intensity':
        if rms is None:
            perc = np.nanpercentile(data, perc)
            rms = np.sqrt(np.nanmean(np.abs(data[data<perc])**2))
            logger.debug('rms (data<%.3e) = %.3e', perc, rms)
        else:
            logger.debug('Input rms = %.3e', rms)
        vmin = nrms*rms
    elif dtype=='velocity':
        vmin = np.nanmin(data)*vfrac
    else:
        raise KeyError('dtype %s not recognized' % dtype)
    logger.debug('vmin: %.3e', vmin)

    return vmin

def auto_vmax(data, frac=0.8, vfrac=0.98, dtype='intensity'):
    if dtype=='intensity':
        vmax = np.nanmax(data)*frac
    elif dtype=='velocity':
        vmax = np.nanmax(data)*vfrac
    else:
        raise KeyError('dtype %s not recognized' % dtype)
    logger.debug('vmax: %.3e', vmax)
    return vmax

def auto_levels(data, n=10, stretch='linear', vmin=None, vmax=None, **kwargs):
    if vmin is None or vmax is None:
        logger.debug('Using automatic vmin, vmax')
        vmin, vmax = auto_vminmax(data, **kwargs)
    logger.debug('Getting auto-levels for vmin, vmax = %f, %f', vmin, vmax)

    if stretch=='log':
        assert vmax>0
        try:
            assert vmin>0
        except AssertionError:
            logger.warn('vmin < 0, using default: 1E-6')
            vmin = 1E-6
        levels = np.logspace(np.log10(vmin), np.log10(vmax), n)
    else:
        levels = np.linspace(vmin, vmax, n)
    return levels
