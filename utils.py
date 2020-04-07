import numpy as np

import logger 
from maths import quick_rms, to_sigfig

LOG = logger.get_logger(__name__)

def auto_vminmax(data, dtype='intensity', **kwargs):
    return auto_vmin(data, dtype=dtype, **kwargs), auto_vmax(data, dtype=dtype)

def auto_vmin(data, rms=None, nrms=1., vfrac=1.02, dtype='intensity'):
    if dtype in ['intensity', 'pvmap']:
        if rms is None:
            rms = quick_rms(data)
            LOG.debug('Quick rms = %.3e', rms)
        else:
            LOG.debug('Input rms = %.3e', rms)
        vmin = nrms*rms
    elif dtype=='velocity':
        vmin = np.nanmin(data)*vfrac
    else:
        raise KeyError('dtype %s not recognized' % dtype)
    LOG.debug('vmin: %.3e', vmin)

    return vmin

def auto_vmax(data, frac=0.8, vfrac=0.98, dtype='intensity'):
    if dtype in ['intensity', 'pvmap']:
        vmax = np.nanmax(data)*frac
    elif dtype=='velocity':
        vmax = np.nanmax(data)*vfrac
    else:
        raise KeyError('dtype %s not recognized' % dtype)
    LOG.debug('vmax: %.3e', vmax)
    return vmax

def auto_levels(data=None, rms=None, nsigma=5., base=2, nlevels=None,
        minnlevels=5, nsigmalevel=None):
    # Determine rms
    LOG.info('Determining levels from data:')
    if rms is None and data is not None:
        LOG.info('Using automatic rms for levels')
        rms = quick_rms(data)
    elif (rms is None and data is None) and \
            (rms is not None and nlevels is None):
        raise Exception('Could not determine rms level')
    rms = to_sigfig(rms)
    LOG.info('Getting levels for rms = %f', rms)

    # One level at nsigmalevel
    if nsigmalevel is not None:
        LOG.info('Setting only one level at %i sigma', nsigmalevel)
        levels = [nsigmalevel*rms]
        LOG.info('Levels = %r', levels)
        return levels

    # Limits
    if data is not None:
        baselevel = rms*nsigma
        maxval = to_sigfig(np.nanmax(data))
        nlevels = int(np.floor(np.log(maxval/baselevel)/np.log(base)))+1
        if nlevels<minnlevels:
            LOG.warn('Minimum number of levels not achieved, refining base ...')
            base = to_sigfig(np.exp(np.log(maxval/baselevel)/(minnlevels-1)))
            if base==1.0:
                LOG.warn('New base = %f', base)
                LOG.warn('Changing base to 1.01')
                base = 1.01
            else:
                LOG.info('New base = %f', base)
            nlevels = int(np.floor(np.log(maxval/baselevel)/np.log(base)))+1
        LOG.info('Number of levels = %i', nlevels)
    else:
        LOG.info('Using nlevels = %i', nlevels)

    # Geometric progression
    aux = []
    i = 1
    while len(aux)<nlevels:
        aux += [to_sigfig(i*nsigma)]
        i = i * base
        #if data is not None and i*nsigma>maxnlevel:
        #    break
        #elif nlevels is not None and len(aux)==nlevels:
        #    break

    # If there are not enough levels
    if len(aux)==1:
        LOG.warn('Not enough levels, creating new levels')
        aux = list(np.linspace(aux[0]*rms, 0.9*np.nanmax(data), 10)/rms)
    LOG.info('Levels/rms = %r', aux)

    # Levels    
    levels = np.array(aux) * rms
    LOG.info('Levels = %r', levels)

    return levels
