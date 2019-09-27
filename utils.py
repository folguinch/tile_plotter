import numpy as np

def auto_vminmax(data, dtype='intensity'):
    return auto_vmin(data, dtype=dtype), auto_vmax(data, dtype=dtype)

def auto_vmin(data, perc=20, nrms=2., vfrac=1.02, dtype='intensity'):
    if dtype=='intensity':
        perc = np.nanpercentile(data, perc)
        rms = np.sqrt(np.nanmean(np.abs(data[data<perc])**2))
        vmin = nrms*rms
    elif dtype=='velocity':
        vmin = np.nanmin(data)*vfrac
    else:
        raise KeyError('dtype %s not recognized' % dtype)

    return vmin

def auto_vmax(data, frac=0.8, vfrac=0.98, dtype='intensity'):
    if dtype=='intensity':
        return np.nanmax(data)*frac
    elif dtype=='velocity':
        return np.nanmax(data)*vfrac
    else:
        raise KeyError('dtype %s not recognized' % dtype)

def auto_levels(data, n=10, stretch='linear', perc=20, vmin=None, vmax=None):
    if vmin is None or vmax is None:
        vmin, vmax = auto_vminmax(data)

    if stretch=='log':
        assert vmax>0
        try:
            assert vmin>0
        except AssertionError:
            print('vmin < 0, using default: 1E-6')
            vmin = 1E-6
        levels = np.logspace(np.log10(vmin), np.log10(vmax), n)
    else:
        levels = np.linspace(vmin, vmax, n)
    return levels
