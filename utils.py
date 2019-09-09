import numpy as np

def auto_vminmax(data):
    return auto_vmin(data), auto_vmax(data)

def auto_vmin(data, perc=20, nrms=2.):
    perc = np.nanpercentile(data, perc)
    rms = np.sqrt(np.nanmean(np.abs(data[data<perc])**2))
    vmin = nrms*rms

    return vmin

def auto_vmax(data, frac=0.8):
    return np.nanmax(data)*frac

def auto_levels(data, n=10, stretch='linear', perc=20):
    if stretch=='log':
        vmin, vmax = auto_vminmax(data)
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
