import math

import astropy.stats as apystats
import numpy as np

def quick_rms(data):
    return 1.482602219*apystats.median_absolute_deviation(data, ignore_nan=True)

def to_sigfig(x, sigfig=2):
    return round(x, -int(math.floor(math.log10(x))) + (sigfig - 1))
