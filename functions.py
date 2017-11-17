import os, logging
from configparser import ConfigParser
from itertools import cycle

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.units.equivalencies import doppler_radio
from astropy.visualization import LogStretch, LinearStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.coordinates import SkyCoord
from pvextractor import PathFromCenter, extract_pv_slice
from spectral_cube import SpectralCube
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Rectangle, Ellipse
from RadModelling.Objects import Source, Profile
from RadModelling.Configuration.config_main import config_main
from matplotlib.ticker import FuncFormatter 
#from hyperion.model import ModelOutput


from geometry import FigGeometry

def get_geometry(rows, cols, nxcbar=0, nycbar=0, xsize=4.5, ysize=4.5, left=1.0,
        right=.15, bottom=0.6, top=0.15, wspace=0.2, hspace=0.2, 
        cbar_width=0.2, cbar_spacing=0.1, sharex=False, sharey=False, 
        share_cbar=False):

    # Determine width and height
    geoms_ax = []
    geoms_cbar = []
    for i in range(rows):
        geom_cbar = None

        if i==rows-1 or not sharex:
            newbottom = bottom
        elif sharex:
            newbottom = 0.

        #if (not share_cbar and nycbar>=1) or (nycbar==1 and i==0):
        #    newtop = cbar_spacing
        #    if i==0:
        #        coltop = top
        #    else:
        #        coltop = top+hspace
        #    geom_cbar = FigGeometry(xsize, cbar_width, bottom=0, top=coltop)
        #elif rows>1 and sharex:
        #    newtop = hspace
        #else:
        #    newtop = top 

        for j in range(cols):
            if (not share_cbar and nycbar>=1) or (nycbar==1 and i==0):
                newtop = cbar_spacing
                if i==0:
                    coltop = top
                else:
                    coltop = top+hspace
                geom_cbar = FigGeometry(xsize, cbar_width, bottom=0, top=coltop)
            elif rows>1 and sharex:
                newtop = hspace
            else:
                newtop = top 

            if j==0 or not sharey:
                newleft = left
            elif sharey:
                newleft = 0.

            if (not share_cbar and nxcbar>=1) or (nxcbar==1 and j==cols-1):
                newright = cbar_spacing
                if j==cols-1:
                    colright = right
                else:
                    colright = right+wspace
                geoms_cbar += [FigGeometry(cbar_width, ysize, 0, colright,
                    newbottom, newtop)]
            elif cols>1 and sharey:
                newright = wspace
            else:
                newright = right

            geoms_ax += [FigGeometry(xsize, ysize, newleft, newright,
                newbottom, newtop)]
            if geom_cbar is not None:
                geom_cbar.left = newleft
                geom_cbar.right = newright
                geoms_cbar += [geom_cbar]

    return geoms_ax, geoms_cbar

def get_axes(axgeom, cbargeom, rows, cols, width, height, cbar_orientation):
    factorx = 1./width
    factory = 1./height

    def get_width(ax):
        return ax.width
    def get_height(ax):
        return ax.height
    vwidth = np.vectorize(get_width)
    vheight = np.vectorize(get_height)

    axheights = vheight(axgeom[::-1]).reshape((rows,cols))
    #axheights = np.cumsum(axheights, axis=0) - axheights
    axwidths = vwidth(axgeom).reshape((rows,cols))
    #axwidths = np.cumsum(axwidths, axis=1) - axwidths
    #if rows==cols==1:

    if len(axgeom) == len(cbargeom):
        if cbar_orientation=='vertical':
            cbheights = vheight(cbargeom[::-1]).reshape((rows,cols))
            cbheights = np.cumsum(cbheights, axis=0) - cbheights
            axheights = np.cumsum(axheights, axis=0) - axheights
            cbwidths = vwidth(cbargeom).reshape((rows,cols))
            aux = np.cumsum(axwidths, axis=1)
            cbwidths = np.cumsum(cbwidths, axis=1) - cbwidths + aux
            axwidths = aux - axwidths + (cbwidths - aux)
        elif cbar_orientation=='horizontal':
            cbheights = vheight(cbargeom[::-1]).reshape((rows,cols))
            aux = np.cumsum(axheights, axis=0)
            cbheights = np.cumsum(cbheights, axis=0) - cbheights + aux
            axheights = aux - axheights
            cbwidths = vwidth(cbargeom).reshape((rows,cols))
            cbwidths = np.cumsum(cbwidths, axis=1) - cbwidths
            axwidths = np.cumsum(axwidths, axis=1) - axwidths

        aux = zip(axheights[::-1].flat, axwidths.flat,
                cbheights[::-1].flat, cbwidths.flat)
        for i,(bottom,left,cbbottom,cbleft) in enumerate(aux): 
            axgeom[i].bottom = bottom + axgeom[i].bottom
            cbargeom[i].bottom = cbbottom + cbargeom[i].bottom
            axgeom[i].left = left + axgeom[i].left
            cbargeom[i].left = cbleft + cbargeom[i].left
            axgeom[i].scalex(factorx)
            axgeom[i].scaley(factory)
            cbargeom[i].scalex(factorx)
            cbargeom[i].scaley(factory)
    else:
        totalx = np.sum(axwidths, axis=1)
        totaly = np.sum(axheights, axis=0)
        axwidths = np.cumsum(axwidths, axis=1) - axwidths
        axheights = np.cumsum(axheights, axis=0) - axheights
        aux = zip(axheights[::-1].flat, axwidths.flat)
        for i,(bottom,left) in enumerate(aux):
            axgeom[i].bottom = bottom + axgeom[i].bottom
            axgeom[i].left = left + axgeom[i].left
            if i%cols==cols-1 and len(cbargeom)!=0 and cbar_orientation=='vertical':
                j = i/cols
                cbargeom[j].left =  totalx[j] + cbargeom[j].left
                cbargeom[j].bottom = axgeom[i].bottom #+ cbargeom[j].bottom
                cbargeom[j].scalex(factorx)
                cbargeom[j].scaley(factory)
            elif i<cols and len(cbargeom)!=0 and cbar_orientation=='horizontal':
                j = i%cols
                cbargeom[i].left = axgeom[i].left #+ cbargeom[j].left
                cbargeom[i].bottom = totaly[i] + cbargeom[i].bottom
                cbargeom[i].scalex(factorx)
                cbargeom[i].scaley(factory)
            axgeom[i].scalex(factorx)
            axgeom[i].scaley(factory)

    return axgeom, cbargeom

def offset_formatter():
    def myformatter(x, pos):
        return '%.2e' % (x*3600.)
    return FuncFormatter(myformatter)

def get_source(name):
    conf = config_main(name)
    source = Source(name, config=conf)
    return source

# THIS SHOULD NOT BE HERE
#def get_hyperion_model(fname):
#    return ModelOutput(fname)

def setup_env(script, name, conf_file='config.cfg'):
    THIS = os.path.dirname(os.path.realpath(script))
    EUROPA = '/mnt/Europa/pyfao'
    FIGS = os.path.realpath(os.path.join(THIS,'../Figs'))
    #figname = script.replace('.py', '.png')
    try:
        os.makedirs(FIGS)
    except:
        pass
    config = ConfigParser()
    config.read(conf_file)
    logger = None

    return EUROPA, FIGS, logger, config

def get_data(name, position=None):
    img = fits.open(name)[0]
    #wcs = WCS(img.header)
    if len(img.data.shape)==4:
        img.data = img.data[0,0,:,:]
        if position:
            img = align_images(img, *position)
        wcs = WCS(img.header).sub(['longitude','latitude'])
    else:
        if position:
            img = align_images(img, *position)
        wcs = WCS(img.header)

    return img, wcs

def plot_beam(ax, header, dx=2, dy=2, pad=2):
    # Beam properties
    wcs = WCS(header).sub(['celestial'])
    pixsize = np.mean(np.abs(wcs.wcs.cdelt))
    bmaj = header['BMAJ']/pixsize
    bmin = header['BMIN']/pixsize
    bpa = header['BPA']

    # Define position
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmin += dx
    ymin += dy
    size = bmaj+pad

    # Plot box and beam
    #rect = Rectangle((xmin,ymin), size, size, fill=True, fc='w', zorder=3)
    beam = Ellipse((xmin+size/2., ymin+size/2.), bmin, bmaj, angle=bpa,
            zorder=4, fc='k')
    #ax.add_patch(rect)
    ax.add_patch(beam)

def get_ticks(vmin, vmax, a=1000, n=5, stretch='log'):
    if stretch=='log':
        x = lambda y: (10.**(y*np.log10(a+1.))-1.)/a
        y = np.linspace(0.,1.,n)
        x = x(y)
        return x*(vmax-vmin) + vmin
    else:
        return np.linspace(vmin, vmax, n)

def recenter(ax, wcs, ra, dec, r):
    x, y = wcs.all_world2pix([[ra,dec]], 0)[0]
    cdelt = np.mean(np.abs(wcs.wcs.cdelt))
    radius = r/cdelt
    ax.set_xlim(x-radius, x+radius)
    ax.set_ylim(y-radius, y+radius)

def config_map(ax, xformat="hh:mm:ss.s", yformat="dd:mm:ss", xlabel=True,
        ylabel=True, xticks=True, yticks=True, xpad=1., ypad=1., tickscolor='w'):
    ra, dec = ax.coords['ra'], ax.coords['dec']

    # Axes labels
    ra.set_axislabel('RA (J2000)' if xlabel else '', 
            size=ax.xaxis.get_label().get_fontsize(),
            family=ax.xaxis.get_label().get_family(),
            fontname=ax.xaxis.get_label().get_fontname(), minpad=xpad)
    dec.set_axislabel('Dec (J2000)' if ylabel else '',
            size=ax.xaxis.get_label().get_fontsize(),
            family=ax.xaxis.get_label().get_family(),
            fontname=ax.xaxis.get_label().get_fontname(), minpad=ypad)

    # Ticks labels
    ra.set_major_formatter(xformat)
    ra.set_ticklabel_visible(xticks)
    dec.set_major_formatter(yformat)
    dec.set_ticklabel_visible(yticks)
    ra.set_ticks(color=tickscolor, exclude_overlapping=True)
    dec.set_ticks(color=tickscolor)

    # Ticks
    ra.set_ticklabel(size=ax.xaxis.get_majorticklabels()[0].get_fontsize(),
            family=ax.xaxis.get_majorticklabels()[0].get_family(),
            fontname=ax.xaxis.get_majorticklabels()[0].get_fontname())
    dec.set_ticklabel(size=ax.yaxis.get_majorticklabels()[0].get_fontsize(),
            family=ax.yaxis.get_majorticklabels()[0].get_family(),
            fontname=ax.yaxis.get_majorticklabels()[0].get_fontname())
    ra.display_minor_ticks(True)
    dec.display_minor_ticks(True)

def config_pvdiag(ax, xlabel=True, ylabel=True, xticks=True, yticks=True):
    ax.set_aspect('auto')
    if xlabel:
        ax.set_xlabel('Offset [arcsec]')
    if ylabel:
        ax.set_ylabel('Velocity [km s$^{-1}$]')

def plot_map(data, fig, config, wcs='rectilinear', subplot=111, contours=None,
        contours_wcs=None, levels=None, colors='g', stretch='linear', vmin=0,
        vmax=None, a=1000, shift=True, extent=None, extent_cont=None):
    # Add subplot
    try:
        ax = fig.add_subplot(subplot, projection=wcs)
    except TypeError:
        ax = fig.add_axes(subplot, projection=wcs)

    # Check wcs and re-centre the image
    if hasattr(wcs, 'wcs') and shift and 'r' in config:
        recenter(ax, wcs, config.getfloat('ra'), config.getfloat('dec'), 
                config.getfloat('r'))

    # Normalisation
    vmin = vmin or config.getfloat('vmin',np.nanmin(data))
    vmax = vmax or config.getfloat('vmax',np.nanmax(data))
    if stretch=='log':
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LogStretch(a=a))
    else:
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LinearStretch())

    # Plot data
    im = ax.imshow(data, norm=norm, zorder=1, extent=extent)

    # Plot contours
    if contours is not None and levels is not None:
        if contours_wcs is not None:
            ax.contour(contours, transform=ax.get_transform(contours_wcs),
                    levels=levels, colors=colors, zorder=2)
        else:
            ax.contour(contours, levels=levels, colors=colors, zorder=2, 
                    extent=extent_cont)

    return ax, im

def plot_cbar(fig, im, ax, config=None, subplot=None, **kwargs):
    # Update kwargs
    if config and 'ticks' not in kwargs:
        kwargs.setdefault('vmin', config.getfloat('vmin'))
        kwargs.setdefault('vmax', config.getfloat('vmax'))
        kwargs.setdefault('a', config.getfloat('a',1000))

    # Ticks
    if kwargs.get('ticks'):
        ticks = kwargs['ticks']
    elif 'vmin' in kwargs and 'vmax' in kwargs:
        ticks = get_ticks(kwargs['vmin'], kwargs['vmax'], 
                kwargs.get('a',1000), stretch=kwargs.get('stretch','log'))
    else:
        ticks = None

    # Create bar
    if subplot is not None:
        try:
            cbax = fig.add_subplot(subplot)
        except TypeError:
            cbax = fig.add_axes(subplot)
        cbar = fig.colorbar(im, ax=ax, cax=cbax,
                orientation=kwargs.get('orientation','vertical'),
                drawedges=False, ticks=ticks)
        # Tick Font properties
        #print cbar.ax.get_yticklabels()[-1].get_fontsize()
        #for label in cbax.get_xticklabels()+cbax.get_yticklabels():
        #    label.set_fontsize(ax.xaxis.get_majorticklabels()[0].get_fontsize())
        #    label.set_family(ax.xaxis.get_majorticklabels()[0].get_family())
        #    label.set_fontname(ax.xaxis.get_majorticklabels()[0].get_fontname())
    else:
        cbar = fig.colorbar(im, ax=ax, 
                orientation=kwargs.get('orientation','vertical'),
                drawedges=False, ticks=ticks)

    # Bar position
    if kwargs.get('position'):
        cbar.ax.xaxis.set_ticks_position(kwargs['position'])

    # Label
    if 'label' in kwargs:
        cbar.set_label(kwargs['label'],
                fontsize=ax.xaxis.get_label().get_fontsize(),
                family=ax.xaxis.get_label().get_family(),
                fontname=ax.xaxis.get_label().get_fontname(),
                weight=ax.xaxis.get_label().get_weight())
        if kwargs.get('label_position'):
            cbar.ax.xaxis.set_label_position(kwargs['label_position'])
    for label in cbar.ax.xaxis.get_ticklabels(which='both')+cbar.ax.yaxis.get_ticklabels(which='both'):
        label.set_fontsize(ax.xaxis.get_majorticklabels()[0].get_fontsize())
        label.set_family(ax.xaxis.get_majorticklabels()[0].get_family())
        label.set_fontname(ax.xaxis.get_majorticklabels()[0].get_fontname())

    return cbar

#def plot_pv(data, fig, config, )

def get_fig(styles, rows=1, cols=1, ncbar=0, xsize=4.5, ysize=4.5, left=1.0,
        right=.95, bottom=0.55, upper=0.1, wspace=0.2, hspace=0., 
        cbar_width=0.2, cbar_spacing=0.1, share_y=True, share_x=True, share_cbar=False):
    try:
        plt.close()
    except:
        pass
    plt.style.use(styles)
    
    # Determine the number of axes
    nx = cols + ncbar
    ny = rows

    # Determine width and height
    if ncbar==1 and rows>1 or cols>1:
        share_cbar = True
        wcbar = 0
    elif ncbar==0:
        wcbar = 0
    else:
        wcbar = cbar_width +  cbar_spacing + right
    if ncbar>1 and not share_cbar:
        newright = right * ncbar
    else:
        newright = right
    if rows>1 and not share_x:
        newbottom = bottom * rows
        newupper = upper * rows
    else:
        newbottom = bottom
        newupper = upper
    width = left + xsize*cols + (cbar_width + cbar_spacing)*ncbar +\
            wspace*(cols-1) + newright
    height = newbottom + newupper + ysize*rows
    print 'Figure size: width=%.1f in height=%.1f' % (width,height)

   
    # Determine axes position left-right then up-down
    # Axes positions
    axes = []
    cbar_axes = []
    for i in range(rows):
        if i==0:
            ypos = bottom
        else:
            ypos = ypos + ysize + wspace + bottom*(not share_x) + wcbar

        for j in range(cols):
            if j==0:
                xpos = left
            else:
                xpos = xpos + xsize + wspace + left*(not share_y) + wcbar
            if ncbar>0 and ((share_cbar and j==cols-1) or not share_cbar):
                cxpos = xpos + xsize + cbar_spacing
                cbar_axes += [[cxpos/width, ypos/height, cbar_width/width,
                    ysize/height]]
            axes += [[xpos/width, ypos/height, xsize/width, ysize/height]]
    

    # Create figure
    fig = plt.figure(figsize=(width, height), dpi=600)

    return fig, axes, cbar_axes

def plot_moments(momt, config, figname, cont=None, mn=0, freqline=None, 
        styles=['paper', '3by2fig'], mask=None):
    plt.style.use(styles)
    fig = plt.figure()
    vmin = config.getfloat('m%i_vmin' % mn)
    vmax = config.getfloat('m%i_vmax' % mn)
    a = config.getfloat('a')

    if cont and mn==0:
        continuum = fits.open(cont)[0]
        continuum_wcs = WCS(continuum.header)
    
    gs = gridspec.GridSpec(5,2, height_ratios=[0.05, 0.02, 1, 1, 1])
    gs.update(wspace=0., hspace=0.)

    for i,m in enumerate(momt):
        m_map = fits.open(m)[0]
        m_wcs = WCS(m_map.header).sub(['longitude', 'latitude'])
        if cont is not None and mn==0:
            levels = np.logspace(config.getfloat('m0_lmin'),
                    config.getfloat('m0_lmax'), 8)
            ax, im = plot_map(continuum.data, fig, config, wcs=continuum_wcs,
                    subplot=gs[i/2+2, i%2],
                    contours=m_map.data[0,0,:,:], contours_wcs=m_wcs,
                    levels=levels, stretch='log', vmin=vmin, vmax=vmax, a=a) 
            config_map(ax, xlabel=i==4, ylabel=i==4, xticks=i in [4,5],
                    yticks=i%2==0)
            plot_beam(ax, m_map.header)
            ax.annotate('K=%i' % i, xy=(0.1,0.9), xytext=(0.1,0.9),
                    xycoords='axes fraction', fontsize=10, zorder=5,
                    backgroundcolor='w')
            if i==0:
                cbar = plot_cbar(fig, im, ax, subplot=gs[0,0],
                        orientation='horizontal', position='top',
                        label='Intensity [Jy beam$^{-1}$]', 
                        label_position='top', vmin=vmin, vmax=vmax, a=a)
        elif mn==1 and freqline is not None:
            # Shift velocity
            freq = freqline[i] * u.GHz
            rest = m_map.header['RESTFRQ'] * u.Hz
            line_vel = freq.to(u.km/u.s, equivalencies=doppler_radio(rest))
            print line_vel
            m_map.data = m_map.data[0,0,:,:] - line_vel.value -\
                    m_map.header['VELO-LSR']*1E-3

            # Plot
            ax, im = plot_map(m_map.data, fig, config, wcs=m_wcs,
                    subplot=gs[i/2+2, i%2], vmin=vmin, vmax=vmax) 
            config_map(ax, xlabel=i==4, ylabel=i==4, xticks=i in [4,5],
                    yticks=i%2==0)
            if i==0:
                cbar = plot_cbar(fig, im, ax, subplot=gs[0,0],
                        orientation='horizontal', position='top',
                        label='Velocity [km s$^{-1}$]', 
                        label_position='top')

    fig.savefig(figname)
    plt.close()

def align_casa_imgs(img, ra, dec):
    im = fits.open(img)[0]

    ymax, xmax = np.unravel_index(np.nanargmax(im.data[:,:]),
            im.data.shape[2:])
    im.header['CRPIX1'] = xmax+1.
    im.header['CRPIX2'] = ymax+1.
    im.header['CRVAL1'] = ra
    im.header['CRVAL2'] = dec
    return im

def align_images(img, ra, dec):
    ymax, xmax = np.unravel_index(np.nanargmax(img.data),
            img.data.shape)
    img.header['CRPIX1'] = xmax+1.
    img.header['CRPIX2'] = ymax+1.
    img.header['CRVAL1'] = ra
    img.header['CRVAL2'] = dec
    return img

# For data cubes
def get_cube(fname):
    try:
        return SpectralCube.read(fname)
    except:
        print 'FIXING CUBE'
        cube = fits.open(fname)[0]
        cube.header['CUNIT3'] = 'm/s'
        cube.header['CRVAL3'] = cube.header['CRVAL3']*1E3
        cube.header['CDELT3'] = cube.header['CDELT3']*1E3
        return  SpectralCube(cube.data, WCS(cube.header))

def get_hdu(val, wcs):
    hdu =  fits.PrimaryHDU(val)
    hdu.header = wcs.sub(['longitude','latitude']).to_header()
    hdu.header.set('SIMPLE','T', before=0)
    return hdu

def shift_vel(mmap, freqline):
    # Shift velocity
    freq = freqline * u.GHz
    rest = mmap.header['RESTFRQ'] * u.Hz
    line_vel = freq.to(u.km/u.s, equivalencies=doppler_radio(rest))
    mmap.data = mmap.data - line_vel.value -\
            mmap.header['VELO-LSR']*1E-3
    return mmap

def rms(x):
    return np.sqrt(np.nansum(x*x)/x.size)

def freq_to_vel(freq, frest):
    return freq.to(u.km/u.s, equivalencies=doppler_radio(frest))

def vel_to_freq(vel, frest):
    return vel.to(u.Hz, equivalencies=doppler_radio(frest))

def get_moment(fname, moment, position=None, fref=None, sigbox=None, nsigma=5):
    cube = get_cube(fname)
    freq = cube.spectral_axis
    rest = cube.header['RESTFRQ'] * u.Hz
    if freq.unit in [u.Hz, u.GHz]:
        vel = freq_to_vel(freq, rest)
    else:
        print 'Check: spectral unit = %r' % freq.unit
        vel = cube.spectral_axis
        freq = vel_to_freq(vel, rest)
    mask = np.ones(cube.shape, dtype=bool)
    if sigbox:
        for i,sl in enumerate(cube._data):
            sigma = rms(sl[sigbox[2]:sigbox[3],sigbox[0]:sigbox[1]])
            mask[i] = sl >= nsigma*sigma

    if moment==0:
        dv = vel.value[1]-vel.value[0]
        mom = get_hdu(dv*np.nansum(cube._data, axis=0), cube.wcs)
    elif moment==1:
        maimg = np.ma.array(cube._data, mask=~mask)
        if fref is not None:
            vel = vel - freq_to_vel(fref, rest) 
        den = np.nansum(maimg, axis=0)
        mom = get_hdu(np.nansum((maimg.T*vel.value).T, axis=0)/den, cube.wcs)
    #if fref:
    #    cube = cube.with_spectral_unit(u.km/u.s, velocity_convention='radio',
    #            rest_value=fref) 
    #mom = cube.moment(order=moment).hdu
    if position:
        mom = align_images(mom, *position)

    return mom, WCS(mom.header)

def plot_moment1(data, config, freq, fig, subplot=111, subplot_cbar=None,
        cblabels=None, cborientation='vertical', sigbox=None, ylabel=True,
        yticks=True, position=None):
    # Data
    mod, mod_wcs = get_moment(data, 1, fref=freq*u.GHz,
            sigbox=sigbox, position=position)
    try:
        ax = fig.add_subplot(subplot, projection=mod_wcs)
    except TypeError:
        ax = fig.add_axes(subplot, projection=mod_wcs)

    # Moment 1
    vmin = config.getfloat('m1_vmin')
    vmax = config.getfloat('m1_vmax')
    ax, im = plot_map(mod.data, fig, config, wcs=mod_wcs,
            subplot=subplot, vmin=vmin, vmax=vmax, stretch='linear')
    config_map(ax, ylabel=ylabel, yticks=yticks, tickscolor='k')
    cbar = plot_cbar(fig, im, ax, subplot=subplot_cbar,
            label='Velocity (km/s)', label_position=cblabels, stretch='linear',
            orientation=cborientation, position=cblabels, vmin=vmin, vmax=vmax) 
    return ax, im

def get_pv(img, x, y, wcs, pa, width, length=10*u.arcsec):
    ra, dec = wcs.all_pix2world([[x, y]], 0)[0]
    pos = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='fk5')

    path = PathFromCenter(center=pos, length=length, angle=pa,
            width=width)

    return extract_pv_slice(img, path, respect_nan=False)

# Others
def formatter(kind, sci=(-3,4)):
	if kind=='log':
		def logformatter(x, pos):
			if x <= 10**sci[0]:
				return '10$^{%i}$' % np.floor(np.log10(x))
			elif x<1:
				return '%g' % x
			elif x<10**sci[1]:
				return '%i' % x
			else:
				return '$10^{%i}$' % np.floor(np.log10(x))
		return FuncFormatter(logformatter)

def plot_seds(seds, ax, legend=None, filename=None, config=None, **kwargs):
    lntype = np.array(["-","--","-.",":"])

    # Configure ax
    ax.set_xlabel(kwargs.get('xlabel', "Wavelength (micron)"))
    ax.set_ylabel(kwargs.get('ylabel', "Flux (Jy)"))
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(formatter('log'))
    ax.yaxis.set_major_formatter(formatter('log'))
    if config:
        xlim = config.get('xlim', fallback='1, 2000')
        ylim = config.get('ylim', fallback='1E-2, 1E4')
        xlim = map(float, xlim.split(','))
        ylim = map(float, ylim.split(','))
    else:
        xlim = [1, 2000]
        ylim = [1E-2, 1E4]

    linecycler = cycle(lntype)
    for i, sed in enumerate(seds):
        if legend:
            label = legend[i]
        else:
            label = None
        if len(sed)==3:
            ylower = np.maximum(1e-10, sed[1] - sed[2])
            yerr_lower = sed[1] - ylower
            ax.errorbar(sed[0], sed[1], yerr=[yerr_lower, sed[2]], fmt='s', 
                    elinewidth=.7, ms=4, mew=.5, label=label)
        elif len(sed)==2:
            ax.plot(sed[0], sed[1], next(linecycler), label=label)

    ax.set_xlim(*tuple(xlim))
    ax.set_ylim(*tuple(ylim))

    if legend: 
        ax.legend(loc=8)

    return ax

def plot_sed(model, source, inc, fig, config=None, subplot=111, comps=None, 
        fname_base=None, logger=None):
    # SEDs
    distance = (source['d']*1E3*u.pc).to(u.cm)
    distance = distance.value
    seds = [(source.sed.wlg, source.sed.F, source.sed.Ferr)]
    wav, fnu = model.get_sed(aperture=-1, distance=distance, 
                             units='Jy', inclination=inc)
    seds += [(wav, fnu)]
    legend = ['Observed', 'Total']
    if comps:
        # Check components
        comps_new = []
        try:
            for i, label in enumerate(comps):
                wav, fnu = model.get_sed(aperture=-1, distance=distance, 
                                         units='Jy', inclination=inc, 
                                         dust_id=i, component='dust_scat')
                comps_new += [label]
        except ValueError:
            comps_new = comps[1:]
            if logger:
                logger.warn('Component missing, assuming disc')
                logger.debug('New components: %r', comps_new)

        for i, label in enumerate(comps_new):
            comps2 = ['source_emit', 'source_scat', 'dust_emit',
                      'dust_scat']
            seds2 = []
            for comp in comps2:
                wav, fnu = model.get_sed(aperture=-1, distance=distance,
                                         units='Jy', inclination=inc,
                                         dust_id=i, component=comp)
                seds2 += [(wav, fnu)]
            
            # Attach to final plot
            seds += [(wav, np.sum(seds2, axis=0)[1])] 
            legend += [label]

    try:
        ax = fig.add_subplot(subplot)
    except:
        ax = fig.add_axes(subplot)

    return plot_seds(seds, ax, legend=legend, config=config)

def plot_profile(prof, ax, ystretch='linear', label=None, pttype='s',
        lntype='k-'):
    if prof.ferr is not None:
        if ystretch=='log':
            ylower = np.maximum(1e-10, prof.f - prof.ferr)
            yerr_lower = prof.f - ylower
        else:
            yerr_lower = prof.ferr
        ax.errorbar(prof.b, prof.f, yerr=[yerr_lower, prof.ferr], fmt=pttype, 
                    elinewidth=.8, ms=4, mew=.5, label=label)
    else:
        ax.plot(prof.b, prof.f, lntype, lw=2, label=label)
    return ax

def get_profile(fname):
    if not os.path.isfile(fname):
        raise Exception
    return Profile(filename=fname)

if __name__=='__main__':
    #print setup_env(__file__)
    fig, gs = get_fig(['poster'])
