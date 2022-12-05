"""
.. include:: README.md
"""
try:
    from cmcrameri import cm
    import matplotlib as mpl

    print(('Registering color maps: '
           f'{cm.broc.name} '
           f'{cm.vik.name} '
           f'{cm.imola.name} '
           f'{cm.oslo_r.name} '
           f'{cm.bamako.name} '
           f'{cm.nuuk.name} '
           f'{cm.tokyo.name} '
           f'{cm.turku.name} '
           ))
    mpl.colormaps.register(cm.broc)
    mpl.colormaps.register(cm.vik)
    mpl.colormaps.register(cm.imola)
    mpl.colormaps.register(cm.bamako)
    mpl.colormaps.register(cm.nuuk)
    mpl.colormaps.register(cm.oslo_r)
    mpl.colormaps.register(cm.tokyo)
    mpl.colormaps.register(cm.turku)
except ImportError:
    pass
