"""
.. include:: README.md
"""
try:
    from cmcrameri import cm
    import matplotlib as mpl

    print(('Registering color maps: '
           f'{cm.broc.name} {cm.vik.name}'))
    mpl.colormaps.register(cm.broc)
    mpl.colormaps.register(cm.vik)
except ImportError:
    pass
