from .base_plotter import BasePlotter, SinglePlotter

class Plotter(SinglePlotter):
    pass

class NPlotter(BasePlotter):

    def __init__(self, styles=[], rows=1, cols=1, xsize=5, ysize=3, left=.7, 
            right=0.15, bottom=0.6, top=0.15, wspace=0.2, hspace=0.2, 
            sharex=False, sharey=False):

        super(NPlotter, self).__init__(styles=styles, rows=rows, cols=cols, 
                xsize=xsize, ysize=ysize, left=left, right=right, bottom=bottom,
                top=top, wspace=wspace, hspace=hspace, sharex=sharex,
                sharey=sharey)

    def get_axis(self, nax=0):
        axis, cbaxis = super(NPlotter,self).get_axis(nax, include_cbar=False)
        return Plotter(axis)

    def init_axis(self, n):
        super(NPlotter, self).init_axis(n)
