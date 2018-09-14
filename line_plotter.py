from .base_plotter import BasePlotter, SinglePlotter

class LinePlotter(SinglePlotter):
    pass

class LinesPlotter(BasePlotter):

    def __init__(self, styles=[], rows=1, cols=1, xsize=3, ysize=5, left=.7, 
            right=0.15, bottom=0.6, top=0.15, wspace=0.2, hspace=0.2, 
            sharex=False, sharey=False):

        super(LinesPlotter, self).__init__(styles=styles, rows=rows, cols=cols, 
                xsize=xsize, ysize=ysize, left=left, right=right, bottom=bottom,
                top=top, wspace=wspace, hspace=hspace, sharex=sharex,
                sharey=sharey)

    def get_axis(self, nax=0):
        axis, cbaxis = super(LinesPlotter,self).get_axis(nax, include_cbar=False)
        return LinePlotter(axis)

    def init_axis(self, n):
        super(LinesPlotter, self).init_axis(n)
