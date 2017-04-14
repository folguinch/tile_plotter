__metaclass__ = type

class FigGeometry:

    def __init__(self, xsize, ysize, left=0, right=0, bottom=0, top=0):
        self.xsize = xsize
        self.ysize = ysize
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top

    def __repr__(self):
        text = 'x size: %.1f\n' % self.xsize
        text += 'y size: %.1f\n' % self.ysize
        text += 'left margin: %.1f\n' % self.left
        text += 'right margin: %.1f\n' % self.right 
        text += 'bottom margin: %.1f\n' % self.bottom 
        text += 'top margin: %.1f\n' % self.top

        return text

    @property
    def height(self):
        return self.bottom+self.ysize+self.top

    @property
    def width(self):
        return self.left+self.xsize+self.right

    @property
    def axis(self):
        return [self.left, self.bottom, self.xsize, self.ysize]

    def scalex(self, factor):
        self.xsize = self.xsize*factor
        self.left = self.left*factor
        self.right = self.right*factor

    def scaley(self, factor):
        self.ysize = self.ysize*factor
        self.bottom = self.bottom*factor
        self.top = self.top*factor
