from typing import List
# Future update for Python 3.7+
# from dataclasses import dataclass

#@dataclass
class FigGeometry(metaclass=type):
    """Figure geometry class"""
    # Future update for Python 3.7+
    #xsize: float
    #ysize: float
    #left: float = 0
    #right: float = 0
    #top: float = 0
    #bottom: float = 0
    #location: List[int] = [0, 0]

    # To be remove in Python 3.7+
    def __init__(self, xsize: float, ysize: float, left: float = 0, 
            right: float = 0, bottom: float = 0, top: float = 0, 
            location: List[int] = [0, 0]):
        self.xsize = xsize
        self.ysize = ysize
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.location = location

    # To be remove in Python 3.7+
    def __repr__(self):
        text = f'x size: {self.xsize}\n'
        text += f'y size: {self.ysize}\n'
        text += f'left margin: {self.left}\n'
        text += f'right margin: {self.rigth}\n'
        text += f'bottom margin: {self.bottom}\n'
        text += f'top margin: {self.top}\n'
        text += f'axis location: {self.location}\n'

        return text

    @property
    def height(self):
        return self.bottom + self.ysize + self.top

    @property
    def width(self):
        return self.left + self.xsize + self.right

    @property
    def axis(self):
        return [self.location[0]+self.left, self.location[1]+self.bottom, 
                self.xsize, self.ysize]

    def scalex(self, factor: float):
        """Scale all the horizontal variables by factor"""
        self.xsize = self.xsize * factor
        self.left = self.left * factor
        self.right = self.right * factor
        self.location[0] = self.location[0]*factor

    def scaley(self, factor: float):
        """Scale all the vertical variables by factor"""
        self.ysize = self.ysize*factor
        self.bottom = self.bottom*factor
        self.top = self.top*factor
        self.location[1] = self.location[1]*factor

    def is_empty(self):
        return self.xsize==self.ysize==self.left==self.right==self.bottom==self.top==0
