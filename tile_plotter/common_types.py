"""Commonly used types."""
from typing import Tuple, TypeVar, Union

Location = Tuple[int, int]
Map = TypeVar('Map', 'astropy.units.Quantity', 'astropy.io.PrimaryHDU')
PlotHandler = TypeVar('PlotHandler')
Position = Tuple[float, float]
Quantity = Union['astropy.units.Quantity', float, 'numpy.typing.ArrayLike']

