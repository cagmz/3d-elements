"""
A 2D numpy array is used to represent an element (fire, water, cloud, vegetation).
Each cell has a 'strength' value associated with it; min_z, max_z are used to normalize the strength values
in order to compare against other neighborhoods.
"""

import numpy as np


class Neighborhood:
    def __init__(self, min_z, max_z, size):
        self.relative_min, self.relative_max = min_z, max_z
        self.cells = np.random.uniform(low=min_z, high=max_z, size=size)
        self.cells_copy = self.cells.copy()
        self.shape = self.cells.shape
