import numpy as np
import matplotlib.pyplot as plt


class Source:
    def __init__(self, function):
        self.function = function

    def update_Source(self, t):
        self.value = self.function(t)
        return self.value

class PointSource(Source):
    def __init__(self, function, position):
        super(PointSource, self).__init__(function)
        self.position = position
    
    def supply_source(self, matrix):
        matrix[*self.position] += self.value
    
    
    