import numpy as np
import matplotlib.pyplot as plt 

    
def f0(t):
    '''constant zero function'''
    return 0

class Source:
    def __init__(self, x, y, z, E_x = f0, E_y = f0, E_z = f0):
        self.positions = {'x' : x, 'y' : y, 'z' : z}
        self.E_x = E_x
        self.E_y = E_y
        self.E_z = E_z
        
    def update_source(self, t):
        self.x = self.E_x(t)
        self.y = self.E_y(t)
        self.z = self.E_z(t)
        
class PointSource(Source):
    def __init__(self, x, y, z, E_x = f0, E_y = f0, E_z = f0):
        super(PointSource, self).__init__(x, y, z, E_x, E_y, E_z)
        
class LineSource(Source):
    def __init__(self, direction, E_x=f0, E_y=f0, E_z=f0, x = None, y = None, z = None):
        super(LineSource, self).__init__(x, y, z, E_x, E_y, E_z)
        if direction == 'x':
            if x != None or y == None or z == None:
                raise ValueError()
            self.positions = {'x' : x, 'y' : y, 'z' : z}
        if direction == 'y':
            if y != None or z == None or x == None:
                raise ValueError()
            self.positions = {'y' : y, 'z' : z, 'x' : x}
        if direction == 'z':
            if z != None or x == None or y == None:
                raise ValueError()
            self.positions = {'z' : z,'x' : x, 'y' : y}
            
class PlaneSource(Source):
    def __init__(self, direction, E_x=f0, E_y=f0, E_z=f0, x = None, y = None, z = None):
        super(PlaneSource, self).__init__(x, y, z, E_x, E_y, E_z)
        if direction == 'xy':
            if x != None or y != None or z == None:
                raise ValueError()
            self.positions = {'x' : x, 'y' : y, 'z' : z}
        if direction == 'yz':
            if y != None or z != None or x == None:
                raise ValueError()
            self.positions = {'y' : y, 'z' : z, 'x' : x}
        if direction == 'zx':
            if z != None or x != None or y == None:
                raise ValueError()
            self.positions = {'z' : z ,'x' : x, 'y' : y}

if __name__ == '__main__':

    def function(t):
        return np.sin(2*np.pi*3*(10**12)*t)

    plane = LineSource(direction='x', y=1, z = 1, E_z = function)
    x = plane.positions['x']
    y = plane.positions['y']
    z = plane.positions['z']

    E_z = np.zeros((32,32,32))
    temp = np.zeros((32,32,32))

    plane.update_source(2)

    temp[x,y,z] = plane.z

    E_z[x,y,z] = temp[x,y,z]

