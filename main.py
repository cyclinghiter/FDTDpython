import numpy as np
import matplotlib.pyplot as plt
from space import *
from source import *

class Solver:
    def __init__(self, EMSpace):
        self.EMSpace = EMSpace
        self.sources = []
        self.structures = []
    
    def append_source(self, source):
        self.sources.append(source)
    
    def update_sources(self):
        for source in self.sources:
            x = source.positions['x']
            y = source.positions['y']
            z = source.positions['z']
            source.update_source(self.EMSpace.t)
            self._none_to_total_(self.EMSpace.E.x, source.x, x,y,z)
            self._none_to_total_(self.EMSpace.E.y, source.y, x,y,z)
            self._none_to_total_(self.EMSpace.E.z, source.z, x,y,z)
            
    def _none_to_total_(self, array, value, x, y, z):
        if x != None and y != None and z != None:
            array[x,y,z] = value
        if x != None and y != None and z == None:
            array[x,y,:] = value 
        if x != None and y == None and z != None:
            array[x,:,z] = value 
        if x != None and y == None and z == None:
            array[x,:,:] = value             
        if x == None and y != None and z != None:
            array[:,y,z] = value 
        if x == None and y != None and z == None:
            array[:,y,:] = value 
        if x == None and y == None and z != None:
            array[:,:,z] = value 
        if x == None and y == None and z == None:
            array[:,:,:] = value
    
    def set_PML(self, depth, direction):
        self.EMSpace.set_PML(depth, direction)
    
    def step(self):
        self.EMSpace.step()
        self.update_sources()
        self.EMSpace.t += self.EMSpace.dt
        self.EMSpace.apply_PEC()
        
shape = (64,64,64)
um = 1e-6
dx = 10 * um
dy = 10 * um
dz = 10 * um
dt = 1/4 * dx / constants.c
space = EMSpace(shape, dx, dy, dz, dt)

def function(t):
    return np.sin(2*np.pi*3*(10**12)*t)

solver = Solver(space)
ps1 = PlaneSource(direction= 'yz', x=12 , E_z = function)
ps2 = PlaneSource(direction= 'yz', x=-12 , E_z = function)

solver.append_source(ps1)
solver.append_source(ps2)

solver.set_PML(10, direction='xyz')

for i in range(1000):
    solver.step()
    if i % 10 == 0:
        plt.imshow(space.E.z[:,32,:], vmax=1, vmin=-1)
        plt.show()
