import numpy as np
import matplotlib.pyplot as plt
from structure import *
from source import *
from space import *
from system import deprecated

class Solver:
    def __init__(self, EMSpace):
        self.EMSpace = EMSpace
        self.sources = []
        self.structures = []
    
    def append_source(self, source):
        self.sources.append(source)
    
    def append_structure(self, structure):
        structure.change_space(self.EMSpace)
        
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
        