import numpy as np
import matplotlib.pyplot as plt
from structure import *
from source import *
from space import *
from system import deprecated

class Solver:
    def __init__(self, TF = None , SF=None, IF=None, mode='SF'):
        self.mode = mode
        if self.mode == 'SF':
            assert SF != None and IF != None and TF != None
            self.TF = TF
            self.SF = SF 
            self.IF = IF 
            self.field_to_update = [self.TF, self.IF]
            self.field_to_state = [self.TF, self.SF]
            
        if self.mode == 'TF':
            assert TF != None
            self.TF = TF
            self.field_to_update = [self.TF]
            self.field_to_state = [self.TF]
            
        if self.mode == 'IF':
            assert IF != None
            self.IF = IF 
            self.field_to_update = [self.IF]
            self.field_to_state = []
            
        
        self.sources = []
        self.structures = []
    
    def append_source(self, source):
        self.sources.append(source)

    def append_structure(self, structure):
        self.structures.append(structure)
        for Field in self.field_to_state:
            structure.change_space(Field)

    def update_sources(self):
        for source in self.sources:
            x = source.positions['x']
            y = source.positions['y']
            z = source.positions['z']
            for Field in self.field_to_update:
                source.update_source(Field.t)
                self._none_to_total_(Field.E.x, source.x, x,y,z)
                self._none_to_total_(Field.E.y, source.y, x,y,z)
                self._none_to_total_(Field.E.z, source.z, x,y,z)
            
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
        for Field in self.field_to_update:
            Field.set_PML(depth, direction)
    
    def step(self):
        self.update_sources()
        if self.mode == 'SF' or self.mode == 'TF':
            self.TF.step()
            self.TF.apply_PEC()
            self.TF.t += self.TF.dt
        
        if self.mode == 'SF' or self.mode == 'IF':
            self.IF.step()
            self.IF.t += self.IF.dt
            
            self.IF.apply_PEC()
        
        if self.mode == 'SF':
            self.SF.E.x = self.TF.E.x - self.IF.E.x
            self.SF.E.y = self.TF.E.y - self.IF.E.y
            self.SF.E.z = self.TF.E.z - self.IF.E.z
            self.SF.H.x = self.TF.H.x - self.IF.H.x
            self.SF.H.y = self.TF.H.y - self.IF.H.y
            self.SF.H.z = self.TF.H.z - self.IF.H.z
    
        