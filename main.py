import numpy as np
import matplotlib.pyplot as plt
from space import *
from source import *
from core import *
from structure import *

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
# ps1 = PlaneSource(direction= 'yz', x=12 , E_z = function)
# ps2 = PlaneSource(direction= 'yz', x=-12 , E_z = function)

ls1 = LineSource(direction ='x', y = 15, z=15, E_x = function)
str1 = rectangle(shape = shape, center = (32,32,32), depth=15, width=15, height=15, eps=4, mu=1)

# solver.append_source(ps1)
solver.append_source(ls1)
solver.append_structure(str1)
solver.set_PML(depth=10, direction='xyz')

for i in range(1000):
    solver.step()
    if i % 10 == 0:
        plt.imshow(space.E.x[32,:,:], vmax=1, vmin=-1)
        plt.show()
