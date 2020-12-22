import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm

from space import *
from source import *
from core import *
from structure import *

shape = (128,128,128)
um = 1e-6
dx = 10 * um
dy = 10 * um
dz = 10 * um
dt = 1/4 * dx / constants.c

TF = EMSpace(shape, dx, dy, dz, dt)
IF = EMSpace(shape, dx, dy, dz, dt)
SF = EMSpace(shape, dx, dy, dz, dt)

def function(t):
    return np.sin(2*np.pi*0.9*(10**12)*t)

solver = Solver(TF=TF, SF=SF, IF=IF)

s1 = PlaneSource(direction = 'xy', z=20, E_x = function)
solver.append_source(s1)

# str1 = Rectangle(shape = shape, center = (32,32,32), depth=5, width=5, height=5, eps=4, mu=1)
# solver.append_structure(str1)

str1 = Sphere(shape = shape, center = (64,64,64), R = 10, eps=4, mu=1)
solver.append_structure(str1)

solver.set_PML(depth=10, direction='xyz')

# from mayavi import mlab
# mlab.contour3d(TF.epsr)
# mlab.show()

def Imshow(input, fig, axes, ax_num, vmax=None, vmin=None, title=None):
        cmap = cm.bwr
        if vmax == None:
            vmax = np.max(input)
        if vmin == None:
            vmin = np.min(input)
        no_cbar_ticks = False
        im = axes[ax_num].imshow(input, vmax=vmax, vmin=vmin, origin='upper')
        axes[ax_num].set_title(title)
        axes[ax_num].grid(False)
        divider = make_axes_locatable(axes[ax_num])
        cax = divider.append_axes('left', size='5%', pad=0.1)
        axes[ax_num].yaxis.set_ticks_position('right')
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cax.yaxis.set_ticks_position('left')    


import tqdm

for i in tqdm.tqdm(range(1000)):
    solver.step()
    if i % 50 == 0:
        fig, axes = plt.subplots(2,4, figsize = (20, 8))
        Imshow(SF.epsr[64,:,:], fig, axes, (0,0), title = "epsilon r")
        Imshow(SF.E.x[64,:,:], fig, axes, (0,1), title = 'Ex center')
        Imshow(SF.E.y[64,:,:], fig, axes, (0,2), title = 'Ey center')
        Imshow(SF.E.z[64,:,:], fig, axes, (0,3), title = 'Ez center')
        Imshow(SF.mur[64,:,:], fig, axes, (1,0), title = "mu r")
        Imshow(SF.E.x[:,:,-11], fig, axes, (1,1), title = 'Ex last')
        Imshow(SF.E.y[:,:,-11], fig, axes, (1,2), title = 'Ey last')
        Imshow(SF.E.z[:,:,-11], fig, axes, (1,3), title = 'Ez last')
        plt.savefig("result/tstep={}.png".format(i))
        plt.close("all")