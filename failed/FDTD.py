import numpy as np 
import matplotlib.pyplot as plt
import sys
import matplotlib.animation as animation
import tqdm

class constant:
    c = 299792458
    mu0 = 4.0 * np.pi * 1e-7
    eps0 = 1.0 / (mu0 * c*c)
    imp0 = np.sqrt(mu0 / eps0)
    um = 1e-6

class VectorField:
    
    """
    
    E field, B field initialization.
    
    """
    def __init__(self, shape):
        self.shape = shape
        self.x = np.zeros(shape)
        self.y = np.zeros(shape) 
        self.z = np.zeros(shape) 
        
    def copy(self):
        new_field = VectorField(self.shape)
        new_field.x = self.x.copy()
        new_field.y = self.y.copy()
        new_field.z = self.z.copy()
        return new_field
    
        
class Space:
    
    """
    loss가 존재하지 않는 space initialization. update equation을 간단하게 작성할 수 있음.
    """
    
    def __init__(self, E, H, dt, dx ,dy, dz):
        assert E.shape == H.shape
        self.shape = E.shape
        self.E = E
        self.H = H
        self.dt = dt
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.eps0 = constant.eps0 
        self.mu0 = constant.mu0
        self.epsr = np.ones(self.shape)
        self.mur = np.ones(self.shape)

    def step(self):
        self.updateH()
        self.updateE()

    def updateE(self):
        self.E.x[:-1, 1:-1, 1:-1] = self.E.x[:-1, 1:-1, 1:-1] + self.dt / (self.epsr * self.eps0)[:-1, 1:-1, 1:-1] * ((self.H.z[:-1, 1:-1, 1:-1] - self.H.z[:-1, :-2, 1:-1])/self.dy - (self.H.y[:-1, 1:-1, 1:-1] - self.H.y[:-1, 1:-1, :-2])/self.dz)
        self.E.y[1:-1, :-1, 1:-1] = self.E.y[1:-1, :-1, 1:-1] + self.dt / (self.epsr * self.eps0)[1:-1, :-1, 1:-1] * ((self.H.x[1:-1, :-1, 1:-1] - self.H.x[1:-1, :-1, :-2])/self.dz - (self.H.z[1:-1, :-1, 1:-1] - self.H.z[:-2, :-1, 1:-1])/self.dx)    
        self.E.z[1:-1, 1:-1, :-1] = self.E.z[1:-1, 1:-1, :-1] + self.dt / (self.epsr * self.eps0)[1:-1, 1:-1, :-1] * ((self.H.y[1:-1, 1:-1, :-1] - self.H.y[:-2, 1:-1, :-1])/self.dx - (self.H.x[1:-1, 1:-1, :-1] - self.H.x[1:-1, :-2, :-1])/self.dy)            
        print(self.dt / (self.epsr * self.eps0))
    def updateH(self):
        self.H.x[:, :-1, :-1] =  self.H.x[:, :-1, :-1] - self.dt / (self.mur * self.mu0)[:, :-1, :-1] * ((self.E.z[:, 1:, :-1] - self.E.z[:, :-1, :-1])/self.dy - (self.E.y[:, :-1, 1:] - self.E.y[:, :-1, :-1])/self.dz)
        self.H.y[:-1, :, :-1] =  self.H.y[:-1, :, :-1] - self.dt / (self.mur * self.mu0)[:-1, :, :-1] * ((self.E.x[:-1, :, 1:] - self.E.x[:-1, :, :-1])/self.dz - (self.E.z[1:, :, :-1] - self.E.z[:-1, :, :-1])/self.dx)    
        self.H.z[:-1, :-1, :] =  self.H.z[:-1, :-1, :] - self.dt / (self.mur * self.mu0)[:-1, :-1, :] * ((self.E.y[1:, :-1, :] - self.E.y[:-1, :-1, :])/self.dx - (self.E.x[:-1, 1:, :] - self.E.x[:-1, :-1, :])/self.dy)  

    def put_source(self, source, t):
        pass 
    
    def append_structure(self, structure):
        self.structure_list.append(structure)
    
    def show_status(self, display):
        pass
    
    

class PMLSpace():
    
    """
    Abstract Class of PML Space. D, E, B, H field must be implemented according to direction.
    """
    
    def __init__(self, Space, sigma_max, depth):
        self.dt = Space.dt
        self.dx = Space.dx
        self.dy = Space.dy
        self.dz = Space.dz
        self.shape = Space.shape
        self.depth = depth
        self.sigma_max = sigma_max
        
    def updateB(self):
        pass
    
    def updateH(self):
        pass
    
    def updateD(self):
        pass
        
    def updateE(self):
        pass
    
    def step(self):
        self.UpdateB()
        self.UpdateH()
        self.UpdateD()
        self.UpdateE()
    

class PMLxSpace(PMLSpace):
    
    """
    
    Uniaxial PML for x direction. Reflection for x direction is 0.
    
    """
    
    def __init__(self, Space, sigma_max, depth):
        super(PMLxSpace, self).__init__(Space, sigma_max, depth)
        self.shape = (depth, self.shape[1], self.shape[2])
        # self.sigma = self.sigma_max * np.ones(self.shape)
        self.sigma = np.zeros(self.shape)
        for i in range(depth):
            self.sigma[i,:,:] = self.sigma_max * i / depth
        self.D = np.zeros(self.shape)
        self.D_temp = np.zeros(self.shape)
        self.E = VectorField(self.shape)
        self.B = np.zeros(self.shape)
        self.B_temp = np.zeros(self.shape)
        self.H = VectorField(self.shape)
        self.eps0 = constant.eps0
        self.mu0 = constant.mu0
        self.epsr = np.ones(self.shape)
        self.mur = np.ones(self.shape)
        
    def UpdateD(self):
        self.D_temp = self.D.copy()
        # self.D[:-1, 1:-1, 1:-1] = self.D[:-1, 1:-1, 1:-1] + self.dt * ((self.H.z[:-1, 1:-1, 1:-1] - self.H.z[:-1, :-2, 1:-1])/self.dy - (self.H.y[:-1, 1:-1, 1:-1] - self.H.y[:-1, 1:-1, :-2])/self.dz)
        self.D[:-1, 1:-1, 1:-1] = self.D[:-1, 1:-1, 1:-1] + self.dt * ((self.H.z[:-1, 1:-1, 1:-1] - self.H.z[:-1, :-2, 1:-1])/self.dy - (self.H.y[:-1, 1:-1, 1:-1] - self.H.y[:-1, 1:-1, :-2])/self.dz)
        
        
    def UpdateE(self):
        Cexe = (2 * self.eps0 - self.sigma * self.dt) / (2 * self.eps0 + self.sigma * self.dt)
        Cexh =  2 * self.dt / self.epsr / (2*self.eps0 + self.sigma * self.dt)
        # self.E.y[1:-1, :-1, 1:-1] = Cexe[1:-1, :-1, 1:-1]  * self.E.y[1:-1, :-1, 1:-1] + Cexh[1:-1, :-1, 1:-1]  * ((self.H.x[1:-1, :-1, 1:-1] - self.H.x[1:-1, :-1, :-2])/self.dz - (self.H.z[1:-1, :-1, 1:-1] - self.H.z[:-2, :-1, 1:-1])/self.dx)    
        # self.E.z[1:-1, 1:-1, :-1] = Cexe[1:-1, 1:-1, :-1] * self.E.z[1:-1, 1:-1, :-1] + Cexh[1:-1, 1:-1, :-1] * ((self.H.y[1:-1,1:-1,:-1] - self.H.y[:-2,1:-1,:-1]) /self.dx - (self.H.x[1:-1,1:-1,:-1] - self.H.x[1:-1,:-2,:-1])/self.dy)
        self.E.y[1:-1, :-1, 1:-1] = Cexe[1:-1, :-1, 1:-1] * self.E.y[1:-1, :-1, 1:-1] + Cexh[1:-1, :-1, 1:-1] * ((self.H.x[1:-1, :-1, 1:-1] - self.H.x[1:-1, :-1, :-2])/self.dz - (self.H.z[1:-1, :-1, 1:-1] - self.H.z[:-2, :-1, 1:-1])/self.dx)    
        self.E.z[1:-1, 1:-1, :-1] = Cexe[1:-1, 1:-1, :-1] * self.E.z[1:-1, 1:-1, :-1] + Cexh[1:-1, 1:-1, :-1] * ((self.H.y[1:-1, 1:-1, :-1] - self.H.y[:-2, 1:-1, :-1])/self.dx - (self.H.x[1:-1, 1:-1, :-1] - self.H.x[1:-1, :-2, :-1])/self.dy)
        self.E.x = self.E.x + 1/(self.epsr *self.eps0) * (self.D*(1 + self.sigma * self.dt / (2 * self.eps0)) - self.D_temp * (1- self.sigma * self.dt / (2 * self.eps0)))
    
    def UpdateB(self):
        self.B_temp = self.B.copy()
        # self.B[:, :-1, :-1] = self.B[:,:-1,:-1] -self.dt * ((self.E.z[:, 1:, :-1] - self.E.z[:, :-1, :-1])/self.dy - (self.E.y[:, :-1, 1:] - self.E.y[:, :-1, :-1])/self.dz)
        self.B[:, :-1, :-1] =  self.B[:, :-1, :-1] - self.dt * ((self.E.z[:, 1:, :-1] - self.E.z[:, :-1, :-1])/self.dy - (self.E.y[:, :-1, 1:] - self.E.y[:, :-1, :-1])/self.dz)


    def UpdateH(self):
        Chxh = (2 * self.eps0 - self.sigma * self.dt) / (2 * self.eps0 + self.sigma * self.dt)
        Chxe = 2 * self.dt * self.eps0 / (self.mur * self.mu0) / (2*self.eps0 + self.sigma * self.dt)
        
        # self.H.y[:-1, :, :-1] = Chxh[:-1,:,:-1] * self.H.y[:-1, :, :-1] - Chxe[:-1,:,:-1] * ((self.E.x[:-1, :, 1:] - self.E.x[:-1, :, :-1])/self.dz - (self.E.z[1:, :, :-1] - self.E.z[:-1, :, :-1])/self.dx)                    
        # self.H.z[:-1, :-1, :] = Chxh[:-1,:-1,:] * self.H.y[:-1, :-1, :] - Chxe[:-1, :-1, :] * ((self.E.y[1:, :-1, :] - self.E.y[:-1, :-1, :])/self.dx - (self.E.x[:-1, 1:, :] - self.E.x[:-1, :-1, :])/self.dy)            
        self.H.y[:-1, :, :-1] =  Chxh[:-1,:,:-1] * self.H.y[:-1, :, :-1] - Chxe[:-1,:,:-1] * ((self.E.x[:-1, :, 1:] - self.E.x[:-1, :, :-1])/self.dz - (self.E.z[1:, :, :-1] - self.E.z[:-1, :, :-1])/self.dx)    
        self.H.z[:-1, :-1, :] =  Chxh[:-1,:-1,:] * self.H.z[:-1, :-1, :] - Chxe[:-1, :-1, :] * ((self.E.y[1:, :-1, :] - self.E.y[:-1, :-1, :])/self.dx - (self.E.x[:-1, 1:, :] - self.E.x[:-1, :-1, :])/self.dy)
        self.H.x = self.H.x + 1/(self.mur * self.mu0) * (self.B*(1 + self.sigma * self.dt / 2 / self.eps0) - self.B_temp*(1- self.sigma * self.dt / 2 / self.eps0))


class PMLySpace(PMLSpace):
    
    """
    
    Uniaxial PML for y direction. Reflection for y direction is 0.
    
    """
    
    def __init__(self, Space, sigma_max, depth):
        super(PMLySpace, self).__init__(Space, sigma_max, depth)
        self.shape = (self.shape[0] , depth, self.shape[2])
        self.sigma = np.zeros(self.shape)
        for i in range(depth):
            self.sigma[:,i,:] = self.sigma_max * i / depth
        self.D = np.zeros(self.shape)
        self.D_temp = np.zeros(self.shape)
        self.E = VectorField(self.shape)
        self.B = np.zeros(self.shape)
        self.B_temp = np.zeros(self.shape)
        self.H = VectorField(self.shape)
        self.eps0 = constant.eps0
        self.mu0 = constant.mu0
        self.epsr = np.ones(self.shape)
        self.mur = np.ones(self.shape)

    def UpdateD(self):
        self.D_temp = self.D.copy()
        self.D[1:-1, :-1, 1:-1] = self.D[1:-1, :-1, 1:-1] + self.dt * ((self.H.x[1:-1, :-1, 1:-1] - self.H.x[1:-1, :-1, :-2])/self.dz - (self.H.z[1:-1, :-1, 1:-1] - self.H.z[:-2, :-1, 1:-1])/self.dx)    


    def UpdateE(self):
        Cexe = (2 * self.eps0 - self.sigma * self.dt) / (2 * self.eps0 + self.sigma * self.dt)
        Cexh =  2 * self.dt / self.epsr / (2*self.eps0 + self.sigma * self.dt)                
        self.E.x[:-1, 1:-1, 1:-1] = Cexe[:-1, 1:-1, 1:-1] * self.E.x[:-1, 1:-1, 1:-1] + Cexh[:-1, 1:-1, 1:-1] * ((self.H.z[:-1, 1:-1, 1:-1] - self.H.z[:-1, :-2, 1:-1])/self.dy - (self.H.y[:-1, 1:-1, 1:-1] - self.H.y[:-1, 1:-1, :-2])/self.dz)
        self.E.z[1:-1, 1:-1, :-1] = Cexe[1:-1, 1:-1, :-1] * self.E.z[1:-1, 1:-1, :-1] + Cexh[1:-1, 1:-1, :-1] * ((self.H.y[1:-1, 1:-1, :-1] - self.H.y[:-2, 1:-1, :-1])/self.dx - (self.H.x[1:-1, 1:-1, :-1] - self.H.x[1:-1, :-2, :-1])/self.dy)        
        self.E.y = self.E.y + 1/(self.epsr *self.eps0) * (self.D*(1 + self.sigma * self.dt / (2 * self.eps0)) - self.D_temp * (1- self.sigma * self.dt / (2 * self.eps0)))
    
    def UpdateB(self):
        self.B_temp = self.B.copy()
        self.B[:-1,:,:-1] = self.B[:-1, :, :-1] - self.dt * ((self.E.x[:-1, :, 1:] - self.E.x[:-1, :, :-1])/self.dz - (self.E.z[1:, :, :-1] - self.E.z[:-1, :, :-1])/self.dx)    


    def UpdateH(self):
        Chxh = (2 * self.eps0 - self.sigma * self.dt) / (2 * self.eps0 + self.sigma * self.dt)
        Chxe = 2 * self.dt * self.eps0 / (self.mur * self.mu0) / (2*self.eps0 + self.sigma * self.dt)
        self.H.x[:, :-1, :-1] = Chxh[:,:-1,:-1] * self.H.x[:, :-1, :-1] - Chxe[:,:-1,:-1] * ((self.E.z[:, 1:, :-1] - self.E.z[:, :-1, :-1])/self.dy - (self.E.y[:, :-1, 1:] - self.E.y[:, :-1, :-1])/self.dz)
        self.H.z[:-1, :-1, :] =  Chxh[:-1,:-1,:] * self.H.z[:-1, :-1, :] - Chxe[:-1, :-1, :] * ((self.E.y[1:, :-1, :] - self.E.y[:-1, :-1, :])/self.dx - (self.E.x[:-1, 1:, :] - self.E.x[:-1, :-1, :])/self.dy)
        self.H.y = self.H.y + 1/(self.mur * self.mu0) * (self.B*(1 + self.sigma * self.dt / 2 / self.eps0) - self.B_temp*(1- self.sigma * self.dt / 2 / self.eps0))


class PMLzSpace(PMLSpace):
    
    """
    
    Uniaxial PML for z direction. Reflection for z direction is 0.
    
    """    
    
    def __init__(self, Space, sigma_max, depth):
        super(PMLzSpace, self).__init__(Space, sigma_max, depth)
        self.shape = (self.shape[0] , self.shape[1], depth)
        self.sigma = np.zeros(self.shape)
        for i in range(depth):
            self.sigma[:,:,i] = self.sigma_max * i / depth
        self.D = np.zeros(self.shape)
        self.D_temp = np.zeros(self.shape)
        self.E = VectorField(self.shape)
        self.B = np.zeros(self.shape)
        self.B_temp = np.zeros(self.shape)
        self.H = VectorField(self.shape)
        self.eps0 = constant.eps0
        self.mu0 = constant.mu0
        self.epsr = np.ones(self.shape)
        self.mur = np.ones(self.shape)

    def UpdateD(self):
        self.D_temp = self.D.copy()
        self.D[1:-1, 1:-1, :-1] = self.D[1:-1, 1:-1, :-1] + self.dt * ((self.H.y[1:-1, 1:-1, :-1] - self.H.y[:-2, 1:-1, :-1])/self.dx - (self.H.x[1:-1, 1:-1, :-1] - self.H.x[1:-1, :-2, :-1])/self.dy)            

    def UpdateE(self):
        Cexe = (2 * self.eps0 - self.sigma * self.dt) / (2 * self.eps0 + self.sigma * self.dt)
        Cexh =  2 * self.dt / self.epsr / (2*self.eps0 + self.sigma * self.dt)                
        self.E.x[:-1, 1:-1, 1:-1] = Cexe[:-1, 1:-1, 1:-1] * self.E.x[:-1, 1:-1, 1:-1] + Cexh[:-1, 1:-1, 1:-1] * ((self.H.z[:-1, 1:-1, 1:-1] - self.H.z[:-1, :-2, 1:-1])/self.dy - (self.H.y[:-1, 1:-1, 1:-1] - self.H.y[:-1, 1:-1, :-2])/self.dz)       
        self.E.y[1:-1, :-1, 1:-1] = Cexe[1:-1, :-1, 1:-1] * self.E.y[1:-1, :-1, 1:-1] + Cexh[1:-1, :-1, 1:-1] * ((self.H.x[1:-1, :-1, 1:-1] - self.H.x[1:-1, :-1, :-2])/self.dz - (self.H.z[1:-1, :-1, 1:-1] - self.H.z[:-2, :-1, 1:-1])/self.dx)            
        self.E.z = self.E.z + 1/(self.epsr *self.eps0) * (self.D*(1 + self.sigma * self.dt / (2 * self.eps0)) - self.D_temp * (1- self.sigma * self.dt / (2 * self.eps0)))
    
    def UpdateB(self):
        self.B_temp = self.B.copy()
        self.B[:-1,:-1,:] = self.B[:-1,:-1,:] - self.dt * ((self.E.y[1:,:-1,:] - self.E.y[:-1,:-1,:]) /self.dx - (self.E.x[:-1,1:,:] - self.E.x[-1:,:-1,:])/self.dy)

    def UpdateH(self):
        Chxh = (2 * self.eps0 - self.sigma * self.dt) / (2 * self.eps0 + self.sigma * self.dt)
        Chxe = 2 * self.dt * self.eps0 / (self.mur * self.mu0) / (2*self.eps0 + self.sigma * self.dt)
        self.H.x[:, :-1, :-1] = Chxh[:,:-1,:-1] * self.H.x[:, :-1, :-1] - Chxe[:,:-1,:-1] * ((self.E.z[:, 1:, :-1] - self.E.z[:, :-1, :-1])/self.dy - (self.E.y[:, :-1, 1:] - self.E.y[:, :-1, :-1])/self.dz)        
        self.H.y[:-1, :, :-1] =  Chxh[:-1,:,:-1] * self.H.y[:-1, :, :-1] - Chxe[:-1,:,:-1] * ((self.E.x[:-1, :, 1:] - self.E.x[:-1, :, :-1])/self.dz - (self.E.z[1:, :, :-1] - self.E.z[:-1, :, :-1])/self.dx)            
        self.H.z = self.H.z + 1/(self.mur * self.mu0) * (self.B*(1 + self.sigma * self.dt / 2 / self.eps0) - self.B_temp*(1- self.sigma * self.dt / 2 / self.eps0))



class FDTDSolver:
    def __init__(self, Space):
        self.Space = Space
        self.E = self.Space.E.copy()
        self.H = self.Space.H.copy()
        self.dt = Space.dt
        self.dx = Space.dx
        self.dy = Space.dy
        self.dz = Space.dz
        self.shape = Space.shape
        self.t = 0
        self.tstep = 0
        self.PML_list = []
        

    def set_PML(self, sigma_max, depth, direction, name=None):
    
        if direction == 'x+':
            self.PML_list.append({'PML':PMLxSpace(self.Space, sigma_max, depth),
                                  'where' : 'x+',
                                  'name' : name})
        if direction == 'x-':
            self.PML_list.append({'PML':PMLxSpace(self.Space, sigma_max, depth),
                                  'where' : 'x-',
                                  'name' : name})        
        if direction == 'y+':
            self.PML_list.append({'PML':PMLySpace(self.Space, sigma_max, depth),
                                  'where' : 'y+',
                                  'name' : name})
        if direction == 'y-':
            self.PML_list.append({'PML':PMLySpace(self.Space, sigma_max, depth),
                                  'where' : 'y-',
                                  'name' : name})         
        if direction == 'z+':
            self.PML_list.append({'PML':PMLzSpace(self.Space, sigma_max, depth),
                                  'where' : 'z+',
                                  'name' : name})
        if direction == 'z-':
            self.PML_list.append({'PML':PMLzSpace(self.Space, sigma_max, depth),
                                  'where' : 'z-',
                                  'name' : name})

    def step_pml(self):
        E = self.Space.E.copy()
        H = self.Space.H.copy()
        for pml_to_update in self.PML_list:
            PML = pml_to_update['PML']
            d = PML.depth
            if pml_to_update['where'] == 'x+':
                 PML.E.x = E.x[-PML.depth:,:,:]
                 PML.E.y = E.y[-PML.depth:,:,:]
                 PML.E.z = E.z[-PML.depth:,:,:]
                 PML.H.x = H.x[-PML.depth:,:,:]
                 PML.H.y = H.y[-PML.depth:,:,:]
                 PML.H.z = H.z[-PML.depth:,:,:]                 
            if pml_to_update['where'] == 'x-':
                 PML.E.x = E.x[:PML.depth,:,:]
                 PML.E.y = E.y[:PML.depth,:,:]
                 PML.E.z = E.z[:PML.depth,:,:]
                 PML.H.x = H.x[:PML.depth,:,:]
                 PML.H.y = H.y[:PML.depth,:,:]
                 PML.H.z = H.z[:PML.depth,:,:]
            if pml_to_update['where'] == 'y+':
                 PML.E.x = E.x[:,-PML.depth:,:]
                 PML.E.y = E.y[:,-PML.depth:,:]
                 PML.E.z = E.z[:,-PML.depth:,:]
                 PML.H.x = H.x[:,-PML.depth:,:]
                 PML.H.y = H.y[:,-PML.depth:,:]
                 PML.H.z = H.z[:,-PML.depth:,:]
            if pml_to_update['where'] == 'y-':
                 PML.E.x = E.x[:,:PML.depth,:]
                 PML.E.y = E.y[:,:PML.depth,:]
                 PML.E.z = E.z[:,:PML.depth,:]
                 PML.H.x = H.x[:,:PML.depth,:]                 
                 PML.H.y = H.y[:,:PML.depth,:]                 
                 PML.H.z = H.z[:,:PML.depth,:]                 
            if pml_to_update['where'] == 'z+':
                 PML.E.x = E.x[:,:,-PML.depth:]
                 PML.E.y = E.y[:,:,-PML.depth:]
                 PML.E.z = E.z[:,:,-PML.depth:]
                 PML.H.x = H.x[:,:,-PML.depth:]
                 PML.H.y = H.y[:,:,-PML.depth:]
                 PML.H.z = H.z[:,:,-PML.depth:]
            if pml_to_update['where'] == 'z-':
                 PML.E.x = E.x[:,:,:PML.depth]
                 PML.E.y = E.y[:,:,:PML.depth]
                 PML.E.z = E.z[:,:,:PML.depth]
                 PML.H.x = H.x[:,:,:PML.depth]
                 PML.H.y = H.y[:,:,:PML.depth]
                 PML.H.z = H.z[:,:,:PML.depth]
         
            PML.step()
    
    
    def apply_PML_to_Space(self):
        for pml_to_update in self.PML_list:            
            PML = pml_to_update['PML']
            d = PML.depth
            update_depth = d - 1
            if pml_to_update['where'] == 'x+':
                 self.Space.E.x[-update_depth:,:, :] = PML.E.x[1:,:,:]
                 self.Space.E.y[-update_depth:,:, :] = PML.E.y[1:,:,:]
                 self.Space.E.z[-update_depth:,:, :] = PML.E.z[1:,:,:]
                 self.Space.H.x[-update_depth:,:, :] = PML.H.x[1:,:,:]
                 self.Space.H.y[-update_depth:,:, :] = PML.H.y[1:,:,:]
                 self.Space.H.z[-update_depth:,:, :] = PML.H.z[1:,:,:]
            if pml_to_update['where'] == 'x-':
                 self.Space.E.x[:update_depth,:,:] = PML.E.x[:-1,:,:]
                 self.Space.E.y[:update_depth,:,:] = PML.E.y[:-1,:,:]
                 self.Space.E.z[:update_depth,:,:] = PML.E.z[:-1,:,:]
                 self.Space.H.x[:update_depth,:,:] = PML.H.x[:-1,:,:]
                 self.Space.H.y[:update_depth,:,:] = PML.H.y[:-1,:,:]
                 self.Space.H.z[:update_depth,:,:] = PML.H.z[:-1,:,:]
            if pml_to_update['where'] == 'y+':
                 self.Space.E.x[:,-update_depth:,:] = PML.E.x[:,1:,:]
                 self.Space.E.y[:,-update_depth:,:] = PML.E.y[:,1:,:]
                 self.Space.E.z[:,-update_depth:,:] = PML.E.z[:,1:,:]
                 self.Space.H.x[:,-update_depth:,:] = PML.H.x[:,1:,:]
                 self.Space.H.y[:,-update_depth:,:] = PML.H.y[:,1:,:]
                 self.Space.H.z[:,-update_depth:,:] = PML.H.z[:,1:,:]
            if pml_to_update['where'] == 'y-':
                 self.Space.E.x[:,:update_depth,:] = PML.E.x[:,:-1,:]
                 self.Space.E.y[:,:update_depth,:] = PML.E.y[:,:-1,:]
                 self.Space.E.z[:,:update_depth,:] = PML.E.z[:,:-1,:]
                 self.Space.H.x[:,:update_depth,:] = PML.H.x[:,:-1,:]       
                 self.Space.H.y[:,:update_depth,:] = PML.H.y[:,:-1,:]         
                 self.Space.H.z[:,:update_depth,:] = PML.H.z[:,:-1,:]  
            if pml_to_update['where'] == 'z+':
                 self.Space.E.x[:,:,-update_depth:] = PML.E.x[:,:,1:]
                 self.Space.E.y[:,:,-update_depth:] = PML.E.y[:,:,1:]
                 self.Space.E.z[:,:,-update_depth:] = PML.E.z[:,:,1:]
                 self.Space.H.x[:,:,-update_depth:] = PML.H.x[:,:,1:]
                 self.Space.H.y[:,:,-update_depth:] = PML.H.y[:,:,1:]
                 self.Space.H.z[:,:,-update_depth:] = PML.H.z[:,:,1:]
            if pml_to_update['where'] == 'z-':
                 self.Space.E.x[:,:,:update_depth] = PML.E.x[:,:,:-1]
                 self.Space.E.y[:,:,:update_depth] = PML.E.y[:,:,:-1]
                 self.Space.E.z[:,:,:update_depth] = PML.E.z[:,:,:-1]
                 self.Space.H.x[:,:,:update_depth] = PML.H.x[:,:,:-1]   
                 self.Space.H.y[:,:,:update_depth] = PML.H.y[:,:,:-1]   
                 self.Space.H.z[:,:,:update_depth] = PML.H.z[:,:,:-1]   


    def step(self):
        self.step_pml()
        self.Space.step()
        self.apply_PML_to_Space()
        self.update_source()
        self.t += self.dt
        self.tstep += 1
        self.E = self.Space.E.copy()
        self.H = self.Space.H.copy()

    def update_source(self):        
        self.Space.E.x[:,:,:] = 0
        self.Space.E.y[:,:,:] = 0
    
        self.Space.E.z[32,32,32] = np.sin(2*np.pi*1.4*(10**12)*self.t)


shape = (64,64,64)
E = VectorField(shape)
H = VectorField(shape)

um = 1e-6
dx = 10 * um
dy = 10 * um
dz = 10 * um

dt = 1/4 * dx / constant.c

depth = 6
A = Space(E,H,dt=dt,dx=dx,dy=dy,dz=dz)

solver = FDTDSolver(A)
solver.set_PML(sigma_max = 0 , depth=depth, direction ='x+', name='x+')
solver.set_PML(sigma_max = 0 , depth=depth, direction ='x-', name='x-')
solver.set_PML(sigma_max = 0 , depth=depth, direction ='y+', name='y+')
solver.set_PML(sigma_max = 0 , depth=depth, direction ='y-', name='y-')
solver.set_PML(sigma_max = 0 , depth=depth, direction ='z+', name='z+')
solver.set_PML(sigma_max = 0 , depth=depth, direction ='z-', name='z-')
solver.PML_list


ims = []
for i in tqdm.tqdm(range(1000)):
    solver.step()
    im = solver.E.z[:,:,32]
    if i % 50 == 0:
        plt.imshow(im.T)
        plt.show()
        plt.close()
    ims.append([im])

# f, ax = plt.subplots(figsize=(12, 6))
# def animate_func(i):
#     ax.clear()
#     plt.imshow(ims[i][0].reshape(64,64), vmax=1, vmin=-1)
#     return 

# plt.imshow(ims[100][0])
# plt.show()

# ani = animation.FuncAnimation(f, animate_func, frames=1000, interval=5)
# ani.save("test.mp4")

##################
# source 추가.
# material 추가. dielectric smoothing.
# simulation 할 방법 생각. 더 큰 용량과 빠른 계산 필요
# GPU로 활용할 방법 공부해야함. GPU 할당 방법.
