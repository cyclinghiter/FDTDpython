import sys
import numpy as np 
import matplotlib.pyplot as plt
from scipy import constants

class VectorField:
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
    def __init__(self, shape, dx, dy, dz, dt):
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.dt = dt
        self.shape = shape
        self.eps0 = constants.epsilon_0 
        self.mu0 = constants.mu_0
        self.epsr = np.ones(self.shape)
        self.mur = np.ones(self.shape)
        self.sigma = np.zeros(self.shape)
        self.sigma_m = np.zeros(self.shape)


class PMLParameter(Space):
    def __init__(self, depth, dx, grading_order = 3):
        super(PMLParameter, self).__init__(depth, dx, dx, dx, dt)
        self.grading_order = grading_order
        self.rc0 =  1.e-16
        self.width = self.dx * depth
        self.imp0 = np.sqrt(self.mu0 / self.eps0)
        self.sigmamax = - (self.grading_order + 1) * np.log(self.rc0) / (2 * self.imp0 * self.width)
        self.kappamax = 1
        self.alphamax = 0.02
        
        self.sigma = np.zeros(depth)
        self.kappa = np.zeros(depth)
        self.alpha = np.zeros(depth)
        
        for i in range(depth):
            loc = (i+1) * self.dx / self.width
            self.sigma[i] = self.sigmamax * (loc ** self.grading_order)
            self.kappa[i] = 1 + ((self.kappamax-1) * ((loc) **self.grading_order))
            self.alpha[i] = self.alphamax * ((1-loc) **self.grading_order)
            
            self.alpha[i] = self.alphamax
        self.b = np.exp(-(self.sigma/self.kappa + self.alpha) * self.dt / self.eps0)
        self.c = self.sigma / (self.sigma*self.kappa + self.alpha*self.kappa**2) * (self.b -1)
                
        
class EMSpace(Space):
    def __init__(self, shape, dx, dy, dz, dt, init_time = 0):
        super(EMSpace, self).__init__(shape, dx, dy, dz, dt)  
        self.dt = dt
        self.t = init_time
        self.E = VectorField(self.shape)
        self.H = VectorField(self.shape)
        self.update_coefficient()
        self.PMLdirection = ''
    
    def update_coefficient(self):        
        self.Ca = ((1 - (self.sigma * self.dt) 
                        / (2*self.eps0 * self.epsr)) /
                    (1 + (self.sigma * self.dt)
                        / (2*self.eps0 * self.epsr)))
        
        self.Cb = ((self.dt 
                / ((self.eps0 * self.epsr)))
                    / (1 + (self.sigma * self.dt)
                        / (2*self.eps0 * self.epsr)))
        
        self.Da = ((1 - (self.sigma_m * self.dt) 
                        / (2*self.mu0 * self.mur)) /
                    (1 + (self.sigma_m * self.dt)
                        / (2*self.mu0 * self.mur)))
        
        self.Db = ((self.dt 
                / ((self.mu0 * self.mur)))
                    / (1 + (self.sigma_m * self.dt)
                        / (2*self.mu0 * self.mur)))
        
    def _get_updated_coefficient(func):
        def wrapper(self):
            func(self)
            self.update_coefficient()
        return wrapper 
    
    @_get_updated_coefficient
    def set_epsr(self):
        pass
    
    @_get_updated_coefficient    
    def set_mur(self):
        pass
    
    def apply_PEC(self):
        self.E.x[:,0,:] = 0
        self.E.x[:,-1,:] = 0
        self.E.x[:,:,0] = 0
        self.E.x[:,:,-1] = 0
        
        self.E.y[0,:,:] = 0
        self.E.y[-1,:,:] = 0
        self.E.y[:,:,0] = 0
        self.E.y[:,:,-1] = 0
        
        self.E.z[0,:,:] = 0        
        self.E.z[-1,:,:] = 0
        self.E.z[:,0,:] = 0
        self.E.z[:,-1,:] = 0

    def set_PML(self, depth, direction):
        
        self.calculate_PML_region = 1
        self.PMLdirection = direction
        self.depth = depth
        self.PML = PMLParameter(depth, dx = self.dx)
        
        if 'x' in self.PMLdirection:
            self.psi_Eyx0 = np.zeros((depth, self.shape[1], self.shape[2]))
            self.psi_Ezx0 = np.zeros((depth, self.shape[1], self.shape[2]))
            self.psi_Eyx1 = np.zeros((depth, self.shape[1], self.shape[2]))
            self.psi_Ezx1 = np.zeros((depth, self.shape[1], self.shape[2]))
            self.psi_Hyx0 = np.zeros((depth, self.shape[1], self.shape[2]))
            self.psi_Hzx0 = np.zeros((depth, self.shape[1], self.shape[2]))
            self.psi_Hyx1 = np.zeros((depth, self.shape[1], self.shape[2]))
            self.psi_Hzx1 = np.zeros((depth, self.shape[1], self.shape[2]))
            self.bx = np.zeros_like(self.psi_Hyx0)
            self.cx = np.zeros_like(self.psi_Hyx0)
        
            for i in range(self.depth):
                self.bx[i,:,:] = self.PML.b[i]
                self.cx[i,:,:] = self.PML.c[i]
                
        if 'y' in self.PMLdirection:
            self.psi_Ezy0 = np.zeros((self.shape[0], depth, self.shape[2]))
            self.psi_Exy0 = np.zeros((self.shape[0], depth, self.shape[2]))
            self.psi_Ezy1 = np.zeros((self.shape[0], depth, self.shape[2]))
            self.psi_Exy1 = np.zeros((self.shape[0], depth, self.shape[2]))
            self.psi_Hzy0 = np.zeros((self.shape[0], depth, self.shape[2]))
            self.psi_Hxy0 = np.zeros((self.shape[0], depth, self.shape[2]))
            self.psi_Hzy1 = np.zeros((self.shape[0], depth, self.shape[2]))
            self.psi_Hxy1 = np.zeros((self.shape[0], depth, self.shape[2]))
            self.by = np.zeros_like(self.psi_Hxy0)        
            self.cy = np.zeros_like(self.psi_Hxy0)    
                
            for i in range(self.depth):
                self.by[:,i,:] = self.PML.b[i]
                self.cy[:,i,:] = self.PML.c[i]
                

        if 'z' in self.PMLdirection:
            self.psi_Exz0 = np.zeros((self.shape[0], self.shape[1], depth))
            self.psi_Eyz0 = np.zeros((self.shape[0], self.shape[1], depth))
            self.psi_Exz1 = np.zeros((self.shape[0], self.shape[1], depth))
            self.psi_Eyz1 = np.zeros((self.shape[0], self.shape[1], depth))
            self.psi_Hxz0 = np.zeros((self.shape[0], self.shape[1], depth))
            self.psi_Hyz0 = np.zeros((self.shape[0], self.shape[1], depth))
            self.psi_Hxz1 = np.zeros((self.shape[0], self.shape[1], depth))
            self.psi_Hyz1 = np.zeros((self.shape[0], self.shape[1], depth))
            self.bz = np.zeros_like(self.psi_Hyz0)    
            self.cz = np.zeros_like(self.psi_Hyz0)    
            
            for i in range(self.depth):
                self.bz[:,:,i] = self.PML.b[i]
                self.cz[:,:,i] = self.PML.c[i]
                                        
    def updateE(self):
        self.E.x[:, 1:, 1:] = ((self.Ca * self.E.x)[:, 1:, 1:]
                            + (self.Cb[:, 1:, 1:] * 
                            ((self.H.z[:, 1:, 1:] - self.H.z[:, :-1, 1:])/self.dy 
                            - (self.H.y[:, 1:, 1:] - self.H.y[:, 1:, :-1])/self.dz)))
        self.E.y[1:, :, 1:] = ((self.Ca * self.E.y)[1:, :, 1:] 
                            + (self.Cb[1:, :, 1:] *
                            ((self.H.x[1:, :, 1:] - self.H.x[1:, :, :-1])/self.dz
                            - (self.H.z[1:, :, 1:] - self.H.z[:-1, :, 1:])/self.dx)))    
        self.E.z[1:, 1:, :] = ((self.Ca * self.E.z)[1:, 1:, :] 
                            + (self.Cb[1:, 1:, :] *
                            ((self.H.y[1:, 1:, :] - self.H.y[:-1, 1:, :])/self.dx 
                            - (self.H.x[1:, 1:, :] - self.H.x[1:, :-1, :])/self.dy)))
        
        if 'x' in self.PMLdirection:
            self.psi_Ezx0[1:,:,:] = (self.bx[::-1,:,:][1:,:,:] * self.psi_Ezx0[1:,:,:] + self.cx[::-1,:,:][1:,:,:] * (self.H.y[1:1+self.depth,:,:] - self.H.y[:self.depth,:,:])[:-1,:,:] / self.dx)
            self.psi_Eyx0[1:,:,:] = (self.bx[::-1,:,:][1:,:,:] * self.psi_Eyx0[1:,:,:] + self.cx[::-1,:,:][1:,:,:] * (self.H.z[1:1+self.depth,:,:] - self.H.z[:self.depth,:,:])[:-1,:,:] / self.dx)
            self.E.z[:self.depth,:, :] += self.Cb[:self.depth,:,:] * self.psi_Ezx0
            self.E.y[:self.depth,:, :] -= self.Cb[:self.depth,:,:] * self.psi_Eyx0
            self.psi_Ezx1 = (self.bx * self.psi_Ezx1 + self.cx * (self.H.y[-self.depth:,:,:] - self.H.y[-self.depth-1:-1,:,:]) / self.dx)
            self.psi_Eyx1 = (self.bx * self.psi_Eyx1 + self.cx * (self.H.z[-self.depth:,:,:] - self.H.z[-self.depth-1:-1,:,:]) / self.dx)
            self.E.z[-self.depth:,:, :] += self.Cb[-self.depth:,:,:] * self.psi_Ezx1
            self.E.y[-self.depth:,:, :] -= self.Cb[-self.depth:,:,:] * self.psi_Eyx1
            
        if 'y' in self.PMLdirection:
            self.psi_Exy0[:,1:,:] = (self.by[:,::-1,:][:,1:,:] * self.psi_Exy0[:,1:,:] + self.cy[:,::-1,:][:,1:,:] * (self.H.z[:,1:1+self.depth,:] - self.H.z[:,:self.depth,:])[:,:-1,:] / self.dy)
            self.psi_Ezy0[:,1:,:] = (self.by[:,::-1,:][:,1:,:] * self.psi_Ezy0[:,1:,:] + self.cy[:,::-1,:][:,1:,:] * (self.H.x[:,1:1+self.depth,:] - self.H.x[:,:self.depth,:])[:,:-1,:] / self.dy)
            self.E.x[:,:self.depth, :] += self.Cb[:,:self.depth,:] * self.psi_Exy0
            self.E.z[:,:self.depth, :] -= self.Cb[:,:self.depth,:] * self.psi_Ezy0
            self.psi_Exy1 = (self.by * self.psi_Exy1 + self.cy * (self.H.z[:,-self.depth:,:] - self.H.z[:,-self.depth-1:-1,:]) / self.dy)
            self.psi_Ezy1 = (self.by * self.psi_Ezy1 + self.cy * (self.H.x[:,-self.depth:,:] - self.H.x[:,-self.depth-1:-1,:]) / self.dy)
            self.E.x[:,-self.depth:, :] += self.Cb[:,-self.depth:,:] * self.psi_Exy1
            self.E.z[:,-self.depth:, :] -= self.Cb[:,-self.depth:,:] * self.psi_Ezy1
            
        if 'z' in self.PMLdirection:
            self.psi_Eyz0[:,:,1:] = (self.bz[:,:,::-1][:,:,1:] * self.psi_Eyz0[:,:,1:] + self.cz[:,:,::-1][:,:,1:] * (self.H.x[:,:,1:1+self.depth] - self.H.x[:,:,:self.depth])[:,:,:-1] / self.dz)
            self.psi_Exz0[:,:,1:] = (self.bz[:,:,::-1][:,:,1:] * self.psi_Exz0[:,:,1:] + self.cz[:,:,::-1][:,:,1:] * (self.H.y[:,:,1:1+self.depth] - self.H.y[:,:,:self.depth])[:,:,:-1] / self.dz)
            self.E.y[:,:,:self.depth] += self.Cb[:,:,:self.depth] * self.psi_Eyz0
            self.E.x[:,:,:self.depth] -= self.Cb[:,:,:self.depth] * self.psi_Exz0
            self.psi_Eyz1 = (self.bz * self.psi_Eyz1 + self.cz * (self.H.x[:,:,-self.depth:] - self.H.x[:,:,-self.depth-1:-1]) / self.dz)
            self.psi_Exz1 = (self.bz * self.psi_Exz1 + self.cz * (self.H.y[:,:,-self.depth:] - self.H.y[:,:,-self.depth-1:-1]) / self.dz)
            self.E.y[:,:,-self.depth:] += self.Cb[:,:,-self.depth:] * self.psi_Eyz1
            self.E.x[:,:,-self.depth:] -= self.Cb[:,:,-self.depth:] * self.psi_Exz1


    def updateH(self):
        self.H.x[:-1, :-1, :-1] = ((self.Da * self.H.x)[:-1, :-1, :-1] 
                             - (self.Db[:-1, :-1, :-1] * 
                             ((self.E.z[:-1, 1:, :-1] - self.E.z[:-1, :-1, :-1])/self.dy
                             - (self.E.y[:-1, :-1, 1:] - self.E.y[:-1, :-1, :-1])/self.dz)))
        self.H.y[:-1, :-1, :-1] =  ((self.Da * self.H.y)[:-1, :-1, :-1]
                             - (self.Db[:-1, :-1, :-1] * 
                             ((self.E.x[:-1, :-1, 1:] - self.E.x[:-1, :-1, :-1])/self.dz 
                             - (self.E.z[1:, :-1, :-1] - self.E.z[:-1, :-1, :-1])/self.dx)))    
        self.H.z[:-1, :-1, :-1] =  ((self.Da * self.H.z)[:-1, :-1, :-1] 
                             - (self.Db[:-1, :-1, :-1] * 
                             ((self.E.y[1:, :-1, :-1] - self.E.y[:-1, :-1, :-1])/self.dx
                             - (self.E.x[:-1, 1:, :-1] - self.E.x[:-1, :-1, :-1])/self.dy)))  

        if 'x' in self.PMLdirection:
            self.psi_Hzx0 = self.bx[::-1,:,:] * self.psi_Hzx0 + self.cx[::-1,:,:] * (self.E.y[1:self.depth+1,:,:] - self.E.y[:self.depth,:,:]) / self.dx
            self.psi_Hyx0 = self.bx[::-1,:,:] * self.psi_Hyx0 + self.cx[::-1,:,:] * (self.E.z[1:self.depth+1,:,:] - self.E.z[:self.depth,:,:]) / self.dx
            self.H.z[:self.depth,:, :] -= self.Db[:self.depth,:,:] * self.psi_Hzx0
            self.H.y[:self.depth,:, :] += self.Db[:self.depth,:,:] * self.psi_Hyx0
            self.psi_Hzx1[:-1,:,:] = (self.bx[:-1,:,:] * self.psi_Hzx1[:-1,:,:] + self.cx[:-1,:,:] * (self.E.y[-self.depth:,:,:] - self.E.y[-self.depth-1:-1,:,:])[1:,:,:] / self.dx)
            self.psi_Hyx1[:-1,:,:] = (self.bx[:-1,:,:] * self.psi_Hyx1[:-1,:,:] + self.cx[:-1,:,:] * (self.E.z[-self.depth:,:,:] - self.E.z[-self.depth-1:-1,:,:])[1:,:,:] / self.dx)
            self.H.z[-self.depth:,:, :] -= self.Db[-self.depth:,:,:] * self.psi_Hzx1
            self.H.y[-self.depth:,:, :] += self.Db[-self.depth:,:,:] * self.psi_Hyx1
            
        if 'y' in self.PMLdirection:
            self.psi_Hxy0 = self.by[:,::-1,:] * self.psi_Hxy0 + self.cy[:,::-1,:] * (self.E.z[:,1:self.depth+1,:] - self.E.z[:,:self.depth,:]) / self.dy
            self.psi_Hzy0 = self.by[:,::-1,:] * self.psi_Hzy0 + self.cy[:,::-1,:] * (self.E.x[:,1:self.depth+1,:] - self.E.x[:,:self.depth,:]) / self.dy
            self.H.x[:,:self.depth, :] -= self.Db[:,:self.depth,:] * self.psi_Hxy0
            self.H.z[:,:self.depth, :] += self.Db[:,:self.depth,:] * self.psi_Hzy0
            self.psi_Hxy1[:,:-1,:] = (self.by[:,:-1,:] * self.psi_Hxy1[:,:-1,:] + self.cy[:,:-1,:] * (self.E.z[:,-self.depth:,:] - self.E.z[:,-self.depth-1:-1,:])[:,1:,:] / self.dy)    
            self.psi_Hzy1[:,:-1,:] = (self.by[:,:-1,:] * self.psi_Hzy1[:,:-1,:] + self.cy[:,:-1,:] * (self.E.x[:,-self.depth:,:] - self.E.x[:,-self.depth-1:-1,:])[:,1:,:] / self.dy)
            self.H.x[:,-self.depth:, :] -= self.Db[:,-self.depth:,:] * self.psi_Hxy1
            self.H.z[:,-self.depth:, :] += self.Db[:,-self.depth:,:] * self.psi_Hzy1
            
        if 'z' in self.PMLdirection:
            self.psi_Hxz0 = self.bz[:,:,::-1] * self.psi_Hxz0 + self.cz[:,:,::-1] * (self.E.y[:,:,1:self.depth+1] - self.E.y[:,:,:self.depth]) / self.dz
            self.psi_Hyz0 = self.bz[:,:,::-1] * self.psi_Hyz0 + self.cz[:,:,::-1] * (self.E.x[:,:,1:self.depth+1] - self.E.x[:,:,:self.depth]) / self.dz
            self.H.y[:,:,:self.depth] -= self.Db[:,:,:self.depth] * self.psi_Hyz0
            self.H.x[:,:,:self.depth] += self.Db[:,:,:self.depth] * self.psi_Hxz0
            self.psi_Hxz1[:,:,:-1] = (self.bz[:,:,:-1] * self.psi_Hxz1[:,:,:-1] + self.cz[:,:,:-1] * (self.E.y[:,:,-self.depth:] - self.E.y[:,:,-self.depth-1:-1])[:,:,1:] / self.dz)
            self.psi_Hyz1[:,:,:-1] = (self.bz[:,:,:-1] * self.psi_Hyz1[:,:,:-1] + self.cz[:,:,:-1] * (self.E.x[:,:,-self.depth:] - self.E.x[:,:,-self.depth-1:-1])[:,:,1:] / self.dz)
            self.H.y[:,:,-self.depth:] -= self.Db[:,:,-self.depth:] * self.psi_Hyz1
            self.H.x[:,:,-self.depth:] += self.Db[:,:,-self.depth:] * self.psi_Hxz1
    
    def put_source(self):
        self.E.y[:,:,:] = 0
        self.E.x[:,:,:] = 0
        self.E.z[64,64,12] = np.sin(2*np.pi*2*(10**12)*self.t)
    
    def step(self):
        self.updateH()
        self.updateE()
        self.t += self.dt
        self.put_source()
        self.apply_PEC()



if __name__ == '__main__':
    shape = (128,128,24)    
    um = 1e-6
    dx = 10 * um
    dy = 10 * um
    dz = 10 * um
    dt = 1/4 * dx / constants.c

    space = EMSpace(shape, dx, dy, dz, dt)
    space.set_PML(depth=10, direction='xyz')

    if __name__ == '__main__':
        import tqdm
        for i in range(1000):
            space.step()
            if i % 10 == 0:
                plt.imshow(space.E.z[:,:,12])
                plt.show()
        