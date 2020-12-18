import numpy as np

class structure:
    def __init__(self, shape, eps, mu):
        self.epsr = np.ones(shape)
        self.mur = np.ones(shape)

    def change_space(self, space):
        space.set_epsr(space.epsr + self.epsr - 1)
        space.set_mur(space.epsr + self.mur - 1)

class rectangle(structure):
    def __init__(self, shape, center, width, depth, height, eps, mu):
        super(rectangle, self).__init__(shape, eps, mu)
        self.center = center
        self.width = width
        self.depth = depth
        self.height = height
        min_x, max_x =  center[0] - width //2 , center[0] + width //2
        min_y, max_y =  center[1] - depth //2 , center[1] + depth //2
        min_z, max_z =  center[1] - height //2 , center[1] + height //2
        
        self.epsr[min_x:max_x, min_y:max_y, min_z:max_z] = eps
        self.mur[min_x:max_x, min_y:max_y, min_z:max_z] = mu
        
    def __str__(self):
        return 'Rectangle({},{},{},{}'.format(self.center, self.width, self.depth, self.height)

