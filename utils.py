import numpy as np
def make_grid(x,y):
    x,y = np.meshgrid(x,y)
    return x.flatten(),y.flatten()
