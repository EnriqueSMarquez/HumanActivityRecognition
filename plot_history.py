import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import _pickle as pickle
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os

save_folder = './Run1_pamap2/'
print(save_folder)

history = pickle.load(open(save_folder+'history.txt','rb'))[0]

for key,value in history.items():
    plt.figure()
    plt.plot(np.asarray(value).reshape(-1))
    plt.xlabel('EPOCH')
    plt.ylabel(key)
    plt.savefig(save_folder + key +'.png', bbox_inches="tight")
