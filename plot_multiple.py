import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import _pickle as pickle
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os

save_folder = './Run1_pamp2_Tuning/'
# save_folder = './Run2_MNIST/'
print(save_folder)
#
# nbclasses_linspace = np.linspace(2,10,9)
# hidden_features_linspace = np.linspace(2,512,100)
# nbclasses_grid,hidden_features_grid = np.meshgrid(nbclasses_linspace,hidden_features_linspace)
# grid = np.vstack([nbclasses_grid.flatten(),hidden_features_grid.flatten()]).T.astype(int)

# kernel_size_linspace = np.arange(3,52,2).reshape(-1).astype(int).tolist()

samplings_steps = 3

train_loss = []
train_acc = []
test_loss = []
test_acc = []
f1 = []
info_dict = {}
for step in range(samplings_steps):
    current_folder = save_folder+str(step)+'/'
    runs = os.listdir(current_folder)
    info_dict[str(step)] = {}
    for run in runs:
        tmp = run.split('_')
        if os.path.isfile(current_folder+run+'/history.txt'):
            history = pickle.load(open(current_folder+run+'/history.txt','rb'),encoding='latin1')
            if str(tmp[1]) not in info_dict[str(step)].keys():
                info_dict[str(step)][str(tmp[1])] = []
            info_dict[str(step)][str(tmp[1])] += [[int(tmp[0]),np.max(history[0]['test_f1'])]]

    # train_loss += [np.min(current_history[0]['train_loss'])]
    # train_acc += [np.max(current_history[0]['train_acc'])]
    # test_loss += [np.min(current_history[0]['test_loss'])]
    # test_acc += [np.max(current_history[0]['test_acc'])]
    # f1 += [np.max(current_history[0]['test_f1'])]

# # ax2 = fig2.gca(projection='3d')
# plot_array = [train_loss,train_acc,test_loss,test_acc,f1]
# strings = ['train_loss','train_acc','test_loss','test_acc','f1']
# surf2 = ax2.plot_surface(nbclasses_grid,hidden_features_grid,grid3d[:,2].reshape(nbclasses_grid.shape),cmap=cm.coolwarm, linewidth=0.2)
# plt.savefig(save_folder+'nbclasses_vs_hiddenfeatures_vs_acc.png')

for step,values in info_dict.items():
    plt.figure()
    for kernel_size,depth_accs in values.items():
        curve = np.sort(np.asarray(depth_accs),axis=0)
        plt.plot(curve[:,0],curve[:,1],label=kernel_size)
        plt.xlabel('DEPTH')
        plt.ylabel('F1')
        plt.legend()
    plt.savefig(save_folder + step +'samplings_f1.png', bbox_inches="tight")

# plt.show()
#
# # ax2.set_zlim([0.75,0.99])
# ax2.set_xlabel('# CLASSES')
# ax2.set_ylabel('# HIDDEN UNITS')
# ax2.set_zlabel('TEST ACC')
# plt.show()
