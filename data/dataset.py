import numpy as np
import torch
from scipy import stats

class HAR_dataset():
    def __init__(self, datapath,dataset,transform,target_transform):
        self.dataset = dataset
        self.datapath = datapath
        self.transform = transform
        self.target_transform = target_transform
    def build_data(self,window_size=30,step=10,cross_validation_index=-1):
        self.read_data(cross_validation_index)
        inputs = []
        targets = []
        counter = 0
        while counter+window_size < len(self.data['inputs']):
            inputs += [self.data['inputs'][counter:window_size+counter,:]]
            targets += [stats.mode(self.data['targets'][counter:window_size+counter],axis=None).mode[0]]
            # targets += [self.data['targets'][window_size+counter]]
            counter += step
        self.data = {'inputs': np.asarray(inputs).transpose(0,2,1), 'targets': np.asarray(targets, dtype=int)}
    def __getitem__(self,index):
        x,y = self.data['inputs'][index],self.data['targets'][index].reshape(1,-1)
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x,y
    def input_shape(self):
        return self.data['inputs'].shape[1::]
    def nb_classes(self):
        return max(self.data['targets'])+1
    def __len__(self):
        return len(self.data['inputs'])
    def get_data_weights(self):
        class_count = np.zeros((self.nb_classes(),)).astype(float)
        for i in range(self.nb_classes()):
            class_count[i] = len(np.where(self.data['targets'] == i)[0])
        weights = (1 / torch.from_numpy(class_count).type(torch.DoubleTensor))
        return weights
