import numpy as np
import csv
import sys
import os
import h5py
import simplejson as json
from tqdm import tqdm
import torchvision
default_data_path = '/ssd/esm1g14/OpportunityUCIDataset/'
import torch
from torch.utils.data.sampler import Sampler

class OpportunityDataset():
    def __init__(self, datapath=default_data_path,dataset='training',transform=None,target_transform=None,window_size=None):
        self.dataset = dataset
        self.datapath = datapath
        self._read_opportunity()
        self.transform = transform
        self.target_transform = target_transform
        if window_size:
            self.build_data(window_size=window_size)
    def save_data(self,file_path):
        f = h5py.File(file_path)
        for key in self.data:
            f.create_group(key)
            for field in self.data[key]:
                f[key].create_dataset(field, data=self.data[key][field])
        f.close()
        with open('opportunity.h5.classes.json', 'w') as f:
            f.write(json.dumps(self.id2label))

    def _read_opportunity(self):
        files = {
            'training': [
                'S1-ADL1.dat',                'S1-ADL3.dat', 'S1-ADL4.dat', 'S1-ADL5.dat', 'S1-Drill.dat',
                'S2-ADL1.dat', 'S2-ADL2.dat',                               'S2-ADL5.dat', 'S2-Drill.dat',
                'S3-ADL1.dat', 'S3-ADL2.dat',                               'S3-ADL5.dat', 'S3-Drill.dat',
                'S4-ADL1.dat', 'S4-ADL2.dat', 'S4-ADL3.dat', 'S4-ADL4.dat', 'S4-ADL5.dat', 'S4-Drill.dat'
            ],
            'validation': [

            ],
            'test': [
                'S2-ADL3.dat', 'S2-ADL4.dat',
                'S3-ADL3.dat', 'S3-ADL4.dat',
                'S1-ADL2.dat'
            ]
        }

        label_map = [
            (0,      'Other'),
            (406516, 'Open Door 1'),
            (406517, 'Open Door 2'),
            (404516, 'Close Door 1'),
            (404517, 'Close Door 2'),
            (406520, 'Open Fridge'),
            (404520, 'Close Fridge'),
            (406505, 'Open Dishwasher'),
            (404505, 'Close Dishwasher'),
            (406519, 'Open Drawer 1'),
            (404519, 'Close Drawer 1'),
            (406511, 'Open Drawer 2'),
            (404511, 'Close Drawer 2'),
            (406508, 'Open Drawer 3'),
            (404508, 'Close Drawer 3'),
            (408512, 'Clean Table'),
            (407521, 'Drink from Cup'),
            (405506, 'Toggle Switch')
        ]
        label2id = {str(x[0]): i for i, x in enumerate(label_map)}
        id2label = [x[1] for x in label_map]

        cols = [
            38, 39, 40, 41, 42, 43, 44, 45, 46, 51, 52, 53, 54, 55, 56, 57, 58, 59,
            64, 65, 66, 67, 68, 69, 70, 71, 72, 77, 78, 79, 80, 81, 82, 83, 84, 85,
            90, 91, 92, 93, 94, 95, 96, 97, 98, 103, 104, 105, 106, 107, 108, 109,
            110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
            125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 250]
        cols = [x-1 for x in cols] # labels for 18 activities (including other)

        self._read_opp_files(files[self.dataset], cols, label2id)
        self.id2label = id2label
        self.label2id = label2id
    def _read_opp_files(self, filelist, cols, label2id):
        data = []
        labels = []
        print('LOADING %s DATA')%(self.dataset)
        filelist = tqdm(filelist)
        for i, filename in enumerate(filelist):
            nancnt = 0
            # print('reading file %d of %d' % (i+1, len(filelist)))
            with open(self.datapath.rstrip('/') + '/dataset/%s' % filename, 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                for line in reader:
                    elem = []
                    for ind in cols:
                        elem.append(line[ind])
                    # we can skip lines that contain NaNs, as they occur in blocks at the start
                    # and end of the recordings.
                    if sum([x == 'NaN' for x in elem]) == 0:
                        data.append([float(x) / 1000 for x in elem[:-1]])
                        labels.append(label2id[elem[-1]])
        self.data = {'inputs': np.asarray(data), 'targets': np.asarray(labels, dtype=int)}

    def reset_data_to_time_series(self):
        self._read_opportunity()

    def build_data(self,window_size=30,overlap=0.5): #30Hz
        inputs = []
        targets = []
        counter = 0
        max_len = np.floor(len(self.data['inputs'])/(overlap*window_size)).astype(int)
        for i in range(max_len-1):
            inputs += [self.data['inputs'][counter:window_size+counter,:]]
            targets += [self.data['targets'][int(window_size+counter)]]
            counter += int(len(inputs[-1])*overlap)
        self.data = {'inputs': np.asarray(inputs), 'targets': np.asarray(targets, dtype=int)}
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
class BalancedBatchSampler(Sampler):
    def __init__(self, labels):
        self.labels = labels
        self.nb_classes = int(max(labels)+1)
        # self.classes_count =[]
        # class_count = np.zeros((self.nb_classes,)).astype(float)
        # for i in range(self.nb_classes()):
        #     class_count += [len(np.where(self.labels == i)[0])]
        # self.classes_count = class_count
        self.build_classes_iterators()
    def __iter__(self):
        return iter(self.merged_iterator())
    def __len__(self):
        return len(self.labels)
    def build_classes_iterators(self):
        iterators = []
        classes_indexes = []
        for i in range(self.nb_classes):
            classes_indexes += [np.where(self.labels == i)[0]]
            permutation = np.random.permutation(len(classes_indexes[-1]))
            iterators += [iter(classes_indexes[-1][permutation])]
        self.classes_indexes = classes_indexes
        self.classes_iterators = iterators
    def merged_iterator(self):
        counter = 0
        while counter < len(self.labels):
            next_index = next(self.classes_iterators[0],None)
            if next_index != None:
                yield next_index
                counter += 1
            else:
                self.buld_class_iterator(0)
                next_index = next(self.classes_iterators[0])
                yield next_index
                counter += 1
            for j,iterator in enumerate(self.classes_iterators):
                next_index = next(iterator,None)
                if next_index != None:
                    yield next_index
                    counter += 1
                else:
                    self.buld_class_iterator(j)
                    next_index = next(self.classes_iterators[j])
                    yield next_index
                    counter += 1
    def buld_class_iterator(self,label):
        permutation = np.random.permutation(len(self.classes_indexes[label]))
        self.classes_iterators[label] = iter(self.classes_indexes[label][permutation])
