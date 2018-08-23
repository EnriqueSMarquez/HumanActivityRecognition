import numpy as np
import csv
import sys
import os
import h5py
import pandas as pd
import simplejson as json
import copy
from scipy import stats
from .dataset import HAR_dataset

default_path = '/ssd/esm1g14/PAMAP2/'
default_frequency = 100
default_test_index = 5
heart_rate_index = [2]
acc16_indexes = [4,5,6,21,22,23,38,39,40]

class PAMAP2_Dataset(HAR_dataset):
    def __init__(self, subjects,datapath=default_path,transform=None,target_transform=None):
        super(PAMAP2_Dataset,self).__init__(datapath=datapath,transform=transform,target_transform=target_transform)
        self.files = ['subject101.dat', 
                      'subject102.dat',
                      'subject103.dat',
                      'subject104.dat',
                      'subject105.dat',
                      'subject106.dat',
                      'subject107.dat',
                      'subject108.dat']
        self.files = [self.files[i] for i in subjects]
    def read_data(self,downsample=1,chop_non_related_activities=True,trim_activities=True):
        files = self.files.copy()
        label_map = [
            (0, 'other'),
            (1, 'lying'),
            (2, 'sitting'),
            (3, 'standing'),
            (4, 'walking'),
            (5, 'running'),
            (6, 'cycling'),
            (7, 'Nordic walking'),
            # (9, 'watching TV'),
            # (10, 'computer work'),
            # (11, 'car driving'),
            (12, 'ascending stairs'),
            (13, 'descending stairs'),
            (16, 'vacuum cleaning'),
            (17, 'ironing'),
            # (18, 'folding laundry'),
            # (19, 'house cleaning'),
            # (20, 'playing soccer'),
            (24, 'rope jumping')
        ]

        label2id = {str(x[0]): i for i, x in enumerate(label_map)}
        id2label = [x[1] for x in label_map]
        cols = [1] + heart_rate_index + acc16_indexes
        self.data = self.read_files(files, cols, label2id)
        self.data['targets'] = np.asarray([int(label2id[str(i)]) for i in self.data['targets'].tolist()]).astype(int)
        if chop_non_related_activities:
            tmp = {'inputs' : [],'targets': []}
            for x,y in zip(self.data['inputs'],self.data['targets']):
                if int(y) != 0:
                    tmp['inputs'] += [x]
                    tmp['targets'] += [y]
            tmp['inputs'] = np.asarray(tmp['inputs'])
            tmp['targets'] = np.asarray(tmp['targets'])
            tmp['targets'] -= 1
        self.data = tmp
        if trim_activities:
            self.trim_activities()
        if downsample > 1:
            self.data['inputs'] = self.data['inputs'][::downsample,:]
            self.data['targets'] = self.data['targets'][::downsample]
        self.id2label = id2label
        self.label2id = label2id

    def read_files(self, filelist, cols, label2id,interpolate_heart_rate=True):
        data = []
        labels = []
        for i, filename in enumerate(filelist):
            current_data = pd.read_csv(self.datapath+'Protocol/%s' % filename,delimiter=' ')
            if interpolate_heart_rate:
                current_data.iloc[:,heart_rate_index] = current_data.iloc[:,heart_rate_index].interpolate()

            current_data = current_data.dropna()
            current_data = current_data.iloc[:,cols]
            data.append(current_data.iloc[:,1::])
            labels.append(current_data.iloc[:,0])
        return {'inputs': np.concatenate([np.asarray(i) for i in data]).astype(float), 'targets': np.concatenate([np.asarray(j) for j in labels]).astype(int)}
    def plot_data(self,saving_folder):
        if not os.path.isdir(saving_folder):
            os.mkdir(saving_folder)
        for i,time_series in enumerate(self.data['inputs'].T.tolist()):
            plt.figure()
            plt.plot(time_series)
            plt.savefig(saving_folder+'time_series' + str(i)+'.png')

    def trim_activities(self):
        inputs = []
        targets = []
        trimmed_inputs = []
        trimmed_targets =[]
        for x,y in zip(self.data['inputs'],self.data['targets']):
            if len(targets) == 0:
                targets += [y]
                inputs += [x]
            else:
                if targets[-1] != y:
                    trimmed_inputs += [inputs[1000:-1000]]
                    trimmed_targets += [targets[1000:-1000]]
                    targets = []
                    inputs = []
                else:
                    targets += [y]
                    inputs += [x]
        self.data['inputs'] = np.concatenate(trimmed_inputs)
        self.data['targets'] = np.concatenate(trimmed_targets)

