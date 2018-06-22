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

class PAMAP2_Dataset(HAR_dataset):
    def __init__(self, datapath=default_path,dataset='training',transform=None,target_transform=None):
        super(PAMAP2_Dataset,self).__init__(datapath=datapath,dataset=dataset,transform=transform,target_transform=target_transform)
        self.files = ['subject101.dat', 'subject102.dat','subject103.dat','subject104.dat', 'subject107.dat', 'subject108.dat', 'subject109.dat','subject106.dat']
    def read_data(self,cross_validation_index=-1):
        files = self.files.copy()
        if self.dataset == 'training':
            files.pop(cross_validation_index)
        else:
            files = [files.pop(cross_validation_index)]
        label_map = [
            # (0, 'other'),
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
        # print "label2id=",labelToId
        id2Label = [x[1] for x in label_map]
        # print "id2label=",idToLabel
        cols = [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53
               ]
        self.data = self.read_files(files, cols, label2id)
        self.id2Label = id2Label
        self.label2id = label2id

    def read_files(self, filelist, cols, label2id):
        data = []
        labels = []
        for i, filename in enumerate(filelist):
            with open(self.datapath+'Protocol/%s' % filename, 'r') as f:
                #print "f",f
                reader = csv.reader(f, delimiter=' ')
                for line in reader:
                    #print "line=",line
                    elem = []
                    #not including the non related activity
                    if line[1] == "0":
                        continue
                    for ind in cols:
                        elem.append(line[ind])
                    if sum([x == 'NaN' for x in elem]) == 0:
                        data.append([float(x) / 1000 for x in elem[:-1]])
                        labels.append(label2id[elem[0]])

        return {'inputs': np.asarray(data), 'targets': np.asarray(labels, dtype=int)}
