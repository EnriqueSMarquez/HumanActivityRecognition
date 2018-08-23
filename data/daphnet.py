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

default_path = '/ssd/esm1g14/dataset_fog_release/'
default_frequency = 60
default_test_index = 1
default_validation_index = 8
class DaphNet(HAR_dataset):
	def __init__(self, subjects,datapath=default_path,transform=None,target_transform=None):
		super(DaphNet,self).__init__(datapath=datapath,transform=transform,target_transform=target_transform)
		self.files = [['S01R01.txt', 'S01R02.txt'],
					 ['S02R01.txt', 'S02R02.txt'],
					 ['S03R01.txt','S03R02.txt', 'S03R03.txt'],
					 ['S04R01.txt'],
					 ['S05R01.txt', 'S05R02.txt'],
					 ['S06R01.txt', 'S06R02.txt'], 
					 ['S07R01.txt', 'S07R02.txt'],
					 ['S08R01.txt'],
					 ['S09R01.txt'],
					 ['S10R01.txt']]
		self.files = [self.files[i] for i in subjects]
	def read_data(self,downsample=1):
		files = self.files.copy()
		files = [item for sublist in files for item in sublist]
		label_map =[(1, 'No freeze'),(2, 'freeze')]
		self.label2id = {str(x[0]): i for i, x in enumerate(label_map)}
		# print "label2id=",labelToId
		self.id2label = [x[1] for x in label_map]
		# print "id2label=",idToLabel
		cols = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
		# print "cols",cols
		self.read_files(files, cols, self.label2id)
		if downsample > 1 and downsample != 0:
			self.data['inputs'] = self.data['inputs'][::downsample,:]
			self.data['targets'] = self.data['targets'][::downsample]
	def read_files(self, filelist, cols, label2id):
		data = []
		labels = []
		for i, filename in enumerate(filelist):
			with open(self.datapath+'/dataset/%s' % filename, 'r') as f:
				#print "f",f
				reader = csv.reader(f, delimiter=' ')
				for line in reader:
					#print "line=",line
					elem = []
					#not including the non related activity
					if line[10] == "0":
						continue
					for ind in cols:
						#print "ind=",ind
						if ind == 10:
							# print "line[ind]",line[ind]
							if line[ind] == "0":
								continue
						elem.append(line[ind])
					if sum([x == 'NaN' for x in elem]) == 0:
						data.append([float(x) / 1000 for x in elem[:-1]])
						labels.append(self.label2id[elem[-1]])
		
		self.data = {'inputs': np.asarray(data), 'targets': np.asarray(labels, dtype=int)}
