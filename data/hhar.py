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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

default_path = '/ssd/esm1g14/HHAR/'

class HeterogeneityHumanActivityRecognition(HAR_dataset):
	def __init__(self, datapath=default_path,dataset='training',transform=None,target_transform=None):
		super(HeterogeneityHumanActivityRecognition,self).__init__(datapath=datapath,dataset=dataset,transform=transform,target_transform=target_transform)
		self.files =['Phones_accelerometer.csv',
					 'Phones_gyroscope.csv',
					 'Watch_accelerometer.csv',
					 'Watch_gyroscope.csv']
		self.subject_ids = ['a','b','c','d','e','f','g','h','i']
		self.id2label = ['bike', 'sit', 'stand', 'walk', 'stairsup', 'stairsdown']
		self.label2id = {key : i for i,key in enumerate(self.id2label)}
	def get_subjects_data(self):
		subjects_data = {i : {} for i in self.subject_ids}
		for current_file in self.files:
			data = pd.read_csv(self.datapath + current_file)
			for subject in self.subject_ids:
				subjects_data[subject][current_file[0:-4]] = data[data['User'] == subject]
				for key,value in self.label2id.items():
					subjects_data[subject][current_file[0:-4]]['gt'][subjects_data[subject][current_file[0:-4]]['gt'] == key] = value
				subjects_data[subject][current_file[0:-4]]['gt'] = pd.to_numeric(subjects_data[subject][current_file[0:-4]]['gt'])
				# subjects_data[subject][current_file[0:-4]].set_index('Arrival_Time',inplace=True)
		self.plot_time_series(subjects_data,folder_path=self.datapath + 'plots/')
		return subjects_data
	def read_data(self,interpolation_f=None,cross_validation_index=-1):
		data = self.get_subjects_data()
		for subject,subject_data in data.items():
			for i,(sensor_name,sensor_data) in enumerate(subject_data.items()):
				if i == 0:
					merged_subject_data1 = sensor_data
					tmp = sensor_name
				elif i == 1:
					merged_subject_data1 = merged_subject_data1.merge(sensor_data,on='Arrival_Time',how='outer',suffixes=['_'+tmp,'_'+sensor_name])
				elif i == 2:
					merged_subject_data2 = sensor_data
					tmp = sensor_name
				elif i == 3:
					merged_subject_data2 = merged_subject_data2.merge(sensor_data,on='Arrival_Time',how='outer',suffixes=['_'+tmp,'_'+sensor_name])
			merged_subject_data = merged_subject_data1.merge(merged_subject_data2,on='Arrival_Time',how='outer')
			# cHANGELABELS
		
	def read_files(self, filelist, cols, label2id):
		data = []
		labels = []
		print('LOADING')
		print(filelist)
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
	def plot_time_series(self,data,folder_path):
		if not os.path.isdir(folder_path):
			os.mkdir(folder_path)
		for subject,subject_data in data.items():
			if not os.path.isdir(folder_path + subject):
				os.mkdir(folder_path+subject)
			for i,(sensor_name,sensor_data) in enumerate(subject_data.items()):
				for sensor_attribute in sensor_data:
					sensor_data_attribute = sensor_data[sensor_attribute]
					if not os.path.isdir(folder_path + subject + '/' + sensor_attribute):
						os.mkdir(folder_path + subject + '/' + sensor_attribute)
					plt.figure()
					plt.plot(sensor_data_attribute)
					plt.savefig(folder_path+subject+'/'+sensor_attribute+'/'+sensor_name+'.png')