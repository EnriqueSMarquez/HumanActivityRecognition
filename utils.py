import numpy as np
import torch

def get_dataset_info(dataset):
	if dataset == 'opp':
	    from data import opportunity
	    dataset = opportunity.OpportunityDataset
	    val_id = [[0,1]]
	    test_id = [[1,3],[1,4],[2,3],[2,4]]
	    train_id = [[0,0],[0,2],[0,3],[0,4],[0,5],
	                [1,0],[1,1],[1,2],[1,5],
	                [2,0],[2,1],[2,2],[2,5],
	                [3,0],[3,1],[3,2],[3,3],[3,4],[3,5]]
	    macro = False
	elif dataset == 'pamap2':
	    from data import pamap2
	    dataset = pamap2.PAMAP2_Dataset
	    val_id = [4]
	    test_id = [5]
	    train_id = [0,1,2,3,6,7]
	    macro = True
	elif dataset == 'daph':
	    from data import daphnet
	    dataset = daphnet.DaphNet
	    val_id = [8]
	    test_id = [1]
	    train_id = [0,2,3,4,5,6,7,9]
	    macro = True
	return {'dataset' : dataset, 'train_subjects' : train_id, 'test_subjects' : test_id, 'validation_subjects' : val_id, 'mean_metric' : macro}

def quick_transforms(type=torch.FloatTensor):
    def f(x):
        return torch.from_numpy(x).type(type)
    return f

def get_datasets(dataset,validation=True,window_size=90,step=45,downsample=1):
	dataset_info = get_dataset_info(dataset)
	input_transform = quick_transforms()
	target_transform = quick_transforms(type=torch.LongTensor)
	if validation:
		train = dataset_info['dataset'](subjects=dataset_info['train_subjects'],transform=input_transform,target_transform=target_transform)
		train.build_data(window_size=window_size,step=step,downsample=downsample)
		val = dataset_info['dataset'](subjects=dataset_info['validation_subjects'],transform=input_transform,target_transform=target_transform)
		val.build_data(window_size=window_size,step=step,downsample=downsample)
		test = dataset_info['dataset'](subjects=dataset_info['test_subjects'],transform=input_transform,target_transform=target_transform)
		test.build_data(window_size=window_size,step=step,downsample=downsample)
		y = {'training_set' : train, 'validation_set' : val, 'testing_set' : test}
	else:
		train = dataset_info['dataset'](subjects=dataset_info['train_subjects'] + dataset_info['validation_subjects'],transform=input_transform,target_transform=target_transform)
		train.build_data(window_size=window_size,step=step,downsample=downsample)
		test = dataset_info['dataset'](subjects=dataset_info['test_subjects'],transform=input_transform,target_transform=target_transform)
		test.build_data(window_size=window_size,step=step,downsample=downsample)
		y = {'training_set' : train, 'testing_set' : test}
	y['mean_metric'] = dataset_info['mean_metric']
	return y