import numpy as np
import torch
from torch.utils.data.sampler import Sampler

class BalancedBatchSampler(Sampler):
    def __init__(self, labels):
        self.labels = labels
        self.nb_classes = int(max(labels)+1)
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
