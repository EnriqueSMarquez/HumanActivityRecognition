import torch
import torchvision
from torch.autograd import Variable
from tqdm import tqdm
import _pickle as pickle
from sklearn.metrics import f1_score,confusion_matrix
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools

class Tester():
    def __init__(self,model,test_loader,id2label,criterion,verbose=False,saving_folder=None,nb_outputs=1,macro=False):
        self.model = model
        self.verbose = verbose
        self.test_loader = test_loader
        self.criterion = criterion
        self.saving_folder = saving_folder
        self.nb_outputs = nb_outputs
        self.macro = macro
        self.id2label = id2label
    def test(self,cm=False):
        total = 0.
        corrects = [0.]*self.nb_outputs
        running_losses = [0.]*self.nb_outputs
        f1s = []
        predicted = []
        labels = []
        for i,(x,y)  in enumerate(self.test_loader):
            batch_losses,batch_corrects,batch_f1s,batch_predicted = self.test_batch(x,y)
            predicted += [batch_predicted[0].cpu().numpy().flatten()]
            labels += [y.cpu().numpy().flatten()]
            running_losses = [0.99 * running_loss + 0.01 * batch_loss.data.item()
                             for running_loss, batch_loss in zip(running_losses, batch_losses)]
            total += x.size(0)
            corrects = [a + b for a, b in zip(corrects, batch_corrects)]
            f1s += [batch_f1s]
        if cm:
            cm = confusion_matrix(np.asarray(labels).flatten(),np.asarray(predicted).flatten())
            self.plot_confusion_matrix(cm,classes=self.id2label,normalize=True)
        return {'loss' : running_losses,
                'acc' : [correct/total for correct in corrects],
                'f1' : np.mean(f1s,axis=0).tolist()}
    def test_batch(self,x,y):
        x = x.cuda()
        y = y.cuda().view(-1)
        outs = self.model(x)
        losses = [self.criterion(out, y) for out in outs]
        predicts = [torch.max(out.data, 1)[1] for out in outs]
        # _,truth = torch.max(y.data,1)
        corrects = [(torch.max(out.data, 1)[1] == y.data).sum().item() for out in outs]
        if self.macro:
            f1s = [f1_score(y.data.cpu().numpy().reshape(-1),
                            predict.cpu().numpy().reshape(-1),average='macro') for predict in predicts]  
        else: 
            f1s = [f1_score(y.data.cpu().numpy().reshape(-1),
                            predict.cpu().numpy().reshape(-1),average='weighted') for predict in predicts]
        return losses,corrects,f1s,predicts
    def plot_confusion_matrix(self,cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, [value for value in classes.values()], rotation=90)
        plt.yticks(tick_marks, [value for value in classes.values()])

        fmt = '.1f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        np.set_printoptions(precision=2)
        plt.savefig(os.path.join(self.saving_folder, 'confusion_matrix.png'),bbox_inches='tight')
