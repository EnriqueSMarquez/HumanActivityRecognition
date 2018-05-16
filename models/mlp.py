import torch
from torch import nn
from torch.nn.init import kaiming_normal

class MLP(nn.Module):
    def __init__(self,input_size,output_size,hidden_units,layers=3):#LEN(HIDDENUNITS) == LAYERS
        super(MLP, self).__init__()
        self.linear_layers = [nn.Sequential(*[nn.Linear(input_size,hidden_units[0]),nn.Dropout(0.5),nn.ReLU()])]
        self.linear_layers += [nn.Sequential(*[nn.Linear(in_features,out_features),nn.Dropout(0.5),nn.ReLU()]) for in_features,out_features in zip(hidden_units[0:-1],hidden_units[1::])]
        self.linear_layers += [nn.Linear(hidden_units[-1],output_size)]
        self.linear_layers = nn.ModuleList(self.linear_layers)
        self.init_weights()
    def init_weights(self):##CHECK INIT WEIGHTS COS OF SEQUENTIAL
        for m in self.modules():
            if isinstance(m, nn.Linear):
                kaiming_normal(m.weight.data)
            elif isinstance(m, nn.ModuleList):
                for mm in m:
                    if isinstance(mm,nn.Sequential):
                        for mmm in mm:
                            if isinstance(mmm,nn.Linear):
                                kaiming_normal(mmm.weight.data)
            elif isinstance(m,nn.Sequential):
                for mm in m:
                    if isinstance(mm,nn.Linear):
                        kaiming_normal(mm.weight.data)
    def forward(self,x):
        x = x.view(x.size(0),-1)
        for linear_layer in self.linear_layers:
            x = linear_layer(x)
        return x
