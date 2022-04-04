import numpy as np
import math
import torch
from torch import Tensor
from utils.nn.model.ParticleNet import ParticleNetTagger

def get_model(data_config, **kwargs):

    ## input numer of point features to EdgeConvBlock
    point_features = 48;
    ## convoluational layers in EdgeConvBlock and kNN
    conv_params = [
        (16, (192, 160, 128)),
        (12, (224, 192, 160)),
        (8,  (224, 192, 160))
        ]
    ## use fusion layer for edge-conv block
    use_fusion = True
    ## fully connected output layers
    fc_params = [
        (192, 0.1),
        (160, 0.1),
        (128, 0.1),
        (96, 0.1),
        (64, 0.1)
    ]

    ## classes and features
    pf_features_dims = len(data_config.input_dicts['pf_features'])
    sv_features_dims = len(data_config.input_dicts['sv_features'])
    num_classes = len(data_config.label_value);
    num_targets = len(data_config.target_value)
    model = ParticleNetTagger(pf_features_dims, 
                              sv_features_dims, 
                              num_classes,
                              num_targets,
                              conv_params, 
                              fc_params,
                              input_dims=point_features, 
                              use_fusion=use_fusion,
                              use_fts_bn=kwargs.get('use_fts_bn', False),
                              use_counts=kwargs.get('use_counts', True),
                              pf_input_dropout=kwargs.get('pf_input_dropout', None),
                              sv_input_dropout=kwargs.get('sv_input_dropout', None),
                              for_inference=kwargs.get('for_inference', False)
                          )

    model_info = {
        'input_names':list(data_config.input_names),
        'input_shapes':{k:((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names':['output'],
        'dynamic_axes':{**{k:{0:'N', 2:'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'output':{0:'N'}}},
        }

    return model, model_info


class CrossEntropyLogCoshLoss(torch.nn.L1Loss):
    __constants__ = ['reduction','nclass','ntarget','loss_lambda']

    def __init__(self, reduction: str = 'mean', nclass: int = 1, ntarget: int = 1, loss_lambda: float = 1.) -> None:
        super(CrossEntropyLogCoshLoss, self).__init__(None, None, reduction)
        self.nclass = nclass;
        self.ntarget = ntarget;
        self.loss_lambda = loss_lambda

    def forward(self, input: Tensor, target: Tensor) -> Tensor:

        input_reg  = input[:,self.nclass:self.nclass+self.ntarget].squeeze();
        target_reg = target[:,1:].squeeze();
        loss_reg = (input_reg-target_reg)+torch.nn.functional.softplus(-2.*(input_reg-target_reg))-math.log(2);

        input_cat  = input[:,:self.nclass].squeeze();
        target_cat = target[:,:1].squeeze().long();
        loss_cat = torch.nn.functional.cross_entropy(input_cat,target_cat,reduction=self.reduction);
                
        if self.reduction == 'none':            
            return loss_cat+self.loss_lambda*loss_reg, loss_cat, loss_reg;
        elif self.reduction == 'mean':
            return loss_cat+self.loss_lambda*loss_reg.mean(), loss_cat, loss_reg.mean()
        elif self.reduction == 'sum':
            return loss_cat+self.loss_lambda*loss_reg.sum(), loss_cat, loss_reg.sum()

class CrossEntropyHuberLoss(torch.nn.L1Loss):

    __constants__ = ['reduction','nclass','ntarget','delta','loss_lambda']

    def __init__(self, reduction: str = 'mean', nclass: int = 1, ntarget: int = 1, delta: float = 1., loss_lambda: float = 1.) -> None:
        super(CrossEntropyHuberLoss, self).__init__(None, None, reduction)
        self.nclass = nclass;
        self.ntarget = ntarget;
        self.delta = delta;
        self.loss_lambda = loss_lambda;

    def forward(self, input: Tensor, target: Tensor) -> Tensor:

        input_reg  = input[:,self.nclass:self.nclass+self.ntarget].squeeze();
        target_reg = target[:,1:].squeeze();
        loss_reg = torch.nn.HuberLoss(input_reg,target_reg,reduction=self.reduction,delta=self.delta);

        input_cat  = input[:,:self.nclass].squeeze();
        target_cat = target[:,:1].squeeze().long();
        loss_cat = torch.nn.functional.cross_entropy(input_cat,target_cat,reduction=self.reduction);

        return loss_cat+loss_lambda*loss_reg, loss_cat, loss_reg

class CrossEntropyMSELoss(torch.nn.L1Loss):

    __constants__ = ['reduction','nclass','ntarget','loss_lambda']

    def __init__(self, reduction: str = 'mean', nclass: int = 1, ntarget: int = 1, loss_lambda: float = 1.) -> None:
        super(CrossEntropyHuberLoss, self).__init__(None,None,reduction)
        self.nclass = nclass;
        self.ntarget = ntarget;
        self.loss_lambda = loss_lambda;

    def forward(self, input: Tensor, target: Tensor) -> Tensor:

        input_reg  = input[:,self.nclass:self.nclass+self.ntarget].squeeze();
        target_reg = target[:,1:].squeeze();
        loss_reg = torch.nn.MSELoss(input_reg,target_reg,reduction=self.reduction);

        input_cat  = input[:,:self.nclass].squeeze();
        target_cat = target[:,:1].squeeze().long();
        loss_cat = torch.nn.functional.cross_entropy(input_cat,target_cat,reduction=self.reduction);

        return loss_cat+loss_lambda*loss_reg, loss_cat, loss_reg


def get_loss(data_config, **kwargs):

    nclass  = len(data_config.label_value);
    ntarget = len(data_config.target_value);

    if kwargs.get('loss_mode', 3) == 1:
        return CrossEntropyMSELoss(reduction=kwargs.get('reduction','mean'),loss_lambda=kwargs.get('loss_lambda',1),nclass=nclass,ntarget=ntarget);
    elif kwargs.get('loss_mode', 3) == 2:
        return CrossEntropyHuberLoss(reduction=kwargs.get('reduction','mean'),loss_lambda=kwargs.get('loss_lambda',1),delta=kwargs.get('delta',1),nclass=nclass,ntarget=ntarget);
    elif kwargs.get('loss_mode', 3) == 3:
        return CrossEntropyLogCoshLoss(reduction=kwargs.get('reduction','mean'),loss_lambda=kwargs.get('loss_lambda',1),nclass=nclass,ntarget=ntarget);
    else:
        return CrossEntropyLogCoshLoss(reduction=kwargs.get('reduction','mean'),loss_lambda=kwargs.get('loss_lambda',1),nclass=nclass,ntarget=ntarget);
