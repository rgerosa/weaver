import torch
from utils.nn.model.ParticleNet import ParticleNetTagger


def get_model(data_config, **kwargs):

    conv_params = [
        (16, (160, 128, 96)),
        (12, (192, 160, 128)),
        (8,  (192, 160, 128)),
        ]

    fc_params = [
        (192, 0.1),
        (160, 0.1),
        (128, 0.1),
        (96, 0.1),
        (64, 0.1)
    ]

    use_fusion = True

    pf_features_dims = len(data_config.input_dicts['pf_features'])
    sv_features_dims = len(data_config.input_dicts['sv_features'])
    num_classes = len(data_config.label_value)
    num_targets = 0;
    model = ParticleNetTagger(pf_features_dims, 
                              sv_features_dims, 
                              num_classes,
                              num_targets,
                              conv_params, 
                              fc_params,
                              input_dims=48,
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
        'output_names':['softmax'],
        'dynamic_axes':{**{k:{0:'N', 2:'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax':{0:'N'}}},
        }

    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()
