import math
from torch import nn
import torch.nn.init as init
from timm.layers import trunc_normal_tf_


def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        match scheme:
            case 'normal':
                nn.init.normal_(module.weight, std=.02)  
            case 'uniform':
                nn.init.uniform_(module.weight, a=0, b=1)  
            case 'constant':
                nn.init.constant_(module.weight, 0.5)  
            case 'trunc_normal':
                trunc_normal_tf_(module.weight, std=.02)  
            case 'xavier_normal':
                nn.init.xavier_normal_(module.weight)
            case 'xavier_uniform':
                nn.init.xavier_uniform_(module.weight)
            case 'kaiming_normal':
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            case 'kaiming_uniform':
                nn.init.kaiming_uniform_(module.weight, mode='fan_out', nonlinearity='relu')
            case _:
                # efficientnet like
                fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                fan_out //= module.groups
                nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        if scheme == 'xavier_normal':
            init.xavier_normal_(module.weight)
        elif scheme == 'kaiming_normal':
            init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        else:
            init.normal_(module.weight, std=0.02)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
        
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.constant_(module.weight, 1.0)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
