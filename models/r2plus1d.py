import torchvision
from torchvision import models
import torch.nn as nn

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

# def initialize_model(arch_name, num_classes)

class MyR2plus1d(nn.Module):
    def __init__(self, num_classes, use_pretrained = True, init_std=0.01, model_name = 'r2plus1d' ):
        super(MyR2plus1d, self).__init__()
        # self.model_ft = models.__dict__[model_name](pretrained=use_pretrained)
        self.model_ft = models.video.r2plus1d_18( pretrained=use_pretrained)
        num_ftrs = self.model_ft.fc.in_features
        self.init_std = init_std
        modules = list(self.model_ft.children())[:-1]
        self.model_ft = nn.Sequential(*modules)
        self.clsfr = nn.Linear( num_ftrs, num_classes )
        normal_init(self.clsfr, std=self.init_std)
    def forward(self, x):
        feat= self.model_ft(x).squeeze()
        if len(feat.size()) == 1:
            feat = feat.unsqueeze(0)
        pred_cls = self.clsfr(feat)
        return pred_cls

