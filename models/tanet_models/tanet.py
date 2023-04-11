# Code for "TAM: Temporal Adaptive Module for Video Recognition"
# arXiv: 2005.06803
# Zhaoyang liu*, Limin Wang, Wayne Wu, Chen Qian, Tong Lu
# zyliumy@gmail.com

from torch import nn

# from ops.basic_ops import ConsensusModule
# from ops.transforms import *
from models.tanet_models.basic_ops import ConsensusModule
from models.tanet_models.transforms import *
from torchvision.transforms import Compose
from torch.nn.init import normal_, constant_


class TSN(nn.Module):
    def __init__(self,
                 num_class,
                 num_segments,
                 modality,
                 base_model='resnet101',
                 new_length=None,
                 consensus_type='avg',
                 before_softmax=True,
                 dropout=0.8,
                 img_feature_dim=256,
                 crop_num=1,
                 partial_bn=True,
                 print_spec=True,
                 pretrain='imagenet',
                 tam=False,
                 fc_lr5=False,
                 non_local=False):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num  # number of spatial crops
        self.consensus_type = consensus_type
        self.img_feature_dim = img_feature_dim  # the dimension of the CNN feature to represent each frame
        self.pretrain = pretrain

        self.tam = tam  # if True, use shift for models
        self.base_model_name = base_model
        self.fc_lr5 = fc_lr5
        self.non_local = non_local  # if True, add non local block

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length
        if print_spec:
            print(("""
                   Initializing TSN with base model: {}.
                   TSN Configurations:
                   input_modality:     {}
                   num_segments:       {}
                   new_length:         {}
                   consensus_module:   {}
                   dropout_ratio:      {}
                   img_feature_dim:    {}
                   """.format(base_model, self.modality, self.num_segments,
                              self.new_length, consensus_type, self.dropout,
                              self.img_feature_dim)))

        self._prepare_base_model(base_model)

        feature_dim = self._prepare_tsn(num_class) #  todo    delete the last FC layer of  ResNet (pretrained on ImageNet, 1000 classes),  replaced with Dropout + new_fc layer

        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")

        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model,
                              self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name,
                    nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:    #  todo    delete the last FC layer of  ResNet (pretrained on ImageNet, 1000 classes),  replaced with Dropout + new_fc layer
            setattr(self.base_model, self.base_model.last_layer_name,
                    nn.Dropout(p=self.dropout))       #   todo last_layer_name is  fc,   replace the last layer of ResNet,  with Dropout,  followed by new_fc layer from feature_dim to num_class
            # self.new_fc = nn.Linear(feature_dim, num_class)
            if self.consensus_type in ['TRN', 'TRNmultiscale']:
                # create a new linear layer as the frame feature
                self.new_fc = nn.Linear(feature_dim, self.img_feature_dim)
            else:
                # the default consensus types in TSN
                self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal_(
                getattr(self.base_model,
                        self.base_model.last_layer_name).weight, 0, std)
            constant_(
                getattr(self.base_model, self.base_model.last_layer_name).bias,
                0)
        else:
            if hasattr(self.new_fc, 'weight'):
                normal_(self.new_fc.weight, 0, std)
                constant_(self.new_fc.bias, 0)
        return feature_dim

    def _prepare_base_model(self, base_model):
        print('=> base model: {}'.format(base_model))

        if 'resnet' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True)  # get resnet model from torchvision.models
            if self.tam:
                print('Adding temporal adaptive moduel...')
                # from ops.temporal_module import make_temporal_modeling
                from models.tanet_models.temporal_module import make_temporal_modeling
                make_temporal_modeling(self.base_model,
                                       self.num_segments,
                                       t_kernel_size=3,
                                       t_stride=1,
                                       t_padding=1)  # adding TAM to each TemporalBottleneck of each stage/layer,    layer1/2/3/4 has  3/4/6/3 TemporalBottleneck modules

            if self.non_local:
                print('Adding non-local module...')
                from ops.non_local import make_non_local
                make_non_local(self.base_model, self.num_segments)

            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406
                                   ] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [
                    np.mean(self.input_std) * 2
                ] * 3 * self.new_length

        elif base_model == 'BNInception':
            from archs.bn_inception import bninception
            self.base_model = bninception(pretrained=self.pretrain)
            self.input_size = self.base_model.input_size
            self.input_mean = self.base_model.mean
            self.input_std = self.base_model.std
            self.base_model.last_layer_name = 'fc'
            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)
            if self.is_shift:
                print('Adding temporal shift...')
                self.base_model.build_temporal_ops(
                    self.num_segments,
                    is_temporal_shift=self.shift_place,
                    shift_div=self.shift_div)
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn and mode:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False


    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []    #  5 x lr  for  first conv layer weight
        first_conv_bias = []   # 10x lr for   first conv layer bias
        normal_weight = []   #  1x  lr  for other conv layer weights
        normal_bias = []    #  2x   lr   for other conv layer bias
        lr5_weight = []   #  5x  lr  for FC layers weight,  if not finetuned from a checkpoint of the same dataset
        lr10_bias = []   #  10x  lr  for FC layers bias,    if not finetuned from a checkpoint of the same dataset
        bn = []   #  1x  lr  for  batch norm parameters
        custom_ops = []

        conv_cnt = 0
        bn_cnt = 0
        conv_op = (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
                   torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d,
                   torch.nn.ConvTranspose3d)
        # from ops import temporal_module
        for m in self.modules():
            if isinstance(m, conv_op):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                if self.fc_lr5:
                    lr5_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                if len(ps) == 2:
                    if self.fc_lr5:
                        lr10_bias.append(ps[1])
                    else:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError(
                        "New atomic module type: {}. Need to give it a learning policy"
                        .format(type(m)))

        params_group = [
            {
                'params': first_conv_weight,
                'lr_mult': 5 if self.modality == 'Flow' else 1,
                'decay_mult': 1,
                'name': "first_conv_weight"
            },
            {
                'params': first_conv_bias,
                'lr_mult': 10 if self.modality == 'Flow' else 2,
                'decay_mult': 0,
                'name': "first_conv_bias"
            },
            {
                'params': normal_weight,
                'lr_mult': 1,
                'decay_mult': 1,
                'name': "normal_weight"
            },
            {
                'params': normal_bias,
                'lr_mult': 2,
                'decay_mult': 0,
                'name': "normal_bias"
            },
            {
                'params': bn,
                'lr_mult': 1,
                'decay_mult': 0,
                'name': "BN scale/shift"
            },
            {
                'params': custom_ops,
                'lr_mult': 1,
                'decay_mult': 1,
                'name': "custom_ops"
            },
            # for fc
            {
                'params': lr5_weight,
                'lr_mult': 5,
                'decay_mult': 1,
                'name': "lr5_weight"
            },
            {
                'params': lr10_bias,
                'lr_mult': 10,
                'decay_mult': 0,
                'name': "lr10_bias"
            },
        ]

        return params_group

    def forward(self, input, no_reshape=False): #  todo input (bz, C*T, 224, 224 )
        if not no_reshape:
            sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

            if self.modality == 'RGBDiff':
                sample_len = 3 * self.new_length
                input = self._get_diff(input)

            base_out = self.base_model(
                input.view((-1, sample_len) + input.size()[-2:]) )  #  (bz, T, C, 224, 224 ) -> (bz * T, C, 224, 224 ) -> (bz * T, 2048)
        else:
            base_out = self.base_model(input)

        #print('1:base_out=', base_out.size())
        if self.dropout > 0:
            base_out = self.new_fc(base_out)  # (bz * T, 2048)  -> (bz * T, n_class)

        #print('2:base_out=', base_out.size())
        if not self.before_softmax:
            base_out = self.softmax(base_out)

        base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])  #  (bz * T, n_class) -> (bz, T, n_class)
        #print('3:base_out=', base_out.size())
        output = self.consensus(base_out)   #  (bz, T, n_class)  -> (bz, 1, n_class)
        #print('4:base_out=', output.size())
        return output.squeeze(1)  #  (bz, 1, n_class) -> (bz,  n_class)

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((
            -1,
            self.num_segments,
            self.new_length + 1,
            input_c,
        ) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :,
                         x, :, :, :] = input_view[:, :,
                                                  x, :, :, :] - input_view[:, :,
                                                                           x -
                                                                           1, :, :, :]
            else:
                new_data[:, :, x -
                         1, :, :, :] = input_view[:, :,
                                                  x, :, :, :] - input_view[:, :,
                                                                           x -
                                                                           1, :, :, :]

        return new_data

    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(
            filter(lambda x: isinstance(modules[x], nn.Conv2d),
                   list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (
            2 * self.new_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(
            dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length,
                             conv_layer.out_channels,
                             conv_layer.kernel_size,
                             conv_layer.stride,
                             conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys(
        ))[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)

        if self.base_model_name == 'BNInception':
            import torch.utils.model_zoo as model_zoo
            sd = model_zoo.load_url(
                'https://www.dropbox.com/s/35ftw2t4mxxgjae/BNInceptionFlow-ef652051.pth.tar?dl=1'
            )
            base_model.load_state_dict(sd)
            print('=> Loading pretrained Flow weight done...')
        else:
            print('#' * 30, 'Warning! No Flow pretrained model is found')
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = filter(lambda x: isinstance(modules[x], nn.Conv2d),
                                list(range(len(modules))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (
                3 * self.new_length, ) + kernel_size[2:]
            new_kernels = params[0].data.mean(
                dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (
                3 * self.new_length, ) + kernel_size[2:]
            new_kernels = torch.cat((params[0].data, params[0].data.mean(
                dim=1, keepdim=True).expand(new_kernel_size).contiguous()), 1)
            new_kernel_size = kernel_size[:1] + (
                3 + 3 * self.new_length, ) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1],
                             conv_layer.out_channels,
                             conv_layer.kernel_size,
                             conv_layer.stride,
                             conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys(
        ))[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self, flip=True, dataset='Kinetics'):   #  todo get training augmentation
        label_transforms = None
        if 'somethingv2' in dataset:
            label_transforms = {
                86: 87,
                87: 86,
                93: 94,
                94: 93,
                166: 167,
                167: 166
            }
        elif 'something' in dataset:
            label_transforms = {3: 5, 5: 3, 31: 42, 42: 31, 53: 67, 67: 53}

        if self.modality == 'RGB':
            if flip:
                crop_op = GroupMultiScaleCrop_TANet(self.input_size, [1, .875, .75, .66])
                flip_op = GroupRandomHorizontalFlip_TANet(False, label_transforms)  #  todo   label_transforms for some classes in SSv2 ????
                return Compose([crop_op, flip_op])
            else:
                print('#' * 20, 'NO FLIP!!!')
                return Compose([
                    GroupMultiScaleCrop_TANet(self.input_size, [1, .875, .75, .66])
                ])
        elif self.modality == 'Flow':
            return Compose([
                GroupMultiScaleCrop_TANet(self.input_size, [1, .875, .75]),
                GroupRandomHorizontalFlip_TANet(is_flow=True)
            ])
        elif self.modality == 'RGBDiff':
            return Compose([
                GroupMultiScaleCrop_TANet(self.input_size, [1, .875, .75]),
                GroupRandomHorizontalFlip_TANet(is_flow=False)
            ])
