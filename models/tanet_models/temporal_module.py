# Code for "TAM: Temporal Adaptive Module for Video Recognition"
# arXiv: 2005.06803
# Zhaoyang liu*, Limin Wang, Wayne Wu, Chen Qian, Tong Lu
# zyliumy@gmail.com

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class TAM(nn.Module):   #  TAM is concatenate after conv1,bn1,relu
    def __init__(self,
                 in_channels,  # the feature dim after conv1
                 n_segment,
                 kernel_size=3,
                 stride=1,
                 padding=1):
        super(TAM, self).__init__()
        self.in_channels = in_channels
        self.n_segment = n_segment
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        print('TAM with kernel_size {}.'.format(kernel_size))

        self.G = nn.Sequential(  #  todo  global branch: to learn the dynamic/adaptive kernels that are used in convolution to aggregate the temporal info
            nn.Linear(n_segment, n_segment * 2, bias=False),
            nn.BatchNorm1d(n_segment * 2), nn.ReLU(inplace=True),
            nn.Linear(n_segment * 2, kernel_size, bias=False), nn.Softmax(-1))

        self.L = nn.Sequential(  # todo local branch: compute local importance map (attention) in shape C x T, using 2 temporal convolutions
            nn.Conv1d(in_channels,
                      in_channels // 4,
                      kernel_size,
                      stride=1,
                      padding=kernel_size // 2,
                      bias=False), nn.BatchNorm1d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // 4, in_channels, 1, bias=False),
            nn.Sigmoid())

    def forward(self, x):    #  todo input is of shape batch*T, C, H, W,  output is of the same shape  batch*T, C, H, W
        # x.size = N*C*T*(H*W)
        nt, c, h, w = x.size() #  x size :    N*T,  C,  H, W
        t = self.n_segment
        n_batch = nt // t
        new_x = x.view(n_batch, t, c, h, w).permute(0, 2, 1, 3,
                                                     4).contiguous()  # ( N*T,  C,  H, W) -> (N, C, T, H, W)
        out = F.adaptive_avg_pool2d(new_x.view(n_batch * c, t, h, w), (1, 1))  #  (N, C, T, H, W) ->   (N * C, T, 1, 1)
        out = out.view(-1, t)  #  (N * C, T, 1, 1) -> (N * C,  T )
        conv_kernel = self.G(out.view(-1, t)).view(n_batch * c, 1, -1, 1)  #  (N * C,  T )  ->  (N * C,   1,  T,  1 )
        local_activation = self.L(out.view(n_batch, c,
                                           t)).view(n_batch, c, t, 1, 1) #    (N * C,  T ) -> (N, C,  T)  ->  (N ,  C,  T,  1, 1 )
        new_x = new_x * local_activation  #  (N, C, T, H, W)
        out = F.conv2d(new_x.view(1, n_batch * c, t, h * w),
                       conv_kernel,
                       bias=None,
                       stride=(self.stride, 1),
                       padding=(self.padding, 0),
                       groups=n_batch * c)  #   (N, C, T, H, W)  ->   (1, N * C, T, H * W)  ->   (1, N * C, T, H * W)
        out = out.view(n_batch, c, t, h, w) #  (1, N * C, T, H * W) ->   (N, C, T, H, W)
        out = out.permute(0, 2, 1, 3, 4).contiguous().view(nt, c, h, w)  #   (N * T, C, H, W)

        return out


class TemporalBottleneck(nn.Module):
    def __init__(self,
                 net,
                 n_segment=8,
                 t_kernel_size=3,
                 t_stride=1,
                 t_padding=1):
        super(TemporalBottleneck, self).__init__()
        self.net = net   #  here net is a Bottleneck module
        assert isinstance(net, torchvision.models.resnet.Bottleneck)
        self.n_segment = n_segment
        self.tam = TAM(in_channels=net.conv1.out_channels,
                       n_segment=n_segment,
                       kernel_size=t_kernel_size,
                       stride=t_stride,
                       padding=t_padding)

    def forward(self, x):
        identity = x

        out = self.net.conv1(x)
        out = self.net.bn1(out)
        out = self.net.relu(out)
        out = self.tam(out)

        out = self.net.conv2(out)
        out = self.net.bn2(out)
        out = self.net.relu(out)

        out = self.net.conv3(out)
        out = self.net.bn3(out)

        if self.net.downsample is not None:
            identity = self.net.downsample(x)

        out += identity
        out = self.net.relu(out)

        return out


def make_temporal_modeling(net,
                           n_segment=8,
                           t_kernel_size=3,
                           t_stride=1,
                           t_padding=1):
    if isinstance(net, torchvision.models.ResNet):
        n_round = 1

        def make_block_temporal(stage,  #  stage is layer1/2/3/4 in ResNet,  each stage has  3/4/6/3 TemporalBottleneck module
                                this_segment,
                                t_kernel_size=3,
                                t_stride=1,
                                t_padding=1):
            blocks = list(stage.children())
            print('=> Processing this stage with {} blocks residual'.format(
                len(blocks)))
            for i, b in enumerate(blocks):
                # if i >= len(blocks)//2:
                if i % n_round == 0:   # todo turn each Bottleneck module into a TemporalBottleneck,    TemporalBottleneck = (Bottleneck + TAM )
                    blocks[i] = TemporalBottleneck(b, this_segment,
                                                   t_kernel_size, t_stride,
                                                   t_padding)
            return nn.Sequential(*blocks)
        # add temporal modules to layer1, layer2, layer3, layer4  in ResNet,  layer1/2/3/4 are 4 stages,  each stage has 3/4/6/3 TemporalBottleneck module,
        net.layer1 = make_block_temporal(net.layer1, n_segment, t_kernel_size,
                                         t_stride, t_padding)
        net.layer2 = make_block_temporal(net.layer2, n_segment, t_kernel_size,
                                         t_stride, t_padding)
        net.layer3 = make_block_temporal(net.layer3, n_segment, t_kernel_size,
                                         t_stride, t_padding)
        net.layer4 = make_block_temporal(net.layer4, n_segment, t_kernel_size,
                                         t_stride, t_padding)


if __name__ == '__main__':
    # test
    pass
