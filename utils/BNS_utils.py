import torch
import torch.nn as nn
from utils.utils_ import AverageMeter, AverageMeterTensor
from utils.norm_stats_utils import compute_regularization

l1_loss = nn.L1Loss(reduction='mean')

def compute_kld(mean_true, mean_pred, var_true, var_pred):
    # mean1 and std1 are for true distribution
    # mean2 and std2 are for pred distribution
    # kld_mv = torch.log(std_pred / std_true) + (std_true ** 2 + (mean_true - mean_pred) ** 2) / (2 * std_pred ** 2) - 0.5

    kld_mv = 0.5 * torch.log(torch.div(var_pred, var_true)) + (var_true + (mean_true - mean_pred) ** 2) / \
             (2 * var_pred) - 0.5
    kld_mv = torch.sum(kld_mv)
    return kld_mv


class BNFeatureHook():
    def __init__(self, module, reg_type='l2norm', running_manner = False, use_src_stat_in_reg = True, momentum = 0.1):

        self.hook = module.register_forward_hook(self.hook_fn)  # register a hook func to a module
        self.reg_type = reg_type
        self.running_manner = running_manner
        self.use_src_stat_in_reg = use_src_stat_in_reg  # whether to use the source statistics in regularization loss
        # todo keep the initial module.running_xx.data (the statistics of source model)
        #   if BN layer is not set to eval,  these statistics will change
        if self.use_src_stat_in_reg:
            self.source_mean = module.running_mean.data
            self.source_var = module.running_var.data
        if self.running_manner:
            # initialize the statistics of computation in running manner
            self.mean = torch.zeros_like( module.running_mean)
            self.var = torch.zeros_like(module.running_var)
        self.momentum = momentum

    def hook_fn(self, module, input, output):  # input in shape (B, C, T, H, W)

        nch = input[0].shape[1]
        if isinstance(module, nn.BatchNorm1d):
            # input in shape (B, C) or (B, C, T)
            if len(input[0].shape) == 2: #  todo  BatchNorm1d in TAM G branch  input is (N*C,  T )
                batch_mean = input[0].mean([0,])
                batch_var = input[0].permute(1, 0,).contiguous().view([nch, -1]).var(1, unbiased=False)  # compute the variance along each channel
            elif len(input[0].shape) == 3:  # todo BatchNorm1d in TAM L branch  input is (N, C, T)
                batch_mean = input[0].mean([0,2])
                batch_var = input[0].permute(1, 0, 2).contiguous().view([nch, -1]).var(1, unbiased=False)  # compute the variance along each channel
        elif isinstance(module, nn.BatchNorm2d):
            # input in shape (B, C, H, W)
            batch_mean = input[0].mean([0, 2, 3])
            batch_var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)  # compute the variance along each channel
        elif isinstance(module, nn.BatchNorm3d):
            # input in shape (B, C, T, H, W)
            batch_mean = input[0].mean([0, 2, 3, 4])
            batch_var = input[0].permute(1, 0, 2, 3, 4).contiguous().view([nch, -1]).var(1,  unbiased=False)  # compute the variance along each channel

        self.mean =  self.momentum * batch_mean + (1.0 - self.momentum) * self.mean.detach() if self.running_manner else batch_mean
        self.var = self.momentum * batch_var + (1.0 - self.momentum) * self.var.detach() if self.running_manner else batch_var
        # todo if BN layer is set to eval, these two are the same;  otherwise, module.running_xx.data keeps changing
        self.mean_true = self.source_mean if self.use_src_stat_in_reg else module.running_mean.data
        self.var_true = self.source_var if self.use_src_stat_in_reg else module.running_var.data
        self.r_feature = compute_regularization(mean_true = self.mean_true, mean_pred = self.mean, var_true=self.var_true, var_pred = self.var, reg_type = self.reg_type)


        # if self.reg_type == 'l2norm':
        #     self.r_feature = torch.norm(self.var_true - self.var, 2) + torch.norm(self.mean_true - self.mean,2)
        # if self.reg_type == 'l1_loss':
        #     self.r_feature = torch.norm(self.var_true  - self.var, 1) + torch.norm(self.mean_true - self.mean, 1)
        # elif self.reg_type == 'kld':
        #     self.r_feature = compute_kld(mean_true=self.mean_true, mean_pred= self.mean,
        #                                             var_true= self.var_true, var_pred= self.var)

    def add_hook_back(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)  # register a hook func to a module

    def close(self):
        self.hook.remove()







class TempStatsRegHook():
    def __init__(self, module, clip_len = None, temp_stats_clean_tuple = None, reg_type='l2norm', ):

        self.hook = module.register_forward_hook(self.hook_fn)  # register a hook func to a module
        self.clip_len = clip_len
        # self.temp_mean_clean, self.temp_var_clean = temp_stats_clean_tuple

        self.reg_type = reg_type
        # self.running_manner = running_manner
        # self.use_src_stat_in_reg = use_src_stat_in_reg  # whether to use the source statistics in regularization loss
        # todo keep the initial module.running_xx.data (the statistics of source model)
        #   if BN layer is not set to eval,  these statistics will change
        # if self.use_src_stat_in_reg:
        #     self.source_mean = module.running_mean.data
        #     self.source_var = module.running_var.data
        self.source_mean, self.source_var = temp_stats_clean_tuple

        self.source_mean = torch.tensor(self.source_mean).cuda()
        self.source_var = torch.tensor(self.source_var).cuda()

        # self.source_mean = self.source_mean.mean((1,2))
        # self.source_var = self.source_var.mean((1,2 ))

        # if self.running_manner:
        #     # initialize the statistics of computation in running manner
        #     self.mean = torch.zeros_like( self.source_mean)
        #     self.var = torch.zeros_like( self.source_var)

        self.mean_avgmeter = AverageMeterTensor()
        self.var_avgmeter = AverageMeterTensor()

        # self.momentum = momentum

    def hook_fn(self, module, input, output):

        if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
            # output is in shape (N, C, T)  or   (N*C, T )
            raise NotImplementedError('Temporal statistics computation for nn.Conv1d not implemented!')
        elif isinstance(module, nn.Conv2d):
            # output is in shape (N*T,  C,  H,  W)
            nt, c, h, w = output.size()
            t = self.clip_len
            bz = nt // t

            output = output.view(bz, t, c, h, w).permute(0, 2, 1, 3,  4).contiguous() # # ( N*T,  C,  H, W) -> (N, C, T, H, W)
        elif isinstance(module, nn.Conv3d):
            # output is in shape (N, C, T, H, W)
            bz, c, t, h, w = output.size()
            output = output
        else:
            raise Exception(f'undefined module {module}')
        # spatial_dim = h * w
        # todo compute the statistics only along the temporal dimension T,  then take the average for all samples  N
        #  the statistics are in shape  (C, H, W),
        batch_mean = output.mean(2).mean(0)  #  (N, C, T, H, W)  ->  (N, C, H, W) ->  (C, H, W)
        # temp_var = new_output.permute(1, 3, 4, 0, 2).contiguous().view([c, t, -1]).var(2, unbiased = False )
        batch_var = output.permute(0, 1, 3, 4, 2).contiguous().var(-1, unbiased=False).mean(0)  # (N, C, T, H, W) -> #  (N, C, H, W, T) -> (N, C, H, W) ->  (C, H, W)

        # batch_mean = output.mean(2).mean((0, 2,3)) #  (N, C, T, H, W)  ->  (N, C, H, W) ->  (C,)
        # batch_var = output.permute(0, 1, 3, 4, 2).contiguous().var(-1, unbiased=False).mean((0, 2,3)) # (N, C, T, H, W) -> #  (N, C, H, W, T) -> (N, C, H, W) ->  (C,)


        self.mean_avgmeter.update(batch_mean, n= bz)
        self.var_avgmeter.update(batch_var, n= bz)

        if self.reg_type == 'l2norm':
            # # todo sum of squared difference,  averaged over  h * w
            # self.r_feature = torch.sum(( self.source_var - self.var_avgmeter.avg )**2 ) / spatial_dim + torch.sum(( self.source_mean - self.mean_avgmeter.avg )**2 ) / spatial_dim
            self.r_feature = torch.norm(self.source_var - self.var_avgmeter.avg, 2) + torch.norm(self.source_mean - self.mean_avgmeter.avg, 2)
        else:
            raise NotImplementedError

    def close(self):
        self.hook.remove()




class ComputeSpatioTemporalStatisticsHook():
    def __init__(self, module, clip_len = None,):

        self.hook = module.register_forward_hook(self.hook_fn)  # register a hook func to a module
        self.clip_len = clip_len

    def hook_fn(self, module, input, output):

        if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
            # output is in shape (N, C, T)  or   (N*C, T )
            raise NotImplementedError('Temporal statistics computation for nn.Conv1d not implemented!')
        elif isinstance(module, nn.Conv2d):
            # output is in shape (N*T,  C,  H,  W)
            nt, c, h, w = output.size()
            t = self.clip_len
            bz = nt // t
            output = output.view(bz, t, c, h, w).permute(0, 2, 1, 3,  4).contiguous() # # ( N*T,  C,  H, W) -> (N, C, T, H, W)
        elif isinstance(module, nn.Conv3d):
            # output is in shape (N, C, T, H, W)
            bz, c, t, h, w = output.size()
            output = output
        else:
            raise Exception(f'undefined module {module}')

        # todo compute the statistics only along the temporal dimension T,  then take the average for all samples  N
        #  the statistics are in shape  (C, H, W),
        self.temp_mean = output.mean((0, 2,3,4)).mean(0) #  (N, C, T, H, W)  ->   (C, )
        self.temp_var = output.permute(1, 0, 2, 3, 4).contiguous().view([c, -1]).var(1, unbiased=False) #  (N, C, T, H, W) -> (C, N, T, H, W) -> (C, )

        # batch_mean = input[0].mean([0, 2, 3, 4])
        # batch_var = input[0].permute(1, 0, 2, 3, 4).contiguous().view([nch, -1]).var(1, unbiased=False)  # compute the variance along each channel

        self.temp_mean = output.mean(2).mean(0)  #  (N, C, T, H, W)  ->  (N, C, H, W) ->  (C, H, W)
        # temp_var = new_output.permute(1, 3, 4, 0, 2).contiguous().view([c, t, -1]).var(2, unbiased = False )
        self.temp_var = output.permute(0, 1, 3, 4, 2).contiguous().var(-1, unbiased=False).mean(0)  # (N, C, T, H, W) -> #  (N, C, H, W, T) -> (N, C, H, W) ->  (C, H, W)

        # self.temp_mean = output.mean(2).mean((0, 2, 3)) #  (N, C, T, H, W)  ->  (N, C, H, W) ->  (C,)
        # self.temp_var = output.permute(0, 1, 3, 4, 2).contiguous().var(-1, unbiased=False).mean((0, 2, 3) )   # (N, C, T, H, W) -> #  (N, C, H, W, T) -> (N, C, H, W) ->  (C,)


    def close(self):
        self.hook.remove()


class ComputeTemporalStatisticsHook():
    def __init__(self, module, clip_len = None,):

        self.hook = module.register_forward_hook(self.hook_fn)  # register a hook func to a module
        self.clip_len = clip_len

    def hook_fn(self, module, input, output):

        if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
            # output is in shape (N, C, T)  or   (N*C, T )
            raise NotImplementedError('Temporal statistics computation for nn.Conv1d not implemented!')
        elif isinstance(module, nn.Conv2d):
            # output is in shape (N*T,  C,  H,  W)
            nt, c, h, w = output.size()
            t = self.clip_len
            bz = nt // t
            output = output.view(bz, t, c, h, w).permute(0, 2, 1, 3,  4).contiguous() # # ( N*T,  C,  H, W) -> (N, C, T, H, W)
        elif isinstance(module, nn.Conv3d):
            # output is in shape (N, C, T, H, W)
            bz, c, t, h, w = output.size()
            output = output
        else:
            raise Exception(f'undefined module {module}')

        # todo compute the statistics only along the temporal dimension T,  then take the average for all samples  N
        #  the statistics are in shape  (C, H, W),
        self.temp_mean = output.mean(2).mean(0)  #  (N, C, T, H, W)  ->  (N, C, H, W) ->  (C, H, W)
        # temp_var = new_output.permute(1, 3, 4, 0, 2).contiguous().view([c, t, -1]).var(2, unbiased = False )
        self.temp_var = output.permute(0, 1, 3, 4, 2).contiguous().var(-1, unbiased=False).mean(0)  # (N, C, T, H, W) -> #  (N, C, H, W, T) -> (N, C, H, W) ->  (C, H, W)

        # self.temp_mean = output.mean(2).mean((0, 2, 3)) #  (N, C, T, H, W)  ->  (N, C, H, W) ->  (C,)
        # self.temp_var = output.permute(0, 1, 3, 4, 2).contiguous().var(-1, unbiased=False).mean((0, 2, 3) )   # (N, C, T, H, W) -> #  (N, C, H, W, T) -> (N, C, H, W) ->  (C,)


    def close(self):
        self.hook.remove()


def choose_layers(model, candidate_layers):

    chosen_layers = []
    # choose all the BN layers
    # candidate_layers = [nn.BatchNorm1d,  nn.BatchNorm2d, nn.BatchNorm3d  ]
    counter = [0] * len(candidate_layers)
    # for m in model.modules():
    for nm, m in model.named_modules():
        for candidate_idx, candidate in enumerate(candidate_layers):
            if isinstance(m, candidate):
                counter[candidate_idx] += 1
                chosen_layers.append((nm, m))
    # for idx in range(len(candidate_layers)):
    #     print(f'Number of {candidate_layers[idx]}  : {counter[idx]}')
    return chosen_layers


def freeze_except_bn(model, bn_condidiate_layers, ):
    """
    freeze the model, except the BN layers
    :param model:
    :param bn_condidiate_layers:
    :return:
    """

    model.train()  #
    model.requires_grad_(False)
    for m in model.modules():
        for candidate in bn_condidiate_layers:
            if isinstance(m, candidate):
                m.requires_grad_(True)
    return model

def collect_bn_params(model, bn_candidate_layers):
    params = []
    names = []
    for nm, m in model.named_modules():
        for candidate in bn_candidate_layers:
            if isinstance(m, candidate):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']: # weight is scale gamma, bias is shift beta
                        params.append(p)
                        names.append( f"{nm}.{np}")
    return params, names