import torch
import torch.nn as nn
from utils.utils_ import AverageMeter, AverageMeterTensor, MovingAverageTensor

l1_loss = nn.L1Loss(reduction='mean')
mse_loss = nn.MSELoss(reduction='mean')

def compute_kld(mean_true, mean_pred, var_true, var_pred):
    # mean1 and std1 are for true distribution
    # mean2 and std2 are for pred distribution
    # kld_mv = torch.log(std_pred / std_true) + (std_true ** 2 + (mean_true - mean_pred) ** 2) / (2 * std_pred ** 2) - 0.5

    kld_mv = 0.5 * torch.log(torch.div(var_pred, var_true)) + (var_true + (mean_true - mean_pred) ** 2) / \
             (2 * var_pred) - 0.5
    kld_mv = torch.sum(kld_mv)
    return kld_mv

class ComputeNormStatsHook():
    """
    this hook is to be placed after the normalization layer.
    """
    def __init__(self, module, clip_len = None, stat_type = None, before_norm = None, batch_size = None):
        self.hook = module.register_forward_hook(self.hook_fn)  # register a hook func to a module
        self.clip_len = clip_len
        self.stat_type = stat_type
        self.before_norm = before_norm
        self.batch_size = batch_size
    def hook_fn(self, module, input, output):

        feature = input[0] if self.before_norm else output

        if isinstance(module, nn.BatchNorm1d):
            # raise NotImplementedError('Statistics computation for nn.BatchNorm1d not implemented! ')

            # output is in shape (N, C, T)  or   (N*C, T )
            assert self.stat_type in ['temp', 'temp_v2']
            if len(feature.size()) == 2:

                # todo should have converted  (N*C, T) to (N, C, T), this requires the ACTUAL batch size
                #    but we do not know the actual batch size
                nc, t = feature.size()
                # if self.stat_type == 'temp':
                self.batch_mean = feature.mean(0) # (N*C, T) -> (T, )
                self.batch_var = feature.permute(1, 0).contiguous().var(1, unbiased = False) # (N*C, T) -> (T, NC ) -> (T, )
                # elif self.stat_type == 'temp_v2':
                #     c = nc // self.batch_size
                #     feature = feature.view(self.batch_size, c, t)
                #     self.batch_mean = feature.mean((0, 2))  # (N, C, T) -> (C, )
                #     self.batch_var = feature.permute(1, 0, 2).contiguous().view([c, -1]).var(1,  unbiased=False)  # (N, C, T) -> (C, N, T) -> (C, )
            elif len(feature.size()) == 3:
                n, c, t = feature.size()
                self.batch_mean = feature.mean((0, 2)) # (N, C, T) -> (C, )
                self.batch_var = feature.permute(1, 0, 2).contiguous().view([c, -1]).var(1, unbiased = False) # (N, C, T) -> (C, N, T) -> (C, )


        elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):

            # todo reshape the output into (N, C, T, H, W )
            if isinstance(module, nn.BatchNorm2d):
                # output is in shape (N*T,  C,  H,  W)
                nt, c, h, w = feature.size()
                t = self.clip_len
                bz = nt // t
                feature = feature.view(bz, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()  # # ( N*T,  C,  H, W) -> (N, C, T, H, W)
            elif isinstance(module, nn.BatchNorm3d):
                # output is in shape (N, C, T, H, W)
                bz, c, t, h, w = feature.size()
                feature = feature
            else:
                raise Exception(f'undefined module {module}')

            self.compute_stat_for_NCTHW(feature)

        elif isinstance(module, nn.LayerNorm):
            # todo output is in shape  B T H W C,  !!!!!!!!!!!!!!  notice that in LayerNorm, the mean and variance are computed on the C-dimension
            assert len(feature.size()) == 5
            bz, t, h, w, c = feature.size()
            feature = feature.permute(0, 4, 1, 2, 3).contiguous() #  bz, t, h, w, c ->  bz, c, t, h, w
            self.compute_stat_for_NCTHW(feature)
    def compute_stat_for_NCTHW(self, output):
        bz, c, t, h, w = output.size()
        if self.stat_type == 'temp':
            # todo compute the statistics along N and T
            #  the statistics are in shape  (C, H, W),
            self.batch_mean = output.mean((0, 2))  # (N, C, T, H, W) ->  (C, H, W)
            # temp_var = new_output.permute(1, 3, 4, 0, 2).contiguous().view([c, t, -1]).var(2, unbiased = False )
            self.batch_var = output.permute(1, 3, 4, 0, 2).contiguous().view([c, h, w, -1]).var(-1, unbiased=False)  # (N, C, T, H, W)  ->  (C, H, W, N, T) -> (C, H, W )
        elif self.stat_type == 'temp_v2':
            output = output.mean((3,4))  # (N, C, T, H, W) -> (N, C, T)
            self.batch_mean = output.mean((0,2)) # (N, C, T) -> (C,)
            self.batch_var = output.permute(1, 0, 2).contiguous().view([c, -1]).var(1, unbiased = False) # (N, C, T) -> (C, N, T) -> (C, )
        elif self.stat_type == 'spatiotemp':
            self.batch_mean = output.mean((0, 2, 3, 4))  # (N, C, T, H, W) ->  (C, )
            # batch_var = input[0].permute(1, 0, 2, 3, 4).contiguous().view([nch, -1]).var(1,  unbiased=False)  # compute the variance along each channel
            self.batch_var = output.permute(1, 0, 2, 3, 4).contiguous().view([c, -1]).var(1, unbiased=False)  # (N, C, T, H, W)  ->  (C, N, T, H, W) -> (C, )
        elif self.stat_type == 'spatial':
            self.batch_mean = output.mean((0, 3, 4))  # (N, C, T, H, W) ->  (C, T)
            self.batch_var = output.permute(1, 2, 0, 3, 4).contiguous().view([c, t, -1]).var(-1, unbiased=False)  # (N, C, T, H, W)  ->  (C, T, N, H, W ) -> (C, T )

    def close(self):
        self.hook.remove()

class CombineNormStatsRegHook_onereg():
    """
    Combine regularization of several types of statistics
    todo if there are multiple views, compute the statistics on the volume of multiple views , and align statistics with the source statistics,  only one regularization
    """
    def __init__(self, module, clip_len = None,
                 spatiotemp_stats_clean_tuple = None,
                 reg_type='mse_loss', moving_avg = None, momentum=0.1, stat_type_list = None, reduce_dim = True,before_norm = None ,

                 if_sample_tta_aug_views = None, n_augmented_views = None, ):

        self.hook = module.register_forward_hook(self.hook_fn)  # register a hook func to a module
        self.clip_len = clip_len
        # self.temp_mean_clean, self.temp_var_clean = temp_stats_clean_tuple

        self.reg_type = reg_type
        self.moving_avg = moving_avg
        self.momentum = momentum
        self.stat_type_list = stat_type_list
        self.reduce_dim = reduce_dim
        self.before_norm = before_norm
        self.if_sample_tta_aug_views = if_sample_tta_aug_views
        self.n_augmented_views = n_augmented_views
        # self.running_manner = running_manner
        # self.use_src_stat_in_reg = use_src_stat_in_reg  # whether to use the source statistics in regularization loss
        # todo keep the initial module.running_xx.data (the statistics of source model)
        #   if BN layer is not set to eval,  these statistics will change

        assert self.stat_type_list == ['spatiotemp']

        # self.source_mean_temp, self.source_var_temp = temp_stats_clean_tuple
        # self.source_mean_spatial, self.source_var_spatial = spatial_stats_clean_tuple
        self.source_mean_spatiotemp, self.source_var_spatiotemp = spatiotemp_stats_clean_tuple

        # if self.source_mean_temp is not None:
        #     self.source_mean_temp, self.source_var_temp = torch.tensor( self.source_mean_temp).cuda(), torch.tensor(self.source_var_temp).cuda()
        # if self.source_mean_spatial is not None:  # todo for BatchNorm1d layer,  there are no spatial or spatiotemporal statistics
        #     self.source_mean_spatial, self.source_var_spatial = torch.tensor( self.source_mean_spatial).cuda(), torch.tensor(self.source_var_spatial).cuda()
        self.device = torch.device("cuda:0")
        if self.source_mean_spatiotemp is not None:
            # self.source_mean_spatiotemp, self.source_var_spatiotemp = torch.tensor( self.source_mean_spatiotemp).cuda(), torch.tensor(self.source_var_spatiotemp).cuda()
            self.source_mean_spatiotemp, self.source_var_spatiotemp = torch.tensor( self.source_mean_spatiotemp).to(self.device), torch.tensor(self.source_var_spatiotemp).to(self.device)

        if self.moving_avg:
            if 'spatiotemp' in self.stat_type_list:
                self.mean_avgmeter_spatiotemp, self.var_avgmeter_spatiotemp = MovingAverageTensor(momentum=self.momentum), MovingAverageTensor(momentum=self.momentum)
        else:
            if 'spatiotemp' in self.stat_type_list:
                self.mean_avgmeter_spatiotemp, self.var_avgmeter_spatiotemp = AverageMeterTensor(), AverageMeterTensor()

    def hook_fn(self, module, input, output):
        feature = input[0] if self.before_norm else output
        # self.r_feature = torch.tensor(0).float().cuda()
        self.r_feature = torch.tensor(0).float().to(self.device)

        if isinstance(module, nn.BatchNorm1d): # todo  on BatchNorm1d, only temporal statistics regularization
            # output is in shape (N, C, T)  or   (N*C, T )
            # raise NotImplementedError('Statistics computation for nn.BatchNorm1d not implemented! ')
            # assert self.stat_type_list == 'temp'
            if 'temp' in self.stat_type_list or 'temp_v2' in self.stat_type_list:
                if self.if_sample_tta_aug_views:
                    raise NotImplementedError('temporal statistics for regularization of multiple augmented views not implemented! ')
                else:
                    if len(feature.size()) == 2:
                        nc, t = feature.size()
                        batch_mean_temp = feature.mean(0) # (N*C, T) -> (T, )
                        batch_var_temp = feature.permute(1, 0).contiguous().var(1, unbiased = False) # (N*C, T) -> (T, NC ) -> (T, )
                        bz = nc
                        self.feature_shape = (nc, t)
                    elif len(feature.size()) == 3:
                        bz, c, t = feature.size()
                        batch_mean_temp = feature.mean((0, 2)) # (N, C, T) -> (C, )
                        batch_var_temp = feature.permute(1, 0, 2).contiguous().view([c, -1]).var(1, unbiased = False) # (N, C, T) -> (C, N, T) -> (C, )
                        self.feature_shape = (bz, c, t)
                    if self.moving_avg:
                        self.mean_avgmeter_temp.update(batch_mean_temp )
                        self.var_avgmeter_temp.update(batch_var_temp )
                    else:
                        self.mean_avgmeter_temp.update(batch_mean_temp, n= bz)
                        self.var_avgmeter_temp.update(batch_var_temp, n= bz)
                    self.r_feature = self.r_feature + compute_regularization(self.source_mean_temp, self.mean_avgmeter_temp.avg, self.source_var_temp, self.var_avgmeter_temp.avg, self.reg_type)

        elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d): #todo on BatchNorm2d and Batchnorm3d, all types of statistics
            if self.if_sample_tta_aug_views:
                # todo   (actual_bz * n_temporal_clips  *   clip_len,  C, 256, 256)
                if isinstance(module, nn.BatchNorm2d):
                    nmt, c, h, w = feature.size()
                    t = self.clip_len
                    m = self.n_augmented_views
                    bz = nmt // (m * t)
                    feature = feature.view(bz*m, t, c, h ,w ).permute(0, 2, 1, 3, 4).contiguous()  # ( N*M*T,  C,  H, W) -> (N*M, T, C, H, W) ->  (N*M,  C, T, H, W)
                    # feature = feature.view(bz, m, t, c, h, w).permute(0, 1, 3, 2, 4, 5).contiguous() # (N*M, T, C, H, W) -> (N, M, T, C, H, W) -> (N, M, C, T, H, W)
                elif isinstance(module, nn.BatchNorm3d):
                    nm, c, t, h, w = feature.size()
                    m = self.n_augmented_views
                    bz = nm // m
                    # feature = feature.view(bz, m, c, t, h, w)
                else:
                    raise Exception(f'undefined module {module}')
                self.feature_shape = (bz*m,   c, t, h, w)
                # self.compute_reg_for_NMCTHW(feature)
                self.compute_reg_for_NCTHW(feature)
            else:
                if isinstance(module, nn.BatchNorm2d):
                    # output is in shape (N*T,  C,  H,  W)
                    nt, c, h, w = feature.size()
                    t = self.clip_len
                    bz = nt // t
                    feature = feature.view(bz, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous() # # ( N*T,  C,  H, W) -> (N, C, T, H, W)
                elif isinstance(module, nn.BatchNorm3d):
                    # output is in shape (N, C, T, H, W)
                    bz, c, t, h, w = feature.size()
                else:
                    raise Exception(f'undefined module {module}')
                self.feature_shape = (bz, c, t, h, w)

                self.compute_reg_for_NCTHW(feature)


        elif isinstance(module, nn.LayerNorm):
            if self.if_sample_tta_aug_views:
                nm, t, h, w, c = feature.size()
                m = self.n_augmented_views
                bz = nm // m
                feature = feature.permute(0, 4, 1, 2, 3).contiguous() # nm, t, h, w, c -> nm, c,  t, h, w,
                # feature = feature.view(bz, m, t, h, w, c).permute(0, 1, 5, 2,3,4).contiguous()
                # self.compute_reg_for_NMCTHW(feature)
                self.compute_reg_for_NCTHW(feature)
            else:
                assert len(feature.size()) == 5
                bz, t, h, w, c = feature.size()
                feature = feature.permute(0, 4, 1, 2, 3).contiguous()  # bz, t, h, w, c ->  bz, c, t, h, w
                self.feature_shape = (bz, c, t, h, w)
                self.compute_reg_for_NCTHW(feature)

    def compute_reg_for_NCTHW(self, output):
        bz, c, t, h, w = output.size()

        if 'spatiotemp' in self.stat_type_list:
            batch_mean_spatiotemp = output.mean((0, 2, 3, 4))  # (N, C, T, H, W) ->  (C, )
            batch_var_spatiotemp = output.permute(1, 0, 2, 3, 4).contiguous().view([c, -1]).var(1, unbiased=False)  # (N, C, T, H, W)  ->  (C, N, T, H, W) -> (C, )
            if self.moving_avg:
                self.mean_avgmeter_spatiotemp.update(batch_mean_spatiotemp)
                self.var_avgmeter_spatiotemp.update(batch_var_spatiotemp)
            else:
                self.mean_avgmeter_spatiotemp.update(batch_mean_spatiotemp, n=bz)
                self.var_avgmeter_spatiotemp.update(batch_var_spatiotemp, n=bz)
            self.r_feature = self.r_feature + compute_regularization(self.source_mean_spatiotemp,
                                                                     self.mean_avgmeter_spatiotemp.avg,
                                                                     self.source_var_spatiotemp,
                                                                     self.var_avgmeter_spatiotemp.avg, self.reg_type)
    def add_hook_back(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)  # register a hook func to a module

    def close(self):
        self.hook.remove()

class CombineNormStatsRegHook():
    """
    Combine regularization of several types of statistics
    todo if there are multiple views, compute the statistics on each view, and align statistics of each view with the source statistics, sum up / average the reguarlizations
    """
    def __init__(self, module, clip_len = None,
                 temp_stats_clean_tuple = None, spatial_stats_clean_tuple = None, spatiotemp_stats_clean_tuple = None,
                 reg_type='mse_loss', moving_avg = None, momentum=0.1, stat_type_list = None, reduce_dim = True,before_norm = None ,

                 if_sample_tta_aug_views = None, n_augmented_views = None, ):

        self.hook = module.register_forward_hook(self.hook_fn)  # register a hook func to a module
        self.clip_len = clip_len
        # self.temp_mean_clean, self.temp_var_clean = temp_stats_clean_tuple

        self.reg_type = reg_type
        self.moving_avg = moving_avg
        self.momentum = momentum
        self.stat_type_list = stat_type_list
        self.reduce_dim = reduce_dim
        self.before_norm = before_norm
        self.if_sample_tta_aug_views = if_sample_tta_aug_views
        self.n_augmented_views = n_augmented_views
        # self.running_manner = running_manner
        # self.use_src_stat_in_reg = use_src_stat_in_reg  # whether to use the source statistics in regularization loss
        # todo keep the initial module.running_xx.data (the statistics of source model)
        #   if BN layer is not set to eval,  these statistics will change

        self.source_mean_temp, self.source_var_temp = temp_stats_clean_tuple
        self.source_mean_spatial, self.source_var_spatial = spatial_stats_clean_tuple
        self.source_mean_spatiotemp, self.source_var_spatiotemp = spatiotemp_stats_clean_tuple

        if self.source_mean_temp is not None:
            self.source_mean_temp, self.source_var_temp = torch.tensor( self.source_mean_temp).cuda(), torch.tensor(self.source_var_temp).cuda()
        if self.source_mean_spatial is not None:  # todo for BatchNorm1d layer,  there are no spatial or spatiotemporal statistics
            self.source_mean_spatial, self.source_var_spatial = torch.tensor( self.source_mean_spatial).cuda(), torch.tensor(self.source_var_spatial).cuda()
        if self.source_mean_spatiotemp is not None:
            self.source_mean_spatiotemp, self.source_var_spatiotemp = torch.tensor( self.source_mean_spatiotemp).cuda(), torch.tensor(self.source_var_spatiotemp).cuda()

        if self.reduce_dim:
            if self.source_mean_temp is not None:
                if len(self.source_mean_temp.size()) == 3:
                    self.source_mean_temp = self.source_mean_temp.mean((1,2)) # (C, H, W) -> (C, )
                    self.source_var_temp = self.source_var_temp.mean((1,2)) # (C, H, W) -> (C, )
            if self.source_mean_spatial is not None:
                self.source_mean_spatial = self.source_mean_spatial.mean(1) # (C, T) -> (C, )
                self.source_var_spatial = self.source_var_spatial.mean(1) # (C, T) -> (C, )


        if self.moving_avg:
            if self.if_sample_tta_aug_views:
                if 'temp' in self.stat_type_list or 'temp_v2' in self.stat_type_list:
                    self.mean_avgmeter_temp_list = [ MovingAverageTensor(momentum=self.momentum) ] * self.n_augmented_views
                    self.var_avgmeter_temp_list = [MovingAverageTensor(momentum= self.momentum )] * self.n_augmented_views
                if 'spatial' in self.stat_type_list:
                    self.mean_avgmeter_spatial_list = [MovingAverageTensor(momentum=self.momentum)] * self.n_augmented_views
                    self.var_avgmeter_spatial_list = [MovingAverageTensor(momentum=self.momentum)] * self.n_augmented_views
                if 'spatiotemp' in self.stat_type_list:
                    self.mean_avgmeter_spatiotemp_list = [MovingAverageTensor(momentum=self.momentum)] * self.n_augmented_views
                    self.var_avgmeter_spatiotemp_list = [MovingAverageTensor(momentum=self.momentum)] * self.n_augmented_views
            else:
                if 'temp' in self.stat_type_list  or  'temp_v2' in self.stat_type_list:
                    self.mean_avgmeter_temp, self.var_avgmeter_temp = MovingAverageTensor(momentum=self.momentum), MovingAverageTensor(momentum=self.momentum)
                if 'spatial' in self.stat_type_list:
                    self.mean_avgmeter_spatial, self.var_avgmeter_spatial = MovingAverageTensor(momentum=self.momentum), MovingAverageTensor(momentum=self.momentum)
                if 'spatiotemp' in self.stat_type_list:
                    self.mean_avgmeter_spatiotemp, self.var_avgmeter_spatiotemp = MovingAverageTensor(momentum=self.momentum), MovingAverageTensor(momentum=self.momentum)

        else:
            if self.if_sample_tta_aug_views:
                if 'temp' in self.stat_type_list or 'temp_v2' in self.stat_type_list:
                    self.mean_avgmeter_temp_list = [ AverageMeterTensor() ] * self.n_augmented_views
                    self.var_avgmeter_temp_list = [AverageMeterTensor()] * self.n_augmented_views
                if 'spatial' in self.stat_type_list:
                    self.mean_avgmeter_spatial_list = [AverageMeterTensor()] * self.n_augmented_views
                    self.var_avgmeter_spatial_list = [AverageMeterTensor()] * self.n_augmented_views
                if 'spatiotemp' in self.stat_type_list:
                    self.mean_avgmeter_spatiotemp_list = [AverageMeterTensor()] * self.n_augmented_views
                    self.var_avgmeter_spatiotemp_list = [AverageMeterTensor()] * self.n_augmented_views
            else:
                if 'temp' in self.stat_type_list  or 'temp_v2' in self.stat_type_list:
                    self.mean_avgmeter_temp, self.var_avgmeter_temp = AverageMeterTensor(), AverageMeterTensor()
                if 'spatial' in self.stat_type_list:
                    self.mean_avgmeter_spatial, self.var_avgmeter_spatial = AverageMeterTensor(), AverageMeterTensor()
                if 'spatiotemp' in self.stat_type_list:
                    self.mean_avgmeter_spatiotemp, self.var_avgmeter_spatiotemp = AverageMeterTensor(), AverageMeterTensor()



    def hook_fn(self, module, input, output):
        feature = input[0] if self.before_norm else output
        self.r_feature = torch.tensor(0).float().cuda()

        if isinstance(module, nn.BatchNorm1d): # todo  on BatchNorm1d, only temporal statistics regularization
            # output is in shape (N, C, T)  or   (N*C, T )
            # raise NotImplementedError('Statistics computation for nn.BatchNorm1d not implemented! ')
            # assert self.stat_type_list == 'temp'
            if 'temp' in self.stat_type_list or 'temp_v2' in self.stat_type_list:
                if self.if_sample_tta_aug_views:
                    raise NotImplementedError('temporal statistics for regularization of multiple augmented views not implemented! ')
                else:
                    if len(feature.size()) == 2:
                        nc, t = feature.size()
                        batch_mean_temp = feature.mean(0) # (N*C, T) -> (T, )
                        batch_var_temp = feature.permute(1, 0).contiguous().var(1, unbiased = False) # (N*C, T) -> (T, NC ) -> (T, )
                        bz = nc
                        self.feature_shape = (nc, t)
                    elif len(feature.size()) == 3:
                        bz, c, t = feature.size()
                        batch_mean_temp = feature.mean((0, 2)) # (N, C, T) -> (C, )
                        batch_var_temp = feature.permute(1, 0, 2).contiguous().view([c, -1]).var(1, unbiased = False) # (N, C, T) -> (C, N, T) -> (C, )
                        self.feature_shape = (bz, c, t)
                    if self.moving_avg:
                        self.mean_avgmeter_temp.update(batch_mean_temp )
                        self.var_avgmeter_temp.update(batch_var_temp )
                    else:
                        self.mean_avgmeter_temp.update(batch_mean_temp, n= bz)
                        self.var_avgmeter_temp.update(batch_var_temp, n= bz)
                    self.r_feature = self.r_feature + compute_regularization(self.source_mean_temp, self.mean_avgmeter_temp.avg, self.source_var_temp, self.var_avgmeter_temp.avg, self.reg_type)

        elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d): #todo on BatchNorm2d and Batchnorm3d, all types of statistics
            if self.if_sample_tta_aug_views:
                # todo   (actual_bz * n_temporal_clips  *   clip_len,  C, 256, 256)
                if isinstance(module, nn.BatchNorm2d):
                    nmt, c, h, w = feature.size()
                    t = self.clip_len
                    m = self.n_augmented_views
                    bz = nmt // (m * t)
                    feature = feature.view(bz*m, t, c, h ,w )  # ( N*M*T,  C,  H, W) -> (N*M, T, C, H, W)
                    feature = feature.view(bz, m, t, c, h, w).permute(0, 1, 3, 2, 4, 5).contiguous() # (N*M, T, C, H, W) -> (N, M, T, C, H, W) -> (N, M, C, T, H, W)
                elif isinstance(module, nn.BatchNorm3d):
                    nm, c, t, h, w = feature.size()
                    m = self.n_augmented_views
                    bz = nm // m
                    feature = feature.view(bz, m, c, t, h, w)
                else:
                    raise Exception(f'undefined module {module}')
                self.feature_shape = (bz, m,  c, t, h, w)
                self.compute_reg_for_NMCTHW(feature)
            else:
                if isinstance(module, nn.BatchNorm2d):
                    # output is in shape (N*T,  C,  H,  W)
                    nt, c, h, w = feature.size()
                    t = self.clip_len
                    bz = nt // t
                    feature = feature.view(bz, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous() # # ( N*T,  C,  H, W) -> (N, C, T, H, W)
                elif isinstance(module, nn.BatchNorm3d):
                    # output is in shape (N, C, T, H, W)
                    bz, c, t, h, w = feature.size()
                else:
                    raise Exception(f'undefined module {module}')
                self.feature_shape = (bz, c, t, h, w)

                self.compute_reg_for_NCTHW(feature)


        elif isinstance(module, nn.LayerNorm):
            if self.if_sample_tta_aug_views:
                nm, t, h, w, c = feature.size()
                m = self.n_augmented_views
                bz = nm // m
                feature = feature.view(bz, m, t, h, w, c).permute(0, 1, 5, 2,3,4).contiguous()
                self.compute_reg_for_NMCTHW(feature)
            else:
                assert len(feature.size()) == 5
                bz, t, h, w, c = feature.size()
                feature = feature.permute(0, 4, 1, 2, 3).contiguous()  # bz, t, h, w, c ->  bz, c, t, h, w
                self.feature_shape = (bz, c, t, h, w)
                self.compute_reg_for_NCTHW(feature)

    def compute_reg_for_NMCTHW(self, output):
        # todo M is the number of augmented views
        bz, m, c, t, h, w = output.size()
        if 'temp' in self.stat_type_list or 'temp_v2' in self.stat_type_list:
            raise Exception('regularization of temporal statistics not implemented')
        if 'spatial' in self.stat_type_list:
            raise Exception('regularization of temporal statistics not implemented')
        if 'spatiotemp' in self.stat_type_list:
            batch_mean_spatiotemp = output.mean((0, 3, 4, 5)) # (N, M, C, T, H, W) -> (M, C)
            batch_var_spatiotemp = output.permute(1, 2, 0, 3, 4, 5).contiguous().view([m, c, -1]).var(2, unbiased = False)  #  (N, M, C, T, H, W) -> ( M, C, N, T, H, W ) -> (M, C)
            if self.moving_avg:
                for idx in range(self.n_augmented_views):
                    self.mean_avgmeter_spatiotemp_list[idx].update(batch_mean_spatiotemp[idx, :])
                    self.var_avgmeter_spatiotemp_list[idx].update(batch_var_spatiotemp[idx, :])
            else:
                for idx in range(self.n_augmented_views):
                    self.mean_avgmeter_spatiotemp_list[idx].update(batch_mean_spatiotemp[idx, :], n=bz)
                    self.var_avgmeter_spatiotemp_list[idx].update(batch_var_spatiotemp[idx, :], n= bz)
            reg_sum = torch.tensor(0).float().cuda()
            for idx in range(self.n_augmented_views):
                reg_sum = reg_sum + compute_regularization(self.source_mean_spatiotemp, self.mean_avgmeter_spatiotemp_list[idx].avg,
                                                                         self.source_var_spatiotemp, self.var_avgmeter_spatiotemp_list[idx].avg,  self.reg_type)
            reg_sum = reg_sum / self.n_augmented_views
            self.r_feature = self.r_feature + reg_sum


    def compute_reg_for_NCTHW(self, output):
        bz, c, t, h, w = output.size()

        if 'temp' in self.stat_type_list:
            if self.reduce_dim:
                batch_mean_temp = output.mean((0, 2, 3, 4))  # (N, C, T, H, W)-> (C,)
                batch_var_temp = output.permute(1, 3, 4, 0, 2).contiguous().view([c, h, w, -1]).var(-1, unbiased=False).mean( (1, 2))  # (N, C, T, H, W)-> (C, H, W, N, T) -> (C, H, W )->(C,)
            else:
                batch_mean_temp = output.mean((0, 2))  # (N, C, T, H, W) ->  (C, H, W)
                batch_var_temp = output.permute(1, 3, 4, 0, 2).contiguous().view([c, h, w, -1]).var(-1, unbiased=False)  # (N, C, T, H, W)  ->  (C, H, W, N, T) -> (C, H, W )
            if self.moving_avg:
                self.mean_avgmeter_temp.update(batch_mean_temp)
                self.var_avgmeter_temp.update(batch_var_temp)
            else:
                self.mean_avgmeter_temp.update(batch_mean_temp, n=bz)
                self.var_avgmeter_temp.update(batch_var_temp, n=bz)
            self.r_feature = self.r_feature + compute_regularization(self.source_mean_temp,
                                                                     self.mean_avgmeter_temp.avg,
                                                                     self.source_var_temp,
                                                                     self.var_avgmeter_temp.avg, self.reg_type)
        if 'temp_v2' in self.stat_type_list:
            output = output.mean((3, 4))  # (N, C, T, H, W) -> (N, C, T)
            batch_mean_temp = output.mean((0, 2))  # (N, C, T) -> (C,)
            batch_var_temp = output.permute(1, 0, 2).contiguous().view([c, -1]).var(1,  unbiased=False)  # (N, C, T) -> (C, N, T) -> (C, )
            if self.moving_avg:
                self.mean_avgmeter_temp.update(batch_mean_temp)
                self.var_avgmeter_temp.update(batch_var_temp)
            else:
                self.mean_avgmeter_temp.update(batch_mean_temp, n=bz)
                self.var_avgmeter_temp.update(batch_var_temp, n=bz)
            self.r_feature = self.r_feature + compute_regularization(self.source_mean_temp,
                                                                     self.mean_avgmeter_temp.avg,
                                                                     self.source_var_temp,
                                                                     self.var_avgmeter_temp.avg, self.reg_type)


        if 'spatiotemp' in self.stat_type_list:
            batch_mean_spatiotemp = output.mean((0, 2, 3, 4))  # (N, C, T, H, W) ->  (C, )
            batch_var_spatiotemp = output.permute(1, 0, 2, 3, 4).contiguous().view([c, -1]).var(1, unbiased=False)  # (N, C, T, H, W)  ->  (C, N, T, H, W) -> (C, )
            if self.moving_avg:
                self.mean_avgmeter_spatiotemp.update(batch_mean_spatiotemp)
                self.var_avgmeter_spatiotemp.update(batch_var_spatiotemp)
            else:
                self.mean_avgmeter_spatiotemp.update(batch_mean_spatiotemp, n=bz)
                self.var_avgmeter_spatiotemp.update(batch_var_spatiotemp, n=bz)
            self.r_feature = self.r_feature + compute_regularization(self.source_mean_spatiotemp,
                                                                     self.mean_avgmeter_spatiotemp.avg,
                                                                     self.source_var_spatiotemp,
                                                                     self.var_avgmeter_spatiotemp.avg, self.reg_type)

        if 'spatial' in self.stat_type_list:
            if self.reduce_dim:
                batch_mean_spatial = output.mean((0, 2, 3, 4))  # (N, C, T, H, W) ->  (C, )
                batch_var_spatial = output.permute(1, 2, 0, 3, 4).contiguous().view([c, t, -1]).var(-1,
                                                                                                    unbiased=False).mean(
                    1)  # (N, C, T, H, W)  ->  (C, T, N, H, W ) -> (C, T ) -> (C, )
            else:
                batch_mean_spatial = output.mean((0, 3, 4))  # (N, C, T, H, W) ->  (C, T)
                batch_var_spatial = output.permute(1, 2, 0, 3, 4).contiguous().view([c, t, -1]).var(-1,
                                                                                                    unbiased=False)  # (N, C, T, H, W)  ->  (C, T, N, H, W ) -> (C, T )
            if self.moving_avg:
                self.mean_avgmeter_spatial.update(batch_mean_spatial)
                self.var_avgmeter_spatial.update(batch_var_spatial)
            else:
                self.mean_avgmeter_spatial.update(batch_mean_spatial, n=bz)
                self.var_avgmeter_spatial.update(batch_var_spatial, n=bz)
            self.r_feature = self.r_feature + compute_regularization(self.source_mean_spatial,
                                                                     self.mean_avgmeter_spatial.avg,
                                                                     self.source_var_spatial,
                                                                     self.var_avgmeter_spatial.avg, self.reg_type)

    def close(self):
        self.hook.remove()


def compute_regularization(mean_true, mean_pred, var_true, var_pred, reg_type):
    # device = torch.device("cuda:0")
    # mean_true = mean_true.to(device)
    mean_pred = mean_pred.to(mean_true.device)
    # var_true = var_true.to(device)
    var_pred = var_pred.to(var_true.device)
    if reg_type == 'mse_loss':
        return mse_loss(var_true, var_pred) + mse_loss(mean_true, mean_pred)
    elif reg_type == 'l1_loss':
        return l1_loss(var_true, var_pred) + l1_loss(mean_true, mean_pred)
    elif reg_type == 'kld':
        return compute_kld(mean_true, mean_pred, var_true, var_pred)


class NormStatsRegHook():
    """
    Regularization of one type of statistics
    todo to be deprecated
    """
    def __init__(self, module, clip_len = None, stats_clean_tuple = None, reg_type='mse_loss', moving_avg = None, momentum=0.1, stat_type = None, reduce_dim = True):
        raise NotImplementedError('args.stat_type of str  is deprecated, use list instead. To add the implementation for case of Video swin transformer. ')

        self.hook = module.register_forward_hook(self.hook_fn)  # register a hook func to a module
        self.clip_len = clip_len
        # self.temp_mean_clean, self.temp_var_clean = temp_stats_clean_tuple

        self.reg_type = reg_type
        self.moving_avg = moving_avg
        self.momentum = momentum
        self.stat_type = stat_type
        self.reduce_dim = reduce_dim
        # self.running_manner = running_manner
        # self.use_src_stat_in_reg = use_src_stat_in_reg  # whether to use the source statistics in regularization loss
        # todo keep the initial module.running_xx.data (the statistics of source model)
        #   if BN layer is not set to eval,  these statistics will change

        self.source_mean, self.source_var = stats_clean_tuple

        self.source_mean = torch.tensor(self.source_mean).cuda()
        self.source_var = torch.tensor(self.source_var).cuda()
        if self.stat_type == 'temp':
            if self.reduce_dim and len(self.source_mean.size())==3 :
                self.source_mean = self.source_mean.mean((1,2)) # (C, H, W) -> (C, )
                self.source_var = self.source_var.mean((1,2 )) # (C, H, W) -> (C, )
        elif self.stat_type == 'spatial':
            if self.reduce_dim:
                self.source_mean = self.source_mean.mean(1) # (C, T) -> (C, )
                self.source_var = self.source_var.mean(1) # (C, T) -> (C, )

        if self.moving_avg:
            self.mean_avgmeter = MovingAverageTensor(momentum=self.momentum)
            self.var_avgmeter = MovingAverageTensor(momentum=self.momentum)
        else:
            self.mean_avgmeter = AverageMeterTensor()
            self.var_avgmeter = AverageMeterTensor()

    def hook_fn(self, module, input, output):

        if isinstance(module, nn.BatchNorm1d):
            # output is in shape (N, C, T)  or   (N*C, T )
            # raise NotImplementedError('Statistics computation for nn.BatchNorm1d not implemented! ')
            assert self.stat_type == 'temp'
            if len(output.size()) == 2:
                nc, t = output.size()
                batch_mean = output.mean(0) # (N*C, T) -> (T, )
                batch_var = output.permute(1,0).contiguous().var(1, unbiased = False) # (N*C, T) -> (T, NC ) -> (T, )
                bz = nc
            elif len(output.size()) == 3:
                bz, c, t = output.size()
                batch_mean = output.mean( (0, 2)) # (N, C, T) -> (C, )
                batch_var = output.permute(1, 0, 2).contiguous().view([c, -1]).var(1, unbiased = False) # (N, C, T) -> (C, N, T) -> (C, )
        elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
            if isinstance(module, nn.BatchNorm2d):
                # output is in shape (N*T,  C,  H,  W)
                nt, c, h, w = output.size()
                t = self.clip_len
                bz = nt // t
                output = output.view(bz, t, c, h, w).permute(0, 2, 1, 3,  4).contiguous() # # ( N*T,  C,  H, W) -> (N, C, T, H, W)
            elif isinstance(module, nn.BatchNorm3d):
                # output is in shape (N, C, T, H, W)
                bz, c, t, h, w = output.size()
                output = output
            else:
                raise Exception(f'undefined module {module}')

            # todo compute the batch statistics
            if self.stat_type == 'temp':
                if self.reduce_dim:
                    batch_mean = output.mean((0, 2, 3, 4))  # (N, C, T, H, W)-> (C,)
                    batch_var = output.permute(1, 3, 4, 0, 2).contiguous().view([c, h, w, -1]).var(-1, unbiased=False).mean((1,2))  # (N, C, T, H, W)-> (C, H, W, N, T) -> (C, H, W )->(C,)
                else:
                    batch_mean = output.mean((0, 2))  # (N, C, T, H, W) ->  (C, H, W)
                    batch_var  = output.permute(1, 3, 4, 0, 2).contiguous().view([c, h, w, -1]).var(-1,unbiased=False)  # (N, C, T, H, W)  ->  (C, H, W, N, T) -> (C, H, W )
            elif self.stat_type == 'spatiotemp':
                batch_mean = output.mean((0, 2, 3, 4))  # (N, C, T, H, W) ->  (C, )
                batch_var = output.permute(1, 0, 2, 3, 4).contiguous().view([c, -1]).var(1, unbiased=False)  # (N, C, T, H, W)  ->  (C, N, T, H, W) -> (C, )
            elif self.stat_type == 'spatial':
                if self.reduce_dim:
                    batch_mean = output.mean((0, 2, 3, 4))  # (N, C, T, H, W) ->  (C, )
                    batch_var = output.permute(1, 2, 0, 3, 4).contiguous().view([c, t, -1]).var(-1, unbiased=False).mean(1)  # (N, C, T, H, W)  ->  (C, T, N, H, W ) -> (C, T ) -> (C, )
                else:
                    batch_mean = output.mean((0, 3, 4))  # (N, C, T, H, W) ->  (C, T)
                    batch_var = output.permute(1, 2, 0, 3, 4).contiguous().view([c, t,  -1]).var(-1,  unbiased=False)  # (N, C, T, H, W)  ->  (C, T, N, H, W ) -> (C, T )

        if self.moving_avg:
            self.mean_avgmeter.update(batch_mean)
            self.var_avgmeter.update(batch_var)
        else:
            self.mean_avgmeter.update(batch_mean, n= bz)
            self.var_avgmeter.update(batch_var, n= bz)

        if self.reg_type == 'mse_loss':
            # # todo sum of squared difference,  averaged over  h * w
            # self.r_feature = torch.sum(( self.source_var - self.var_avgmeter.avg )**2 ) / spatial_dim + torch.sum(( self.source_mean - self.mean_avgmeter.avg )**2 ) / spatial_dim
            # self.r_feature = torch.norm(self.source_var - self.var_avgmeter.avg, 2) + torch.norm(self.source_mean - self.mean_avgmeter.avg, 2)
            self.r_feature = mse_loss(self.source_var, self.var_avgmeter.avg) + mse_loss(self.source_mean, self.mean_avgmeter.avg)
        elif self.reg_type == 'l1_loss':
            self.r_feature = l1_loss(self.source_var, self.var_avgmeter.avg) + l1_loss(self.source_mean, self.mean_avgmeter.avg)
        else:
            raise NotImplementedError

    def close(self):
        self.hook.remove()