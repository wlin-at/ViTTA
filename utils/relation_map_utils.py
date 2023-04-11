

import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.utils_ import AverageMeter, AverageMeterTensor, MovingAverageTensor
from einops import rearrange


l1_loss = nn.L1Loss(reduction='mean')
mse_loss = nn.MSELoss(reduction='mean')

def compute_exp_norm_relation_map(batch_sym_matrix):
    # todo compute the expoential values in a relation map, normalized by dividing the sum of each row/column
    # batch_sym_matrix in shape (N, T, T)
    exp_sym_matrix = torch.exp(batch_sym_matrix) # (N,T,T)
    sym_dim = exp_sym_matrix.size(2)
    return torch.div(exp_sym_matrix,  torch.sum(exp_sym_matrix, 2).expand(-1,  sym_dim )   )  # (N, T, T)  divided by (N,T,T)

def get_upper_traingle_idx_pair(t):
    idx_list1 = []
    idx_list2 = []

    for value in range(0, t-1):  # value is from 0 to  (t-2)
        n_duplicates = t-1 - value
        idx_list1 +=  [value] * n_duplicates

    for start in range(1, t): # start value from 1 to  t-1
        idx_list2 +=  list(range(start, t))
    return idx_list1, idx_list2

def compute_upper_triangle_similarity(feature):

    N, n_elements, dim = feature.size()
    # if n_elements > 1000:
    #     return None
    # else:
        # todo feature is in shape (N, n_elements, dim)
        #    compute the similarity for n_elements * (n_elements-1)/2 pairs  of elements
        #    the two vectos should be both in shape (N,  n_elements * (n_elements-1)/2,  dim  )
    idx_list1, idx_list2 = get_upper_traingle_idx_pair(
        n_elements)  # len of these lists are   n_elements * (n_elements-1)/2
    return F.cosine_similarity(feature[:, idx_list1, :], feature[:, idx_list2, :],  dim=-1)  # todo (N,  n_elements * (n_elements-1)/2 )




class ComputeRelationMapHook():
    def __init__(self, module, clip_len = None, stat_type = None, before_norm = None, batch_size = None):
        self.hook = module.register_forward_hook(self.hook_fn)  # register a hook func to a module
        self.clip_len = clip_len
        self.stat_type = stat_type
        self.before_norm = before_norm
        self.batch_size = batch_size
    def hook_fn(self, module, input, output):

        feature = input[0] if self.before_norm else output

        if isinstance(module, nn.BatchNorm1d):
            assert self.stat_type in ['temp', 'channel']  #  temporal relation map
            if len(feature.size()) == 2:
                # todo  we do not compute relation map for  feature in shape (nc, t )
                nc, t = feature.size()
                self.relation_map = None
            elif len(feature.size()) == 3:
                n, c, t = feature.size()
                self.compute_relation_map_for_NCT(feature)
        elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
            if isinstance(module, nn.BatchNorm2d):
                nt, c, h, w = feature.size()
                t = self.clip_len
                bz = nt // t
                feature = feature.view(bz, t, c, h, w).permute(0, 2, 1, 3,  4).contiguous()  # # ( N*T,  C,  H, W) -> (N, C, T, H, W)
            elif isinstance(module, nn.BatchNorm3d):
                # output is in shape (N, C, T, H, W)
                bz, c, t, h, w = feature.size()
            self.compute_relation_map_for_NCTHW(feature)
        elif isinstance(module, nn.LayerNorm):
            assert len(feature.size()) == 5
            bz, t, h, w, c = feature.size()
            feature = rearrange(feature, 'b t h w c -> b c t h w')
            self.compute_relation_map_for_NCTHW(feature)
        else:
            raise Exception(f'undefined module {module}')
    def compute_relation_map_for_NCT(self, output):

        bz, c, t  = output.size()
        if self.stat_type == 'temp':
            output = rearrange(output, 'n c t -> n t c')
        # elif self.stat_type == 'channel':
        #     output = rearrange(output, '')
        output = torch.matmul(output, torch.transpose(output, 1, 2))
        output = compute_exp_norm_relation_map(output)
        self.relation_map = output.mean(0)
    def compute_relation_map_for_NCTHW(self, output):
        bz, c, t, h, w = output.size()
        if self.stat_type == 'temp':
            # output = output.permute(0, 2, 1, 3, 4).contiguous().view([bz, t, -1]) # (N,C,T,H,W) -> (N,T,C,H,W) -> (N,T,CHW)
            output = rearrange(output, 'n c t h w -> n t (c h w)')
        elif self.stat_type == 'spatial':
            # output = output.view([bz, c, t, -1])   # (N,C,T,H,W) -> (N, C, T, HW) ->
            # output = rearrange(output, 'n c t h w -> n (c t) (h w)')  # (N,C,T,H,W) -> (N, CT, HW) ->
            output  = rearrange(output, 'n c t h w -> n (h w) (c t)')  # (N,C,T,H,W)  ->  # (N, HW, CT)
        elif self.stat_type == 'spatiotemp':
            output = rearrange(output, 'n c t h w -> n (t h w) c')
        elif self.stat_type == 'channel':
            output = rearrange(output, 'n c t h w -> n c (t h w)')
        else:
            raise Exception(f'undefined stat type {self.stat_type}')
        # todo batched matrix multiplication   batched matrix x batched matrix
        output = torch.matmul(output, torch.transpose(output, 1, 2))  # (N, HW, CT) ->  (N, HW, HW)
        output = compute_exp_norm_relation_map(output)
        self.relation_map = output.mean(0)  # (N, HW, HW) ->  ( HW, HW)


class ComputePairwiseSimilarityHook():
    def __init__(self, module, clip_len = None, stat_type = None, before_norm = None, batch_size = None):
        self.hook = module.register_forward_hook(self.hook_fn)  # register a hook func to a module
        self.clip_len = clip_len
        self.stat_type = stat_type
        self.before_norm = before_norm
        self.batch_size = batch_size
    def hook_fn(self, module, input, output):

        feature = input[0] if self.before_norm else output

        if isinstance(module, nn.BatchNorm1d):
            assert self.stat_type in ['temp', 'channel']  #  temporal relation map
            if len(feature.size()) == 2:
                # todo  we do not compute relation map for  feature in shape (nc, t )
                nc, t = feature.size()
                self.sim_vec = None
            elif len(feature.size()) == 3:
                n, c, t = feature.size()
                self.compute_sim_for_NCT(feature)
        elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
            if isinstance(module, nn.BatchNorm2d):
                nt, c, h, w = feature.size()
                t = self.clip_len
                bz = nt // t
                feature = feature.view(bz, t, c, h, w).permute(0, 2, 1, 3,  4).contiguous()  # # ( N*T,  C,  H, W) -> (N, C, T, H, W)
            elif isinstance(module, nn.BatchNorm3d):
                # output is in shape (N, C, T, H, W)
                bz, c, t, h, w = feature.size()
            self.compute_sim_for_NCTHW(feature)
        elif isinstance(module, nn.LayerNorm):
            assert len(feature.size()) == 5
            bz, t, h, w, c = feature.size()
            feature = rearrange(feature, 'b t h w c -> b c t h w')
            self.compute_sim_for_NCTHW(feature)
        else:
            raise Exception(f'undefined module {module}')
    def compute_sim_for_NCT(self, output):
        bz, c, t  = output.size()
        if self.stat_type == 'temp':
            output = rearrange(output, 'n c t -> n t c')
        # elif self.stat_type == 'channel':
        #     output = rearrange(output, '')
        # output = F.cosine_similarity(output[..., None, :, :], output[..., :, None, :], dim=-1) # (N, T, T ) or (N, C, C )
        output = compute_upper_triangle_similarity(output) # (N,  n_elements * (n_elements-1)/2 )

        self.sim_vec = None if output is None else output.mean(0)
    def compute_sim_for_NCTHW(self, output):
        bz, c, t, h, w = output.size()
        if self.stat_type == 'temp':
            # output = output.permute(0, 2, 1, 3, 4).contiguous().view([bz, t, -1]) # (N,C,T,H,W) -> (N,T,C,H,W) -> (N,T,CHW)
            output = rearrange(output, 'n c t h w -> n t (c h w)')
        elif self.stat_type == 'spatial':
            # output = output.view([bz, c, t, -1])   # (N,C,T,H,W) -> (N, C, T, HW) ->
            # output = rearrange(output, 'n c t h w -> n (c t) (h w)')  # (N,C,T,H,W) -> (N, CT, HW) ->
            # todo pca reduce the HW dimension to be same as T dimension
            output  = rearrange(output, 'n c t h w -> (n c t) (h w)')  # (N,C,T,H,W)  ->  # (N, HW, CT)
            output, _, _ = torch.pca_lowrank(output, q= t  )
            output = rearrange(output,'(n ct) hw -> n hw ct' , n= bz)

        elif self.stat_type == 'spatiotemp':
            output = rearrange(output, 'n c t h w -> n (t h w) c')
        elif self.stat_type == 'channel':
            output = rearrange(output, 'n c t h w -> n c (t h w)')
        else:
            raise Exception(f'undefined stat type {self.stat_type}')
        # output = F.cosine_similarity(output[..., None, :, :], output[..., :, None, :], dim=-1)  # (N, T, T ) or (N, C, C )
        output = compute_upper_triangle_similarity(output)  # (N,  n_elements * (n_elements-1)/2 )
        self.sim_vec = None if output is None else output.mean(0)

class CombineCossimRegHook():
    """
    Combine regularization of several types of statistics
    """
    def __init__(self, module, clip_len = None,
                 temp_cossim = None,
                 reg_type='mse_loss', moving_avg = None, momentum=0.1, stat_type_list = None, before_norm = None):

        self.hook = module.register_forward_hook(self.hook_fn)  # register a hook func to a module
        self.clip_len = clip_len
        # self.temp_mean_clean, self.temp_var_clean = temp_stats_clean_tuple

        self.reg_type = reg_type
        self.moving_avg = moving_avg
        self.momentum = momentum
        self.stat_type_list = stat_type_list
        # self.reduce_dim = reduce_dim
        self.before_norm = before_norm
        # self.running_manner = running_manner
        # self.use_src_stat_in_reg = use_src_stat_in_reg  # whether to use the source statistics in regularization loss
        # todo keep the initial module.running_xx.data (the statistics of source model)
        #   if BN layer is not set to eval,  these statistics will change

        self.temp_cossim = temp_cossim
        if self.temp_cossim is not None:
            # todo for some BatchNorm1d layer, there is no cossim computed
            self.temp_cossim = torch.tensor(self.temp_cossim).cuda()
        # self.source_mean_spatial, self.source_var_spatial = spatial_stats_clean_tuple
        # self.source_mean_spatiotemp, self.source_var_spatiotemp = spatiotemp_stats_clean_tuple

        # self.source_mean = torch.tensor(self.source_mean).cuda()
        # self.source_var = torch.tensor(self.source_var).cuda()

        # self.source_mean_temp, self.source_var_temp = torch.tensor( self.source_mean_temp).cuda(), torch.tensor(self.source_var_temp).cuda()
        # if self.source_mean_spatial is not None:  # todo for BatchNorm1d layer,  there are no spatial or spatiotemporal statistics
        #     self.source_mean_spatial, self.source_var_spatial = torch.tensor( self.source_mean_spatial).cuda(), torch.tensor(self.source_var_spatial).cuda()
        #     self.source_mean_spatiotemp, self.source_var_spatiotemp = torch.tensor( self.source_mean_spatiotemp).cuda(), torch.tensor(self.source_var_spatiotemp).cuda()

        # if self.reduce_dim:
        #     if len(self.source_mean_temp.size()) == 3:
        #         self.source_mean_temp = self.source_mean_temp.mean((1,2)) # (C, H, W) -> (C, )
        #         self.source_var_temp = self.source_var_temp.mean((1,2)) # (C, H, W) -> (C, )
        #     if self.source_mean_spatial is not None:
        #         self.source_mean_spatial = self.source_mean_spatial.mean(1) # (C, T) -> (C, )
        #         self.source_var_spatial = self.source_var_spatial.mean(1) # (C, T) -> (C, )


        if self.moving_avg:
            if 'temp' in self.stat_type_list  or  'temp_v2' in self.stat_type_list:
                self.cossim_avgmeter_temp= MovingAverageTensor(momentum=self.momentum)
            # if 'spatial' in self.stat_type_list:
            #     self.mean_avgmeter_spatial, self.var_avgmeter_spatial = MovingAverageTensor(momentum=self.momentum), MovingAverageTensor(momentum=self.momentum)
            # if 'spatiotemp' in self.stat_type_list:
            #     self.mean_avgmeter_spatiotemp, self.var_avgmeter_spatiotemp = MovingAverageTensor(momentum=self.momentum), MovingAverageTensor(momentum=self.momentum)
        else:
            if 'temp' in self.stat_type_list  or 'temp_v2' in self.stat_type_list:
                self.cossim_avgmeter_temp = AverageMeterTensor()
            # if 'spatial' in self.stat_type_list:
            #     self.mean_avgmeter_spatial, self.var_avgmeter_spatial = AverageMeterTensor(), AverageMeterTensor()
            # if 'spatiotemp' in self.stat_type_list:
            #     self.mean_avgmeter_spatiotemp, self.var_avgmeter_spatiotemp = AverageMeterTensor(), AverageMeterTensor()



    def hook_fn(self, module, input, output):
        feature = input[0] if self.before_norm else output
        self.r_feature = torch.tensor(0).float().cuda()

        if isinstance(module, nn.BatchNorm1d): # todo  on BatchNorm1d, only temporal statistics regularization
            # output is in shape (N, C, T)  or   (N*C, T )
            # raise NotImplementedError('Statistics computation for nn.BatchNorm1d not implemented! ')
            # assert self.stat_type_list == 'temp'
            if 'temp' in self.stat_type_list :
                if len(feature.size()) == 2:
                    pass
                elif len(feature.size()) == 3:
                    bz, c, t = feature.size()
                    feature = rearrange(feature, 'n c t -> n t c')
                    # batch_mean_temp = feature.mean((0, 2)) # (N, C, T) -> (C, )
                    # batch_var_temp = feature.permute(1, 0, 2).contiguous().view([c, -1]).var(1, unbiased = False) # (N, C, T) -> (C, N, T) -> (C, )
                    output = compute_upper_triangle_similarity(feature).mean(0)  # mean of cosine imilarities on a batch
                    # self.feature_shape = (bz, c, t)
                    if self.moving_avg:
                        self.cossim_avgmeter_temp.update(output)
                        # self.var_avgmeter_temp.update(batch_var_temp )
                    else:
                        self.cossim_avgmeter_temp.update(output, n= bz)
                        # self.var_avgmeter_temp.update(batch_var_temp, n= bz)
                    self.r_feature = self.r_feature + compute_regularization(cossim_pred= self.cossim_avgmeter_temp.avg, cossim_true=self.temp_cossim, reg_type= self.reg_type)

        elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d): #todo on BatchNorm2d and Batchnorm3d, all types of statistics
            if isinstance(module, nn.BatchNorm2d):
                # output is in shape (N*T,  C,  H,  W)
                nt, c, h, w = feature.size()
                t = self.clip_len
                bz = nt // t
                feature = feature.view(bz, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous() # # ( N*T,  C,  H, W) -> (N, C, T, H, W)
            elif isinstance(module, nn.BatchNorm3d):
                # output is in shape (N, C, T, H, W)
                bz, c, t, h, w = feature.size()
                feature = feature
            else:
                raise Exception(f'undefined module {module}')
            self.feature_shape = (bz, c, t, h, w)

            self.compute_reg_for_NCTHW(feature)


        elif isinstance(module, nn.LayerNorm):
            assert len(feature.size()) == 5
            bz, t, h, w, c = feature.size()
            feature = feature.permute(0, 4, 1, 2, 3).contiguous()  # bz, t, h, w, c ->  bz, c, t, h, w
            self.feature_shape = (bz, c, t, h, w)
            self.compute_reg_for_NCTHW(feature)

    def compute_reg_for_NCTHW(self, output):
        bz, c, t, h, w = output.size()
        # if self.stat_type_list == 'temp':
        if 'temp' in self.stat_type_list:
            output = rearrange(output, 'n c t h w -> n t (c h w)')
            output = compute_upper_triangle_similarity(output).mean(0)
            if self.moving_avg:
                self.cossim_avgmeter_temp.update(output)
            else:
                self.cossim_avgmeter_temp.update(output, n=bz)
            self.r_feature = self.r_feature + compute_regularization(cossim_pred=self.cossim_avgmeter_temp.avg,
                                                                     cossim_true=self.temp_cossim,
                                                                     reg_type=self.reg_type)


        # elif self.stat_type_list == 'spatiotemp':
        if 'spatiotemp' in self.stat_type_list:
            pass
        # elif self.stat_type_list == 'spatial':
        if 'spatial' in self.stat_type_list:
            pass

    def close(self):
        self.hook.remove()

def compute_regularization(cossim_pred, cossim_true, reg_type):
    if reg_type == 'mse_loss':
        return mse_loss(cossim_pred, cossim_true)
    elif reg_type == 'l1_loss':
        return l1_loss(cossim_pred, cossim_true)
    elif reg_type == 'kld':
        return compute_kld(mean_true, mean_pred, var_true, var_pred)