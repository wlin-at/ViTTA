
import torch.nn as nn
from models.videoswintransformer_models.swin_transformer import SwinTransformer3D
from models.videoswintransformer_models.i3d_head import I3DHead
import torch.nn.functional as F

# <class 'tuple'>: (2, 4, 4)
#  <class 'list'>: [2, 2, 18, 2]
# <class 'list'>: [4, 8, 16, 32]
# <class 'tuple'>: (8, 7, 7)
# todo swin base
# model = dict(
#     type='Recognizer3D',
#     backbone=dict(
#         type='SwinTransformer3D',
#         patch_size=(4,4,4),
#         embed_dim=128,
#         depths=[2, 2, 18, 2],
#         num_heads=[4, 8, 16, 32]),
#         window_size=(8,7,7),
#         mlp_ratio=4.,
#         qkv_bias=True,
#         qk_scale=None,
#         drop_rate=0.,
#         attn_drop_rate=0.,
#         drop_path_rate=0.2,
#         patch_norm=True),
#     cls_head=dict(
#         type='I3DHead',
#         in_channels=1024,
#         num_classes=400,
#         spatial_type='avg',
#         dropout_ratio=0.5),
#     test_cfg = dict(average_clips='prob'))

# todo swin_base_patch244_window1677_sthv2.py
# model=dict(backbone=dict(patch_size=(2,4,4), window_size=(16,7,7), drop_path_rate=0.4),
#            cls_head=dict(num_classes=174),
#            test_cfg=dict(max_testing_views=2),
#            train_cfg=dict(blending=dict(type='LabelSmoothing', num_classes=174, smoothing=0.1)))

# todo swin_base_patch244_window877_kinetics400_1k.py
# model=dict(backbone=dict(patch_size=(2,4,4), drop_path_rate=0.3), test_cfg=dict(max_testing_views=4))

class Recognizer3D(nn.Module):
    def __init__(self, num_classes = None, patch_size = None, window_size = None, drop_path_rate = None, ):
        super(Recognizer3D, self).__init__()
        # backbone params
        self.pretrained = None
        self.pretrained2d = True
        self.patch_size = patch_size
        self.in_chans = 3
        self.embed_dim = 128
        self.depths = [2, 2, 18, 2]
        self.num_heads = [4, 8, 16, 32]
        self.window_size = window_size
        self.mlp_ratio = 4.0
        self.qkv_bias = True
        self.qk_scale = None
        self.drop_rate = 0.
        self.attn_drop_rate = 0.
        self.drop_path_rate = drop_path_rate
        self.patch_norm = True

        # head params
        self.num_classes = num_classes
        self.in_channels = 1024
        self.spatial_type = 'avg'
        self.dropout_ratio = 0.5

        self.score_type = 'score'  #

        self.backbone = SwinTransformer3D(pretrained= self.pretrained,
                          pretrained2d= self.pretrained2d,
                          patch_size= self.patch_size,
                          in_chans= self.in_chans,
                          embed_dim= self.embed_dim,
                          depths= self.depths ,
                          num_heads= self.num_heads ,
                          window_size= self.window_size,
                          mlp_ratio= self.mlp_ratio,
                          qkv_bias= self.qkv_bias,
                          qk_scale= self.qk_scale,
                          drop_rate= self.drop_rate,
                          attn_drop_rate= self.attn_drop_rate,
                          drop_path_rate=  self.drop_path_rate,
                          patch_norm= self.patch_norm,
                          )
        self.cls_head = I3DHead(num_classes=self.num_classes,
                in_channels=self.in_channels,
                spatial_type=self.spatial_type,
                dropout_ratio=self.dropout_ratio
                )

    def forward(self, x): # x   (batch, n_views, C, T, H, W)
        n = x.shape[0]
        n_views = x.shape[1]  # n_views   n_spatial_crops * n_temporal clips
        x = x.reshape((-1,) + x.shape[2:])  # (N, n_views, C, T, H, W) ->  (N * n_views, C, T, H, W)
        feat = self.backbone(x)  # (N * n_views, C, T, H, W) ->  (N * n_views, C,  T/2,  H/32,  W/32)
        cls_score = self.cls_head(feat) # (N * n_views, 1024, 16, 7, 7) -> (N * n_views, 600)
        # cls_score = self.average_clips(cls_score, num_segs= n_views)  # todo  average over  n_views
        vid_cls_score, view_cls_score = self.average_clips(cls_score, num_segs= n_views)
        return vid_cls_score, view_cls_score


    def average_clips(self, cls_score, num_segs = 1):
        bz = cls_score.shape[0]
        cls_score = cls_score.view(bz // num_segs, num_segs, -1) #  (bz, n_views, n_class)
        if self.score_type == 'prob':
            cls_score = F.softmax(cls_score, dim=2).mean(dim=2)
            return cls_score
        elif self.score_type == 'score':

            vid_cls_score = cls_score.mean(dim=1)
            return vid_cls_score, cls_score


