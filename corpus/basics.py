


import time
# import os
from torch.nn.utils import clip_grad_norm
import torch.nn as nn
from einops import rearrange
# import os.path as osp
# from tensorboardX import SummaryWriter
# import torch.backends.cudnn as cudnn
from datasets_.dataset_deprecated import MyTSNDataset
from datasets_.video_dataset import MyTSNVideoDataset, MyVideoDataset


from models.r2plus1d import MyR2plus1d
from models import i3d
from models.i3d_incep import InceptionI3d
from models.tanet_models.tanet import TSN
from models.videoswintransformer_models.recognizer3d import Recognizer3D


from timm.models import create_model

from utils.transforms import *
from utils.utils_ import AverageMeter, accuracy,  get_augmentation
from utils.BNS_utils import BNFeatureHook, choose_layers
import baselines.tent as tent
import os.path as osp
from utils.pred_consistency_utils import compute_pred_consis
import copy as cp



def train(train_loader, model, criterion, optimizer, epoch, args=None, logger=None, writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):  # input (bz, n_clips,  3, clip_len, 224, 224), target (bz, )
        input = input.reshape(
            (-1,) + input.shape[2:])  # (batch, n_views, 3, T, 224,224 ) -> (batch * n_views, 3, T, 224,224 )
        target = target.reshape((target.shape[0], 1)).repeat(1, args.num_clips)
        target = target.reshape((-1,) + target.shape[2:])  # (batch * n_views, )

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        output = model(input)  # (batch * n_views, 3, T, 224,224 ) ->  (batch * n_views, n_class)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                logger.debug("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose:
            if i % args.print_freq == 0:
                logger.debug(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))
    writer.add_scalars('loss', {'losses': losses.avg}, global_step=epoch)
    writer.add_scalars('acc', {'train_acc': top1.avg}, global_step=epoch)
    writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], global_step=epoch)


def validate(val_loader, model, criterion, iter, epoch=None, args=None, logger=None, writer=None, optimizer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode

    pred_concat = []
    gt_concat = []
    if args.arch == 'tanet':
        n_clips = int(args.sample_style.split("-")[-1])
    elif args.arch == 'videoswintransformer':
        n_clips = args.num_clips

    if args.evaluate_baselines:
        if args.baseline == 'source':
            logger.debug(f'Starting ---- {args.corruptions} ---- evaluation for Source...')
        elif args.baseline == 'tent':
            from baselines.tent import forward_and_adapt
            logger.debug(f'Starting ---- {args.corruptions} ---- adaptation for TENT...')
            for i, (input, target) in enumerate(val_loader):
                actual_bz = input.shape[0]
                input = input.cuda()
                if args.arch == 'tanet':
                    input = input.view(-1, 3, input.size(2), input.size(3))
                    input = input.view(actual_bz * args.test_crops * n_clips,
                                       args.clip_length, 3, input.size(2), input.size(3))
                    _ = forward_and_adapt(input, model, optimizer, args, actual_bz, n_clips)
                else:
                    input = input.reshape((-1,) + input.shape[2:])
                    _ = forward_and_adapt(input, model, optimizer)
            logger.debug(f'TENT Adaptation Finished --- Now Evaluating')
        elif args.baseline == 'norm':
            logger.debug(f'Starting ---- {args.corruptions} ---- adaptation for NORM...')
            with torch.no_grad():
                for i, (input, target) in enumerate(val_loader):
                    actual_bz = input.shape[0]
                    input = input.cuda()
                    if args.arch == 'tanet':
                        input = input.view(-1, 3, input.size(2), input.size(3))
                        input = input.view(actual_bz * args.test_crops * n_clips,
                                           args.clip_length, 3, input.size(2), input.size(3))
                        _ = model(input)
                    else:
                        input = input.reshape((-1,) + input.shape[2:])
                        _ = model(input)
            logger.debug(f'NORM Adaptation Finished --- Now Evaluating')
        elif args.baseline == 'shot':
            logger.debug(f'Starting ---- {args.corruptions} ---- evaluation for SHOT...')
        elif args.baseline == 'dua':
            logger.debug(f'Starting ---- {args.corruptions} ---- evaluation for DUA...')

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):  #
            model.eval()
            actual_bz = input.shape[0]
            input = input.cuda()
            target = target.cuda()
            if args.arch == 'tanet':
                # (actual_bz,    C* spatial_crops * temporal_clips* clip_len, 256, 256) ->   (actual_bz * spatial_crops * temporal_clips* clip_len,  C, 256, 256)
                input = input.view(-1, 3, input.size(2), input.size(3))
                input = input.view(actual_bz * args.test_crops * n_clips,
                                       args.clip_length, 3, input.size(2),input.size(3))  # (actual_bz * spatial_crops * temporal_clips* clip_len,  C, 256, 256) -> (actual_bz * spatial_crops * temporal_clips,  clip_len,  C, 256, 256)

                output = model(input) #  (actual_bz * spatial_crops * temporal_clips,         clip_len,  C, 256, 256)   ->     (actual_bz * spatial_crops * temporal_clips,       n_class )
                # take the average among all spatial_crops * temporal_clips,   (actual_bz * spatial_crops * temporal_clips,       n_class )  ->   (actual_bz,       n_class )
                output = output.reshape(actual_bz, args.test_crops * n_clips, -1).mean(1)
            elif args.arch == 'videoswintransformer':
                # the format shape is N C T H W
                # (actual_bz,   C* spatial_crops * temporal_clips* clip_len,    256,     256)   -> (batch, n_views, C, T, H, W)
                n_views = args.test_crops * n_clips
                # input = input.view(-1, n_views, 3, args.clip_length, input.size(3), input.size(4))
                output, _ = model(  input)  # (batch, n_views, C, T, H, W) ->  (batch, n_class), todo outputs are unnormalized scores



            else:
                input = input.reshape( (-1,) + input.shape[2:])  # (batch, n_views, 3, T, 224,224 ) -> (batch * n_views, 3, T, 224,224 )
                output = model( input)  # (batch * n_views, 3, T, 224,224 ) ->  (batch * n_views,  n_class)  todo  reshape clip prediction into video prediction

                output = torch.squeeze(output)
                output = rearrange(output, '(d0 d1) d2 -> d0 d1 d2', d0=actual_bz)  # (batch * n_views,  n_class) ->  (batch, n_views,  n_class)  todo  reshape clip prediction into video prediction
                output = torch.mean(output, dim=1)  # (batch, n_views,  n_class) ->  (batch,  n_class), take the average scores of multiple views

            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            _, preds = torch.max(output, 1)
            # pred_concat = np.concatenate([pred_concat, preds.detach().cpu().numpy()])
            # gt_concat = np.concatenate([gt_concat, target.detach().cpu().numpy()])

            losses.update(loss.item(), actual_bz)
            top1.update(prec1.item(), actual_bz)
            top5.update(prec5.item(), actual_bz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.verbose:
                if i % args.print_freq == 0:
                    logger.debug(('Test: [{0}/{1}]\t'
                                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5)))

    logger.debug(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'.format(top1=top1,
                                                                                                            top5=top5,
                                                                                                            loss=losses)))
    if writer is not None:
        writer.add_scalars('loss', {'val_loss': losses.avg}, global_step=epoch)
        writer.add_scalars('acc', {'val_acc': top1.avg}, global_step=epoch)
    logger.debug(f'Validation acc {top1.avg} ')

    # logger.debug(classification_report(pred_concat, gt_concat))

    return top1.avg


def compute_statistics(model = None, args=None, logger = None, log_time = None):
    # from utils.BNS_utils import ComputeTemporalStatisticsHook
    from utils.norm_stats_utils import ComputeNormStatsHook
    # todo candidate layers are conv layers
    # candidate_layers = [nn.Conv2d, nn.Conv3d]
    # chosen_conv_layers = choose_layers(model, candidate_layers)

    # candidate_layers = [nn.BatchNorm2d, nn.BatchNorm3d]
    compute_stat_hooks = []
    list_stat_mean = []
    list_stat_var = []
    if args.arch == 'tanet':
        if args.stat_type in ['temp', 'temp_v2']:
            # todo temporal statistics computed on all types of BN layers,  nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d
            candidate_layers = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
        elif args.stat_type  in ['spatial', 'spatiotemp'] :
            # todo spatial and spatiotemporal statistics computed only on   nn.BatchNorm2d, nn.BatchNorm3d
            candidate_layers = [ nn.BatchNorm2d, nn.BatchNorm3d]
        chosen_layers = choose_layers(model, candidate_layers)
    elif args.arch == 'videoswintransformer':
        # todo   on Video Swin Transformer,
        #     statistics are computed on all LayerNorm layers (feature in shape BTHWC), except for the first LayerNorm after Conv3D (feature in shape B,combined_dim,C)
        candidate_layers = [nn.LayerNorm]
        chosen_layers = choose_layers(model, candidate_layers)
        chosen_layers = chosen_layers[1:]

    for layer_id, (layer_name, layer_) in enumerate(chosen_layers):
        compute_stat_hooks.append( ComputeNormStatsHook(layer_, clip_len= args.clip_length, stat_type=args.stat_type, before_norm= args.before_norm, batch_size=args.batch_size))
        list_stat_mean.append(AverageMeter())
        list_stat_var.append(AverageMeter())

    if args.arch == 'tanet':
        n_clips = int(args.sample_style.split("-")[-1])
    elif args.arch == 'videoswintransformer':
        n_clips = args.num_clips
    if args.arch == 'tanet':
        data_loader = torch.utils.data.DataLoader(
            get_dataset_tanet(args,  split='val', dataset_type='eval'),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, )
    elif args.arch == 'videoswintransformer':
        data_loader = torch.utils.data.DataLoader(
            get_dataset_videoswin(args, split='val', dataset_type='eval'),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )
    else:
        # I3D
        data_loader = torch.utils.data.DataLoader(
            get_dataset(args, split='val'),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)


    model.eval()  # fix the BNS during computation todo notice that this is already done in setup_model()
    with torch.no_grad():
        for batch_id, (input, target) in enumerate(data_loader):
            actual_bz = input.shape[0]
            input = input.cuda()
            if args.arch == 'tanet':
                # (actual_bz, C* spatial_crops * temporal_clips* clip_len, 256, 256) ->   (actual_bz * spatial_crops * temporal_clips* clip_len,  C, 256, 256)
                input = input.view(-1, 3, input.size(2), input.size(3))
                input = input.view(actual_bz * args.test_crops * n_clips,
                                   args.clip_length, 3, input.size(2), input.size(3))  # (actual_bz * spatial_crops * temporal_clips* clip_len,  C, 256, 256) -> (actual_bz * spatial_crops * temporal_clips,  clip_len,  C, 256, 256)
                _ = model( input)  # (actual_bz * spatial_crops * temporal_clips,         clip_len,  C, 256, 256)   ->     (actual_bz * spatial_crops * temporal_clips,       n_class )
            elif args.arch == 'videoswintransformer':
                # the format shape is N C T H W
                # (actual_bz,   C* spatial_crops * temporal_clips* clip_len,    256,     256)   -> (batch, n_views, C, T, H, W)
                n_views = args.test_crops * n_clips
                # input = input.view(-1, n_views, 3, args.clip_length, input.size(3), input.size(4))
                _ = model( input)  # (batch, n_views, C, T, H, W) ->  (batch, n_class), todo outputs are unnormalized scores
            else:
                input = input.reshape( (-1,) + input.shape[2:])  # (batch, n_views, 3, T, 224,224 ) -> (batch * n_views, 3, T, 224,224 )
                # forward pass
                _ = model( input)  # (batch * n_views, 3, T, 224,224 ) ->  (batch * n_views,  n_class)  todo  reshape clip prediction into video prediction
            #
            if batch_id % 1000 == 0:
                print(f'{batch_id}/{len(data_loader)} batches completed ...')
            for hook_id, stat_hook in enumerate(compute_stat_hooks):
                list_stat_mean[hook_id].update(stat_hook.batch_mean, n= actual_bz)
                list_stat_var[hook_id].update(stat_hook.batch_var, n= actual_bz)

    for hook_id, stat_hook in enumerate(compute_stat_hooks):
        list_stat_mean[hook_id] = list_stat_mean[hook_id].avg.cpu().numpy()
        list_stat_var[hook_id] = list_stat_var[hook_id].avg.cpu().numpy()

    np.save( osp.join(args.result_dir,  f'list_{args.stat_type}_mean_{log_time}.npy'), list_stat_mean, allow_pickle=True)
    np.save( osp.join(args.result_dir, f'list_{args.stat_type}_var_{log_time}.npy'), list_stat_var, allow_pickle=True)

    # step1_img_ps = np.load(step1_img_ps_file, allow_pickle=True).item()

def compute_cos_similarity(model = None, args = None, log_time = None):
    from utils.relation_map_utils import ComputePairwiseSimilarityHook
    compute_stat_hooks = []
    list_cos_sim_mat = []
    if args.arch == 'tanet':
        candidate_layers = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
        # if args.stat_type in ['temp']:
        #
        # elif args.stat_type  in ['spatial', 'spatiotemp', 'channel'] :
        #     candidate_layers = [ nn.BatchNorm2d, nn.BatchNorm3d]
        chosen_layers = choose_layers(model, candidate_layers)
    elif args.arch == 'videoswintransformer':
        # todo   on Video Swin Transformer,
        #     statistics are computed on all LayerNorm layers (feature in shape BTHWC), except for the first LayerNorm after Conv3D (feature in shape B,combined_dim,C)
        candidate_layers = [nn.LayerNorm]
        chosen_layers = choose_layers(model, candidate_layers)
        chosen_layers = chosen_layers[1:]
    for layer_id, (layer_name, layer_) in enumerate(chosen_layers):
        # if args.stat_type in ['spatial', 'spatiotemp'] and layer_name in ['module.base_model.bn1']:
        #     compute_stat_hooks.append(None)
        #     list_cos_sim_mat.append(None)
        # else:
        if isinstance(layer_, nn.BatchNorm1d ) and args.stat_type  in ['spatial', 'spatiotemp', 'channel'] :
            compute_stat_hooks.append(None)
            list_cos_sim_mat.append(None)
        else:
            compute_stat_hooks.append( ComputePairwiseSimilarityHook(layer_, clip_len= args.clip_length, stat_type=args.stat_type, before_norm= args.before_norm, batch_size=args.batch_size))
            list_cos_sim_mat.append(AverageMeter())
    if args.arch == 'tanet':
        n_clips = int(args.sample_style.split("-")[-1])
    elif args.arch == 'videoswintransformer':
        n_clips = args.num_clips
    if args.arch == 'tanet':
        data_loader = torch.utils.data.DataLoader(
            get_dataset_tanet(args,  split='val'),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, )
    elif args.arch == 'videoswintransformer':
        data_loader = torch.utils.data.DataLoader(
            get_dataset_videoswin(args, split='val'),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )
    else:
        # I3D
        data_loader = torch.utils.data.DataLoader(
            get_dataset(args, split='val'),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    model.eval()  # fix the BNS during computation todo notice that this is already done in setup_model()
    with torch.no_grad():
        for batch_id, (input, target) in enumerate(data_loader):
            actual_bz = input.shape[0]
            input = input.cuda()
            if args.arch == 'tanet':
                # (actual_bz, C* spatial_crops * temporal_clips* clip_len, 256, 256) ->   (actual_bz * spatial_crops * temporal_clips* clip_len,  C, 256, 256)
                input = input.view(-1, 3, input.size(2), input.size(3))
                input = input.view(actual_bz * args.test_crops * n_clips,
                                   args.clip_length, 3, input.size(2), input.size(3))  # (actual_bz * spatial_crops * temporal_clips* clip_len,  C, 256, 256) -> (actual_bz * spatial_crops * temporal_clips,  clip_len,  C, 256, 256)
                _ = model( input)  # (actual_bz * spatial_crops * temporal_clips,         clip_len,  C, 256, 256)   ->     (actual_bz * spatial_crops * temporal_clips,       n_class )
            elif args.arch == 'videoswintransformer':
                # the format shape is N C T H W
                # (actual_bz,   C* spatial_crops * temporal_clips* clip_len,    256,     256)   -> (batch, n_views, C, T, H, W)
                n_views = args.test_crops * n_clips
                input = input.view(-1, n_views, 3, args.clip_length, input.size(3), input.size(4))
                _ = model( input)  # (batch, n_views, C, T, H, W) ->  (batch, n_class), todo outputs are unnormalized scores
            else: # todo I3D
                input = input.reshape( (-1,) + input.shape[2:])  # (batch, n_views, 3, T, 224,224 ) -> (batch * n_views, 3, T, 224,224 )
                # forward pass
                _ = model( input)  # (batch * n_views, 3, T, 224,224 ) ->  (batch * n_views,  n_class)  todo  reshape clip prediction into video prediction
            #
            if batch_id % 1000 == 0:
                print(f'{batch_id}/{len(data_loader)} batches completed ...')
            for hook_id, stat_hook in enumerate(compute_stat_hooks):
                get_hook_result = True
                if stat_hook is None:
                    get_hook_result = False
                else:
                    if stat_hook.sim_vec is None:
                        get_hook_result = False

                if not get_hook_result:
                    list_cos_sim_mat[hook_id] = None
                else:
                    list_cos_sim_mat[hook_id].update(stat_hook.sim_vec, n=actual_bz)

    for hook_id, stat_hook in enumerate(compute_stat_hooks):
        if list_cos_sim_mat[hook_id] is not None:
            list_cos_sim_mat[hook_id] = list_cos_sim_mat[hook_id].avg.cpu().numpy()

    np.save( osp.join(args.result_dir,  f'list_{args.stat_type}_relationmap_{log_time}.npy'), list_cos_sim_mat, allow_pickle=True)

def tta_standard(model_origin, criterion, args=None, logger = None, writer =None):
    """
    todo  tta_standard: during adaptation, overfit to one sample, and evaluate on this sample right after adaptation. re-initilaize the model when the next sample comes
        tta_online: during adaptation, one gradient step per sample, and evaluate on this sample right after adaptation. do not re-initiaize the model when the next sample comes
    :param model:
    :param criterion:
    :param args:
    :param logger:
    :param writer:
    :return:
    """
    if args.if_tta_standard == 'tta_standard':
        # todo  overfit to one sample,  do not accumulate the target statistics in each forward step
        #   do not accumulate the target statistics for one sample in multiple gradient steps (multiple forward pass)
        #   do not accumulate the target statistics between different samples
        assert args.momentum_mvg == 1.0
        assert args.n_epoch_adapat == 1
    elif args.if_tta_standard == 'tta_online':
        assert args.momentum_mvg != 1.0  # todo accumulate the target statistics for different samples
        assert args.n_gradient_steps == 1 # todo one gradient step per sample (on forward pass per sample )
        assert args.n_epoch_adapat == 1
    from utils.norm_stats_utils import CombineNormStatsRegHook_onereg
    # from utils.relation_map_utils import CombineCossimRegHook

    candidate_bn_layers = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]



    if args.arch == 'tanet':
        tta_loader = torch.utils.data.DataLoader(
            get_dataset_tanet(args,  split='val', dataset_type='tta'),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
        eval_loader = torch.utils.data.DataLoader(
            get_dataset_tanet(args, split='val', dataset_type='eval'),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )

    elif args.arch == 'videoswintransformer':
        tta_loader = torch.utils.data.DataLoader(
            get_dataset_videoswin(args,  split='val', dataset_type= 'tta'),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
        eval_loader = torch.utils.data.DataLoader(
            get_dataset_videoswin(args, split='val', dataset_type='eval'),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
    # global_iter = 0

    epoch_result_list = []
    if args.arch == 'tanet':
        n_clips = int(args.sample_style.split("-")[-1])
    elif args.arch == 'videoswintransformer':
        n_clips = args.num_clips
    if args.if_sample_tta_aug_views:
        assert n_clips == 1
        n_augmented_views = args.n_augmented_views
    if_pred_consistency = args.if_pred_consistency if args.if_sample_tta_aug_views else False

    batch_time = AverageMeter()
    losses_ce = AverageMeter()
    losses_reg = AverageMeter()
    losses_consis = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    pred_concat = []
    gt_concat = []
    end = time.time()
    eval_loader_iterator = iter(eval_loader)

    # todo ############################################################
    # todo ##################################### choose layers
    # todo ############################################################
    if args.stat_reg == 'mean_var':
        assert args.stat_type == ['spatiotemp']
        list_spatiotemp_mean_clean = list(np.load(args.spatiotemp_mean_clean_file, allow_pickle=True))
        list_spatiotemp_var_clean = list(np.load(args.spatiotemp_var_clean_file, allow_pickle=True))
        if args.arch == 'tanet':
            bn_layers = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
            chosen_layers = choose_layers(model_origin, bn_layers)

            list_spatiotemp_mean_clean_new, list_spatiotemp_var_clean_new = [], []
            counter = 0
            for layer_id, (chosen_layer_name, chosen_layer) in enumerate(chosen_layers):
                if isinstance(chosen_layer, nn.BatchNorm1d):
                    # at the position of Batchnorm1d, add None as placeholder in the list for spatial and spatiotemporal statistics
                    list_spatiotemp_mean_clean_new.append(None)
                    list_spatiotemp_var_clean_new.append(None)
                elif isinstance(chosen_layer, nn.BatchNorm2d) or isinstance(chosen_layer, nn.BatchNorm3d):
                    list_spatiotemp_mean_clean_new.append(list_spatiotemp_mean_clean[counter])
                    list_spatiotemp_var_clean_new.append(list_spatiotemp_var_clean[counter])
                    counter += 1

        elif args.arch == 'videoswintransformer':
            # todo   on Video Swin Transformer,
            #     statistics are computed on all LayerNorm layers (feature in shape BTHWC), except for the first LayerNorm after Conv3D (feature in shape B,combined_dim,C)
            candidate_layers = [nn.LayerNorm]
            chosen_layers = choose_layers(model_origin, candidate_layers)
            chosen_layers = chosen_layers[1:]

            list_spatiotemp_mean_clean_new, list_spatiotemp_var_clean_new = list_spatiotemp_mean_clean, list_spatiotemp_var_clean

        assert len(list_spatiotemp_mean_clean_new) == len(chosen_layers)

    if not hasattr(args, 'moving_avg'):
        args.moving_avg = False
    if not hasattr(args, 'momentum_mvg'):
        args.momentum_mvg = 0.1

    for batch_id, (input, target) in enumerate(tta_loader):  #

        setup_model_optimizer = False
        if args.if_tta_standard == 'tta_standard':
            setup_model_optimizer = True  #  setup model and optimizer before every sample comes
        elif args.if_tta_standard == 'tta_online':
            if batch_id == 0:
                setup_model_optimizer = True #  setup model and optimizer only before the first sample comes

        if setup_model_optimizer:
            print(f'Batch {batch_id}, initialize the model, update chosen layers, initialize hooks, intialize average meter')
            # todo ############################################################
            # todo #####################################  re-intialize the model, update chosen_layers from this new model
            # todo ############################################################
            model = cp.deepcopy(model_origin)
            # when we initialize the model, we have to re-choose the layers from it.
            if args.arch == 'tanet':
                # todo  temporal statistics are computed on  nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d
                #      spatial statistics are computed on nn.BatchNorm2d, nn.BatchNorm3d,   not on Batchnorm1d
                #      spatiotemporal statistics are computed on nn.BatchNorm2d, nn.BatchNorm3d,  not on Batchnorm1d
                bn_layers = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
                chosen_layers = choose_layers(model, bn_layers)
            elif args.arch == 'videoswintransformer':
                # todo   on Video Swin Transformer,
                #     statistics are computed on all LayerNorm layers (feature in shape BTHWC), except for the first LayerNorm after Conv3D (feature in shape B,combined_dim,C)
                candidate_layers = [nn.LayerNorm]
                chosen_layers = choose_layers(model, candidate_layers)
                chosen_layers = chosen_layers[1:]
            # todo ############################################################
            # todo ##################################### set up the optimizer
            # todo ############################################################
            if args.update_only_bn_affine:
                from utils.BNS_utils import freeze_except_bn, collect_bn_params
                if args.arch == 'tanet':
                    model = freeze_except_bn(model, bn_condidiate_layers=candidate_bn_layers)  # set only Batchnorm layers to trainable,   freeze all the other layers
                    params, param_names = collect_bn_params(model,  bn_candidate_layers=candidate_bn_layers)  # collecting gamma and beta in all Batchnorm layers
                elif args.arch == 'videoswintransformer':
                    model = freeze_except_bn(model,
                                             bn_condidiate_layers=[nn.LayerNorm])  # set only Batchnorm layers to trainable,   freeze all the other layers
                    params, param_names = collect_bn_params(model,
                                                            bn_candidate_layers=[nn.LayerNorm])  # collecting gamma and beta in all Batchnorm layers
                optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.)
            else:
                optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum,
                                            weight_decay=args.weight_decay)

            # todo ############################################################
            # todo ##################################### initialize hooks to the chosen layers for computing statistics, initialize average meter
            # todo ############################################################
            if args.stat_reg == 'mean_var':
                if isinstance(args.stat_type, str):
                    raise NotImplementedError(
                        'args.stat_type of str  is deprecated, use list instead. To add the implementation for case of Video swin transformer. ')
                elif isinstance(args.stat_type, list):
                    stat_reg_hooks = []
                    for layer_id, (chosen_layer_name, chosen_layer) in enumerate(chosen_layers):
                        for block_name in args.chosen_blocks:
                            if block_name in chosen_layer_name:
                                stat_reg_hooks.append(
                                    CombineNormStatsRegHook_onereg(chosen_layer, clip_len=args.clip_length, # todo load statistcs, initialize the average meter
                                                                   spatiotemp_stats_clean_tuple=(
                                                                       list_spatiotemp_mean_clean_new[layer_id],
                                                                       list_spatiotemp_var_clean_new[layer_id]),
                                                                   reg_type=args.reg_type,
                                                                   moving_avg=args.moving_avg,
                                                                   momentum=args.momentum_mvg,
                                                                   stat_type_list=args.stat_type,
                                                                   reduce_dim=args.reduce_dim,
                                                                   before_norm=args.before_norm,
                                                                   if_sample_tta_aug_views=args.if_sample_tta_aug_views,
                                                                   n_augmented_views=args.n_augmented_views))
                                break
            elif args.stat_reg == 'BNS':
                # todo  regularization on BNS statistics
                # regularization on BNS statistics
                # bns_feature_hooks = []
                stat_reg_hooks = []
                chosen_layers = choose_layers(model, candidate_bn_layers)
                # for chosen_layer in chosen_layers:
                for layer_id, (chosen_layer_name, chosen_layer) in enumerate(chosen_layers):
                    for block_name in args.chosen_blocks:
                        if block_name in chosen_layer_name:
                            # regularization between manually computed target batch statistics (whether or not in running manner) between source statistics
                            stat_reg_hooks.append(BNFeatureHook(chosen_layer, reg_type= args.reg_type, running_manner=args.running_manner,
                                              use_src_stat_in_reg=args.use_src_stat_in_reg, momentum=args.momentum_bns))
            else:
                raise Exception(f'undefined regularization type {args.stat_reg}')
        # todo ############################################################
        # todo ##################################### set the model to train mode,  freeze BN statistics
        # todo ############################################################
        model.train()  # BN layers are set to train mode
        if args.fix_BNS:  # fix the BNS during forward pass
            for m in model.modules():
                for candidate in candidate_bn_layers:
                    if isinstance(m, candidate):
                        m.eval()
        actual_bz = input.shape[0]
        input = input.cuda()
        target = target.cuda()
        # todo ############################################################
        # todo ##################################### reshape the input
        # todo ############################################################
        if args.arch == 'tanet':
            input = input.view(-1, 3, input.size(2), input.size(3))
            if args.if_sample_tta_aug_views:
                input = input.view(actual_bz * args.test_crops * n_augmented_views, args.clip_length, 3, input.size(2),  input.size(3))
            else:
                input = input.view(actual_bz * args.test_crops * n_clips, args.clip_length, 3, input.size(2), input.size(3))
        elif args.arch == 'videoswintransformer':
            pass
        else:
            raise NotImplementedError(f'Incorrect model type {args.arch}')
        # todo ############################################################
        # todo ##################################### train on one sample for multiple steps
        # todo ############################################################
        n_gradient_steps = args.n_gradient_steps
        for step_id in range(n_gradient_steps):
            if args.arch == 'tanet':
                output = model(input)
                if args.if_sample_tta_aug_views:
                    output = output.reshape(actual_bz, args.test_crops * n_augmented_views, -1)  # (N, n_views, n_class )
                    if if_pred_consistency:
                        loss_consis = compute_pred_consis(output)
                    output = output.mean(1)
                else:
                    output = output.reshape(actual_bz, args.test_crops * n_clips, -1).mean(1)

            elif args.arch == 'videoswintransformer':
                if args.if_sample_tta_aug_views:
                    n_views = args.test_crops * n_augmented_views
                else:
                    n_views = args.test_crops * n_clips
                # input = input.view(-1, n_views, 3, args.clip_length, input.size(3), input.size(4))
                if args.if_sample_tta_aug_views:
                    if if_pred_consistency:
                        output, view_cls_score = model( input)  # (batch, n_views, C, T, H, W) ->  (batch, n_class), todo outputs are unnormalized scores
                        loss_consis = compute_pred_consis(view_cls_score)
                else:
                    output, _ = model( input)
            else:
                raise NotImplementedError(f'Incorrect model type {args.arch}')
            loss_ce = criterion(output, target)
            loss_reg = torch.tensor(0).float().cuda()
            if args.stat_reg:
                for hook in stat_reg_hooks:
                    loss_reg += hook.r_feature.cuda()
            else:
                raise Exception(f'undefined regularization type {args.stat_reg}')
            if if_pred_consistency:
                loss = args.lambda_feature_reg*loss_reg + args.lambda_pred_consis * loss_consis
            else:
                loss = loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # global_iter += 1

        losses_ce.update(loss_ce.item(), actual_bz)
        losses_reg.update(loss_reg.item(), actual_bz)
        if if_pred_consistency:
            losses_consis.update(loss_consis.item(), actual_bz)

        # todo ############################################################
        # todo ##################################### remove all the hooks, no computation of statistics during inference
        # todo ############################################################
        if args.stat_reg:
            for stat_reg_hook in stat_reg_hooks:
                stat_reg_hook.close()
        else:
            raise Exception(f'undefined regularization type {args.stat_reg}')

        # todo ##########################################################################################
        # todo ################### Inference on the same batch ##############################################
        # todo ##########################################################################################
        with torch.no_grad():
            model.eval()
            input, target = next(eval_loader_iterator)
            input, target = input.cuda(), target.cuda()
            if args.arch == 'tanet':
                # (actual_bz, C* spatial_crops * temporal_clips* clip_len, 256, 256) ->   (actual_bz * spatial_crops * temporal_clips* clip_len,  C, 256, 256)
                input = input.view(-1, 3, input.size(2), input.size(3))
                input = input.view(actual_bz * args.test_crops * n_clips,
                                   args.clip_length, 3, input.size(2), input.size(3))  # (actual_bz * spatial_crops * temporal_clips* clip_len,  C, 256, 256) -> (actual_bz * spatial_crops * temporal_clips,  clip_len,  C, 256, 256)
                output = model( input)  # (actual_bz * spatial_crops * temporal_clips,         clip_len,  C, 256, 256)   ->     (actual_bz * spatial_crops * temporal_clips,       n_class )
                # take the average among all spatial_crops * temporal_clips,   (actual_bz * spatial_crops * temporal_clips,       n_class )  ->   (actual_bz,       n_class )
                output = output.reshape(actual_bz, args.test_crops * n_clips, -1).mean(1)
            elif args.arch == 'videoswintransformer':
                # the format shape is N C T H W         if  collapse in datsaet is True, then shape is  (actual_bz,   C* spatial_crops * temporal_clips* clip_len,    256,     256)
                # (batch, n_views, C, T, H, W)
                n_views = args.test_crops * n_clips
                # input = input.view(-1, n_views, 3, args.clip_length, input.size(3), input.size(4))
                output, _ = model( input)  # (batch, n_views, C, T, H, W) ->  (batch, n_class), todo outputs are unnormalized scores
            else:
                raise NotImplementedError(f'Incorrect model type {args.arch}')
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            top1.update(prec1.item(), actual_bz)
            top5.update(prec5.item(), actual_bz)

            batch_time.update(time.time() - end)
            end = time.time()

        # todo ##########################################################################################
        # todo ################### In the case of tta_online, after inference, add the hooks back  ##############################################
        # todo ##########################################################################################
        if args.if_tta_standard == 'tta_online':
            hook_layer_counter = 0
            for layer_id, (chosen_layer_name, chosen_layer) in enumerate(chosen_layers):
                for block_name in args.chosen_blocks:
                    if block_name in chosen_layer_name:
                        stat_reg_hooks[hook_layer_counter].add_hook_back(chosen_layer)
                        hook_layer_counter += 1
            assert hook_layer_counter == len(stat_reg_hooks)

        if args.verbose:
            logger.debug(('TTA Epoch{epoch}: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss reg {loss_reg.val:.4f} ({loss_reg.avg:.4f})\t'
                          'Loss consis {loss_consis.val:.4f} ({loss_consis.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                batch_id, len(tta_loader), epoch=1, batch_time=batch_time, loss_reg=losses_reg, loss_consis = losses_consis,
                top1=top1, top5=top5)))

    epoch_result_list.append(top1.avg)

    # model_path = osp.join(  args.result_dir, f'{args.corruptions}.model' )
    # print(f'Saving models to {model_path}')
    #
    # torch.save( model.state_dict(), model_path )

    return epoch_result_list

def load_precomputed_statistics(args, n_layers):

    list_temp_mean_clean = list(np.load(args.temp_mean_clean_file, allow_pickle=True)) if 'temp' in args.stat_type or 'temp_v2' in args.stat_type else [None]* n_layers
    list_temp_var_clean = list(np.load(args.temp_var_clean_file, allow_pickle=True)) if 'temp' in args.stat_type or 'temp_v2' in args.stat_type else [None]* n_layers
    list_spatiotemp_mean_clean = list(np.load(args.spatiotemp_mean_clean_file, allow_pickle=True)) if 'spatiotemp' in args.stat_type else [None]* n_layers
    list_spatiotemp_var_clean = list(np.load(args.spatiotemp_var_clean_file, allow_pickle=True)) if 'spatiotemp' in args.stat_type else [None]* n_layers
    list_spatial_mean_clean = list(np.load(args.spatial_mean_clean_file, allow_pickle=True)) if 'spatial' in args.stat_type else [None]* n_layers
    list_spatial_var_clean = list(np.load(args.spatial_var_clean_file, allow_pickle=True)) if 'spatial' in args.stat_type else [None]* n_layers
    return list_temp_mean_clean, list_temp_var_clean, list_spatiotemp_mean_clean, list_spatiotemp_var_clean, list_spatial_mean_clean, list_spatial_var_clean


def test_time_adapt(model, criterion, args=None, logger=None, writer=None):
    # test time adaptation for several epochs
    # from utils.norm_stats_utils import  CombineNormStatsRegHook
    from utils.norm_stats_utils import  CombineNormStatsRegHook_onereg
    from utils.relation_map_utils import CombineCossimRegHook
    candidate_bn_layers = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]

    if args.update_only_bn_affine:
        from utils.BNS_utils import freeze_except_bn, collect_bn_params
        model = freeze_except_bn(model, bn_condidiate_layers=candidate_bn_layers) # set only Batchnorm layers to trainable,   freeze all the other layers
        params, param_names = collect_bn_params(model, bn_candidate_layers=candidate_bn_layers) #  collecting gamma and beta in all Batchnorm layers
        optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.)
    else:
        optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)


    if args.arch == 'tanet':
        tta_loader = torch.utils.data.DataLoader(
            get_dataset_tanet(args,  split='val', dataset_type='tta'),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )

        eval_loader = torch.utils.data.DataLoader(
            get_dataset_tanet(args,  split='val', dataset_type='eval'),
            batch_size=args.batch_size_eval, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
    elif args.arch == 'videoswintransformer':
        tta_loader = torch.utils.data.DataLoader(
            get_dataset_videoswin(args,  split='val', dataset_type= 'tta'),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
        eval_loader = torch.utils.data.DataLoader(
            get_dataset_videoswin(args, split='val', dataset_type='eval'),
            batch_size=args.batch_size_eval, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
    else:
        tta_loader = torch.utils.data.DataLoader(
            get_dataset(args, split='val'),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )

        eval_loader = torch.utils.data.DataLoader(
            get_dataset(args, split='val'),
            batch_size=args.batch_size_eval, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
    global_iter = 0

    if args.stat_reg == 'mean_var':
        # regularization with prior of temporal statistics
        stat_reg_hooks = []
        # candidate_conv_layers = [nn.Conv2d, nn.Conv3d]

        if not hasattr(args, 'moving_avg'):
            args.moving_avg = False
        if not hasattr(args, 'momentum_mvg'):
            args.momentum_mvg = 0.1

        if isinstance(args.stat_type, str):
            """
            regularization of one type of statistics
            load one type of statistics 
            """
            raise NotImplementedError('args.stat_type of str  is deprecated, use list instead. To add the implementation for case of Video swin transformer. ')
            if args.stat_type == 'temp':
                bn_layers = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
            elif args.stat_type in ['spatial', 'spatiotemp' ]:
                bn_layers = [nn.BatchNorm2d, nn.BatchNorm3d]
            chosen_layers = choose_layers(model, bn_layers)
            # chosen_conv_layers = choose_layers(model, candidate_conv_layers)

            list_stat_mean_clean = list(np.load(args.stat_mean_clean_file, allow_pickle=True))
            list_stat_var_clean = list(np.load(args.stat_var_clean_file, allow_pickle=True))
            # assert len(list_temp_mean_clean) == len(chosen_conv_layers)
            assert len(list_stat_mean_clean) == len(chosen_layers)
            for layer_id, (chosen_layer_name, chosen_layer) in enumerate(  chosen_layers ):
                # if isinstance(bn_layer, nn.BatchNorm2d):
                # for nm, m in model.named_modules():
                for block_name in  args.chosen_blocks:
                    if block_name in chosen_layer_name :
                        stat_reg_hooks.append(NormStatsRegHook(chosen_layer, clip_len= args.clip_length,
                                                               stats_clean_tuple=(list_stat_mean_clean[layer_id], list_stat_var_clean[layer_id]), reg_type=args.reg_type, moving_avg= args.moving_avg, momentum= args.momentum_mvg,
                                                               stat_type=args.stat_type, reduce_dim=args.reduce_dim))
                        break
        elif isinstance(args.stat_type, list):
            """
            combination of regularizations of multiple types of statistics 
            load multiple types of statistics 
            """
            if args.arch == 'tanet':
                # todo  temporal statistics are computed on  nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d
                #      spatial statistics are computed on nn.BatchNorm2d, nn.BatchNorm3d,   not on Batchnorm1d
                #      spatiotemporal statistics are computed on nn.BatchNorm2d, nn.BatchNorm3d,  not on Batchnorm1d
                bn_layers = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
                chosen_layers = choose_layers(model, bn_layers)
                list_temp_mean_clean, list_temp_var_clean, list_spatiotemp_mean_clean, list_spatiotemp_var_clean, list_spatial_mean_clean, list_spatial_var_clean = load_precomputed_statistics(args, len(chosen_layers))
                if 'spatiotemp' in args.stat_type:
                    list_spatiotemp_mean_clean_new, list_spatiotemp_var_clean_new = [], []
                else:
                    list_spatiotemp_mean_clean_new, list_spatiotemp_var_clean_new = [None] * len(chosen_layers), [None] * len(chosen_layers)
                if 'spatial' in args.stat_type:
                    list_spatial_mean_clean_new, list_spatial_var_clean_new = [], []
                else:
                    list_spatial_mean_clean_new, list_spatial_var_clean_new =[None] * len(chosen_layers), [None] * len(chosen_layers)
                counter = 0
                for layer_id,(chosen_layer_name, chosen_layer) in enumerate(chosen_layers):
                    if isinstance(chosen_layer, nn.BatchNorm1d):
                        # at the position of Batchnorm1d, add None as placeholder in the list for spatial and spatiotemporal statistics
                        if 'spatiotemp' in args.stat_type:
                            list_spatiotemp_mean_clean_new.append(None)
                            list_spatiotemp_var_clean_new.append(None)
                        if 'spatial' in args.stat_type:
                            list_spatial_mean_clean_new.append(None)
                            list_spatial_var_clean_new.append(None)
                    elif isinstance(chosen_layer, nn.BatchNorm2d) or isinstance(chosen_layer, nn.BatchNorm3d):
                        if 'spatiotemp' in args.stat_type:
                            list_spatiotemp_mean_clean_new.append( list_spatiotemp_mean_clean[counter] )
                            list_spatiotemp_var_clean_new.append( list_spatiotemp_var_clean[counter])
                        if 'spatial' in args.stat_type:
                            list_spatial_mean_clean_new.append(list_spatial_mean_clean[counter])
                            list_spatial_var_clean_new.append(list_spatial_var_clean[counter])

                        counter +=1
            elif args.arch == 'videoswintransformer':
                # todo   on Video Swin Transformer,
                #     statistics are computed on all LayerNorm layers (feature in shape BTHWC), except for the first LayerNorm after Conv3D (feature in shape B,combined_dim,C)
                candidate_layers = [nn.LayerNorm]
                chosen_layers = choose_layers(model, candidate_layers)
                chosen_layers = chosen_layers[1:]
                list_temp_mean_clean, list_temp_var_clean, list_spatiotemp_mean_clean, list_spatiotemp_var_clean, list_spatial_mean_clean, list_spatial_var_clean = load_precomputed_statistics( args, len(chosen_layers))
                list_spatial_mean_clean_new, list_spatial_var_clean_new= list_spatial_mean_clean, list_spatial_var_clean
                list_spatiotemp_mean_clean_new, list_spatiotemp_var_clean_new  = list_spatiotemp_mean_clean, list_spatiotemp_var_clean

            assert len(list_temp_mean_clean) == len(chosen_layers)
            for layer_id, (chosen_layer_name, chosen_layer) in enumerate(chosen_layers):
                for block_name in  args.chosen_blocks:
                    if block_name in chosen_layer_name :
                        stat_reg_hooks.append(CombineNormStatsRegHook_onereg(chosen_layer, clip_len = args.clip_length,
                            spatiotemp_stats_clean_tuple = (list_spatiotemp_mean_clean_new[layer_id], list_spatiotemp_var_clean_new[layer_id]),
                            reg_type=args.reg_type, moving_avg= args.moving_avg,  momentum=args.momentum_mvg,  stat_type_list = args.stat_type, reduce_dim = args.reduce_dim,
                            before_norm= args.before_norm, if_sample_tta_aug_views= args.if_sample_tta_aug_views, n_augmented_views=args.n_augmented_views ))
                        break
    elif args.stat_reg == 'cossim':
        stat_reg_hooks = []
        list_temp_cossim = list(np.load( args.temp_cossim_clean_file, allow_pickle=True ))
        if args.arch == 'tanet':
            bn_layers = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
            chosen_layers = choose_layers(model, bn_layers)
        assert len(list_temp_cossim) == len(chosen_layers)
        for layer_id, (chosen_layer_name, chosen_layer) in enumerate(chosen_layers):
            if list_temp_cossim[layer_id] is not None:
                for block_name in args.chosen_blocks:
                    if block_name in chosen_layer_name:
                        stat_reg_hooks.append(CombineCossimRegHook(chosen_layer, clip_len=args.clip_length,
                                                                   temp_cossim= list_temp_cossim[layer_id],
                                                                      reg_type=args.reg_type, moving_avg=args.moving_avg,
                                                                      momentum=args.momentum_mvg,
                                                                      stat_type_list=args.stat_type,
                                                                      before_norm=args.before_norm))
                        break

    elif args.stat_reg == 'BNS':
        # todo regularization on BNS statistics
        # regularization on BNS staticstics
        bns_feature_hooks = []
        chosen_layers = choose_layers(model, candidate_bn_layers)
        for chosen_layer in chosen_layers:
            # regularize between manually computed target batch statistics (whether or not in running manner) between  source stastistics
            bns_feature_hooks.append(BNFeatureHook(chosen_layer, reg_type='l2norm', running_manner=args.running_manner,
                                                   use_src_stat_in_reg=args.use_src_stat_in_reg, momentum=args.momentum_bns))
    else:
        raise Exception(f'undefined regularization type {args.stat_reg}')

    epoch_result_list = []

    if args.arch == 'tanet':
        n_clips = int(args.sample_style.split("-")[-1])
    elif args.arch == 'videoswintransformer':
        n_clips = args.num_clips
    if args.if_sample_tta_aug_views:
        assert n_clips == 1
        n_augmented_views = args.n_augmented_views

    if_pred_consistency = args.if_pred_consistency if args.if_sample_tta_aug_views else False



    for epoch in range(args.n_epoch_adapat):
        batch_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_reg = AverageMeter()
        losses_consis = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        pred_concat = []
        gt_concat = []
        end = time.time()
        # with torch.autograd.set_detect_anomaly(True):
        for i, (input, target) in enumerate(tta_loader):  #
            # model.eval()
            model.train()  # BN layers are set to train mode
            if args.fix_BNS:  # fix the BNS during forward pass
                for m in model.modules():
                    for candidate in candidate_bn_layers:
                        if isinstance(m, candidate):
                            m.eval()
            actual_bz = input.shape[0]
            input = input.cuda()
            target = target.cuda()
            if args.arch == 'tanet':
                # (actual_bz,          C* spatial_crops * temporal_clips* clip_len,          256, 256) ->                  (actual_bz * spatial_crops * temporal_clips* clip_len,             C, 256, 256)
                input = input.view(-1, 3, input.size(2), input.size(3))
                # input = input.view(actual_bz, -1, 3, input.size(2), input.size(3))
                # input = input.view(-1, 3, 224, 224)
                if args.if_sample_tta_aug_views:
                    input = input.view(actual_bz * args.test_crops * n_augmented_views,   args.clip_length,   3, input.size(2), input.size(3) )
                else:
                    input = input.view(actual_bz * args.test_crops * n_clips,
                                   args.clip_length, 3, input.size(2), input.size(3))  # (actual_bz * spatial_crops * temporal_clips* clip_len,    C,     256, 256) -> (actual_bz * spatial_crops * temporal_clips,   clip_len,   C,  256,   256)
                output = model(input)  # (actual_bz * spatial_crops * temporal_clips,         clip_len,   C,   256, 256)   ->     (actual_bz * spatial_crops * temporal_clips,       n_class )
                # take the average among all spatial_crops * temporal_clips,   (actual_bz * spatial_crops * temporal_clips,       n_class )  ->   (actual_bz,       n_class )

                # if_pred_consistency = False
                if args.if_sample_tta_aug_views:
                    output = output.reshape(actual_bz, args.test_crops * n_augmented_views, -1)  # (N, n_views, n_class )
                    if if_pred_consistency:
                        loss_consis = compute_pred_consis(output)
                    output = output.mean(1)
                else:
                    output = output.reshape(actual_bz, args.test_crops * n_clips, -1).mean(1)

            elif args.arch == 'videoswintransformer':
                # the format shape is N C T H W
                # todo (batch, n_views, C, T, H, W)
                if args.if_sample_tta_aug_views:
                    n_views = args.test_crops * n_augmented_views
                else:
                    n_views = args.test_crops * n_clips
                # input = input.view(-1, n_views, 3, args.clip_length, input.size(3), input.size(4))
                if args.if_sample_tta_aug_views:
                    if if_pred_consistency:
                        output, view_cls_score = model( input)  # (batch, n_views, C, T, H, W) ->  (batch, n_class), todo outputs are unnormalized scores
                        loss_consis = compute_pred_consis(view_cls_score)
                else:
                    output, _ = model( input)
            else:
                input = input.reshape( (-1,) + input.shape[2:])  # (batch, n_views, 3, T, 224,224 ) -> (batch * n_views, 3, T, 224,224 )
                # forward pass
                output = model( input)  # (batch * n_views, 3, T, 224,224 ) ->  (batch * n_views,  n_class)  todo  reshape clip prediction into video prediction
                output = rearrange(output, '(d0 d1) d2 -> d0 d1 d2', d0=actual_bz)  # (batch * n_views,  n_class) ->  (batch, n_views,  n_class)  todo  reshape clip prediction into video prediction
                output = torch.mean(output, dim=1)  # (batch, n_views,  n_class) ->  (batch,  n_class)
            loss_ce = criterion(output, target)

            loss_reg = torch.tensor(0).float().cuda()
            if args.stat_reg:
                for hook in stat_reg_hooks:
                    loss_reg += hook.r_feature.cuda()
            else:
                for hook in bns_feature_hooks:
                    loss_reg += hook.r_feature.cuda()
            if if_pred_consistency:
                loss = args.lambda_feature_reg*loss_reg + args.lambda_pred_consis * loss_consis
            else:
                loss = loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_iter += 1

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            _, preds = torch.max(output, 1)
            # pred_concat = np.concatenate([pred_concat, preds.detach().cpu().numpy()])
            # gt_concat = np.concatenate([gt_concat, target.detach().cpu().numpy()])

            losses_ce.update(loss_ce.item(), actual_bz)
            losses_reg.update(loss_reg.item(), actual_bz)
            if if_pred_consistency:
                losses_consis.update(loss_consis.item(), actual_bz)
            top1.update(prec1.item(), actual_bz)
            top5.update(prec5.item(), actual_bz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.verbose:
                if i % args.print_freq == 0:
                    logger.debug(('TTA Epoch{epoch}: [{0}/{1}]\t'
                                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                  'Loss reg {loss_reg.val:.4f} ({loss_reg.avg:.4f})\t'
                                  'Loss consis {loss_consis.val:.4f} ({loss_consis.avg:.4f})\t'
                                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(tta_loader), epoch=epoch, batch_time=batch_time, loss_reg=losses_reg, loss_consis = losses_consis,
                        top1=top1, top5=top5)))

            if writer is not None:
                writer.add_scalars('loss', {'loss_reg': loss_reg.item()}, global_step=global_iter)
                if if_pred_consistency:
                    writer.add_scalars('loss', {'loss_consis': loss_consis.item()}, global_step=global_iter)
                writer.add_scalars('loss', {'loss_ce': loss_ce.item()}, global_step=global_iter)
            # writer.add_scalars('acc', {'val_acc': top1.avg}, global_step=epoch)
        # logger.debug(f'Validation acc {top1.avg} ')
        # logger.debug(classification_report(pred_concat, gt_concat))

        # evaluate on the entire test set at the end of each epoch
        # todo remove all the hooks
        if args.stat_reg in ['mean_var', 'cossim']:
            for stat_reg_hook in stat_reg_hooks:
                stat_reg_hook.close()
        elif args.stat_reg == 'BNS':
            for bns_feature_hook in bns_feature_hooks:
                bns_feature_hook.close()
        top1_acc = validate_brief(eval_loader=eval_loader, model=model, global_iter=global_iter, epoch=epoch, args=args,
                       logger=logger, writer=writer)
        epoch_result_list.append(top1_acc)
    return epoch_result_list, model

def evaluate_baselines(model, args=None, logger=None, writer=None):

    tta_loader = torch.utils.data.DataLoader(
        get_dataset(args, split='val'),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    eval_loader = torch.utils.data.DataLoader(
        get_dataset(args, split='val'),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    global_iter = 0
    validate_brief(eval_loader=eval_loader, model=model, global_iter=global_iter, args=args,
                   logger=logger, writer=writer)



def validate_brief(eval_loader, model, global_iter, epoch=None, args=None, logger=None, writer=None):
    batch_time = AverageMeter()
    # losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    if args.arch == 'tanet':
        n_clips = int(args.sample_style.split("-")[-1])
    elif args.arch == 'videoswintransformer':
        n_clips = args.num_clips

    pred_concat = []
    gt_concat = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(eval_loader):  #
            model.eval()
            actual_bz = input.shape[0]
            input = input.cuda()
            target = target.cuda()
            if args.arch == 'tanet':
                # (actual_bz, C* spatial_crops * temporal_clips* clip_len, 256, 256) ->   (actual_bz * spatial_crops * temporal_clips* clip_len,  C, 256, 256)
                input = input.view(-1, 3, input.size(2), input.size(3))
                input = input.view(actual_bz * args.test_crops * n_clips,
                                       args.clip_length, 3, input.size(2),input.size(3))  # (actual_bz * spatial_crops * temporal_clips* clip_len,  C, 256, 256) -> (actual_bz * spatial_crops * temporal_clips,  clip_len,  C, 256, 256)
                output = model(input) #  (actual_bz * spatial_crops * temporal_clips,         clip_len,  C, 256, 256)   ->     (actual_bz * spatial_crops * temporal_clips,       n_class )
                # take the average among all spatial_crops * temporal_clips,   (actual_bz * spatial_crops * temporal_clips,       n_class )  ->   (actual_bz,       n_class )
                output = output.reshape(actual_bz, args.test_crops * n_clips, -1).mean(1)
            elif args.arch == 'videoswintransformer':
                # the format shape is N C T H W         if  collapse in datsaet is True, then shape is  (actual_bz,   C* spatial_crops * temporal_clips* clip_len,    256,     256)
                # (batch, n_views, C, T, H, W)
                n_views = args.test_crops * n_clips
                # input = input.view(-1, n_views, 3, args.clip_length, input.size(3), input.size(4))
                output, _ = model(  input)  # (batch, n_views, C, T, H, W) ->  (batch, n_class), todo outputs are unnormalized scores
            else:
                input = input.reshape( (-1,) + input.shape[2:])  # (batch, n_views, 3, T, 224,224 ) -> (batch * n_views, 3, T, 224,224 )
                output = model(input)  # (batch * n_views, 3, T, 224,224 ) ->  (batch * n_views,  n_class)  todo  reshape clip prediction into video prediction
                output = rearrange(output, '(d0 d1) d2 -> d0 d1 d2', d0=actual_bz)  # (batch * n_views,  n_class) ->  (batch, n_views,  n_class)  todo  reshape clip prediction into video prediction
                output = torch.mean(output, dim=1)  # (batch, n_views,  n_class) ->  (batch,  n_class)


            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            _, preds = torch.max(output, 1)
            # pred_concat = np.concatenate([pred_concat, preds.detach().cpu().numpy()])
            # gt_concat = np.concatenate([gt_concat, target.detach().cpu().numpy()])

            # losses.update(loss.item(), actual_bz)
            top1.update(prec1.item(), actual_bz)
            top5.update(prec5.item(), actual_bz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.verbose:
                if i % args.print_freq == 0:
                    logger.debug(('  \tTest Epoch {epoch}: [{0}/{1}]\t'
                                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(i, len(eval_loader), epoch=epoch,
                                                                                  batch_time=batch_time, top1=top1,
                                                                                  top5=top5)))

    eval_dua = False
    if args.evaluate_baselines:
        if args.baseline == 'dua':
            eval_dua == True
    if not eval_dua:
            logger.debug(('  \tTesting Results Epoch {epoch}: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(epoch=epoch,
                                                                                                          top1=top1,
                                                                                                          top5=top5)))
    else:
        logger.debug(
            ('\tTesting Results for DUA after adaptation on video {epoch}: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(epoch=epoch,
                                                                                                     top1=top1,
                                                                                                     top5=top5)))
        return top1.avg

    if writer is not None:
        # writer.add_scalars('loss', {'val_loss': losses.avg}, global_step=epoch)
        writer.add_scalars('acc', {'test_acc': top1.avg}, global_step=global_iter)
    logger.debug(f'  \tTest Epoch {epoch} acc {top1.avg} ')
    return top1.avg

def get_dataset_videoswin(args, split = 'train', dataset_type = None):
    from models.videoswintransformer_models.video_dataset import Video_SwinDataset
    # from models.videoswintransformer_models.trans
    if split == 'train':
        raise NotImplementedError('Training dataset processing for Video Swin Transformer to be added!')
    elif split == 'val':
        if dataset_type == 'tta':
            if_sample_tta_aug_views = args.if_sample_tta_aug_views
        elif dataset_type == 'eval':
            if_sample_tta_aug_views = False
        tta_view_sample_style_list = args.tta_view_sample_style_list if if_sample_tta_aug_views else None
        return Video_SwinDataset( args.val_vid_list,
                 num_segments=args.clip_length,  # clip_length
                 frame_interval = args.frame_interval,
                 num_clips = args.num_clips, # number of temporal clips
                 frame_uniform = args.frame_uniform,
                 test_mode = True,
                 flip_ratio = args.flip_ratio,
                 scale_size = args.scale_size,
                 input_size= args.input_size,
                 img_norm_cfg  = args.img_norm_cfg,

                 vid_format=args.vid_format,
                 video_data_dir = args.video_data_dir,
                 remove_missing = False,
                 debug = args.debug,
                if_sample_tta_aug_views= if_sample_tta_aug_views,
                tta_view_sample_style_list=tta_view_sample_style_list,
                n_augmented_views=args.n_augmented_views )

def get_dataset_tanet(args, split = 'train', dataset_type = None):
    from models.tanet_models.transforms import GroupFullResSample_TANet, GroupScale_TANet, GroupCenterCrop_TANet, Stack_TANet, ToTorchFormatTensor_TANet, \
        GroupNormalize_TANet, GroupMultiScaleCrop_TANet, SubgroupWise_MultiScaleCrop_TANet, SubgroupWise_RandomHorizontalFlip_TANet
    from models.tanet_models.video_dataset import Video_TANetDataSet
    if split == 'train':
        raise NotImplementedError('Training dataset processing for TANet to be added!')
    elif split == 'val':
        # if args.full_res,  feed 256x256 to the network
        # input_size = tanet_model.scale_size if args.full_res else tanet_model.input_size
        input_size = args.scale_size if args.full_res else args.input_size

        if dataset_type == 'tta':
            if_sample_tta_aug_views = args.if_sample_tta_aug_views
        elif dataset_type == 'eval':
            if_sample_tta_aug_views = False
        tta_view_sample_style_list = args.tta_view_sample_style_list if if_sample_tta_aug_views else None
        n_augmented_views = args.n_augmented_views if if_sample_tta_aug_views else None
        if_spatial_rand_cropping = args.if_spatial_rand_cropping if if_sample_tta_aug_views else False

        if args.test_crops == 1:
            if if_spatial_rand_cropping:
                # cropping = torchvision.transforms.Compose([GroupMultiScaleCrop_TANet(input_size)] )
                cropping = torchvision.transforms.Compose([SubgroupWise_MultiScaleCrop_TANet(input_size=input_size,
                                                                                             n_temp_clips=n_augmented_views,
                                                                                             clip_len=args.clip_length)])
                # label_transforms =  {
                # 86: 87,
                # 87: 86,
                # 93: 94,
                # 94: 93,
                # 166: 167,
                # 167: 166 } if args.dataset == 'somethingv2' else None
                # cropping = torchvision.transforms.Compose([SubgroupWise_MultiScaleCrop_TANet(input_size=input_size,
                #                                                                              n_temp_clips=n_augmented_views,
                #                                                                              clip_len=args.clip_length),
                #                                            SubgroupWise_RandomHorizontalFlip_TANet(label_transforms= label_transforms,
                #                                                                                    n_temp_clips=n_augmented_views,
                #                                                                                    clip_len=args.clip_length)])
            else:
                cropping = torchvision.transforms.Compose([  # scale to scale_size, then center crop to input_size
                    GroupScale_TANet(args.scale_size), # scale size is 256, input_size is 224, todo here scale_size is the size of the smaller edge
                    GroupCenterCrop_TANet(input_size),
                ])
        elif args.test_crops == 3:
            cropping = torchvision.transforms.Compose([GroupFullResSample_TANet(input_size, args.scale_size, flip=False)])
        else:
            raise NotImplementedError(f'{args.test_crops} spatial crops not implemented!')



        return Video_TANetDataSet(
            args.val_vid_list,
            num_segments=args.clip_length,
            new_length=1 if args.modality == "RGB" else 5,
            modality=args.modality,
            # image_tmpl=prefix,
            vid_format=args.vid_format,
            test_mode=True,
            remove_missing= True,
            transform=torchvision.transforms.Compose([
                cropping,   #  GroupFullResSample,  scale to scale size, and crop  left, right, center  3 spatial crops    10 temporal clips,  16 frames,     480 frames in total   256 256
                Stack_TANet(roll=False),  #  stack the temporal dimension into channel dimension,   (*, C, T, H, W) -> (*, C*T, H, W)   ( *,   480*3, 256, 256)
                ToTorchFormatTensor_TANet(div= True),  # todo divide by 255   [0.485, 0.456, 0.406] * 255 = [123.675, 116.28, 103.53]      [0.229, 0.224, 0.225] * 255= [58.395,57.12, 57.375]
                GroupNormalize_TANet(args.input_mean, args.input_std),
            ]),
            video_data_dir=args.video_data_dir,
            test_sample=args.sample_style, #  'uniform-x' or 'dense-x'
            debug=args.debug,
            if_sample_tta_aug_views= if_sample_tta_aug_views,
            tta_view_sample_style_list=tta_view_sample_style_list,
            n_tta_aug_views=n_augmented_views)


def get_dataset_tanet_dua(args, tanet_model = None,  split = 'train'):
    from models.tanet_models.transforms import GroupFullResSample_TANet, GroupScale_TANet, GroupCenterCrop_TANet, Stack_TANet, ToTorchFormatTensor_TANet, GroupNormalize_TANet
    from models.tanet_models.video_dataset import Video_TANetDataSet
    if split == 'train':
        raise NotImplementedError('Training dataset processing for TANet to be added!')
    elif split == 'val':
        # if args.full_res,  feed 256x256 to the network
        input_size = tanet_model.scale_size if args.full_res else tanet_model.input_size
        if args.test_crops == 1:
            cropping = torchvision.transforms.Compose([  # scale to scale_size, then center crop to input_size
                GroupScale_TANet(tanet_model.scale_size),
                GroupCenterCrop_TANet(input_size),
            ])
        elif args.test_crops == 3:
            cropping = torchvision.transforms.Compose([GroupFullResSample_TANet(input_size, tanet_model.scale_size, flip=False)])
        else:
            raise NotImplementedError(f'{args.test_crops} spatial crops not implemented!')

        return Video_TANetDataSet(
            args.val_vid_list,
            num_segments=args.clip_length,
            new_length=1 if args.modality == "RGB" else 5,
            modality=args.modality,
            # image_tmpl=prefix,
            vid_format=args.vid_format,
            test_mode=True,
            remove_missing= True,
            transform=torchvision.transforms.Compose([
                cropping,   #  GroupFullResSample,  scale to scale size, and crop  left, right, center  3 spatial crops    10 temporal clips,  16 frames,     480 frames in total   256 256
                Stack_TANet(roll=False),  #  stack the temporal dimension into channel dimension,   (*, C, T, H, W) -> (*, C*T, H, W)   ( *,   480*3, 256, 256)
                ToTorchFormatTensor_TANet(div= True), # todo divide by 255
                GroupNormalize_TANet(tanet_model.input_mean, tanet_model.input_std),
            ]), video_data_dir=args.video_data_dir,
            test_sample=args.sample_style,
            debug=args.debug),\
            Video_TANetDataSet( args.val_vid_list,
            num_segments=args.clip_length,
            new_length=1 if args.modality == "RGB" else 5,
            modality=args.modality,
            # image_tmpl=prefix,
            vid_format=args.vid_format,
            test_mode=True,
            remove_missing=True,
            transform=torchvision.transforms.Compose([
                # cropping,
                # GroupFullResSample,  scale to scale size, and crop  left, right, center  3 spatial crops    10 temporal clips,  16 frames,     480 frames in total   256 256
                Stack_TANet(roll=False),
                # stack the temporal dimension into channel dimension,   (*, C, T, H, W) -> (*, C*T, H, W)   ( *,   480*3, 256, 256)
                ToTorchFormatTensor_TANet(div=True),
                # GroupNormalize_TANet(tanet_model.input_mean, tanet_model.input_std),
            ]),
            video_data_dir=args.video_data_dir,
            test_sample=args.sample_style,
            debug=args.debug)


def get_dataset(args, split='train'):
    train_augmentation = get_augmentation(args, args.modality, args.input_size)  # GroupMultiScaleCrop  amd   GroupRandomHorizontalFlip

    train_transform = torchvision.transforms.Compose([
        train_augmentation,  # GroupMultiScaleCrop  amd   GroupRandomHorizontalFlip
        fromListToTorchFormatTensor(clip_len=args.clip_length, num_clips=args.num_clips),
        GroupNormalize(args.input_mean, args.input_std)])

    dua_transform = torchvision.transforms.Compose([
        GroupScale(int(args.scale_size)),
        GroupCenterCrop(args.crop_size),
        fromListToTorchFormatTensor(clip_len=args.clip_length, num_clips=args.num_clips),
    ])

    val_transform = torchvision.transforms.Compose([
        GroupScale(int(args.scale_size)),
        GroupCenterCrop(args.crop_size),
        fromListToTorchFormatTensor(clip_len=args.clip_length, num_clips=args.num_clips),
        GroupNormalize(args.input_mean, args.input_std),
    ])
    if args.datatype == 'vid':
        if split == 'train':
            if args.tsn_style:  # vid, train, tsn
                return MyTSNVideoDataset(args, args.root_path, args.train_vid_list, clip_length=args.clip_length,
                                         frame_interval=args.frame_interval,
                                         num_clips=args.num_clips, modality=args.modality, vid_format=args.vid_format,
                                         # load images
                                         transform=train_transform, video_data_dir=args.video_data_dir,
                                         debug=args.debug)
            else:  # vid, train,  non-tsn
                return MyVideoDataset(args, args.root_path, args.train_vid_list, clip_length=args.clip_length,
                                      frame_interval=args.frame_interval, num_clips=args.num_clips,
                                      modality=args.modality, vid_format=args.vid_format,
                                      transform=train_transform, video_data_dir=args.video_data_dir, debug=args.debug)


        use_dua_val = False
        if args.evaluate_baselines:
            if args.baseline == 'dua':
                use_dua_val = True  #  use dua transformation for evaluation

        if split == 'val' and (not use_dua_val):
            if args.tsn_style:  # vid, val, tsn
                return MyTSNVideoDataset(args, args.root_path, args.val_vid_list, clip_length=args.clip_length,
                                         frame_interval=args.frame_interval,
                                         num_clips=args.num_clips, modality=args.modality, vid_format=args.vid_format,
                                         transform=val_transform, video_data_dir=args.video_data_dir, test_mode=True,
                                         debug=args.debug)
            else:  # vid, val, non-tsn
                return MyVideoDataset(args, args.root_path, args.val_vid_list, clip_length=args.clip_length,
                                      frame_interval=args.frame_interval, num_clips=args.num_clips,
                                      modality=args.modality, vid_format=args.vid_format,
                                      transform=val_transform, test_mode=True, video_data_dir=args.video_data_dir,
                                      debug=args.debug)

        elif split == 'val' and use_dua_val:
            if args.tsn_style:  # vid, val, tsn
                return MyTSNVideoDataset(args, args.root_path, args.val_vid_list, clip_length=args.clip_length,
                                         frame_interval=args.frame_interval,
                                         num_clips=args.num_clips, modality=args.modality, vid_format=args.vid_format,
                                         transform=dua_transform, video_data_dir=args.video_data_dir, test_mode=True,
                                         debug=args.debug), \
                       MyTSNVideoDataset(args, args.root_path, args.val_vid_list, clip_length=args.clip_length,
                                         frame_interval=args.frame_interval,
                                         num_clips=args.num_clips, modality=args.modality, vid_format=args.vid_format,
                                         transform=val_transform, video_data_dir=args.video_data_dir, test_mode=True,
                                         debug=args.debug)
            else:  # vid, val, non-tsn
                return MyVideoDataset(args, args.root_path, args.val_vid_list, clip_length=args.clip_length,
                                      frame_interval=args.frame_interval, num_clips=args.num_clips,
                                      modality=args.modality, vid_format=args.vid_format,
                                      transform=val_transform, test_mode=True, video_data_dir=args.video_data_dir,
                                      debug=args.debug)

    elif args.datatype == 'frame':
        if split == 'train':
            if args.tsn_style:  # frame, train, tsn
                return MyTSNDataset(args, args.root_path, args.train_frame_list, clip_length=args.clip_length,
                                    frame_interval=args.frame_interval,
                                    num_clips=args.num_clips, modality=args.modality,
                                    image_tmpl=args.img_tmpl if args.modality == "RGB" else args.flow_prefix + "{}_{:05d}.jpg",
                                    # load images
                                    transform=train_transform, data_dir=args.frame_data_dir, debug=args.debug)
            else:  # vid, train, non-tsn
                raise Exception('not implemented yet!')
        elif split == 'val':
            if args.tsn_style:  # frame, val, tsn
                return MyTSNDataset(args, args.root_path, args.val_frame_list, clip_length=args.clip_length,
                                    frame_interval=args.frame_interval,
                                    num_clips=args.num_clips, modality=args.modality,
                                    image_tmpl=args.img_tmpl if args.modality == "RGB" else args.flow_prefix + "{}_{:05d}.jpg",
                                    transform=val_transform, data_dir=args.frame_data_dir, test_mode=True,
                                    debug=args.debug)
            else:  # vid, val, non-tsn
                raise Exception('not implemented yet!')


def get_model(args, num_classes, logger):
    if args.arch == 'r2plus1d':
        model = MyR2plus1d(num_classes=num_classes, use_pretrained=args.use_pretrained)
    elif args.arch == 'i3d_incep':
        model = InceptionI3d(num_classes=400, in_channels=3)
        if args.use_pretrained:
            model.load_state_dict(torch.load(args.pretrained_model))
            logger.debug(f'Loaded pretrained I3D Inception model {args.pretrained_model}')
        model.replace_logits(num_classes=num_classes)
    elif 'i3d_resnet' in args.arch:
        if args.arch in ['i3d_resnet18', 'i3d_resnet34', ]:
            in_channel = 512
        elif args.arch in ['i3d_resnet50', 'i3d_resnet101', 'i3d_resnet152', ]:
            in_channel = 2048
        model = getattr(i3d, args.arch)(modality=args.modality, num_classes=num_classes, in_channel=in_channel,
                                        dropout_ratio=args.dropout)
    elif args.arch == 'tanet':
        model = TSN(
            num_classes,
            args.clip_length,
            args.modality,
            base_model='resnet50',
            consensus_type= 'avg',
            img_feature_dim=args.img_feature_dim,
            tam= True,
            non_local=False,
            partial_bn=args.partial_bn
        )
    elif args.arch == 'videomae':
        model = create_model(
            args.model, # 'vit_base_patch16_224'
            pretrained=False,
            num_classes= num_classes,
            all_frames=args.num_frames * args.num_segments,
            tubelet_size=args.tubelet_size,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            attn_drop_rate=args.attn_drop_rate,
            drop_block_rate=None,
            use_mean_pooling=args.use_mean_pooling,
            init_scale=args.init_scale,
        )
    elif args.arch == 'videoswintransformer':
        model = Recognizer3D(num_classes =  num_classes, patch_size= args.patch_size, window_size=args.window_size, drop_path_rate=args.drop_path_rate)
    else:
        raise Exception(f'{args.arch} is not a valid model!')
    return model


def validate_old(val_loader, model, criterion, iter, epoch=None, args=None, logger=None, writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.reshape(
                (-1,) + input.shape[2:])  # (batch, n_views, 3, T, 224,224 ) -> (batch * n_views, 3, T, 224,224 )
            target = target.reshape((target.shape[0], 1)).repeat(1, args.num_clips)
            target = target.reshape((-1,) + target.shape[2:])

            target = target.cuda()
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(
                1, 5))  # todo this is clip accuracy,  we should take the average of all clip predictions from a video

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logger.debug(('Test: [{0}/{1}]\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5)))

    logger.debug(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                  .format(top1=top1, top5=top5, loss=losses)))
    writer.add_scalars('loss', {'val_loss': losses.avg}, global_step=epoch)
    writer.add_scalars('acc', {'val_acc': top1.avg}, global_step=epoch)

    return top1.avg