import argparse

# for TANet
input_mean = [0.485, 0.456, 0.406]
input_std = [0.229, 0.224, 0.225]

# for Video Swin Transformer
img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

parser = argparse.ArgumentParser(description="ViTTA")

# ========================= Data Configs ==========================
# parser.add_argument('--data_dir', default='/media/data_8T', type=str, help='main data directory')
parser.add_argument('--dataset', type=str, default='ucf101', choices=['ucf101', 'somethingv2', 'kinetics'])
parser.add_argument('--modality', type=str, default='RGB')
parser.add_argument('--root_path', default='None', type=str)
parser.add_argument('--video_data_dir',
                    default='/home/ivanl/data/UCF-HMDB/video_pertubations/UCF101/level_5_ucf_val_split_1', type=str,
                    help='directory of the corrupted videos')  # to specify
parser.add_argument('--vid_format', default='', type=str,
                    help='if video format is not given in the filenames in the list, the video format can be specified here')
parser.add_argument('--datatype', default='vid', type=str, choices=['vid', 'frame'])


parser.add_argument('--spatiotemp_mean_clean_file', type=str,
                    default='/home/ivanl/data/UCF-HMDB/UCF-HMDB_all/corruptions_results/source/tanet_ucf101/compute_norm_spatiotempstats_clean_train_bn2d/list_spatiotemp_mean_20220908_235138.npy',
                    help='spatiotemporal statistics - mean')  # to specify
parser.add_argument('--spatiotemp_var_clean_file', type=str,
                    default='/home/ivanl/data/UCF-HMDB/UCF-HMDB_all/corruptions_results/source/tanet_ucf101/compute_norm_spatiotempstats_clean_train_bn2d/list_spatiotemp_var_20220908_235138.npy',
                    help='spatiotemporal statistics - variance')  # to specify

parser.add_argument('--val_vid_list', type=str,
                    default='/home/ivanl/data/UCF-HMDB/video_pertubations/UCF101/list_video_perturbations/{}.txt',
                    help='list of corrupted videos to adapt to, list is named after the corruption type name')  # to specify

parser.add_argument('--result_dir', type=str,
                    default='/home/ivanl/data/UCF-HMDB/UCF-HMDB_all/corruptions_results/source/{}_{}/tta_{}',
                    help='result directory')   # to specify


# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default='tanet', choices=['tanet', 'videoswintransformer'],
                    help='network architecture')
parser.add_argument('--model_path', type=str,
                    default='/home/ivanl/data/DeepInversion_results/train_models/models/UCF/tanet/20220815_122340_ckpt.pth.tar')  # to specify
parser.add_argument('--img_feature_dim', type=int, default=256, help='dimension of image feature on ResNet50')
parser.add_argument('--partial_bn', action='store_true', )

# ========================= Model Configs for Video Swin Transformer ==========================
parser.add_argument('--num_clips', type=int, default=1, help='number of temporal clips')
parser.add_argument('--frame_uniform', type=bool, default=True, help='whether uniform sampling or dense sampling') # uniform sampling is better than dense sampling when using only 1 clip
parser.add_argument('--frame_interval', type=int, default=2)
parser.add_argument('--flip_ratio', type=int, default=0)
parser.add_argument('--img_norm_cfg',  default=img_norm_cfg)
parser.add_argument('--patch_size',  default=(2,4,4))
parser.add_argument('--window_size',  default=(8, 7, 7))
parser.add_argument('--drop_path_rate',  default=0.2)


# ========================= Runtime Configs ==========================
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--norm', action='store_true')
parser.add_argument('--debug', action='store_true', help='if debug, loading only the first 50 videos in the list')
parser.add_argument('--verbose', type=bool, default=True, help='more details in the logging file')
parser.add_argument('--print-freq', '-p', default=20, type=int,  metavar='N', help='print frequency (default: 10)')


# ========================= Learning Configs ==========================
parser.add_argument('--tta', type=bool, default=True, help='perform test-time adaptation')
parser.add_argument('--use_src_stat_in_reg', type=bool, default=True, help='whether to use source statistics in the regularization loss')
parser.add_argument('--fix_BNS', type=bool, default=True, help='whether fix the BNS of target model during forward pass')
parser.add_argument('--running_manner', type=bool, default=True, help='whether to manually compute the target statistics in running manner')
parser.add_argument('--momentum_bns', type=float, default=0.1)
parser.add_argument('--update_only_bn_affine', action='store_true')
parser.add_argument('--compute_stat', action='store_true')
parser.add_argument('--momentum_mvg', type=float, default=0.1)
parser.add_argument('--stat_reg', type=str, default='mean_var', help='statistics regularization')
parser.add_argument('--if_tta_standard', type=str, default='tta_online')
parser.add_argument('--loss_type', type=str, default="nll", choices=['nll'])

parser.add_argument('--if_sample_tta_aug_views', type=bool, default=True)
parser.add_argument('--if_spatial_rand_cropping', type=bool, default=True)
parser.add_argument('--if_pred_consistency', type=bool, default=True)
parser.add_argument('--lambda_pred_consis', type=float, default=0.1)
parser.add_argument('--lambda_feature_reg', type=int, default=1)
parser.add_argument('--n_augmented_views', type=int, default=2)
parser.add_argument('--tta_view_sample_style_list', default=['uniform_equidist'])
parser.add_argument('--stat_type', default=['spatiotemp'])
parser.add_argument('--before_norm', action='store_true')
parser.add_argument('--reduce_dim', type=bool, default=True)
parser.add_argument('--reg_type', type=str, default='l1_loss')

parser.add_argument('--chosen_blocks', default=['layer3', 'layer4'] )
parser.add_argument('--moving_avg', type=bool, default=True )

parser.add_argument('--n_gradient_steps', type=int, default=1, help='number of gradient steps per sample')





parser.add_argument('--full_res', action='store_true')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--scale_size', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--clip_length', type=int, default=16)
parser.add_argument('--sample_style', type=str, default='uniform-1',
                    help="either 'dense-xx' (dense sampling, sample from 64 consecutive frames) or 'uniform-xx' (uniform sampling, TSN style), last number is the number of temporal clips")
parser.add_argument('--test_crops', type=int, default=1, help="number of spatial crops")
parser.add_argument('--use_pretrained', action='store_true',
                    help='whether to use pretrained model for training, set to False during evaluation')
parser.add_argument('--input_mean', default=input_mean)
parser.add_argument('--input_std', default=input_std)

parser.add_argument('--lr', default=0.00005)
parser.add_argument('--n_epoch_adapat', default=1)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)')




def get_opts():
    args = parser.parse_args()

    args.evaluate_baselines = not args.tta
    args.baseline = 'source'

    return args








# parser = argparse.ArgumentParser(description="Implementation of ViTTA")
# parser.add_argument('--dataset', type=str, default='ucf101', choices=['ucf101', 'somethingv2', 'kinetics'])
# parser.add_argument('--modality', type=str, default='RGB', choices=['RGB', 'Flow'])
# parser.add_argument('--root_path', default='None', type=str)
# parser.add_argument('--train_list', default='None', type=str)
# parser.add_argument('--val_list', default='None', type=str)
#
# # ========================= Model Configs ==========================
# parser.add_argument('--arch', type=str, default="i3d_resnet50")
# parser.add_argument('--dropout', '--do', default=0.5, type=float,
#                     metavar='DO', help='dropout ratio (default: 0.5)')  # used in i3d
# parser.add_argument('--clip_length', default=64, type=int, metavar='N',
#                     help='length of sequential frames (default: 64)')
# parser.add_argument('--input_size', default=224, type=int, metavar='N',
#                     help='size of input (default: 224)')
# parser.add_argument('--loss_type', type=str, default="nll",
#                     choices=['nll'])
#
# # ========================= Learning Configs ==========================
# parser.add_argument('--epochs', default=80, type=int, metavar='N',
#                     help='number of total epochs to run')
# parser.add_argument('-b', '--batch-size', default=16, type=int,
#                     metavar='N', help='mini-batch size (default: 16)')
# parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
#                     metavar='LR', help='initial learning rate')
# parser.add_argument('--lr_steps', default=[30, 60], type=float, nargs="+",
#                     metavar='LRSteps', help='epochs to decay learning rate by 10')
# parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                     help='momentum')
# parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
#                     metavar='W', help='weight decay (default: 5e-4)')
# parser.add_argument('--clip-gradient', '--gd', default=None, type=float,
#                     metavar='W', help='gradient norm clipping (default: disabled)')
#
# # ========================= Monitor Configs ==========================
# parser.add_argument('--print-freq', '-p', default=20, type=int,
#                     metavar='N', help='print frequency (default: 10)')
# parser.add_argument('--eval-freq', '-ef', default=5, type=int,
#                     metavar='N', help='evaluation frequency (default: 5)')
#
# # ========================= Runtime Configs ==========================
# parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
#                     help='number of data loading workers (default: 4)')
# parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')
# parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
#                     help='evaluate model on validation set')
# parser.add_argument('--snapshot_pref', type=str, default="")
# parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
#                     help='manual epoch number (useful on restarts)')
# parser.add_argument('--gpus', nargs='+', type=int, default=None)
# parser.add_argument('--flow_prefix', default="flow_", type=str)