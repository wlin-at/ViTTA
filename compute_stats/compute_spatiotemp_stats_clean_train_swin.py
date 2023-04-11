import os
import sys
sys.path.append(os.path.abspath('..'))    # last level
# import os.path as osp
# from utils.opts import parser
from utils.opts import get_opts
from corpus.main_eval import eval

corruptions = ['clean' ]

if __name__ == '__main__':
    global args
    args = get_opts()
    args.gpus = [0]
    args.arch = 'videoswintransformer'
    args.dataset = 'ucf101'
    # todo ========================= To Specify ==========================
    args.model_path = '.../swin_base_patch244_window877_pretrain_kinetics400_30epoch_lr3e-5.pth'
    args.video_data_dir = '...'  #  main directory of the video data,  [args.video_data_dir] + [path in file list] should be complete absolute path for a video file
    args.val_vid_list = '...' # list of training data for computing statistics, with lines in format :   file_path n_frames class_id
    args.result_dir = '.../{}_{}/compute_norm_{}stats_{}_bz{}'
    # todo ========================= To Specify ==========================


    args.batch_size = 32  # 12
    args.clip_length = 16
    args.num_clips = 1  # number of temporal clips
    args.test_crops = 1  # number of spatial crops
    args.frame_uniform = True
    args.frame_interval = 2
    args.scale_size = 224

    args.patch_size = (2, 4, 4)
    args.window_size =(8, 7, 7)

    args.tta = True
    args.evaluate_baselines = not args.tta
    args.baseline = 'source'


    args.n_augmented_views = None
    args.n_epoch_adapat = 1

    args.compute_stat = 'mean_var'
    args.stat_type = 'spatiotemp'
    args.corruptions = 'clean'
    args.result_dir = args.result_dir.format(args.arch, args.dataset, args.stat_type, args.corruptions, args.batch_size)
    eval(args=args, )


