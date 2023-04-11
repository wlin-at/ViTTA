import os
import sys
print(os.getcwd())
sys.path.append(os.path.abspath('..'))
from utils.opts import get_opts
from corpus.main_eval import eval


if __name__ == '__main__':
    global args
    args = get_opts()
    args.gpus = [0]
    args.arch = 'tanet'
    args.dataset = 'ucf101'
    # todo ========================= To Specify ==========================
    args.model_path = '.../tanet_ucf.pth.tar'
    args.video_data_dir = '...' #  main directory of the video data,  [args.video_data_dir] + [path in file list] should be complete absolute path for a video file
    args.val_vid_list = '...' # list of training data for computing statistics, with lines in format :   file_path n_frames class_id
    args.result_dir =  '.../{}_{}/compute_norm_{}stats_{}_bz{}'
    # todo ========================= To Specify ==========================

    args.clip_length = 16
    args.batch_size = 32  # 12
    args.sample_style = 'uniform-1'  # number of temporal clips
    args.test_crops = 1  # number of spatial crops

    args.tta = True
    args.evaluate_baselines = not args.tta
    args.baseline = 'source'

    args.compute_stat = 'mean_var'
    args.stat_type = 'spatiotemp'

    args.corruptions = 'clean'
    args.result_dir = args.result_dir.format(args.arch, args.dataset, args.stat_type, args.corruptions, args.batch_size)
    eval(args=args, )


