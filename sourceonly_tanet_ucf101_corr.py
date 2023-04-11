# import os.path as osp
# from utils.opts import parser
from utils.opts import get_opts
from utils.utils_ import  get_writer_to_all_result
from corpus.main_eval import eval

# best_prec1 = 0

corruptions = ['gauss', 'pepper', 'salt','shot',
               'zoom', 'impulse', 'defocus', 'motion',
               'jpeg', 'contrast', 'rain', 'h265_abr'  ]

if __name__ == '__main__':
    global args
    args = get_opts()
    args.gpus = [0]
    args.arch = 'tanet'
    args.dataset = 'ucf101'
    # todo ========================= To Specify ==========================
    args.model_path = '.../tanet_ucf.pth.tar'
    args.video_data_dir = '.../level_5_ucf_val_split_1' #  main directory of the video data,  [args.video_data_dir] + [path in file list] should be complete absolute path for a video file
    args.val_vid_list = '.../list_video_perturbations_ucf/{}.txt'
    args.result_dir = '.../{}_{}/eval_{}'
    # todo ========================= To Specify ==========================

    args.batch_size = 32  # 12
    args.clip_length = 16  # 32
    args.sample_style = 'uniform-1'  # number of temporal clips
    args.test_crops = 1  # number of spatial crops

    args.tta = False
    args.evaluate_baselines = not args.tta
    args.baseline = 'source'

    for corr_id, args.corruptions in enumerate(corruptions):
        print(f'####Starting Evaluation for ::: {args.corruptions} corruption####')
        args.val_vid_list = args.val_vid_list.format(args.corruptions)
        args.result_dir = args.result_dir.format( args.arch, args.dataset, args.corruptions )

        epoch_result_list = eval(args=args, )

        if corr_id == 0:
            f_write = get_writer_to_all_result(args)
        f_write.write(' '.join([str(round(float(xx), 3)) for xx in epoch_result_list]) + '\n')

        f_write.flush()
        if corr_id == len(corruptions) - 1:
            f_write.close()
