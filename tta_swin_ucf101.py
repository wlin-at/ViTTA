# import os.path as osp
from utils.opts import get_opts
from utils.utils_ import  get_writer_to_all_result
from corpus.main_eval import eval

corruptions = ['gauss_shuffled',  'pepper_shuffled', 'salt_shuffled', 'shot_shuffled',
               'zoom_shuffled', 'impulse_shuffled', 'defocus_shuffled', 'motion_shuffled',
               'jpeg_shuffled', 'contrast_shuffled', 'rain_shuffled', 'h265_abr_shuffled',  ]

if __name__ == '__main__':
    global args
    args = get_opts()
    args.gpus = [0]
    args.arch = 'videoswintransformer'
    args.dataset = 'ucf101'
    # todo ========================= To Specify ==========================
    args.model_path = '.../swin_base_patch244_window877_pretrain_kinetics400_30epoch_lr3e-5.pth'
    args.video_data_dir = '.../level_5_ucf_val_split_1' #  main directory of the video data,  [args.video_data_dir] + [path in file list] should be complete absolute path for a video file
    args.spatiotemp_mean_clean_file = '.../source_statistics_tanet_ucf/list_spatiotemp_mean_20221004_192722.npy'
    args.spatiotemp_var_clean_file = '.../source_statistics_tanet_ucf/list_spatiotemp_var_20221004_192722.npy'
    args.val_vid_list = '.../list_video_perturbations_ucf/{}.txt'
    args.result_dir = '.../{}_{}/tta_{}'
    # todo ========================= To Specify ==========================



    args.clip_length = 16
    args.num_clips = 1  # number of temporal clips
    args.test_crops = 1  # number of spatial crops
    args.frame_uniform = True
    args.frame_interval = 2
    args.scale_size = 224  #  different than TANet

    args.patch_size = (2,4,4)
    args.window_size = (8, 7, 7)

    args.lr = 0.00001
    args.lambda_pred_consis = 0.05
    args.momentum_mvg = 0.05
    args.chosen_blocks = ['module.backbone.layers.2', 'module.backbone.layers.3', 'module.backbone.norm']


    for corr_id, args.corruptions in enumerate(corruptions):
        print(f'####Starting Evaluation for ::: {args.corruptions} corruption####')
        args.val_vid_list = args.val_vid_list.format(args.corruptions)
        args.result_dir = args.result_dir.format( args.arch, args.dataset, args.corruptions )

        epoch_result_list, _ = eval(args=args, )
        if corr_id == 0:
            f_write = get_writer_to_all_result(args)
        f_write.write(' '.join([str(round(float(xx), 3)) for xx in epoch_result_list]) + '\n')

        f_write.flush()
        if corr_id == len(corruptions) - 1:
            f_write.close()
