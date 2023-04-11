
import os.path as osp
from models.tanet_models.video_dataset import VideoRecord
import torch.utils.data as data
from models.videoswintransformer_models.transforms_backup import DecordInit, SampleFrames, DecordDecode, Resize, CenterCrop, Flip, Normalize, FormatShape, Collect, ToTensor, \
    RandomResizedCrop
# import decord
class Video_SwinDataset(data.Dataset):
    def __init__(self, list_file,
                 num_segments=3,  # clip_length
                 frame_interval = 2,
                 num_clips = 1,
                 frame_uniform = True,
                 test_mode = False,
                 flip_ratio = None,
                 scale_size = None,
                 input_size=None,
                 img_norm_cfg  = None,

                 vid_format='.mp4',
                 video_data_dir = None,
                 remove_missing = False,
                 if_sample_tta_aug_views=None,
                 tta_view_sample_style_list=None,
                 n_augmented_views = None,
                 debug = False, debug_vid = 50,
                 ):
        self.list_file = list_file
        self.num_segments = num_segments
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.frame_uniform = frame_uniform
        self.test_mode = test_mode
        self.flip_ratio = flip_ratio
        if self.test_mode:
            assert self.flip_ratio == 0
        self.scale_size = scale_size
        self.input_size = input_size
        self.img_norm_cfg = img_norm_cfg

        self.vid_format = vid_format
        self.video_data_dir = video_data_dir
        self.remove_missing = remove_missing
        self.debug = debug
        self.debug_vid = debug_vid
        self.if_sample_tta_aug_views = if_sample_tta_aug_views
        self.tta_view_sample_style_list = tta_view_sample_style_list
        self.n_augmented_views = n_augmented_views
        self.__parse_list()
    def __parse_list(self):
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        if not self.test_mode or self.remove_missing:
            tmp = [item for item in tmp if int(item[1]) >= 3]
        self.video_list = [VideoRecord(item) for item in tmp]

        if self.debug:
            self.video_list = self.video_list[:self.debug_vid]

    def __getitem__(self, index):
        record = self.video_list[index]
        vid_path = osp.join(self.video_data_dir, f'{record.path}{self.vid_format}')
        results = { 'filename': vid_path, 'start_index':0, 'modality': 'RGB'}
        if self.test_mode:
            if self.if_sample_tta_aug_views:
                func_list = [DecordInit(),
                             SampleFrames(clip_len=self.num_segments, frame_interval=self.frame_interval, # todo frame_interval is only used for dense sampling
                                          num_clips=self.num_clips,
                                          frame_uniform=self.frame_uniform, test_mode=self.test_mode,
                                          if_sample_tta_aug_views= self.if_sample_tta_aug_views, tta_view_sample_style_list= self.tta_view_sample_style_list,
                                          n_augmented_views= self.n_augmented_views
                                          ), # todo uniform sampling (instead of dense sampling)
                             DecordDecode(),
                             Resize( scale=(-1, self.scale_size)), # todo always resize the height to 224
                             RandomResizedCrop(),
                             Resize(scale= (self.input_size, self.input_size), keep_ratio= False ),
                             # CenterCrop(crop_size=( self.input_size)),
                             Flip(flip_ratio= self.flip_ratio),
                             Normalize(**self.img_norm_cfg),
                             FormatShape(input_format='NCTHW' ), # , collapse=True  #  todo collapse = False default,     (n_clips,  3,  T, H, W )
                             Collect(keys=['imgs'], meta_keys=[] ),
                             ToTensor(keys=['imgs'])
                             ]
            else:
                func_list = [DecordInit(),
                             SampleFrames(clip_len=self.num_segments, frame_interval=self.frame_interval,
                                          # todo frame_interval is only used for dense sampling
                                          num_clips=self.num_clips,
                                          frame_uniform=self.frame_uniform, test_mode=self.test_mode,
                                          if_sample_tta_aug_views=self.if_sample_tta_aug_views,
                                          tta_view_sample_style_list=self.tta_view_sample_style_list,
                                          n_augmented_views=self.n_augmented_views
                                          ),  # todo uniform sampling (instead of dense sampling)
                             DecordDecode(),
                             Resize(scale=(-1, self.scale_size)),  # todo always resize the height to 224
                             CenterCrop(crop_size=(self.input_size)),
                             Flip(flip_ratio=self.flip_ratio),
                             Normalize(**self.img_norm_cfg),
                             FormatShape(input_format='NCTHW'),
                             # , collapse=True  #  todo collapse = False default,     (n_clips,  3,  T, H, W )
                             Collect(keys=['imgs'], meta_keys=[]),
                             ToTensor(keys=['imgs'])
                             ]
        else:
            raise NotImplementedError('Transformation for training not implemented ')
        for func_ in func_list:
            results = func_(results)
        return results['imgs'], record.label
    def __len__(self):
        return len(self.video_list)

    def get(self, record, indices):
        pass