import io
import mmcv
from mmcv.fileio import FileClient
import numpy as np
import warnings
import random
from torch.nn.modules.utils import _pair
from collections.abc import Sequence
from mmcv.parallel import DataContainer as DC
import torch

import warnings


def _init_lazy_if_proper(results, lazy):
    """Initialize lazy operation properly.

    Make sure that a lazy operation is properly initialized,
    and avoid a non-lazy operation accidentally getting mixed in.

    Required keys in results are "imgs" if "img_shape" not in results,
    otherwise, Required keys in results are "img_shape", add or modified keys
    are "img_shape", "lazy".
    Add or modified keys in "lazy" are "original_shape", "crop_bbox", "flip",
    "flip_direction", "interpolation".

    Args:
        results (dict): A dict stores data pipeline result.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    if 'img_shape' not in results:
        results['img_shape'] = results['imgs'][0].shape[:2]
    if lazy:
        if 'lazy' not in results:
            img_h, img_w = results['img_shape']
            lazyop = dict()
            lazyop['original_shape'] = results['img_shape']
            lazyop['crop_bbox'] = np.array([0, 0, img_w, img_h],
                                           dtype=np.float32)
            lazyop['flip'] = False
            lazyop['flip_direction'] = None
            lazyop['interpolation'] = None
            results['lazy'] = lazyop
    else:
        assert 'lazy' not in results, 'Use Fuse after lazy operations'

class RandomCrop:
    """Vanilla square random crop that specifics the output size.

    Required keys in results are "img_shape", "keypoint" (optional), "imgs"
    (optional), added or modified keys are "keypoint", "imgs", "lazy"; Required
    keys in "lazy" are "flip", "crop_bbox", added or modified key is
    "crop_bbox".

    Args:
        size (int): The output size of the images.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self, size, lazy=False):
        if not isinstance(size, int):
            raise TypeError(f'Size must be an int, but got {type(size)}')
        self.size = size
        self.lazy = lazy

    @staticmethod
    def _crop_kps(kps, crop_bbox):
        return kps - crop_bbox[:2]

    @staticmethod
    def _crop_imgs(imgs, crop_bbox):
        x1, y1, x2, y2 = crop_bbox
        return [img[y1:y2, x1:x2] for img in imgs]

    @staticmethod
    def _box_crop(box, crop_bbox):
        """Crop the bounding boxes according to the crop_bbox.

        Args:
            box (np.ndarray): The bounding boxes.
            crop_bbox(np.ndarray): The bbox used to crop the original image.
        """

        x1, y1, x2, y2 = crop_bbox
        img_w, img_h = x2 - x1, y2 - y1

        box_ = box.copy()
        box_[..., 0::2] = np.clip(box[..., 0::2] - x1, 0, img_w - 1)
        box_[..., 1::2] = np.clip(box[..., 1::2] - y1, 0, img_h - 1)
        return box_

    def _all_box_crop(self, results, crop_bbox):
        """Crop the gt_bboxes and proposals in results according to crop_bbox.

        Args:
            results (dict): All information about the sample, which contain
                'gt_bboxes' and 'proposals' (optional).
            crop_bbox(np.ndarray): The bbox used to crop the original image.
        """
        results['gt_bboxes'] = self._box_crop(results['gt_bboxes'], crop_bbox)
        if 'proposals' in results and results['proposals'] is not None:
            assert results['proposals'].shape[1] == 4
            results['proposals'] = self._box_crop(results['proposals'],
                                                  crop_bbox)
        return results

    def __call__(self, results):
        """Performs the RandomCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)
        if 'keypoint' in results:
            assert not self.lazy, ('Keypoint Augmentations are not compatible '
                                   'with lazy == True')

        img_h, img_w = results['img_shape']
        assert self.size <= img_h and self.size <= img_w

        y_offset = 0
        x_offset = 0
        if img_h > self.size:
            y_offset = int(np.random.randint(0, img_h - self.size))
        if img_w > self.size:
            x_offset = int(np.random.randint(0, img_w - self.size))

        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        x_ratio, y_ratio = x_offset / img_w, y_offset / img_h
        w_ratio, h_ratio = self.size / img_w, self.size / img_h

        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_x_ratio
        ]
        results['crop_quadruple'] = np.array(
            new_crop_quadruple, dtype=np.float32)

        new_h, new_w = self.size, self.size

        crop_bbox = np.array(
            [x_offset, y_offset, x_offset + new_w, y_offset + new_h])
        results['crop_bbox'] = crop_bbox

        results['img_shape'] = (new_h, new_w)

        if not self.lazy:
            if 'keypoint' in results:
                results['keypoint'] = self._crop_kps(results['keypoint'],
                                                     crop_bbox)
            if 'imgs' in results:
                results['imgs'] = self._crop_imgs(results['imgs'], crop_bbox)
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = x_offset * (lazy_right - lazy_left) / img_w
            right = (x_offset + new_w) * (lazy_right - lazy_left) / img_w
            top = y_offset * (lazy_bottom - lazy_top) / img_h
            bottom = (y_offset + new_h) * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        # Process entity boxes
        if 'gt_bboxes' in results:
            assert not self.lazy
            results = self._all_box_crop(results, results['crop_bbox'])

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(size={self.size}, '
                    f'lazy={self.lazy})')
        return repr_str

def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    if isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    if isinstance(data, int):
        return torch.LongTensor([data])
    if isinstance(data, float):
        return torch.FloatTensor([data])
    raise TypeError(f'type {type(data)} cannot be converted to tensor.')

# class DecordInit:
#     """Using decord to initialize the video_reader.
#
#     Decord: https://github.com/dmlc/decord
#
#     Required keys are "filename",
#     added or modified keys are "video_reader" and "total_frames".
#     """
#
#     def __init__(self, io_backend='disk', num_threads=1, **kwargs):
#         self.io_backend = io_backend
#         self.num_threads = num_threads
#         self.kwargs = kwargs
#         self.file_client = None
#
#     def __call__(self, filename):
#         """Perform the Decord initialization.
#
#         Args:
#             results (dict): The resulting dict to be modified and passed
#                 to the next transform in pipeline.
#         """
#         try:
#             import decord
#         except ImportError:
#             raise ImportError(
#                 'Please run "pip install decord" to install Decord first.')
#
#         if self.file_client is None:
#             self.file_client = FileClient(self.io_backend, **self.kwargs)
#
#         file_obj = io.BytesIO(self.file_client.get(filename))
#         container = decord.VideoReader(file_obj, num_threads=self.num_threads)
#         # results['video_reader'] = container
#         # results['total_frames'] = len(container)
#         return container
#
#     def __repr__(self):
#         repr_str = (f'{self.__class__.__name__}('
#                     f'io_backend={self.io_backend}, '
#                     f'num_threads={self.num_threads})')
#         return repr_str
#
# class SampleFrames:
#     """Sample frames from the video.
#
#     Required keys are "total_frames", "start_index" , added or modified keys
#     are "frame_inds", "frame_interval" and "num_clips".
#
#     Args:
#         clip_len (int): Frames of each sampled output clip.
#         frame_interval (int): Temporal interval of adjacent sampled frames.
#             Default: 1.
#         num_clips (int): Number of clips to be sampled. Default: 1.
#         temporal_jitter (bool): Whether to apply temporal jittering.
#             Default: False.
#         twice_sample (bool): Whether to use twice sample when testing.
#             If set to True, it will sample frames with and without fixed shift,
#             which is commonly used for testing in TSM model. Default: False.
#         out_of_bound_opt (str): The way to deal with out of bounds frame
#             indexes. Available options are 'loop', 'repeat_last'.
#             Default: 'loop'.
#         test_mode (bool): Store True when building test or validation dataset.
#             Default: False.
#         start_index (None): This argument is deprecated and moved to dataset
#             class (``BaseDataset``, ``VideoDatset``, ``RawframeDataset``, etc),
#             see this: https://github.com/open-mmlab/mmaction2/pull/89.
#     """
#
#     def __init__(self,
#                  clip_len,
#                  frame_interval=1,
#                  num_clips=1,
#                  temporal_jitter=False,
#                  twice_sample=False,
#                  out_of_bound_opt='loop',
#                  test_mode=False,
#                  start_index=None,
#                  frame_uniform=False):
#
#         self.clip_len = clip_len
#         self.frame_interval = frame_interval
#         self.num_clips = num_clips
#         self.temporal_jitter = temporal_jitter
#         self.twice_sample = twice_sample
#         self.out_of_bound_opt = out_of_bound_opt
#         self.test_mode = test_mode
#         self.frame_uniform = frame_uniform
#         assert self.out_of_bound_opt in ['loop', 'repeat_last']
#
#         if start_index is not None:
#             warnings.warn('No longer support "start_index" in "SampleFrames", '
#                           'it should be set in dataset class, see this pr: '
#                           'https://github.com/open-mmlab/mmaction2/pull/89')
#
#     def _get_train_clips(self, num_frames):
#         """Get clip offsets in train mode.
#
#         It will calculate the average interval for selected frames,
#         and randomly shift them within offsets between [0, avg_interval].
#         If the total number of frames is smaller than clips num or origin
#         frames length, it will return all zero indices.
#
#         Args:
#             num_frames (int): Total number of frame in the video.
#
#         Returns:
#             np.ndarray: Sampled frame indices in train mode.
#         """
#         ori_clip_len = self.clip_len * self.frame_interval
#         avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips
#
#         if avg_interval > 0:
#             base_offsets = np.arange(self.num_clips) * avg_interval
#             clip_offsets = base_offsets + np.random.randint(
#                 avg_interval, size=self.num_clips)
#         elif num_frames > max(self.num_clips, ori_clip_len):
#             clip_offsets = np.sort(
#                 np.random.randint(
#                     num_frames - ori_clip_len + 1, size=self.num_clips))
#         elif avg_interval == 0:
#             ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
#             clip_offsets = np.around(np.arange(self.num_clips) * ratio)
#         else:
#             clip_offsets = np.zeros((self.num_clips, ), dtype=np.int)
#
#         return clip_offsets
#
#     def _get_test_clips(self, num_frames):
#         """Get clip offsets in test mode.
#
#         Calculate the average interval for selected frames, and shift them
#         fixedly by avg_interval/2. If set twice_sample True, it will sample
#         frames together without fixed shift. If the total number of frames is
#         not enough, it will return all zero indices.
#
#         Args:
#             num_frames (int): Total number of frame in the video.
#
#         Returns:
#             np.ndarray: Sampled frame indices in test mode.
#         """
#         ori_clip_len = self.clip_len * self.frame_interval   # clip_len and frame_interval are defined in config file,
#         avg_interval = (num_frames - ori_clip_len + 1) / float(self.num_clips)  # 59.25
#         if num_frames > ori_clip_len - 1:
#             base_offsets = np.arange(self.num_clips) * avg_interval #  0, 59.25, 118.5, 177.75
#             clip_offsets = (base_offsets + avg_interval / 2.0).astype(np.int)  # 29, 88, 148, 207
#             if self.twice_sample:
#                 clip_offsets = np.concatenate([clip_offsets, base_offsets])
#         else:
#             clip_offsets = np.zeros((self.num_clips, ), dtype=np.int)
#         return clip_offsets
#
#     def _sample_clips(self, num_frames):
#         """Choose clip offsets for the video in a given mode.
#
#         Args:
#             num_frames (int): Total number of frame in the video.
#
#         Returns:
#             np.ndarray: Sampled frame indices.
#         """
#         if self.test_mode:
#             clip_offsets = self._get_test_clips(num_frames)
#         else:
#             clip_offsets = self._get_train_clips(num_frames)
#
#         return clip_offsets
#
#     def get_seq_frames(self, num_frames):
#         """
#         Modified from https://github.com/facebookresearch/SlowFast/blob/64abcc90ccfdcbb11cf91d6e525bed60e92a8796/slowfast/datasets/ssv2.py#L159
#         Given the video index, return the list of sampled frame indexes.
#         Args:
#             num_frames (int): Total number of frame in the video.
#         Returns:
#             seq (list): the indexes of frames of sampled from the video.
#         """
#         seg_size = float(num_frames - 1) / self.clip_len
#         seq = []
#         for i in range(self.clip_len):
#             start = int(np.round(seg_size * i))
#             end = int(np.round(seg_size * (i + 1)))
#             if not self.test_mode:
#                 seq.append(random.randint(start, end))
#             else:
#                 seq.append((start + end) // 2)
#
#         return np.array(seq)
#
#     def __call__(self,  total_frames, start_index = 0):
#         """Perform the SampleFrames loading.
#
#         Args:
#             results (dict): The resulting dict to be modified and passed
#                 to the next transform in pipeline.
#         """
#         # total_frames = results['total_frames']  # this is assigned in  DecordInit
#         if self.frame_uniform:  # sthv2 sampling strategy  todo uniform sampling instead dense sampling
#             assert start_index == 0
#             frame_inds = self.get_seq_frames(total_frames)
#         else:
#             clip_offsets = self._sample_clips(total_frames)  #  # 29, 88, 148, 207
#             frame_inds = clip_offsets[:, None] + np.arange(
#                 self.clip_len)[None, :] * self.frame_interval
#             frame_inds = np.concatenate(frame_inds)
#             # frame_inds  (4, 32),  the area for frame sampling is in the middle, there is space of avg_interval_len/2  in the beginning and in the end
#             if self.temporal_jitter:
#                 perframe_offsets = np.random.randint(
#                     self.frame_interval, size=len(frame_inds))
#                 frame_inds += perframe_offsets
#
#             frame_inds = frame_inds.reshape((-1, self.clip_len))
#             if self.out_of_bound_opt == 'loop':
#                 frame_inds = np.mod(frame_inds, total_frames)
#             elif self.out_of_bound_opt == 'repeat_last':
#                 safe_inds = frame_inds < total_frames
#                 unsafe_inds = 1 - safe_inds
#                 last_ind = np.max(safe_inds * frame_inds, axis=1)
#                 new_inds = (safe_inds * frame_inds + (unsafe_inds.T * last_ind).T)
#                 frame_inds = new_inds
#             else:
#                 raise ValueError('Illegal out_of_bound option.')
#
#             # start_index = results['start_index']
#             frame_inds = np.concatenate(frame_inds) + start_index
#
#         return frame_inds.astype(np.int)
#         # results['frame_inds'] = frame_inds.astype(np.int)
#         # results['clip_len'] = self.clip_len
#         # results['frame_interval'] = self.frame_interval
#         # results['num_clips'] = self.num_clips
#         # return results
#
#     def __repr__(self):
#         repr_str = (f'{self.__class__.__name__}('
#                     f'clip_len={self.clip_len}, '
#                     f'frame_interval={self.frame_interval}, '
#                     f'num_clips={self.num_clips}, '
#                     f'temporal_jitter={self.temporal_jitter}, '
#                     f'twice_sample={self.twice_sample}, '
#                     f'out_of_bound_opt={self.out_of_bound_opt}, '
#                     f'test_mode={self.test_mode})')
#         return repr_str
#
# class DecordDecode:
#     """Using decord to decode the video.
#
#     Decord: https://github.com/dmlc/decord
#
#     Required keys are "video_reader", "filename" and "frame_inds",
#     added or modified keys are "imgs" and "original_shape".
#     """
#
#     def __call__(self, video_reader, frame_inds ):
#         """Perform the Decord decoding.
#
#         Args:
#             results (dict): The resulting dict to be modified and passed
#                 to the next transform in pipeline.
#         """
#         container = video_reader  # video reader that reads the video
#
#         if frame_inds.ndim != 1:
#             frame_inds = np.squeeze(frame_inds)
#
#         # frame_inds = results['frame_inds'] #  indices of frames to sample
#         # Generate frame index mapping in order
#         frame_dict = {
#             idx: container[idx].asnumpy()
#             for idx in np.unique(frame_inds)  # there could be repetitive frame indices in frame_inds
#         }  # the frames are read in the original resolution
#         # frame_dict contains the unique frames,  there might be duplicated frames in different segments
#         imgs = [frame_dict[idx] for idx in frame_inds]  # sampled frames   every frame is  (360, 640, 3)
#
#         # results['video_reader'] = None  #  clear video reader
#         del container
#
#         # results['imgs'] = imgs
#         # results['original_shape'] = imgs[0].shape[:2]  # (360, 640)
#         # results['img_shape'] = imgs[0].shape[:2]  # (360, 640)
#
#         return imgs, imgs[0].shape[:2], frame_inds
#
# class Resize:
#     """Resize images to a specific size.
#
#     Required keys are "img_shape", "modality", "imgs" (optional), "keypoint"
#     (optional), added or modified keys are "imgs", "img_shape", "keep_ratio",
#     "scale_factor", "lazy", "resize_size". Required keys in "lazy" is None,
#     added or modified key is "interpolation".
#
#     Args:
#         scale (float | Tuple[int]): If keep_ratio is True, it serves as scaling
#             factor or maximum size:
#             If it is a float number, the image will be rescaled by this
#             factor, else if it is a tuple of 2 integers, the image will
#             be rescaled as large as possible within the scale.
#             Otherwise, it serves as (w, h) of output size.
#         keep_ratio (bool): If set to True, Images will be resized without
#             changing the aspect ratio. Otherwise, it will resize images to a
#             given size. Default: True.
#         interpolation (str): Algorithm used for interpolation:
#             "nearest" | "bilinear". Default: "bilinear".
#         lazy (bool): Determine whether to apply lazy operation. Default: False.
#     """
#
#     def __init__(self,
#                  scale,
#                  keep_ratio=True,
#                  interpolation='bilinear',
#                  lazy=False):
#         if isinstance(scale, float):
#             if scale <= 0:
#                 raise ValueError(f'Invalid scale {scale}, must be positive.')
#         elif isinstance(scale, tuple):
#             max_long_edge = max(scale)
#             max_short_edge = min(scale)
#             if max_short_edge == -1:
#                 # assign np.inf to long edge for rescaling short edge later.
#                 scale = (np.inf, max_long_edge)
#         else:
#             raise TypeError(
#                 f'Scale must be float or tuple of int, but got {type(scale)}')
#         self.scale = scale
#         self.keep_ratio = keep_ratio
#         self.interpolation = interpolation
#         self.lazy = lazy
#
#     def _resize_imgs(self, imgs, new_w, new_h):
#         return [
#             mmcv.imresize(
#                 img, (new_w, new_h), interpolation=self.interpolation)
#             for img in imgs
#         ]
#
#     @staticmethod
#     def _resize_kps(kps, scale_factor):
#         return kps * scale_factor
#
#     @staticmethod
#     def _box_resize(box, scale_factor):
#         """Rescale the bounding boxes according to the scale_factor.
#
#         Args:
#             box (np.ndarray): The bounding boxes.
#             scale_factor (np.ndarray): The scale factor used for rescaling.
#         """
#         assert len(scale_factor) == 2
#         scale_factor = np.concatenate([scale_factor, scale_factor])
#         return box * scale_factor
#
#     def __call__(self,  img_shape, ):
#         """Performs the Resize augmentation.
#
#         Args:
#             results (dict): The resulting dict to be modified and passed
#                 to the next transform in pipeline.
#         """
#         if self.lazy:
#             _init_lazy_if_proper(results, self.lazy)
#         # if 'keypoint' in results:
#         #     assert not self.lazy, ('Keypoint Augmentations are not compatible '
#         #                            'with lazy == True')
#
#         # if 'scale_factor' not in results:
#         #     results['scale_factor'] = np.array([1, 1], dtype=np.float32)
#         scale_factor = np.array([1, 1], dtype=np.float32)
#         img_h, img_w = img_shape
#
#         if self.keep_ratio:
#             new_w, new_h = mmcv.rescale_size((img_w, img_h), self.scale)  # Calculate the new size to be rescaled to.
#         else:
#             new_w, new_h = self.scale
#
#         self.scale_factor = np.array([new_w / img_w, new_h / img_h],
#                                      dtype=np.float32)
#
#         img_shape = (new_h, new_w)
#         keep_ratio = self.keep_ratio
#         scale_factor = scale_factor * self.scale_factor
#
#         if not self.lazy:
#             if 'imgs' in results:
#                 results['imgs'] = self._resize_imgs(results['imgs'], new_w,
#                                                     new_h)  # resize image
#             if 'keypoint' in results:
#                 results['keypoint'] = self._resize_kps(results['keypoint'],
#                                                        self.scale_factor)
#         else:
#             lazyop = results['lazy']
#             if lazyop['flip']:
#                 raise NotImplementedError('Put Flip at last for now')
#             lazyop['interpolation'] = self.interpolation
#
#         # if 'gt_bboxes' in results:
#         #     assert not self.lazy
#         #     results['gt_bboxes'] = self._box_resize(results['gt_bboxes'],
#         #                                             self.scale_factor)
#         #     if 'proposals' in results and results['proposals'] is not None:
#         #         assert results['proposals'].shape[1] == 4
#         #         results['proposals'] = self._box_resize(
#         #             results['proposals'], self.scale_factor)
#
#         return results
#
#     def __repr__(self):
#         repr_str = (f'{self.__class__.__name__}('
#                     f'scale={self.scale}, keep_ratio={self.keep_ratio}, '
#                     f'interpolation={self.interpolation}, '
#                     f'lazy={self.lazy})')
#         return repr_str

class DecordInit:
    """Using decord to initialize the video_reader.

    Decord: https://github.com/dmlc/decord

    Required keys are "filename",
    added or modified keys are "video_reader" and "total_frames".
    """

    def __init__(self, io_backend='disk', num_threads=1, **kwargs):
        self.io_backend = io_backend
        self.num_threads = num_threads
        self.kwargs = kwargs
        self.file_client = None

    def __call__(self, results):
        """Perform the Decord initialization.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        try:
            import decord
        except ImportError:
            raise ImportError(
                'Please run "pip install decord" to install Decord first.')

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        file_obj = io.BytesIO(self.file_client.get(results['filename']))
        container = decord.VideoReader(file_obj, num_threads=self.num_threads)
        results['video_reader'] = container
        results['total_frames'] = len(container)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend}, '
                    f'num_threads={self.num_threads})')
        return repr_str

class SampleFrames:
    """Sample frames from the video.

    Required keys are "total_frames", "start_index" , added or modified keys
    are "frame_inds", "frame_interval" and "num_clips".

    Args:
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 1.
        num_clips (int): Number of clips to be sampled. Default: 1.
        temporal_jitter (bool): Whether to apply temporal jittering.
            Default: False.
        twice_sample (bool): Whether to use twice sample when testing.
            If set to True, it will sample frames with and without fixed shift,
            which is commonly used for testing in TSM model. Default: False.
        out_of_bound_opt (str): The way to deal with out of bounds frame
            indexes. Available options are 'loop', 'repeat_last'.
            Default: 'loop'.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        start_index (None): This argument is deprecated and moved to dataset
            class (``BaseDataset``, ``VideoDatset``, ``RawframeDataset``, etc),
            see this: https://github.com/open-mmlab/mmaction2/pull/89.
    """

    def __init__(self,
                 clip_len,
                 frame_interval=1,
                 num_clips=1,
                 temporal_jitter=False,
                 twice_sample=False,
                 out_of_bound_opt='loop',
                 test_mode=False,
                 start_index=None,
                 frame_uniform=False):

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.temporal_jitter = temporal_jitter
        self.twice_sample = twice_sample
        self.out_of_bound_opt = out_of_bound_opt
        self.test_mode = test_mode
        self.frame_uniform = frame_uniform
        assert self.out_of_bound_opt in ['loop', 'repeat_last']

        if start_index is not None:
            warnings.warn('No longer support "start_index" in "SampleFrames", '
                          'it should be set in dataset class, see this pr: '
                          'https://github.com/open-mmlab/mmaction2/pull/89')

    def _get_train_clips(self, num_frames):
        """Get clip offsets in train mode.

        It will calculate the average interval for selected frames,
        and randomly shift them within offsets between [0, avg_interval].
        If the total number of frames is smaller than clips num or origin
        frames length, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

        if avg_interval > 0:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = base_offsets + np.random.randint(
                avg_interval, size=self.num_clips)
        elif num_frames > max(self.num_clips, ori_clip_len):
            clip_offsets = np.sort(
                np.random.randint(
                    num_frames - ori_clip_len + 1, size=self.num_clips))
        elif avg_interval == 0:
            ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
            clip_offsets = np.around(np.arange(self.num_clips) * ratio)
        else:
            clip_offsets = np.zeros((self.num_clips, ), dtype=np.int)

        return clip_offsets

    def _get_test_clips(self, num_frames):
        """Get clip offsets in test mode.

        Calculate the average interval for selected frames, and shift them
        fixedly by avg_interval/2. If set twice_sample True, it will sample
        frames together without fixed shift. If the total number of frames is
        not enough, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval   # clip_len and frame_interval are defined in config file,
        avg_interval = (num_frames - ori_clip_len + 1) / float(self.num_clips)  # 59.25
        if num_frames > ori_clip_len - 1:
            base_offsets = np.arange(self.num_clips) * avg_interval #  0, 59.25, 118.5, 177.75
            clip_offsets = (base_offsets + avg_interval / 2.0).astype(np.int)  # 29, 88, 148, 207
            if self.twice_sample:
                clip_offsets = np.concatenate([clip_offsets, base_offsets])
        else:
            clip_offsets = np.zeros((self.num_clips, ), dtype=np.int)
        return clip_offsets

    def _sample_clips(self, num_frames):
        """Choose clip offsets for the video in a given mode.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices.
        """
        if self.test_mode:
            clip_offsets = self._get_test_clips(num_frames)
        else:
            clip_offsets = self._get_train_clips(num_frames)

        return clip_offsets

    def get_seq_frames(self, num_frames):
        """
        Modified from https://github.com/facebookresearch/SlowFast/blob/64abcc90ccfdcbb11cf91d6e525bed60e92a8796/slowfast/datasets_/ssv2.py#L159
        Given the video index, return the list of sampled frame indexes.
        Args:
            num_frames (int): Total number of frame in the video.
        Returns:
            seq (list): the indexes of frames of sampled from the video.
        """
        seg_size = float(num_frames - 1) / self.clip_len
        seq = []
        for i in range(self.clip_len):
            start = int(np.round(seg_size * i))
            end = int(np.round(seg_size * (i + 1)))
            if not self.test_mode:
                seq.append(random.randint(start, end))
            else:
                seq.append((start + end) // 2)

        return np.array(seq)

    def __call__(self, results):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results['total_frames']  # this is assigned in  DecordInit
        if self.frame_uniform:  # sthv2 sampling strategy  todo uniform sampling instead dense sampling
            assert results['start_index'] == 0
            frame_inds = self.get_seq_frames(total_frames)
        else:
            clip_offsets = self._sample_clips(total_frames)  #  # 29, 88, 148, 207
            frame_inds = clip_offsets[:, None] + np.arange(
                self.clip_len)[None, :] * self.frame_interval
            frame_inds = np.concatenate(frame_inds)
            # frame_inds  (4, 32),  the area for frame sampling is in the middle, there is space of avg_interval_len/2  in the beginning and in the end
            if self.temporal_jitter:
                perframe_offsets = np.random.randint(
                    self.frame_interval, size=len(frame_inds))
                frame_inds += perframe_offsets

            frame_inds = frame_inds.reshape((-1, self.clip_len))
            if self.out_of_bound_opt == 'loop':
                frame_inds = np.mod(frame_inds, total_frames)
            elif self.out_of_bound_opt == 'repeat_last':
                safe_inds = frame_inds < total_frames
                unsafe_inds = 1 - safe_inds
                last_ind = np.max(safe_inds * frame_inds, axis=1)
                new_inds = (safe_inds * frame_inds + (unsafe_inds.T * last_ind).T)
                frame_inds = new_inds
            else:
                raise ValueError('Illegal out_of_bound option.')

            start_index = results['start_index']
            frame_inds = np.concatenate(frame_inds) + start_index

        results['frame_inds'] = frame_inds.astype(np.int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = self.num_clips
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval}, '
                    f'num_clips={self.num_clips}, '
                    f'temporal_jitter={self.temporal_jitter}, '
                    f'twice_sample={self.twice_sample}, '
                    f'out_of_bound_opt={self.out_of_bound_opt}, '
                    f'test_mode={self.test_mode})')
        return repr_str