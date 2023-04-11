import torch.utils.data as data

from PIL import Image
import numpy as np
import os
import os.path as osp
from numpy.random import randint
import cv2
import decord
import random


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class MyVideoDataset(data.Dataset):
    """
    sample several clips of consecutive frames
    """
    def __init__(self, root_path, list_file, clip_length=64, frame_interval=1, num_clips=1, frame_size=(320, 240),
                 modality='RGB', vid_format='.mp4',
                 transform=None, test_mode=False, video_data_dir=None, debug=False, debug_vid=50):
        self.root_path = root_path
        self.list_file = list_file
        self.clip_len = clip_length
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        # self.frame_size = frame_size
        self.modality = modality
        # self.image_tmpl = image_tmpl
        self.vid_format = vid_format
        self.transform = transform
        # self.random_shift = random_shift
        self.test_mode = test_mode
        self.video_data_dir = video_data_dir

        self.debug = debug
        self.debug_vid = debug_vid

        self._parse_list()

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in
                           open(os.path.join(self.list_file))]
        if self.debug:
            self.video_list = self.video_list[:self.debug_vid]

    def _sample_indices_old(self, record):
        if not self.test_mode and self.random_shift:
            average_duration = record.num_frames // self.clip_len
            if average_duration > 0:
                # uniformly divide and then randomly sample from each segment
                offsets = np.sort(
                    np.multiply(list(range(self.clip_len)), average_duration) + randint(average_duration,
                                                                                        size=self.clip_len))
            else:
                # randomly sample with repetitions
                offsets = np.sort(randint(record.num_frames, size=self.clip_len))
        else:  # equi-distant sampling
            tick = record.num_frames / float(self.clip_len)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.clip_len)])
        return offsets + 1

    def _get_train_clips(self, num_frames):
        ori_clip_len = self.clip_len * self.frame_interval
        # the average interval between two clips (the clip can only be sampled within the video,  last start position is   (num_frames - ori_clip_len +1)
        avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips
        if avg_interval > 0:  # randomly sample the starting position of each clip
            base_offsets = np.arange(self.num_clips) * avg_interval  # starting positions of all clips
            clip_offsets = base_offsets + np.random.randint(
                avg_interval, size=self.num_clips)  # randomly shift each starting posiiton within one interval
        elif num_frames > max(self.num_clips, ori_clip_len):
            clip_offsets = np.sort(
                np.random.randint(
                    num_frames - ori_clip_len + 1,
                    size=self.num_clips))  # in the interval of (0, num_frames - ori_clip_len + 1), randomly choose 4 starting positions
        elif avg_interval == 0:
            ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
            clip_offsets = np.around(np.arange(self.num_clips) * ratio)
        else:
            clip_offsets = np.zeros((self.num_clips,), dtype=np.int)
        return clip_offsets

    def _get_test_clips(self, num_frames):  # uniformly sample the starting position of each clip
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) / float(self.num_clips)
        if num_frames > ori_clip_len - 1:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = (base_offsets + avg_interval / 2.0).astype(np.int)
        else:
            clip_offsets = np.zeros((self.num_clips,), dtype=np.int)
        return clip_offsets

    def _sample_clips(self, num_frames):
        if self.test_mode:
            clip_offsets = self._get_test_clips(num_frames)
        else:
            clip_offsets = self._get_train_clips(num_frames)
        return clip_offsets

    def _sample_indices(self, record):
        num_frames = record.num_frames
        clip_offsets = self._sample_clips(num_frames)
        frame_inds = clip_offsets[:, None] + np.arange(
            self.clip_len)[None, :] * self.frame_interval
        frame_inds = np.concatenate(frame_inds)
        frame_inds = frame_inds.reshape((-1, self.clip_len))  # (clip_len, )  ->  (n_clips, clip_len )
        frame_inds = np.mod(frame_inds, num_frames)
        return frame_inds

    def __getitem__(self, index):
        record = self.video_list[index]  # video file
        indices = self._sample_indices(record)  # get the frame indices
        return self.get(record, indices)

    def get(self, record, indices):  # indices (n_clips, clip_len )
        vid_path = os.path.join(self.video_data_dir, f'{record.path}{self.vid_format}')
        container = decord.VideoReader(vid_path)
        frame_indices = np.concatenate(indices)  # flatten the frame indices
        # accurate mode
        images = container.get_batch(frame_indices).asnumpy()
        images = list(images)
        images = [Image.fromarray(image).convert('RGB') for image in images]

        # process_data = self.transform(images)
        # return process_data, record.label
        process_data, label = self.transform( (images, record.label) )
        return process_data, label

    def __len__(self):
        return len(self.video_list)


class MyTSNVideoDataset(data.Dataset):
    """
    sample 1 clip in uniform sampling (TSN style )
    """
    def __init__(self, args, root_path, list_file, clip_length=64, frame_interval=1, num_clips=1, frame_size=(320, 240),
                 modality='RGB', image_tmpl='img_{:05d}.jpg', vid_format='.mp4',
                 transform=None, random_shift=True, test_mode=False, video_data_dir=None, debug=False, debug_vid=50):
        self.root_path = root_path
        self.list_file = list_file
        self.clip_len = clip_length
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        # self.frame_size = frame_size
        self.modality = modality
        # self.image_tmpl = image_tmpl
        self.vid_format = vid_format
        self.transform = transform
        # self.random_shift = random_shift
        self.test_mode = test_mode
        self.video_data_dir = video_data_dir

        self.debug = debug
        self.debug_vid = debug_vid

        self._parse_list()
        self.args = args

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in
                           open(os.path.join(self.list_file))]

        if self.debug:
            self.video_list = self.video_list[:self.debug_vid]

    def _sample_indices_old(self, record):
        if not self.test_mode and self.random_shift:
            average_duration = record.num_frames // self.clip_len
            if average_duration > 0:
                # uniformly divide and then randomly sample from each segment
                offsets = np.sort(
                    np.multiply(list(range(self.clip_len)), average_duration) + randint(average_duration,
                                                                                        size=self.clip_len))
            else:
                # randomly sample with repetitions
                offsets = np.sort(randint(record.num_frames, size=self.clip_len))
        else:  # equi-distant sampling
            tick = record.num_frames / float(self.clip_len)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.clip_len)])
        return offsets + 1

    def _get_train_clips(self, num_frames):
        ori_clip_len = self.clip_len * self.frame_interval
        # the average interval between two clips (the clip can only be sampled within the video,  last start position is   (num_frames - ori_clip_len +1)
        avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips
        if avg_interval > 0:  # randomly sample the starting position of each clip
            base_offsets = np.arange(self.num_clips) * avg_interval  # starting positions of all clips
            clip_offsets = base_offsets + np.random.randint(
                avg_interval, size=self.num_clips)  # randomly shift each starting posiiton within one interval
        elif num_frames > max(self.num_clips, ori_clip_len):
            clip_offsets = np.sort(
                np.random.randint(
                    num_frames - ori_clip_len + 1,
                    size=self.num_clips))  # in the interval of (0, num_frames - ori_clip_len + 1), randomly choose 4 starting positions
        elif avg_interval == 0:
            ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
            clip_offsets = np.around(np.arange(self.num_clips) * ratio)
        else:
            clip_offsets = np.zeros((self.num_clips,), dtype=np.int)
        return clip_offsets

    def _get_test_clips(self, num_frames):  # uniformly sample the starting position of each clip
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) / float(self.num_clips)
        if num_frames > ori_clip_len - 1:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = (base_offsets + avg_interval / 2.0).astype(np.int)
        else:
            clip_offsets = np.zeros((self.num_clips,), dtype=np.int)
        return clip_offsets

    def _sample_clips(self, num_frames):
        if self.test_mode:
            clip_offsets = self._get_test_clips(num_frames)
        else:
            clip_offsets = self._get_train_clips(num_frames)
        return clip_offsets

    def uniform_divide_segment(self, record):
        vid_len = record.num_frames
        # n_segments = self.num_clips
        n_segments = self.clip_len
        seg_len = int(np.floor(float(vid_len) / n_segments))
        seg_len_list = [seg_len] * n_segments
        for idx in range(vid_len - seg_len * n_segments):
            seg_len_list[idx] += 1
        return seg_len_list

    def _sample_indices_train(self, record):
        selected_frames_ = np.zeros((self.num_clips, self.clip_len))

        if record.num_frames >= self.clip_len:
            seg_len_list = self.uniform_divide_segment(record)
            # selected_frames = []

            for clip_id in range(self.num_clips):
                start = 0
                for seg_id, seg_len in enumerate(seg_len_list):
                    end = start + seg_len - 1  # (0, 5)  (6, 11),  (12, 17)
                    selected_frames_[clip_id, seg_id] = random.randint(start,
                                                                       end)  # todo random.randint returns a random number, both borders are included
                    # selected_frames_per_clip.append( random.randint( start,  end ) )
                    start = end + 1
        else:  # clip_len > num_frames
            selected_frames = list(range(record.num_frames)) + [record.num_frames - 1] * (
                        self.clip_len - record.num_frames)
            for clip_id in range(self.num_clips):
                selected_frames_[clip_id, :] = selected_frames
        return selected_frames_

    def _sample_indices_test(self, record):
        # sample one clip for test
        selected_frames_ = np.zeros((1, self.clip_len))
        if record.num_frames >= self.clip_len:
            seg_len = int(np.floor(float(record.num_frames) / self.clip_len))
            half_seg_len = int(np.floor(seg_len / 2.0))
            selected_frames = np.arange(self.clip_len) * seg_len + half_seg_len
        else:  # clip_len > num_frames
            # repeat the last frame
            selected_frames = list(range(record.num_frames)) + [record.num_frames - 1] * (
                        self.clip_len - record.num_frames)
            selected_frames = np.array(selected_frames)
        selected_frames_[0, :] = selected_frames
        return selected_frames_

    def _sample_indices(self, record):
        frame_inds = self._sample_indices_test(record) if self.test_mode else self._sample_indices_train(record)
        return frame_inds

    def __getitem__(self, index):
        record = self.video_list[index]  # video file
        indices = self._sample_indices(record)  # get the frame indices
        return self.get(record, indices)

    def get(self, record, indices):  # indices (n_clips, clip_len )
        vid_path = os.path.join(self.video_data_dir, f'{record.path}{self.vid_format}')
        container = decord.VideoReader(vid_path)  # read video into a container
        frame_indices = np.concatenate(indices)  # flatten the frame indices
        # accurate mode

        frame_indices = np.minimum(frame_indices, container._num_frame - 1)
        images = container.get_batch(frame_indices).asnumpy()
        # try:
        #     images = container.get_batch(frame_indices).asnumpy()
        # except:
        #     print(frame_indices)

        images = list(images)
        images = [Image.fromarray(image).convert('RGB') for image in images]

        # process_data = self.transform(images)
        process_data, label = self.transform( (images, record.label) )
        return process_data, label

    def __len__(self):
        return len(self.video_list)
