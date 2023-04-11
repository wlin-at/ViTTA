
import torch.utils.data as data

from PIL import Image
import os
import numpy as np
from numpy.random import randint
# from ops.dataset import VideoRecord
import os.path as osp
import decord

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

class Video_TANetDataSet(data.Dataset):
    """  directly load frames from video file, instead of loading extracted image frames """
    def __init__(self,
                 # root_path,
                 list_file,
                 num_segments=3,  # clip_length
                 new_length=1,
                 modality='RGB',
                 # image_tmpl='img_{:05d}.jpg',
                 vid_format = '.mp4',
                 transform=None,
                 random_shift=True,
                 test_mode=False,
                 video_data_dir = None,
                 remove_missing=False,
                 dense_sample=False,
                 test_sample="dense-10",  # only for test data

                 if_sample_tta_aug_views = None,
                 tta_view_sample_style_list = None,
                 n_tta_aug_views = None,
                 debug = False, debug_vid = 50,
                 ):

        # self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments  # clip_length
        self.new_length = new_length
        self.modality = modality
        # self.image_tmpl = image_tmpl
        self.vid_format = vid_format
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.video_data_dir = video_data_dir
        self.remove_missing = remove_missing
        self.dense_sample = dense_sample  # using dense sample as I3D   todo  frames are sampled from 64 consecutive frames in the video
        self.test_sample = test_sample

        self.debug = debug
        self.debug_vid = debug_vid

        self.if_sample_tta_aug_views = if_sample_tta_aug_views
        self.tta_view_sample_style_list = tta_view_sample_style_list
        self.n_tta_aug_views = n_tta_aug_views

        if self.dense_sample:
            print('=> Using dense sample for the dataset...')  #  todo  frames are sampled from 64 consecutive frames in the video

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image_deprecated(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            try:
                return [
                    Image.open(
                        os.path.join(
                            self.root_path, directory,
                            self.image_tmpl.format(idx))).convert('RGB')
                ]
            except Exception:
                print(
                    'error loading image:',
                    os.path.join(self.root_path, directory,
                                 self.image_tmpl.format(idx)))
                return [
                    Image.open(
                        os.path.join(self.root_path, directory,
                                     self.image_tmpl.format(1))).convert('RGB')
                ]
        elif self.modality == 'Flow':
            if self.image_tmpl == 'flow_{}_{:05d}.jpg':  # ucf
                x_img = Image.open(
                    os.path.join(self.root_path, directory,
                                 self.image_tmpl.format('x',
                                                        idx))).convert('L')
                y_img = Image.open(
                    os.path.join(self.root_path, directory,
                                 self.image_tmpl.format('y',
                                                        idx))).convert('L')
            elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':  # something v1 flow
                x_img = Image.open(
                    os.path.join(
                        self.root_path, '{:06d}'.format(int(directory)),
                        self.image_tmpl.format(int(directory), 'x',
                                               idx))).convert('L')
                y_img = Image.open(
                    os.path.join(
                        self.root_path, '{:06d}'.format(int(directory)),
                        self.image_tmpl.format(int(directory), 'y',
                                               idx))).convert('L')
            else:
                try:
                    # idx_skip = 1 + (idx-1)*5
                    flow = Image.open(
                        os.path.join(
                            self.root_path, directory,
                            self.image_tmpl.format(idx))).convert('RGB')
                except Exception:
                    print(
                        'error loading flow file:',
                        os.path.join(self.root_path, directory,
                                     self.image_tmpl.format(idx)))
                    flow = Image.open(
                        os.path.join(self.root_path, directory,
                                     self.image_tmpl.format(1))).convert('RGB')
                # the input flow file is RGB image with (flow_x, flow_y, blank) for each channel
                flow_x, flow_y, _ = flow.split()
                x_img = flow_x.convert('L')
                y_img = flow_y.convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        # check the frame number is large >3:
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        if not self.test_mode or self.remove_missing:
            tmp = [item for item in tmp if int(item[1]) >= 3]
        self.video_list = [VideoRecord(item) for item in tmp]

        if self.debug:
            self.video_list = self.video_list[:self.debug_vid]

        # if self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
        #     for v in self.video_list:
        #         v._data[1] = int(v._data[1]) / 2
        # print('video number:%d' % (len(self.video_list)))

    def _sample_tta_augmented_views(self, record, tta_view_sample_style):
        #  uniform, dense,  uniform_rand,  dense_rand,  random
        if tta_view_sample_style == 'uniform':
            # todo uniformly divde a video into several segments, then sample the middle frame in each segment
            num_clips = 1
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = [int(tick / 2.0 + tick * x) for x in range(self.num_segments)]
            return np.array(offsets) + 1
        elif tta_view_sample_style == 'dense':
            # todo choose 64 conseutive frames in the center of video, from these 64 frames, sample [self.num_segment] frames with fixed stride (64//self.num_segments)
            num_clips = 1
            t_stride = 64 // self.num_segments  # num_segments is the clip_length
            sample_pos = max(1, 1 + record.num_frames - t_stride * self.num_segments)
            start_idx = sample_pos // 2
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(
                self.num_segments)]  # todo  notice that when record.num_frames < 64, the frames come recurrently !!!!!!!!!!!
            return np.array(offsets) + 1


        elif tta_view_sample_style == 'uniform_equidist':
            # todo uniformly divide a video into several segments,  in the first segment, equidistantly choose 2 starting positions for uniform sampling
            num_clips = self.n_tta_aug_views  # todo the frame indices of the 2 clips are concatenated
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            start_list = np.linspace(0, tick - 1, num=num_clips, dtype=int)
            offsets = []   # offsets of the two views are concatenated in a list
            for start_idx in start_list.tolist():
                offsets += [int(start_idx + tick * x) % record.num_frames for x in range(self.num_segments)]
            return np.array(offsets) + 1
        elif tta_view_sample_style == 'dense_equidist':
            # todo equi-distantly choose the starting points of  [num_clips]  64-consecutive-frame-segments, then sample [self.num_segment] frames  from each  64-consecutive-frame-segment
            num_clips = self.n_tta_aug_views  # todo the frame indices of the 2 clips are concatenated
            t_stride = 64 // self.num_segments
            sample_pos = max(1, 1 + record.num_frames - t_stride * self.num_segments)
            start_list = np.linspace(0, sample_pos - 1, num=num_clips, dtype=int)  #
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        elif tta_view_sample_style == 'uniform_rand':
            # todo uniformly divde a video into sevral segments, then randomly sample one frame from each segment
            num_clips = 1
            average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                                  size=self.num_segments)
            elif record.num_frames > self.num_segments:
                # todo if a video cannot be uniformly into several frames, then just randomly sample  frames (with replacement) from the video
                offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
            else:
                # todo duplicate sampling of the first frame
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

        elif tta_view_sample_style == 'dense_rand':
            # todo  randomly choose 64 consecutive frames in the video, from these 64 frames,  then sample [self.num_segment] frames with fixed stride (64//self.num_segments)
            num_clips = 1
            t_stride = 64 // self.num_segments  # number of frames in each segment
            # t_stride = 128 // self.num_segments
            sample_pos = max(1, 1 + record.num_frames - t_stride * self.num_segments)
            start_idx = 0 if sample_pos == 1 else randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1

        elif tta_view_sample_style == 'random':
            # todo  completely randomly sample  [self.num_segments]  frames from the video,  randomly sample without replacement
            num_clips = 1
            if record.num_frames >= self.num_segments:
                offsets = np.sort(np.random.choice(record.num_frames, size=self.num_segments, replace=False))
            else:
                # todo duplicating the last frame
                offsets = np.array( list(range(record.num_frames)) + [record.num_frames-1]* (self.num_segments - record.num_frames))
            return np.array(offsets)
    def _sample_indices(self, record):  # todo get training sample indices,  only sample 1 clip during training
        """
        :param record: VideoRecord
        :return: list
        """
        if self.dense_sample:  # todo  i3d dense sample :  randomly choose 64 consecutive frames in the video, from these 64 frames,  then sample [self.num_segment] frames with fixed stride (64//self.num_segments)
            t_stride = 64 // self.num_segments   #  number of frames in each segment
            # t_stride = 128 // self.num_segments
            sample_pos = max( 1, 1 + record.num_frames - t_stride * self.num_segments)
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:  # normal sample   todo TSN sampling : uniformly divide a video into several segments, then randomly sample one frame from each segment
            average_duration = (record.num_frames - self.new_length +1) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
            elif record.num_frames > self.num_segments:
                offsets = np.sort( randint(record.num_frames - self.new_length + 1, size=self.num_segments))
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_val_indices(self, record):  #  todo get validation sample indices, random shift is False
        if self.dense_sample:  # todo   i3d dense sample : choose 64 consecutive frames in the center of the video, from these 64 frames, sample [self.num_segment] frames with fixed stride (64//self.num_segments)
            # t_stride = 8
            t_stride = 64 // self.num_segments
            # t_stride = 128 // self.num_segments
            sample_pos = max( 1, 1 + record.num_frames - t_stride * self.num_segments)
            # start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            start_idx = sample_pos // 2
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:  # todo TSN sampling :  uniformly divide a video into several segments, then sample the middle frame in each segment
            if record.num_frames > self.num_segments + self.new_length - 1:
                tick = (record.num_frames - self.new_length + 1) / float( self.num_segments)
                offsets = np.array([ int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_test_indices(self, record):  # todo   get test sample indices,
        if "dense" in self.test_sample:
            num_clips = int(self.test_sample.split("-")[-1])
            # t_stride = 8
            t_stride = 64 // self.num_segments  #  num_segments is the clip_length
            # t_stride = 128 // self.num_segments
            sample_pos = max( 1, 1 + record.num_frames - t_stride * self.num_segments)
            if num_clips == 1:   # todo choose 64 consecutive frames in the center of the video, from these 64 frames, sample [self.num_segment] frames with fixed stride (64//self.num_segments)
                start_idx = sample_pos // 2
                offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]  # todo  notice that when record.num_frames < 64, the frames come recurrently !!!!!!!!!!!
            else:  #  todo equi-distantly choose the starting points of  [num_clips]  64-consecutive-frame-segments, then sample [self.num_segment] frames  from each  64-consecutive-frame-segment
                start_list = np.linspace(0, sample_pos - 1, num=num_clips, dtype=int)  #
                offsets = []
                for start_idx in start_list.tolist():
                    offsets += [ (idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        elif "uniform" in self.test_sample:
            num_clips = int(self.test_sample.split("-")[-1])
            if num_clips == 1:  #  todo uniformly divide a video into several segments, then sample the middle frame in each segment
                tick = (record.num_frames - self.new_length + 1) / float( self.num_segments)
                offsets = [ int(tick / 2.0 + tick * x) for x in range(self.num_segments)]
            else:#  todo uniformly divide a video into several segments,  in the first segment, equidistantly choose [num_clips] starting positions for uniform sampling
                tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
                start_list = np.linspace(0, tick - 1, num=num_clips, dtype=int)
                offsets = []
                # print(start_list.tolist())
                # print(tick)
                for start_idx in start_list.tolist():
                    offsets += [ int(start_idx + tick * x) % record.num_frames for x in range(self.num_segments)]

            return np.array(offsets) + 1
        else:
            raise NotImplementedError("{} not exist".format(self.test_sample))

    def __getitem__(self, index):
        record = self.video_list[index]
        if self.if_sample_tta_aug_views:
            segment_indices = []
            for tta_view_sample_style in self.tta_view_sample_style_list:
                # segment_indices =  self._sample_tta_augmented_views( record, tta_view_sample_style)
                segment_indices += list( self._sample_tta_augmented_views( record, tta_view_sample_style) )
        else:
            if not self.test_mode:
                segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
            else:
                segment_indices = self._get_test_indices(record)  #  the frame indices for multiple temporal clips are concatenated

        return self.get(record, segment_indices)

    def get(self, record, indices):
        vid_path = osp.join( self.video_data_dir,  f'{record.path}{self.vid_format}' )
        container = decord.VideoReader(vid_path)
        # frame_indices = np.concatenate(indices) #  flatten the frame indices
        # todo during training, there is only one clip
        #    during test, there could be multiple clips whose frame indices are concatenated in   indices
        frame_indices = indices

        frame_indices = np.minimum(frame_indices, container._num_frame - 1 )
        # try:
        images = container.get_batch(frame_indices).asnumpy()
        # except:
        #     t = 1
        images = list(images)
        images = [Image.fromarray(image).convert('RGB') for image in images]
        # todo  train augmentation #  GroupMultiScaleCrop and  GroupRandomHorizontalFlip ,
        #    for SSv2,   labels for some classes are modified in  GroupRandomHorizontalFlip in
        #   label_transforms  is hard coded to swap the labels for 3 groups of classes: 86 and 87, 93 and 94, 166 and 167
        #   because after horizontal flip,  "left to right" becomes "right to left"

        process_data, label = self.transform((images, record.label))
        return process_data, label

    def get_img_file_deprecated(self, record, indices):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):  #  for RGB, load 1 image
                seg_imgs = self._load_image(record.path, p)  # load image into PIL image
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data, label = self.transform((images, record.label))
        return process_data, label

    def __len__(self):
        return len(self.video_list)