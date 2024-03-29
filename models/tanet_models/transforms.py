import os
import sys
sys.path.append(os.getcwd())
import torchvision
import random
from PIL import Image, ImageOps
import numpy as np
import numbers
import math
import torch
import torchvision.transforms as T



# todo   in all the transformations implemented for TANet,   both video data and labels are passed to the transformation function (this is different from normal transformation function where only data are passed)
#    because for some actions (e.g. "xxx from left to right") on SSv2,  transformations like  HorzitonalFlip might change the label of the action


class GroupRandomCrop_TANet(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgs):
        img_group, label = imgs
        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert (img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return (out_images, label)


class GroupCenterCrop_TANet(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, imgs):
        img_group, label = imgs
        return ([self.worker(img) for img in img_group], label)



class SubgroupWise_RandomHorizontalFlip_TANet(object):
    """
    Each gropu consists of several temporal clips,
    split the gropu into subgroup, each goup consists of a temporal clip
    perform random flip for each temporal clip
    Randomly horizontally flips the given Image with a probability of 0.5
    todo  if label is in  label_transforms, do not flip
    """
    def __init__(self, is_flow=False, label_transforms=None, n_temp_clips = None, clip_len = None):
        self.is_flow = is_flow
        self.label_transforms = label_transforms
        self.n_temp_clips = n_temp_clips
        self.clip_len = clip_len
    def __call__(self, imgs):
        img_group, label = imgs

        do_random_flip = True
        if self.label_transforms is not None:
            if label in self.label_transforms.keys():
                do_random_flip = False
        if do_random_flip:
            assert len(img_group) == self.n_temp_clips * self.clip_len
            ret_img_group = []
            for temp_clip_id in range(self.n_temp_clips):
                subgroup = img_group[temp_clip_id * self.clip_len: (temp_clip_id + 1) * self.clip_len]
                v = random.random()
                if v < 0.5:
                    subgroup = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in subgroup]
                ret_img_group += subgroup
            return (ret_img_group, label)
        else:
            return (img_group, label)

            # if v < 0.5:
            #     ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            #     return (ret, label)
            # else:
            #     return (img_group, label)


class GroupRandomHorizontalFlip_TANet(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __init__(self, is_flow=False, label_transforms=None):
        self.is_flow = is_flow
        self.label_transforms = label_transforms

    def __call__(self, imgs):
        img_group, label = imgs
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    ret[i] = ImageOps.invert(
                        ret[i])  # invert flow pixel values when flipping
            if self.label_transforms is not None:
                if label in self.label_transforms.keys():
                    label = self.label_transforms[label]
            return (ret, label)
        else:
            return (img_group, label)

'''
class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    ret[i] = ImageOps.invert(
                        ret[i])  # invert flow pixel values when flipping
            return ret
        else:
            return img_group
'''


class GroupNormalize_TANet(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, imgs):
        inputs, label = imgs
        rep_mean = self.mean * (inputs.size()[0] // len(self.mean))
        rep_std = self.std * (inputs.size()[0] // len(self.std))
        # TODO: make efficient
        for t, m, s in zip(inputs, rep_mean, rep_std):
            t.sub_(m).div_(s)
        return (inputs, label)


class GroupNormalize_TANet_dua(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, imgs):
        inputs, label = imgs
        rep_mean = self.mean * (inputs.size()[0] // len(self.mean))
        rep_std = self.std * (inputs.size()[0] // len(self.std))
        # TODO: make efficient
        for t, m, s in zip(inputs, rep_mean, rep_std):
            t.float().sub_(m).div_(s)
        return (inputs, label)


class GroupScale_TANet(object):  # todo aspect ratio is kept
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, imgs):
        img_group, label = imgs
        return ([self.worker(img) for img in img_group], label)


class GroupOverSample_TANet(object):
    def __init__(self, crop_size, scale_size=None, flip=True):
        self.crop_size = crop_size if not isinstance(crop_size, int) else (
            crop_size, crop_size)

        if scale_size is not None:
            self.scale_worker = GroupScale_TANet(scale_size)
        else:
            self.scale_worker = None
        self.flip = flip

    def __call__(self, imgs):
        img_group, label = imgs
        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        offsets = GroupMultiScaleCrop_TANet.fill_fix_offset(False, image_w, image_h,
                                                      crop_w, crop_h)
        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)

                if img.mode == 'L' and i % 2 == 0:
                    flip_group.append(ImageOps.invert(flip_crop))
                else:
                    flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            if self.flip:
                oversample_group.extend(flip_group)
        return (oversample_group, label)


class GroupFullResSample_TANet(object):
    def __init__(self, crop_size, scale_size=None, flip=True):
        self.crop_size = crop_size if not isinstance(crop_size, int) else (
            crop_size, crop_size)

        if scale_size is not None:
            self.scale_worker = GroupScale_TANet(scale_size)  #  scale to scale size
        else:
            self.scale_worker = None
        self.flip = flip

    def __call__(self, imgs):
        img_group, label = imgs

        if self.scale_worker is not None:
            img_group, label = self.scale_worker(imgs)   #  scale to scale size

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        offsets = list()
        offsets.append((0 * w_step, 2 * h_step))  # left
        offsets.append((4 * w_step, 2 * h_step))  # right
        offsets.append((2 * w_step, 2 * h_step))  # center

        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                if self.flip:
                    flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)

                    if img.mode == 'L' and i % 2 == 0:
                        flip_group.append(ImageOps.invert(flip_crop))
                    else:
                        flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            oversample_group.extend(flip_group)
        return (oversample_group, label)


# def random_crop_img_group( sub_img_group,  origin_img_size,   )

class SubgroupWise_MultiScaleCrop_TANet(object):
    """
    Each group consists of several temporal clips,
    split the group into subgroup, each group consists of a temporal clip
    perform a random scale crop for each temporal clip
    """
    def __init__(self,
                 input_size,
                 scales=None,
                 max_distort=1,
                 fix_crop=True,
                 more_fix_crop=True,
                 n_temp_clips = None,
                 clip_len = None):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [
            input_size, input_size
        ]
        self.interpolation = Image.BILINEAR
        self.n_temp_clips = n_temp_clips
        self.clip_len = clip_len
    def __call__(self, imgs):
        img_group, label = imgs
        assert len(img_group) == self.n_temp_clips * self.clip_len
        im_size = img_group[0].size
        ret_img_group = []
        for temp_clip_id in range(self.n_temp_clips):
            subgroup = img_group[ temp_clip_id*self.clip_len : (temp_clip_id+1)*self.clip_len   ]
            ret_img_group +=  self.crop_scale_subgroup(origin_im_size=im_size, img_group= subgroup )

        return (ret_img_group, label)

    def crop_scale_subgroup(self, origin_im_size, img_group):
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(origin_im_size)
        crop_img_group = [
            img.crop(
                (offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
            for img in img_group
        ]
        ret_img_group = [
            img.resize((self.input_size[0], self.input_size[1]),
                       self.interpolation) for img in crop_img_group
        ]
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [
            self.input_size[1] if abs(x - self.input_size[1]) < 3 else x
            for x in crop_sizes
        ]
        crop_w = [
            self.input_size[0] if abs(x - self.input_size[0]) < 3 else x
            for x in crop_sizes
        ]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(
                image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h,
                                       crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret





class GroupMultiScaleCrop_TANet(object):
    def __init__(self,
                 input_size,
                 scales=None,
                 max_distort=1,
                 fix_crop=True,
                 more_fix_crop=True):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [
            input_size, input_size
        ]
        self.interpolation = Image.BILINEAR

    def __call__(self, imgs):
        img_group, label = imgs
        im_size = img_group[0].size

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [
            img.crop(
                (offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
            for img in img_group
        ]
        ret_img_group = [
            img.resize((self.input_size[0], self.input_size[1]),
                       self.interpolation) for img in crop_img_group
        ]
        return (ret_img_group, label)

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [
            self.input_size[1] if abs(x - self.input_size[1]) < 3 else x
            for x in crop_sizes
        ]
        crop_w = [
            self.input_size[0] if abs(x - self.input_size[0]) < 3 else x
            for x in crop_sizes
        ]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(
                image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h,
                                       crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret


class GroupMultiScaleCrop_TANet_tensor(object):
    def __init__(self,
                 input_size,
                 scales=None,
                 max_distort=1,
                 fix_crop=True,
                 more_fix_crop=True):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [
            input_size, input_size
        ]
        self.interpolation = Image.BILINEAR

    def __call__(self, imgs):

        img_group, label = imgs
        img_group = torch.squeeze(img_group)
        transform = T.ToPILImage()
        img_group = [img_group[x, :, :, :] for x in range(img_group.shape[0])] # form list

        im_size = img_group[0].shape
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [
            transform(img).crop(
                (offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)) # crop and transpose it back to PIL
            for img in img_group
        ]
        ret_img_group = [
            img.resize((self.input_size[0], self.input_size[1]),
                       self.interpolation) for img in crop_img_group
        ]
        return (ret_img_group, label)

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[2], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [
            self.input_size[1] if abs(x - self.input_size[1]) < 3 else x
            for x in crop_sizes
        ]
        crop_w = [
            self.input_size[0] if abs(x - self.input_size[0]) < 3 else x
            for x in crop_sizes
        ]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(
                image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h,
                                       crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret

class GroupRandomSizedCrop_TANet(object):
    """Random crop the given PIL.Image to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgs):
        img_group, label = imgs
        for attempt in range(10):
            area = img_group[0].size[0] * img_group[0].size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img_group[0].size[0] and h <= img_group[0].size[1]:
                x1 = random.randint(0, img_group[0].size[0] - w)
                y1 = random.randint(0, img_group[0].size[1] - h)
                found = True
                break
        else:
            found = False
            x1 = 0
            y1 = 0

        if found:
            out_group = list()
            for img in img_group:
                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))
                out_group.append(
                    img.resize((self.size, self.size), self.interpolation))
            return out_group
        else:
            # Fallback
            scale = GroupScale_TANet(self.size, interpolation=self.interpolation)
            crop = GroupRandomCrop_TANet(self.size)
            return crop(scale((img_group, label)))

# class Stack_aug_views_TANet(object):
#     # todo  stack the multiple augmented views into the channel dimension (just like TANet stacks multiple temporal clips into the channel dimension)
#     #   here we assume that the  number of temporal clips is 1,  but there are multiple augmented views
#     #   a list with the length of temporal_clips* clip_len  of Image objects      -> numpy array  (H, W, temporal_clips* clip_len * 3 )
#     def __int__(self, roll = False):
#         self.roll = roll


class Stack_TANet(object):
    def __init__(self, roll=False):
        # todo #  stack the multiple temporal clips into channel dimension,  a list with the length of temporal_clips* clip_len  of Image objects      -> numpy array  (H, W, temporal_clips* clip_len * 3 )
        self.roll = roll

    def __call__(self, imgs):
        img_group, label = imgs
        if img_group[0].mode == 'L':
            img_group = np.concatenate(
                [np.expand_dims(x, 2) for x in img_group], axis=2)
            return (img_group, label)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                img_group = np.concatenate(
                    [np.array(x)[:, :, ::-1] for x in img_group], axis=2)
                return (img_group, label)
            else:
                return (np.concatenate(img_group, axis=2), label)


class ToTorchFormatTensor_TANet(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, imgs):
        pic, label = imgs
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()  # todo  numpy array (H, W, temporal_clips* clip_len * 3 ) ->  torch tensor  (temporal_clips* clip_len * 3 , H, W)
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(
                pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if self.div:
            img = img.float().div(255) if self.div else img.float()  # todo   torch tensor  (temporal_clips* clip_len * 3 , H, W), divided by 255
        return (img, label)


class ToTorchFormatTensor_TANet_dua(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, imgs):
        pic, label = imgs
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            all_imgs = []
            for img in pic:
                img_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(
                    img.tobytes()))
                img_tensor = img_tensor.view(img.size[1], img.size[0], len(img.mode))
                # put it from HWC to CHW format
                # yikes, this transpose takes 80% of the loading time/CPU
                img_tensor = img_tensor.transpose(0, 1).transpose(0, 2).contiguous()
                all_imgs.append(img_tensor)
            img_final = torch.stack(all_imgs)

        if self.div:
            img_final = img_final.float().div(255) if self.div else img_final.float()
        return (img_final, label)


class IdentityTransform_TANet(object):
    def __call__(self, data):
        return data


if __name__ == "__main__":
    trans = torchvision.transforms.Compose([
        GroupScale(256),
        GroupRandomCrop(224),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(mean=[.485, .456, .406], std=[.229, .224, .225])
    ])

    im = Image.open('../tensorflow-model-zoo.torch/lena_299.png')

    color_group = [im] * 3
    rst = trans(color_group)

    gray_group = [im.convert('L')] * 9
    gray_rst = trans(gray_group)

    trans2 = torchvision.transforms.Compose([
        GroupRandomSizedCrop(256),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(mean=[.485, .456, .406], std=[.229, .224, .225])
    ])
    print(trans2(color_group))