import sys

sys.path.append(".")  # last level
import os
import getpass
import logging
import numpy as np
import torchvision
# import torchvision.transforms


from utils.transforms import *
# from transforms import *
import os.path as osp
import csv
import shutil
import time

def make_dir(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)


def get_class_dict(file_):
    class_to_id_dict = dict()
    id_to_class_dict = dict()
    for line in open(file_):
        items = line.strip('\n').split(' ')
        class_id, class_name = int(items[0]), items[1]
        class_to_id_dict.update({class_name: class_id})
        id_to_class_dict.update({class_id: class_name})
    return class_to_id_dict, id_to_class_dict


def read_mapping(file_):
    class_to_id_dict = dict()
    id_to_class_dict = dict()
    for line_id, line in enumerate(open(file_)):
        class_id = line_id
        class_name = line.strip('\n')
        # items = line.strip('\n').split(' ')
        # class_id, class_name = int(items[0]), items[1]
        class_to_id_dict.update({class_name: class_id})
        id_to_class_dict.update({class_id: class_name})
    return class_to_id_dict, id_to_class_dict


def read_csv(csv_file, class_to_id_dict):
    vid_label_dict = dict()
    with open(csv_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        fields = next(csvreader)
        # rows = []
        for row in csvreader:
            action = row[0]
            youtube_id = row[1]
            class_id = class_to_id_dict[action]
            vid_label_dict.update({youtube_id: (class_id, action)})
    return vid_label_dict


def get_env_id():
    if getpass.getuser() == 'mirza':
        env_id = 0
    elif getpass.getuser() == 'jmie01':
        env_id = 1
    elif getpass.getuser() == 'lin':
        env_id = 2
    elif getpass.getuser() == 'ivanl':
        env_id = 3
    elif getpass.getuser() == 'eicg':
        env_id = 4
    elif getpass.getuser() == 'wlin':
        env_id = 5
    else:
        raise Exception("Unknown username!")
    return env_id


def get_list_files_da(dataset_da, debug, data_dir):
    dummy_str = '_dummy' if debug else ''
    if dataset_da == 'u2h':
        train_list = osp.join('UCF-HMDB/UCF-HMDB12/list_nframes_label', f'list_ucf12_train_nframes{dummy_str}.txt')
        val_list = osp.join('UCF-HMDB/UCF-HMDB12/list_nframes_label', f'list_hmdb12_val_nframes{dummy_str}.txt')
    elif dataset_da == 'h2u':
        train_list = osp.join('UCF-HMDB/UCF-HMDB12/list_nframes_label', f'list_hmdb12_train_nframes{dummy_str}.txt')
        val_list = osp.join('UCF-HMDB/UCF-HMDB12/list_nframes_label', f'list_ucf12_val_nframes{dummy_str}.txt')
    train_list, val_list = osp.join(data_dir, train_list), osp.join(data_dir, val_list)
    return train_list, val_list


def path_logger(result_dir, log_time):
    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(logging.DEBUG)
    global logger

    logger = logging.getLogger('basic')
    logger.setLevel(logging.DEBUG)

    path_logging = os.path.join(result_dir, f'{log_time}')

    fileHandler = logging.FileHandler(path_logging, mode='w')
    fileHandler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelno)s - %(filename)s - %(funcName)s - %(message)s')
    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    return logger


def model_analysis(model, logger):
    print("Model Structure")
    print(model)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logger.debug('#################################################')
    logger.debug(f'Number of trainable parameters: {params}')
    logger.debug('#################################################')


def get_augmentation(args, modality, input_size):
    if modality == 'Flow':
        raise NotImplementedError('Flow not implemented!')
        # return torchvision.transforms.Compose([GroupMultiScaleCrop(input_size, [1, .875, .75]),
        #                                        GroupRandomHorizontalFlip(is_flow=True)])


    #   todo  for SSv2,   labels for some classes are modified  when  GroupRandomHorizontalFlip is performed
    #       label_transforms  is hard coded to swap the labels for 3 groups of classes: 86 and 87, 93 and 94, 166 and 167
    #       because after horizontal flip,  "left to right" becomes "right to left"
    if args.dataset == 'somethingv2':
        label_transforms = {
            86: 87,
            87: 86,
            93: 94,
            94: 93,
            166: 167,
            167: 166
        }
    else:
        label_transforms = None

    if args.evaluate_baselines:
        if modality == 'RGB' and args.baseline != 'dua':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False, label_transforms=label_transforms)])
        elif modality == 'RGB' and args.baseline == 'dua':

            from models.tanet_models.transforms import GroupMultiScaleCrop_TANet_tensor, GroupRandomHorizontalFlip_TANet

            if args.arch == 'tanet':
                return torchvision.transforms.Compose([
                    GroupMultiScaleCrop_TANet_tensor(input_size, [1, .875, .75, .66]),
                    GroupRandomHorizontalFlip_TANet(is_flow=False, label_transforms=label_transforms)])
            else:
                return torchvision.transforms.Compose([
                    GroupMultiScaleCrop_tensors(input_size, [1, .875, .75, .66]),
                    GroupRandomHorizontalFlip(is_flow=False, label_transforms=label_transforms)])
        # elif modality == 'RGB' and args.baseline == 'dua' and args.arch == 'tanet':
        #     return torchvision.transforms.Compose([])
    else:
        # todo pure evaluation  or  TTA
        if modality == 'RGB':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False, label_transforms=label_transforms)])


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageMeterTensor(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = torch.tensor(0).float().cuda()
        self.avg = torch.tensor(0).float().cuda()
        self.sum = torch.tensor(0).float().cuda()
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum = self.sum.detach() + val * n
        self.count += n
        self.avg = self.sum / self.count

class MovingAverageTensor(object):
    def __init__(self, momentum=0.1):
        self.momentum = momentum
        self.reset()
    def reset(self):
        self.avg = torch.tensor(0).float().cuda()
    def update(self, val ):
        self.avg = self.momentum * val  + (1.0 - self.momentum) * self.avg.detach().to(val.device)


def adjust_learning_rate(optimizer, epoch, lr_steps, args=None):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['weight_decay'] = decay


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', result_dir=None, log_time=None, logger=None,
                    args=None):
    filename = '_'.join((log_time, args.snapshot_pref, args.modality.lower(), filename))
    file_path = osp.join(result_dir, filename)
    torch.save(state, file_path)
    if is_best:
        best_name = '_'.join((log_time, args.snapshot_pref, args.modality.lower(), 'model_best.pth.tar'))
        best_model_path = osp.join(result_dir, best_name)
        shutil.copyfile(file_path, best_model_path)
        logger.debug(f'Best Checkpoint saved!')


def get_writer_to_all_result(args, custom_path=None):
    log_time = time.strftime("%Y%m%d_%H%M%S")

    if custom_path is None:
        f_write = open(osp.join(args.result_dir, f'{log_time}_all_result'), 'w+')
    else:
        f_write = open(osp.join(custom_path, f'{args.baseline}_{log_time}_all_result'), 'w+')

    for arg in dir(args):
        if arg[0] != '_':
            f_write.write(f'{arg} {getattr(args, arg)}\n')
    f_write.write(f'#############################\n')
    f_write.write(f'#############################\n')
    f_write.write('\n')
    f_write.write('\n')
    return f_write
