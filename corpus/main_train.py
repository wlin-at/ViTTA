import time
import os
# from torch.nn.utils import clip_grad_norm
# import torch.nn as nn
# from einops import rearrange
import os.path as osp
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
# from datasets_.dataset import MyTSNDataset
# from datasets_.video_dataset import MyTSNVideoDataset, MyVideoDataset

# from models.r2plus1d import MyR2plus1d
# from models import i3d
# from models.i3d_incep import InceptionI3d
# from models.tanet_models.tanet import TSN
from utils.transforms import *
from utils.utils_ import  make_dir, path_logger, model_analysis, \
    adjust_learning_rate, save_checkpoint
# from utils.BNS_utils import BN3DFeatureHook, choose_BN_layers
# import baselines.tent as tent
from corpus.basics import train, validate,  get_dataset, get_model

def main_train(args=None, best_prec1=0, ):
    log_time = time.strftime("%Y%m%d_%H%M%S")

    make_dir(args.result_dir)
    logger = path_logger(args.result_dir, log_time)
    writer = SummaryWriter(log_dir=osp.join(args.result_dir, f'{log_time}_tb'))

    for arg in dir(args):
        logger.debug(f'{arg} {getattr(args, arg)}')

    if args.dataset == 'ucf101':
        num_classes = 101
    elif args.dataset == 'hmdb51':
        num_classes = 51
    elif args.dataset == 'kinetics':
        num_classes = 400
    elif args.dataset == 'kth':
        num_classes = 6
    elif args.dataset in ['u2h', 'h2u']:
        num_classes = 12
    else:
        raise ValueError('Unknown dataset ' + args.dataset)

    model = get_model(args, num_classes, logger)

    if args.verbose:
        model_analysis(model, logger)
    args.crop_size = args.input_size

    if args.modality == 'Flow':
        args.input_mean = [0.5]
        args.input_std = [np.mean(args.input_std)]

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
    # for k, v in model.named_parameters():
    #     print(k)
    # quit()

    if args.resume:
        if os.path.isfile(args.resume):
            logger.debug("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logger.debug("=> loaded checkpoint '{}' (epoch {})"
                         .format(args.evaluate, checkpoint['epoch']))
        else:
            logger.debug("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    train_loader = torch.utils.data.DataLoader(  # TSN video dataset
        get_dataset(args, split='train'),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, )

    val_loader = torch.utils.data.DataLoader(
        get_dataset(args, split='val'),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.evaluate:
        validate(val_loader, model, criterion, 0, epoch=0, args=args, logger=logger, writer=writer)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps, args=args)  # learning rate decay is 0.1

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args=args, logger=logger, writer=writer)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1 = validate(val_loader, model, criterion, (epoch + 1) * len(train_loader), epoch=epoch, args=args,
                             logger=logger, writer=writer)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            if args.if_save_model:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                }, is_best, result_dir=args.result_dir, log_time=log_time, logger=logger, args=args)
                logger.debug(f'Checkpoint epoch {epoch} saved!')



