import os
import time
import torch.utils.data.dataloader
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

from utils.transforms import *
from utils.utils_ import make_dir, path_logger, model_analysis
# from models.r2plus1d import MyR2plus1d
# from models import i3d

from corpus.basics import validate, get_dataset, get_dataset_tanet, get_dataset_videoswin, \
    get_dataset_tanet_dua, get_model, test_time_adapt, compute_statistics, compute_cos_similarity, \
    tta_standard
import os.path as osp
from tensorboardX import SummaryWriter
from baselines.setup_baseline import setup_model
from baselines.shot import train as train_shot
from baselines.dua import dua_adaptation as adapt_dua
from baselines.t3a import get_cls_ext, t3a_forward_and_adapt
# import torch.nn as nn


# def compute_temp_statistics(args = None,):




def eval(args=None, model = None ):
    log_time = time.strftime("%Y%m%d_%H%M%S")
    make_dir(args.result_dir)
    logger = path_logger(args.result_dir, log_time)
    # writer = SummaryWriter(log_dir=osp.join(result_dir, f'{log_time}_tb'))
    if args.verbose:
        for arg in dir(args):
            if arg[0] != '_':
                logger.debug(f'{arg} {getattr(args, arg)}')
    num_class_dict = {
        'ucf101' : 101,
        'hmdb51': 51,
        'kinetics': 400,
        'somethingv2': 174,
        'kth': 6,
        'u2h':12,
        'h2u':12,
    }
    num_classes = num_class_dict[args.dataset]
    args.num_classes = num_classes

    if model is None:
        # todo  initialize the model if the model is not provided
        model = get_model(args, num_classes, logger)
        # todo  load model weights
        checkpoint = torch.load(args.model_path)
        logger.debug(f'Loading {args.model_path}')
        if args.arch == 'tanet':
            print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

        if 'module.' in list(checkpoint['state_dict'].keys())[0]:
            model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
            model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
    if args.verbose:
        model_analysis(model, logger)

    args.crop_size = args.input_size

    if args.modality == 'Flow':
        args.input_mean = [0.5]
        args.input_std = [np.mean(args.input_std)]

    # train_augmentation = get_augmentation(args.modality, args.input_size)  #  GroupMultiScaleCrop  amd   GroupRandomHorizontalFlip

    cudnn.benchmark = True

    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")
    if args.tta:
        # TTA
        writer = SummaryWriter(log_dir=osp.join(args.result_dir, f'{log_time}_tb'))
        if args.compute_stat == 'mean_var':
            compute_statistics(model, args=args, log_time=log_time)
            epoch_result_list = None
        elif args.compute_stat == 'cossim':
            compute_cos_similarity(model, args=args, log_time=log_time)
            epoch_result_list = None
        elif args.compute_stat == False:
            if args.if_tta_standard:
                epoch_result_list = tta_standard(model, criterion, args=args, logger=logger, writer=writer)
                model = None
            else:
                # todo  return the adapted model
                epoch_result_list, model = test_time_adapt(model, criterion, args=args, logger=logger, writer=writer)

    elif args.evaluate_baselines:
        # evaluate baselines
        if args.baseline == 'source': #  source only evaluation
            if args.arch == 'tanet':
                val_loader = torch.utils.data.DataLoader(
                    get_dataset_tanet(args,  split='val', dataset_type='eval'),
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True, )
            elif args.arch == 'videoswintransformer':
                val_loader = torch.utils.data.DataLoader(
                    get_dataset_videoswin(args, split='val', dataset_type='eval'),
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True, )
            else:
                # I3D
                val_loader = torch.utils.data.DataLoader(
                    get_dataset(args, split='val'),
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

            model_baseline = setup_model(args, base_model=model, logger=logger)  #  set model to eval mode


            top1_acc = validate(val_loader, model_baseline, criterion, 0, epoch=0, args=args, logger=logger)
            epoch_result_list = [top1_acc]

        elif args.baseline == 'norm':

            if args.arch == 'tanet':
                val_loader = torch.utils.data.DataLoader(
                    get_dataset_tanet(args, split='val'),
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True, )
            else:
                val_loader = torch.utils.data.DataLoader(
                    get_dataset(args, split='val'),
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

            model_baseline = setup_model(args, base_model=model, logger=logger)
            top1_acc = validate(val_loader, model_baseline, criterion, 0, epoch=0, args=args, logger=logger)
            epoch_result_list = [top1_acc]

        elif args.baseline == 'tent':

            if args.arch == 'tanet':
                val_loader = torch.utils.data.DataLoader(
                    get_dataset_tanet(args,  split='val'),
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True, )
            else:
                val_loader = torch.utils.data.DataLoader(
                    get_dataset(args, split='val'),
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True)
            # set only Batchnorm3d layers to trainable,   freeze all the other layers ;    collecting gamma and beta in all Batchnorm3d layers
            model_baseline, optimizer = setup_model(args, base_model=model, logger=logger)
            top1_acc = validate(val_loader, model_baseline, criterion, 0, epoch=0,
                                args=args, logger=logger, optimizer = optimizer)
            epoch_result_list = [top1_acc]

        elif args.baseline == 'shot':
            if args.arch == 'tanet':
                val_loader = torch.utils.data.DataLoader(
                    get_dataset_tanet(args, split='val'),
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True, )
            else:
                val_loader = torch.utils.data.DataLoader(
                    get_dataset(args, split='val'),
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

            optimizer, classifier, ext = setup_model(args, base_model=model, logger=logger)
            top1_acc = train_shot(args, criterion, optimizer, classifier, ext, val_loader, logger) # train and validate
            epoch_result_list = [top1_acc]

        elif args.baseline == 'dua':
            from utils.utils_ import get_augmentation
            aug = get_augmentation(args, args.modality,
                                   args.input_size)
            if args.arch == 'tanet':
                val_loader_adapt = torch.utils.data.DataLoader(
                    get_dataset_tanet_dua(args, tanet_model=model.module, split='val')[1],
                    batch_size=1, shuffle=False,
                    num_workers=args.workers, pin_memory=True, )
                te_loader = torch.utils.data.DataLoader(
                    get_dataset_tanet_dua(args, tanet_model=model.module, split='val')[0],
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True, )
            else:
                val_loader_adapt = torch.utils.data.DataLoader(
                    get_dataset(args, split='val')[0],
                    batch_size=1, shuffle=True,
                    num_workers=args.workers, pin_memory=True
                )
                te_loader = torch.utils.data.DataLoader(
                    get_dataset(args, split='val')[1],
                    batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=True
                )

            dua_model = setup_model(args, base_model=model, logger=logger)
            top1_acc = adapt_dua(args=args, model=dua_model, batchsize=16, logger=logger,
                      no_vids=int(len(val_loader_adapt) * 1 / 100),
                      adapt_loader=val_loader_adapt, te_loader=te_loader, augmentations=aug)

            epoch_result_list = [top1_acc]

        elif args.baseline == 't3a':
            logger.debug(f'Baseline :::::: {args.baseline}')
            if args.arch == 'tanet':
                val_loader = torch.utils.data.DataLoader(
                    get_dataset_tanet(args,  split='val'),
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True, )
            else:
                val_loader = torch.utils.data.DataLoader(
                    get_dataset(args, split='val'),
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

            ext, classifier = get_cls_ext(args, model)
            top1_acc = t3a_forward_and_adapt(args, ext, classifier, val_loader)
            logger.debug(f'Top1 Accuracy After Adaptation ::: {args.corruptions} ::: {top1_acc}')
            epoch_result_list = [top1_acc]
        else:
            raise NotImplementedError('The Baseline is not Implemented')

    # validate(val_loader, model, criterion, iter,  epoch = None, args = None, logger= None, writer = None)
    logger.handlers.clear()
    # todo return the adapted model,  if no adaptation, returned model is None
    return epoch_result_list, model


