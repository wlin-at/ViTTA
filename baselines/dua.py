import torch.nn as nn
from utils.utils_ import *
# from corpus.main_train import validate_brief
from corpus.basics import validate_brief
from baselines.dua_utils import rotate_batch


def DUA(model):
    model = configure_model(model)
    return model


def configure_model(model):
    """Configure model for adaptation by test-time normalization."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.train()
    return model


def dua_adaptation(args, model, te_loader, adapt_loader, logger, batchsize, augmentations, no_vids):
    """
    :param model: After configuring the DUA model
    :param te_loader: The test set for Test-Time-Training
    :param logger: Logger for logging results
    :param batchsize: Batchsize to use for adaptation
    :param augmentations: augmentations to form a batch from a single video
    :param no_vids: total number of videos for adaptation

    """
    if args.arch == 'tanet':
        from models.tanet_models.transforms import ToTorchFormatTensor_TANet_dua, GroupNormalize_TANet_dua
        adapt_transforms = torchvision.transforms.Compose([
            augmentations,  # GroupMultiScaleCrop  amd   GroupRandomHorizontalFlip
            ToTorchFormatTensor_TANet_dua(div=True),
            GroupNormalize_TANet_dua(args.input_mean, args.input_std)
        ])
    else:
        adapt_transforms = torchvision.transforms.Compose([
            augmentations,  # GroupMultiScaleCrop  amd   GroupRandomHorizontalFlip
            fromListToTorchFormatTensor(clip_len=args.clip_length, num_clips=args.num_clips),
            GroupNormalize(args.input_mean, args.input_std)
            # Normalize later in the DUA adaptation loop after making a batch
        ])
    logger.debug('---- Starting adaptation for DUA ----')
    all_acc = []
    for i, (inputs, target) in enumerate(adapt_loader):
        model.train()
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.train()

        with torch.no_grad():
            if args.arch == 'tanet':
                n_clips = int(args.sample_style.split("-")[-1])
                inputs = inputs.cuda()
                actual_bz = inputs.shape[0]

                inputs = inputs.view(-1, 3, inputs.size(2), inputs.size(3))
                inputs = inputs.view(actual_bz * args.test_crops * n_clips,
                                     args.clip_length, 3, inputs.size(2), inputs.size(3))  # [1, 16, 3, 224, 224]
                inputs = [(adapt_transforms([inputs, target])[0]) for _ in
                          range(batchsize)]  # pass image, label together
                inputs = torch.stack(inputs)  # only stack images
                inputs = inputs.cuda()
                rot_img = rotate_batch(inputs)
                _ = model(rot_img.float())
            else:
                inputs = [(adapt_transforms([inputs, target])[0]) for _ in
                          range(batchsize)]  # pass image, label together
                inputs = torch.stack(inputs)  # only stack images
                inputs = inputs.cuda()
                inputs = inputs.reshape(
                    (-1,) + inputs.shape[2:])  # [b, channel, frames, h, w]
                rot_img = rotate_batch(inputs)
                _ = model(rot_img)

            logger.debug(f'---- Starting evaluation for DUA after video {i} ----')

        if i % 1 == 0 or i == len(adapt_loader) - 1:
            top1 = validate_brief(eval_loader=te_loader, model=model, global_iter=i, args=args,
                                  logger=logger, writer=None, epoch=i)
            all_acc.append(top1)

        if len(all_acc) >= 3:
            if all(top1 < i for i in all_acc[-3:]):
                logger.debug('---- Model Performance Degrading Consistently ::: Quitting Now ----')
                return max(all_acc)

        if i == no_vids:
            logger.debug(f' --- Best Accuracy for {args.corruptions} --- {max(all_acc)}')
            logger.debug(f' --- Stopping DUA adaptation ---')
            return max(all_acc)

    return max(all_acc)


