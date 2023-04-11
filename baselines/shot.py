from baselines.shot_utils import *
import torch.optim as optim
import argparse
from utils import utils_ as utils
from corpus.main_train import validate

parser = argparse.ArgumentParser()

parser.add_argument('--lr', default=0.00005, type=float)
parser.add_argument('--nepoch', default=1, type=int, help='maximum number of epoch for SHOT')
parser.add_argument('--bnepoch', default=2, type=int, help='first few epochs to update bn stat')
parser.add_argument('--delayepoch', default=0, type=int)
parser.add_argument('--stopepoch', default=25, type=int)
########################################################################
parser.add_argument('--outf', default='.')
########################################################################
parser.add_argument('--level', default=5, type=int)
parser.add_argument('--corruption', default='')
parser.add_argument('--resume', default=None, help='directory of pretrained model')
parser.add_argument('--ckpt', default=None, type=int)
parser.add_argument('--fix_ssh', action='store_true')
parser.add_argument('--batch_size', default=12, type=int)
########################################################################
parser.add_argument('--method', default='shot', choices=['shot'])
########################################################################
parser.add_argument('--model', default='resnet50', help='resnet50')
parser.add_argument('--save_every', default=100, type=int)
########################################################################
parser.add_argument('--tsne', action='store_true')
########################################################################
parser.add_argument('--cls_par', type=float, default=0.001)
parser.add_argument('--ent_par', type=float, default=1.0)
parser.add_argument('--gent', type=bool, default=True)
parser.add_argument('--ent', type=bool, default=True)
########################################################################
parser.add_argument('--seed', default=0, type=int)

args_shot = parser.parse_args()


def configure_shot(net, logger, args):
    logger.debug('---- Configuring SHOT ----')
    if args.arch == 'tanet':
        classifier = net.module.new_fc
        ext = net
        ext.module.new_fc = nn.Identity()

        for k, v in classifier.named_parameters():
            v.requires_grad = False
    else:
        for k, v in net.named_parameters():
            if 'logits' in k:
                v.requires_grad = False  # freeze the  classifier
        classifier = nn.Sequential(*list(net.module.logits.children()))
        ext = list(net.module.children())[3:] + list(net.module.children())[:2]
        ext = nn.Sequential(*ext)

    optimizer = optim.SGD(ext.parameters(), lr=args_shot.lr, momentum=0.9)
    return optimizer, classifier, ext


def train(args, criterion, optimizer, classifier, ext, teloader, logger):
    logger.debug('---- Training SHOT ----')
    losses = utils.AverageMeter()
    shot_acc = list()
    if args.arch == 'tanet':
        n_clips = int(args.sample_style.split("-")[-1])

    for epoch in range(1, args_shot.nepoch + 1):
        ext.eval()
        mem_label = obtain_shot_label(teloader, ext, classifier, n_clips=n_clips, args=args)  # compute the pseudo label
        mem_label = torch.from_numpy(mem_label).cuda()
        ext.train()

        for batch_idx, (inputs, labels) in enumerate(teloader):

            optimizer.zero_grad()
            actual_bz = inputs.shape[0]
            inputs = inputs.cuda()
            labels = labels.cuda()

            if args.arch == 'tanet':
                classifier_loss = 0
                inputs = inputs.view(-1, 3, inputs.size(2), inputs.size(3))
                inputs = inputs.view(actual_bz * args.test_crops * n_clips,
                                     args.clip_length, 3, inputs.size(2), inputs.size(3))
                features_test = ext(inputs.cuda())

                outputs_test = classifier(features_test)

                outputs_test = torch.squeeze(outputs_test)
                outputs_test = outputs_test.reshape(actual_bz, args.test_crops * n_clips, -1).mean(1)

            else:
                classifier_loss = 0
                inputs = inputs.reshape(
                    (-1,) + inputs.shape[2:])
                features_test = ext(inputs.cuda())
                outputs_test = classifier(features_test)
                outputs_test = torch.squeeze(outputs_test)

            if args_shot.cls_par > 0:
                pred = mem_label[batch_idx * args_shot.batch_size:(batch_idx + 1) * args_shot.batch_size]
                classifier_loss = args_shot.cls_par * nn.CrossEntropyLoss()(outputs_test,
                                                                            pred)  # CE loss using the pseudo labels
            else:
                classifier_loss = torch.tensor(0.0).cuda()

            if args_shot.ent:
                softmax_out = nn.Softmax(dim=1)(outputs_test)
                entropy_loss = torch.mean(Entropy(softmax_out))
                if args_shot.gent:
                    msoftmax = softmax_out.mean(dim=0)
                    entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

                im_loss = entropy_loss * args_shot.ent_par
                classifier_loss += im_loss

            classifier_loss.backward()
            optimizer.step()
            losses.update(classifier_loss.item(), labels.size(0))
            if args.verbose:
                logger.debug(('SHOT Training: [{0}/{1}]\t'
                              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(batch_idx, len(teloader), loss=losses)))

        if args.arch == 'tanet':
            ext.module.new_fc = classifier # simply put the classifier back in place of nn.Identity()
            top_1_acc = validate(teloader, ext, criterion, 0, epoch=epoch, args=args, logger=logger)
            shot_acc.append(top_1_acc)
            ext.module.new_fc = nn.Identity()
        else:
            adapted_model = nn.Sequential(*(list(ext.children()) + list(classifier.children())))
            adapted_model = adapted_model.cuda()
            top_1_acc = validate(teloader, adapted_model, criterion, 0, epoch=epoch, args=args, logger=logger)
            shot_acc.append(top_1_acc)
    return max(shot_acc)
