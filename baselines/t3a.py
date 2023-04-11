import torch.nn as nn
import torch
from utils.utils_ import *

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def get_cls_ext(args, net):

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

    return ext, classifier


class T3A(nn.Module):
    """
    Test Time Template Adjustments (T3A)

    """

    def __init__(self, args, ext, classifier):
        super().__init__()
        self.args = args
        self.model = ext
        self.classifier = classifier
        self.classifier.weight.requires_grad = False  # To save memory ...
        self.classifier.bias.requires_grad = False  # To save memory ...

        self.warmup_supports = self.classifier.weight.data
        warmup_prob = self.classifier(self.warmup_supports)
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.warmup_labels = torch.nn.functional.one_hot(warmup_prob.argmax(1), num_classes=args.num_classes).float()

        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data

        self.filter_K = args.t3a_filter_k
        self.num_classes = args.num_classes
        self.softmax = torch.nn.Softmax(-1)

    def forward(self, x):
        with torch.no_grad():
            z = self.model(x)
        # online adaptation
        p = self.classifier(z)
        yhat = torch.nn.functional.one_hot(p.argmax(1), num_classes=self.num_classes).float()
        ent = softmax_entropy(p)

        # prediction
        self.supports = self.supports.to(z.device)
        self.labels = self.labels.to(z.device)
        self.ent = self.ent.to(z.device)
        self.supports = torch.cat([self.supports, z])
        self.labels = torch.cat([self.labels, yhat])
        self.ent = torch.cat([self.ent, ent])

        supports, labels = self.select_supports()
        supports = torch.nn.functional.normalize(supports, dim=1)
        weights = (supports.T @ (labels))
        return z @ torch.nn.functional.normalize(weights, dim=0)

    def select_supports(self):
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s))))

        indices = []
        indices1 = torch.LongTensor(list(range(len(ent_s))))
        for i in range(self.num_classes):
            _, indices2 = torch.sort(ent_s[y_hat == i])
            indices.append(indices1[y_hat == i][indices2][:filter_K])
        indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]

        return self.supports, self.labels


def t3a_forward_and_adapt(args, ext, cls, val_loader):
    model = T3A(args, ext, cls)
    with torch.no_grad():
        total = 0
        correct_list = []
        top1 = AverageMeter()

        for i, (input, target) in enumerate(val_loader):  #
            ext.eval()
            cls.eval()
            actual_bz = input.shape[0]
            input = input.cuda()
            target = target.cuda()
            if args.arch == 'tanet':
                n_clips = int(args.sample_style.split("-")[-1])
                input = input.view(-1, 3, input.size(2), input.size(3))
                input = input.view(actual_bz * args.test_crops * n_clips,
                                       args.clip_length, 3, input.size(2), input.size(3))
                output = model(input)
                output = output.reshape(actual_bz, args.test_crops * n_clips, -1).mean(1)
                logits = torch.squeeze(output)
                prec1, prec5 = accuracy(logits.data, target, topk=(1, 5))
                top1.update(prec1.item(), actual_bz)
            else:
                input = input.reshape((-1,) + input.shape[2:])
                output = model(input)
                logits = torch.squeeze(output)
                prec1, prec5 = accuracy(logits.data, target, topk=(1, 5))
                top1.update(prec1.item(), actual_bz)
    return top1.avg

