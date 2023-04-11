import torch
import torch.nn as nn
import torch.nn.functional as F

# def kl_div(input, targets):
# 	return F.kl_div(F.log_softmax(input, dim=1), targets, reduction='none').sum(1)

l1_loss = nn.L1Loss(reduction='sum')


def kl_div(input, targets):
	# return F.kl_div( input.log(), targets, reduction='none').sum(1)
	return F.kl_div( F.log_softmax(input, dim=1), targets, reduction='none' ).sum(1)

def compute_pred_consis( preds ):
    """
    :param preds:  in shape (batch_size, n_views, n_class) before softmax
    :return:
    """
    bz, n_views, n_class = preds.size()
    softmaxs = []
    for view_id in range(n_views):
        softmaxs += [F.softmax( preds[:, view_id, :], dim=1)]

    # avg_softmax = torch.stack(softmaxs, dim=0).mean(0).detach()
    avg_softmax = torch.stack(softmaxs, dim=0).mean(0)

    loss_consis = [ l1_loss(  softmaxs[view_id] , avg_softmax)        for view_id in range(n_views) ]
    # loss_consis = [  kl_div(  preds[:, view_id, :]  , avg_softmax)     for view_id in range(n_views)   ]
    loss_consis = sum(loss_consis) / n_views
    return loss_consis

    # avg_softmax = sum(softmaxs) / n_views



    # softmaxs = [ F.softmax()  for logit in preds[:, ]]