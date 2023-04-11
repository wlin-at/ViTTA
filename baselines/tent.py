from copy import deepcopy
import torch.jit


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer, args=None, actual_bz=None, n_clips=None):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    outputs = model(x) # (batch * n_views, 3, T, 224,224 )  -> (batch * n_views, n_class ) todo clip-level prediction
    if args.arch == 'tanet':
        outputs = outputs.reshape(actual_bz, args.test_crops * n_clips, -1).mean(1)
    # adapt
    loss = softmax_entropy(outputs).mean(0)   #   todo compute the entropy for all clip-level predictions   then take the averaga among all samples
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale gamma, bias is shift beta
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.requires_grad_(True)
        #m.track_running_stats = True # for original implementation this is False
        #m.running_mean = None # for original implementation uncomment this
        #m.running_var = None # for original implementation uncomment this
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"

    has_bn = any([isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
