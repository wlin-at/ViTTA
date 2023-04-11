import torch.optim as optim
import baselines.tent as tent
import baselines.norm as norm
from utils.opts import parser
import baselines.shot as shot
import baselines.dua as dua

# hard coding tent arguments in order not to incorporate configs (taken from their github)
tent_args = parser.parse_args()
tent_args.STEPS = 1
tent_args.LR = 1e-5 # for i3d (batchsize 16) set it to 1e-3 --- (tanet batchsize is 12)
tent_args.BETA = 0.9
tent_args.WD = 0.
tent_args.EPISODIC = False


def setup_model(args, base_model, logger):
    """
    :param args: argument from the main file
    :param base_model: model to set up for adaptation
    :param logger: logger for keeping the logs
    :return: returns the base_model after setting up adaptation baseline
    """

    if args.baseline == "source":
        logger.info("test-time adaptation: NONE")
        model = setup_source(args, base_model, logger)  #  set model to eval()
        return model
    elif args.baseline == "norm":
        logger.info("test-time adaptation: NORM")
        model = setup_norm(args, base_model, logger)
        return model
    elif args.baseline == "tent":
        logger.info("test-time adaptation: TENT")
        model, optimizer = setup_tent(args, base_model, logger)
        return model, optimizer
    elif args.baseline == "shot":
        optimizer, classfier, ext = setup_shot(args, base_model, logger)
        return optimizer, classfier, ext
    elif args.baseline == "dua":
        model = setup_dua(args, base_model, logger)
        return model
    else:
        raise NotImplementedError('Baseline not implemented')


def setup_source(args, model, logger):
    """Set up the baseline source model without adaptation."""
    model.eval()
    # if args.verbose:
    #     logger.info(f"model for adaptation: %s", model)
    return model


def setup_dua(args, model, logger):
    """
    Set up DUA model.
    Do not reset stats. Freeze entire model except the Batch Normalization layer.
    """
    dua_model = dua.DUA(model)
    if args.verbose:
        logger.info(f"model for adaptation: %s", model)
    return dua_model


def setup_shot(args, model, logger):
    """Set up test-time shot.

    Adapts the feature extractor by keeping source predictions as hypothesis and entropy minimization.
    """
    optimizer, classifier, ext = shot.configure_shot(model, logger, args)
    if args.verbose:
        logger.info(f"model for adaptation: %s", model)
    return optimizer, classifier, ext


def setup_norm(args, model, logger):
    """Set up test-time normalization adaptation.

    Adapt by normalizing features with test batch statistics.
    The statistics are measured independently for each batch;
    no running average or other cross-batch estimation is used.
    """
    norm_model = norm.Norm(model)
    stats, stat_names = norm.collect_stats(model)
    if args.verbose:
        logger.info(f"model for adaptation: %s", model)
        logger.debug(f"stats for adaptation: %s", stat_names)
    return norm_model


def setup_tent(args, model, logger):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tent.configure_model(model)  #  set only Batchnorm3d layers to trainable,   freeze all the other layers
    params, param_names = tent.collect_params(model)  # collecting gamma and beta in all Batchnorm3d layers
    optimizer = setup_optimizer(params)  # todo hyperparameters are hard-coded above
    if args.verbose:
        logger.debug(f"model for adaptation: %s", model)
        logger.debug(f"params for adaptation: %s", param_names)
        logger.debug(f"optimizer for adaptation: %s", optimizer)
    return model, optimizer


def setup_optimizer(params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    return optim.Adam(params,
                      lr=tent_args.LR,
                      betas=(tent_args.BETA, 0.999),
                      weight_decay=tent_args.WD)
