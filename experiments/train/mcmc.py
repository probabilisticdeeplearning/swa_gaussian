"""Swag based MCMC sampling"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import tabulate
import torch
import torch.nn.functional as F
import torchvision

from swag import data, models, utils, losses
from swag.posteriors import SWAG


def parse_args():
    """Parse args"""
    parser = argparse.ArgumentParser(description="SGD/SWA training")
    parser.add_argument("--dir", type=Path, required=True,
                        help="training directory (default: None)")

    parser.add_argument("--dataset", type=str, default="CIFAR10",
                        help="dataset name (default: CIFAR10)")
    parser.add_argument("--data_path", type=str, required=True, metavar="PATH",
                        help="path to datasets location (default: None)")
    parser.add_argument("--use_test", dest="use_test", action="store_true",
                        help="use test dataset instead of validation")
    parser.add_argument("--split_classes", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=128, metavar="N",
                        help="input batch size (default: 128)")
    parser.add_argument("--num_workers", type=int, default=4, metavar="N",
                        help="number of workers (default: 4)")
    parser.add_argument("--model", type=str, required=True, metavar="MODEL",
                        help="model name (default: None)")

    parser.add_argument("--resume", type=str, default=None, metavar="CKPT",
                        help="checkpoint to resume training from")

    parser.add_argument("--epochs", type=int, default=200, metavar="N",
                        help="number of epochs to train (default: 200)")
    parser.add_argument("--save_freq", type=int, default=25, metavar="N",
                        help="save frequency (default: 25)")
    parser.add_argument("--eval_freq", type=int, default=5, metavar="N",
                        help="evaluation frequency (default: 5)")
    parser.add_argument("--learning_rate_init", type=float, default=0.01,
                        metavar="LR",
                        help="initial learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.9, metavar="M",
                        help="SGD momentum (default: 0.9)")
    parser.add_argument("--wd", type=float, default=1e-4,
                        help="weight decay (default: 1e-4)")

    parser.add_argument("--swa", action="store_true",
                        help="swa usage flag (default: off)")
    parser.add_argument("--swa_start", type=float, default=161, metavar="N",
                        help="SWA start epoch number (default: 161)")
    parser.add_argument("--swa_learning_rate", type=float, default=0.02,
                        metavar="LR",
                        help="SWA LR (default: 0.02)")
    parser.add_argument("--swa_c_epochs", type=int, default=1, metavar="N",
                        help="SWA model collect. freq/cycle length in epochs")
    parser.add_argument("--cov_mat", action="store_true",
                        help="save sample covariance")
    parser.add_argument("--max_num_models", type=int, default=20,
                        help="maximum number of SWAG models to save")

    parser.add_argument("--swa_resume", type=str, default=None, metavar="CKPT",
                        help="checkpoint to restor SWA from (default: None)")
    parser.add_argument("--loss", type=str, default="CE",
                        help="loss to use for training model")

    parser.add_argument("--seed", type=int, default=1, metavar="S",
                        help="random seed (default: 1)")
    parser.add_argument("--no_schedule", action="store_true",
                        help="store schedule")

    return parser.parse_args()


def prepare_directory(args):
    """Setup directory structure"""
    print("Preparing directory {}".format(args.dir))
    args.dir.mkdir(exist_ok=True)
    with open(os.path.join(args.dir, "command.sh"), "w") as cmd_file:
        cmd_file.write(" ".join(sys.argv))
        cmd_file.write("\n")


def torch_settings(args):
    """Pytorch settings"""
    args.device = None
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def init_model(args, model_cfg, num_classes):
    """Initialize pytorch model"""

    print("Preparing model")
    print(*model_cfg.args)
    swag_model = None
    model = model_cfg.base(*model_cfg.args, num_classes=num_classes,
                           **model_cfg.kwargs)
    model.to(args.device)

    if args.cov_mat:
        args.no_cov_mat = False
    else:
        args.no_cov_mat = True
    if args.swa:
        print("SWAG training")
        swag_model = SWAG(model_cfg.base, no_cov_mat=args.no_cov_mat,
                          max_num_models=args.max_num_models,
                          *model_cfg.args, num_classes=num_classes,
                          **model_cfg.kwargs)
        swag_model.to(args.device)
    else:
        print("SGD training")
    return model, swag_model


def update_learning_rate(args, epoch, optimizer):
    """Update learning rate

    Based on epoch and optimizer
    """

    learning_rate = float()
    if not args.no_schedule:
        learning_rate = schedule(args, epoch)
        utils.adjust_learning_rate(optimizer, learning_rate)
    else:
        learning_rate = args.learning_rate_init

    return learning_rate


def schedule(args, epoch):
    """Scheduler"""
    progress = epoch / (args.swa_start if args.swa else args.epochs)
    if args.swa:
        learning_rate_ratio = args.swa_learning_rate / args.learning_rate_init
    else:
        learning_rate_ratio = 0.01
    if progress <= 0.5:
        factor = 1.0
    elif progress <= 0.9:
        factor = 1.0 - (1.0 - learning_rate_ratio) * (progress - 0.5) / 0.4
    else:
        factor = learning_rate_ratio
    return args.learning_rate_init * factor


def setup_optimisation(args, model):
    """Setup optimization problem
    Use a slightly modified loss function that allows input of model
    """
    if args.loss == "CE":
        criterion = losses.cross_entropy
    elif args.loss == "adv_CE":
        criterion = losses.adversarial_cross_entropy

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate_init,
        momentum=args.momentum,
        weight_decay=args.wd
    )
    return criterion, optimizer


def load_checkpoint(args, model, model_cfg, optimizer, num_classes):
    """Restore stored network if given"""
    start_epoch = 0
    if args.resume is not None:
        print("Resume training from %s" % args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    if args.swa and args.swa_resume is not None:
        checkpoint = torch.load(args.swa_resume)
        swag_model = SWAG(model_cfg.base, no_cov_mat=args.no_cov_mat,
                          max_num_models=args.max_num_models,
                          loading=True, *model_cfg.args,
                          num_classes=num_classes, **model_cfg.kwargs)
        swag_model.to(args.device)
        swag_model.load_state_dict(checkpoint["state_dict"])

    return start_epoch


def display_progress(args, swag_res, values, weird_split):
    """Print training progress

    TODO: Figure out weird split
    """

    columns = ["ep", "learning_rate", "tr_loss", "tr_acc", "te_loss",
               "te_acc", "time", "mem_usage"]

    if args.swa:
        values = values[:-2] + [swag_res["loss"],
                                swag_res["accuracy"]] + values[-2:]
    table = tabulate.tabulate([values], columns,
                              tablefmt="simple", floatfmt="8.4f")
    if weird_split:
        table = table.split("\n")
        table = "\n".join([table[1]] + table)
    else:
        table = table.split("\n")[2]
    print(table)


def save_model(args, epoch, model, optimizer, swag_model):
    """Save model to file"""
    utils.save_checkpoint(
        args.dir,
        epoch + 1,
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict()
    )
    if args.swa:
        utils.save_checkpoint(
            args.dir,
            epoch + 1,
            name="swag",
            state_dict=swag_model.state_dict(),
        )


def calculate_test_scores(args, epoch, loaders, model, criterion):
    """Test (validation) scores

    Runs first, last and every args.freq'th epoch
    """
    test_res = {"loss": None, "accuracy": None}
    if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
        test_res = utils.eval(loaders["test"], model, criterion)

    return test_res


def calculate_swag_metrics(args, epoch, loaders, model, criterion,
                           sgd_targets, sgd_ens_preds,
                           n_ensembled, swag_model):
    """Calculate and store swag metrics"""

    swag_res = {"loss": None, "accuracy": None}
    if args.swa and (epoch + 1) > args.swa_start\
            and (epoch + 1 - args.swa_start) % args.swa_c_epochs == 0:
        sgd_res = utils.predict(loaders["test"], model)
        sgd_preds = sgd_res["predictions"]
        sgd_targets = sgd_res["targets"]
        print("updating sgd_ens")
        if sgd_ens_preds is None:
            sgd_ens_preds = sgd_preds.copy()
        else:
            # TODO: rewrite in a numerically stable way
            term_1 = sgd_ens_preds * n_ensembled / (n_ensembled + 1)
            term_2 = sgd_preds / (n_ensembled + 1)
            sgd_ens_preds = term_1 + term_2
        n_ensembled += 1
        swag_model.collect_model(model)
        if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or\
                epoch == args.epochs - 1:
            swag_model.sample(0.0)
            utils.bn_update(loaders["train"], swag_model)
            swag_res = utils.eval(loaders["test"], swag_model, criterion)
        else:
            swag_res = {"loss": None, "accuracy": None}

    return swag_res, sgd_targets


def main():
    """Main entry point"""
    args = parse_args()
    prepare_directory(args)
    torch_settings(args)

    print("Using model {}".format(args.model))
    model_cfg = getattr(models, args.model)

    print("Loading dataset {} from {}".format(args.dataset, args.data_path))
    loaders, num_classes = data.loaders(
        args.dataset,
        args.data_path,
        args.batch_size,
        args.num_workers,
        model_cfg.transform_train,
        model_cfg.transform_test,
        use_validation=not args.use_test,
        split_classes=args.split_classes
    )

    model, swag_model = init_model(args, model_cfg, num_classes)
    criterion, optimizer = setup_optimisation(args, model)
    start_epoch = load_checkpoint(args, model, model_cfg,
                                  optimizer, num_classes)

    utils.save_checkpoint(
        args.dir,
        start_epoch,
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict()
    )

    sgd_ens_preds = None
    sgd_targets = None
    n_ensembled = 0.

    for epoch in range(start_epoch, args.epochs):
        time_ep = time.time()

        learning_rate = update_learning_rate(args, epoch, optimizer)
        train_res = utils.train_epoch(loaders["train"],
                                      model, criterion, optimizer)

        test_res = calculate_test_scores(args, epoch,
                                         loaders, model, criterion)
        swag_res, sgd_targets = calculate_swag_metrics(args, epoch, loaders,
                                                       model, criterion,
                                                       sgd_targets,
                                                       sgd_ens_preds,
                                                       n_ensembled, swag_model)
        if (epoch + 1) % args.save_freq == 0:
            save_model(args, epoch, model, optimizer, swag_model)

        time_ep = time.time() - time_ep
        memory_usage = torch.cuda.memory_allocated()/(1024.0 ** 3)
        values = [epoch + 1,
                  learning_rate,
                  train_res["loss"],
                  train_res["accuracy"],
                  test_res["loss"],
                  test_res["accuracy"],
                  time_ep,
                  memory_usage]
        display_progress(args, swag_res, values, epoch % 40 == 0)

    if args.epochs % args.save_freq != 0:
        utils.save_checkpoint(
            args.dir,
            args.epochs,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict()
        )
        if args.swa and args.epochs > args.swa_start:
            utils.save_checkpoint(
                args.dir,
                args.epochs,
                name="swag",
                state_dict=swag_model.state_dict(),
            )

    if args.swa:
        np.savez(args.dir / "sgd_ens_preds.npz",
                 predictions=sgd_ens_preds,
                 targets=sgd_targets)


if __name__ == "__main__":
    main()
