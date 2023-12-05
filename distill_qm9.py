import argparse
import getpass
import pickle
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
from configs.datasets_config import get_dataset_info
from equivariant_diffusion.utils import (
    assert_mean_zero_with_mask,
    remove_mean_with_mask,
    sample_center_gravity_zero_gaussian_with_mask,
)
from qm9 import dataset
from qm9.models import get_latent_consistency, get_latent_diffusion, get_optim
from train_test import check_mask_correct
import utils


def parse_args():
    parser = argparse.ArgumentParser(description="distill ldm model into lcm")

    # Model Loading Args
    model_args = parser.add_argument_group("Model Loading Args")
    model_args.add_argument("--teacher_model", type=str, default=None, required=True)
    # model_args.add_argument(
    #     "--ae_path", type=str, default=None, help="Specify first stage model path"
    # )

    # Data Args
    data_args = parser.add_argument_group("Data Args")
    data_args.add_argument("--dataset", type=str, default="qm9")
    data_args.add_argument(
        "--datadir", type=str, default="qm9/temp", help="qm9 directory"
    )
    data_args.add_argument(
        "--remove_h", action="store_true", help="Remove node features"
    )
    data_args.add_argument("--batch_size", type=int, default=128)
    data_args.add_argument(
        "--num_workers", type=int, default=0, help="Num workers data loaders"
    )
    data_args.add_argument("--pin_memory", type=bool, default=True)
    data_args.add_argument(
        "--filter_n_atoms",
        type=int,
        default=None,
        help="When set to an integer value, QM9 will only contain molecules of that amount of atoms",
    )
    data_args.add_argument(
        "--include_charges", type=eval, default=True, help="include atom charge or not"
    )


    # Training Args
    training_args = parser.add_argument_group("Training Args")
    training_args.add_argument("--output_dir", type=str, default="geolcm_distilled")
    training_args.add_argument("--n_epochs", type=int, default=200)
    training_args.add_argument(
        "--ddp", type=eval, default=True, help="Use DistributedDataParallel"
    )
    training_args.add_argument(
        "--trainable_ae",
        action="store_true",
        help="Train first stage AutoEncoder model",
    )
    training_args.add_argument("--lr", type=float, default=2e-4)
    # training_args.add_argument(
    #     "--conditioning",
    #     nargs="+",
    #     default=[],
    #     help="arguments : homo | lumo | alpha | gap | mu | Cv",
    # )

    # Logging Args
    logging_args = parser.add_argument_group("Logging Args")
    logging_args.add_argument("--no_wandb", action="store_true")
    logging_args.add_argument("--run_name", type=str, default="debug")

    args = parser.parse_args()
    return args


def preprocess_data(args, data, device, dtype):
    x = data["positions"].to(device, dtype)
    node_mask = data["atom_mask"].to(device, dtype).unsqueeze(2)
    edge_mask = data["edge_mask"].to(device, dtype)
    one_hot = data["one_hot"].to(device, dtype)
    charges = (data["charges"] if args.include_charges else torch.zeros(0)).to(
        device, dtype
    )

    x = remove_mean_with_mask(x, node_mask)

    if args.augment_noise > 0:
        # Add noise eps ~ N(0, augment_noise) around points.
        eps = sample_center_gravity_zero_gaussian_with_mask(
            x.size(), x.device, node_mask
        )
        x = x + eps * args.augment_noise

    x = remove_mean_with_mask(x, node_mask)
    if args.data_augmentation:
        x = utils.random_rotation(x).detach()

    check_mask_correct([x, one_hot, charges], node_mask)
    assert_mean_zero_with_mask(x, node_mask)

    h = {"categorical": one_hot, "integer": charges}

    return x, h, node_mask, edge_mask


def train_epoch(
    args, loader, epoch, model, model_ema, ema, optim, gradnorm_queue, device, dtype
):
    model.train()
    for i, data in enumerate(loader):
        x, h, node_mask, edge_mask = preprocess_data(args, data, device, dtype)

        # TODO: implement conditioning
        context = None

        optim.zero_grad()

        loss = model(x, h, node_mask, edge_mask, context)
        loss.backward()

        if args.clip_grad:
            grad_norm = utils.gradient_clipping(model, gradnorm_queue)
        else:
            grad_norm = 0.0

        optim.step()

        # Update EMA
        if args.ema_decay > 0:
            ema.update_model_average(model_ema, model)


def main(args):
    # Set up DDP -------------
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()
    map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
    dtype = torch.float32
    print(device_id)

    # Set up Logging -----------
    wandb_usr = getpass.getuser()
    if args.no_wandb:
        mode = "disabled"
    else:
        mode = "online"
    wandb_kwargs = {
        "entity": wandb_usr,
        "name": args.run_name,
        "project": "latent_distillation_qm9",
        "config": args,
        "settings": wandb.Settings(_disable_stats=True),
        "reinit": True,
        "mode": mode,
    }
    wandb.init(**wandb_kwargs)
    wandb.save("*.txt")

    # Set up data -------------
    dataset_info = get_dataset_info(args.dataset, args.remove_h)
    dataloaders, charge_scale = dataset.retrieve_dataloaders(args)

    ## TODO: implement conditioning
    context_node_nf = 0
    property_norms = None

    # Load Model ---------------

    ## Load Teacher Model
    with open(f"{args.teacher_model}/args.pickle", "rb") as argfile:
        teacher_args = pickle.load(argfile)
    args.augment_noise = teacher_args.augment_noise
    args.data_augmentation = teacher_args.data_augmentation

    teacher_model, nodes_dist, prop_dist = get_latent_diffusion(
        teacher_args, device_id, dataset_info, dataloaders["train"], map_location
    )
    teacher_model.load_state_dict(
        torch.load(
            f"{args.teacher_model}/generative_model_ema.npy", map_location=map_location
        )
    )

    ## Load Student Model
    student_model, nodes_dist, prop_dist = get_latent_consistency(
        teacher_args, device_id, dataset_info, dataloaders["train"], map_location
    )
    student_model.load_state_dict(
        torch.load(
            f"{args.teacher_model}/generative_model_ema.npy", map_location=map_location
        )
    )
    student_model = DDP(student_model, device_ids=[rank])

    # Create Optimizer -----------
    optim = get_optim(args, student_model)
    gradnorm_queue = utils.Queue()
    gradnorm_queue.add(3000)  # Add large value that will be flushed.

    # Start Training --------------
    for epoch in range(args.n_epochs):
        start_epoch = time.time()
        train_epoch(
            args=args,
            loader=dataloaders["train"],
            epoch=epoch,
            model=student_model,
            model_ema=None,
            ema=None,
            optim=optim,
            gradnorm_queue=gradnorm_queue,
            device=device_id,
            dtype=dtype
        )
        print(f"Epoch {epoch} took {time.time() - start_epoch:.1f} seconds.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
