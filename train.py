from src.dataset import CelebADataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from src.model import *
from src.trainer import Trainer
import gc
import json
import os
import random
import numpy as np
import argparse


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="base_config",
        type=str,
        help="config name",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    set_seed()
    args = parse_args()

    with open(f"configs/{args.config}.json") as f:
        config = json.load(f)

    transform = transforms.Compose(
        [
            transforms.Resize((config["data"]["size"], config["data"]["size"])),
            transforms.CenterCrop(config["data"]["size"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    dataset = CelebADataset(
        root_dir="./data/",
        transform=transform,
        domains=config["data"]["domains"],
        limit=config["data"]["limit"],
    )

    val_size = config["data"]["val_size"]
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [len(dataset) - val_size, val_size]
    )

    device = "cpu"
    num_workers = 0 if device == "cuda" else 0
    pin_memory = True if device == "cuda" else False
    train_dataloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=True,
    )
    reference_dataloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"] * 2,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        drop_last=True,
    )

    trainer = Trainer(
        config=config,
        dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        reference_dataloader=reference_dataloader,
        log=config["training"]["log"],
    )

    trainer.train()
    gc.collect()
