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


with open("configs/base_config.json") as f:
    config = json.load(f)

device = "cpu"


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


set_seed()

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
num_workers = 0 if device == "cuda" else 0
pin_memory = True if device == "cuda" else False
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=config["training"]["batch_size"],
    num_workers=num_workers,
    pin_memory=pin_memory,
    shuffle=True,
    drop_last=True,
)

trainer = Trainer(config=config, dataloader=dataloader, log=config["training"]["log"])

if __name__ == "__main__":

    trainer.train()
    gc.collect()
