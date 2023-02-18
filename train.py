from src.dataset import CelebADataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from src.model import *
from src.trainer import Trainer
import gc
import json


with open("configs/base_config.json") as f:
    config = json.load(f)

device = "cpu"

transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


dataset = CelebADataset(
    root_dir="./data/", transform=transform, domains=config["data"]["domains"]
)
num_workers = 0 if device == "cuda" else 2
pin_memory = True if device == "cuda" else False
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=config["training"]["batch_size"],
    num_workers=num_workers,
    pin_memory=pin_memory,
    shuffle=True,
)

trainer = Trainer(config=config, dataloader=dataloader, log=False)


if __name__ == "__main__":
    trainer.train()
