from src.dataset import CelebADataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from src.model import *
from src.trainer import Trainer
import gc


device = "cpu"
BS = 2
D = 64  # stylecode size
K = 40  # number of domains

transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

dataset = CelebADataset(root_dir="./data/", transform=transform)
num_workers = 0 if device == "cuda" else 2
pin_memory = True if device == "cuda" else False
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BS,
    num_workers=num_workers,
    pin_memory=pin_memory,
    shuffle=True,
)
# generator = Generator()
# discriminator = Discriminator(K=K)
# mapping_network = MappingNetwork(K=K, D=D)
# style_encoder = StyleEncoder(K=K, D=D)

trainer = Trainer(device="cpu", K=40, D=64, BS=2, EPOCHS=1, dataloader=dataloader)


if __name__ == "__main__":
    trainer.train()
    # for i in range(3):
    #     batch = next(iter(dataloader))
    #     y_trg = torch.randint(size=(1, 1), low=0, high=K - 1).item()
    #     im = batch[0]
    #     z = torch.randn((BS, 16))
    #     s = mapping_network(z, y_trg)
    #     out = generator(im, s)
    #     s_fake = style_encoder(out, y_trg)
    #     out_reversed = generator(im, s_fake)
    #     dout = discriminator(out_reversed, y_trg)
    #     l1 = adversarial_loss(im, out)  # fix
    #     l2 = style_rec_loss(s, s_fake)
    #     l3 = style_div_loss(out_reversed, out_reversed)
    #     l4 = cycle_loss(out_reversed, im)
    #     l1.backward()
    #     # print(l1, l2, l3, l4)
    # gc.collect()
