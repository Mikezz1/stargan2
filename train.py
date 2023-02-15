from torchvision.datasets.celeba import CelebADataset
from torchvision import transforms

transform = transforms.Compose(
    [
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = CelebADataset(split="train", transform=transform)
test_dataset = CelebADataset(split="test", transform=transform)
