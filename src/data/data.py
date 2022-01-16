import numpy as np
import torch
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class MNISTdata(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = self.transform(x)

        return x.float(), y

    def __len__(self):
        return len(self.data)


def mnist():
    # exchange with the corrupted mnist dataset
    # Define a transform to normalize the data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    train_paths = [f"corruptmnist/train_{i}.npz" for i in range(5)]

    X_train = np.concatenate(
        [np.load(train_file)["images"] for train_file in train_paths]
    )
    Y_train = np.concatenate(
        [np.load(train_file)["labels"] for train_file in train_paths]
    )

    X_test = np.load("corruptmnist/test.npz")["images"]
    Y_test = np.load("corruptmnist/test.npz")["labels"]

    train = MNISTdata(X_train, Y_train, transform=transform)
    trainloader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)

    test = MNISTdata(X_test, Y_test, transform=transform)
    testloader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)

    return trainloader, testloader
