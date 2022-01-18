import os
import sys
from typing import Callable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt  # type: ignore
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import loggers
from torch import nn
from torch.utils.data import Dataset

sys.path.insert(1, os.path.join(sys.path[0], ".."))
import wandb
from data.make_dataset import MNISTdata


class MyLightningModel(pl.LightningModule):
    def __init__(self, hidden_size: int, output_size: int, drop_p: float = 0.3) -> None:
        """Builds a feedforward network with arbitrary hidden layers.

        Arguments
        ---------
        hidden_size: integer, size of dense layer
        output_size: number of classes
        drop_p: dropout rate

        """
        super().__init__()
        # Input to a hidden layer
        self.num_classes = output_size

        self.arch = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=3, padding=1, stride=1
            ),
            # convolution output dim (16, 28, 28)
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # pooling output dim (16, 14, 14)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=5, padding=2),
            nn.Dropout2d(p=drop_p),
            # convolution output dim (8, 14, 14)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # polling output dim (8, 7, 7)
            nn.ReLU(inplace=True),
        )

        # fully connected output layers
        # [(W−K+2P)/S]+1
        self.fc1_features = 8 * 7 * 7
        self.fc1 = nn.Linear(in_features=self.fc1_features, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=self.num_classes)

    def forward(self, x):

        x = self.arch(x)
        x = x.view(-1, self.fc1_features)
        x = F.relu(self.fc1(x))

        return F.log_softmax(self.fc2(x), dim=1)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        images, labels = batch
        x = self.arch(images)
        x = x.view(-1, self.fc1_features)
        x = F.relu(self.fc1(x))
        x_hat = F.log_softmax(self.fc2(x), dim=1)
        loss = F.nll_loss(x_hat, labels)
        self.log("train_loss", loss)
        self.logger.experiment.log({"logits": wandb.Histogram(x_hat.detach().numpy())})
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        y_hat = self(images)
        val_loss = F.nll_loss(y_hat, labels)
        self.log("val_loss", val_loss)
        return val_loss


def main():

    train_data = torch.load("data/processed/train.pth")
    test_data = torch.load("data/processed/test.pth")

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

    model = MyLightningModel(128, 10)
    wd_logger = loggers.WandbLogger(name="test")
    trainer = pl.Trainer(logger=wd_logger, max_epochs=5)

    trainer.fit(model, trainloader, testloader)


if __name__ == "__main__":
    main()
