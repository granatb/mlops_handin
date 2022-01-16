from typing import Callable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt # type: ignore
import torch
import torch.nn.functional as F
from torch import nn


class MyAwesomeModel(nn.Module):
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


def validation(
    model: nn.Module,
    testloader: torch.utils.data.DataLoader,
    criterion: Union[Callable, nn.Module],
) -> Tuple[float, float]:
    accuracy = 0
    test_loss = 0
    for images, labels in testloader:

        # images = images.resize_(images.size()[0], 784)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ## Calculating the accuracy
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = labels.data == ps.max(1)[1]
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss, accuracy


def train(
    model: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    criterion: Union[Callable, nn.Module],
    optimizer: Optional[torch.optim.Optimizer] = None,
    epochs: int = 5,
    print_every: int = 40,
) -> None:
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    steps = 0
    running_loss = 0
    train_losses = []
    val_losses = []
    iter_num = []

    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        for images, labels in trainloader:
            steps += 1

            # Flatten images into a 784 long vector
            # images.resize_(images.size()[0], 784)

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Model in inference mode, dropout is off
                model.eval()

                # Turn off gradients for validation, will speed up inference
                with torch.no_grad():
                    test_loss, accuracy = validation(model, testloader, criterion)

                print(
                    "Epoch: {}/{}.. ".format(e + 1, epochs),
                    "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                    "Test Loss: {:.3f}.. ".format(test_loss / len(testloader)),
                    "Test Accuracy: {:.3f}".format(accuracy / len(testloader)),
                )

                train_losses.append(running_loss / print_every)
                val_losses.append(test_loss / len(testloader))
                iter_num.append(steps)

                running_loss = 0

                # Make sure dropout and grads are on for training
                model.train()

    plt.plot(iter_num, train_losses, label="train_loss")
    plt.plot(iter_num, val_losses, label="test_loss")
    plt.legend()
    plt.savefig("reports/figures/training.png")
