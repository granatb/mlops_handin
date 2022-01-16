from model import MyAwesomeModel
import model

import argparse
import sys

import torch
from torch import nn
from torch.utils.data import Dataset
from torch import optim

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from data.make_dataset import MNISTdata

def main():
    
    my_model = MyAwesomeModel(128, 10)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(my_model.parameters(), lr=0.001)

    train_data = torch.load('data/processed/train.pth')
    test_data = torch.load('data/processed/test.pth')

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

    model.train(my_model, trainloader, testloader, criterion, optimizer, epochs=5)

    checkpoint = {'hidden_size': 128,
            'output_size': 10,
            'state_dict': my_model.state_dict()}

    torch.save(checkpoint, 'models/checkpoint.pth')

if __name__ == "__main__":
    main()