from model import MyAwesomeModel
import model
import click

import argparse
import sys

import torch
from torch import nn
from torch.utils.data import Dataset
from torch import optim
import numpy as np
from torchvision import transforms
import logging
from dotenv import find_dotenv, load_dotenv

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from data.make_dataset import MNISTdata


@click.command()
@click.argument('model_filepath', type=click.Path(exists=True))
@click.argument('data_filepath', type=click.Path())
def main(model_filepath, data_filepath):
    '''
    Returns the loss and accuracy for a given data using a pretrained network.

            Parameters:
                    model_filepath (string): path to a pretrained model
                    data_filepath (string): path to raw data

    '''
    def load_checkpoint(filepath):
        checkpoint = torch.load(filepath)
        my_model = MyAwesomeModel(checkpoint['hidden_size'],
                                checkpoint['output_size'])
        my_model.load_state_dict(checkpoint['state_dict'])
        
        return my_model

    print("Evaluating until hitting the ceiling")
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('load_model_from', default="")
    # add any additional argument that you want
    args = parser.parse_args(sys.argv[2:])
    print(args)
    
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    # TODO: Implement evaluation logic here
    my_model = load_checkpoint(model_filepath)
    
    X_test = np.load(data_filepath)["images"]
    Y_test = np.load(data_filepath)["labels"]
    sample_data = MNISTdata(X_test, Y_test, transform=transform)
    sampleloader = torch.utils.data.DataLoader(sample_data, batch_size=64, shuffle=True)
    criterion = nn.NLLLoss()

    sample_loss, accuracy = model.validation(my_model, sampleloader, criterion)
    print(f'test loss: {sample_loss}, accuracy: {accuracy}')

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())

    main()