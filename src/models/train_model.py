import argparse
import os
import sys

import hydra
from omegaconf import OmegaConf
import model
import torch
from model import MyAwesomeModel
from torch import nn, optim
from torch.utils.data import Dataset

import argparse


sys.path.insert(1, os.path.join(sys.path[0], ".."))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from data.make_dataset import MNISTdata
import logging

log = logging.getLogger(__name__)
print("Working directory : {}".format(os.getcwd()))
@hydra.main(config_name="training_conf.yml", config_path="config") # hydra currently supports only 1 config file
def main(cfg):
    print(f"configuration: \n {OmegaConf.to_yaml(cfg)}")
    os.chdir(hydra.utils.get_original_cwd())
    hparams = cfg.experiment
    torch.manual_seed(hparams["seed"])

    my_model = MyAwesomeModel(128, 10)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(my_model.parameters(), lr=hparams['lr'])

    train_data = torch.load("data/processed/train.pth")
    test_data = torch.load("data/processed/test.pth")

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=hparams["batch_size"], shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=hparams["batch_size"], shuffle=True)
    
    log.info("Start training...")
    model.train(my_model, trainloader, testloader, criterion, optimizer, epochs=hparams["n_epochs"])

    checkpoint = {
        "hidden_size": 128,
        "output_size": 10,
        "state_dict": my_model.state_dict(),
    }
    log.info("Finish!!")

    torch.save(checkpoint, "models/checkpoint.pth")


if __name__ == "__main__":
    main()
