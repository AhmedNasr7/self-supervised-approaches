import torch
from torch import nn
import numpy as np 
from models.resnet1d import ResNet1d_Backbone
from models.simsiam import SimSiam
import lightning as L
from transforms import *
from datasets import LitDataModule
import torch.nn.functional as F
from watermark import watermark

from tensorboardX import SummaryWriter
from torchsummary import summary

from models.simsiam import SimSiam
from SimSiam_trainer import SimSiam_LitModel


def train():

    L.seed_everything(42) # for reproducibility 


    samples_n = 1
    channels_n  = 1
    length = 1200
    classes_n = 3

    x = torch.zeros(samples_n, channels_n, length).cuda()

    latent_dim: int = 1024
    proj_hidden_dim: int = 1024 
    pred_hidden_dim: int = 256

    model = SimSiam(length, latent_dim, proj_hidden_dim, pred_hidden_dim) # change model 
    lightning_model = SimSiam_LitModel(model=model, learning_rate=1e-4) # change lit module


    data_module = LitDataModule(root_dir="./data/", batch_size=32)
    
    
    trainer = L.Trainer(
            max_epochs=10, accelerator="cuda", devices="auto", deterministic=True
            )

    trainer.fit(model=lightning_model, datamodule=data_module)






if __name__ == "__main__":

    print(watermark(packages="numpy,torch,lightning", python=True))


    train()