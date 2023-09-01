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
from utils import *

class SimSiam_LitModel(L.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()

        self.learning_rate = learning_rate
        self.model = model
 


        # Save settings and hyperparameters to the log directory
        # but skip the model parameters
        self.save_hyperparameters(ignore=["model"])
        self.criterion = nn.CrossEntropyLoss()

    def _transforms(self, x: torch.Tensor):

        x1, x2 = TwoCropsTransform()(x)

        return x1, x2
      

    def _negative_cosine_similarity(self, p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """ D(p, z) = -(p*z).sum(dim=1).mean() """

        return - F.cosine_similarity(p, z.detach(), dim=-1).mean() # cosine similarity already normalize input
    
    def _criterion(self, p1: torch.Tensor, z1: torch.Tensor, p2: torch.Tensor, z2: torch.Tensor):
        loss_1 = self._negative_cosine_similarity(p1, z1)
        loss_2 = self._negative_cosine_similarity(p2, z2)
    
        return loss_1 /2 + loss_2 /2

    def forward(self, x1, x2):
        return self.model(x1, x2)

    def _forward_step(self, batch):
        x, y = batch

        x1, x2 = self._transforms(x)


        p1, p2, z1, z2 = self(x1, x2)

        loss = self._criterion(p1, z1, p2, z2)


        return loss

    def training_step(self, batch, batch_idx):

        loss = self._forward_step(batch)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=True)
      
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._forward_step(batch)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=True)
      
        return loss
        

  

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer
    



if __name__ == "__main__":

    print(watermark(packages="numpy,torch,lightning", python=True))


    samples_n = 1
    channels_n  = 1
    length = 1200
    classes_n = 3

    x = torch.zeros(samples_n, channels_n, length).cuda()

    latent_dim: int = 1024
    proj_hidden_dim: int = 1024 
    pred_hidden_dim: int = 256

    model = SimSiam(length, latent_dim, proj_hidden_dim, pred_hidden_dim)

    # lit_model = SimSiam_LitModel(model, 1e-4).cuda()
    # output = lit_model(x, x)


    lightning_model = SimSiam_LitModel(model=model, learning_rate=1e-4)

    trainer = L.Trainer(
        max_epochs=10, accelerator="cuda", devices="auto", deterministic=True
    )

    data_module = LitDataModule(root_dir="./data/", batch_size=32)

    trainer.fit(model=lightning_model, datamodule=data_module)

