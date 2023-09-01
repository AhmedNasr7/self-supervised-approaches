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

from models.resnet_modules import *
from models.simclr import SimCLR
from utils import * 

   
class InfoNceLoss(nn.Module):
   """
   Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper
   """
   def __init__(self, batch_size, temperature=0.5):
       super().__init__()
       self.batch_size = batch_size
       self.temperature = temperature
       self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()

   def calc_similarity_batch(self, a, b):
       representations = torch.cat([a, b], dim=0)
       return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

   def forward(self, proj_1, proj_2):
       """
       proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
       where corresponding indices are pairs
       z_i, z_j in the SimCLR paper
       """
       batch_size = proj_1.shape[0]
       z_i = F.normalize(proj_1, p=2, dim=1)
       z_j = F.normalize(proj_2, p=2, dim=1)
       
       similarity_matrix = self.calc_similarity_batch(z_i, z_j)

       sim_ij = torch.diag(similarity_matrix, batch_size)
       sim_ji = torch.diag(similarity_matrix, -batch_size)

       positives = torch.cat([sim_ij, sim_ji], dim=0)

       nominator = torch.exp(positives / self.temperature)

       denominator = device_as(self.mask,similarity_matrix)  * torch.exp(similarity_matrix / self.temperature) 
       all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
       loss = torch.sum(all_losses) / (2 * self.batch_size)
       return loss


class SimCLR_LitModel(L.LightningModule):
   
   def __init__(self, model, learning_rate, batch_size=32):
       super().__init__()
       self.learning_rate = learning_rate
       self.model = model
       self.batch_size = batch_size
    #    self.max_epoch = 100

       self.save_hyperparameters(ignore=["model"])


       self._loss = InfoNceLoss(self.batch_size)

   def _transforms(self, x: torch.Tensor):

       x1, x2 = TwoCropsTransform()(x)

       return x1, x2
   

   
   def compute_loss(self, z1, z2):
        return self._loss(z1, z2)
        
   def forward(self, x1, x2):
        return self.model(x1, x2)

   def _forward_step(self, batch):
        x, y = batch

        x1, x2 = self._transforms(x)


        z1, z2 = self(x1, x2)

        loss = self.compute_loss(z1, z2)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=True)


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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=self.learning_rate / 50
        )

        return [optimizer], [lr_scheduler]
   
    



if __name__ == "__main__":

    print(watermark(packages="numpy,torch,lightning", python=True))

    max_epoch = 100
    batch_size = 4
    samples_n = 1
    channels_n  = 1
    length = 1200
    classes_n = 3

    x = torch.zeros(samples_n, channels_n, length).cuda()

    latent_dim: int = 1024
    proj_hidden_dim: int = 1024 
    pred_hidden_dim: int = 256

    model = SimCLR(length, latent_dim, proj_hidden_dim, pred_hidden_dim)


    lightning_model = SimCLR_LitModel(model=model, learning_rate=1e-4, batch_size=batch_size)

    trainer = L.Trainer(
        max_epochs=10, accelerator="cuda", devices="auto", deterministic=True
    )

    data_module = LitDataModule(root_dir="./data/", batch_size=batch_size)

    trainer.fit(model=lightning_model, datamodule=data_module)

