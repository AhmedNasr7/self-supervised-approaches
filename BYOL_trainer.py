import torch
from torch import nn
import numpy as np 
# from models.resnet1d import ResNet1d_Backbone
from models.byol import BYOLModel
import lightning as L
from transforms import *
from datasets import LitDataModule
import torch.nn.functional as F
from watermark import watermark
import os


class BYOLTrainer(L.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()

        self.model = model
        self.online_network = self.model.online_network
        self.target_network = self.model.target_network

        self.learning_rate = learning_rate

        # Save settings and hyperparameters to the log directory
        # but skip the model parameters
        self.save_hyperparameters(ignore=["model"])
        self.m = 0.996 # momentum 

    
    def _transforms(self, x: torch.Tensor):

        x1, x2 = TwoCropsTransform()(x)

        return x1, x2

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def _compute_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def forward(self, batch_view_1, batch_view_2):
        # compute query feature
        predictions_from_view_1 = self.predictor(self.online_network(batch_view_1))
        predictions_from_view_2 = self.predictor(self.online_network(batch_view_2))

        # compute key features
        with torch.no_grad():
            targets_to_view_1 = self.target_network(batch_view_1)
            targets_to_view_2 = self.target_network(batch_view_2)

        return predictions_from_view_1, predictions_from_view_2, targets_to_view_1, targets_to_view_2

        

    def _forward_step(self, batch):
        x, y = batch

        x1, x2 = self._transforms(x)


        predictions_from_view_1, predictions_from_view_2, targets_to_view_1, targets_to_view_2 = self(x1, x2)

        loss = self._criterion(predictions_from_view_1, predictions_from_view_2, targets_to_view_1, targets_to_view_2)

        return loss

    def training_step(self, batch, batch_idx):

        self.initializes_target_network()
        loss = self._forward_step(batch)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=True)

        
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        

        # self._update_target_network_parameters()  # update the key encoder


        # save checkpoints
        self.save_model(os.path.join("./", 'model.pth'))
      
        return loss
    

    def on_before_zero_grad(self, _):
        self._update_target_network_parameters()

    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer
    

    def validation_step(self, batch, batch_idx):
        pass
        

    def test_step(self, batch, batch_idx):
        pass
  

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer
    
           

    def _critetion(self, predictions_from_view_1, predictions_from_view_2, targets_to_view_1, targets_to_view_2):

        
        loss = self._compute_loss(predictions_from_view_1, targets_to_view_1)
        loss += self._compute_loss(predictions_from_view_2, targets_to_view_2)

        return loss.mean()

    def save_model(self, PATH):

        torch.save({
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            # 'optimizer_state_dict': self.optimizer.state_dict(),
        }, PATH)


# get_transforms_list

class BYOL(L.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()
        self.automatic_optimization = False
        self.learning_rate = learning_rate
        self.model = model
 


        # Save settings and hyperparameters to the log directory
        # but skip the model parameters
        self.save_hyperparameters(ignore=["model"])
        self.criterion = nn.CrossEntropyLoss()

    def _transforms(self, x: torch.Tensor):
        transforms_list = get_transforms_list()
        x = MultiViewDataInjector(transforms_list)(x)

        print("x shape mv: ", x.shape)

        return x
      

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

        x = self._transforms(x)
        exit(0)

        p1, p2, z1, z2 = self(x1, x2)

        loss = self._criterion(p1, z1, p2, z2)


        return loss

    def training_step(self, batch, batch_idx):
        loss = self._forward_step(batch)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=True)
      
        return loss

    def validation_step(self, batch, batch_idx):
        pass
        

    def test_step(self, batch, batch_idx):
        pass
  

    def configure_optimizers(self):
        optimizer = LARS(self.parameters(), lr=self.learning_rate)
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

    model = BYOLModel()

    # lit_model = SimSiam_LitModel(model, 1e-4).cuda()
    # output = lit_model(x, x)


    lightning_model = BYOL(model=model, learning_rate=1e-4)

    trainer = L.Trainer(
        max_epochs=10, accelerator="cuda", devices="auto", deterministic=True
    )

    data_module = LitDataModule(root_dir="./data/", batch_size=32)

    trainer.fit(model=lightning_model, datamodule=data_module)

