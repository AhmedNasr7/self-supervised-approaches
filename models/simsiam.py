
import torch
import torch.nn as nn

import torch
import torch.nn as nn
from .resnet_modules import *


class SimSiam(nn.Module):

    def __init__(
        self,
        length: int,
        latent_dim: int,
        proj_hidden_dim: int,
        pred_hidden_dim: int,
    ) -> None:

        super().__init__()

        # Encoder network
        self.encoder = Encoder(length)

        # Projection (mlp) network
        self.projection_mlp = ProjectionMLP(
            input_dim=self.encoder.emb_dim,
            hidden_dim=proj_hidden_dim,
            output_dim=latent_dim
        )

        # Predictor network (h)
        self.predictor_mlp = PredictorMLP(
            input_dim=latent_dim,
            hidden_dim=pred_hidden_dim,
            output_dim=latent_dim
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        e1, e2 = self.encode(x1), self.encode(x2)
        z1, z2 = self.project(e1), self.project(e2)
        p1, p2 = self.predict(z1), self.predict(z2)

        return p1, p2, z1.detach(), z2.detach()


    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def project(self, e: torch.Tensor) -> torch.Tensor:
        return self.projection_mlp(e)

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        return self.predictor_mlp(z)

