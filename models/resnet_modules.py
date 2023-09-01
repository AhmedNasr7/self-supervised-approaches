import torch
import torch.nn as nn


from .resnet1d import ResNet1D



class EncoderwithProjection(nn.Module):
    
    def __init__(
        self,
        length: int = 1200,

        freeze: bool = False
    ) -> None:

        super().__init__()

        # Encoder network
        self.encoder = Encoder(length=length)

        if freeze:
            for param in self.encoder.parameters():
                param.requres_grad = False

        # Linear projector
        self.projector = MLP

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.encoder(x)
        return self.projector(e)
    

class MLP(nn.Module):
    
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 2048,
        output_dim: int = 256
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)



class LinearClassifier(nn.Module):

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(                                                                                                           
            nn.Linear(input_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ProjectionMLP(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            # nn.BatchNorm1d(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class PredictorMLP(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Encoder(nn.Module):

    def __init__(
        self, length
    ):
        super().__init__()
        self.model = ResNet1D(
                        in_channels=1, 
                        base_filters=128, 
                        kernel_size=16, 
                        stride=2, 
                        n_block=16, 
                        groups=32,
                        n_classes=3, 
                        downsample_gap=6, 
                        increasefilter_gap=12,
                        zero_init_residual=True)
        
        self.emb_dim = length // 8 # for 1d signal
        
        # self.model = resnet #nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze()
    

class ResNet(nn.Module):
    
    def __init__(
        self,
        length: int,
        num_classes: int,
        freeze: bool
    ) -> None:

        super().__init__()

        # Encoder network
        self.encoder = Encoder(length=length)

        if freeze:
            for param in self.encoder.parameters():
                param.requres_grad = False

        # Linear classifier
        self.classifier = LinearClassifier(self.encoder.emb_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.encoder(x)
        return self.classifier(e)
    


