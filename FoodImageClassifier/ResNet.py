from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn

class ResNetFood(nn.Module):
    def __init__(self, num_classes=11, dropout=0.4, pretrained=True):
        super().__init__()

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.model = resnet50(weights=weights)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)
