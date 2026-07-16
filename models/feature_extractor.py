import torch
import torch.nn as nn

from torchvision.models import resnet18, ResNet18_Weights

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


class FeatureExtractor(nn.Module):
    def __init__(self, backbone="resnet", trainable_layers=None, classifier_type="default"):
        super(FeatureExtractor, self).__init__()
        self.backbone = backbone
        self.trainable_layers = trainable_layers
        self.classifier_type = classifier_type
        self.num_ftrs = None
        self.output_features = None
        self._build_extractor()
        self._unfreeze_layers()

    def _build_extractor(self):
        self.weights = ResNet18_Weights.IMAGENET1K_V1
        base_model = resnet18(weights=self.weights)
        self.num_ftrs = base_model.fc.in_features

        self.backbone_model = nn.Sequential(*list(base_model.children())[:-1])
        self.output_features = self.num_ftrs

    def forward(self, x):
        x = self.backbone_model(x)
        x = x.view(x.size(0), -1)
        return x

    def _unfreeze_layers(self):
        # Congelar todo o backbone primeiro
        for p in self.backbone_model.parameters():
            p.requires_grad = False

        # Se trainable_layers for None, "none", ou <= 0, manter tudo congelado
        if self.trainable_layers is None or self.trainable_layers == "none":
            return

        if isinstance(self.trainable_layers, int) and self.trainable_layers <= 0:
            return

        # Se trainable_layers == "all", descongelar tudo
        if self.trainable_layers == "all":
            for p in self.backbone_model.parameters():
                p.requires_grad = True
            return

        # Se trainable_layers é um número, descongelar apenas as últimas N camadas Conv2d
        if isinstance(self.trainable_layers, int):
            indexed = [
                (idx, name, module)
                for idx, (name, module) in enumerate(self.named_modules())
            ]
            convs = [
                (idx, name, module)
                for idx, name, module in indexed
                if isinstance(module, nn.Conv2d)
            ]
            if len(convs) < self.trainable_layers:
                raise ValueError("O modelo não contém camadas Conv2d suficientes.")

            conv_idx = convs[-self.trainable_layers][0]
            for idx, _, module in indexed:
                if idx >= conv_idx:
                    for p in module.parameters(recurse=False):
                        p.requires_grad = True
