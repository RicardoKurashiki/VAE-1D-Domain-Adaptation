import os
import yaml
from datetime import datetime
import torch
import torch.nn as nn

from .feature_extractor import FeatureExtractor
from .feature_classifier import FeatureClassifier

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


class CompleteModel(nn.Module):
    def __init__(self, num_classes, backbone="resnet", trainable_layers=None, classifier_type="default"):
        super(CompleteModel, self).__init__()
        self.extractor = FeatureExtractor(
            backbone=backbone,
            trainable_layers=trainable_layers,
            classifier_type=classifier_type,
        )
        input_features = self.extractor.output_features
        self.classifier = FeatureClassifier(num_classes=num_classes, in_features=input_features)

    def forward(self, x):
        f = self.extractor(x)
        c = self.classifier(f)
        return c

    def save_weights(self, output_path, config=None, epoch=None, metrics=None):
        os.makedirs(output_path, exist_ok=True)

        torch.save(
            self.extractor.state_dict(),
            os.path.join(output_path, "extractor_weights.pt"),
        )
        torch.save(
            self.classifier.state_dict(),
            os.path.join(output_path, "classifier_weights.pt"),
        )

        if config is not None:
            checkpoint = {
                'extractor_state_dict': self.extractor.state_dict(),
                'classifier_state_dict': self.classifier.state_dict(),
                'config': config,
                'epoch': epoch,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat(),
            }
            torch.save(checkpoint, os.path.join(output_path, "checkpoint.pt"))

            with open(os.path.join(output_path, "training_config.yaml"), 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

        return output_path

    def load_weights(self, input_path):
        self.extractor.load_state_dict(
            torch.load(
                os.path.join(input_path, "extractor_weights.pt"),
                map_location=device,
                weights_only=True,
            )
        )
        self.classifier.load_state_dict(
            torch.load(
                os.path.join(input_path, "classifier_weights.pt"),
                map_location=device,
                weights_only=True,
            )
        )
        return input_path
