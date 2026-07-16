import os
import torch

from .complete_model import CompleteModel

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


class ClassificationModel:
    def __init__(self, num_classes, backbone="resnet", trainable_layers=None,
                 classifier_type="default"):
        self.num_classes = num_classes
        self.backbone = backbone
        self.trainable_layers = trainable_layers
        self.classifier_type = classifier_type
        self.model = CompleteModel(
            num_classes=num_classes,
            backbone=backbone,
            trainable_layers=trainable_layers,
            classifier_type=classifier_type,
        )

    def summary(self):
        print(f"{'Layer Name':50} {'Type':20} {'Trainable'}")
        print("-" * 90)

        for name, module in self.model.named_modules():
            if name == "":
                continue
            has_params = any(True for _ in module.parameters(recurse=False))
            if has_params:
                trainable = all(
                    p.requires_grad for p in module.parameters(recurse=False)
                )
                print(f"{name:50} {module.__class__.__name__:20} {trainable}")

    def save_architecture(self, output_path):
        os.makedirs(output_path, exist_ok=True)

        architecture_text = []
        architecture_text.append("=" * 90)
        architecture_text.append("MODEL ARCHITECTURE")
        architecture_text.append("=" * 90)
        architecture_text.append(f"\nBackbone: {self.backbone}")
        architecture_text.append(f"Number of classes: {self.num_classes}")
        architecture_text.append(f"Trainable layers: {self.trainable_layers}")
        architecture_text.append(f"Classifier type: {self.classifier_type}")
        architecture_text.append("\n" + "-" * 90)
        architecture_text.append(
            f"{'Layer Name':50} {'Type':20} {'Trainable':10} {'Parameters'}"
        )
        architecture_text.append("-" * 90)

        total_params = 0
        trainable_params = 0

        for name, module in self.model.named_modules():
            if name == "":
                continue
            has_params = any(True for _ in module.parameters(recurse=False))
            if has_params:
                trainable = all(
                    p.requires_grad for p in module.parameters(recurse=False)
                )
                num_params = sum(p.numel() for p in module.parameters(recurse=False))
                total_params += num_params
                if trainable:
                    trainable_params += num_params
                architecture_text.append(
                    f"{name:50} {module.__class__.__name__:20} {str(trainable):10} {num_params:,}"
                )

        architecture_text.append("-" * 90)
        architecture_text.append(f"\nTotal parameters: {total_params:,}")
        architecture_text.append(f"Trainable parameters: {trainable_params:,}")
        architecture_text.append(
            f"Frozen parameters: {total_params - trainable_params:,}"
        )
        architecture_text.append("=" * 90)

        architecture_text.append("\n\nFULL MODEL REPRESENTATION:\n")
        architecture_text.append(str(self.model))

        architecture_file = os.path.join(output_path, "model_architecture.txt")
        with open(architecture_file, "w", encoding="utf-8") as f:
            f.write("\n".join(architecture_text))

        return architecture_file

    def save_weights(self, output_path, config=None, epoch=None, metrics=None):
        self.model.save_weights(output_path, config=config, epoch=epoch, metrics=metrics)
        return output_path

    def load_weights(self, input_path):
        self.model.load_weights(input_path)
        print(f"Weights loaded from {input_path}")
        return input_path
