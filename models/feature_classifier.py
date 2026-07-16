import torch.nn as nn


class FeatureClassifier(nn.Module):
    def __init__(self, num_classes, in_features, dropout=0.2):
        super(FeatureClassifier, self).__init__()
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)
