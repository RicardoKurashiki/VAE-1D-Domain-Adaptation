from models.vae import VariationalAutoEncoder


class AutoEncoder():
    def __init__(self, architecture, input_dim=256, n_classes=2):
        self.architecture = architecture
        if architecture == "vae":
            self.model = VariationalAutoEncoder(input_dim=input_dim)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
