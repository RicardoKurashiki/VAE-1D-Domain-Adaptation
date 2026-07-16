import random
import torch


class AddRandomNoise:
    def __init__(self, std_range=(0.01, 0.05)):
        self.std_range = std_range

    def __call__(self, tensor):
        std = random.uniform(self.std_range[0], self.std_range[1])
        noise = torch.randn_like(tensor) * std
        noisy_tensor = torch.clamp(tensor + noise, 0.0, 1.0)
        return noisy_tensor
