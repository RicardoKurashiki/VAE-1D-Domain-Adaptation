import torch
import torch.nn as nn

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


class _Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(_Encoder, self).__init__()
        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        h_ = self.LeakyReLU(self.FC_input(x))
        h_ = self.LeakyReLU(self.FC_input2(h_))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)
        return mean, log_var


class _Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, latent_dim):
        super(_Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.LeakyReLU(self.FC_hidden(x))
        h = self.LeakyReLU(self.FC_hidden2(h))
        x_hat = self.FC_output(h)
        return x_hat


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, latent_dim=64):
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = _Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = _Decoder(input_dim, hidden_dim, latent_dim)

    def reparameterization(self, mean, std):
        epsilon = torch.randn_like(std)
        z = mean + std * epsilon
        return z

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decoder(z)
        return x_hat, z, mean, log_var
