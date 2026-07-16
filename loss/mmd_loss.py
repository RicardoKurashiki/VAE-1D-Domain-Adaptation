import torch
import torch.nn as nn


class MMDLoss(nn.Module):
    """Maximum Mean Discrepancy (MMD) with a multi-kernel RBF.

    Mede a distância entre duas distribuições comparando seus mean embeddings
    num RKHS. É label-free e captura todos os momentos, sendo o critério canônico
    de domain adaptation (DAN/DDC) para aproximar a distribuição alvo da source.

    Usa um banco de kernels RBF com bandwidths derivados da heurística da mediana
    (median heuristic), evitando ter que escolher sigma na mão.

    Args:
        kernel_mul (float): fator multiplicativo entre bandwidths consecutivos.
        kernel_num (int): número de kernels RBF combinados.
    """

    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMDLoss, self).__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num

    def _gaussian_kernel(self, source, target):
        n_total = source.size(0) + target.size(0)
        total = torch.cat([source, target], dim=0)

        # distâncias L2 ao quadrado entre todos os pares
        l2_distance = torch.cdist(total, total).pow(2)

        # median heuristic: bandwidth base = soma das distâncias / nº de pares
        bandwidth = l2_distance.detach().sum() / (n_total ** 2 - n_total)
        bandwidth = torch.clamp(bandwidth, min=1e-8)
        bandwidth = bandwidth / (self.kernel_mul ** (self.kernel_num // 2))

        bandwidths = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        kernels = [torch.exp(-l2_distance / bw) for bw in bandwidths]
        return sum(kernels)

    def forward(self, source, target):
        """
        Args:
            source: features alinhadas (x_recon) com shape (batch_size, feat_dim).
            target: amostras reais do source/Kermany com shape (batch_size, feat_dim).
        """
        batch_size = source.size(0)
        kernels = self._gaussian_kernel(source, target)

        xx = kernels[:batch_size, :batch_size]
        yy = kernels[batch_size:, batch_size:]
        xy = kernels[:batch_size, batch_size:]
        yx = kernels[batch_size:, :batch_size]

        loss = xx.mean() + yy.mean() - xy.mean() - yx.mean()
        return loss
