import torch
import torch.nn as nn


def rbf_kernel(x, y, sigma=1.0):
    """
    Radial Basis Function (Gaussian) kernel

    Args:
        x: Tensor of shape (N, D)
        y: Tensor of shape (M, D)
        sigma: Bandwidth of the RBF kernel

    Returns:
        Kernel matrix of shape (N, M)
    """
    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    y_norm = (y ** 2).sum(dim=1, keepdim=True)

    # Pairwise squared Euclidean distances
    dist = x_norm + y_norm.T - 2.0 * torch.matmul(x, y.T)

    return torch.exp(-dist / (2 * sigma ** 2))


class MMDLoss(nn.Module):
    """
    MMD loss between aggregated posterior q(z)
    and prior p(z) ~ N(0, I)
    """

    def __init__(self, sigma=1.0):
        super(MMDLoss, self).__init__()
        self.sigma = sigma

    def forward(self, z):
        """
        Args:
            z: Latent samples from encoder, shape (batch_size, z_dim)

        Returns:
            Scalar MMD loss
        """
        batch_size, z_dim = z.size()

        # Sample from prior p(z) ~ N(0, I)
        prior_z = torch.randn_like(z)

        K_zz = rbf_kernel(z, z, self.sigma)
        K_pp = rbf_kernel(prior_z, prior_z, self.sigma)
        K_zp = rbf_kernel(z, prior_z, self.sigma)

        mmd = K_zz.mean() + K_pp.mean() - 2.0 * K_zp.mean()
        return mmd
