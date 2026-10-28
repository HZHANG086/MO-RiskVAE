import torch
import torch.nn as nn

class HSICLoss(nn.Module):
    def __init__(self, sigma_x=1.0, sigma_y=1.0):
        super().__init__()
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def rbf_kernel(self, x, sigma):
        pairwise_sq_dists = torch.cdist(x, x, p=2).pow(2)
        return torch.exp(-pairwise_sq_dists / (2 * sigma ** 2))

    def forward(self, X, Y):
        """
        X: (n, d_x)  输入（多组学 concat 后）
        Y: (n, 1)    residual
        """
        n = X.size(0)
        K = self.rbf_kernel(X, self.sigma_x)
        L = self.rbf_kernel(Y, self.sigma_y)

        H = torch.eye(n, device=X.device) - 1.0 / n * torch.ones(n, n, device=X.device)

        hsic = torch.trace(K @ H @ L @ H) / ((n - 1) ** 2)
        return hsic
