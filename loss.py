from utils import *

class PEALoss(nn.Module):
    def __init__(
        self, tau=0.01, epsilon=0.005, delta=0.001, alpha=0.5, reduction="mean"
    ):
        super(PEALoss, self).__init__()
        self.tau = tau
        self.epsilon = epsilon
        self.delta = delta
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target):
        mse_per_pixel = torch.mean((pred - target) ** 2, dim=1)
        mask = (mse_per_pixel > self.tau).float()  # shape: (B, H, W)
        masked_error = mse_per_pixel * mask  # (B, H, W)
        L_masked = masked_error.mean()  # scalar
        indicator = (mse_per_pixel < self.epsilon).float()  # (B, H, W)
        boosted_per_pixel = mse_per_pixel + self.delta * indicator
        L_boost = boosted_per_pixel.mean()  # scalar
        loss = L_masked + self.alpha * L_boost

        if self.reduction == "none":
            mask_sample = (mse_per_pixel > self.tau).float()
            masked_error_sample = (
                (mse_per_pixel * mask_sample).view(pred.size(0), -1).mean(dim=1)
            )
            indicator_sample = (mse_per_pixel < self.epsilon).float()
            boosted_sample = (
                (mse_per_pixel + self.delta * indicator_sample)
                .view(pred.size(0), -1)
                .mean(dim=1)
            )
            loss_sample = masked_error_sample + self.alpha * boosted_sample
            return loss_sample

        return loss