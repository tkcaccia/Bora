"""RGB decoder for final Virchow2 spatial tokens; no source-pixel input."""
import torch
from torch import nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(width, width, 3, padding=1))

    def forward(self, x):
        return x + .2 * self.net(x)


class Decoder(nn.Module):
    def __init__(self, mean, scale, width=256, blocks=3):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean).float().reshape(1, 1280, 1, 1))
        self.register_buffer("scale", torch.as_tensor(scale).float().reshape(1, 1280, 1, 1))
        self.net = nn.Sequential(nn.Conv2d(1280, width, 1), nn.GELU(),
                                 *[Residual(width) for _ in range(blocks)],
                                 nn.Conv2d(width, 3 * 14 * 14, 1), nn.PixelShuffle(14))

    def forward(self, tokens):
        return self.net((tokens - self.mean) / self.scale).sigmoid()


def reconstruction_loss(prediction, target):
    l1 = F.l1_loss(prediction, target)
    gradient = (F.l1_loss(prediction[..., 1:, :]-prediction[..., :-1, :],
                          target[..., 1:, :]-target[..., :-1, :]) +
                F.l1_loss(prediction[..., :, 1:]-prediction[..., :, :-1],
                          target[..., :, 1:]-target[..., :, :-1])) / 2
    # Local SSIM in native pixel coordinates; not a learned perceptual prior.
    pool = lambda x: F.avg_pool2d(x, 7, stride=2)
    mu, mt = pool(prediction), pool(target)
    vp = (pool(prediction.square())-mu.square()).clamp_min(0)
    vt = (pool(target.square())-mt.square()).clamp_min(0)
    cov = pool(prediction*target)-mu*mt
    ssim = (((2*mu*mt+.01**2)*(2*cov+.03**2)) /
            ((mu.square()+mt.square()+.01**2)*(vp+vt+.03**2))).mean()
    return l1 + .2*gradient + .15*(1-ssim), (l1, gradient, ssim)
