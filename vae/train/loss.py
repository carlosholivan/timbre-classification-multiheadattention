import torch
import torch.nn.functional as F

# Our modules
from vae import configs


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, loss='bce'):

    if loss == 'bce':
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    elif loss == 'mse':
        recon_loss = F.mse_loss(recon_x, x)
    elif loss == 'l1':
        recon_loss = F.l1_loss(recon_x, x)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + configs.ParamsConfig.VAE_BETA * kld, kld, recon_loss
