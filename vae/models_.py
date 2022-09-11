import torch
from torch.nn import functional as F
from torch import nn

# Our modules
from vae import configs


class Encoder(nn.Module):
    def __init__(self,
                 num_channels=configs.ParamsConfig.NUM_CHANNELS,
                 latent_dims=configs.ParamsConfig.LATENT_DIMS):
        super(Encoder, self).__init__()

        self.num_channels = num_channels
        self.latent_dims = latent_dims

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=self.num_channels*16,
                               kernel_size=(5, 5),
                               stride=(1, 1),
                               padding=(2, 2))

        self.conv2 = nn.Conv2d(in_channels=self.num_channels*16,
                               out_channels=self.num_channels*8,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1))

        self.conv3 = nn.Conv2d(in_channels=self.num_channels*8,
                               out_channels=self.num_channels*8,
                               kernel_size=(100, 3),
                               stride=(1, 1),
                               padding=(2, 1))

        self.conv4 = nn.Conv2d(in_channels=self.num_channels*8,
                               out_channels=self.num_channels*8,
                               kernel_size=(3, 3),
                               stride=(1, 2),
                               padding=(1, 1))

        self.conv5 = nn.Conv2d(in_channels=self.num_channels*8,
                               out_channels=self.num_channels,
                               kernel_size=(50, 13),
                               stride=(1, 8),
                               padding=(2, 1))

        self.fc_mu = nn.Linear(in_features=self.num_channels*50*5,
                               out_features=self.latent_dims)

        self.fc_logvar = nn.Linear(in_features=self.num_channels*50*5,
                                   out_features=self.latent_dims)

    def forward(self, x):
        #print('Input size to Conv1 encoder:', x.shape)

        x_conv1 = F.relu(self.conv1(x))
        #print('Output size Conv1 encoder', x_conv1.shape)

        x_conv2 = F.relu(self.conv2(x_conv1))
        #print('Output size Conv2 encoder', x_conv2.shape)

        x_conv3 = F.relu(self.conv3(x_conv2))
        #print('Output size Conv3 encoder', x_conv3.shape)

        x_conv4 = F.relu(self.conv4(x_conv3))
        #print('Output size Conv4 encoder', x_conv4.shape)

        x_conv5 = F.relu(self.conv5(x_conv4))
        #print('Output size Conv5 encoder', x_conv5.shape)

        x_flatten = x_conv5.view(x_conv5.size(0), -1)  # flatten batch of feature maps
        #print('Output size after flatten', x_flatten.shape)

        x_mu = self.fc_mu(x_flatten)
        #print('Output size of mu after fc', x_mu.shape)
        x_logvar = self.fc_logvar(x_flatten)
        #print('Output size of logvar after fc', x_logvar.shape)
        return x_mu, x_logvar


class Decoder(nn.Module):
    global input_shape

    def __init__(self,
                 num_channels=configs.ParamsConfig.NUM_CHANNELS,
                 latent_dims=configs.ParamsConfig.LATENT_DIMS):
        super(Decoder, self).__init__()

        self.num_channels = num_channels
        self.latent_dims = latent_dims

        self.fc = nn.Linear(in_features=self.latent_dims,
                            out_features=self.num_channels*50*5)

        self.conv5 = nn.ConvTranspose2d(in_channels=self.num_channels,
                                        out_channels=self.num_channels*8,
                                        kernel_size=(50, 16),
                                        stride=(1, 9),
                                        padding=(2, 1))

        self.conv4 = nn.ConvTranspose2d(in_channels=self.num_channels*8,
                                        out_channels=self.num_channels*8,
                                        kernel_size=(3, 4),
                                        stride=(1, 2),
                                        padding=(1, 1))

        self.conv3 = nn.ConvTranspose2d(in_channels=self.num_channels*8,
                                        out_channels=self.num_channels*8,
                                        kernel_size=(100, 3),
                                        stride=(1, 1),
                                        padding=(2, 1))

        self.conv2 = nn.ConvTranspose2d(in_channels=self.num_channels*8,
                                        out_channels=self.num_channels*16,
                                        kernel_size=(3, 3),
                                        stride=(1, 1),
                                        padding=(1, 1))

        self.conv1 = nn.ConvTranspose2d(in_channels=self.num_channels*16,
                                        out_channels=1,
                                        kernel_size=(5, 5),
                                        stride=(1, 1),
                                        padding=(2, 2))

    def forward(self, x):
        #print('Input size to decoder:', x.shape)

        x = self.fc(x)
        #print('Output size of fc decoder:', x.shape)

        x = x.view(x.size(0), self.num_channels, 50, -1)  # unflatten batch of feature vectors to a batch of multi-channel feature maps
        #print('Output size after unflatten:', x.shape)

        x = F.relu(self.conv5(x))
        #print('Output size after conv5:', x.shape)

        x = F.relu(self.conv4(x))
        #print('Output size after conv4:', x.shape)

        x = F.relu(self.conv3(x))
        #print('Output size after conv3:', x.shape)

        x = F.relu(self.conv2(x))
        #print('Output size after conv2:', x.shape)

        x = torch.sigmoid(self.conv1(x))  # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        #print('Output size after conv1 with Sigmoid:', x.shape)
        return x


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
"""
#-----------------------------------------------------------------
class Encoder2(nn.Module):
    def __init__(self,
                 num_channels=configs.ParamsConfig.NUM_CHANNELS,
                 latent_dims=configs.ParamsConfig.LATENT_DIMS):
        super(Encoder2, self).__init__()

        self.num_channels = num_channels
        self.latent_dims = latent_dims

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=self.num_channels*16,
                               kernel_size=(5, 5),
                               stride=(1, 1),
                               padding=(2, 2))

        self.conv2 = nn.Conv2d(in_channels=self.num_channels*16,
                               out_channels=self.num_channels*8,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1))

        self.conv3 = nn.Conv2d(in_channels=self.num_channels*8,
                               out_channels=self.num_channels*8,
                               kernel_size=(100, 3),
                               stride=(1, 1),
                               padding=(2, 1))

        self.conv4 = nn.Conv2d(in_channels=self.num_channels*8,
                               out_channels=self.num_channels*8,
                               kernel_size=(3, 55),
                               stride=(1, 1),
                               padding=(1, 2))

        self.conv5 = nn.Conv2d(in_channels=self.num_channels*8,
                               out_channels=self.num_channels,
                               kernel_size=(50, 50),
                               stride=(1, 1),
                               padding=(2, 2))

        self.fc_mu = nn.Linear(in_features=self.num_channels*50*5,
                               out_features=self.latent_dims)

        self.fc_logvar = nn.Linear(in_features=self.num_channels*50*5,
                                   out_features=self.latent_dims)

    def forward(self, x):
        #print('Input size to Conv1 encoder:', x.shape)

        x_conv1 = F.relu(self.conv1(x))
        #print('Output size Conv1 encoder', x_conv1.shape)

        x_conv2 = F.relu(self.conv2(x_conv1))
        #print('Output size Conv2 encoder', x_conv2.shape)

        x_conv3 = F.relu(self.conv3(x_conv2))
        #print('Output size Conv3 encoder', x_conv3.shape)

        x_conv4 = F.relu(self.conv4(x_conv3))
        #print('Output size Conv4 encoder', x_conv4.shape)

        x_conv5 = F.relu(self.conv5(x_conv4))
        #print('Output size Conv5 encoder', x_conv5.shape)

        x_flatten = x_conv5.view(x_conv5.size(0), -1)  # flatten batch of feature maps
        #print('Output size after flatten', x_flatten.shape)

        x_mu = self.fc_mu(x_flatten)
        #print('Output size of mu after fc', x_mu.shape)
        x_logvar = self.fc_logvar(x_flatten)
        #print('Output size of logvar after fc', x_logvar.shape)
        return x_mu, x_logvar


class Decoder2(nn.Module):
    global input_shape

    def __init__(self,
                 num_channels=configs.ParamsConfig.NUM_CHANNELS,
                 latent_dims=configs.ParamsConfig.LATENT_DIMS):
        super(Decoder2, self).__init__()

        self.num_channels = num_channels
        self.latent_dims = latent_dims

        self.fc = nn.Linear(in_features=self.latent_dims,
                            out_features=self.num_channels*50*5)

        self.conv5 = nn.ConvTranspose2d(in_channels=self.num_channels,
                                        out_channels=self.num_channels*8,
                                        kernel_size=(50, 50),
                                        stride=(1, 1),
                                        padding=(2, 2))

        self.conv4 = nn.ConvTranspose2d(in_channels=self.num_channels*8,
                                        out_channels=self.num_channels*8,
                                        kernel_size=(3, 55),
                                        stride=(1, 1),
                                        padding=(1, 2))

        self.conv3 = nn.ConvTranspose2d(in_channels=self.num_channels*8,
                                        out_channels=self.num_channels*8,
                                        kernel_size=(100, 3),
                                        stride=(1, 1),
                                        padding=(2, 1))

        self.conv2 = nn.ConvTranspose2d(in_channels=self.num_channels*8,
                                        out_channels=self.num_channels*16,
                                        kernel_size=(3, 3),
                                        stride=(1, 1),
                                        padding=(1, 1))

        self.conv1 = nn.ConvTranspose2d(in_channels=self.num_channels*16,
                                        out_channels=1,
                                        kernel_size=(5, 5),
                                        stride=(1, 1),
                                        padding=(2, 2))

    def forward(self, x):
        #print('Input size to decoder:', x.shape)

        x = self.fc(x)
        #print('Output size of fc decoder:', x.shape)

        x = x.view(x.size(0), self.num_channels, 50, -1)  # unflatten batch of feature vectors to a batch of multi-channel feature maps
        #print('Output size after unflatten:', x.shape)

        x = F.relu(self.conv5(x))
        #print('Output size after conv5:', x.shape)

        x = F.relu(self.conv4(x))
        #print('Output size after conv4:', x.shape)

        x = F.relu(self.conv3(x))
        #print('Output size after conv3:', x.shape)

        x = F.relu(self.conv2(x))
        #print('Output size after conv2:', x.shape)

        x = torch.sigmoid(self.conv1(x))  # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        #print('Output size after conv1 with Sigmoid:', x.shape)
        return x


class VAE2(nn.Module):
    def __init__(self):
        super(VAE2, self).__init__()
        self.encoder = Encoder2()
        self.decoder = Decoder2()

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
"""

class SmallEncoder(nn.Module):
    def __init__(self,
                 num_channels=configs.ParamsConfig.NUM_CHANNELS,
                 latent_dims=configs.ParamsConfig.LATENT_DIMS):
        super(SmallEncoder, self).__init__()

        self.num_channels = num_channels
        self.latent_dims = latent_dims

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=self.num_channels*16,
                               kernel_size=(7, 7),
                               stride=(1, 1),
                               padding=(3, 3))

        self.conv2 = nn.Conv2d(in_channels=self.num_channels*16,
                               out_channels=self.num_channels*8,
                               kernel_size=(5, 5),
                               stride=(1, 1),
                               padding=(2, 2))

        self.conv3 = nn.Conv2d(in_channels=self.num_channels*8,
                               out_channels=self.num_channels*8,
                               kernel_size=(5, 5),
                               stride=(2, 1),
                               padding=(2, 2))

        self.conv4 = nn.Conv2d(in_channels=self.num_channels*8,
                               out_channels=self.num_channels*8,
                               kernel_size=(5, 5),
                               stride=(1, 2),
                               padding=(2, 2))

        self.conv5 = nn.Conv2d(in_channels=self.num_channels*8,
                               out_channels=self.num_channels,
                               kernel_size=(46, 9),
                               stride=(1, 4),
                               padding=(0, 0),
                               dilation=(1, 4))

        self.fc_mu = nn.Linear(in_features=self.num_channels*50*5,
                               out_features=self.latent_dims)

        self.fc_logvar = nn.Linear(in_features=self.num_channels*50*5,
                                   out_features=self.latent_dims)

    def forward(self, x):
        #print('Input size to Conv1 encoder:', x.shape)

        x_conv1 = F.relu(self.conv1(x))
        #print('Output size Conv1 encoder', x_conv1.shape)

        x_conv2 = F.relu(self.conv2(x_conv1))
        #print('Output size Conv2 encoder', x_conv2.shape)

        x_conv3 = F.relu(self.conv3(x_conv2))
        #print('Output size Conv3 encoder', x_conv3.shape)

        x_conv4 = F.relu(self.conv4(x_conv3))
        #print('Output size Conv4 encoder', x_conv4.shape)

        x_conv5 = F.relu(self.conv5(x_conv4))
        #print('Output size Conv5 encoder', x_conv5.shape)

        x_flatten = x_conv5.view(x_conv5.size(0), -1)  # flatten batch of feature maps
        #print('Output size after flatten', x_flatten.shape)

        x_mu = self.fc_mu(x_flatten)
        #print('Output size of mu after fc', x_mu.shape)
        x_logvar = self.fc_logvar(x_flatten)
        #print('Output size of logvar after fc', x_logvar.shape)
        return x_mu, x_logvar


class SmallDecoder(nn.Module):
    global input_shape

    def __init__(self,
                 num_channels=configs.ParamsConfig.NUM_CHANNELS,
                 latent_dims=configs.ParamsConfig.LATENT_DIMS):
        super(SmallDecoder, self).__init__()

        self.num_channels = num_channels
        self.latent_dims = latent_dims

        self.fc = nn.Linear(in_features=self.latent_dims,
                            out_features=self.num_channels*50*5)

        self.conv5 = nn.ConvTranspose2d(in_channels=self.num_channels,
                                        out_channels=self.num_channels*8,
                                        kernel_size=(46, 10),
                                        stride=(1, 4),
                                        padding=(0, 1),
                                        dilation=(1, 4))

        self.conv4 = nn.ConvTranspose2d(in_channels=self.num_channels*8,
                                        out_channels=self.num_channels*8,
                                        kernel_size=(5, 5),
                                        stride=(1, 2),
                                        padding=(2, 2))

        self.conv3 = nn.ConvTranspose2d(in_channels=self.num_channels*8,
                                        out_channels=self.num_channels*8,
                                        kernel_size=(5, 5),
                                        stride=(2, 1),
                                        padding=(2, 2))

        self.conv2 = nn.ConvTranspose2d(in_channels=self.num_channels*8,
                                        out_channels=self.num_channels*16,
                                        kernel_size=(6, 6),
                                        stride=(1, 1),
                                        padding=(2, 3))

        self.conv1 = nn.ConvTranspose2d(in_channels=self.num_channels*16,
                                        out_channels=1,
                                        kernel_size=(5, 5),
                                        stride=(1, 1),
                                        padding=(2, 2))

    def forward(self, x):
        #print('Input size to decoder:', x.shape)

        x = self.fc(x)
        #print('Output size of fc decoder:', x.shape)

        x = x.view(x.size(0), self.num_channels, 50, -1)  # unflatten batch of feature vectors to a batch of multi-channel feature maps
        #print('Output size after unflatten:', x.shape)

        x = F.relu(self.conv5(x))
        #print('Output size after conv5:', x.shape)

        x = F.relu(self.conv4(x))
        #print('Output size after conv4:', x.shape)

        x = F.relu(self.conv3(x))
        #print('Output size after conv3:', x.shape)

        x = F.relu(self.conv2(x))
        #print('Output size after conv2:', x.shape)

        x = torch.sigmoid(self.conv1(x))  # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        #print('Output size after conv1 with Sigmoid:', x.shape)
        return x


class SmallVAE(nn.Module):
    def __init__(self):
        super(SmallVAE, self).__init__()
        self.encoder = SmallEncoder()
        self.decoder = SmallDecoder()

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

#-----------------------------------------------------------------
class VerySmallEncoder(nn.Module):
    def __init__(self,
                 num_channels=configs.ParamsConfig.NUM_CHANNELS,
                 latent_dims=configs.ParamsConfig.LATENT_DIMS):
        super(VerySmallEncoder, self).__init__()

        self.num_channels = num_channels
        self.latent_dims = latent_dims

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=self.num_channels*16,
                               kernel_size=(7, 7),
                               stride=(1, 1),
                               padding=(3, 3))

        self.conv3 = nn.Conv2d(in_channels=self.num_channels*16,
                               out_channels=self.num_channels*8,
                               kernel_size=(5, 5),
                               stride=(2, 1),
                               padding=(2, 2))

        self.conv4 = nn.Conv2d(in_channels=self.num_channels*8,
                               out_channels=self.num_channels*8,
                               kernel_size=(5, 5),
                               stride=(1, 2),
                               padding=(2, 2))

        self.conv5 = nn.Conv2d(in_channels=self.num_channels*8,
                               out_channels=self.num_channels,
                               kernel_size=(46, 9),
                               stride=(1, 4),
                               padding=(0, 0),
                               dilation=(1, 4))

        self.fc_mu = nn.Linear(in_features=self.num_channels*50*5,
                               out_features=self.latent_dims)

        self.fc_logvar = nn.Linear(in_features=self.num_channels*50*5,
                                   out_features=self.latent_dims)

    def forward(self, x):
        #print('Input size to Conv1 encoder:', x.shape)

        x_conv1 = F.relu(self.conv1(x))
        #print('Output size Conv1 encoder', x_conv1.shape)

        x_conv3 = F.relu(self.conv3(x_conv1))
        #print('Output size Conv3 encoder', x_conv3.shape)

        x_conv4 = F.relu(self.conv4(x_conv3))
        #print('Output size Conv4 encoder', x_conv4.shape)

        x_conv5 = F.relu(self.conv5(x_conv4))
        #print('Output size Conv5 encoder', x_conv5.shape)

        x_flatten = x_conv5.view(x_conv5.size(0), -1)  # flatten batch of feature maps
        #print('Output size after flatten', x_flatten.shape)

        x_mu = self.fc_mu(x_flatten)
        #print('Output size of mu after fc', x_mu.shape)
        x_logvar = self.fc_logvar(x_flatten)
        #print('Output size of logvar after fc', x_logvar.shape)
        return x_mu, x_logvar


class VerySmallDecoder(nn.Module):
    global input_shape

    def __init__(self,
                 num_channels=configs.ParamsConfig.NUM_CHANNELS,
                 latent_dims=configs.ParamsConfig.LATENT_DIMS):
        super(VerySmallDecoder, self).__init__()

        self.num_channels = num_channels
        self.latent_dims = latent_dims

        self.fc = nn.Linear(in_features=self.latent_dims,
                            out_features=self.num_channels*50*5)

        self.conv5 = nn.ConvTranspose2d(in_channels=self.num_channels,
                                        out_channels=self.num_channels*8,
                                        kernel_size=(46, 10),
                                        stride=(1, 4),
                                        padding=(0, 1),
                                        dilation=(1, 4))

        self.conv4 = nn.ConvTranspose2d(in_channels=self.num_channels*8,
                                        out_channels=self.num_channels*8,
                                        kernel_size=(5, 5),
                                        stride=(1, 2),
                                        padding=(2, 2))

        self.conv3 = nn.ConvTranspose2d(in_channels=self.num_channels*8,
                                        out_channels=self.num_channels*8,
                                        kernel_size=(5, 5),
                                        stride=(2, 1),
                                        padding=(2, 2))

        self.conv2 = nn.ConvTranspose2d(in_channels=self.num_channels*8,
                                        out_channels=1,
                                        kernel_size=(6, 6),
                                        stride=(1, 1),
                                        padding=(2, 3))


    def forward(self, x):
        #print('Input size to decoder:', x.shape)

        x = self.fc(x)
        #print('Output size of fc decoder:', x.shape)

        x = x.view(x.size(0), self.num_channels, 50, -1)  # unflatten batch of feature vectors to a batch of multi-channel feature maps
        #print('Output size after unflatten:', x.shape)

        x = F.relu(self.conv5(x))
        #print('Output size after conv5:', x.shape)

        x = F.relu(self.conv4(x))
        #print('Output size after conv4:', x.shape)

        x = F.relu(self.conv3(x))
        #print('Output size after conv3:', x.shape)

        x = torch.sigmoid(self.conv2(x))  # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        #print('Output size after conv1 with Sigmoid:', x.shape)
        return x


class VerySmallVAE(nn.Module):
    def __init__(self):
        super(VerySmallVAE, self).__init__()
        self.encoder = VerySmallEncoder()
        self.decoder = VerySmallDecoder()

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu


#---------------------------------------------------------------------------------
class VerySmallTimbreEncoder(nn.Module):
    def __init__(self,
                 num_channels=configs.ParamsConfig.NUM_CHANNELS,
                 latent_dims=configs.ParamsConfig.LATENT_DIMS):
        super(VerySmallTimbreEncoder, self).__init__()

        self.num_channels = num_channels
        self.latent_dims = latent_dims

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=self.num_channels*16,
                               kernel_size=(7, 7),
                               stride=(1, 1),
                               padding=(3, 3))

        self.conv3 = nn.Conv2d(in_channels=self.num_channels*16,
                               out_channels=self.num_channels*8,
                               kernel_size=(5, 5),
                               stride=(2, 1),
                               padding=(2, 2))

        self.conv4 = nn.Conv2d(in_channels=self.num_channels*8,
                               out_channels=self.num_channels*8,
                               kernel_size=(5, 5),
                               stride=(1, 2),
                               padding=(2, 2))

        self.conv5 = nn.Conv2d(in_channels=self.num_channels*8,
                               out_channels=self.num_channels,
                               kernel_size=(46, 9),
                               stride=(1, 4),
                               padding=(0, 0),
                               dilation=(1, 3))

        self.fc_mu = nn.Linear(in_features=self.num_channels*50*10,
                               out_features=self.latent_dims)

        self.fc_logvar = nn.Linear(in_features=self.num_channels*50*10,
                                   out_features=self.latent_dims)

    def forward(self, x):
        #print('Input size to Conv1 encoder:', x.shape)

        x_conv1 = F.relu(self.conv1(x))
        #print('Output size Conv1 encoder', x_conv1.shape)

        x_conv3 = F.relu(self.conv3(x_conv1))
        #print('Output size Conv3 encoder', x_conv3.shape)

        x_conv4 = F.relu(self.conv4(x_conv3))
        #print('Output size Conv4 encoder', x_conv4.shape)

        x_conv5 = F.relu(self.conv5(x_conv4))
        #print('Output size Conv5 encoder', x_conv5.shape)

        x_flatten = x_conv5.view(x_conv5.size(0), -1)  # flatten batch of feature maps
        #print('Output size after flatten', x_flatten.shape)

        x_mu = self.fc_mu(x_flatten)
        #print('Output size of mu after fc', x_mu.shape)
        x_logvar = self.fc_logvar(x_flatten)
        #print('Output size of logvar after fc', x_logvar.shape)
        return x_mu, x_logvar


class VerySmallTimbreDecoder(nn.Module):
    global input_shape

    def __init__(self,
                 num_channels=configs.ParamsConfig.NUM_CHANNELS,
                 latent_dims=configs.ParamsConfig.LATENT_DIMS):
        super(VerySmallTimbreDecoder, self).__init__()

        self.num_channels = num_channels
        self.latent_dims = latent_dims

        self.fc = nn.Linear(in_features=self.latent_dims,
                            out_features=self.num_channels*50*10)

        self.conv5 = nn.ConvTranspose2d(in_channels=self.num_channels,
                                        out_channels=self.num_channels*8,
                                        kernel_size=(46, 10),
                                        stride=(1, 4),
                                        padding=(0, 0),
                                        dilation=(1, 3))

        self.conv4 = nn.ConvTranspose2d(in_channels=self.num_channels*8,
                                        out_channels=self.num_channels*8,
                                        kernel_size=(5, 5),
                                        stride=(1, 2),
                                        padding=(2, 2))

        self.conv3 = nn.ConvTranspose2d(in_channels=self.num_channels*8,
                                        out_channels=self.num_channels*8,
                                        kernel_size=(5, 5),
                                        stride=(2, 1),
                                        padding=(2, 2))

        self.conv2 = nn.ConvTranspose2d(in_channels=self.num_channels*8,
                                        out_channels=1,
                                        kernel_size=(6, 6),
                                        stride=(1, 1),
                                        padding=(2, 2))


    def forward(self, x):
        #print('Input size to decoder:', x.shape)

        x = self.fc(x)
        #print('Output size of fc decoder:', x.shape)

        x = x.view(x.size(0), self.num_channels, 50, -1)  # unflatten batch of feature vectors to a batch of multi-channel feature maps
        #print('Output size after unflatten:', x.shape)

        x = F.relu(self.conv5(x))
        #print('Output size after conv5:', x.shape)

        x = F.relu(self.conv4(x))
        #print('Output size after conv4:', x.shape)

        x = F.relu(self.conv3(x))
        #print('Output size after conv3:', x.shape)

        x = torch.sigmoid(self.conv2(x))  # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        #print('Output size after conv1 with Sigmoid:', x.shape)
        return x


class VerySmallTimbreVAE(nn.Module):
    def __init__(self):
        super(VerySmallTimbreVAE, self).__init__()
        self.encoder = VerySmallTimbreEncoder()
        self.decoder = VerySmallTimbreDecoder()

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

#---------------------------------------------------------------------
class TimbreEncoder(nn.Module):
    def __init__(self,
                 input='cqt',
                 gru_units=configs.ParamsConfig.GRU_UNITS,
                 gru_layers=configs.ParamsConfig.GRU_LAYERS,
                 n_frames=configs.InputsConfig.N_FRAMES,
                 latent_dims=configs.ParamsConfig.LATENT_DIMS,
                 bidirectional=True
                 ):
        super(TimbreEncoder, self).__init__()

        if input == 'cqt':
            self.freq_dim = configs.InputsConfig.FREQ_BINS
        elif input == 'mel':
            self.freq_dim = configs.InputsConfig.MELS

        self.gru_units = gru_units
        self.gru_layers = gru_layers
        self.n_frames = n_frames
        self.latent_dims = latent_dims

        self.gru = nn.GRU(
                        input_size=self.freq_dim,
                        hidden_size=self.gru_units,
                        num_layers=self.gru_layers,
                        batch_first=True,
                        bidirectional=bidirectional
                        )

        self.fc = nn.Linear(in_features=2*self.gru_units if bidirectional else self.gru_units,
                                    out_features=1)

        self.fc_mu = nn.Linear(in_features=self.n_frames,
                               out_features=self.latent_dims)

        self.fc_logvar = nn.Linear(in_features=self.n_frames,
                                   out_features=self.latent_dims)

    def forward(self, x):

        #print('Input size to encoder:', x.shape)
        x = x.permute(0, 1, 3, 2)
        x = x.view(-1, self.n_frames, self.freq_dim)
        #print('Input reshaped shape:', x.shape)

        #print('Input size to encoder:', x.shape)

        x_gru, _ = self.gru(x)
        x = F.relu(x_gru)
        #print('Output size gru encoder', x_gru.shape)

        x = F.relu(self.fc(x))
        #print('Output size after fc', x.shape)

        x_flatten = x.contiguous().view(-1, self.n_frames)  # flatten batch of feature maps
        #print('Output size after flatten', x_flatten.shape)

        x_mu = F.relu(self.fc_mu(x_flatten))
        #print('Output size of mu after fc', x_mu.shape)
        x_logvar = F.relu(self.fc_logvar(x_flatten))
        #print('Output size of logvar after fc', x_logvar.shape)
        return x_mu, x_logvar


class TimbreDecoder(nn.Module):

    def __init__(self,
                 input='cqt',
                 gru_units=configs.ParamsConfig.GRU_UNITS,
                 gru_layers=configs.ParamsConfig.GRU_LAYERS,
                 n_frames=configs.InputsConfig.N_FRAMES,
                 latent_dims=configs.ParamsConfig.LATENT_DIMS,
                 bidirectional=True
                 ):
        super(TimbreDecoder, self).__init__()

        if input == 'cqt':
            self.freq_dim = configs.InputsConfig.FREQ_BINS
        elif input == 'mel':
            self.freq_dim = configs.InputsConfig.MELS

        self.gru_units = gru_units
        self.gru_layers = gru_layers
        self.n_frames = n_frames
        self.latent_dims = latent_dims

        self.fc_mu = nn.Linear(in_features=self.latent_dims,
                               out_features=self.n_frames)

        self.gru = nn.GRU(
                        input_size=1,
                        hidden_size=self.gru_units,
                        num_layers=self.gru_layers,
                        batch_first=True,
                        bidirectional=bidirectional
                        )

        self.fc = nn.Linear(in_features=2*self.gru_units if bidirectional else self.gru_units,
                                    out_features=self.freq_dim)

    def forward(self, x):
        #print('Input size to decoder:', x.shape)

        x = F.relu(self.fc_mu(x))
        #print('Output size of fc decoder:', x.shape)

        x = x.view(-1, self.n_frames, 1)  # unflatten batch of feature vectors to a batch of multi-channel feature maps
        #print('Output size after unflatten:', x.shape)

        x, _ = self.gru(x)
        x = F.relu(x)
        #print('Output size after gru:', x.shape)

        x = self.fc(x)
        #print('Output size of fc decoder:', x.shape)

        x = torch.sigmoid(x)  # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        #print('Output size after Sigmoid:', x.shape)

        #x = x.permute(0, 2, 1)
        #print('Output size of decoder:', x.shape)

        x = x.view(-1, 1, self.freq_dim, self.n_frames)
        #x = x.permute(0, 1, 3, 2)
        #print('Output reshaped shape:', x.shape)

        return x


class TimbreVAE(nn.Module):
    def __init__(self, input='cqt'):
        super(TimbreVAE, self).__init__()
        self.encoder = TimbreEncoder(input)
        self.decoder = TimbreDecoder(input)

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu


#---------------------------------------------------------------------
class TimeTimbreEncoder(nn.Module):
    def __init__(self,
                 input='cqt',
                 gru_units=configs.ParamsConfig.GRU_UNITS,
                 gru_layers=configs.ParamsConfig.GRU_LAYERS,
                 n_frames=configs.InputsConfig.N_FRAMES,
                 latent_dims=configs.ParamsConfig.LATENT_DIMS,
                 bidirectional=True
                 ):
        super(TimeTimbreEncoder, self).__init__()

        if input == 'cqt':
            self.freq_dim = configs.InputsConfig.FREQ_BINS
        elif input == 'mel':
            self.freq_dim = configs.InputsConfig.MELS

        self.gru_units = gru_units
        self.gru_layers = gru_layers
        self.n_frames = n_frames
        self.latent_dims = latent_dims

        self.norm = nn.InstanceNorm1d(self.freq_dim, affine=False)

        self.gru = nn.GRU(
                        input_size=self.n_frames,
                        hidden_size=self.gru_units,
                        num_layers=self.gru_layers,
                        batch_first=True,
                        bidirectional=bidirectional
                        )
        self.fc = nn.Linear(in_features=2*self.gru_units if bidirectional else self.gru_units,
                                    out_features=1)

        self.fc_mu = nn.Linear(in_features=self.freq_dim,
                               out_features=self.latent_dims)

        self.fc_logvar = nn.Linear(in_features=self.freq_dim,
                                   out_features=self.latent_dims)

    def forward(self, x):

        #print('Input size to encoder:', x.shape)

        x = x.view(-1, self.freq_dim, self.n_frames)
        #print('Input reshaped shape:', x.shape)

        #x = x.permute(0, 2, 1)
        #print('Input size to encoder:', x.shape)

        x_gru, _ = self.gru(x)
        x = F.relu(x_gru)
        #print('Output size gru encoder', x_gru.shape)

        x = F.relu(self.fc(x))
        #print('Output size after fc', x.shape)

        x_flatten = x.contiguous().view(-1, self.freq_dim)  # flatten batch of feature maps
        #print('Output size after flatten', x_flatten.shape)

        x_mu = F.relu(self.fc_mu(x_flatten))
        #print('Output size of mu after fc', x_mu.shape)
        x_logvar = F.relu(self.fc_logvar(x_flatten))
        #print('Output size of logvar after fc', x_logvar.shape)
        return x_mu, x_logvar


class TimeTimbreDecoder(nn.Module):

    def __init__(self,
                 input='cqt',
                 gru_units=configs.ParamsConfig.GRU_UNITS,
                 gru_layers=configs.ParamsConfig.GRU_LAYERS,
                 n_frames=configs.InputsConfig.N_FRAMES,
                 latent_dims=configs.ParamsConfig.LATENT_DIMS,
                 bidirectional=True
                 ):
        super(TimeTimbreDecoder, self).__init__()

        if input == 'cqt':
            self.freq_dim = configs.InputsConfig.FREQ_BINS
        elif input == 'mel':
            self.freq_dim = configs.InputsConfig.MELS

        self.gru_units = gru_units
        self.gru_layers = gru_layers
        self.n_frames = n_frames
        self.latent_dims = latent_dims

        self.fc_mu = nn.Linear(in_features=self.latent_dims,
                               out_features=self.freq_dim)

        self.gru = nn.GRU(
                        input_size=1,
                        hidden_size=self.gru_units,
                        num_layers=self.gru_layers,
                        batch_first=True,
                        bidirectional=bidirectional
                        )

        self.fc = nn.Linear(in_features=2*self.gru_units if bidirectional else self.gru_units,
                                    out_features=self.n_frames)

    def forward(self, x):
        #print('Input size to decoder:', x.shape)

        x = F.relu(self.fc_mu(x))
        #print('Output size of fc decoder:', x.shape)

        x = x.view(-1, self.freq_dim, 1)  # unflatten batch of feature vectors to a batch of multi-channel feature maps
        #print('Output size after unflatten:', x.shape)

        x, _ = self.gru(x)
        x = F.relu(x)
        #print('Output size after gru:', x.shape)

        x = self.fc(x)
        #print('Output size of fc decoder:', x.shape)

        x = torch.sigmoid(x)  # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        #print('Output size after Sigmoid:', x.shape)

        #x = x.permute(0, 2, 1)
        #print('Output size of decoder:', x.shape)

        x = x.view(-1, 1, self.freq_dim, self.n_frames)
        #print('Output reshaped shape:', x.shape)

        return x


class TimeTimbreVAE(nn.Module):
    def __init__(self, input='cqt'):
        super(TimeTimbreVAE, self).__init__()
        self.encoder = TimeTimbreEncoder(input)
        self.decoder = TimeTimbreDecoder(input)

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    
#---------------------------------------------------------------------------------
class Conv2DFreqTimbreEncoder(nn.Module):
    def __init__(self,
                 input='cqt',
                 num_channels=configs.ParamsConfig.NUM_CHANNELS,
                 latent_dims=configs.ParamsConfig.LATENT_DIMS):
        super(Conv2DFreqTimbreEncoder, self).__init__()

        self.input = input
        self.num_channels = num_channels
        self.latent_dims = latent_dims

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=self.num_channels*16,
                               kernel_size=(7, 7),
                               stride=(1, 1),
                               padding=(3, 3))

        self.conv3 = nn.Conv2d(in_channels=self.num_channels*16,
                               out_channels=self.num_channels*8,
                               kernel_size=(5, 5),
                               stride=(2, 1),
                               padding=(2, 2))

        self.conv4 = nn.Conv2d(in_channels=self.num_channels*8,
                               out_channels=self.num_channels*8,
                               kernel_size=(5, 5),
                               stride=(1, 2),
                               padding=(2, 2))

        self.conv5 = nn.Conv2d(in_channels=self.num_channels*8,
                               out_channels=self.num_channels,
                               kernel_size=(46, 9),
                               stride=(1, 4),
                               padding=(0, 0),
                               dilation=(1, 3))

        if self.input == 'cqt':
            dim = 50

        elif self.input == 'mel':
            dim = 19

        self.fc_mu = nn.Linear(in_features=self.num_channels * dim * 10,
                               out_features=self.latent_dims)

        self.fc_logvar = nn.Linear(in_features=self.num_channels * dim * 10,
                                   out_features=self.latent_dims)



    def forward(self, x):
        #print('Input size to Conv1 encoder:', x.shape)

        x_conv1 = F.relu(self.conv1(x))
        #print('Output size Conv1 encoder', x_conv1.shape)

        x_conv3 = F.relu(self.conv3(x_conv1))
        #print('Output size Conv3 encoder', x_conv3.shape)

        x_conv4 = F.relu(self.conv4(x_conv3))
        #print('Output size Conv4 encoder', x_conv4.shape)

        x_conv5 = F.relu(self.conv5(x_conv4))
        #print('Output size Conv5 encoder', x_conv5.shape)

        x_flatten = x_conv5.view(x_conv5.size(0), -1)  # flatten batch of feature maps
        #print('Output size after flatten', x_flatten.shape)

        x_mu = self.fc_mu(x_flatten)
        #print('Output size of mu after fc', x_mu.shape)
        x_logvar = self.fc_logvar(x_flatten)
        #print('Output size of logvar after fc', x_logvar.shape)
        return x_mu, x_logvar


class Conv2DFreqTimbreDecoder(nn.Module):
    global input_shape

    def __init__(self,
                 input = 'cqt',
                 num_channels=configs.ParamsConfig.NUM_CHANNELS,
                 latent_dims=configs.ParamsConfig.LATENT_DIMS):
        super(Conv2DFreqTimbreDecoder, self).__init__()

        self.input = input
        self.num_channels = num_channels
        self.latent_dims = latent_dims

        if self.input == 'cqt':
            self.dim = 50

        elif self.input == 'mel':
            self.dim = 19

        self.fc = nn.Linear(in_features=self.latent_dims,
                            out_features=self.num_channels*self.dim*10)

        self.conv5 = nn.ConvTranspose2d(in_channels=self.num_channels,
                                        out_channels=self.num_channels*8,
                                        kernel_size=(46, 10),
                                        stride=(1, 4),
                                        padding=(0, 0),
                                        dilation=(1, 3))

        self.conv4 = nn.ConvTranspose2d(in_channels=self.num_channels*8,
                                        out_channels=self.num_channels*8,
                                        kernel_size=(5, 5),
                                        stride=(1, 2),
                                        padding=(2, 2))

        self.conv3 = nn.ConvTranspose2d(in_channels=self.num_channels*8,
                                        out_channels=self.num_channels*8,
                                        kernel_size=(5, 5),
                                        stride=(2, 1),
                                        padding=(2, 2))

        self.conv2 = nn.ConvTranspose2d(in_channels=self.num_channels*8,
                                        out_channels=1,
                                        kernel_size=(6, 6),
                                        stride=(1, 1),
                                        padding=(2, 2))


    def forward(self, x):
        #print('Input size to decoder:', x.shape)

        x = self.fc(x)
        #print('Output size of fc decoder:', x.shape)

        x = x.view(x.size(0), self.num_channels, self.dim, -1)  # unflatten batch of feature vectors to a batch of multi-channel feature maps
        #print('Output size after unflatten:', x.shape)

        x = F.relu(self.conv5(x))
        #print('Output size after conv5:', x.shape)

        x = F.relu(self.conv4(x))
        #print('Output size after conv4:', x.shape)

        x = F.relu(self.conv3(x))
        #print('Output size after conv3:', x.shape)

        x = torch.sigmoid(self.conv2(x))  # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        #print('Output size after conv1 with Sigmoid:', x.shape)
        return x


class Conv2DFreqTimbreVAE(nn.Module):
    def __init__(self, input):
        super(Conv2DFreqTimbreVAE, self).__init__()
        self.encoder = Conv2DFreqTimbreEncoder(input)
        self.decoder = Conv2DFreqTimbreDecoder(input)

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

#-----------------------------------------------
class AttentionTimbreEncoder(nn.Module):
    def __init__(self,
                 input='cqt',
                 embed_dim=configs.InputsConfig.N_FRAMES,
                 num_heads=configs.ParamsConfig.NUM_HEADS,
                 n_frames=configs.InputsConfig.N_FRAMES,
                 latent_dims=configs.ParamsConfig.LATENT_DIMS):
        super(AttentionTimbreEncoder, self).__init__()

        if input == 'cqt':
            self.freq_dim = configs.InputsConfig.FREQ_BINS
        elif input == 'mel':
            self.freq_dim = configs.InputsConfig.MELS

        self.input = input
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.n_frames = n_frames
        self.latent_dims = latent_dims

        self.multihead_attn = nn.MultiheadAttention(self.embed_dim, num_heads=self.num_heads)

        self.fc_mu = nn.Linear(in_features=self.n_frames * self.embed_dim,
                               out_features=self.latent_dims)

        self.fc_logvar = nn.Linear(in_features=self.n_frames * self.embed_dim,
                                   out_features=self.latent_dims)

    def forward(self, x):
        #print('Input size to encoder:', x.shape)

        x_reshaped = x.view(self.n_frames, -1, self.freq_dim)
        #print('Input reshaped:', x_reshaped.shape)

        x_attn, _ = self.multihead_attn(x_reshaped, x_reshaped, x_reshaped)
        #print('Output size attention encoder', x_attn.shape)

        x_flatten = x_attn.view(-1, self.n_frames*self.embed_dim)  # flatten batch of feature maps
        #print('Output size after flatten', x_flatten.shape)

        x_mu = self.fc_mu(x_flatten)
        #print('Output size of mu after fc', x_mu.shape)
        x_logvar = self.fc_logvar(x_flatten)
        #print('Output size of logvar after fc', x_logvar.shape)
        return x_mu, x_logvar


class AttentionTimbreDecoder(nn.Module):

    def __init__(self,
                 input='cqt',
                 embed_dim=configs.InputsConfig.N_FRAMES,
                 num_heads=configs.ParamsConfig.NUM_HEADS,
                 n_frames=configs.InputsConfig.N_FRAMES,
                 latent_dims=configs.ParamsConfig.LATENT_DIMS):
        super(AttentionTimbreDecoder, self).__init__()

        if input == 'cqt':
            self.freq_dim = configs.InputsConfig.FREQ_BINS
        elif input == 'mel':
            self.freq_dim = configs.InputsConfig.MELS

        self.input = input
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.n_frames = n_frames
        self.latent_dims = latent_dims

        self.fc = nn.Linear(in_features=self.latent_dims,
                            out_features=self.n_frames * self.embed_dim)

        self.multihead_attn = nn.MultiheadAttention(self.embed_dim, num_heads=self.num_heads)

        self.fc_out = nn.Linear(in_features=self.embed_dim,
                            out_features=self.freq_dim)


    def forward(self, x):
        #print('Input size to decoder:', x.shape)

        x = F.relu(self.fc(x))
        #print('Queries size of fc decoder:', x.shape)

        x_flatten = x.view(self.n_frames, -1, self.embed_dim)  # unflatten batch of feature vectors to a batch of multi-channel feature maps
        #print('Output size after unflatten:', x_flatten.shape)

        x_attn, _ = self.multihead_attn(x_flatten, x_flatten, x_flatten)
        #print('Output size attention decoder', x_attn.shape)

        x = torch.sigmoid(self.fc_out(x_attn))  # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        #print('Output size after Sigmoid:', x.shape)

        x_reshaped = x.view(-1, 1, self.freq_dim, self.n_frames)
        #print('Output size reshaped:', x_reshaped.shape)

        return x_reshaped


class AttentionTimbreVAE(nn.Module):
    def __init__(self, input):
        super(AttentionTimbreVAE, self).__init__()
        self.encoder = AttentionTimbreEncoder(input)
        self.decoder = AttentionTimbreDecoder(input)

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

#--------------------------------------------
class FreqAttentionTimbreEncoder(nn.Module):
    def __init__(self,
                 input='cqt',
                 num_heads=configs.ParamsConfig.NUM_HEADS,
                 n_frames=configs.InputsConfig.N_FRAMES,
                 latent_dims=configs.ParamsConfig.LATENT_DIMS):
        super(FreqAttentionTimbreEncoder, self).__init__()

        if input == 'cqt':
            self.freq_dim = configs.InputsConfig.FREQ_BINS
        elif input == 'mel':
            self.freq_dim = configs.InputsConfig.MELS

        self.input = input
        self.embed_dim = self.freq_dim
        self.num_heads = num_heads
        self.n_frames = n_frames
        self.latent_dims = latent_dims

        self.multihead_attn = nn.MultiheadAttention(self.embed_dim, num_heads=self.num_heads)

        self.fc_mu = nn.Linear(in_features=self.freq_dim * self.embed_dim,
                               out_features=self.latent_dims)

        self.fc_logvar = nn.Linear(in_features=self.freq_dim * self.embed_dim,
                                   out_features=self.latent_dims)

    def forward(self, x):
        #print('Input size to encoder:', x.shape)

        x = x.permute(2, 0, 1, 3)
        x_reshaped = x.view(self.freq_dim, -1, self.n_frames)
        #print('Input reshaped:', x_reshaped.shape)

        x_attn, _ = self.multihead_attn(x_reshaped, x_reshaped, x_reshaped)
        #print('Output size attention encoder', x_attn.shape)

        x_flatten = x_attn.view(-1, self.freq_dim*self.embed_dim)  # flatten batch of feature maps
        #print('Output size after flatten', x_flatten.shape)

        x_mu = self.fc_mu(x_flatten)
        #print('Output size of mu after fc', x_mu.shape)
        x_logvar = self.fc_logvar(x_flatten)
        #print('Output size of logvar after fc', x_logvar.shape)
        return x_mu, x_logvar


class FreqAttentionTimbreDecoder(nn.Module):

    def __init__(self,
                 input='cqt',
                 embed_dim=configs.InputsConfig.N_FRAMES,
                 num_heads=configs.ParamsConfig.NUM_HEADS,
                 n_frames=configs.InputsConfig.N_FRAMES,
                 latent_dims=configs.ParamsConfig.LATENT_DIMS):
        super(FreqAttentionTimbreDecoder, self).__init__()

        if input == 'cqt':
            self.freq_dim = configs.InputsConfig.FREQ_BINS
        elif input == 'mel':
            self.freq_dim = configs.InputsConfig.MELS

        self.input = input
        self.num_heads = num_heads
        self.embed_dim = self.freq_dim
        self.n_frames = n_frames
        self.latent_dims = latent_dims

        self.fc = nn.Linear(in_features=self.latent_dims,
                            out_features=self.freq_dim * self.embed_dim)

        self.multihead_attn = nn.MultiheadAttention(self.embed_dim, num_heads=self.num_heads)

        self.fc_out = nn.Linear(in_features=self.embed_dim,
                            out_features=self.n_frames)


    def forward(self, x):
        #print('Input size to decoder:', x.shape)

        x = F.relu(self.fc(x))
        #print('Queries size of fc decoder:', x.shape)

        x_flatten = x.view(self.embed_dim, -1, self.n_frames)  # unflatten batch of feature vectors to a batch of multi-channel feature maps
        #print('Output size after unflatten:', x_flatten.shape)

        x_attn, _ = self.multihead_attn(x_flatten, x_flatten, x_flatten)
        #print('Output size attention decoder', x_attn.shape)

        x = torch.sigmoid(self.fc_out(x_attn))  # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        #print('Output size after Sigmoid:', x.shape)

        x_reshaped = x.view(-1, 1, self.n_frames, self.freq_dim)
        #print('Output size reshaped:', x_reshaped.shape)

        return x_reshaped


class FreqAttentionTimbreVAE(nn.Module):
    def __init__(self, input):
        super(FreqAttentionTimbreVAE, self).__init__()
        self.encoder = FreqAttentionTimbreEncoder(input)
        self.decoder = FreqAttentionTimbreDecoder(input)

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu