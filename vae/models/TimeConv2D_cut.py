import torch
from torch.nn import functional as F
from torch import nn

# Our modules
from vae import configs

"""
class VerySmallEncoder(nn.Module):
    def __init__(self,
                 num_channels=configs.ParamsConfig.NUM_CHANNELS,
                 latent_dims=configs.ParamsConfig.LATENT_DIMS):
        super(VerySmallEncoder, self).__init__()

        self.num_channels = num_channels
        self.latent_dims = latent_dims

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=self.num_channels*16,
                               kernel_size=(5, 5),
                               stride=(1, 1),
                               padding=(2, 2),
                               dilation=(1, 2))

        self.conv3 = nn.Conv2d(in_channels=self.num_channels*16,
                               out_channels=self.num_channels*8,
                               kernel_size=(5, 5),
                               stride=(2, 1),
                               padding=(2, 2),
                               dilation=(1, 1))

        self.conv4 = nn.Conv2d(in_channels=self.num_channels*8,
                               out_channels=self.num_channels*8,
                               kernel_size=(5, 5),
                               stride=(1, 2),
                               padding=(2, 2))

        self.conv5 = nn.Conv2d(in_channels=self.num_channels*8,
                               out_channels=self.num_channels,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(0, 0),
                               dilation=(2, 2))

        self.fc_mu = nn.Linear(in_features=self.num_channels*60*5,
                               out_features=self.latent_dims)

        self.fc_logvar = nn.Linear(in_features=self.num_channels*60*5,
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
                            out_features=self.num_channels*60*5)

        self.conv5 = nn.ConvTranspose2d(in_channels=self.num_channels,
                                        out_channels=self.num_channels*8,
                                        kernel_size=(3, 3),
                                        stride=(1, 1),
                                        padding=(0, 0),
                                        dilation=(2, 2))

        self.conv4 = nn.ConvTranspose2d(in_channels=self.num_channels*8,
                                        out_channels=self.num_channels*8,
                                        kernel_size=(5, 5),
                                        stride=(1, 1),
                                        padding=(2, 1),
                                        dilation=(1, 2))

        self.conv3 = nn.ConvTranspose2d(in_channels=self.num_channels*8,
                                        out_channels=self.num_channels*8,
                                        kernel_size=(5, 5),
                                        stride=(2, 1),
                                        padding=(2, 2))

        self.conv2 = nn.ConvTranspose2d(in_channels=self.num_channels*8,
                                        out_channels=1,
                                        kernel_size=(6, 4),
                                        stride=(1, 1),
                                        padding=(2, 1),
                                        dilation=(1, 3))


    def forward(self, x):
        #print('Input size to decoder:', x.shape)

        x = self.fc(x)
        #print('Output size of fc decoder:', x.shape)

        x = x.view(-1, self.num_channels, 60, 5)  # unflatten batch of feature vectors to a batch of multi-channel feature maps
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
"""


class VerySmallEncoder(nn.Module):
    def __init__(self,
                 num_channels=configs.ParamsConfig.NUM_CHANNELS,
                 latent_dims=configs.ParamsConfig.LATENT_DIMS):
        super(VerySmallEncoder, self).__init__()

        self.num_channels = num_channels
        self.latent_dims = latent_dims

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=self.num_channels*16,
                               kernel_size=(5, 22),
                               stride=(1, 1),
                               padding=(2, 2),
                               dilation=(1, 1))

        self.conv3 = nn.Conv2d(in_channels=self.num_channels*16,
                               out_channels=self.num_channels*8,
                               kernel_size=(5, 5),
                               stride=(2, 1),
                               padding=(2, 2),
                               dilation=(1, 1))

        self.conv4 = nn.Conv2d(in_channels=self.num_channels*8,
                               out_channels=self.num_channels*8,
                               kernel_size=(5, 5),
                               stride=(2, 1),
                               padding=(2, 2))

        self.conv5 = nn.Conv2d(in_channels=self.num_channels*8,
                               out_channels=self.num_channels,
                               kernel_size=(32, 5),
                               stride=(1, 1),
                               padding=(2, 2))

        self.fc_mu = nn.Linear(in_features=self.num_channels*5*5,
                               out_features=self.latent_dims)

        self.fc_logvar = nn.Linear(in_features=self.num_channels*5*5,
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

        self.fc = nn.Linear(in_features=self.latent_dims*2, #*2 bc we concat 2 encoder both with 128 latent vars so 128*2 = 256
                            out_features=self.num_channels*5*5)

        self.conv5 = nn.ConvTranspose2d(in_channels=self.num_channels,
                                        out_channels=self.num_channels*8,
                                        kernel_size=(32, 5),
                                        stride=(1, 1),
                                        padding=(2, 2),
                                        dilation=(1, 1))

        self.conv4 = nn.ConvTranspose2d(in_channels=self.num_channels*8,
                                        out_channels=self.num_channels*8,
                                        kernel_size=(6, 5),
                                        stride=(2, 2),
                                        padding=(2, 2),
                                        dilation=(1, 1))

        self.conv3 = nn.ConvTranspose2d(in_channels=self.num_channels*8,
                                        out_channels=self.num_channels*8,
                                        kernel_size=(6, 5),
                                        stride=(2, 2),
                                        padding=(2, 2))

        self.conv2 = nn.ConvTranspose2d(in_channels=self.num_channels*8,
                                        out_channels=1,
                                        kernel_size=(5, 6),
                                        stride=(1, 1),
                                        padding=(2, 0),
                                        dilation=(1, 1))


    def forward(self, x):
        #print('Input size to decoder:', x.shape)

        x = self.fc(x)
        #print('Output size of fc decoder:', x.shape)

        x = x.view(-1, self.num_channels, 5, 5)  # unflatten batch of feature vectors to a batch of multi-channel feature maps
        #print('Output size after unflatten:', x.shape)

        x = F.relu(self.conv5(x))
        #print('Output size after conv5:', x.shape)

        x = F.relu(self.conv4(x))
        #print('Output size after conv4:', x.shape)

        x = F.relu(self.conv3(x))
        #print('Output size after conv3:', x.shape)

        x = torch.sigmoid(self.conv2(x))  # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        #x = self.conv2(x)
        #print('Output size after conv1 with Sigmoid:', x.shape)
        return x


class VerySmallVAE(nn.Module):
    def __init__(self):
        super(VerySmallVAE, self).__init__()
        self.encoder1 = VerySmallEncoder()
        self.encoder2 = VerySmallEncoder()
        self.decoder = VerySmallDecoder()

    def forward(self, input1, input2):
        encoder1_output_mu, encoder1_output_logvar = self.encoder1(input1)
        latent1 = self.latent_sample(encoder1_output_mu, encoder1_output_logvar)

        encoder2_output_mu, encoder2_output_logvar = self.encoder2(input2)
        latent2 = self.latent_sample(encoder2_output_mu, encoder2_output_logvar)

        #concat both encoders
        latent_concat = torch.cat((latent1, latent2), 1)
        #print("Size latent:", latent_concat.shape)

        x_recon = self.decoder(latent_concat)
        return x_recon, encoder1_output_mu, encoder1_output_logvar, encoder2_output_mu, encoder2_output_logvar

    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
