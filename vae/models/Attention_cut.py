import torch
from torch.nn import functional as F
from torch import nn

# Our modules
from vae import configs

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
        elif input == 'mel' or input == 'mel_cut':
            self.freq_dim = configs.InputsConfig.MELS

        self.input = input
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.n_frames = n_frames
        self.latent_dims = latent_dims

        self.multihead_attn = nn.MultiheadAttention(self.freq_dim, num_heads=self.num_heads)

        self.fc_mu = nn.Linear(in_features=self.n_frames * self.freq_dim,
                               out_features=self.latent_dims)

        self.fc_logvar = nn.Linear(in_features=self.n_frames * self.freq_dim,
                                   out_features=self.latent_dims)

    def forward(self, x):
        #print('Input size to encoder:', x.shape)

        x_reshaped = x.view(self.n_frames, -1, self.freq_dim)
        #print('Input reshaped:', x_reshaped.shape)

        x_attn, _ = self.multihead_attn(x_reshaped, x_reshaped, x_reshaped)
        #print('Output size attention encoder', x_attn.shape)

        x_flatten = x_attn.view(-1, self.n_frames*self.freq_dim)  # flatten batch of feature maps
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
        elif input == 'mel' or input == 'mel_cut':
            self.freq_dim = configs.InputsConfig.MELS

        self.input = input
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.n_frames = n_frames
        self.latent_dims = latent_dims

        self.fc = nn.Linear(in_features=self.latent_dims,
                            out_features=self.n_frames * self.freq_dim)

        self.multihead_attn = nn.MultiheadAttention(self.freq_dim, num_heads=self.num_heads)

        self.fc_out = nn.Linear(in_features=self.freq_dim,
                            out_features=self.freq_dim)


    def forward(self, x):
        #print('Input size to decoder:', x.shape)

        x = F.relu(self.fc(x))
        #print('Queries size of fc decoder:', x.shape)

        x_flatten = x.view(self.n_frames, -1, self.freq_dim)  # unflatten batch of feature vectors to a batch of multi-channel feature maps
        #print('Output size after unflatten:', x_flatten.shape)

        x_attn, _ = self.multihead_attn(x_flatten, x_flatten, x_flatten)
        #print('Output size attention decoder', x_attn.shape)

        x_flatten = x.view(-1, self.n_frames, self.freq_dim)
        # print('Output size after unflatten:', x_flatten.shape)

        x = self.fc_out(x_attn)
        #print('Output fc:', x.shape)

        x_reshaped = x.view(-1, 1, self.freq_dim, self.n_frames)
        #print('Output size reshaped:', x_reshaped.shape)

        return x_reshaped


class AttentionTimbreVAE(nn.Module):
    def __init__(self, input):
        super(AttentionTimbreVAE, self).__init__()
        self.encoder = AttentionTimbreEncoder(input)
        #self.encoder2 = AttentionTimbreEncoder(input)
        self.decoder = AttentionTimbreDecoder(input)

    def forward(self, x):
        encoder1_output_mu, encoder1_output_logvar = self.encoder(x)
        latent1 = self.latent_sample(encoder1_output_mu, encoder1_output_logvar)

        #encoder2_output_mu, encoder2_output_logvar = self.encoder2(input2)
        #latent2 = self.latent_sample(encoder2_output_mu, encoder2_output_logvar)

        #concat both encoders
        #latent_concat = torch.cat((latent1, latent2), 1)
        #print("Size latent:", latent_concat.shape)

        x_recon = self.decoder(latent1) #(latent_concat)
        return x_recon, encoder1_output_mu, encoder1_output_logvar
        #return x_recon, encoder1_output_mu, encoder1_output_logvar, encoder2_output_mu, encoder2_output_logvar

    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

