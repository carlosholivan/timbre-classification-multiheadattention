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

        self.fc_1 = nn.Linear(in_features=self.n_frames * self.freq_dim,
                               out_features=20)


    def forward(self, x):
        #print('Input size to encoder:', x.shape)

        x_reshaped = x.view(self.n_frames, -1, self.freq_dim)
        #print('Input reshaped:', x_reshaped.shape)

        x_attn, w = self.multihead_attn(x_reshaped, x_reshaped, x_reshaped)
        #print('Output size attention encoder', x_attn.shape)

        x_flatten = x_attn.view(-1, self.n_frames*self.freq_dim)  # flatten batch of feature maps
        #print('Output size after flatten', x_flatten.shape)

        x = self.fc_1(x_flatten)
        #print('Output size of mu after fc', x_mu.shape)
        
        return x, x_attn, w


class AttentionTimbreFreqFC(nn.Module):
    def __init__(self,
                 input='cqt',
                 n_frames=configs.InputsConfig.N_FRAMES):
        super(AttentionTimbreFreqFC, self).__init__()

        if input == 'cqt':
            self.freq_dim = configs.InputsConfig.FREQ_BINS
        elif input == 'mel' or input == 'mel_cut':
            self.freq_dim = configs.InputsConfig.MELS

        self.input = input
        self.n_frames = n_frames

        self.fc_0 = nn.Linear(in_features=self.freq_dim,
                              out_features=128)

        self.fc_1 = nn.Linear(in_features=self.n_frames * 128,
                               out_features=20)


    def forward(self, x):
        #print('Input size to encoder:', x.shape)

        x_reshaped = x.view(-1, self.n_frames, self.freq_dim)
        #print('Input reshaped:', x_reshaped.shape)

        x = torch.relu(self.fc_0(x_reshaped))

        x = x.view(-1, self.n_frames*128)

        x = self.fc_1(x)
        #print('Output size of mu after fc', x_mu.shape)

        return x