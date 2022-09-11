#from torchsummary import summary
import torch
from torchsummary import summary

# Our modules
from vae import configs

def import_model(model_name, input):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_name == 'TimeConv2D':
        from vae.models.TimeConv2D import VerySmallVAE
        model = VerySmallVAE().to(device)
    if model_name == 'supervised_timbre':
        from vae.models.TimbreAttention import AttentionTimbreEncoder
        model = AttentionTimbreEncoder(input).to(device)
    if model_name == 'supervised_timbre_fc-freq':
        from vae.models.TimbreAttention import AttentionTimbreFreqFC
        model = AttentionTimbreFreqFC(input).to(device)
    elif  model_name == 'TimeConv2D_cut':
        from vae.models.TimeConv2D_cut import VerySmallVAE
        model = VerySmallVAE().to(device)
    elif model_name == 'Attention_cut':
        from vae.models.Attention_cut import AttentionTimbreVAE
        model = AttentionTimbreVAE('mel_cut').to(device)
    return model


def show_model(model, input_channels=1, input_heigh=128, input_width=128):
    if input == 'mel':
        input_heigh = configs.InputsConfig.MELS
        
    elif input == 'cqt':
        input_heigh = configs.InputsConfig.FREQ_BINS

    input_channels = 1
    input_width = configs.InputsConfig.N_FRAMES
    print(summary(model, input_size=(input_channels, input_heigh, input_width)))
    

def show_total_params(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters: %d' % num_params)
    print(model)

    return