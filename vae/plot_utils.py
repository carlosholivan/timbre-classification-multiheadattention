import os
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

import itertools

from IPython.core.display import HTML

import torch
from torchvision import transforms
from torch import nn

import functools
import time

# Our modules
from vae import configs
from vae.data import data_utils


def timeit(func):
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        startTime = time.time()
        func(*args, **kwargs)
        elapsedTime = time.time() - startTime
        print('function [{}] finished in {} s'.format(
            func.__name__, int(elapsedTime)))
    return newfunc


def import_model(model, device):
    if device == 'cpu':
        model = model
    if device == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
        else:
            raise ValueError('cuda is not available')
    return model


def get_latentdims_from_model(model):
    for param_tensor in model.state_dict():
        if param_tensor == 'encoder.fc_mu.bias':
            latent_dims = model.state_dict()[param_tensor].size()[0]
    return latent_dims


def load_pretrained_model(model, trained_epochs):
    model.load_state_dict(torch.load(configs.ParamsConfig.TRAINED_MODELS_PATH + '/saved_model_' + str(trained_epochs) + "epochs.pth"))
    return

def plot_supervised_losses(model, model_name, trained_epochs):

    checkpoint = torch.load(configs.ParamsConfig.TRAINED_MODELS_PATH + '/saved_model_' + str(trained_epochs) + "epochs.pth")
    model.load_state_dict(checkpoint['model'])

    y_val = checkpoint['val loss'][5:-1]
    x_val = np.arange(0, len(y_val))
    y_train = checkpoint['train loss'][5:-1]
    x_train = np.arange(0, len(y_train))

    plt.title('Average Losses')
    plt.plot(x_train, y_train, label='train loss')
    plt.plot(x_val, y_val, label='val loss')
        
    plt.legend()
    plt.show()
    
    return


def plot_losses(model, model_name, trained_epochs, plot='all'):

    checkpoint = torch.load(configs.ParamsConfig.TRAINED_MODELS_PATH + '/saved_model_' + str(trained_epochs) + "epochs.pth")
    model.load_state_dict(checkpoint['model'])

    y_val = checkpoint['val loss'][5:-1]
    x_val = np.arange(0, len(y_val))
    y_train = checkpoint['train loss'][5:-1]
    x_train = np.arange(0, len(y_train))

    if model_name == 'TimeConv2D_cut':
        y_kld_train1 = checkpoint['train kld_1'][5:-1]
        x_kld_train1 = np.arange(0, len(y_kld_train1))
        y_kld_train2 = checkpoint['train kld_2'][5:-1]
        x_kld_train2 = np.arange(0, len(y_kld_train2))
        y_kld_val1 = checkpoint['val kld_1'][5:-1]
        x_kld_val1 = np.arange(0, len(y_kld_val1))
        y_kld_val2 = checkpoint['val kld_2'][5:-1]
        x_kld_val2 = np.arange(0, len(y_kld_val2))
    else:
        y_kld_train = checkpoint['train kld'][5:-1]
        x_kld_train = np.arange(0, len(y_kld_train))
        y_kld_val = checkpoint['val kld'][5:-1]
        x_kld_val = np.arange(0, len(y_kld_val))
    
    y_rec_train = checkpoint['train bce'][5:-1]
    x_rec_train = np.arange(0, len(y_rec_train))
    y_rec_val = checkpoint['val bce'][5:-1]
    x_rec_val = np.arange(0, len(y_rec_val))

    if plot == 'all':
        plt.title('Average Losses')
        plt.plot(x_train, y_train, label='train loss')
        plt.plot(x_val, y_val, label='val loss')
        
    elif plot == 'kld':
        if model_name == 'TimeConv2D_cut':
            plt.title('KLD Losses Encoder 1')
            plt.plot(x_kld_train1, y_kld_train1, label='train KLD loss 1')
            plt.plot(x_kld_val1, y_kld_val1, label='val KLD loss 1')
            plt.legend()
            plt.show()

            plt.title('KLD Losses Encoder 2')
            plt.plot(x_kld_train2, y_kld_train2, label='train KLD loss 2')
            plt.plot(x_kld_val2, y_kld_val2, label='val KLD loss 2')

        else:
            plt.title('KLD Losses')
            plt.plot(x_kld_train, y_kld_train, label='train KLD loss')
            plt.plot(x_kld_val, y_kld_val, label='val KLD loss')
        
    elif plot == 'reconstruction':
        plt.title('Reconstruction Losses')
        plt.plot(x_rec_train, y_rec_train, label='train Rec. loss')
        plt.plot(x_rec_val, y_rec_val, label='val Rec. loss')
        
    plt.legend()
    plt.show()
    
    return


def data_to_pandas_dataframe(model, model_name, trained_epochs, dataset, num_batches=1):

    if num_batches > len(dataset):
        raise ValueError('Selected num_batches > batches in dataset =', len(dataset))
            
    n_points = num_batches * configs.ParamsConfig.BATCH_SIZE
    print('Getting {} means of epoch {}...'.format(n_points, trained_epochs))

    num_batches = num_batches - 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    checkpoint = torch.load(configs.ParamsConfig.TRAINED_MODELS_PATH + '/saved_model_' + str(trained_epochs) + "epochs.pth")
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    df_total = pd.DataFrame()

    for i, (x, y) in enumerate(dataset):
        with torch.no_grad():

            if i > num_batches:
                break

            if model_name == 'TimeConv2D_cut':
                latent_mu1, latent_logvar1 = model.encoder1(x.to(device, dtype=torch.float))
                latent_mu2, latent_logvar2 = model.encoder2(x.to(device, dtype=torch.float))
                z1 = latent_mu1.to('cpu').detach().numpy()
                z1_sigma = latent_logvar1.to('cpu').detach().numpy()
                z2 = latent_mu2.to('cpu').detach().numpy()
                z2_sigma = latent_logvar2.to('cpu').detach().numpy()
                z = z1
            else:
                latent_mu, latent_logvar = model.encoder(x.to(device, dtype=torch.float))
                z = latent_mu.to('cpu').detach().numpy()
                z_sigma = latent_logvar.to('cpu').detach().numpy()

            for batch_idx in range(z.shape[0]):
                if model_name == 'TimeConv2D_cut':
                    latent_dims = z1.shape[1]
                else:
                    latent_dims = z.shape[1]
                file = y[0][batch_idx]
                name_split = y[0][batch_idx].split('_')
                instrument = name_split[0]
                note = name_split[1]
                dynamic = name_split[3]
                technique = name_split[4].split('.')[0]
                for key, value in configs.PlotsConfig.INSTRUMENTS_FAMILIES.items():
                    for instr in value:
                        if instrument == instr:
                            family = key

                if model_name == 'TimeConv2D_cut':
                    df_total = df_total.append({"epochs": trained_epochs,
                                                "means_1" : [z1[:, latent_dim][batch_idx] for latent_dim in range(latent_dims)], #z[:, 0] is the 1st latent dim
                                                "variances_1": [np.exp(z1_sigma[:, latent_dim][batch_idx]) for latent_dim in range(latent_dims)],
                                                "means_2" : [z2[:, latent_dim][batch_idx] for latent_dim in range(latent_dims)], #z[:, 0] is the 1st latent dim
                                                "variances_2": [np.exp(z2_sigma[:, latent_dim][batch_idx]) for latent_dim in range(latent_dims)],
                                                "latent_dims": latent_dims,
                                                "instrument": instrument,
                                                "note": note,
                                                "dynamic": dynamic,
                                                "technique": technique,
                                                "file": file,
                                                "family": family
                                            }, ignore_index=True)
                else:
                    df_total = df_total.append({"epochs": trained_epochs,
                                                "means" : [z[:, latent_dim][batch_idx] for latent_dim in range(latent_dims)], #z[:, 0] is the 1st latent dim
                                                "variances": [np.exp(z_sigma[:, latent_dim][batch_idx]) for latent_dim in range(latent_dims)],
                                                "latent_dims": latent_dims,
                                                "instrument": instrument,
                                                "note": note,
                                                "dynamic": dynamic,
                                                "technique": technique,
                                                "file": file,
                                                "family": family
                                            }, ignore_index=True)

    return df_total

#--------------PASAR A OTRO SCRIPT------------------
def latent_space_animation(model,
                           trained_epochs,
                           dataset,
                           num_batches=1,
                           projection='3d',
                           save_mp4=False,
                           save_gif=False,
                           name_fig='plot'):

    fig = plt.figure(figsize=(6, 6))

    if projection == '2d':
        ax = fig.gca()

    elif projection == '3d':
        ax = fig.gca(projection='3d')

    else:
        raise ValueError('Error.')

    def update_plot(epoch):
        ax.cla()
        df = data_to_pandas_dataframe(model, epoch, dataset, projection, num_batches)
        groups = df.groupby("instrument")
        for name, group in groups:
            ax.scatter(group["x"], group["y"], marker="o", s=8, label=name, alpha=0.5)

        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
        title = 'Latent space of VAE trained %d epochs'% epoch
        ax.set_title(title)
        return ax

    ani = animation.FuncAnimation(fig, update_plot, interval=250, frames=trained_epochs, repeat_delay=10000)

    if save_mp4:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(configs.PlotsConfig.PLOTS_PATH + '/' + name_fig + '.mp4', writer=writer)
        print(name_fig + '.mp4', 'saved in', configs.PlotsConfig.PLOTS_PATH)
    if save_gif:
        ani.save(configs.PlotsConfig.PLOTS_PATH + '/' + name_fig + '.gif')
        print(name_fig + '.gif', 'saved in', configs.PlotsConfig.PLOTS_PATH)

    plt.close(fig)

    return ani


def latent_space_animation_plotly(model, trained_epochs, dataset, projection='2d', num_batches=1, save_html=True, name_fig='plotly_anim'):
    df = px.data.gapminder()

    data = pd.DataFrame([])
    for epoch in range(trained_epochs):
        df = data_to_pandas_dataframe(model, epoch, dataset, projection, num_batches)
        data = data.append(df)

    if projection == '2d':
        fig = px.scatter(data, x="x", y="y", animation_frame="epochs", color="instrument")
    elif projection == '3d':
        fig = px.scatter_3d(data, x="x", y="y", z="z", animation_frame="epochs", color="instrument")

    fig.update_traces(marker=dict(size=5, opacity=0.5))
    fig.update_xaxes(visible=False, showticklabels=False)
    fig.update_yaxes(visible=False, showticklabels=False)
    fig.update_layout(xaxis_range=(df["x"].min()-1, df["x"].max()+1),
                      yaxis_range=(df["y"].min()-1, df["y"].max()+1))
    fig.show()

    if save_html:
        fig.write_html(configs.PlotsConfig.PLOTS_PATH + '/' + name_fig + ".html")
    return


def plot_activations(model, input_npy_path):

    # load npy file
    img = np.load(input_npy_path)

    # define the transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # apply the transforms
    img = transform(img)
    # unsqueeze to add a batch dimension
    img = img.unsqueeze(0)

    plt.figure(figsize=(5, 5))
    plt.imshow(img[0,0,:], origin='lower', cmap='viridis')
    plt.title(input_npy_path.split("/", 1)[1].split("/", 1)[1].split("/", 1)[1])
    plt.show()

    # append all the conv layers
    model_encoder = list(model.encoder.children())
    model_decoder = list(model.decoder.children())
    model_list = model_encoder + model_decoder
    conv_layers = []
    model_weights = []
    counter = 0
    for i in range(len(model_list)):
        if type(model_list[i]) == nn.Conv2d or type(model_list[i]) == nn.ConvTranspose2d:
            counter += 1
            model_weights.append(model_list[i].weight)
            conv_layers.append(model_list[i])


    # pass the image through all the layers
    results = [conv_layers[0](img.to(dtype=torch.float))]
    for i in range(1, len(conv_layers)):
        # pass the result from the last layer to the next layer
        results.append(conv_layers[i](results[-1]))
    outputs = results


    # visualize 64 features from each layer 
    # (although there are more feature maps in the upper layers)
    for num_layer in range(len(outputs)):
        plt.figure(figsize=(30, 30))
        layer_viz = outputs[num_layer][0, :, :, :]
        layer_viz = layer_viz.data
        for i, filter in enumerate(layer_viz):
            if i == 64: # we will visualize only 8x8 blocks from each layer
                break
            plt.subplot(8, 8, i + 1)
            plt.imshow(filter.cpu(), origin='lower', cmap='viridis')
            plt.axis("off")
        print(f"Saving layer {num_layer} feature maps...")
        print(layer_viz.size())

        if not os.path.exists(configs.PlotsConfig.ACTIVATION_PLOTS):
            os.mkdir(configs.PlotsConfig.ACTIVATION_PLOTS)

        plt.savefig(configs.PlotsConfig.ACTIVATION_PLOTS + f"/layer_{num_layer}.png")
        print(f"/layer_{num_layer}.png", 'Plot saved in', configs.PlotsConfig.ACTIVATION_PLOTS)
        plt.show()

    return


def plot_kernels(model):
    model_encoder = list(model.encoder.children())
    model_decoder = list(model.decoder.children())
    model_list = model_encoder + model_decoder
    conv_layers = []
    model_weights = []
    counter = 0
    for i in range(len(model_list)):
        if type(model_list[i]) == nn.Conv2d or type(model_list[i]) == nn.ConvTranspose2d:
            counter += 1
            model_weights.append(model_list[i].weight)
            conv_layers.append(model_list[i])

            kernels = model_list[i].weight.cpu().detach().clone()
            print('Layer:', model_list[i])
            print('Number of kernels:', kernels.shape[0], ', feature maps:', kernels.shape[1], ', kernels size:', kernels.shape[2], 'x', kernels.shape[3])
            plt.figure(figsize=(20, 17))

            for i, filter in enumerate(kernels):
                plt.subplot(8, 8, i+1) # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
                plt.imshow(filter[0, :, :].detach(), cmap='viridis')
                plt.axis('off')
            plt.show()
    return

@timeit
def plot_reconstruction_animation(trained_epochs,
                                  dataset,
                                  plot='train sample',
                                  idx=2000,
                                  input_path=None,
                                  save_mp4=True,
                                  save_gif=True,
                                  filename='reconstruction_animation'):

    if plot == 'train sample':
        # sample from training set
        k = int(np.floor(idx / configs.ParamsConfig.BATCH_SIZE))
        image = next(itertools.islice(dataset, k, None))
        image['file'] = image['file'][0]

    elif plot == 'from npy':
        name = input_path.split("/", 1)[1].split("/", 1)[1].split("/", 1)[1]
        img = np.load(input_path)
        image = {
                'input': img,
                'file': name
                }

        # apply the transforms
        image['input'] = image['input'][np.newaxis, ...]
        image = data_utils.padding(image)
        image['input'] = torch.Tensor(image['input'])
        image['input'] = image['input'].unsqueeze(0)

    elif plot == 'from audio':
        input = data_utils.compute_input(input_path)

        name = input_path.split("/", 1)[1].split("/", 1)[1].split("/", 1)[1]
        image = {
                'input': input,
                'file': name
                }

        # apply the transforms
        image['input'] = image['input'][np.newaxis, ...]
        image = data_utils.padding(image)
        image['input'] = torch.Tensor(image['input'])
        image['input'] = image['input'].unsqueeze(0)

    else:
        raise ValueError('Non valid plot argument.')


    fig, ax = plt.subplots(1, 2)
    fig.suptitle(image['file'], y=1)
    fig.subplots_adjust(hspace = .5, wspace=.5)

    #aspect = image['input'][0, 0,...].shape[1] * 0.1 // image['input'][0, 0,...].shape[0]

    ax[0].imshow(image['input'][0,0,:,:], cmap='viridis', interpolation='none', aspect=1, origin='lower')
    ax[0].text(0.5, 1.02, 'Input CQT',
               size=plt.rcParams["axes.titlesize"],
               ha="center", transform=ax[0].transAxes, )

    ims = []
    for epoch in range(trained_epochs):
        load_pretrained_model(model, epoch)
        reconstruction = model(image['input'].to(dtype=torch.float))

        im = ax[1].imshow(reconstruction[0][0, 0, :, :].detach().numpy(),
                          cmap='viridis', interpolation='none', aspect=1, origin='lower')

        title = ax[1].text(0.5, 1.02, 'Reconstructed CQT, epoch: ' + str(epoch),
                            size=plt.rcParams["axes.titlesize"],
                            ha="center", transform=ax[1].transAxes,)

        ims.append([im, title])

    ani = animation.ArtistAnimation(fig, ims, interval=250, blit=True, repeat_delay=10000)

    if save_mp4:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(configs.PlotsConfig.PLOTS_PATH + '/' + filename + '.mp4', writer=writer)
        print(filename + '.mp4', 'saved in', configs.PlotsConfig.PLOTS_PATH)

    if save_gif:
        ani.save(configs.PlotsConfig.PLOTS_PATH + '/' + filename + '.gif')
        print(filename + '.gif', 'saved in', configs.PlotsConfig.PLOTS_PATH)

    plt.close(fig)

    return ani
