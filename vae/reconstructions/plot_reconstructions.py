import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

# Our modules
from vae import configs


def get_batch_reconstruction_from_dataset(model, model_name, batch_num, trained_epochs, dataloader):

    k = int(np.floor(batch_num/configs.ParamsConfig.BATCH_SIZE))

    input_image, y = next(itertools.islice(dataloader, k, None))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

     # Reconstruction
    checkpoint = torch.load(configs.ParamsConfig.TRAINED_MODELS_PATH + '/saved_model_' + str(trained_epochs) + "epochs.pth")
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device)

    if model_name == 'TimeConv2D_cut':
        reconstruction, *_ = model(input_image.to(device, dtype=torch.float), input_image.to(device, dtype=torch.float))
    else:
        reconstruction, *_ = model(input_image.to(device, dtype=torch.float))

    return input_image[:,0,:,:].cpu().detach().numpy(), reconstruction[:,0,:,:].cpu().detach().numpy(), y[0][:]


def plot_batch_reconstruction_from_dataset(model, model_name, batch_num, trained_epochs, dataloader, sample):
    
    input_image, reconstruction, y = get_batch_reconstruction_from_dataset(model, model_name, batch_num, trained_epochs, dataloader)
    
    ncol = 3
    nrow = input_image.shape[0]

    fig, ax = plt.subplots(nrow, ncol, sharex=True, sharey=True, figsize=(6, 20),
                            gridspec_kw={'wspace': 0.01, 'hspace': 0.00})

    
    for sample in range(nrow):
        ax[sample, 0].imshow(input_image[sample, ...], cmap='viridis', origin='lower', aspect='auto')
        ax[sample, 0].set_xticks([]) 
        ax[sample, 0].set_ylabel(y[sample], rotation='horizontal', ha='right', fontsize=10)
            
        ax[sample, 1].imshow(reconstruction[sample, ...], cmap='viridis', origin='lower', aspect='auto')
        ax[sample, 1].axis('off')

        ax[sample, 2].imshow(input_image[sample, ...]-reconstruction[sample, ...], cmap='viridis', origin='lower', aspect='auto')
        ax[sample, 2].axis('off')

        if sample == 0:
            ax[sample, 0].set_title('Input')
            ax[sample, 1].set_title('Reconstr.')
            ax[sample, 2].set_title('Difference')

    plt.show()
    
    return

def plot_single_reconstruction_from_dataset(model, model_name, batch_num, trained_epochs, dataloader, sample):
    
    input_image, reconstruction, y = get_batch_reconstruction_from_dataset(model, model_name, batch_num, trained_epochs, dataloader)

    fig, ax = plt.subplots(1, 3, figsize=(8, 8))
    fig.subplots_adjust(hspace=0, wspace=.1)

    ax[0].imshow(input_image, cmap='viridis', interpolation='none', aspect=1, origin='lower')
    ax[0].set_title(y[sample])
        
    ax[1].imshow(reconstruction, cmap='viridis', interpolation='none', aspect=1, origin='lower')
    ax[1].set_title('Reconstr., epoch: ' + str(trained_epochs))
    ax[1].set_yticklabels([]) #remove y axis

    ax[2].imshow(input_image-reconstruction, cmap='viridis', interpolation='none', aspect=1, origin='lower')
    ax[2].set_title('Diference')
    ax[2].set_yticklabels([]) #remove y axis
    
    plt.show()
    
    return

