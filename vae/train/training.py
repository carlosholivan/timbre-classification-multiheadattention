import os
import glob
import csv
import sys, os.path

import sklearn
import torch
from torch import optim, nn

import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Our modules
sys.path.append('.')
sys.path.append('..')

from vae import configs, train, plot_utils, models
from vae.data import build_dataloader
from vae.latent_spaces import dimensionality_reduction, plot_spaces
from vae.reconstructions import plot_reconstructions
from vae.models import model_utils
from vae.train import loss

import numpy as np


def train_model(input, model_name, loss_f, epochs=configs.ParamsConfig.NUM_EPOCHS,
                optimizer=None, lr=configs.ParamsConfig.LEARNING_RATE,
                tensorboard=False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Building dataloaders...')
    train_dataset, train_dataloader, val_dataset, val_dataloader = build_dataloader.build_dataset(input)
    print('Dataloaders have been built.')
    print('Number of files in the training dataset:', len(train_dataset))
    print('Number of files in the validation dataset:', len(val_dataset))

    # show configs
    configs.show_configs(model=model_name)

    # import model
    model = model_utils.import_model(model_name)

    # show model
    #model_utils.show_model(model, 'mel')

    if optimizer == None:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)


    if tensorboard:

        if not os.path.exists('../runs'):
            os.mkdir('../runs')

        # delete contents in runs directory
        if os.listdir('../runs'):
            r = glob.glob('../runs/*')
            for i in r:
                os.remove(i)

        writer_losses = SummaryWriter('../runs')
        writer_kld = SummaryWriter('../runs')


    train_loss = []
    val_loss = []
    train_kld = []
    train_kld1 = []
    train_kld2 = []
    val_kld = []
    val_kld1 = []
    val_kld2 = []
    train_bce = []
    val_bce = []

    print('Start training model on', device, '...')

    for epoch in range(epochs):
        model.train()

        train_loss.append(0)
        train_kld.append(0)
        train_kld1.append(0)
        train_kld2.append(0)
        train_bce.append(0)
        num_batches = 0

        pbar = tqdm(total=len(train_dataloader))
        print("Epoch:", epoch)

        for image_batch, _ in train_dataloader:
            image_batch = image_batch.to(device, dtype=torch.float)

            if model_name == 'TimeConv2D_cut':
                image_batch_recon, latent_mu1, latent_logvar1, latent_mu2, latent_logvar2 = model(image_batch, image_batch)
                total_loss, kld1, kld2, bce = loss.loss_function_combined(image_batch_recon, image_batch,
                                                                                                        latent_mu1, latent_logvar1,
                                                                                                        latent_mu2, latent_logvar2,
                                                                                                        torch_loss=loss_f)

            else:
                # vae reconstruction
                image_batch_recon, latent_mu, latent_logvar = model(image_batch)
                # reconstruction error
                total_loss, kld, bce = loss.loss_function(image_batch_recon, image_batch, latent_mu, latent_logvar, torch_loss=loss_f)

            # backpropagation
            optimizer.zero_grad()
            total_loss.backward()

            # one step of the optmizer (using the gradients from backpropagation)
            optimizer.step()


            if model_name == 'TimeConv2D_cut':
                train_kld1[-1] += kld1.item()
                train_kld2[-1] += kld2.item()
            else:
                train_kld[-1] += kld.item()

            train_loss[-1] += total_loss.item()
            train_bce[-1] += bce.item()
            num_batches += 1

            pbar.update()

        train_loss[-1] /= num_batches
        train_kld[-1] /= num_batches
        train_bce[-1] /= num_batches

        pbar.close()

        if model_name == 'TimeConv2D_cut':
            print("training_avg_loss={:.2f}\n training_avg_kld1={:.2f}\n training_avg_kld2={:.2f}\n training_avg_rec={:.2f}\n".format(train_loss[-1],train_kld1[-1], train_kld1[-1], train_bce[-1]))
        else:
            print("training_avg_loss={:.2f}\n training_avg_kld={:.2f}\n training_avg_rec={:.2f}\n".format(train_loss[-1], train_kld[-1], train_bce[-1]))


        model.eval()
        val_loss.append(0)
        val_kld.append(0)
        val_kld1.append(0)
        val_kld2.append(0)
        val_bce.append(0)
        num_batches = 0
        with torch.no_grad():
            for image_batch, _ in val_dataloader:
                image_batch = image_batch.to(device, dtype=torch.float)

                if model_name == 'TimeConv2D_cut':
                    image_batch_recon, latent_mu1, latent_logvar1, latent_mu2, latent_logvar2 = model(image_batch, image_batch)
                    total_val_loss, kld1, kld2, bce = loss.loss_function_combined(image_batch_recon, image_batch,
                                                                                                          latent_mu1, latent_logvar1,
                                                                                                          latent_mu2, latent_logvar2,
                                                                                                          torch_loss = loss_f)
                else:
                    # vae reconstruction
                    image_batch_recon, latent_mu, latent_logvar = model(image_batch)
                    total_val_loss, kld, bce = loss.loss_function(image_batch_recon, image_batch, latent_mu, latent_logvar, torch_loss=loss_f)

                if model_name == 'TimeConv2D_cut':
                    val_kld1[-1] += kld1.item()
                    val_kld2[-1] += kld2.item()
                else:
                    val_kld[-1] += kld.item()

                val_loss[-1] += total_val_loss.item()
                val_bce[-1] += bce.item()
                num_batches += 1

        if model_name == 'TimeConv2D_cut':
            val_kld1[-1] /= num_batches
            val_kld2[-1] /= num_batches
        else:
            val_kld[-1] /= num_batches
        val_loss[-1] /= num_batches
        val_bce[-1] /= num_batches

        print("val_avg_loss={:.2f}\n".format(val_loss[-1]))
        
        if tensorboard:
            # Tensorboard plots
            writer_losses.add_scalar('Train loss', train_epoch_loss, epoch)
            writer_losses.add_scalar('Val loss', val_epoch_loss, epoch)
            writer_kld.add_scalar('Train KLD', train_epoch_kld, epoch)
            writer_kld.add_scalar('Val KLD', val_epoch_kld, epoch)
            writer_losses.close()
            writer_kld.close()

        if model_name == 'TimeConv2D_cut':
            checkpoint = {
                'model': model.state_dict(),
                'train loss': train_loss,
                'train kld_1': train_kld1,
                'train kld_2': train_kld2,
                'train bce': train_bce,
                'val loss': val_loss,
                'val kld_1': val_kld1,
                'val kld_2': val_kld2,
                'val bce': val_bce
            }

        else:
            checkpoint = {
                'model': model.state_dict(),
                'train loss': train_loss,
                'train kld': train_kld,
                'train bce': train_bce,
                'val loss': val_loss,
                'val kld':  val_kld,
                'val bce': val_bce
                                    }



        if not os.path.exists(configs.ParamsConfig.TRAINED_MODELS_PATH):
            os.mkdir(configs.ParamsConfig.TRAINED_MODELS_PATH)

        # save trained model every 10 epochs
        if epoch % 10 == 0:

            torch.save(checkpoint,
                                os.path.join(configs.ParamsConfig.TRAINED_MODELS_PATH, 'saved_model_' + str(epoch) + "epochs.pth"))

            plot_reconstructions.plot_batch_reconstruction_from_dataset(model=model,
                                                                                                        model_name=model_name,
                                                                                                        batch_num=300,
                                                                                                        trained_epochs=epoch,
                                                                                                        dataloader=val_dataloader,
                                                                                                        sample=1)

            plot_utils.plot_losses(model=model, model_name=model_name, trained_epochs=epoch, plot='all')
            plot_utils.plot_losses(model=model, model_name=model_name, trained_epochs=epoch, plot='kld')
            plot_utils.plot_losses(model=model, model_name=model_name, trained_epochs=epoch, plot='reconstruction')

    
