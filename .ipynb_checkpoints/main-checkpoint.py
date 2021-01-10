import torch
import torch.utils.data
from torch import nn, optim
# from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
#from tqdm import tqdm

import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from model.core import VAE
from model.module_utils import *
from model.vae_utility import *
from dataset.loader import *
from config import *

import random as rn

np.random.seed(42)
# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
rn.seed(12345)



def main():
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        print('added visible gpu')
        ngpus = torch.cuda.device_count()
 
    #####Load data#########
    trainx, valx, ohe = get_train_val(0.2, cols, cat_cols, preprocessing = 'log')
    testx, testy, orig_labels= get_test(cols, cat_cols, ohe, preprocessing='log')

    trainloader = torch.utils.data.DataLoader(trainx, batch_size=batch_size*ngpus, shuffle=True)
    valloader = torch.utils.data.DataLoader(valx, batch_size=batch_size*ngpus, shuffle=True)
    
    ####initialize model####
    input_size = trainx.shape[1]
    Nf_binomial = input_size-Nf_lognorm
    
   
    vae = VAE(input_size, hidden_size
          , latent_dim, Nf_lognorm, Nf_binomial).to(device)
    
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    
    ####Train Loop####
    """
    Set the model to the training mode first and train
    """
    vae.train()
    train_loss = []
    val_loss = 10**16
    for epoch in range(epochs):
        for i, data in enumerate(trainloader):
            data = (data.type(torch.FloatTensor)).to(device)
            optimizer.zero_grad()
            pars, mu, sigma = vae(data)

            recon_loss = loss_function(data, pars, Nf_lognorm,
                                      Nf_binomial).mean()

            KLD = KL_loss_forVAE(mu,sigma).mean()
            loss = recon_loss + weight_KL_loss*KLD  # the mean of KL is added to the mean of MSE
            loss.backward()
            train_loss.append(loss.item())
            optimizer.step()

            if i % 10 == 0:
                print("Loss:")
    #             print(loss.item()/len(images))
                print(loss.item())
        
        ###############################################
        #eval mode for evaluation on validation dataset
        ###############################################
        vae.eval()
        vectors = []
        with torch.no_grad():
            for i, data in enumerate(valloader):
                data = (data.type(torch.FloatTensor)).to(device)
                pars, mu, sigma = vae(data)

                recon_loss = loss_function(data, pars, Nf_lognorm,
                                          Nf_binomial).mean()

                KLD = KL_loss_forVAE(mu,sigma).mean()
                loss += recon_loss + weight_KL_loss*KLD

            loss = loss / len(valloader)
            if loss < val_loss:
                print('val_loss improved from {} to {}, saving model to {}'\
                      .format(val_loss, loss, save_model_path))
                torch.save(vae.state_dict(), save_model_path / model_name)
                val_loss = loss
    
    
if __name__ == "__main__":
    
    ###############################################
    # TO DO: add parser for parse command line args
    ###############################################
    
    batch_size = 400
    
    hidden_size = 20
    latent_dim = 4
    act_fun = 'relu'
    # kernel_max_norm = 1000
    
    weight_KL_loss = 0.8
    Nf_lognorm = 34
    
    epochs = 20
    lr = 0.001
    
    save_model_path = Path(model_results_path)
    model_name = 'vae_nls_kdd.h5'
    if not(save_model_path.exists()):
        print('creating path')
        os.makedirs(save_model_path)

    main()
