import numpy as np
import torch

clip_x_to0 = 1e-4

def SmashTo0(x):
    return 0*x

def loss_function(x, pars, Nf_lognorm, Nf_binomial):
    recon_loss = RecoProb_forVAE(x, pars[0], pars[1], pars[2], Nf_lognorm, Nf_binomial)
    return recon_loss

def RecoProb_forVAE(x, par1, par2, par3, Nf_lognorm, Nf_binomial):

    N = 0
    nll_loss = 0

    #Log-Normal distributed variables
    mu = par1[:,:Nf_lognorm]
    sigma = par2[:,:Nf_lognorm]
    fraction = par3[:,:Nf_lognorm]

    x_clipped = torch.clamp(x[:,:Nf_lognorm], clip_x_to0, 1e8)
    single_NLL = torch.where(torch.le(x[:,:Nf_lognorm], clip_x_to0),
                            -torch.log(fraction),
                                -torch.log(1-fraction)
                                + torch.log(sigma)
                                + torch.log(x_clipped)
                                + 0.5*torch.mul(torch.div(torch.log(x_clipped) - mu, sigma),
                                                  torch.div(torch.log(x_clipped) - mu, sigma)))
    nll_loss += torch.sum(single_NLL, axis=-1)

    N += Nf_lognorm

    #Binomial distributed variables
    p = 0.5*(1+0.98*torch.tanh(par1[:, N: N+Nf_binomial]))
    single_NLL = -torch.where(torch.eq(x[:, N: N+Nf_binomial],1), torch.log(p), torch.log(1-p))
    nll_loss += torch.sum(single_NLL, axis=-1)
    N += Nf_binomial

    return nll_loss

def KL_loss_forVAE(mu, sigma):
    mu_prior = torch.tensor(0)
    sigma_prior = torch.tensor(1)
    kl_loss = torch.mul(torch.mul(sigma, sigma), torch.mul(sigma_prior,sigma_prior))
    div = torch.div(mu_prior - mu, sigma_prior)
    kl_loss += torch.mul(div, div)
    kl_loss += torch.log(torch.div(sigma_prior, sigma)) -1
    return 0.5 * torch.sum(kl_loss, axis=-1)

