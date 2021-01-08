from torch import nn, optim
from model.module_utils import *
# from module_utils import * import so adding in init.py

class VAE(nn.Module):
    def __init__(self, original_dim, intermediate_dim,
                latent_dim, Nf_lognorm, Nf_binomial):
        
        self.original_dim = original_dim
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        self.Nf_lognorm = Nf_lognorm
        self.Nf_binomial = Nf_binomial
        
        super(VAE, self).__init__()
        self.enc = nn.Sequential(nn.Linear(self.original_dim, self.intermediate_dim), nn.ReLU(True)\
                                 , nn.Linear(self.intermediate_dim, self.intermediate_dim), nn.ReLU(True))
        
        self.mu = nn.Linear(self.intermediate_dim, self.latent_dim)
        self.sigma = nn.Linear(self.intermediate_dim, self.latent_dim)
        
        
        self.dec = nn.Sequential(nn.Linear(self.latent_dim, self.intermediate_dim), nn.ReLU(True),
                                 nn.Linear(self.intermediate_dim, self.intermediate_dim), nn.ReLU(True))
        
        self.par1 = nn.Linear(self.intermediate_dim, self.original_dim) 
        self.par2 = nn.Linear(self.intermediate_dim, self.Nf_lognorm)
        self.par3 = nn.Linear(self.intermediate_dim, self.Nf_lognorm)  
        
        self.act2 = InverseSquareRootLinearUnit()
        self.act3 = ClippedTanh()    
        
    def encode(self, x):
        enc = self.enc(x)
        mu = self.mu(enc)
        sigma_pre = self.sigma(enc)
        return mu, self.act2(sigma_pre)

    def sample(self, mu, sigma):
        std = sigma
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        z = z.view(z.size(0), self.latent_dim)
        d = self.dec(z)
        par2 = self.par2(d)
        par3 = self.par3(d)
        return self.par1(d), self.act2(par2), self.act3(par3)

    def forward(self, x):
        mu, sigma = self.encode(x.view(x.size(0), self.original_dim))

        z = self.sample(mu, sigma)
        
        return self.decode(z), mu, sigma
