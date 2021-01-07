import sys, scipy
from scipy.stats import chi2, poisson
from scipy.special import erf
import numpy as np

from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import layers as KL
from keras import metrics
from keras import optimizers

from keras.callbacks import  ModelCheckpoint

clip_x_to0 = 1e-4


def IdentityLoss(y_train, NETout):
    return K.mean(NETout)

def InverseSquareRootLinearUnit(args, min_value = 5e-3):
    return 1. + min_value + K.tf.where(K.tf.greater(args, 0), args, K.tf.divide(args, K.sqrt(1+K.square(args))))

def ClippedTanh(x):
    return 0.5*(1+0.999*K.tanh(x))

def SmashTo0(x):
    return 0*x

def sampling(args):
    z_mean, z_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], K.shape(z_mean)[1]), mean=0.,
                              stddev=1.)
    return z_mean + z_sigma * epsilon

def sampleZ(z):
    '''Sampling function of Gaussian latent space.
    Combines mu and sigma tensors with Z = mu + eps*sig, where eps is random normal centred on zero'''
    mu, logsig = z
    epsilon = K.random_normal( K.shape(mu) ,dtype='float32')
    return mu+epsilon*K.exp(logsig)

def sum_of_gaussians(x, mu_vec, sigma_vec):
    x = np.atleast_2d(x)
    if x.shape[0] <= x.shape[1]:
        x = x.T
    x_norm = (x - mu_vec)/sigma_vec
    single_gaus_val = np.exp(-0.5*np.square(x_norm))/(sigma_vec*np.sqrt(2*np.pi))
    return np.sum(single_gaus_val, axis=1)/mu_vec.shape[0]

def sum_of_possion(x_in, mu_vec):
    out = np.zeros_like(x_in)
    for i, aux in enumerate(x_in):
        out[i] = np.sum(poisson.pmf(aux, mu_vec))
    return out

def sum_of_lognorm(x, f, mu_vec, sigma_vec):
    x = np.atleast_2d(x)
    if x.shape[0] <= x.shape[1]:
        x = x.T

    x_clipped = np.clip(x, clip_x_to0, 1e8)
    x_norm = (np.log(x_clipped) - mu_vec)/sigma_vec
    single_prob = np.where(np.less(x, clip_x_to0),
                               f,
                               (1-f)*np.exp(-0.5*np.square(x_norm))/(x_clipped*sigma_vec*np.sqrt(2*np.pi))
    )
    return np.sum(single_prob, axis=1)/mu_vec.shape[0]

def sum_of_PDgauss(x, mu, sigma):
    x = np.atleast_2d(x)
    if x.shape[0] <= x.shape[1]:
        x = x.T

    zp = (x + 0.5 - mu)/sigma
    zm = (x - 0.5 - mu)/sigma

    norm_0 = (-0.5 - mu)/sigma

    aNorm = 1 + 0.5*(1 + erf(norm_0/np.sqrt(2)))
    single_prob = aNorm*0.5*(erf(zp/np.sqrt(2)) - erf(zm/np.sqrt(2)))
    return np.sum(single_prob, axis=1)/mu.shape[0]

def sum_of_Pgauss(x, mu, sigma):
    x = np.atleast_2d(x)
    if x.shape[0] <= x.shape[1]:
        x = x.T
    x_norm = (x - mu)/sigma
    norm_0 = - mu/sigma
    aNorm = 1 + 0.5*(1 + erf(norm_0/np.sqrt(2)))

    single_prob = aNorm*np.exp(-0.5*np.square(x_norm))/(sigma*np.sqrt(2*np.pi))
    return np.sum(single_prob, axis=1)/mu.shape[0]

def ROC_curve(p_BSM, p_SM, eval_q_SM):
    eval_p = np.percentile(1-p_SM, q=100*eval_q_SM)

    out = (1-p_BSM) < eval_p
    out = np.sum(out, axis=0)
    q_BSM = out/float(p_BSM.shape[0])

    AUC = np.trapz(q_BSM, eval_q_SM)

    return q_BSM, AUC


class SaveAutoencoder(ModelCheckpoint):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
         save_best_only=False, save_weights_only=False, mode='auto', period=1):

        super(ModelCheckpoint, self).__init__()

        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:

            warnings.warn('ModelCheckpoint mode %s is unknown, '
                      'fallback to auto mode.' % (mode),
                      RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                    self.monitor_op = np.greater
                    self.best = -np.Inf
            else:
                    self.monitor_op = np.less
                    self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):

        autoencoder = Model(inputs=self.model.input,
                                 outputs=[self.model.get_layer('Output_par1').output,
                                         self.model.get_layer('Output_par2').output,
                                          self.model.get_layer('Output_par3').output])


        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            autoencoder.save_weights(filepath, overwrite=True)
                        else:
                            autoencoder.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    autoencoder.save_weights(filepath, overwrite=True)
                else:
                    autoencoder.save(filepath, overwrite=True)

                    
def KL_loss_forVAE(mu, sigma, mu_prior, sigma_prior):
    kl_loss = K.tf.multiply(K.square(sigma), K.square(sigma_prior))
    kl_loss += K.square(K.tf.divide(mu_prior - mu, sigma_prior))
    kl_loss += K.log(K.tf.divide(sigma_prior, sigma)) -1
    return 0.5 * K.sum(kl_loss, axis=-1)

class CustomRecoProbLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomRecoProbLayer, self).__init__(**kwargs)

    def call(self, inputs):
        x, par1, par2, par3 = inputs
        return RecoProb_forVAE() # REMOVED THE w weights

def RecoProb_forVAE():
    
    def loss(x, par1, par2, par3, w):
        N = 0
        nll_loss = 0

        if Nf_lognorm != 0:
            for i in range(Nf_lognorm):
                #Log-Normal distributed variables
                mu = par1[:,i:i+1]
                sigma = par2[:,i:i+1]
                fraction = par3[:,i:i+1]
                x_clipped = K.clip(x[:,i:i+1], clip_x_to0, 1e8)
                single_NLL = K.tf.where(K.less(x[:,i:i+1], clip_x_to0),
                                        -K.log(fraction),
                                            -K.log(1-fraction)
                                            + K.log(sigma)
                                            + K.log(x_clipped)
                                            + 0.5*K.square(K.tf.divide(K.log(x_clipped) - mu, sigma)))
                nll_loss += K.sum(w[i]*single_NLL, axis=-1)

            N += Nf_lognorm

        if Nf_PDgauss != 0:

            for i in range(N, N+Nf_PDgauss):

                mu = par1[:,i:i+1]
                sigma = par2[:,i:i+1]
                norm_xp = K.tf.divide(x[:,i:i+1] + 0.5 - mu, sigma)
                norm_xm = K.tf.divide(x[:,i:i+1] - 0.5 - mu, sigma)
                sqrt2 = 1.4142135624
                single_LL = 0.5*(K.tf.erf(norm_xp/sqrt2) - K.tf.erf(norm_xm/sqrt2))

                norm_0 = K.tf.divide(-0.5 - mu, sigma)
                aNorm = 1 + 0.5*(1 + K.tf.erf(norm_0/sqrt2))
                single_NLL = -K.log(K.clip(single_LL, 1e-10, 1e40)) -K.log(aNorm)

                nll_loss += K.sum(w[i]*single_NLL, axis=-1)
        return nll_loss

    return loss


def IndividualRecoProb_forVAE_lognorm_1():
    
    def loss(x, par1, par2, par3, w):
        N = 0
        nll_loss = 0
        w = w[0]

        mu = par1[:,:1]
        sigma = par2[:,:1]
        fraction = par3[:,:1]
        x_clipped = K.clip(x[:,:1], clip_x_to0, 1e8)
        single_NLL = K.tf.where(K.less(x[:,:1], clip_x_to0),
                                -K.log(fraction),
                                    -K.log(1-fraction)
                                    + K.log(sigma)
                                    + K.log(x_clipped)
                                    + 0.5*K.square(K.tf.divide(K.log(x_clipped) - mu, sigma)))
        nll_loss += K.sum(w*single_NLL, axis=-1)
        return nll_loss

    return loss

def IndividualRecoProb_forVAE_lognorm_2():
    
    def loss(x, par1, par2, par3):
        N = 0
        nll_loss = 0
        w = w[1]

        mu = par1[:,1:2]
        sigma = par2[:,1:2]
        fraction = par3[:,1:2]
        x_clipped = K.clip(x[:,1:2], clip_x_to0, 1e8)
        single_NLL = K.tf.where(K.less(x[:,1:2], clip_x_to0),
                                -K.log(fraction),
                                    -K.log(1-fraction)
                                    + K.log(sigma)
                                    + K.log(x_clipped)
                                    + 0.5*K.square(K.tf.divide(K.log(x_clipped) - mu, sigma)))
        nll_loss += K.sum(w*single_NLL, axis=-1)
        return nll_loss

    return loss

def IndividualRecoProb_forVAE_lognorm_3():
    
    def loss(x, par1, par2, par3):
        N = 0
        nll_loss = 0
        w = w[2]

        mu = par1[:,2:3]
        sigma = par2[:,2:3]
        fraction = par3[:,2:3]
        x_clipped = K.clip(x[:,2:3], clip_x_to0, 1e8)
        single_NLL = K.tf.where(K.less(x[:,2:3], clip_x_to0),
                                -K.log(fraction),
                                    -K.log(1-fraction)
                                    + K.log(sigma)
                                    + K.log(x_clipped)
                                    + 0.5*K.square(K.tf.divide(K.log(x_clipped) - mu, sigma)))
        nll_loss += K.sum(w*single_NLL, axis=-1)
        return nll_loss

    return loss

def IndividualRecoProb_forVAE_lognorm_4(x, par1, par2, par3, w):
    N = 0
    nll_loss = 0
    w = w[3]

    mu = par1[:,3:4]
    sigma = par2[:,3:4]
    fraction = par3[:,3:4]
    x_clipped = K.clip(x[:,3:4], clip_x_to0, 1e8)
    single_NLL = K.tf.where(K.less(x[:,3:4], clip_x_to0),
                            -K.log(fraction),
                                -K.log(1-fraction)
                                + K.log(sigma)
                                + K.log(x_clipped)
                                + 0.5*K.square(K.tf.divide(K.log(x_clipped) - mu, sigma)))
    nll_loss += K.sum(w*single_NLL, axis=-1)

    return nll_loss

def IndividualRecoProb_forVAE_lognorm_5(x, par1, par2, par3, w):
    N = 0
    nll_loss = 0
    w = w[4]
    
    mu = par1[:,4:5]
    sigma = par2[:,4:5]
    fraction = par3[:,4:5]
    x_clipped = K.clip(x[:,4:5], clip_x_to0, 1e8)
    single_NLL = K.tf.where(K.less(x[:,4:5], clip_x_to0),
                            -K.log(fraction),
                                -K.log(1-fraction)
                                + K.log(sigma)
                                + K.log(x_clipped)
                                + 0.5*K.square(K.tf.divide(K.log(x_clipped) - mu, sigma)))
    nll_loss += K.sum(w*single_NLL, axis=-1)

    return nll_loss


def individualRecoProb_forVAE_discrete_6(x, par1, par2, w):
    nll_loss = 0
    w = w[5]
    mu = par1[:,5:6]
    sigma = par2[:,5:6]
    norm_xp = K.tf.divide(x[:,5:6] + 0.5 - mu, sigma)
    norm_xm = K.tf.divide(x[:,5:6] - 0.5 - mu, sigma)
    sqrt2 = 1.4142135624
    single_LL = 0.5*(K.tf.erf(norm_xp/sqrt2) - K.tf.erf(norm_xm/sqrt2))

    norm_0 = K.tf.divide(-0.5 - mu, sigma)
    aNorm = 1 + 0.5*(1 + K.tf.erf(norm_0/sqrt2))
    single_NLL = -K.log(K.clip(single_LL, 1e-10, 1e40)) -K.log(aNorm)

    nll_loss += K.sum(w*single_NLL, axis=-1)

    return nll_loss


def individualRecoProb_forVAE_discrete_7(x, par1, par2, w):
    nll_loss = 0
    w = w[6]
    mu = par1[:,6:7]
    sigma = par2[:,6:7]
    norm_xp = K.tf.divide(x[:,6:7] + 0.5 - mu, sigma)
    norm_xm = K.tf.divide(x[:,6:7] - 0.5 - mu, sigma)
    sqrt2 = 1.4142135624
    single_LL = 0.5*(K.tf.erf(norm_xp/sqrt2) - K.tf.erf(norm_xm/sqrt2))

    norm_0 = K.tf.divide(-0.5 - mu, sigma)
    aNorm = 1 + 0.5*(1 + K.tf.erf(norm_0/sqrt2))
    single_NLL = -K.log(K.clip(single_LL, 1e-10, 1e40)) -K.log(aNorm)

    nll_loss += K.sum(w*single_NLL, axis=-1)

    return nll_loss

def IndividualRecoProb_forVAE_lognorm_6(x, par1, par2, par3, w):
    N = 0
    nll_loss = 0
    w = w[5]
    mu = par1[:,5:6]
    sigma = par2[:,5:6]
    fraction = par3[:,5:6]
    x_clipped = K.clip(x[:,5:6], clip_x_to0, 1e8)
    single_NLL = K.tf.where(K.less(x[:,5:6], clip_x_to0),
                            -K.log(fraction),
                                -K.log(1-fraction)
                                + K.log(sigma)
                                + K.log(x_clipped)
                                + 0.5*K.square(K.tf.divide(K.log(x_clipped) - mu, sigma)))
    nll_loss += K.sum(w*single_NLL, axis=-1)

    return nll_loss



def individualRecoProb_forVAE_discrete_8(x, par1, par2, w):
    N = Nf_lognorm
    nll_loss = 0
    w = w[7]
    mu = par1[:,7:8]
    sigma = par2[:,7:8]
    norm_xp = K.tf.divide(x[:,7:8] + 0.5 - mu, sigma)
    norm_xm = K.tf.divide(x[:,7:8] - 0.5 - mu, sigma)
    sqrt2 = 1.4142135624
    single_LL = 0.5*(K.tf.erf(norm_xp/sqrt2) - K.tf.erf(norm_xm/sqrt2))

    norm_0 = K.tf.divide(-0.5 - mu, sigma)
    aNorm = 1 + 0.5*(1 + K.tf.erf(norm_0/sqrt2))
    single_NLL = -K.log(K.clip(single_LL, 1e-10, 1e40)) -K.log(aNorm)

    nll_loss += K.sum(w*single_NLL, axis=-1)

    return nll_loss


class CustomKLLossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomKLLossLayer, self).__init__(**kwargs)

    def call(self, inputs):
        mu, sigma, mu_prior, sigma_prior = inputs
        return KL_loss_forVAE(mu, sigma, mu_prior, sigma_prior)

#################################################################################################à
class CustomIndividualLogNorLayer_1(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomIndividualLogNorLayer_1, self).__init__(**kwargs)

    def call(self, inputs):
        x, par1, par2, par3 = inputs
        return IndividualRecoProb_forVAE_lognorm_1()

class CustomIndividualLogNorLayer_2(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomIndividualLogNorLayer_2, self).__init__(**kwargs)

    def call(self, inputs):
        x, par1, par2, par3 = inputs
        return IndividualRecoProb_forVAE_lognorm_2()


class CustomIndividualLogNorLayer_3(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomIndividualLogNorLayer_3, self).__init__(**kwargs)

    def call(self, inputs):
        x, par1, par2, par3 = inputs
        return IndividualRecoProb_forVAE_lognorm_3()

class CustomIndividualLogNorLayer_4(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomIndividualLogNorLayer_4, self).__init__(**kwargs)

    def call(self, inputs):
        x, par1, par2, par3 = inputs
        return IndividualRecoProb_forVAE_lognorm_4(x, par1, par2, par3, w)

class CustomIndividualLogNorLayer_5(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomIndividualLogNorLayer_5, self).__init__(**kwargs)

    def call(self, inputs):
        x, par1, par2, par3 = inputs
        return IndividualRecoProb_forVAE_lognorm_5(x, par1, par2, par3, w)

class CustomIndividualLogNorLayer_6(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomIndividualLogNorLayer_6, self).__init__(**kwargs)

    def call(self, inputs):
        x, par1, par2, par3 = inputs
        return IndividualRecoProb_forVAE_lognorm_6(x, par1, par2, par3, w)

class CustomIndividualTruGauLayer_6(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomIndividualTruGauLayer_6, self).__init__(**kwargs)

    def call(self, inputs):
        x, par1, par2 = inputs
        return individualRecoProb_forVAE_discrete_6(x, par1, par2, w)

class CustomIndividualTruGauLayer_7(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomIndividualTruGauLayer_7, self).__init__(**kwargs)

    def call(self, inputs):
        x, par1, par2 = inputs
        return individualRecoProb_forVAE_discrete_7(x, par1, par2, w)
    
class CustomIndividualTruGauLayer_8(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomIndividualTruGauLayer_8, self).__init__(**kwargs)

    def call(self, inputs):
        x, par1, par2 = inputs
        return individualRecoProb_forVAE_discrete_8(x, par1, par2, w)