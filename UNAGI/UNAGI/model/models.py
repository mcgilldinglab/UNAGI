'''
The module contains the VAE model and the discriminator model.
'''
import numpy as np
import gc
import anndata
import pandas as pd
from torch.nn.modules.module import Module
import scanpy as sc
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import KernelDensity
from sklearn import cluster
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm, entropy, multivariate_normal, gamma
from scipy import stats 
import torch
from pyro.primitives import param,deterministic
from torch.nn import functional as F
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.distributions.gamma import Gamma
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy.stats import entropy 
from scipy import stats
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import threading
from scipy.stats import multivariate_normal
import pyro
import pyro.distributions as dist
from ..dynamic_graphs.distDistance import *
from scvi.module.base import PyroBaseModuleClass
import gc
import math
from sklearn.neighbors import kneighbors_graph

TTT = 0
from torch.distributions import constraints
from torch.distributions.utils import (
    broadcast_all,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)
from torch.nn.functional import softplus

from pyro.distributions import TorchDistribution, LogNormal, Poisson, Gamma, Weibull, Chi2
from pyro.distributions.util import broadcast_shape
import torch
import torch.nn.functional as F
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all, probs_to_logits, lazy_property, logits_to_probs

from numbers import Number

import torch
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all
class ZeroInflatedDistribution(TorchDistribution):
    """
    Generic Zero Inflated distribution.

    This can be used directly or can be used as a base class as e.g. for
    :class:`ZeroInflatedPoisson` and :class:`ZeroInflatedNegativeBinomial`.

    :param TorchDistribution base_dist: the base distribution.
    :param torch.Tensor gate: probability of extra zeros given via a Bernoulli distribution.
    :param torch.Tensor gate_logits: logits of extra zeros given via a Bernoulli distribution.
    """

    arg_constraints = {
        "gate": constraints.unit_interval,
        "gate_logits": constraints.real,
    }

    def __init__(self, base_dist, *, gate=None, gate_logits=None, validate_args=None):
        if (gate is None) == (gate_logits is None):
            raise ValueError(
                "Either `gate` or `gate_logits` must be specified, but not both."
            )
        if gate is not None:
            batch_shape = broadcast_shape(gate.shape, base_dist.batch_shape)
            self.gate = gate.expand(batch_shape)
        else:
            batch_shape = broadcast_shape(gate_logits.shape, base_dist.batch_shape)
            self.gate_logits = gate_logits.expand(batch_shape)
        if base_dist.event_shape:
            raise ValueError(
                "ZeroInflatedDistribution expected empty "
                "base_dist.event_shape but got {}".format(base_dist.event_shape)
            )

        self.base_dist = base_dist.expand(batch_shape)
        event_shape = torch.Size()
        
        super().__init__(batch_shape, event_shape, validate_args=False)

    

    @lazy_property
    def gate(self):
      
        return logits_to_probs(self.gate_logits, is_binary=True)

    @lazy_property
    def gate_logits(self):
        return probs_to_logits(self.gate, is_binary=True)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        
        if "gate" in self.__dict__:

            gate, value = broadcast_all(self.gate, value)
            temp_value = value.clone()
            temp_value[temp_value==0] = 1e-7
            log_prob = (-gate).log1p() + self.base_dist.log_prob(temp_value)
            log_prob = torch.where(value == 0, (gate).log(), log_prob)
        else:
            gate_logits, value = broadcast_all(self.gate_logits, value)
            temp_value = value.clone()
            temp_value[temp_value==0] = 1e-7
            temp_base_log_prob= self.base_dist.log_prob(temp_value)
            log_prob_minus_log_gate = -gate_logits + temp_base_log_prob#self.base_dist.log_prob(temp_value)
            log_gate = -softplus(-gate_logits)
            
            log_prob = log_prob_minus_log_gate + log_gate
            zero_log_prob = log_gate#-0.5#softplus(log_prob_minus_log_gate) + log_gate#+1
            log_prob = torch.where(value == 0, zero_log_prob, log_prob)
        return log_prob


    def sample(self, sample_shape=torch.Size()):
        
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            mask = torch.bernoulli(self.gate.expand(shape)).bool()
            samples = self.base_dist.expand(shape).sample()
            samples = torch.where(mask, samples.new_zeros(()), samples)
        return samples


    @lazy_property
    def mean(self):
        return (1 - self.gate) * self.base_dist.mean

    @lazy_property
    def variance(self):
        return (1 - self.gate) * (
            self.base_dist.mean**2 + self.base_dist.variance
        ) - (self.mean) ** 2

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(type(self), _instance)
        batch_shape = torch.Size(batch_shape)
        gate = self.gate.expand(batch_shape) if "gate" in self.__dict__ else None
        gate_logits = (
            self.gate_logits.expand(batch_shape)
            if "gate_logits" in self.__dict__
            else None
        )
        base_dist = self.base_dist.expand(batch_shape)
        ZeroInflatedDistribution.__init__(
            new, base_dist, gate=gate, gate_logits=gate_logits, validate_args=False
        )
        new._validate_args = self._validate_args
        return new
class myZeroInflatedGamma(ZeroInflatedDistribution):
    """
    A Zero Inflated Gamma distribution.

    :param total_count: non-negative number of negative Bernoulli trials.
    :type total_count: float or torch.Tensor
    :param torch.Tensor probs: Event probabilities of success in the half open interval [0, 1).
    :param torch.Tensor logits: Event log-odds for probabilities of success.
    :param torch.Tensor gate: probability of extra zeros.
    :param torch.Tensor gate_logits: logits of extra zeros.
    """

    arg_constraints = {
        'loc': constraints.real, 'scale': constraints.positive,
        "gate": constraints.unit_interval,
        "gate_logits": constraints.real,
    }
   
    support = constraints.positive

    def __init__(
        self,
        loc,
        scale=None,
        gate=None,
        gate_logits=None,
        validate_args=None
    ):
        base_dist = Gamma(concentration=loc,rate=scale,validate_args=False)
        base_dist._validate_args = validate_args
        super().__init__(
            base_dist, gate=gate, gate_logits=gate_logits, validate_args=validate_args
        )

class myZeroInflatedLogNormal(ZeroInflatedDistribution):
    """
    A Zero Inflated Log Normal distribution.

    :param total_count: non-negative number of negative Bernoulli trials.
    :type total_count: float or torch.Tensor
    :param torch.Tensor probs: Event probabilities of success in the half open interval [0, 1).
    :param torch.Tensor logits: Event log-odds for probabilities of success.
    :param torch.Tensor gate: probability of extra zeros.
    :param torch.Tensor gate_logits: logits of extra zeros.
    """

    arg_constraints = {
        'loc': constraints.real, 'scale': constraints.positive,
        "gate": constraints.unit_interval,
        "gate_logits": constraints.real,
    }
   
    support = constraints.positive

    def __init__(
        self,
        loc,
        scale=None,
        gate=None,
        gate_logits=None,
        validate_args=None
    ):
        base_dist = LogNormal(loc=loc,scale=scale,validate_args=False)
        base_dist._validate_args = validate_args
        super().__init__(
            base_dist, gate=gate, gate_logits=gate_logits, validate_args=validate_args
        )
        
    
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # print(input.shape,self.weight.shape)
        # support = self.softplus(self.fc(input.float()))
        # support = self.BN2(support)
        
        # # print(support.shape)
        # support = torch.mm(input.float(), self.weight.float())
        # output = torch.spmm(adj, support)
        # print(torch.sum(input))

        support = torch.spmm(adj, input)
        # print(torch.sum(support))
        
        # output = self.fc(support)
        # output = torch.mm(support, self.weight.float())
        return support, support 

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
class PlainEncoder(nn.Module):
    '''
    The plain encoder of VAE, which encodes the single-cell data into the latent space.

    parameters:
    in_dim: the input dimension of the encoder.
    z_dim: the latent dimension of the encoder.
    hidden_dim: the GCN dimension of the encoder.

    return:
    z_loc: the mean of the latent space.
    z_scale: the variance of the latent space.
    '''
    def __init__(self, in_dim, z_dim, hidden_dim):
        super().__init__()
        
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        self.softplus = nn.Softplus()
    
    def forward(self, x,adj,  start, end, test=False):

        hidden = self.softplus(self.fc1(x))
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(torch.clamp(self.fc22(hidden), -5, 5))
        return z_loc, z_scale
class GraphEncoder(nn.Module):
    '''
    The GCN encoder of VAE, which encodes the single-cell data into the latent space.

    parameters:
    in_dim: the input dimension of the encoder.
    z_dim: the latent dimension of the encoder.
    hidden_dim: the GCN dimension of the encoder.

    return:
    z_loc: the mean of the latent space.
    z_scale: the variance of the latent space.
    
    '''
    def __init__(self, in_dim, z_dim, hidden_dim):
        super().__init__()
        
        self.gc1 = GraphConvolution(in_dim, hidden_dim)
        # self.gc1 = GraphConvolution(in_dim, hidden_dim)
        # self.dropout = dropout
        #self.fc1 = nn.Linear(nhid, 512)
        #self.fc2 = nn.Linear(512, hidden_dim)
        #self.fc3 = nn.Linear(256, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        #self.BN1 = nn.BatchNorm1d(hidden_dim)
        
        self.test = nn.Linear(in_dim, hidden_dim)
        #self.test1 = nn.Linear(hidden_dim, hidden_dim)
        self.softplus = nn.Softplus()

    def forward(self, x,adj,  start, end, test=False): 
        if test == False:
            y, x = self.gc1(x, adj)
            hidden = x.clone()
            # x = self.softplus(x)
            hidden1 = x
            # hidden=x[start:end,:] #temporarily remove to test full batch vs small batch
            # hidden = self.softplus(self.BN1(x)) #temporarily remove to test full batch vs small batch
            y=y[start:end,:]

            hidden1=x[start:end,:]
        else:
            y = x
            hidden1 = x.clone()
        
        
        hidden1 = self.softplus(self.test(hidden1))
        z_loc = self.fc21(hidden1)
        z_scale = torch.exp(torch.clamp(self.fc22(hidden1), -5, 5))
        #print(z_loc)
        return z_loc, z_scale
    
class Discriminator(nn.Module):
    '''
        The discriminator of UNAGI, which distinguishes the real data from the generated data.
    '''
    def __init__(self,input_dim,hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        #self.fc3 = nn.Linear(64, 1)
        self.ReLU6 = nn.Softplus()
        self.BN1 = nn.BatchNorm1d(hidden_dim)
        
        self.bce = nn.BCEWithLogitsLoss()
    def forward(self, x, y):
        ys = y
        xs = x
        xs = self.ReLU6(self.fc1(xs))#+1e-6

        #print(xs)
        #xs = self.ReLU6(self.fc2(xs))+1e-6
        xs = self.fc2(xs)

        #ys = torch.LongTensor(ys)
        return self.bce(xs,ys)
    
class Decoder(nn.Module):
    def __init__(self, in_dim, z_dim, hidden_dim):
        super().__init__()

        self.fc1 = nn.Linear(z_dim, hidden_dim)
        #self.fc2 = nn.Linear(512, hidden_dim)
        #self.test = nn.Linear(hidden_dim,hidden_dim)
        #self.fc3 = nn.Linear(512, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, in_dim)
        self.fc22 = nn.Linear(hidden_dim, in_dim)
        self.fc23 = nn.Linear(hidden_dim, in_dim)
        #self.fc24 = nn.Linear(hidden_dim, in_dim)
        self.softplus = nn.Softplus()
        self.ReLU6 = nn.ReLU6()
        self.ReLU = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, z):

        hidden =self.softplus(self.fc1(z))#+1e-7
        

        loc =  self.fc21(hidden) #mu

        scale = self.softplus(self.fc22(hidden))
        dropout = self.sigmoid(torch.clamp(self.fc23(hidden),-3,3))
        return loc, scale, dropout
    
class VAE(PyroBaseModuleClass):
    '''
    The VAE model of UNAGI, which contains the encoder and decoder of VAE.
    '''
    def __init__(self,  n_input, n_latent,n_graph,beta,distribution): 
        super().__init__()
        self.n_latent = n_latent
        self.n_input = n_input
        self.n_graph = n_graph
        self.beta = beta
        self.distribution = distribution
       # in the init, we create the parameters of our elementary stochastic computation unit.
       
        # First, we setup the parameters of the generative model
        self.decoder = Decoder(self.n_input,self.n_latent,self.n_graph)
        self.log_theta = torch.nn.Parameter(torch.randn(self.n_input))
        self.gate_logits = torch.nn.Parameter(torch.randn(self.n_input))
        self.discriminator = Discriminator(self.n_input, self.n_latent)
        # Second, we setup the parameters of the variational distribution
        self.encoder = GraphEncoder(self.n_input, self.n_latent,self.n_graph)
        # self.gc = GraphConvolution(n_input, nhid)
        # self.softplus = nn.Sigmoid()
        
    def model(self, x,adj,batch,  start, end):
        '''
        define the model p(x|z)p(z) using the generative network p(x|z) and prior p(z) = N(0,1). The distribution of x is chosen from the `dist` parameter in the `setup_training` function. The default is `ziln` (zero-inflated log normal).

        parameters:
        x: the single-cell data.
        adj: the cell graph.
        batch: the batch information of the single-cell data.
        start: the start index of the single-cell data.
        end: the end index of the single-cell data.

        '''
        # register PyTorch module `decoder` with Pyro p(z) p(x|z)
        pyro.module("decoder", self)
        with pyro.plate("data", x[start:end,:].shape[0]):

            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x[start:end,:].shape[0], self.n_latent)))
            z_scale = x.new_ones(torch.Size((x[start: end,:].shape[0], self.n_latent)))
            # z_p = Variable(torch.randn(64,128))
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # get the "normalized" mean of the negative binomial
            if self.distribution == 'ziln':
                dec_loc, dec_mu, dec_dropout  = self.decoder(z)
                scale = torch.exp(self.log_theta)
                loc = (dec_mu+1e-5).log()-scale.pow(2)/2
                x_dist = myZeroInflatedLogNormal(loc=loc, scale=scale, gate=dec_dropout)
                # score against actual counts
                rx=pyro.sample("obs", x_dist.to_event(1), obs=x[start:end,:])
                a = deterministic('recon',x_dist.mean)
            elif self.distribution == 'zinb':
                _, mu, dec_dropout  = self.decoder(z)
                theta = self.log_theta.exp()
                nb_logits = (mu+1e-4).log() - (theta+1e-4).log()
            
                x_dist = dist.ZeroInflatedNegativeBinomial(total_count=theta, logits=nb_logits,gate= dec_dropout, validate_args=False)
                # score against actual counts
                rx=pyro.sample("obs", x_dist.to_event(1), obs=x[start:end,:])
                a = deterministic('recon',x_dist.mean)
            elif self.distribution =='nb':
                _, mu, dec_dropout  = self.decoder(z)
                theta = self.log_theta.exp()
                nb_logits = (mu+1e-4).log() - (theta+1e-4).log()
            
                x_dist = dist.NegativeBinomial(total_count=theta, logits=nb_logits, validate_args=False)
                # score against actual counts
                rx=pyro.sample("obs", x_dist.to_event(1), obs=x[start:end,:])
                a = deterministic('recon',x_dist.mean)

            elif self.distribution =='zig':
                _, mu, dec_dropout = self.decode(z)
                scale = F.softplus(self.log_theta)#self.log_theta.exp()
                loc = mu*scale
                rx = myZeroInflatedGamma(loc=loc, scale=scale, gate=dec_dropout, validate_args=False)
                a = deterministic('recon',rx.mean)
            else:
                raise ValueError('distribution not supported')
            return rx
        
    def guide(self, x, adj, batch, start, end):
        '''
        define the guide (i.e. variational distribution) q(z|x)
        '''
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self)
        
        with pyro.plate("data", x[start:end,:].shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            #x_ = torch.log(1 + x)
            [qz_m,qz_v] = self.encoder(x, adj,  start, end)
          
            #qz_v = self.var_encoder(x_)
            # sample the latent code z
            rz=pyro.sample("latent", dist.Normal(qz_m, qz_v).to_event(1))
            return rz
        
    def getZ(self, x,adj,batch, start, end,test=True):
        '''
        Get the latent space of the single-cell data.

        parameters:
        x: the single-cell data.
        adj: the cell graph.
        batch: the batch information of the single-cell data.
        start: the start index of the single-cell data.
        end: the end index of the single-cell data.
        test: whether to use the test mode. Default is True.
        '''
        # x = torch.log(1+x)
        z_loc, z_scale = self.encoder(x,adj,  start, end,test=test)
        # sample in latent space
        # z = dist.Normal(z_loc, z_scale).sample()
        
        return z_loc+z_scale,z_loc, z_scale

    def generate1(self,x,adj,batch, start, end):
        '''
        Reconstruct the single-cell data from the latent space given inputs.

        parameters:
        x: the single-cell data.
        adj: the cell graph.
        batch: the batch information of the single-cell data.
        start: the start index of the single-cell data.
        end: the end index of the single-cell data.

        return:
        rx: the reconstructed single-cell data.
        '''
        # x = torch.log(1 + x)
        z_loc, z_scale = self.encoder(x,adj,batch,  start, end,test=True)
        z = dist.Normal(z_loc, z_scale).sample()
        if self.distribution == 'ziln':
            dec_loc, dec_mu, dec_dropout  = self.decoder(z_loc+z_scale)
            scale = torch.exp(self.log_theta)
            loc = (dec_mu+1e-5).log()-scale.pow(2)/2
            x_dist = myZeroInflatedLogNormal(loc=loc, scale=scale, gate=dec_dropout)
            rx=x_dist.mean #-> expectation
        elif self.distribution == 'zinb':
            _, mu, dec_dropout  = self.decoder(z_loc+z_scale)
            theta = self.log_theta.exp()
            nb_logits = (mu+1e-4).log() - (theta+1e-4).log()
            x_dist = dist.ZeroInflatedNegativeBinomial(total_count=theta, logits=nb_logits,gate= dec_dropout, validate_args=False)
            rx=x_dist.mean
        elif self.distribution =='zig':
            _, mu, dec_dropout = self.decoder(z_loc+z_scale)
            scale = F.softplus(self.log_theta)
            loc = mu*scale
            x_dist = myZeroInflatedGamma(loc=loc, scale=scale, gate=dec_dropout, validate_args=False)
            rx = x_dist.mean
        # rx=x_dist.sample() #-> sample
        elif self.distribution =='nb':
            _, mu, dec_dropout  = self.decoder(z_loc+z_scale)
            theta = self.log_theta.exp()
            nb_logits = (mu+1e-4).log() - (theta+1e-4).log()
            x_dist = dist.NegativeBinomial(total_count=theta, logits=nb_logits, validate_args=False)
            rx=x_dist.mean
        else:
            raise ValueError('distribution not supported')
        return  rx#, None, None,None, None
    
class VAE_no_GCN(PyroBaseModuleClass):
    '''
    The VAE model without GCN, which contains the encoder and decoder of VAE.
    '''
    def __init__(self,  n_input, n_latent,n_graph,beta,distribution): 
        super().__init__()
        self.n_latent = n_latent
        self.n_input = n_input
        self.n_graph = n_graph
        self.beta = beta
        self.distribution = distribution
       # in the init, we create the parameters of our elementary stochastic computation unit.
       
        # First, we setup the parameters of the generative model
        self.decoder = Decoder(self.n_input,self.n_latent,self.n_graph)
        self.log_theta = torch.nn.Parameter(torch.randn(self.n_input))
        self.gate_logits = torch.nn.Parameter(torch.randn(self.n_input))
        # Second, we setup the parameters of the variational distribution
        self.encoder = PlainEncoder(self.n_input, self.n_latent,self.n_graph)
        # self.gc = GraphConvolution(n_input, nhid)
        # self.softplus = nn.Sigmoid()
        
    def model(self, x,adj,batch,  start, end):
        '''
        define the model p(x|z)p(z) using the generative network p(x|z) and prior p(z) = N(0,1). The distribution of x is chosen from the `dist` parameter in the `setup_training` function. The default is `ziln` (zero-inflated log normal).

        parameters:
        x: the single-cell data.
        adj: the cell graph.
        batch: the batch information of the single-cell data.
        start: the start index of the single-cell data.
        end: the end index of the single-cell data.

        '''
        # register PyTorch module `decoder` with Pyro p(z) p(x|z)
        pyro.module("decoder", self)
        with pyro.plate("data", x[start:end,:].shape[0]):

            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x[start:end,:].shape[0], self.n_latent)))
            z_scale = x.new_ones(torch.Size((x[start: end,:].shape[0], self.n_latent)))
            # z_p = Variable(torch.randn(64,128))
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # get the "normalized" mean of the negative binomial
            if self.distribution == 'ziln':
                dec_loc, dec_mu, dec_dropout  = self.decoder(z)
                scale = torch.exp(self.log_theta)
                loc = (dec_mu+1e-5).log()-scale.pow(2)/2
                x_dist = myZeroInflatedLogNormal(loc=loc, scale=scale, gate=dec_dropout)
                # score against actual counts
                rx=pyro.sample("obs", x_dist.to_event(1), obs=x[start:end,:])
                a = deterministic('recon',x_dist.mean)
            elif self.distribution == 'zinb':
                _, mu, dec_dropout  = self.decoder(z)
                theta = self.log_theta.exp()
                nb_logits = (mu+1e-4).log() - (theta+1e-4).log()
            
                x_dist = dist.ZeroInflatedNegativeBinomial(total_count=theta, logits=nb_logits,gate= dec_dropout, validate_args=False)
                # score against actual counts
                rx=pyro.sample("obs", x_dist.to_event(1), obs=x[start:end,:])
                a = deterministic('recon',x_dist.mean)
            elif self.distribution =='nb':
                _, mu, dec_dropout  = self.decoder(z)
                theta = self.log_theta.exp()
                nb_logits = (mu+1e-4).log() - (theta+1e-4).log()
            
                x_dist = dist.NegativeBinomial(total_count=theta, logits=nb_logits, validate_args=False)
                # score against actual counts
                rx=pyro.sample("obs", x_dist.to_event(1), obs=x[start:end,:])
                a = deterministic('recon',x_dist.mean)

            elif self.distribution =='zig':
                _, mu, dec_dropout = self.decode(z)
                scale = F.softplus(self.log_theta)#self.log_theta.exp()
                loc = mu*scale
                rx = myZeroInflatedGamma(loc=loc, scale=scale, gate=dec_dropout, validate_args=False)
                a = deterministic('recon',rx.mean)
            else:
                raise ValueError('distribution not supported')
            return rx
        
    def guide(self, x, adj, batch, start, end):
        '''
        define the guide (i.e. variational distribution) q(z|x)
        '''
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self)
        
        with pyro.plate("data", x[start:end,:].shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            #x_ = torch.log(1 + x)
            [qz_m,qz_v] = self.encoder(x, adj,  start, end)
          
            #qz_v = self.var_encoder(x_)
            # sample the latent code z
            rz=pyro.sample("latent", dist.Normal(qz_m, qz_v).to_event(1))
            return rz
        
    def getZ(self, x,adj,batch, start, end,test=True):
        '''
        Get the latent space of the single-cell data.

        parameters:
        x: the single-cell data.
        adj: the cell graph.
        batch: the batch information of the single-cell data.
        start: the start index of the single-cell data.
        end: the end index of the single-cell data.
        test: whether to use the test mode. Default is True.
        '''
        # x = torch.log(1+x)
        z_loc, z_scale = self.encoder(x,adj,  start, end,test=test)
        # sample in latent space
        # z = dist.Normal(z_loc, z_scale).sample()
        
        return z_loc+z_scale,z_loc, z_scale

    def generate1(self,x,adj,batch, start, end):
        '''
        Reconstruct the single-cell data from the latent space given inputs.

        parameters:
        x: the single-cell data.
        adj: the cell graph.
        batch: the batch information of the single-cell data.
        start: the start index of the single-cell data.
        end: the end index of the single-cell data.

        return:
        rx: the reconstructed single-cell data.
        '''
        # x = torch.log(1 + x)
        z_loc, z_scale = self.encoder(x,adj,batch,  start, end,test=True)
        z = dist.Normal(z_loc, z_scale).sample()
        if self.distribution == 'ziln':
            dec_loc, dec_mu, dec_dropout  = self.decoder(z_loc+z_scale)
            scale = torch.exp(self.log_theta)
            loc = (dec_mu+1e-5).log()-scale.pow(2)/2
            x_dist = myZeroInflatedLogNormal(loc=loc, scale=scale, gate=dec_dropout)
            rx=x_dist.mean #-> expectation
        elif self.distribution == 'zinb':
            _, mu, dec_dropout  = self.decoder(z_loc+z_scale)
            theta = self.log_theta.exp()
            nb_logits = (mu+1e-4).log() - (theta+1e-4).log()
            x_dist = dist.ZeroInflatedNegativeBinomial(total_count=theta, logits=nb_logits,gate= dec_dropout, validate_args=False)
            rx=x_dist.mean
        elif self.distribution =='zig':
            _, mu, dec_dropout = self.decoder(z_loc+z_scale)
            scale = F.softplus(self.log_theta)
            loc = mu*scale
            x_dist = myZeroInflatedGamma(loc=loc, scale=scale, gate=dec_dropout, validate_args=False)
            rx = x_dist.mean
        # rx=x_dist.sample() #-> sample
        elif self.distribution =='nb':
            _, mu, dec_dropout  = self.decoder(z_loc+z_scale)
            theta = self.log_theta.exp()
            nb_logits = (mu+1e-4).log() - (theta+1e-4).log()
            x_dist = dist.NegativeBinomial(total_count=theta, logits=nb_logits, validate_args=False)
            rx=x_dist.mean
        else:
            raise ValueError('distribution not supported')
        return  rx#, None, None,None, None