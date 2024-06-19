import torch
from torch.nn import functional as F
from torch import nn
from .distributions import ZeroInflatedGamma, ZeroInflatedLogNormal, ZeroInflatedExponential
from pyro.distributions.zero_inflated import ZeroInflatedNegativeBinomial

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, use_bias=True):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(torch.zeros(size=(in_features, out_features))))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(torch.zeros(size=(out_features,))))
        else:
            self.register_parameter('bias', None)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        x = x @ self.weight
        if self.bias is not None:
            x += self.bias

        return torch.sparse.mm(adj, x)
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.sigmoid(self.fc3(x))
class Plain_encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, graph_dim, latent_dim):
        super(Plain_encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.graph_dim = graph_dim
        self.fc0 = nn.Linear(input_dim, graph_dim)
        self.fc1 = nn.Linear(graph_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        self.BN = nn.BatchNorm1d(graph_dim)
        self.BN1 = nn.BatchNorm1d(hidden_dim)
    def forward(self, x, idx=None):
        if idx is not None:
            h1 = F.softplus(self.BN(self.fc0(x[idx,:])))
            h1 = F.softplus(self.BN1(self.fc1(h1)))
        else:
            h1 = F.softplus(self.BN(self.fc0(x)))
            h1 = F.softplus(self.BN1(self.fc1(h1)))
        # return self.fc21(h1), torch.sqrt(F.softplus(self.fc22(h1)))
        return self.fc21(h1), self.fc22(h1)
class Graph_encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, graph_dim, latent_dim):
        super(Graph_encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.graph_dim = graph_dim
        self.fc_graph = GCNLayer(input_dim, graph_dim)
        self.fc1 = nn.Linear(graph_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        self.BN = nn.BatchNorm1d(graph_dim)
        self.BN1 = nn.BatchNorm1d(hidden_dim)
    def forward(self, x, adj,idx=None):
        if idx is not None:
            # h1 = F.softplus(self.fc0(x[idx,:]))
            h0 = F.softplus(self.BN(self.fc_graph(x,adj)))
            h0 = h0[idx,:]
            
            h1 = F.softplus(self.fc1(h0))
        else:
            # h1 = F.softplus(self.fc0(x))
            h0 = F.softplus(self.BN(self.fc_graph(x,adj)))
            # print(h0)
            h1 = F.softplus(self.BN1(self.fc1(h0)))
        # return self.fc21(h1), torch.sqrt(F.softplus(self.fc22(h1)))
        return self.fc21(h1), self.fc22(h1)
class VAE_decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim,distribution):
        super(VAE_decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.distribution = distribution
        self.latent_dim = latent_dim
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        self.fc5 = nn.Linear(hidden_dim, input_dim)
    def forward(self, z):
        h3 = F.softplus(self.fc3(z))
        if self.distribution == 'zig':
            mu = self.fc4(h3)
            dropout_logits = self.fc5(h3)
            return  F.softplus(torch.clamp(mu,min=-1)), dropout_logits
        elif self.distribution == 'ziln':
            mu = self.fc4(h3)
            dropout_logits = self.fc5(h3)
            dropout_logits = torch.clamp(dropout_logits,-3,3)
            return  F.softplus(torch.clamp(mu,min=-1)), dropout_logits
        elif self.distribution == 'zinb':
            mu = self.fc4(h3)
            dropout_logits = self.fc5(h3)
            return  torch.exp(mu), dropout_logits
        elif self.distribution == 'zie':
            mu = self.fc4(h3)
            dropout_logits = self.fc5(h3)
            return  F.softplus(mu), dropout_logits
class Plain_VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, graph_dim, latent_dim,beta=1,distribution='zinb'):
        super(Plain_VAE, self).__init__()
        self.input_dim = input_dim
        self.beta = beta
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.graph_dim = graph_dim
        self.encoder = Plain_encoder(input_dim, hidden_dim, graph_dim, latent_dim)
        self.decoder = VAE_decoder(input_dim, hidden_dim, latent_dim,distribution)
        self.distribution = distribution
        self.log_theta = torch.nn.Parameter(torch.rand(input_dim))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x,idx=None):
        # x = torch.log(x+1)

        mu, logvar = self.encoder(x.view(-1, self.input_dim),idx)
        z = self.reparameterize(mu, logvar)
        #zinb distribution
        
        de_mean, de_dropout = self.decoder(z)
        
        recons = self.generate_inside(de_mean, de_dropout)
        return de_mean, de_dropout, mu, logvar, recons
    

    def get_latent_representation(self, x,idx=None):
        mu, logvar = self.encoder(x.view(-1, self.input_dim),idx)
        return mu+torch.exp(0.5 * logvar)
    # def getnerate
    def generate_inside(self, mean, dropout_logits):
        # scale = F.softplus(self.log_theta)#self.log_theta.exp()
        # loc = (mean+1e-5).log()-scale.pow(2)/2
        if self.distribution == 'zig':
            scale = F.softplus(self.log_theta)#self.log_theta.exp()
            loc = mean*scale
            distribution = ZeroInflatedGamma(loc=loc, scale=scale, gate_logits=dropout_logits, validate_args=False)
        elif self.distribution == 'ziln':
            scale = F.softplus(self.log_theta)#self.log_theta.exp()
            loc = (mean+1e-5).log()- (scale).pow(2)/2
            distribution = ZeroInflatedLogNormal(loc=loc, scale=scale, gate_logits=dropout_logits, validate_args=False)
        elif self.distribution == 'zinb':
            theta = F.softplus(self.log_theta)
            nb_logits = (mean+1e-5).log()-(theta+1e-5).log()
            distribution = ZeroInflatedNegativeBinomial(total_count=theta, logits=nb_logits,gate_logits = dropout_logits, validate_args=False)
        elif self.distribution == 'zie':
            loc = 1/(mean+1e-5)
            distribution = ZeroInflatedExponential(rate=loc, gate_logits=dropout_logits, validate_args=False)
        return distribution.mean
    def decode(self, z):
        mu, dropout_logits = self.decoder(z)
        # scale = F.softplus(self.log_theta)#self.log_theta.exp()
        # loc = (mu+1e-5).log()-scale.pow(2)/2
        if self.distribution == 'zig':
            scale = F.softplus(self.log_theta)#self.log_theta.exp()
            loc = mu*scale
            distribution = ZeroInflatedGamma(loc=loc, scale=scale, gate_logits=dropout_logits, validate_args=False)
        elif self.distribution == 'ziln':
            scale = F.softplus(self.log_theta)
            loc = (mu+1e-5).log()- (scale+1e-5).pow(2)/2
            distribution = ZeroInflatedLogNormal(loc=loc, scale=scale, gate_logits=dropout_logits, validate_args=False)
        elif self.distribution == 'zinb':
            theta = F.softplus(self.log_theta)
            nb_logits = (mu+1e-5).log()-(theta+1e-5).log()
            distribution = ZeroInflatedNegativeBinomial(total_count=theta, logits=nb_logits,gate_logits = dropout_logits, validate_args=False)
        elif self.distribution == 'zie':
            loc = 1/(mu+1e-5)
            distribution = ZeroInflatedExponential(rate=loc, gate_logits=dropout_logits, validate_args=False)
        return distribution.mean #return the mean of zinb distribution
    def generate(self, x,sample_shape=None,random=False):
        '''
        generate samples from the model
        sample_shape: shape of sample
        '''
        mu, logvar = self.encoder(x.view(-1, self.input_dim))
        if random:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu+torch.exp(0.5 * logvar) #not using reparameterize
        mu, dropout_logits = self.decoder(z)
        # scale = F.softplus(self.log_theta)#self.log_theta.exp()
        # loc = (mu+1e-5).log()-scale.pow(2)/2
        if self.distribution == 'zig':
            scale = F.softplus(self.log_theta)#self.log_theta.exp()
            loc = mu*scale
            distribution = ZeroInflatedGamma(loc=loc, scale=scale, gate_logits=dropout_logits, validate_args=False)
        elif self.distribution == 'ziln':
            scale = F.softplus(self.log_theta)
            loc = (mu+1e-5).log()- (scale+1e-5).pow(2)/2
            distribution = ZeroInflatedLogNormal(loc=loc, scale=scale, gate_logits=dropout_logits, validate_args=False)
        elif self.distribution == 'zinb':
            theta = F.softplus(self.log_theta)
            nb_logits = (mu+1e-5).log()-(theta+1e-5).log()
            distribution = ZeroInflatedNegativeBinomial(total_count=theta, logits=nb_logits,gate_logits = dropout_logits, validate_args=False)
        elif self.distribution == 'zie':
            loc = 1/(mu+1e-5)
            distribution = ZeroInflatedExponential(rate=loc, gate_logits=dropout_logits, validate_args=False)
        # theta = F.softplus(self.log_theta)#self.log_theta.exp()

        # nb_logits = (mu+1e-5).log() - (theta+1e-5).log()
        # distribution = ZeroInflatedNegativeBinomial(total_count=theta, logits=nb_logits,gate_logits = dropout_logits, validate_args=False)

        if random:
            return distribution.sample(sample_shape) #return the sample of zinb distribution
        else:
            return distribution.mean #return the mean of zinb distribution
    def kl_d(self, mu, logvar):
        return (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),dim=-1))#/len(mu)
    def reconstruction_loss(self, x, mu, dropout_logits,gene_weights=None):
        '''
        x: input data
        mu: output of decoder
        dropout_logits: dropout logits of zinb distribution
        gene_weights: weights of genes
        '''
        if self.distribution == 'zig':

            scale = F.softplus(self.log_theta)#self.log_theta.exp()
            loc = mu * scale
            distribution = ZeroInflatedGamma(loc=loc, scale=scale, gate_logits=dropout_logits, validate_args=False)
        elif self.distribution == 'ziln':
            scale = F.softplus(self.log_theta)
            loc = (mu+1e-5).log()- (scale).pow(2)/2
            distribution = ZeroInflatedLogNormal(loc=loc, scale=scale, gate_logits=dropout_logits, validate_args=False)
        elif self.distribution == 'zinb':
            theta = F.softplus(self.log_theta)
            nb_logits = (mu+1e-5).log()-(theta+1e-5).log()
            distribution = ZeroInflatedNegativeBinomial(total_count=theta, logits=nb_logits,gate_logits = dropout_logits, validate_args=False)
        elif self.distribution == 'zie':
            loc = 1/(mu+1e-5)
            distribution = ZeroInflatedExponential(rate=loc, gate_logits=dropout_logits, validate_args=False)
        if gene_weights is not None:
            return (distribution.log_prob(x)*gene_weights).sum(-1)
        return distribution.log_prob(x).sum(-1)
    def loss_function(self, x, mu, dropout_logits, mu_, logvar_,gene_weights=None):
        reconstruction_loss = self.reconstruction_loss(x, mu, dropout_logits,gene_weights=gene_weights)
        kl_div = self.kl_d(mu_, logvar_)
        # print(-torch.mean(reconstruction_loss), torch.mean(kl_div))
        return -(torch.mean(reconstruction_loss,dim=0) - torch.mean(self.beta * kl_div,dim=0))
    
class VAE(Plain_VAE):
    def __init__(self, input_dim, hidden_dim, graph_dim, latent_dim,beta=1,distribution='zinb'):
        super(Plain_VAE, self).__init__()
        self.input_dim = input_dim
        self.beta = beta
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.graph_dim = graph_dim
        self.encoder = Graph_encoder(input_dim, hidden_dim, graph_dim, latent_dim)
        self.decoder = VAE_decoder(input_dim, hidden_dim, latent_dim,distribution)
        self.distribution = distribution
        self.log_theta = torch.nn.Parameter(torch.rand(input_dim))

    # def reparameterize(self, mu, logvar):
    #     std = torch.exp(0.5 * logvar)
    #     eps = torch.randn_like(std)
    #     return mu + eps*std

    def forward(self, x,adj,idx=None):
        # x = torch.log(x+1)

        mu, logvar = self.encoder(x.view(-1, self.input_dim),adj,idx)
        z = self.reparameterize(mu, logvar)
        #zinb distribution
        
        de_mean, de_dropout = self.decoder(z)
        
        recons = self.generate_inside(de_mean, de_dropout)
        return de_mean, de_dropout, mu, logvar, recons
    

    def get_latent_representation(self, x,adj,idx=None):
        mu, logvar = self.encoder(x.view(-1, self.input_dim),adj,idx)
        return mu+torch.exp(0.5 * logvar)

    # def generate_inside(self, mean, dropout_logits):
    #     # scale = F.softplus(self.log_theta)#self.log_theta.exp()
    #     # loc = (mean+1e-5).log()-scale.pow(2)/2
    #     if self.distribution == 'zig':
    #         scale = F.softplus(self.log_theta)#self.log_theta.exp()
    #         loc = mean*scale
    #         distribution = ZeroInflatedGamma(loc=loc, scale=scale, gate_logits=dropout_logits, validate_args=False)
    #     elif self.distribution == 'ziln':
    #         scale = F.softplus(self.log_theta)#self.log_theta.exp()
    #         loc = (mean+1e-5).log()- (scale).pow(2)/2
    #         distribution = ZeroInflatedLogNormal(loc=loc, scale=scale, gate_logits=dropout_logits, validate_args=False)
    #     elif self.distribution == 'zinb':
    #         theta = F.softplus(self.log_theta)
    #         nb_logits = (mean+1e-5).log()-(theta+1e-5).log()
    #         distribution = ZeroInflatedNegativeBinomial(total_count=theta, logits=nb_logits,gate_logits = dropout_logits, validate_args=False)
    #     return distribution.mean
    # def decode(self, z):
    #     mu, dropout_logits = self.decoder(z)
    #     # scale = F.softplus(self.log_theta)#self.log_theta.exp()
    #     # loc = (mu+1e-5).log()-scale.pow(2)/2
    #     if self.distribution == 'zig':
    #         scale = F.softplus(self.log_theta)#self.log_theta.exp()
    #         loc = mu*scale
    #         distribution = ZeroInflatedGamma(loc=loc, scale=scale, gate_logits=dropout_logits, validate_args=False)
    #     elif self.distribution == 'ziln':
    #         scale = F.softplus(self.log_theta)
    #         loc = (mu+1e-5).log()- (scale+1e-5).pow(2)/2
    #         distribution = ZeroInflatedLogNormal(loc=loc, scale=scale, gate_logits=dropout_logits, validate_args=False)
    #     elif self.distribution == 'zinb':
    #         theta = F.softplus(self.log_theta)
    #         nb_logits = (mu+1e-5).log()-(theta+1e-5).log()
    #         distribution = ZeroInflatedNegativeBinomial(total_count=theta, logits=nb_logits,gate_logits = dropout_logits, validate_args=False)
    #     return distribution.mean #return the mean of zinb distribution
    def generate(self, x, adj,sample_shape=None,random=False):
        '''
        generate samples from the model
        sample_shape: shape of sample
        '''
        mu, logvar = self.encoder(x.view(-1, self.input_dim),adj)
        if random:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu+torch.exp(0.5 * logvar) #not using reparameterize
        mu, dropout_logits = self.decoder(z)
        # scale = F.softplus(self.log_theta)#self.log_theta.exp()
        # loc = (mu+1e-5).log()-scale.pow(2)/2
        if self.distribution == 'zig':
            scale = F.softplus(self.log_theta)#self.log_theta.exp()
            loc = mu*scale
            distribution = ZeroInflatedGamma(loc=loc, scale=scale, gate_logits=dropout_logits, validate_args=False)
        elif self.distribution == 'ziln':
            scale = F.softplus(self.log_theta)
            loc = (mu+1e-5).log()- (scale+1e-5).pow(2)/2
            distribution = ZeroInflatedLogNormal(loc=loc, scale=scale, gate_logits=dropout_logits, validate_args=False)
        elif self.distribution == 'zinb':
            theta = F.softplus(self.log_theta)
            nb_logits = (mu+1e-5).log()-(theta+1e-5).log()
            distribution = ZeroInflatedNegativeBinomial(total_count=theta, logits=nb_logits,gate_logits = dropout_logits, validate_args=False)
        elif self.distribution == 'zie':
            loc = 1/(mu+1e-5)
            distribution = ZeroInflatedExponential(rate=loc, gate_logits=dropout_logits, validate_args=False)
        # theta = F.softplus(self.log_theta)#self.log_theta.exp()

        # nb_logits = (mu+1e-5).log() - (theta+1e-5).log()
        # distribution = ZeroInflatedNegativeBinomial(total_count=theta, logits=nb_logits,gate_logits = dropout_logits, validate_args=False)

        if random:
            return distribution.sample(sample_shape) #return the sample of zinb distribution
        else:
            return distribution.mean #return the mean of zinb distribution
    # def kl_d(self, mu, logvar):
    #     return (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))/len(mu)
    # def reconstruction_loss(self, x, mu, dropout_logits,gene_weights=None):
    #     '''
    #     x: input data
    #     mu: output of decoder
    #     dropout_logits: dropout logits of zinb distribution
    #     gene_weights: weights of genes
    #     '''
    #     if self.distribution == 'zig':

    #         scale = F.softplus(self.log_theta)#self.log_theta.exp()
    #         loc = mu * scale
    #         distribution = ZeroInflatedGamma(loc=loc, scale=scale, gate_logits=dropout_logits, validate_args=False)
    #     elif self.distribution == 'ziln':
    #         scale = F.softplus(self.log_theta)
    #         loc = (mu+1e-5).log()- (scale).pow(2)/2
    #         distribution = ZeroInflatedLogNormal(loc=loc, scale=scale, gate_logits=dropout_logits, validate_args=False)
    #     elif self.distribution == 'zinb':
    #         theta = F.softplus(self.log_theta)
    #         nb_logits = (mu+1e-5).log()-(theta+1e-5).log()
    #         distribution = ZeroInflatedNegativeBinomial(total_count=theta, logits=nb_logits,gate_logits = dropout_logits, validate_args=False)
    #     if gene_weights is not None:
    #         return (distribution.log_prob(x)*gene_weights).sum(-1)
    #     return distribution.log_prob(x).sum(-1)
    # def loss_function(self, x, mu, dropout_logits, mu_, logvar_,gene_weights=None):
    #     reconstruction_loss = self.reconstruction_loss(x, mu, dropout_logits,gene_weights=gene_weights)
    #     kl_div = self.kl_d(mu_, logvar_)
    #     # print(-torch.mean(reconstruction_loss), torch.mean(kl_div))
    #     return -torch.mean(reconstruction_loss - self.beta * kl_div)