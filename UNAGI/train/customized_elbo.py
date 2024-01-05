'''
The customized elbo to support the VAE and discriminator loss.
'''
from pyro import poutine
import pyro.ops.jit
import numpy as np
from torch.nn.parameter import Parameter
from pyro.infer.util import is_validation_enabled
from pyro.infer.enum import get_importance_trace
from pyro.poutine.util import prune_subsample_sites
from pyro.distributions.util import is_identically_zero, scale_and_mask
from pyro.infer.elbo import ELBO
from pyro.util import check_model_guide_match, check_site_shape, ignore_jit_warnings, warn_if_nan, warn_if_inf
from pyro.infer.util import (
    is_validation_enabled,
    torch_item,
)
import torch
from pyro.util import check_if_enumerated, warn_if_nan

def graphUpdater(loss, model, guide,discriminator, optim,x,adj,i, start,end,device='cuda:0',second_optimizer=None,two=False):
    '''
    updater of Graph VAE-GAN.

    parameters
    --------------
    loss: 
        loss function
    model: 
        VAE model
    guide: 
        guide function of the model
    discriminator: 
        adversarial discriminator model
    optim: 
        optimizer
    x: 
        gene expression data
    adj: 
        cell graph
    i: 
        index of the batch
    start: 
        start index of the batch
    end: 
        end index of the batch
    device: 
        device to run the model
    second_optimizer: 
        optimizer for the discriminator
    two: 
        whether to return the loss of the VAE and the discriminator separately

    return
    ---------------
    loss: np.float
        loss of the VAE
    loss_discriminator: np.float
        loss of the discriminator

    '''
    
    with poutine.trace(param_only=True) as param_capture:
        loss_vae,surrogate_loss_particle,loss_discriminator = loss.loss_and_grads(model, guide,discriminator, device, x, adj,i, start,end)
        surrogate_loss_particle.backward()
    params = set(site["value"].unconstrained() for site in param_capture.trace.nodes.values()  if 'discriminator' not in site['name']  and 'recon' not in site['name'])
    optim(params)
    
    # zero gradients
    pyro.infer.util.zero_grads(params)
   
    if second_optimizer is not None:
        optim = second_optimizer
    with poutine.trace(param_only=True) as param_capture:
        loss_vae,surrogate_loss_particle,loss_discriminator = loss.loss_and_grads(model, guide,discriminator, device, x, adj,i, start,end)
    params = set(site["value"].unconstrained() for site in param_capture.trace.nodes.values() if 'discriminator' in site['name'] and 'recon' not in site['name'])
    loss_discriminator.backward()
    optim(params)
    # zero gradients
    pyro.infer.util.zero_grads(params)
    loss = loss_vae
    loss+=loss_discriminator
    #print(loss_discriminator)
#     loss = loss_vae
    if two == True:
        return torch_item(loss_vae), torch_item(loss_discriminator)
    if isinstance(loss, tuple):
        return type(loss)(map(torch_item, loss))
    else:
        return torch_item(loss)
    
class myELBO(ELBO):
    '''
    The customized ELBO function for the VAE-GAN model. The ELBO function is modified to include the discriminator loss.

    parameters
    ----------------

    geneWeight: torch.tensor
        The weight of the gene expression data. Default is None.

    pushback_Score: torch.tensor
        The pushback score for the discriminator. Default is None.

    '''
    def __init__(self, geneWeight=None, pushback_Score = None):
    
        super(myELBO, self).__init__()
        self.geneWeight = geneWeight
        self.pushback_Score = pushback_Score
    def get_importance_trace(self,graph_type, max_plate_nesting, model, guide,discriminator, args, kwargs,detach=False):
        """
        Returns a single trace from the guide, which can optionally be detached,
        and the model that is run against it.
        """
        pyro.module("discriminator", discriminator)
        
        guide_trace = poutine.trace(guide, graph_type=graph_type).get_trace(*args, **kwargs)
        if detach:
            guide_trace.detach_()
        model_trace = poutine.trace(poutine.replay(model, trace=guide_trace), graph_type=graph_type).get_trace(*args, **kwargs)
        if is_validation_enabled():
            check_model_guide_match(model_trace, guide_trace, max_plate_nesting)

        guide_trace = prune_subsample_sites(guide_trace)
        model_trace = prune_subsample_sites(model_trace)

        model_trace = self.compute_log_prob(model_trace)
        guide_trace.compute_score_parts()

            
        if is_validation_enabled():
            for site in model_trace.nodes.values():
                if site["type"] == "sample":
                    check_site_shape(site, max_plate_nesting)
            for site in guide_trace.nodes.values():
                if site["type"] == "sample":
                    check_site_shape(site, max_plate_nesting)

        return model_trace, guide_trace
    def _get_traces(self, model, guide,discriminator, args, kwargs):
        """
        Runs the guide and runs the model against the guide with
        the result packaged as a trace generator.
        """
        if self.vectorize_particles:
            if self.max_plate_nesting == float("inf"):
                self._guess_max_plate_nesting(model, guide,discriminator, args, kwargs)
            yield self._get_vectorized_trace(model, guide,discriminator, args, kwargs)
        else:
            for i in range(self.num_particles):
                yield self._get_trace(model, guide,discriminator, args, kwargs)
    def _get_trace(self, model, guide,discriminator, args, kwargs):
        """
        Returns a single trace from the guide, and the model that is run
        against it.
        """
        model_trace, guide_trace = self.get_importance_trace(
            "flat", self.max_plate_nesting, model, guide,discriminator, args, kwargs
        )
        if is_validation_enabled():
            check_if_enumerated(guide_trace)
        return model_trace, guide_trace
    def differentiable_loss(self, model, guide, *args, **kwargs):
        """
        Computes the surrogate loss that can be differentiated with autograd
        to produce gradient estimates for the model and guide parameters
        """
        loss = 0.0
        surrogate_loss = 0.0
        for model_trace, guide_trace in self._get_traces(model, guide, args, kwargs):
            loss_particle, surrogate_loss_particle = self._differentiable_loss_particle(
                model_trace, guide_trace
            )
            surrogate_loss += surrogate_loss_particle / self.num_particles
            loss += loss_particle / self.num_particles
        warn_if_nan(surrogate_loss, "loss")
        return loss + (surrogate_loss - torch_item(surrogate_loss))


    def _differentiable_loss_particle(self, model_trace, guide_trace):
        elbo_particle = 0
        surrogate_elbo_particle = 0
        log_r = None
        # compute elbo and surrogate elbo
        for name, site in model_trace.nodes.items():
            if site["type"] == "sample":
                elbo_particle = elbo_particle + torch_item(site["log_prob_sum"])
                surrogate_elbo_particle = surrogate_elbo_particle + site["log_prob_sum"]
        
        for name, site in guide_trace.nodes.items():
            if site["type"] == "sample":
                
                log_prob, score_function_term, entropy_term = site["score_parts"]

                elbo_particle = elbo_particle - torch_item(site["log_prob_sum"])

                if not is_identically_zero(entropy_term):
                    surrogate_elbo_particle = (
                        surrogate_elbo_particle - entropy_term.sum()
                    )

                if not is_identically_zero(score_function_term):
                    if log_r is None:
                        log_r = _compute_log_r(model_trace, guide_trace)
                    site = log_r.sum_to(site["cond_indep_stack"])
                    surrogate_elbo_particle = (
                        surrogate_elbo_particle + (site * score_function_term).sum()
                    )

        return -elbo_particle, -surrogate_elbo_particle
    def _sum_rightmost(self,value, dim):
        r"""
        Sum out ``dim`` many rightmost dimensions of a given tensor.

        Args:
            value (Tensor): A tensor of ``.dim()`` at least ``dim``.
            dim (int): The number of rightmost dims to sum out.
        """
        if dim == 0:
            return value
        required_shape = value.shape[:-dim] + (-1,)
        return value.reshape(required_shape).sum(-1)
    def loss_and_grads(self, model, guide, discriminator, device, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Computes the ELBO as well as the surrogate ELBO that is used to form the gradient estimator.
        Performs backward on the latter. Num_particle many samples are used to form the estimators.
        """
        loss = 0.0
        
        # grab a trace from the generator
        for model_trace, guide_trace in self._get_traces(model, guide,discriminator, args, kwargs): 
            
            loss_particle, surrogate_loss_particle = self._differentiable_loss_particle(
                model_trace, guide_trace
            )


            x_true = model_trace.nodes['obs']['value'].to(device)
            y_true = torch.ones(x_true.shape[0],1).float().to(device)
            x_fake = model_trace.nodes['recon']['value'].to(device)
            y_fake = torch.zeros(x_true.shape[0],1).float().to(device)
            loss_discriminator = discriminator(x_true,y_true)

            discriminator_fake = discriminator(x_fake,y_fake)

            loss_discriminator += discriminator_fake
            loss_discriminator = (loss_discriminator / self.num_particles)
            vae_add = discriminator(x_fake,y_true)

            
            loss_generator = surrogate_loss_particle + vae_add

            
            if self.pushback_Score is not None:
                pushback_Score = Parameter(torch.sum(self.pushback_Score))

                loss += 0.00001*(loss_particle) / self.num_particles + vae_add + 10000*pushback_Score#self.pushback_Score
            
            else:
                loss += (loss_particle) / self.num_particles + vae_add
            
            # collect parameters to train from model and guide
            trainable_params = any(
                site["type"] == "param"
                for trace in (model_trace, guide_trace)
                for site in trace.nodes.values()
                
            )
           
            if trainable_params and getattr(
                surrogate_loss_particle, "requires_grad", False
            ):
                if self.pushback_Score is not None:
                    surrogate_loss_particle = 0.00001* surrogate_loss_particle / self.num_particles + vae_add + 10000*pushback_Score
                else:
                    surrogate_loss_particle = surrogate_loss_particle / self.num_particles + vae_add
        warn_if_nan(loss, "loss")
        return loss,surrogate_loss_particle,loss_discriminator
    def compute_log_prob(self,trace, site_filter=lambda name, site: True):
        """
        Compute the site-wise log probabilities of the trace.
        Each ``log_prob`` has shape equal to the corresponding ``batch_shape``.
        Each ``log_prob_sum`` is a scalar.
        Both computations are memoized.
        """
        
        for name, site in trace.nodes.items():
            if site["type"] == "sample" and site_filter(name, site):
                if "log_prob" not in site:
                    try:
                        if self.geneWeight is not None and name == 'obs':
                            #print(site["value"].shape)
                            #log_p_gene = site["fn"].base_dist.log_prob(site["value"])*self.geneWeight
                            # print(site["fn"].log_prob(site["value"]))
                            
                            log_p_gene = site["fn"].base_dist.log_prob(site["value"])*self.geneWeight
                            log_p = self._sum_rightmost(log_p_gene,1)#1 is the indicator of event, showed in model.sample.event
                            #log_p = site["fn"].log_prob(site["value"], *site["args"], **site["kwargs"])
                        else:
                            log_p = site["fn"].log_prob(
                            site["value"], *site["args"], **site["kwargs"])
                            # log_p = log_p.mean()
                        #print(log_p.shape)
                        
                    except ValueError as e:
                        _, exc_value, traceback = sys.exc_info()
                        shapes = self.format_shapes(last_site=site["name"])
                        raise ValueError(
                            "Error while computing log_prob at site '{}':\n{}\n{}".format(
                                name, exc_value, shapes
                            )
                        ).with_traceback(traceback) from e
                    site["unscaled_log_prob"] = log_p
                    log_p = scale_and_mask(log_p, site["scale"], site["mask"])
                    site["log_prob"] = log_p
                    site["log_prob_sum"] = log_p.sum()
                    # site["log_prob_sum"] = log_p.mean() #use mean to try to tackle large batch size nan issue
                    if is_validation_enabled():
                        warn_if_nan(
                            site["log_prob_sum"],
                            "log_prob_sum at site '{}'".format(name),
                        )
                        warn_if_inf(
                            site["log_prob_sum"],
                            "log_prob_sum at site '{}'".format(name),
                            allow_neginf=True,
                        )
        return trace