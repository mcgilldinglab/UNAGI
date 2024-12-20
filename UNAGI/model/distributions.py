import torch
from torch import broadcast_shapes
from torch.distributions.gamma import Gamma
from torch.distributions import constraints
from torch.distributions.exponential import Exponential
from torch.distributions.utils import (
    broadcast_all,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)

from pyro.distributions import TorchDistribution, LogNormal, Poisson, Gamma, Weibull, Chi2
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
            batch_shape = broadcast_shapes(gate.shape, base_dist.batch_shape)
            self.gate = gate.expand(batch_shape)
        else:
            batch_shape = broadcast_shapes(gate_logits.shape, base_dist.batch_shape)
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
            temp_base_log_prob = self.base_dist.log_prob(temp_value)
            log_prob_minus_log_gate = -gate_logits + temp_base_log_prob#self.base_dist.log_prob(temp_value)
            log_gate = -torch.nn.functional.softplus(-gate_logits)
            log_prob = log_prob_minus_log_gate + log_gate
            zero_log_prob = log_gate
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

class ZeroInflatedGamma(ZeroInflatedDistribution):
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
class ZeroInflatedLogNormal(ZeroInflatedDistribution):
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
class ZeroInflatedExponential(ZeroInflatedDistribution):
    """
    A Zero Inflated Exponential distribution.

    :param total_count: non-negative number of negative Bernoulli trials.
    :type total_count: float or torch.Tensor
    :param torch.Tensor probs: Event probabilities of success in the half open interval [0, 1).
    :param torch.Tensor logits: Event log-odds for probabilities of success.
    :param torch.Tensor gate: probability of extra zeros.
    :param torch.Tensor gate_logits: logits of extra zeros.
    """

    arg_constraints = {
        'rate': constraints.positive,
        "gate": constraints.unit_interval,
        "gate_logits": constraints.real,
    }
   
    support = constraints.positive

    def __init__(
        self,
        rate,
        gate=None,
        gate_logits=None,
        validate_args=None
    ):
        base_dist = Exponential(rate=rate,validate_args=False)
        base_dist._validate_args = validate_args
        super().__init__(
            base_dist, gate=gate, gate_logits=gate_logits, validate_args=validate_args
        )