"""
  Contains implementations of various Bayesian Layers and likelihood layers
"""
import torch
import math
from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from scipy.special import gammaln
td = torch.distributions


def reparam(mu, logvar, do_sample=True):
    if do_sample:
        std = 0.5*logvar.exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return mu + eps * std
    else:
        return mu


class BayesianLinearLayerCorrelated(torch.nn.Module):
    """
        Affine layer with N(0, v/H) priors on weights and
        matrix normal variational Gaussian approximation
        q(w) = MN (M, U, V). V is diagonal, U = hh' + \Psi (rank one + noise). assumes W is n_in * n_out (eventhough
        is implemented in transpose)
    """
    def __init__(self, in_features, out_features, cuda=False, init_weight=None, init_bias=None, init_to_prior=False):
        super(BayesianLinearLayerCorrelated, self).__init__()
        self.cuda = cuda
        self.in_features = in_features
        self.out_features = out_features

        # weight mean params
        self.weights = Parameter(torch.FloatTensor(out_features, in_features))
        self.bias = Parameter(torch.FloatTensor(out_features))
        # weight variance params
        # U = hh' + Psi
        self.h = Parameter(torch.FloatTensor(in_features, 1))
        self.psi_log = Parameter(torch.FloatTensor(in_features))
        self.V_log = Parameter(torch.FloatTensor(out_features))
        self.bias_logvar = Parameter(torch.FloatTensor(out_features))

        # numerical stability
        self.fudge_factor = 1e-8

        # We will use a N(0, 1/num_inputs) prior over weights
        self.prior_U = torch.FloatTensor([1. / np.sqrt(self.weights.size(1))])
        self.prior_mean = torch.FloatTensor([0.])
        # for Bias use a prior of N(0, 1)
        self.prior_bias_stdv = torch.FloatTensor([1.])
        self.prior_bias_mean = torch.FloatTensor([0.])

        # init params either random or with pretrained net
        self.init_parameters(init_weight, init_bias)

    def init_parameters(self, init_weight, init_bias):
        # init means
        if init_weight is not None:
            self.weights.data = torch.FloatTensor(init_weight)
        else:
            self.weights.data.normal_(0, np.float(self.prior_U.numpy()[0]))

        if init_bias is not None:
            self.bias.data = torch.FloatTensor(init_bias)
        else:
            self.bias.data.normal_(0, 1)

        # init variances
        self.h.data.normal_(0, 1e-2)
        self.psi_log.data.normal_(-5, 1e-2)
        self.V_log.data.normal_(-5, 1e-2)
        self.bias_logvar.data.normal_(-9, 1e-2)

    def forward(self, x, do_sample=True):
        # local reparameterization trick
        mu_activations = F.linear(x, self.weights, self.bias)
        # var a'Ua * V = a'(Psi + hh')a * V
        # super inefficient FIX
        U = torch.diagflat(self.psi_log.exp()) + self.h @ self.h.t()
        psi_half_x = self.psi_log.mul(0.5).exp().unsqueeze(dim=1) * x.t() # x.t() must be D*N
        psi_half_x = (psi_half_x**2).sum(axis=0, keepdims=True)
        h_x = x @ self.h
        h_x = (h_x**2).sum(axis=1, keepdims=True)
        aUa = psi_half_x.t() + h_x
        var_activations = aUa * self.V_log.exp() + self.bias_logvar.exp()
        # for i, a in enumerate(x):
        #     var_activations[i] = (a.t() @ U) @ a * self.V_log.exp() + self.bias_logvar.exp()
        #     print (var_activations[i], activs[i])

        # var_activations = F.linear(x.pow(2), self.weights_logvar.exp(), self.bias_logvar.exp())
        activ = reparam(mu_activations, var_activations.log(), do_sample=do_sample)
        return activ

    def kl_divergence(self):
        """
        KL divergence (q(W) || p(W))
        :return:
        """
        U = torch.diagflat(self.psi_log.exp()) + self.h @ self.h.t()
        kld_weights = 0.5 * (self.in_features * (torch.trace(U)) * self.V_log.exp().sum() +
                             self.in_features * torch.trace(self.weights @ self.weights.t()) -
                             self.in_features * self.out_features -
                             self.in_features * self.out_features * np.log(self.in_features) -
                             self.out_features * torch.logdet(U) - self.in_features * self.V_log.sum()
        )
        kld_bias = self.prior_bias_stdv.log() - self.bias_logvar.mul(0.5) + \
            (self.bias_logvar.exp() + (self.bias.pow(2) - self.prior_bias_mean)) / (2 * self.prior_bias_stdv.pow(2)) \
            - 0.5
        return kld_weights + kld_bias.sum()


class NaiveBayesianLinearLayerCorrelated(BayesianLinearLayerCorrelated):
    """
        Affine layer with N(0, I) priors on weights and
        matrix normal variational Gaussian approximation
        q(w) = MN (M, U, V). V is diagonal, U = hh' + \Psi (rank one + noise). assumes W is n_in * n_out (eventhough
        is implemented in transpose)
    """
    def __init__(self, in_features, out_features, cuda=False, init_weight=None, init_bias=None):
        super(NaiveBayesianLinearLayerCorrelated, self).__init__(in_features, out_features, cuda, init_weight,
                                                                 init_bias)

    def kl_divergence(self):
        """
        KL divergence (q(W) || p(W))
        :return:
        """
        U = torch.diagflat(self.psi_log.exp()) + self.h @ self.h.t()
        kld_weights = 0.5 * ((torch.trace(U)) * self.V_log.exp().sum() +
                             torch.trace(self.weights @ self.weights.t()) -
                             self.in_features * self.out_features -
                             self.out_features * torch.logdet(U) - self.in_features * self.V_log.sum()
        )
        kld_bias = self.prior_bias_stdv.log() - self.bias_logvar.mul(0.5) + \
            (self.bias_logvar.exp() + (self.bias.pow(2) - self.prior_bias_mean)) / (2 * self.prior_bias_stdv.pow(2)) \
            - 0.5
        return kld_weights + kld_bias.sum()




class BayesianLinearLayer(torch.nn.Module):
    """
    Affine layer with N(0, v/H) priors on weights and
    fully factorized variational Gaussian approximation
    """

    def __init__(self, in_features, out_features, cuda=False, init_weight=None, init_bias=None, init_to_prior=False):
        super(BayesianLinearLayer, self).__init__()
        self.cuda = cuda
        self.in_features = in_features
        self.out_features = out_features

        # weight mean params
        self.weights = Parameter(torch.FloatTensor(out_features, in_features))
        self.bias = Parameter(torch.FloatTensor(out_features))
        # weight variance params
        self.weights_logvar = Parameter(torch.FloatTensor(out_features, in_features))
        self.bias_logvar = Parameter(torch.FloatTensor(out_features))

        # numerical stability
        self.fudge_factor = 1e-8

        # We will use a N(0, 1/num_inputs) prior over weights
        self.prior_stdv = torch.FloatTensor([1. / np.sqrt(self.weights.size(1))])
        # self.prior_stdv = torch.FloatTensor([1. / np.sqrt(1e+3)])
        self.prior_mean = torch.FloatTensor([0.])
        # for Bias use a prior of N(0, 1)
        self.prior_bias_stdv = torch.FloatTensor([1.])
        self.prior_bias_mean = torch.FloatTensor([0.])

        # init params either random or with pretrained net
        self.init_parameters(init_weight, init_bias)

    def init_parameters(self, init_weight, init_bias):
        # init means
        if init_weight is not None:
            self.weights.data = torch.FloatTensor(init_weight)
        else:
            self.weights.data.normal_(0, np.float(self.prior_stdv.numpy()[0]))

        if init_bias is not None:
            self.bias.data = torch.FloatTensor(init_bias)
        else:
            self.bias.data.normal_(0, 1)

        # init variances
        self.weights_logvar.data.normal_(-9, 1e-2)
        self.bias_logvar.data.normal_(-9, 1e-2)

    def forward(self, x, do_sample=True):
        # local reparameterization trick
        mu_activations = F.linear(x, self.weights, self.bias)
        var_activations = F.linear(x.pow(2), self.weights_logvar.exp(), self.bias_logvar.exp())
        activ = reparam(mu_activations, var_activations.log(), do_sample=do_sample)
        return activ

    def kl_divergence(self):
        """
        KL divergence (q(W) || p(W))
        :return:
        """
        kld_weights = self.prior_stdv.log() - self.weights_logvar.mul(0.5) + \
            (self.weights_logvar.exp() + (self.weights.pow(2) - self.prior_mean)) / (2 * self.prior_stdv.pow(2)) - 0.5
        kld_bias = self.prior_bias_stdv.log() - self.bias_logvar.mul(0.5) + \
            (self.bias_logvar.exp() + (self.bias.pow(2) - self.prior_bias_mean)) / (2 * self.prior_bias_stdv.pow(2)) \
            - 0.5
        return kld_weights.sum() + kld_bias.sum()


class BayesianLinearLayerFixedVariance(torch.nn.Module):
    """
    Affine layer with N(0, v) priors on weights and
    fully factorized variational Gaussian approximation
    """

    def __init__(self, in_features, out_features, cuda=False, init_weight=None, init_bias=None, init_to_prior=False):
        super(BayesianLinearLayerFixedVariance, self).__init__()
        self.cuda = cuda
        self.in_features = in_features
        self.out_features = out_features

        # weight mean params
        self.weights = Parameter(torch.FloatTensor(out_features, in_features))
        self.bias = Parameter(torch.FloatTensor(out_features))
        # weight variance params
        self.weights_logvar = Parameter(torch.FloatTensor(out_features, in_features))
        self.bias_logvar = Parameter(torch.FloatTensor(out_features))

        # numerical stability
        self.fudge_factor = 1e-8

        # We will use a N(0, 1) prior over weights
        self.prior_stdv = torch.FloatTensor([1.])
        self.prior_mean = torch.FloatTensor([0.])
        # for Bias use a prior of N(0, 1)
        self.prior_bias_stdv = torch.FloatTensor([1.])
        self.prior_bias_mean = torch.FloatTensor([0.])
        if not init_to_prior:
            # init params either random or with pretrained net
            self.init_parameters(init_weight, init_bias)
        else:
            self.init_parameters_prior()

    def init_parameters(self, init_weight=None, init_bias=None):
        # init means
        if init_weight is not None:
            self.weights.data = torch.FloatTensor(init_weight)
        else:
            self.weights.data.normal_(0, np.float(self.prior_stdv.numpy()[0]))

        if init_bias is not None:
            self.bias.data = torch.FloatTensor(init_bias)
        else:
            self.bias.data.normal_(0, 1)

        # init variances
        self.weights_logvar.data.normal_(-9, 1e-2)
        self.bias_logvar.data.normal_(-9, 1e-2)

    def init_parameters_prior(self):
        """
        Set the variational parameters to the prior N(0, 1)
        Useful for sampling from the prior.
        :return:
        """
        self.weights.data = torch.zeros_like(self.weights)
        self.weights_logvar.data = torch.zeros_like(self.weights_logvar)
        self.bias.data = torch.zeros_like(self.bias)
        self.bias_logvar.data = torch.zeros_like(self.bias_logvar)

    def forward(self, x, do_sample=True):
        # local reparameterization trick
        mu_activations = F.linear(x, self.weights, self.bias)
        var_activations = F.linear(x.pow(2), self.weights_logvar.exp(), self.bias_logvar.exp())
        activ = reparam(mu_activations, var_activations.log(), do_sample=do_sample)
        return activ

    def kl_divergence(self):
        """
        KL divergence (q(W) || p(W))
        :return:
        """
        kld_weights = self.prior_stdv.log() - self.weights_logvar.mul(0.5) + \
            (self.weights_logvar.exp() + (self.weights.pow(2) - self.prior_mean)) / (2 * self.prior_stdv.pow(2)) - 0.5
        kld_bias = self.prior_bias_stdv.log() - self.bias_logvar.mul(0.5) + \
            (self.bias_logvar.exp() + (self.bias.pow(2) - self.prior_bias_mean)) / (2 * self.prior_bias_stdv.pow(2)) \
            - 0.5
        return kld_weights.sum() + kld_bias.sum()

class ScaledBayesianLinearLayer(BayesianLinearLayer):
    """
    Affine layer with N(0, v/H) priors on weights and N(mu, sigma^2/H) variational approximation
    """

    def __init__(self, in_features, out_features, cuda=False, init_weight=None, init_bias=None,
                 map_type='linear_map', clip_var=None, init_to_prior=None):
        super(ScaledBayesianLinearLayer, self).__init__(in_features, out_features, cuda, init_weight, init_bias,
                                                        init_to_prior=init_to_prior)
        self.H = np.sqrt(np.max(self.weights.size()))


    def forward(self, x, do_sample=True):
        # local reparameterization trick
        x = x / self.H
        mu_activations = F.linear(x, self.weights, self.bias)
        var_activations = F.linear(x.pow(2), self.weights_logvar.exp(), self.bias_logvar.exp())
        activ = reparam(mu_activations, var_activations.log(), do_sample=do_sample)
        return activ

    def kl_divergence(self):
        """
        KL divergence (q(W) || p(W))
        :return:
        """
        kld_weights = self.prior_stdv.log() - self.weights_logvar.mul(0.5) + np.log(self.H) + \
            (self.weights_logvar.exp()/self.H + (self.weights.pow(2) - self.prior_mean)) / (2 * self.prior_stdv.pow(2)) - 0.5
        kld_bias = self.prior_bias_stdv.log() - self.bias_logvar.mul(0.5) + \
            (self.bias_logvar.exp() + (self.bias.pow(2) - self.prior_bias_mean)) / (2 * self.prior_bias_stdv.pow(2)) \
            - 0.5
        return kld_weights.sum() + kld_bias.sum()


class GaussianLikelihoodGammaPrecision(torch.nn.Module):
    """
        N(y | f(x, w), \lambda^-1); \lambda ~ Gamma(a, b)
    """

    def __init__(self, a0=6, b0=6, cuda=False):
        super(GaussianLikelihoodGammaPrecision, self).__init__()
        self.cuda = cuda
        self.a0 = a0
        self.b0 = b0
        # variational parameters
        self.ahat = Parameter(torch.FloatTensor([10.]))
        self.bhat = Parameter(torch.FloatTensor([3.]))
        self.const = torch.log(torch.FloatTensor([2 * math.pi]))

    def likelihood(self,  y_pred=None, y=None):
        """
        computes E_q(\lambda)[ln N (y_pred | y, \lambda^-1)], where q(lambda) = Gamma(ahat, bhat)
        :param y_pred:
        :param y:
        :return:
        """
        n = y_pred.shape[0]
        return -0.5 * n * self.const + 0.5 * n * (torch.digamma(self.ahat) - torch.log(self.bhat)) - 0.5 * (self.ahat/
        self.bhat) * ((y_pred - y) ** 2).sum()

    def kl_lambda(self):
        return (self.ahat - self.a0) * torch.digamma(self.ahat) - torch.lgamma(self.ahat) + gammaln(self.a0) + \
            self.a0 * (torch.log(self.bhat) - np.log(self.b0)) + self.ahat * (self.b0 - self.bhat) / self.bhat


