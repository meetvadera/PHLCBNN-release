import torch
import torch.nn as nn
from models.variational_layers import BayesianLinearLayer
import numpy as np

td = torch.distributions

class VariationalClassificationMLP(torch.nn.Module):
    """
     Variational BNN MLP
    """
    def __init__(self, layer=BayesianLinearLayer, ip_dim=1, op_dim=1, num_nodes=50, num_layers=1,
                 init_to_prior=False):
        super(VariationalClassificationMLP, self).__init__()
        self.num_layers = num_layers
        self.activation = nn.ReLU()
        self.fc_hidden = []
        self.fc1 = layer(ip_dim, num_nodes, init_to_prior=init_to_prior)
        for _ in np.arange(self.num_layers - 1):
            self.fc_hidden.append(layer(num_nodes, num_nodes, init_to_prior=init_to_prior))
        self.fc_out = layer(num_nodes, op_dim, init_to_prior=init_to_prior)
        self.noise_layer = torch.nn.CrossEntropyLoss(reduction='sum')

    def forward(self, x, do_sample=True):
        x = self.fc1(x, do_sample=do_sample)
        x = self.activation(x)
        for layer in self.fc_hidden:
            x = layer(x, do_sample=do_sample)
            x = self.activation(x)
        return self.fc_out(x, do_sample=do_sample)

    def kl_divergence_w(self):
        kld = self.fc1.kl_divergence() + self.fc_out.kl_divergence()
        for layer in self.fc_hidden:
            kld += layer.kl_divergence()
        return kld

    def likelihood(self, x=None, y=None):
        out = self.forward(x)
        return - self.noise_layer(out, y)

    def test_log_likelihood(self, x=None, y=None, n_sample=100.):
        neg_ll = 0.
        for _ in np.arange(n_sample):
            neg_ll += self.likelihood(x, y)
        return -1 * neg_ll / n_sample

    def neg_elbo(self, num_batches, x=None, y=None):
        # scale the KL terms by number of batches so that the minibatch elbo is an unbiased estiamte of the true elbo.
        Elik = self.likelihood(x, y)
        return self.kl_divergence_w() / num_batches - Elik


if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt
    from utils_data import gen_synthetic_2d

    np.random.seed(123)
    torch.manual_seed(123)
    x, y = gen_synthetic_2d()
    bnn = VariationalClassificationMLP(ip_dim=2, op_dim=2, num_nodes=100)
    optimizer = torch.optim.Adam(bnn.parameters(), lr=1e-2)
    num_epochs = 5000
    neg_elbo = torch.zeros([num_epochs, 1])
    for i in np.arange(num_epochs):
        optimizer.zero_grad()
        loss = bnn.neg_elbo(num_batches=1, x=x, y=y)
        loss.backward()
        optimizer.step()
        neg_elbo[i] = loss.data

    plt.plot(neg_elbo, 'ro-')


    viz = True
    if viz:
        plt.figure()
        # generate posterior predictive samples on a grid
        x_g = np.linspace(-3.5, 3.5, 200)
        y_g = np.linspace(-4.5, 3.5, 200)
        xv, yv = np.meshgrid(x_g, y_g)
        grid = torch.FloatTensor(np.dstack([xv, yv]).reshape(-1, 2))
        num_mc_samples = 1000
        preds = np.zeros([num_mc_samples, grid.shape[0], 2])
        for i in np.arange(num_mc_samples):
            preds[i] = torch.softmax(bnn.forward(grid), dim=1).data.numpy()
        # compute posterior predictive by averaging over number of samples
        preds = preds.mean(axis=0)
        cmap = matplotlib.cm.get_cmap('RdBu')
        for i in np.arange(grid.shape[0]):
            plt.plot(grid[i, 0], grid[i, 1], 'o', color=cmap(1-preds[i,0]))
            # plt.plot(grid[i, 0], grid[i, 1], 'bo', alpha=preds[i,1])
        # plot training data
        plt.plot(x[:50, 0].data.numpy(), x[:50, 1].data.numpy(), 'ro')
        plt.plot(x[50:, 0].data.numpy(), x[50:, 1].data.numpy(), 'bo')
    plt.show()
