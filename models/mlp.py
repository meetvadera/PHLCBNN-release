import torch

__all__ = ['mlp1_100', 'mlp1_200', 'mlp2_100', 'mlp2_200']


class _MLP_1(torch.nn.Module):
    def __init__(self, hidden_dim_1=200):
        super(_MLP_1, self).__init__()
        self.hidden_dim_1 = hidden_dim_1
        self.h1 = torch.nn.Linear(784, self.hidden_dim_1)
        self.out = torch.nn.Linear(self.hidden_dim_1, 10)

    def forward(self, x):
        h1 = self.h1(x.view(x.shape[0], -1))
        out = self.out(h1)
        return out


class _MLP_2(torch.nn.Module):
    def __init__(self, hidden_dim_1=200, hidden_dim_2=200):
        super(_MLP_2, self).__init__()
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.h1 = torch.nn.Linear(784, self.hidden_dim_1)
        self.h2 = torch.nn.Linear(self.hidden_dim_1, self.hidden_dim_2)
        self.out = torch.nn.Linear(self.hidden_dim_2, 10)

    def forward(self, x):
        h1 = self.h1(x.view(x.shape[0], -1))
        h2 = self.h2(h1)
        out = self.out(h2)
        return out


def mlp1_100():
    return _MLP_1(hidden_dim_1=100)


def mlp1_200():
    return _MLP_1(hidden_dim_1=200)


def mlp2_100():
    return _MLP_2(hidden_dim_1=100, hidden_dim_2=100)


def mlp2_200():
    return _MLP_2(hidden_dim_1=200, hidden_dim_2=200)
