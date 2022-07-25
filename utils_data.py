import torch
import numpy as np
import numpy.random as npr

def gen_synthetic_2d():
    mu_1 = np.ones(2)
    mu_2 = -np.ones(2)
    x1 = npr.randn(50, 2) + mu_1
    x2 = npr.randn(50, 2) + mu_2
    x = np.concatenate([x1, x2])
    y = np.zeros([100])
    y[50:] = 1
    return torch.FloatTensor(x), torch.LongTensor(y)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x, y = gen_synthetic_2d()
    x = x.numpy()
    plt.plot(x[:50, 0], x[:50, 1], 'ro')
    plt.plot(x[50:, 0], x[50:, 1], 'bo')
    plt.show()
