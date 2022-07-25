import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.adam import Adam

from models.mlp import mlp_1_synthetic
from models.variational_mlp import VariationalClassificationMLP
from utils_data import gen_synthetic_2d, gen_uniform_2d

# import seaborn as sns



if __name__ == "__main__":
    num_trials = 10
    variational_bnn_decision_cost_list = []
    q_model_decision_cost_list = []
    for trial in range(num_trials):
        np.random.seed(10 * trial)
        torch.manual_seed(10 * trial)

        # define cost matrix

        lm = torch.ones(2, 2)
        lm[0, 0] = 0
        lm[1, 1] = 0
        lm[0, 1] = 1.  # False negatives are X times as costly as false positives.
        lm[1, 0] = 0.1
        lm.requires_grad = False

        BETA = 1000.

        # generate data with class imbalance; 2d.
        imbalance_pct = 0.1
        x, y = gen_synthetic_2d(imbalance_pct=imbalance_pct)
        x_test, y_test = gen_synthetic_2d(imbalance_pct=imbalance_pct)
        # variational BNN - 1 hidden layer with 50 units
        bnn = VariationalClassificationMLP(ip_dim=2, op_dim=2, num_layers=1, num_nodes=50)
        bnn.fit(x, y)  # learn by maximizing elbo

        # generate validation data for loss based calibration.
        # since this is just 2d we will uniformly sample the space
        xval = gen_uniform_2d()
        # to get a single sample posterior prediction at xval
        p = torch.softmax(bnn.forward(xval, do_sample=True), dim=1)
        print(xval.shape, p.shape)

        # if you want to persist the predictions, which would be more reliable do
        num_variational_samples = 100
        p = torch.zeros(xval.shape[0], 2)
        for _ in np.arange(num_variational_samples):
            p += torch.softmax(bnn.forward(xval, do_sample=True), dim=1)
        p /= num_variational_samples

        q_model = mlp_1_synthetic(hidden_dim_1=50)

        q_model_train_iters = 500
        q_model_optimizer = Adam(q_model.parameters(), lr=1e-1)
        # q_model_optimizer = SGD(q_model.parameters(), lr=5e-3, momentum=0.9, weight_decay=5e-4)
        num_validation_points = len(xval)

        for train_iter in range(q_model_train_iters):
            # print("Q-model training iter: ", train_iter)
            logits_q_model = q_model(xval)
            ce_loss = -torch.sum(F.log_softmax(logits_q_model, dim=-1).exp() *
                                 torch.log(p), dim=1).mean()
            decision_cost = torch.matmul(F.log_softmax(logits_q_model, dim=-1).exp().unsqueeze(dim=1),
                                         lm).squeeze()
            decision_softmax = F.log_softmax(-BETA * decision_cost, dim=-1).exp().detach()
            argmaxed_decision_cost_mean = (decision_cost * decision_softmax).sum(dim=-1).mean()
            entropy_loss = -(F.log_softmax(logits_q_model, dim=-1).exp() *
                             F.log_softmax(logits_q_model, dim=-1)).sum(dim=-1).mean()
            total_loss = ce_loss - entropy_loss + argmaxed_decision_cost_mean
            q_model_optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            q_model_optimizer.step()
            # print("Train Loss: %f, Test loss: %f" % (total_loss.item(), F.cross_entropy(q_model(x_test), y_test).item()))

        p_test = torch.zeros(x_test.shape[0], 2)
        for _ in np.arange(num_variational_samples):
            p_test += torch.softmax(bnn.forward(x_test, do_sample=True), dim=1)
        p_test /= num_variational_samples

        q_model_test_proba = F.log_softmax(q_model(x_test)).exp()
        q_model_decision_cost = torch.matmul(q_model_test_proba.unsqueeze(dim=1), lm).squeeze().detach()
        q_model_decision_softmax = F.log_softmax(-BETA * q_model_decision_cost, dim=-1).exp()
        ##TODO: Give meaningful names below
        A = np.expand_dims(torch.matmul(q_model_decision_softmax, lm.transpose(0, 1)).cpu().numpy(),
                           1)
        B = np.expand_dims(F.one_hot(y_test.long(), num_classes=2).cpu().numpy().T, 0)
        B = np.matmul(A, B.T)
        # q_model_decision_cost_mean = (q_model_decision_cost * q_model_decision_softmax).sum(dim=-1).mean()
        q_model_decision_cost_mean = B.squeeze().mean()

        print("Q Model decision cost: ", q_model_decision_cost_mean)

        bnn_decision_cost = torch.matmul(p_test.unsqueeze(dim=1), lm).squeeze()
        bnn_decision_softmax = F.log_softmax(-BETA * bnn_decision_cost, dim=-1).exp().detach()
        ##TODO: Give meaningful names below
        A = np.expand_dims(torch.matmul(bnn_decision_softmax, lm.transpose(0, 1)).cpu().numpy(),
                           1)
        B = np.expand_dims(F.one_hot(y_test.long(), num_classes=2).cpu().numpy().T, 0)
        B = np.matmul(A, B.T)
        # q_model_decision_cost_mean = (q_model_decision_cost * q_model_decision_softmax).sum(dim=-1).mean()
        bnn_decision_cost_mean = B.squeeze().mean()
        print("Variational BNN decision cost: ", bnn_decision_cost_mean)

        variational_bnn_decision_cost_list.append(bnn_decision_cost_mean)
        q_model_decision_cost_list.append(q_model_decision_cost_mean)

        # visualize training and validation data.
        # sns.set_style("whitegrid")
        # sns.set_context("paper")
        plt.rcParams.update({"mathtext.fallback_to_cm": True,
                             'font.family': 'normal'})
        fig, axs = plt.subplots(1, 2, figsize=(4.5, 2))
        n1 = np.int(imbalance_pct * x.shape[0])  # number of examples in class 1
        axs[0].plot(x[:n1, 0].data.numpy(), x[:n1, 1].data.numpy(), 'ro', alpha=0.9)
        axs[0].plot(x[n1:, 0].data.numpy(), x[n1:, 1].data.numpy(), 'bo', alpha=0.9)
        axs[0].set_title(r"Original data")
        # axs[1].plot(x_test[:n1, 0].data.numpy(), x_test[:n1, 1].data.numpy(), 'ro', alpha=0.9)
        # axs[1].plot(x_test[n1:, 0].data.numpy(), x_test[n1:, 1].data.numpy(), 'bo', alpha=0.9)
        axs[1].set_title(r"Additional training data")
        axs[1].plot(xval[:, 0].data.numpy(), xval[:, 1].data.numpy(), 'ko', alpha=0.9)
        # axs[1].set_title(r"Additional Data $\mathcal{D^\prime}$")
        # plt.title(r"Original Data ($\mathcal{D})\textrm{ and Additional Data }($\mathcal{D^\prime}$)")
        # plt.show()
        # plt.savefig('../toy_data.pdf', format='pdf', bbox_inches='tight')

    print("Mean decision cost - Variational BNN: ", np.mean(np.array(variational_bnn_decision_cost_list)))
    print("Mean decision cost - Q model: ", np.mean(np.array(q_model_decision_cost_list)))
    print("Std dev decision cost - Variational BNN: ", np.std(np.array(variational_bnn_decision_cost_list)))
    print("Std dev decision cost - Q model: ", np.std(np.array(q_model_decision_cost_list)))
