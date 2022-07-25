import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.adam import Adam

from models.mlp import mlp_1_synthetic
from optimizers.sghmc import SGHMC
from utils_data import gen_synthetic_2d, gen_uniform_2d

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
        # generate validation data for loss based calibration.
        # since this is just 2d we will uniformly sample the space
        xval = gen_uniform_2d()
        # variational BNN - 1 hidden layer with 50 units
        bnn = mlp_1_synthetic(hidden_dim_1=50)
        sghmc_optimizer = SGHMC(bnn.parameters(), lr=1e-1, momentum=0.5, weight_decay=len(x) * 1e-2,
                                num_training_samples=len(x))

        lc_sghmc_factor = 1.
        burn_in_iters = 300
        num_samples = 100
        thinning_interval = 50
        p = torch.zeros(xval.shape[0], 2)
        p_test = torch.zeros(x_test.shape[0], 2)
        counter = 0
        for iter in range(burn_in_iters + num_samples * thinning_interval):
            bnn_logits = bnn(x)
            decision_cost = torch.matmul(F.log_softmax(bnn_logits, dim=-1).exp().unsqueeze(dim=1),
                                         lm).squeeze()
            decision_softmax = F.log_softmax(-BETA * decision_cost, dim=-1).exp().detach()
            argmaxed_decision_cost_mean = (decision_cost * decision_softmax).sum(dim=-1).mean()
            loss = F.cross_entropy(bnn_logits, y) + lc_sghmc_factor * argmaxed_decision_cost_mean
            sghmc_optimizer.zero_grad()
            loss.backward()
            sghmc_optimizer.step()
            # print("SGHMC train loss: ", loss.item())
            if iter >= burn_in_iters and (iter % thinning_interval == 0):
                p += torch.softmax(bnn(xval), dim=-1).detach()
                p_test += torch.softmax(bnn(x_test), dim=-1).detach()
                # print("SGHMC test loss: ", F.cross_entropy(bnn(x_test), y_test).item())
                counter += 1

        num_samples = counter
        print("Counter: ", counter)
        p /= num_samples
        p_test /= num_samples

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
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        n1 = np.int(imbalance_pct * x.shape[0])  # number of examples in class 1
        axs[0].plot(x[:n1, 0].data.numpy(), x[:n1, 1].data.numpy(), 'ro', alpha=0.9)
        axs[0].plot(x[n1:, 0].data.numpy(), x[n1:, 1].data.numpy(), 'bo', alpha=0.9)
        axs[0].set_title("Training data")
        axs[1].plot(x_test[:n1, 0].data.numpy(), x_test[:n1, 1].data.numpy(), 'ro', alpha=0.9)
        axs[1].plot(x_test[n1:, 0].data.numpy(), x_test[n1:, 1].data.numpy(), 'bo', alpha=0.9)
        axs[1].set_title("Testing data")
        axs[2].plot(xval[:, 0].data.numpy(), xval[:, 1].data.numpy(), 'ko', alpha=0.9)
        axs[2].set_title("Validation data")
        # plt.show()

    print("Mean decision cost - Variational BNN: ", np.mean(np.array(variational_bnn_decision_cost_list)))
    print("Mean decision cost - Q model: ", np.mean(np.array(q_model_decision_cost_list)))
    print("Std dev decision cost - Variational BNN: ", np.std(np.array(variational_bnn_decision_cost_list)))
    print("Std dev decision cost - Q model: ", np.std(np.array(q_model_decision_cost_list)))
