import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.adam import Adam

from laplace.curvature.curvature.fisher import KFAC
from laplace.curvature.curvature.sampling import invert_factors, sample_and_replace_weights
from models.mlp import mlp_1_synthetic
from utils_data import gen_synthetic_2d, gen_uniform_2d


class MLP(torch.nn.Module):
    def __init__(self, num_nodes=50, ip_dim=2, op_dim=2):
        super(MLP, self).__init__()
        self.hidden_dim_1 = num_nodes
        self.h1 = torch.nn.Linear(ip_dim, self.hidden_dim_1)
        # self.h2 = torch.nn.Linear(self.hidden_dim_1, self.hidden_dim_1)
        self.out = torch.nn.Linear(self.hidden_dim_1, op_dim)

    def forward(self, x):
        h1 = F.relu(self.h1(x))
        # h1 = F.relu(self.h2(h1))
        out = self.out(h1)
        return out


device = 'cpu'


def train(model, x, y, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
            logits = model(x)
            loss = criterion(logits, y)
            model.zero_grad()
            loss.backward()
            optimizer.step()


def eval(model, x, y):
    model.eval()
    logits = torch.Tensor().to(device)
    targets = torch.LongTensor()

    with torch.no_grad():
        logits = torch.cat([logits, model(x.to(device))])
        targets = torch.cat([targets, y])
    return torch.nn.functional.softmax(logits, dim=1), targets


def accuracy(predictions, labels):
    print(f"Accuracy: {100 * np.mean(np.argmax(predictions.cpu().numpy(), axis=1) == labels.numpy()):.2f}%")


def convert_nn_to_bnn(model, x):
    # learn kroecker fractored Hessian
    kfac = KFAC(model)
    logits = model(x)
    dist = torch.distributions.Categorical(logits=logits)

    # A rank-1 Kronecker factored IM approximation.
    labels = dist.sample()

    loss = criterion(logits, labels)
    model.zero_grad()
    loss.backward()

    kfac.update(batch_size=x.shape[0])
    factors = list(kfac.state.values())
    inv_factors = invert_factors(factors, norm=1., scale=1e2)
    return inv_factors


def get_posterior_predictive_samples(model, inv_factors, xval, num_samples=100):
    posterior_mean = copy.deepcopy(model.state_dict())
    posterior_preds = torch.zeros(num_samples, xval.shape[0], 2)
    for sample in range(num_samples):
        sample_and_replace_weights(model, inv_factors)
        posterior_preds[sample] = torch.nn.functional.softmax(model(xval), dim=1)
        model.load_state_dict(posterior_mean)
    return posterior_preds


if __name__ == "__main__":
    variational_bnn_decision_cost_list = []
    q_model_decision_cost_list = []
    lm = torch.ones(2, 2)
    lm[0, 0] = 0
    lm[1, 1] = 0
    lm[0, 1] = 1.  # False negatives are X times as costly as false positives.
    lm[1, 0] = 0.1
    lm.requires_grad = False

    BETA = 1000.
    num_trials = 10
    for trial in range(num_trials):
        np.random.seed(10 * trial)
        torch.manual_seed(10 * trial)

        # generate data with class imbalance; 2d.
        imbalance_pct = 0.1
        x, y = gen_synthetic_2d(imbalance_pct=imbalance_pct)
        model = MLP()

        # fit NN
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=1e-2)
        criterion = torch.nn.CrossEntropyLoss()
        train(model, x, y, criterion, optimizer, epochs=2000)
        # np.random.seed(132)
        x_test, y_test = gen_synthetic_2d(imbalance_pct=imbalance_pct)
        point_predictions, point_labels = eval(model, x_test, y_test)
        # accuracy(point_predictions, point_labels)

        # learn bnn
        inv_factors = convert_nn_to_bnn(model, x)

        # generate validation data for loss based calibration.
        # since this is just 2d we will uniformly sample the space
        xval = gen_uniform_2d()

        posterior_preds = get_posterior_predictive_samples(model, inv_factors, xval, num_samples=100)
        # print(posterior_preds.shape) # shape = num_samples * N (number of xval data) * 2

        p = posterior_preds.mean(dim=0).detach()
        q_model = mlp_1_synthetic(hidden_dim_1=50)

        q_model_train_iters = 500
        q_model_optimizer = Adam(q_model.parameters(), lr=1e-1)
        
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

        # p_test = torch.zeros(x_test.shape[0], 2)
        # # for _ in np.arange(num_variational_samples):
        # #     p_test += torch.softmax(bnn.forward(x_test, do_sample=True), dim=1)
        # # p_test /= num_variational_samples

        p_test_ensemble = get_posterior_predictive_samples(model, inv_factors, x_test, num_samples=100)
        p_test = p_test_ensemble.mean(dim=0).detach()

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
        print("Laplace Approximation BNN decision cost: ", bnn_decision_cost_mean)

        variational_bnn_decision_cost_list.append(bnn_decision_cost_mean)
        q_model_decision_cost_list.append(q_model_decision_cost_mean)

    print("Mean decision cost - Laplace Approximation BNN: ", np.mean(np.array(variational_bnn_decision_cost_list)))
    print("Mean decision cost - Q model: ", np.mean(np.array(q_model_decision_cost_list)))
    print("Std dev decision cost - Laplace Approximation BNN: ", np.std(np.array(variational_bnn_decision_cost_list)))
    print("Std dev decision cost - Q model: ", np.std(np.array(q_model_decision_cost_list)))
