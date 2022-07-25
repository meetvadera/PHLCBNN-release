import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data.dataset import Dataset

global device
device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
BETA = 1000.


def noise_loss(net, lr, alpha):
    noise_loss = 0.0
    noise_std = (2 / lr * alpha) ** 0.5
    for var in net.parameters():
        means = torch.zeros(var.size()).to(device)
        noise_loss += torch.sum(var * torch.normal(means, std=noise_std)).to(device)
    return noise_loss


def get_validation_proba(model, val_data, minibatch_size=128):
    val_proba_array = []
    num_minibatch_iterations = np.ceil(len(val_data) / minibatch_size)
    with torch.no_grad():
        for iteration in range(int(num_minibatch_iterations)):
            if iteration != num_minibatch_iterations - 1:
                minibatch_data = val_data[iteration * minibatch_size: (iteration + 1) * minibatch_size].to(device)
            else:
                minibatch_data = val_data[iteration * minibatch_size:].to(device)
            batch_logits = model(minibatch_data)
            batch_proba = F.log_softmax(batch_logits, dim=-1).exp().cpu().detach()
            val_proba_array.append(batch_proba)
    return torch.cat(val_proba_array)


def test_model(model, val_loader, cost_matrix, num_classes):
    model.eval()
    with torch.no_grad():
        val_proba = torch.zeros(len(val_loader.dataset), num_classes)
        val_labels = torch.zeros(len(val_loader.dataset)).long()
        start_idx = 0
        total_val_loss = 0.
        total_correct_predictions = 0
        for batch_idx, (batch_data, batch_labels) in enumerate(val_loader):
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            batch_logits = model(batch_data)
            val_proba[start_idx: start_idx + len(batch_data)] = F.log_softmax(batch_logits, dim=-1).exp().cpu()
            val_labels[start_idx: start_idx + len(batch_data)] = batch_labels.cpu()
            total_val_loss += F.cross_entropy(batch_logits, batch_labels) * len(batch_data)
            decision_cost = torch.matmul(F.log_softmax(batch_logits, dim=-1).exp().unsqueeze(dim=1),
                                         cost_matrix).squeeze()
            decision_softmax = F.log_softmax(-BETA * decision_cost, dim=-1).exp()
            preds = decision_softmax.max(dim=1)[1]
            total_correct_predictions += (preds == batch_labels).sum().cpu().item()
            start_idx = start_idx + len(batch_data)
    weighted_f1_score = f1_score(val_labels.numpy(), val_proba.max(dim=1)[1].numpy().squeeze(), average='weighted')
    micro_f1_score = f1_score(val_labels.numpy(), val_proba.max(dim=1)[1].numpy().squeeze(), average='micro')
    val_loss = total_val_loss / len(val_loader.dataset)
    val_accuracy = total_correct_predictions / len(val_loader.dataset)
    return val_loss, val_accuracy, val_proba, weighted_f1_score, micro_f1_score


## Below function is for the case where the last class is NOT absent during training
def test_model_with_rejection(model, val_loader, cost_matrix, num_classes):
    model.eval()
    with torch.no_grad():
        val_proba = torch.zeros(len(val_loader.dataset), num_classes)
        val_labels = torch.zeros(len(val_loader.dataset)).long()
        start_idx = 0
        for batch_idx, (batch_data, batch_labels) in enumerate(val_loader):
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            batch_logits = model(batch_data)
            val_proba[start_idx: start_idx + len(batch_data)] = F.log_softmax(batch_logits, dim=-1).exp().cpu()
            val_labels[start_idx: start_idx + len(batch_data)] = batch_labels.cpu()
            start_idx += len(batch_data)
    decision_cost = torch.matmul(val_proba.to(device).unsqueeze(dim=1),
                                 cost_matrix).squeeze().cpu()
    decision_softmax = F.log_softmax(-BETA * decision_cost, dim=-1).exp()
    preds = decision_softmax.max(dim=1)[1]
    non_reject_indices = (preds != num_classes).nonzero().squeeze()
    reject_indices = (preds == num_classes).nonzero().squeeze()
    val_loss = F.nll_loss(val_proba.log()[non_reject_indices], val_labels[non_reject_indices])
    total_correct_predictions = (preds[non_reject_indices] == val_labels[non_reject_indices]).sum().cpu().item()
    total_rejected_samples = len(reject_indices) if reject_indices.dim() != 0 else 0
    total_non_rejected_samples = len(non_reject_indices)
    if total_non_rejected_samples != 0:
        val_accuracy = total_correct_predictions / total_non_rejected_samples
    else:
        val_accuracy = 0.
    return val_loss, val_accuracy, val_proba, total_rejected_samples, reject_indices, non_reject_indices


## Below function is for the case where the last class is absent during training
# def test_model_with_rejection(model, val_loader, cost_matrix, num_classes):
#     model.eval()
#     with torch.no_grad():
#         val_proba = torch.zeros(len(val_loader.dataset), num_classes)
#         val_labels = torch.zeros(len(val_loader.dataset)).long()
#         start_idx = 0
#         for batch_idx, (batch_data, batch_labels) in enumerate(val_loader):
#             batch_data = batch_data.to(device)
#             batch_labels = batch_labels.to(device)
#             batch_logits = model(batch_data)
#             val_proba[start_idx: start_idx + len(batch_data)] = F.log_softmax(batch_logits, dim=-1).exp().cpu()
#             val_labels[start_idx: start_idx + len(batch_data)] = batch_labels.cpu()
#             start_idx += len(batch_data)
#     decision_cost = torch.matmul(val_proba.to(device).unsqueeze(dim=1),
#                                  cost_matrix).squeeze().cpu()
#     decision_softmax = F.log_softmax(-BETA * decision_cost, dim=-1).exp()
#     preds = decision_softmax.max(dim=1)[1]
#     non_reject_indices = (preds != num_classes).nonzero().squeeze()
#     reject_indices = (preds == num_classes).nonzero().squeeze()
#     non_rejected_val_proba = val_proba[non_reject_indices]
#     non_rejected_labels = val_labels[non_reject_indices]
#     val_loss = F.nll_loss(non_rejected_val_proba.log()[(non_rejected_labels != num_classes).nonzero().squeeze()],
#                           non_rejected_labels[(non_rejected_labels != num_classes).nonzero().squeeze()])
#     # val_loss = F.nll_loss(val_proba.log()[non_reject_indices], val_labels[non_reject_indices])
#     ood_preds = torch.zeros_like(val_labels)
#     ood_preds[(preds == num_classes).nonzero().squeeze()] = 1
#     ood_labels = torch.zeros_like(val_labels)
#     ood_labels[(val_labels == num_classes).nonzero().squeeze()] = 1
#     ood_precision_score = precision_score(ood_labels.numpy().squeeze(), ood_preds.numpy().squeeze())
#     ood_recall_score = recall_score(ood_labels.numpy().squeeze(), ood_preds.numpy().squeeze())
#     total_correct_predictions = (preds[non_reject_indices] == val_labels[non_reject_indices]).sum().cpu().item()
#     total_rejected_samples = len(reject_indices) if reject_indices.dim() != 0 else 0
#     total_non_rejected_samples = len(non_reject_indices)
#     val_accuracy = total_correct_predictions / total_non_rejected_samples
#     return val_loss, val_accuracy, val_proba, total_rejected_samples, reject_indices, non_reject_indices, \
#            ood_precision_score, ood_recall_score

def test_model_tf_dataset(model, val_ds, cost_matrix, num_samples, num_classes):
    model.eval()
    with torch.no_grad():
        val_proba = torch.zeros(num_samples, num_classes)
        val_labels = torch.zeros(num_samples).long()
        start_idx = 0
        total_val_loss = 0.
        total_correct_predictions = 0
        for batch_idx, (batch_data, batch_labels) in enumerate(val_ds.as_numpy_iterator()):
            batch_data = torch.tensor(batch_data).transpose(2, 3).transpose(1, 2).to(device)
            batch_labels = torch.tensor(batch_labels).long().to(device)
            batch_logits = model(batch_data)
            val_proba[start_idx: start_idx + len(batch_data)] = F.log_softmax(batch_logits, dim=-1).exp().cpu()
            val_labels[start_idx: start_idx + len(batch_data)] = batch_labels.cpu()
            total_val_loss += F.cross_entropy(batch_logits, batch_labels) * len(batch_data)
            decision_cost = torch.matmul(F.log_softmax(batch_logits, dim=-1).exp().unsqueeze(dim=1),
                                         cost_matrix).squeeze()
            decision_softmax = F.log_softmax(-BETA * decision_cost, dim=-1).exp()
            preds = decision_softmax.max(dim=1)[1]
            total_correct_predictions += (preds == batch_labels).sum().cpu().item()
            start_idx = start_idx + len(batch_data)
    weighted_f1_score = f1_score(val_labels.numpy(), val_proba.max(dim=1)[1].numpy().squeeze(), average='weighted')
    micro_f1_score = f1_score(val_labels.numpy(), val_proba.max(dim=1)[1].numpy().squeeze(), average='micro')
    val_loss = total_val_loss / num_samples
    val_accuracy = total_correct_predictions / num_samples
    return val_loss, val_accuracy, val_proba, weighted_f1_score, micro_f1_score, val_labels


def test_model_target(model, val_loader, target):
    model.eval()
    with torch.no_grad():
        total_target_val_loss = 0.
        total_target_correct_predictions = 0
        num_targets = 0
        for batch_idx, (batch_data, batch_labels) in enumerate(val_loader):
            target_indices = (batch_labels == target).nonzero()
            if len(target_indices) > 0:
                num_targets += len(target_indices)
                batch_data = batch_data[target_indices].to(device)
                batch_labels = batch_labels[target_indices].to(device)
                batch_logits = model(batch_data)
                total_target_val_loss += F.cross_entropy(batch_logits, batch_labels) * len(batch_data)
                preds = batch_logits.max(dim=1)[1]
                total_target_correct_predictions += (preds == batch_labels).sum().cpu().item()
    target_val_loss = total_target_val_loss / num_targets
    target_val_accuracy = total_target_correct_predictions / num_targets
    return target_val_loss, target_val_accuracy


def targeted_nll(proba, labels, target):
    target_indices = (labels == target).nonzero()
    return F.nll_loss(proba[target_indices].log().to(device).squeeze(),
                      labels[target_indices].to(device).squeeze()).cpu().item()


def downsample_class(dataset: Dataset, target, downsample_factor):
    mask = torch.ones(len(dataset.targets)).bool()
    target_indices = (dataset.targets == target).nonzero()
    downsampled_num_targets = int((1 - downsample_factor) * len(target_indices))
    mask[target_indices[:downsampled_num_targets]] = False
    dataset.data = dataset.data[mask]
    dataset.targets = dataset.targets[mask]
    return dataset

def remove_class(dataset: Dataset, target_class):
    mask = torch.ones(len(dataset.targets)).bool()
    dataset.targets = torch.tensor(dataset.targets).long().squeeze()
    target_indices = (dataset.targets == target_class).nonzero()
    mask[target_indices] = False
    dataset.data = dataset.data[mask]
    dataset.targets = dataset.targets[mask]
    return dataset

def targeted_f1(preds, labels, target):
    return f1_score(labels.cpu().numpy().squeeze(), preds.cpu().numpy().squeeze(), labels=[target], average=None)[0]


class IndexDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]

        return data, target, torch.tensor(index)

    def __len__(self):
        return len(self.dataset)


def get_ece(preds, targets, n_bins=15):
    """
    ECE ported from Asukha et al., 2020.
    :param preds: Prediction probabilities in a Numpy array
    :param targets: Targets in a numpy array
    :param n_bins: Total number of bins to use.
    :return: Expected calibration error.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences, predictions = np.max(preds, 1), np.argmax(preds, 1)
    accuracies = (predictions == targets)

    ece = 0.0
    avg_confs_in_bins = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            delta = avg_confidence_in_bin - accuracy_in_bin
            avg_confs_in_bins.append(delta)
            ece += np.abs(delta) * prop_in_bin
        else:
            avg_confs_in_bins.append(None)
    # For reliability diagrams, also need to return these:
    # return ece, bin_lowers, avg_confs_in_bins
    return ece


class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """

    def __init__(self, tensors, transform=None):
        self.targets = tensors[1]
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)
