import argparse
import csv

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.sgd import SGD
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import models
from transforms import get_transforms
from utils import test_model, downsample_class, get_ece

parser = argparse.ArgumentParser(description='Arguments for the experiment')
parser.add_argument('--model', type=str, default='mlp1_200', help='Model name (default: mlp1_200)')
parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset (default: MNIST)')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate to use for teacher (default: 1e-3)')
parser.add_argument('--wd', type=float, default=1e-4,
                    help='Learning rate to use for training teacher (default: 1e-4)')
parser.add_argument('--label_corruption_proportion', type=float, default=0.,
                    help='Amount of label corruption noise to apply on dataset (default: 0.0)')
parser.add_argument('--data_path', type=str, default='../data/',
                    help='Directory in which to store datasets (default: ../data/')
parser.add_argument('--num_workers', type=int, default=0, help='Num workers for DataLoader (default: 0)')
parser.add_argument('--batch_size', type=int, default=64, help='Minibatch size (default: 64)')
parser.add_argument('--num_epochs', type=int, default=100,
                    help='Number of epochs for distilling posterior predictive distribution (default:100)')
parser.add_argument('--downsample_factor', type=float, default=1.0,
                    help='Downsampling factor for low cost classes (default: 1.0)')
parser.add_argument('--subsample_factor', type=float, default=1.,
                    help='Proportion of data to be subsampled from the training set (default: 1.0)')
parser.add_argument('--save_path', type=str, default=None,
                    help='Directory in which to checkpoint model (default: None)')
parser.add_argument('--seed', type=int, default=10,
                    help='Random seed for experiment (default: 10)')
parser.add_argument('--use_loss_calibration', type=int, default=False,
                    help='Use loss-calibrated objective for SGD (default: False)')
args = parser.parse_args()

print(vars(args))
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if args.dataset == 'MNIST':
    transforms = get_transforms(dataset=args.dataset)
    train_set = torchvision.datasets.MNIST(root=args.data_path, train=True, transform=transforms['train'],
                                           download=True)
    test_set = torchvision.datasets.MNIST(root=args.data_path, train=False, transform=transforms['test'], download=True)
elif args.dataset == 'CIFAR10':
    transforms = get_transforms(dataset=args.dataset)
    train_set = torchvision.datasets.CIFAR10(root=args.data_path, train=True, transform=transforms['train'],
                                             download=True)
    test_set = torchvision.datasets.CIFAR10(root=args.data_path, train=False, transform=transforms['test'],
                                            download=True)
else:
    raise NotImplementedError("%s dataset not available at present" % args.dataset)

if args.subsample_factor < 1:
    subsampled_length = int(args.subsample_factor * len(train_set))
    mask = torch.zeros(len(train_set.targets)).bool()
    mask[torch.randperm(len(train_set))[:subsampled_length]] = True
    train_set.data = train_set.data[mask]
    train_set.targets = train_set.targets[mask]

if args.downsample_factor < 1.:
    train_set = downsample_class(train_set, target=3, downsample_factor=args.downsample_factor)
    train_set = downsample_class(train_set, target=8, downsample_factor=args.downsample_factor)

num_targets = len(train_set.targets)

train_set.targets = torch.tensor(train_set.targets)
test_set.targets = torch.tensor(test_set.targets)
if args.label_corruption_proportion > 0:
    num_corrupt_labels = int(args.label_corruption_proportion * num_targets)
    corrupt_labels = torch.randint(low=0, high=torch.max(torch.tensor(train_set.targets).squeeze()).item() + 1,
                                   size=(num_corrupt_labels,))
    train_set.targets[-num_corrupt_labels:] = corrupt_labels

if args.label_corruption_proportion > 0:
    num_corrupt_labels = int(args.label_corruption_proportion * num_targets)
    corrupt_labels = torch.randint(low=0, high=torch.max(train_set.targets).item() + 1, size=(num_corrupt_labels,))
    train_set.targets[-num_corrupt_labels:] = corrupt_labels

num_classes = int(max(train_set.targets)) + 1
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = getattr(models, args.model)().to(args.device)

BETA = 1000.
if args.dataset == 'MNIST':
    cost_matrix = torch.eye(10, 10, requires_grad=False)
    cost_matrix = 1 - cost_matrix
    cost_matrix[:, [3, 8]] = 0.7
    cost_matrix[3, 3] = 0.
    cost_matrix[8, 8] = 0.
    cost_matrix = cost_matrix.to(args.device)
    ce_weights = torch.tensor([1., 1., 1., 1.4, 1., 1., 1., 1., 1.4, 1.]).to(args.device)
elif args.dataset == 'CIFAR10':
    cost_matrix = torch.eye(10, 10, requires_grad=False)
    cost_matrix = 1 - cost_matrix
    cost_matrix[:, [1, 9]] = 0.7
    cost_matrix[1, 1] = 0.
    cost_matrix[9, 9] = 0.
    cost_matrix = cost_matrix.to(args.device)
    ce_weights = torch.tensor([1., 1.4, 1., 1., 1., 1., 1., 1., 1., 1.4]).to(args.device)

#Uncomment the line below if you want to use class weights.
#ce_weights = torch.ones_like(ce_weights)
optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
# optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
#optimizer_scheduler = OneCycleLR(optimizer, max_lr=0.05, epochs=args.num_epochs, steps_per_epoch=len(train_loader))
#optimizer_scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * args.num_epochs)
optimizer_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=len(train_loader) * 20)

teacher_model_ensemble_proba = []
output_results_list = []
output_results_list += [vars(args)[key] for key in sorted(vars(args).keys())]
print("Args keys order: ", sorted(vars(args).keys()))

for epoch in tqdm(range(int(args.num_epochs))):
    model.train()
    total_epoch_ce_loss = 0.
    total_epoch_decision_cost = 0.
    for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):
        batch_data = batch_data.to(args.device)
        batch_labels = batch_labels.to(args.device)
        batch_logits = model(batch_data)
        #ce_loss = F.cross_entropy(batch_logits, batch_labels, weight=ce_weights)
        ce_loss = F.cross_entropy(batch_logits, batch_labels)
        decision_cost = torch.matmul(F.log_softmax(batch_logits, dim=-1).exp().unsqueeze(dim=1), cost_matrix).squeeze()
        decision_softmax = F.log_softmax(-BETA * decision_cost, dim=-1).exp()
        argmaxed_decision_cost_mean = (decision_cost * decision_softmax).sum(dim=-1).mean()
        total_loss = ce_loss + argmaxed_decision_cost_mean
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        optimizer_scheduler.step()
        total_epoch_ce_loss += ce_loss.cpu().item() * len(batch_data)
        total_epoch_decision_cost += argmaxed_decision_cost_mean.cpu().item() * len(batch_data)
    avg_epoch_ce_loss = total_epoch_ce_loss / len(train_loader.dataset)
    avg_epoch_decision_cost = total_epoch_decision_cost / len(train_loader.dataset)
    print("Epoch: %d, avg classification loss: %f, avg decision cost: %f\n" %
          (epoch, avg_epoch_ce_loss, avg_epoch_decision_cost))
    if epoch % 5 == 0:
        val_loss, val_accuracy, val_proba, weighted_f1_score, micro_f1_score = test_model(model, test_loader,
                                                                                          cost_matrix, num_classes)
        decision_cost = torch.matmul(val_proba.to(args.device).unsqueeze(dim=1), cost_matrix).squeeze()
        decision_softmax = F.log_softmax(-BETA * decision_cost, dim=-1).exp()
        decision_cost_mean = (decision_cost * decision_softmax).sum(dim=-1).mean()
        teacher_model_ensemble_proba.append(val_proba)
        print("Epoch: %d, teacher test loss: %f, teacher test accuracy: %f, teacher decision cost: %f\n" % (
            epoch, val_loss, val_accuracy, decision_cost_mean.cpu().item()))
        if args.save_path is not None:
            torch.save(model.state_dict(), args.save_path + '%s_%s_%s_%s_lc.pt' % (args.dataset, args.model,
                                                                                   args.label_corruption_proportion,
                                                                                   args.seed))

val_loss, val_accuracy, val_proba, weighted_f1_score, micro_f1_score = test_model(model, test_loader, cost_matrix,
                                                                                  num_classes)
val_proba = val_proba.to(args.device)
decision_cost = torch.matmul(val_proba.unsqueeze(dim=1), cost_matrix).squeeze()
decision_softmax = F.log_softmax(-BETA * decision_cost, dim=-1).exp()
A = np.expand_dims(torch.matmul(decision_softmax, cost_matrix.transpose(0, 1)).cpu().numpy(), 1)
B = np.expand_dims(F.one_hot(test_set.targets.long(), num_classes=num_classes).cpu().numpy().T, 0)
B = np.matmul(A, B.T)
# decision_cost_mean = (decision_cost * decision_softmax).sum(dim=-1).mean()
decision_cost_mean = B.squeeze().mean()
ece = get_ece(val_proba.cpu().numpy(), test_set.targets.cpu().numpy())
output_results_list += [val_loss.cpu().item(), val_accuracy, decision_cost_mean, weighted_f1_score,
                        micro_f1_score]