import argparse
import csv

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from sklearn.metrics import f1_score
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, OneCycleLR
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import models
from optimizers import SGHMC
from transforms import get_transforms
from utils import test_model, downsample_class, targeted_nll, targeted_f1, get_ece

parser = argparse.ArgumentParser(description='Arguments for the experiment')
parser.add_argument('--model', type=str, default='mlp1_200', help='Model name (default: mlp1_100)')
parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset (default: MNIST)')
parser.add_argument('--use_val_split', dest='use_val_split', action='store_true',
                    help='hold out a validation set from train set (default: False)')
parser.add_argument('--val_proportion', type=float, default=0.2,
                    help='Proportion of dataset to use as validation for retrain')
parser.add_argument('--use_loss_calibration_teacher', dest='use_loss_calibration_teacher', action='store_true',
                    help='Use calibrated posterior while sampling from teacher')
parser.add_argument('--use_loss_calibration_student', dest='use_loss_calibration_student', action='store_true',
                    help='Use calibrated loss while training student')
parser.add_argument('--run_baseline_only', dest='run_baseline_only', action='store_true',
                    help='Run baseline distillation only')
parser.add_argument('--alpha', type=float, default=1., help='Alpha term used for SGHMC (default: 1.)')
parser.add_argument('--teacher_lr', type=float, default=1e-3, help='Learning rate to use for teacher (default: 1e-3)')
parser.add_argument('--student_lr', type=float, default=1e-3, help='Learning rate to use for teacher (default: 1e-3)')
parser.add_argument('--teacher_wd', type=float, default=1e-4,
                    help='Learning rate to use for training teacher (default: 1e-4)')
parser.add_argument('--student_wd', type=float, default=1e-4,
                    help='Learning rate to use for training student (default: 1e-4)')
parser.add_argument('--label_corruption_proportion', type=float, default=0.,
                    help='Amount of label corruption noise to apply on dataset (default: 0.)')
parser.add_argument('--data_path', type=str, default='../data/',
                    help='Directory in which to store datasets (default: ../data/')
parser.add_argument('--num_workers', type=int, default=0, help='Num workers for DataLoader (default: 0)')
parser.add_argument('--batch_size', type=int, default=64, help='Minibatch size (default: 64)')
parser.add_argument('--num_distillation_epochs', type=int, default=100,
                    help='Number of epochs for distilling posterior predictive distribution (default:100)')
parser.add_argument('--num_epochs_q_training', type=int, default=100,
                    help='Number of epochs for training the q network (default: 100)')
parser.add_argument('--downsample_factor', type=float, default=1.0,
                    help='Downsampling factor for low cost classes (default: 1.0)')
parser.add_argument('--subsample_factor', type=float, default=1.,
                    help='Proportion of data to be subsampled from the training set (default: 1.0)')
parser.add_argument('--burn_in_iters', type=int, default=0,
                    help='Number of burn in iterations for sampler before starting distillation (default: 0)')
parser.add_argument('--data_noise', type=float, default=0.01,
                    help='Scale of gaussian perturbation to training set for q_training (default: 0.01)')
parser.add_argument('--checkpoint_path', type=str, default=None,
                    help='Path to a pretrained model (default: None)')
parser.add_argument('--seed', type=int, default=10,
                    help='Random seed for experiment (default: 10)')
parser.add_argument('--pretrain_epochs', type=int, default=None,
                    help='Number of epochs for pre-training teacher model (default: 50)')
parser.add_argument('--pretrain_lr', type=float, default=1e-3,
                    help='Pre-training learning rate (default: 1e-3)')
parser.add_argument('--pretrain_wd', type=float, default=1e-4,
                    help='Pre-training weight decay (default: 1e-4)')
parser.add_argument('--save_path', type=str, default=None,
                    help='Directory to save the pre-trained model (default: None)')
parser.add_argument('--load_path', type=str, default=None,
                    help='Directory to load the pre-trained model from (default: None)')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(vars(args))
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

num_targets = len(train_set)
train_set.targets = torch.tensor(train_set.targets)
test_set.targets = torch.tensor(test_set.targets)
num_classes = int(max(train_set.targets)) + 1
if args.label_corruption_proportion > 0:
    num_corrupt_labels = int(args.label_corruption_proportion * num_targets)
    corrupt_labels = torch.randint(low=0, high=torch.max(torch.tensor(train_set.targets).squeeze()).item() + 1,
                                   size=(num_corrupt_labels,))
    train_set.targets[-num_corrupt_labels:] = corrupt_labels

if args.downsample_factor < 1.:
    train_set = downsample_class(train_set, target=3, downsample_factor=args.downsample_factor)
    train_set = downsample_class(train_set, target=8, downsample_factor=args.downsample_factor)
    # val_set = downsample_class(val_set, target=3, downsample_factor=args.downsample_factor)
    # val_set = downsample_class(val_set, target=8, downsample_factor=args.downsample_factor)

num_targets = len(train_set)
if args.use_val_split and args.val_proportion < 1:
    val_set_len = int(num_targets * args.val_proportion)
    split_datasets = random_split(train_set, lengths=[num_targets - val_set_len, val_set_len])
    train_set = split_datasets[0]
    val_set = split_datasets[1]
else:
    val_set = train_set

# train_set = IndexDataset(train_set)
# val_set = IndexDataset(val_set)
# test_set = IndexDataset(test_set)
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

teacher_model = getattr(models, args.model)().to(args.device)
student_model = getattr(models, args.model)().to(args.device)
q_model = getattr(models, args.model)().to(args.device)

if args.checkpoint_path is not None:
    teacher_model.load_state_dict(torch.load(args.checkpoint_path))

BETA = 1000.
if args.dataset == 'MNIST':
    cost_matrix = torch.eye(10, 10, requires_grad=False)
    cost_matrix = 1 - cost_matrix
    cost_matrix[:, [3, 8]] = 0.7
    cost_matrix[3, 3] = 0.
    cost_matrix[8, 8] = 0.
    cost_matrix = cost_matrix.to(args.device)
elif args.dataset == 'CIFAR10':
    cost_matrix = torch.eye(10, 10, requires_grad=False)
    cost_matrix = 1 - cost_matrix
    cost_matrix[:, [1, 9]] = 0.7
    cost_matrix[1, 1] = 0.
    cost_matrix[9, 9] = 0.
    cost_matrix = cost_matrix.to(args.device)

teacher_optimizer = SGHMC(teacher_model.parameters(), lr=args.teacher_lr, weight_decay=args.teacher_wd,
                          momentum=1 - args.alpha, num_training_samples=len(train_loader.dataset))
student_optimizer = SGD(student_model.parameters(), lr=args.student_lr, weight_decay=args.student_wd,
                        momentum=0.9)
q_model_optimizer = SGD(q_model.parameters(), lr=args.student_lr, weight_decay=0.,
                        momentum=0.9)
teacher_optimizer_scheduler = CosineAnnealingLR(teacher_optimizer,
                                                T_max=len(train_loader) * args.num_distillation_epochs,
                                                #                                                eta_min=args.teacher_lr / 2
                                                )
student_optimizer_scheduler = OneCycleLR(student_optimizer, max_lr=0.05,
                                         total_steps=(len(
                                             train_loader) * args.num_distillation_epochs) - args.burn_in_iters + 1)
q_model_optimizer_scheduler = OneCycleLR(q_model_optimizer, max_lr=0.05, epochs=args.num_epochs_q_training,
                                         steps_per_epoch=len(val_loader))
if args.use_loss_calibration_teacher:
    teacher_decision_cost_factor = 1.
else:
    teacher_decision_cost_factor = 0.

if args.use_loss_calibration_student:
    student_decision_cost_factor = 1.
else:
    student_decision_cost_factor = 0.

if args.pretrain_epochs is not None:
    pretrain_optimizer = SGD(teacher_model.parameters(), lr=args.pretrain_lr, weight_decay=args.pretrain_wd,
                             momentum=0.9)
    pretrain_optimizer_scheduler = CosineAnnealingWarmRestarts(pretrain_optimizer, T_0=len(train_loader) * 20)
    for epoch in tqdm(range(int(args.pretrain_epochs))):
        teacher_model.train()
        total_epoch_ce_loss = 0.
        total_epoch_decision_cost = 0.
        total_epoch_student_loss = 0.
        for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):
            batch_data = batch_data.to(args.device)
            batch_labels = batch_labels.to(args.device)
            batch_logits = teacher_model(batch_data)
            ce_loss = F.cross_entropy(batch_logits, batch_labels)
            decision_cost = torch.matmul(F.log_softmax(batch_logits, dim=-1).exp().unsqueeze(dim=1),
                                         cost_matrix).squeeze()
            decision_softmax = F.log_softmax(-BETA * decision_cost, dim=-1).exp().detach()
            argmaxed_decision_cost_mean = (decision_cost * decision_softmax).sum(dim=-1).mean()
            total_loss = ce_loss + teacher_decision_cost_factor * argmaxed_decision_cost_mean
            pretrain_optimizer.zero_grad()
            total_loss.backward()
            pretrain_optimizer.step()
            pretrain_optimizer_scheduler.step()
            total_epoch_ce_loss += ce_loss.cpu().item() * len(batch_data)
        print("Epoch: %d, avg classification loss: %f,\n" %
              (epoch, total_epoch_ce_loss / len(train_loader.dataset)))
        if epoch % 5 == 0 or epoch == args.pretrain_epochs - 1:
            val_loss, val_accuracy, val_proba, weighted_f1_score, micro_f1_score = test_model(teacher_model,
                                                                                              test_loader,
                                                                                              cost_matrix, num_classes)
            decision_cost = torch.matmul(val_proba.to(args.device).unsqueeze(dim=1), cost_matrix).squeeze()
            decision_softmax = F.log_softmax(-BETA * decision_cost, dim=-1).exp()
            decision_cost_mean = (decision_cost * decision_softmax).sum(dim=-1).mean()
            print("Epoch: %d, teacher test loss: %f, teacher test accuracy: %f, teacher decision cost: %f\n" % (
                epoch, val_loss, val_accuracy, decision_cost_mean.cpu().item()))
    if args.save_path is not None:
        torch.save(teacher_model.state_dict(), args.save_path + '%s_%s_seed_%d_corruption_%f.pt' %
                   (args.model, args.dataset, args.seed, args.label_corruption_proportion))


if args.load_path is not None:
    teacher_model.load_state_dict(torch.load(args.load_path + '%s_%s_seed_%d_corruption_%f.pt' %
                   (args.model, args.dataset, args.seed, args.label_corruption_proportion)))

teacher_model_ensemble_proba = []
output_results_list = []
output_results_list += [vars(args)[key] for key in sorted(vars(args).keys())]
print("Args keys order: ", sorted(vars(args).keys()))
t = 0
for epoch in tqdm(range(int(args.num_distillation_epochs))):
    teacher_model.train()
    student_model.train()
    total_epoch_ce_loss = 0.
    total_epoch_decision_cost = 0.
    total_epoch_student_loss = 0.
    for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):
        t += 1
        batch_data = batch_data.to(args.device)
        batch_labels = batch_labels.to(args.device)
        batch_logits = teacher_model(batch_data)
        ce_loss = F.cross_entropy(batch_logits, batch_labels)
        decision_cost = torch.matmul(F.log_softmax(batch_logits, dim=-1).exp().unsqueeze(dim=1), cost_matrix).squeeze()
        decision_softmax = F.log_softmax(-BETA * decision_cost, dim=-1).exp().detach()
        argmaxed_decision_cost_mean = (decision_cost * decision_softmax).sum(dim=-1).mean()
        total_loss = ce_loss + teacher_decision_cost_factor * argmaxed_decision_cost_mean
        teacher_optimizer.zero_grad()
        total_loss.backward()
        teacher_optimizer.step()
        #teacher_optimizer_scheduler.step()
        total_epoch_ce_loss += ce_loss.cpu().item() * len(batch_data)
        if t > args.burn_in_iters:
            student_model_batch_logits = student_model(batch_data)
            student_decision_cost = torch.matmul(
                F.log_softmax(student_model_batch_logits, dim=-1).exp().unsqueeze(dim=1),
                cost_matrix).squeeze()
            student_decision_softmax = F.log_softmax(-BETA * student_decision_cost, dim=-1).exp().detach()
            student_argmaxed_decision_cost_mean = (student_decision_cost * student_decision_softmax).sum(dim=-1).mean()
            teacher_model_batch_proba = F.log_softmax(teacher_model(batch_data), dim=-1).exp()
            student_loss = -torch.sum(teacher_model_batch_proba *
                                      F.log_softmax(student_model_batch_logits, dim=-1), dim=1).mean() + \
                           student_argmaxed_decision_cost_mean * student_decision_cost_factor
            student_optimizer.zero_grad()
            student_loss.backward()
            student_optimizer.step()
            student_optimizer_scheduler.step()
            total_epoch_decision_cost += argmaxed_decision_cost_mean.cpu().item() * len(batch_data)
            total_epoch_student_loss += student_loss.cpu().item() * len(batch_data)
        avg_epoch_ce_loss = total_epoch_ce_loss / len(train_loader.dataset)
        avg_epoch_student_loss = total_epoch_student_loss / len(train_loader.dataset)
        avg_epoch_decision_cost = total_epoch_decision_cost / len(train_loader.dataset)
    if t > args.burn_in_iters:
        print("Epoch: %d, avg classification loss: %f, avg student loss: %f\n" % (
            epoch, avg_epoch_ce_loss, avg_epoch_student_loss))
        if epoch % 5 == 0:
            teacher_val_loss, teacher_val_accuracy, teacher_val_proba, teacher_weighted_f1_score, teacher_micro_f1_score = \
                test_model(teacher_model, test_loader, cost_matrix, num_classes)
            teacher_model_ensemble_proba.append(teacher_val_proba)
            student_val_loss, student_val_accuracy, student_val_proba, student_weighted_f1_score, student_micro_f1_score = \
                test_model(student_model, test_loader, cost_matrix, num_classes)
            print("Epoch: %d, teacher test loss: %f, teacher test accuracy: %f, teacher weighted f1 score: %f, "
                  "teacher micro f1 score: %f, student test loss: %f, student test accuracy: %f, "
                  "student weighted f1 score: %f, student micro f1 score: %f\n" % (
                      epoch, teacher_val_loss, teacher_val_accuracy, teacher_weighted_f1_score, teacher_micro_f1_score,
                      student_val_loss, student_val_accuracy, student_weighted_f1_score, student_micro_f1_score))

model_ensemble_proba = torch.stack(teacher_model_ensemble_proba[-30:]).mean(dim=0).to(args.device)
teacher_model_ensemble_decision_cost = torch.matmul(model_ensemble_proba.unsqueeze(dim=1), cost_matrix).squeeze()
teacher_model_ensemble_decision_softmax = F.log_softmax(-BETA * teacher_model_ensemble_decision_cost, dim=-1).exp()
##TODO: Give meaningful names below
A = np.expand_dims(torch.matmul(teacher_model_ensemble_decision_softmax, cost_matrix.transpose(0, 1)).cpu().numpy(), 1)
B = np.expand_dims(F.one_hot(test_set.targets.long(), num_classes=num_classes).cpu().numpy().T, 0)
B = np.matmul(A, B.T)
# teacher_model_ensemble_decision_cost_mean = (teacher_model_ensemble_decision_cost *
#                                              teacher_model_ensemble_decision_softmax).sum(dim=-1).mean()
teacher_model_ensemble_decision_cost_mean = B.squeeze().mean()

teacher_model_ensemble_proba = torch.stack(teacher_model_ensemble_proba[5:]).mean(dim=0).to(args.device)
teacher_model_ensemble_preds = teacher_model_ensemble_decision_softmax.max(dim=1)[1]
teacher_model_ensemble_accuracy = (teacher_model_ensemble_preds == test_set.targets.to(
    args.device)).sum().cpu().item() / len(test_set.targets)
teacher_model_ensemble_loss = F.nll_loss(teacher_model_ensemble_proba.log(), test_set.targets.to(args.device))

teacher_model_ensemble_weighted_f1_score = f1_score(test_set.targets.cpu().numpy().squeeze(),
                                                    teacher_model_ensemble_preds.cpu().numpy().squeeze(),
                                                    average='weighted')
teacher_model_ensemble_micro_f1_score = f1_score(test_set.targets.cpu().numpy().squeeze(),
                                                 teacher_model_ensemble_preds.cpu().numpy().squeeze(),
                                                 average='micro')
teacher_model_ensemble_ece = get_ece(teacher_model_ensemble_proba.cpu().numpy(), test_set.targets.cpu().numpy())
output_results_list += [teacher_model_ensemble_loss.cpu().item(), teacher_model_ensemble_accuracy,
                        teacher_model_ensemble_decision_cost_mean,
                        targeted_nll(teacher_model_ensemble_proba, test_set.targets.to(args.device), target=1),
                        targeted_nll(teacher_model_ensemble_proba, test_set.targets.to(args.device), target=9),
                        teacher_model_ensemble_weighted_f1_score, teacher_model_ensemble_micro_f1_score,
                        targeted_f1(teacher_model_ensemble_proba.max(dim=1)[1], test_set.targets, target=1),
                        targeted_f1(teacher_model_ensemble_proba.max(dim=1)[1], test_set.targets, target=9),
                        teacher_model_ensemble_ece]
val_loss, val_accuracy, val_proba, student_weighted_f1_score, student_micro_f1_score = \
    test_model(student_model, test_loader, cost_matrix, num_classes)
student_val_proba = val_proba.to(args.device)
student_decision_cost = torch.matmul(student_val_proba.unsqueeze(dim=1), cost_matrix).squeeze()
student_decision_softmax = F.log_softmax(-BETA * student_decision_cost, dim=-1).exp()
##TODO: Give meaningful names below
A = np.expand_dims(torch.matmul(student_decision_softmax, cost_matrix.transpose(0, 1)).cpu().numpy(), 1)
B = np.expand_dims(F.one_hot(test_set.targets.long(), num_classes=num_classes).cpu().numpy().T, 0)
B = np.matmul(A, B.T)
# student_decision_cost_mean = (student_decision_cost * student_decision_softmax).sum(dim=-1).mean()
student_decision_cost_mean = B.squeeze().mean()
student_ece = get_ece(student_val_proba.cpu().numpy(), test_set.targets.cpu().numpy())
output_results_list += [val_loss.cpu().item(), val_accuracy, student_decision_cost_mean,
                        targeted_nll(student_val_proba, test_set.targets.to(args.device), target=1),
                        targeted_nll(student_val_proba, test_set.targets.to(args.device), target=9),
                        student_weighted_f1_score, student_micro_f1_score,
                        targeted_f1(student_val_proba.max(dim=1)[1], test_set.targets, target=1),
                        targeted_f1(student_val_proba.max(dim=1)[1], test_set.targets, target=9),
                        student_ece]
if not args.run_baseline_only:
    # q_model.load_state_dict(student_model.state_dict())
    for epoch in tqdm(range(int(args.num_epochs_q_training))):
        student_model.eval()
        q_model.train()
        total_epoch_ce_loss = 0.
        total_epoch_decision_cost = 0.
        for batch_idx, (batch_data, batch_labels) in enumerate(val_loader):
            batch_data = batch_data.to(args.device)
            batch_labels = batch_labels.to(args.device)
            batch_data += torch.randn_like(batch_data, requires_grad=False) * args.data_noise
            batch_logits = q_model(batch_data)
            student_model_batch_logits = student_model(batch_data)
            ce_loss = -torch.sum(F.log_softmax(batch_logits, dim=-1).exp() *
                                 F.log_softmax(student_model_batch_logits, dim=-1), dim=1).mean()
            decision_cost = torch.matmul(F.log_softmax(batch_logits, dim=-1).exp().unsqueeze(dim=1),
                                         cost_matrix).squeeze()
            decision_softmax = F.log_softmax(-BETA * decision_cost, dim=-1).exp().detach()
            argmaxed_decision_cost_mean = (decision_cost * decision_softmax).sum(dim=-1).mean()
            entropy_loss = -(F.log_softmax(batch_logits, dim=-1).exp() *
                             F.log_softmax(batch_logits, dim=-1)).sum(dim=-1).mean()
            total_loss = ce_loss - entropy_loss + argmaxed_decision_cost_mean
            q_model_optimizer.zero_grad()
            total_loss.backward()
            q_model_optimizer.step()
            q_model_optimizer_scheduler.step()
            total_epoch_ce_loss += ce_loss.cpu().item() * len(batch_data)
            total_epoch_decision_cost += argmaxed_decision_cost_mean.cpu().item() * len(batch_data)
        avg_epoch_ce_loss = total_epoch_ce_loss / len(train_loader.dataset)
        avg_epoch_decision_cost = total_epoch_decision_cost / len(train_loader.dataset)
        print("Epoch: %d, avg classification loss: %f, avg decision cost: %f" %
              (epoch, avg_epoch_ce_loss, avg_epoch_decision_cost))
        if epoch % 5 == 0:
            val_loss, val_accuracy, val_proba, weighted_f1_score, micro_f1_score = test_model(q_model, test_loader,
                                                                                              cost_matrix, num_classes)
            print("Epoch: %d, avg test classification loss: %f, avg accuracy: %f" % (epoch, val_loss, val_accuracy))

    val_loss, val_accuracy, val_proba, weighted_f1_score, micro_f1_score = test_model(q_model, test_loader, cost_matrix,
                                                                                      num_classes)
    val_proba = val_proba.to(args.device)
    q_model_decision_cost = torch.matmul(val_proba.unsqueeze(dim=1), cost_matrix).squeeze()
    q_model_decision_softmax = F.log_softmax(-BETA * q_model_decision_cost, dim=-1).exp()
    ##TODO: Give meaningful names below
    A = np.expand_dims(torch.matmul(q_model_decision_softmax, cost_matrix.transpose(0, 1)).cpu().numpy(),
                       1)
    B = np.expand_dims(F.one_hot(test_set.targets.long(), num_classes=num_classes).cpu().numpy().T, 0)
    B = np.matmul(A, B.T)
    # q_model_decision_cost_mean = (q_model_decision_cost * q_model_decision_softmax).sum(dim=-1).mean()
    q_model_decision_cost_mean = B.squeeze().mean()
    q_model_ece = get_ece(val_proba.cpu().numpy(), test_set.targets.cpu().numpy())
    output_results_list += [val_loss.cpu().item(), val_accuracy, q_model_decision_cost_mean,
                            targeted_nll(val_proba, test_set.targets.to(args.device), target=1),
                            targeted_nll(val_proba, test_set.targets.to(args.device), target=9),
                            weighted_f1_score, micro_f1_score,
                            targeted_f1(val_proba.max(dim=1)[1], test_set.targets, target=1),
                            targeted_f1(val_proba.max(dim=1)[1], test_set.targets, target=9),
                            q_model_ece]

print(output_results_list)
