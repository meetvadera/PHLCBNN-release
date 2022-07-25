import argparse
import csv
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
from tqdm import tqdm

import models
from optimizers import SGHMC
from transforms import get_transforms_with_rotation
from utils import test_model_with_rejection

parser = argparse.ArgumentParser(description='Arguments for the experiment')
parser.add_argument('--model', type=str, default='mlp1_200', help='Model name (default: mlp1_100)')
parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset (default: MNIST)')
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
parser.add_argument('--data_path', type=str, default='../data/',
                    help='Directory in which to store datasets (default: ../data/')
parser.add_argument('--num_workers', type=int, default=0, help='Num workers for DataLoader (default: 0)')
parser.add_argument('--batch_size', type=int, default=64, help='Minibatch size (default: 64)')
parser.add_argument('--num_distillation_epochs', type=int, default=100,
                    help='Number of epochs for distilling posterior predictive distribution (default:100)')
parser.add_argument('--num_epochs_q_training', type=int, default=100,
                    help='Number of epochs for training the q network (default: 100)')
parser.add_argument('--burn_in_iters', type=int, default=0,
                    help='Number of burn in iterations for sampler before starting distillation (default: 0)')
parser.add_argument('--data_noise', type=float, default=0.01,
                    help='Scale of gaussian perturbation to training set for q_training (default: 0.01)')
parser.add_argument('--checkpoint_path', type=str, default=None,
                    help='Path to a pretrained model (default: None)')
parser.add_argument('--referral_cost', type=float, default=None, help='Referral cost')
parser.add_argument('--pretrain_epochs', type=int, default=None, help='No. of pretrain epochs')
parser.add_argument('--pretrain_lr', type=float, default=1e-3, help='Pre-train lr')
parser.add_argument('--pretrain_wd', type=float, default=1e-4, help='Pre-train wd')
parser.add_argument('--seed', type=int, default=10,
                    help='Random seed for experiment (default: 10)')
parser.add_argument('--save_path', type=str, default=None,
                    help='Save path for model (default: None)')
parser.add_argument('--temp', type=float, default=1.,
                    help='Temp used for distillation (default: 1)')
parser.add_argument('--student_model', type=str, default=None, help='Student model arch (default: None)')
parser.add_argument('--rotation_angle', type=float, default=45., help='Rotation angle limit (default: 45)')

args = parser.parse_args()

args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(vars(args))
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


print("Loading Data sets...\n")
if args.dataset == 'MNIST':
    transforms = get_transforms_with_rotation(dataset=args.dataset, rotation_angle=args.rotation_angle)
    train_set = torchvision.datasets.MNIST(root=args.data_path, train=True, transform=transforms['train'],
                                           download=True)
    test_set = torchvision.datasets.MNIST(root=args.data_path, train=False, transform=transforms['test'], download=True)

elif args.dataset == 'CIFAR10':
    transforms = get_transforms_with_rotation(dataset=args.dataset, rotation_angle=args.rotation_angle)
    train_set = torchvision.datasets.CIFAR10(root=args.data_path, train=True, transform=transforms['train'],
                                             download=True)
    test_set = torchvision.datasets.CIFAR10(root=args.data_path, train=False, transform=transforms['test'],
                                            download=True)
else:
    raise NotImplementedError("%s dataset not available at present" % args.dataset)

# train_set = remove_class(train_set, target_class=9)

num_classes = 10

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                          pin_memory=True)

student_train_loader = DataLoader(deepcopy(train_set), batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True)

student_train_data = []
student_train_labels = []

for i in range(10):
    for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):
        student_train_data.append(batch_data)
        student_train_labels.append(batch_labels)

student_train_data = torch.cat(student_train_data)
student_train_labels = torch.cat(student_train_labels)

train_data = []
train_labels = []

for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):
    train_data.append(batch_data)
    train_labels.append(batch_labels)

train_data = torch.cat(train_data)
train_labels = torch.cat(train_labels)

train_set = TensorDataset(train_data.squeeze(), train_labels.squeeze())
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                          pin_memory=True)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)

test_data = []
test_labels = []

for batch_idx, (batch_data, batch_labels) in enumerate(test_loader):
    test_data.append(batch_data)
    test_labels.append(batch_labels)

test_data = torch.cat(test_data)
test_labels = torch.cat(test_labels)

test_set = TensorDataset(test_data.squeeze(), test_labels.squeeze())
test_set.targets = test_labels.squeeze()
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)

print("Finished loading data sets...\n")

print("Initializing models...\n")

teacher_model = getattr(models, args.model)(num_classes=num_classes).to(args.device)
if args.student_model is None:
    args.student_model = args.model

student_model = getattr(models, args.student_model)(num_classes=num_classes).to(args.device)
q_model = getattr(models, args.student_model)(num_classes=num_classes).to(args.device)

print("Finished initializing models...\n")
if args.checkpoint_path is not None:
    teacher_model.load_state_dict(torch.load(args.checkpoint_path))

BETA = 1000.
if args.dataset == 'MNIST':
    cost_matrix = torch.eye(num_classes + 1, num_classes + 1, requires_grad=False)
    cost_matrix = 1 - cost_matrix
    # cost_matrix[:, [3, 8]] = 0.7
    # cost_matrix[3, 3] = 0.
    # cost_matrix[8, 8] = 0.
    cost_matrix[:, num_classes] = args.referral_cost
    cost_matrix = cost_matrix.to(args.device)
elif args.dataset == 'CIFAR10':
    cost_matrix = torch.eye(num_classes + 1, num_classes + 1, requires_grad=False)
    cost_matrix = 1 - cost_matrix
    # cost_matrix[:, [1, 9]] = 0.7
    # cost_matrix[1, 1] = 0.
    # cost_matrix[9, 9] = 0.
    cost_matrix[:, num_classes] = args.referral_cost
    cost_matrix = cost_matrix.to(args.device)

test_cost_matrix = deepcopy(cost_matrix)
# test_cost_matrix[9, 9] = 0.
cost_matrix = cost_matrix[:-1, :]

teacher_optimizer = SGHMC(teacher_model.parameters(), lr=args.teacher_lr, weight_decay=args.teacher_wd,
                          momentum=1 - args.alpha, num_training_samples=len(train_loader.dataset))
student_optimizer = SGD(student_model.parameters(), lr=args.student_lr, weight_decay=args.student_wd,
                        momentum=0.9)
q_model_optimizer = SGD(q_model.parameters(), lr=args.student_lr, weight_decay=args.student_wd,
                        momentum=0.9)
student_optimizer_scheduler = OneCycleLR(student_optimizer, max_lr=0.1,
                                         total_steps=(len(
                                             train_loader) * args.num_distillation_epochs) - args.burn_in_iters + 1)
# q_model_optimizer_scheduler = OneCycleLR(q_model_optimizer, max_lr=0.25, epochs=args.num_epochs_q_training,
#                                         steps_per_epoch=len(train_loader))
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
    # pretrain_optimizer_scheduler = CosineAnnealingWarmRestarts(pretrain_optimizer, T_0=len(train_loader) * 20)
    pretrain_optimizer_scheduler = OneCycleLR(pretrain_optimizer, max_lr=0.05, epochs=args.pretrain_epochs,
                                              steps_per_epoch=len(train_loader))
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
            _, teacher_val_accuracy, teacher_val_proba, _, _, _ = \
                test_model_with_rejection(teacher_model, test_loader, cost_matrix, num_classes)
            decision_cost = torch.matmul(teacher_val_proba.to(args.device).unsqueeze(dim=1), cost_matrix).squeeze()
            decision_softmax = F.log_softmax(-BETA * decision_cost, dim=-1).exp()
            decision_cost_mean = (decision_cost * decision_softmax).sum(dim=-1).mean()
            print("Epoch: %d, teacher test accuracy: %f, teacher decision cost: %f\n" % (
                epoch, teacher_val_accuracy, decision_cost_mean.cpu().item()))
    if args.save_path is not None:
        torch.save(teacher_model.state_dict(), args.save_path + '%s_%s_seed_%d_rotation_%f.pt' %
                   (args.model, args.dataset, args.seed, args.rotation_angle))

teacher_model_ensemble_proba = []
output_results_list = []
output_results_list += [vars(args)[key] for key in sorted(vars(args).keys())]
print("Args keys order: ", sorted(vars(args).keys()))
t = 0
print("Starting training...\n")
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
        decision_softmax = F.log_softmax(-BETA * decision_cost, dim=-1).exp()
        argmaxed_decision_cost_mean = (decision_cost * decision_softmax).sum(dim=-1).mean()
        total_loss = ce_loss + teacher_decision_cost_factor * argmaxed_decision_cost_mean
        teacher_optimizer.zero_grad()
        total_loss.backward()
        teacher_optimizer.step()
        total_epoch_ce_loss += ce_loss.cpu().item() * len(batch_data)
        if t > args.burn_in_iters:
            # for batch_data, _ in student_train_loader:
            batch_data = student_train_data[torch.randperm(len(student_train_labels))[:args.batch_size]].to(args.device)
            student_model_batch_logits = student_model(batch_data)
            teacher_model_batch_logits = teacher_model(batch_data)
            ce_loss = -torch.sum(F.log_softmax(student_model_batch_logits / args.temp, dim=-1).exp() *
                                 F.log_softmax(teacher_model_batch_logits / args.temp, dim=-1), dim=1).mean()
            decision_cost = torch.matmul(F.log_softmax(student_model_batch_logits, dim=-1).exp().unsqueeze(dim=1),
                                         cost_matrix).squeeze()
            decision_softmax = F.log_softmax(-BETA * decision_cost, dim=-1).exp()
            argmaxed_decision_cost_mean = (decision_cost * decision_softmax).sum(dim=-1).mean()
            entropy_loss = -(F.log_softmax(student_model_batch_logits, dim=-1).exp() *
                             F.log_softmax(student_model_batch_logits, dim=-1)).sum(dim=-1).mean()
            total_loss = ce_loss - entropy_loss + argmaxed_decision_cost_mean
            student_optimizer.zero_grad()
            total_loss.backward()
            student_optimizer.step()
            student_optimizer_scheduler.step()
            total_epoch_decision_cost += argmaxed_decision_cost_mean.cpu().item() * len(batch_data)
            total_epoch_student_loss += total_loss.cpu().item() * len(batch_data)
        avg_epoch_ce_loss = total_epoch_ce_loss / len(train_loader.dataset)
        avg_epoch_student_loss = total_epoch_student_loss / len(train_loader.dataset)
        avg_epoch_decision_cost = total_epoch_decision_cost / len(train_loader.dataset)
    if t > args.burn_in_iters:
        print("Epoch: %d, avg classification loss: %f, avg student loss: %f\n" % (
            epoch, avg_epoch_ce_loss, avg_epoch_student_loss))
        if epoch % 5 == 0:
            _, teacher_val_accuracy, teacher_val_proba, _, _, _ = \
                test_model_with_rejection(teacher_model, test_loader, cost_matrix, num_classes)
            teacher_model_ensemble_proba.append(teacher_val_proba)
            _, student_val_accuracy, student_val_proba, _, _, _ = \
                test_model_with_rejection(student_model, test_loader, cost_matrix, num_classes)
            print("Epoch: %d, teacher test accuracy: %f, student test accuracy: %f \n" % (
                epoch, teacher_val_accuracy, student_val_accuracy))

model_ensemble_proba = torch.stack(teacher_model_ensemble_proba[-30:]).mean(dim=0).to(args.device)
teacher_model_ensemble_decision_cost = torch.matmul(model_ensemble_proba.unsqueeze(dim=1), cost_matrix).squeeze()
teacher_model_ensemble_decision_softmax = F.log_softmax(-BETA * teacher_model_ensemble_decision_cost, dim=-1).exp()
##TODO: Give meaningful names below
A = np.expand_dims(
    torch.matmul(teacher_model_ensemble_decision_softmax, test_cost_matrix.transpose(0, 1)).cpu().numpy(), 1)
B = np.expand_dims(
    F.one_hot(torch.tensor(test_set.targets).long().squeeze(), num_classes=num_classes + 1).cpu().numpy().T,
    0)
B = np.matmul(A, B.T)
# teacher_model_ensemble_decision_cost_mean = (teacher_model_ensemble_decision_cost *
#                                              teacher_model_ensemble_decision_softmax).sum(dim=-1).mean()
teacher_model_ensemble_decision_cost_mean = B.squeeze().mean()

teacher_model_ensemble_proba = torch.stack(teacher_model_ensemble_proba[-30:]).mean(dim=0).to(args.device)
teacher_model_ensemble_preds = teacher_model_ensemble_decision_softmax.max(dim=1)[1]
rejection_indices = (teacher_model_ensemble_preds == num_classes).nonzero().squeeze()
non_rejection_indices = (teacher_model_ensemble_preds != num_classes).nonzero().squeeze()
non_rejected_val_proba = teacher_model_ensemble_proba[non_rejection_indices]
non_rejected_labels = torch.tensor(test_set.targets).long().squeeze()[non_rejection_indices].to(args.device)
teacher_model_ensemble_loss = F.nll_loss(
    non_rejected_val_proba.log()[(non_rejected_labels != num_classes).nonzero().squeeze()],
    non_rejected_labels[(non_rejected_labels != num_classes).nonzero().squeeze()])
teacher_model_ensemble_accuracy = ((teacher_model_ensemble_preds[non_rejection_indices] ==
                                    torch.tensor(test_set.targets).long().squeeze().to(
                                        args.device)[non_rejection_indices]).sum().cpu().item()) / len(
    non_rejection_indices)
num_rejected_samples = len(rejection_indices) if rejection_indices.dim() != 0 else 0

output_results_list += [teacher_model_ensemble_loss.cpu().item(), teacher_model_ensemble_accuracy,
                        teacher_model_ensemble_decision_cost_mean,
                        num_rejected_samples]
val_loss, val_accuracy, val_proba, num_rejected_samples, student_reject_indices, student_non_reject_indices, \
    = test_model_with_rejection(student_model, test_loader, cost_matrix, num_classes)
student_val_proba = val_proba.to(args.device)
student_decision_cost = torch.matmul(student_val_proba.unsqueeze(dim=1), cost_matrix).squeeze()
student_decision_softmax = F.log_softmax(-BETA * student_decision_cost, dim=-1).exp()
##TODO: Give meaningful names below
A = np.expand_dims(torch.matmul(student_decision_softmax, test_cost_matrix.transpose(0, 1)).cpu().numpy(), 1)
B = np.expand_dims(
    F.one_hot(torch.tensor(test_set.targets).long().squeeze(), num_classes=num_classes + 1).cpu().numpy().T,
    0)
B = np.matmul(A, B.T)
# student_decision_cost_mean = (student_decision_cost * student_decision_softmax).sum(dim=-1).mean()
student_decision_cost_mean = B.squeeze().mean()
# student_ece = get_ece(student_val_proba.cpu().numpy()[student_non_reject_indices],
#                       test_set.targets.cpu().numpy()[student_non_reject_indices])
# student_auroc = roc_auc_score(test_set.targets.cpu().numpy()[student_non_reject_indices],
#                               student_val_proba.cpu().numpy()[student_non_reject_indices][:, 1])
# student_ap_score = average_precision_score(test_set.targets.cpu().numpy()[student_non_reject_indices],
#                                            student_val_proba.cpu().numpy()[student_non_reject_indices][:, 1])
output_results_list += [val_loss.cpu().item(), val_accuracy, student_decision_cost_mean, num_rejected_samples, ]

print(output_results_list)
with open('rotation_rejection_results_warm_start_no_student .csv', 'a') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',')
    csvwriter.writerow(output_results_list)
