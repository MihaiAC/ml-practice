import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import prune

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import argparse
import copy
import sys
import types

# The "prune" file is a modified version of the official pytorch version, made to suit our needs.

#-------------------------------------------------------------------------------
# Useful functions.
def reinit_and_apply_mask(model, model_init):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            with torch.no_grad():
                module.weight = torch.nn.Parameter((getattr(
                    model_init, module_name).weight * module.weight_mask).clone().to(device).detach().requires_grad_(True))
                module.bias = torch.nn.Parameter((getattr(model_init, module_name).bias).clone(
                ).to(device).detach().requires_grad_(True))


def count_unpruned_weights(model):
    count = 0
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
                count += module.weight_mask.sum()
        return count

def count_all_weights(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def flip_masks(model):
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
                module.weight_mask.mul_(-1)
                module.weight_mask.add_(1)

def prune_model_l1_unstructured(model, pruning_rate):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if name == 'fc_out':
                print('Debug: fc_out pruned.')
                prune.l1_unstructured(
                    module, name='weight', amount=pruning_rate/2)
            else:
                prune.l1_unstructured(
                    module, name='weight', amount=pruning_rate)
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=pruning_rate/2)


# The SNIP methods and adapted methods are modified versions of: https://github.com/mi-lad/snip/blob/master/snip.py.
def apply_prune_mask(net, keep_masks):
    prunable_layers = filter(
        lambda layer: isinstance(layer, nn.Conv2d) or isinstance(
            layer, nn.Linear), net.modules())

    for layer, keep_mask in zip(prunable_layers, keep_masks):
        assert (layer.weight.shape == keep_mask.shape)

        layer.register_buffer("weight_mask", keep_mask)

        def hook_factory(keep_mask):
            """
            The hook function can't be defined directly here because of Python's
            late binding which would result in all hooks getting the very last
            mask! Getting it through another function forces early binding.
            """

            def hook(grads):
                return grads * keep_mask

            return hook

        # mask[i] == 0 --> Prune parameter
        # mask[i] == 1 --> Keep parameter

        # Step 1: Set the masked weights to zero (NB the biases are ignored)
        # Step 2: Make sure their gradients remain zero
        layer.weight.data[keep_mask == 0.] = 0.
        layer.weight.register_hook(hook_factory(keep_mask))

def snip_forward_conv2d(self, x):
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


def snip_forward_linear(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)


def iterative_SNIP(net, type_prune, keep_list, train_dataloader, device):
    last_masks = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            mask = torch.ones_like(layer.weight)
            last_masks.append(mask)

    for keep in keep_list:

        inputs, targets = next(iter(train_dataloader))
        inputs = inputs.to(device)
        targets = targets.to(device)

        aux_net = copy.deepcopy(net)

        position = 0
        for layer in aux_net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.weight_mask = nn.Parameter(last_masks[position].clone().detach())
                layer.weight.requires_grad = False
                position += 1

            if isinstance(layer, nn.Conv2d):
                layer.forward = types.MethodType(snip_forward_conv2d, layer)

            if isinstance(layer, nn.Linear):
                layer.forward = types.MethodType(snip_forward_linear, layer)

        aux_net.zero_grad()
        outputs = aux_net.forward(inputs)
        loss = F.nll_loss(outputs, targets)
        loss.backward()

        grads_abs = []
        for layer in aux_net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                grads_abs.append(torch.abs(layer.weight_mask.grad))

        all_scores = []
        for index, x in enumerate(grads_abs):
            idx_mask = last_masks[index] == 1
            all_scores.append(x[idx_mask])
        
        all_scores = torch.cat(all_scores)
        norm_factor = torch.sum(all_scores)
        all_scores.div_(norm_factor)

        if type_prune == 'num':
            num_params_to_keep = keep
        else:
            num_params_to_keep = int(len(all_scores) * keep)
        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]

        keep_masks = []
        with torch.no_grad():
            for index, g in enumerate(grads_abs):
                g = g * last_masks[index]
                keep_masks.append(((g / norm_factor) >= acceptable_score).float())

        print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks])))

        del last_masks
        last_masks = keep_masks

    return(keep_masks)

def SNIP(net, type_prune, keep, train_dataloader, device):

    inputs, targets = next(iter(train_dataloader))
    inputs = inputs.to(device)
    targets = targets.to(device)

    net = copy.deepcopy(net)

    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            layer.weight.requires_grad = False

        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)

        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(snip_forward_linear, layer)

    net.zero_grad()
    outputs = net.forward(inputs)
    loss = F.nll_loss(outputs, targets)
    loss.backward()

    grads_abs = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            grads_abs.append(torch.abs(layer.weight_mask.grad))

    all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
    norm_factor = torch.sum(all_scores)
    all_scores.div_(norm_factor)

    if type_prune == 'num':
        num_params_to_keep = keep
    else:
        num_params_to_keep = int(len(all_scores) * keep)
    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    acceptable_score = threshold[-1]

    keep_masks = []
    for g in grads_abs:
        keep_masks.append(((g / norm_factor) >= acceptable_score).float())

    print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks])))

    return(keep_masks)

# EarlyStopping class taken from:
# https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py.
class EarlyStopping(object):
    def __init__(self, mode='max', min_delta=0, patience=4, percentage='False'):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode' + str(mode) + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best - min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - \
                    (best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best - \
                    (best * min_delta / 100)

# Function for weight initialisation (taken from
# https://github.com/rahulvigneswaran/Lottery-Ticket-Hypothesis-in-Pytorch/blob/master/main.py)


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)

# Not sure if the loss calculation is correct here.


def calculate_accuracy_and_loss(model, loader, criterion):
    # Put the model in evaluation mode.
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nr_batches = len(loader)
    total_loss = 0
    accuracy = 0

    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (imgs, targets) in enumerate(loader):
            imgs, targets = imgs.to(device), targets.to(device)

            output = model(imgs)

            total_loss += (1/nr_batches) * criterion(output, targets).item()

            _, predicted = torch.max(output.data, 1)
            total += targets.shape[0]
            correct += (predicted == targets).sum().item()

    accuracy = correct/total * 100
    return accuracy, total_loss


def get_data_loaders(dataset):
    # TODO: Add more dataset-dependent data loaders.
    if dataset == 'mnist':
        # Transforms which will be applied to the data.
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(
                                            (0.1307,), (0.3081, ))
                                        ])

        # 0.1307, 0.3081 represent the mean + std of the mnist dataset.

        # Split the train dataset into a train + valid datasets.
        # Must set the values of the samples in each split (here, 50000, 10000).
        dataset = datasets.MNIST(
            root=os.getcwd() + '/data', train=True, download=True, transform=transform)
        train_set, valid_set = torch.utils.data.random_split(dataset, [
                                                             55000, 5000])

        # Load the test dataset.
        test_set = datasets.MNIST(
            root=os.getcwd() + '/data', train=False, transform=transform)

    elif dataset == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(
                                            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                        ])
        dataset = datasets.CIFAR10(
            root=os.getcwd() + '/data', train=True, download=True, transform=transform)
        train_set, valid_set = torch.utils.data.random_split(dataset, [
                                                             45000, 5000])
        test_set = datasets.CIFAR10(
            root=os.getcwd() + '/data', train=False, transform=transform)

    # Transformations will not be applied until you call a DataLoader on it.
    # make valid_loader = test_loader if the option for the split is 0.
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    return train_loader, valid_loader, test_loader


def init_optimizer(optimizer_name, model, kwargs):
    # parameters = []
    # for module_name, module in model.named_modules():
    #     if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
    #         parameters.append(module.weight)
    #         parameters.append(module.bias)
    params = []
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            params.append(module.weight)
            params.append(module.bias)
    if optimizer_name == 'adam':
        return torch.optim.Adam(params, **kwargs)
    elif optimizer_name == 'sgd':
        return torch.optim.SGD(params, **kwargs)
#------------------------------------------------------------------------------------------------------------------------------
# Our neural nets architectures.


class LeNet300_100(nn.Module):
    def __init__(self):
        super(LeNet300_100, self).__init__()
        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc_out = nn.Linear(100, 10)

    def forward(self, input):
        x = input.flatten(start_dim=1, end_dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_out(x)
        return x


class Conv_2(nn.Module):
    def __init__(self):
        super(Conv_2, self).__init__()
        self.conv11 = nn.Conv2d(3, 64, 3)
        self.conv12 = nn.Conv2d(64, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(14*14*64, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, 10)

    def forward(self, input_x):
        x = F.relu(self.conv11(input_x))
        x = F.relu(self.conv12(x))
        x = self.pool(x)
        x = x.view(-1, 14*14*64)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_out(x)
        return x


class Conv_4(nn.Module):
    def __init__(self):
        super(Conv_4, self).__init__()
        self.conv11 = nn.Conv2d(3, 64, 3)
        self.conv12 = nn.Conv2d(64, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv21 = nn.Conv2d(64, 128, 3)
        self.conv22 = nn.Conv2d(128, 128, 3)
        self.fc1 = nn.Linear(5*5*128, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, 10)

    def forward(self, input_x):
        x = F.relu(self.conv11(input_x))
        x = F.relu(self.conv12(x))
        x = self.pool(x)
        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x = self.pool(x)
        x = x.view(-1, 5*5*128)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_out(x)
        return x


# ------------------------------------------------------------------------------
# Argument parser:
parser = argparse.ArgumentParser()

# General arguments:
parser.add_argument("--experiment_name", type=str)
parser.add_argument("--learning_rate", default=0, type=float,
                    help="LTH fc 1.2e-3, conv2 2e-4, conv4 3e-4")
parser.add_argument("--batch_size", default=60, type=int)
parser.add_argument("--eval_freq", default=500, type=int)
parser.add_argument("--max_nr_epochs", default=25,
                    type=int, help="Maximum number of epochs")
parser.add_argument("--model", help="fc | conv2 | conv4")
parser.add_argument("--experiment_type", type=str,
                    default="LTH", help="LTH | SNIP | LTH_flip")
parser.add_argument("--optimizer", type=str, default="adam", help="adam | sgd")
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--patience", type=float, default=4,
                    help="Roughly 2.5 epochs. patience*eval_freq = number of iterations we have patience for. An epoch has training_samples_nr/batch_size iterations.")
parser.add_argument("--random_seed", type=int,
                    help="Need to set experiment seed")

# Method specific arguments:
parser.add_argument("--prune_iterations", default=10, type=int)
parser.add_argument("--pruning_criterion", type=str, default="weight_magnitude",
                    help="Pruning criterion, available only for LTH (so far)")
parser.add_argument("--pruning_rate", type=float, default=0.2)

# Arguments for SGD.
parser.add_argument("--momentum", type=float, default=0)

# Argument used only for the LTH_flip experiment.
parser.add_argument("--flip_LTH_experiment", type=str, default=None)
parser.add_argument("--flip_LTH_prune_iteration", type=int, default=None)

# Arguments used only for a SNIP experiment.
parser.add_argument("--snip_LTH_experiment", type=str, default=None)
parser.add_argument("--is_iterative_SNIP", type=str, default=None)
#parser.add_argument("--prune_type", default="lt", type=str, help="lt | reinit")

args = parser.parse_args()

#-------------------------------------------------------------------------------
# All the variables + initialisations we need for an experiment.
experiment_name = args.experiment_name
learning_rate = args.learning_rate
batch_size = args.batch_size
eval_freq = args.eval_freq
epochs = args.max_nr_epochs
model_name = args.model
experiment_type = args.experiment_type
prune_iterations = args.prune_iterations
pruning_criterion = args.pruning_criterion
pruning_rate = args.pruning_rate
patience = args.patience
random_seed = args.random_seed

# Optimizer params.
optimizer_name = args.optimizer
weight_decay = args.weight_decay
momentum = args.momentum

# LTH_flip experiments
flip_LTH_experiment = args.flip_LTH_experiment
flip_LTH_prune_iteration = args.flip_LTH_prune_iteration

# SNIP experiments.
snip_LTH_experiment = args.snip_LTH_experiment
if args.is_iterative_SNIP is None:
    is_iterative_SNIP = False
else:
    is_iterative_SNIP = True

# Setting the randoms seeds for torch and numpy.
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# Put the optimizer params in a dictionary (will be passed to init_optimizer).
optimizer_args = dict()
optimizer_args["lr"] = learning_rate
optimizer_args["weight_decay"] = weight_decay
if optimizer_name == "sgd":
    optimizer_args['momentum'] = momentum

sns.set_style('darkgrid')

# List of unpruned weights to train SNIP
unpruned_weights_fc = [213060, 170538, 136511, 109282, 87490, 70051, 56094, 44923, 35981, 28823]
unpruned_weights_conv = [2658477, 2130485, 1707721, 1369177, 1098041, 880863, 706877, 567470, 455748, 366193]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Our experiments are hardcoded to use certain datasets:
# FC -> MNIST, Conv-2, Conv-4 -> CIFAR10.
if model_name == "fc":
    model = LeNet300_100().to(device)
    dataset = "mnist"
elif model_name == "conv2":
    model = Conv_2().to(device)
    dataset = "cifar10"
elif model_name == "conv4":
    model = Conv_4().to(device)
    dataset = "cifar10"
else:
    print("Invalid model.")
    sys.exit()

# Get the data loaders for MNIST/CIFAR10.
train_loader, valid_loader, test_loader = get_data_loaders(dataset)

# If the default value of 0 was given to learning_rate, give it the appropriate
# value instead.
if learning_rate == 0:
    if model_name == "fc":
        learning_rate = 0.0012
    elif model_name == "conv2":
        learning_rate = 0.0002
    elif model_name == "conv4":
        learning_rate = 0.0003
    else:
        print("Invalid model.")
        sys.exit()

# Initialise weights.
model.apply(weight_init)

# Save initial model for reference.
os.makedirs(os.getcwd() + '/' + experiment_name, exist_ok=True)
torch.save(model, os.getcwd() + '/' + experiment_name + '/model_init.pth')

# Weight decay? - SNIP does have it with SGD, doesn't have it with SGD.
#               - LTH seems to have some experiments with weight decay, and some without + unspecified value.
optimizer = init_optimizer(optimizer_name, model, optimizer_args)

criterion = nn.CrossEntropyLoss()

os.makedirs(os.getcwd() + '/' + experiment_name + '/stats', exist_ok=True)


def train_SNIP(pruning_no):
    prune_iterations = 1
    global optimizer
    global model

    print('\nBeginning experiment: ' + experiment_name + '\n')

    best_train_accs = np.zeros((prune_iterations, ))
    best_valid_accs = np.zeros((prune_iterations, ))
    best_test_accs = np.zeros((prune_iterations, ))
    best_train_losses = np.zeros((prune_iterations, ))
    best_valid_losses = np.zeros((prune_iterations, ))
    best_test_losses = np.zeros((prune_iterations, ))
    early_stop_iterations = np.zeros((prune_iterations, ))
    unpruned_weights_counts = np.zeros((prune_iterations, ))

    experiment_PATH = os.getcwd() + '/' + experiment_name

    # Re-init iteration number.
    iteration_nr = 0

    # Re-init best validation acc.
    best_valid_acc = 0

    unpruned_weights_counts[0] = count_all_weights(model)

    if is_iterative_SNIP == False:
        keep_masks = SNIP(model, 'num', pruning_no, train_loader, device)
        apply_prune_mask(model, keep_masks)
    else:
        keep_masks = iterative_SNIP(model, 'num', pruning_no, train_loader, device)
        apply_prune_mask(model, keep_masks)

    # Initializing EarlyStopping
    es = EarlyStopping(patience=patience)

    train_acc, train_loss = calculate_accuracy_and_loss(
        model, train_loader, criterion)
    valid_acc, valid_loss = calculate_accuracy_and_loss(
        model, valid_loader, criterion)
    test_acc, test_loss = calculate_accuracy_and_loss(
        model, test_loader, criterion)

    prune_iteration = 0
    train_accs = []
    train_losses = []
    valid_accs = []
    valid_losses = []
    test_accs = []
    test_losses = []

    train_accs.append(train_acc)
    train_losses.append(train_loss)
    valid_accs.append(valid_acc)
    valid_losses.append(valid_loss)
    test_accs.append(test_acc)
    test_losses.append(test_loss)

    print('Stats before training: Train loss: {:.4f}, Train Acc: {:.2f}, Valid loss: {:.4f}, Valid Acc: {:.2f}'.format(
        train_loss, train_acc, valid_loss, valid_acc))

    early_stop = False
    for epoch in range(epochs):
        if early_stop:
            break

        model.train()
        for batch_idx, (imgs, targets) in enumerate(train_loader):

            iteration_nr += 1

            optimizer.zero_grad()
            imgs, targets = imgs.to(device), targets.to(device)

            output = model(imgs)

            train_loss = criterion(output, targets)
            train_loss.backward()

            optimizer.step()

            if iteration_nr % eval_freq == 0:
                train_acc, train_loss = calculate_accuracy_and_loss(
                    model, train_loader, criterion)
                valid_acc, valid_loss = calculate_accuracy_and_loss(
                    model, valid_loader, criterion)
                test_acc, test_loss = calculate_accuracy_and_loss(
                    model, test_loader, criterion)

                train_accs.append(train_acc)
                train_losses.append(train_loss)
                valid_accs.append(valid_acc)
                valid_losses.append(valid_loss)
                test_accs.append(test_acc)
                test_losses.append(test_loss)

                # Maybe save the best models here, now we're interested only in best accs/losses.
                if(valid_acc > best_valid_acc):
                    best_train_accs[0] = train_acc
                    best_train_losses[0] = train_loss
                    best_valid_accs[0] = valid_acc
                    best_valid_losses[0] = valid_loss
                    best_test_accs[0] = test_acc
                    best_test_losses[0] = test_loss
                    early_stop_iterations[0] = iteration_nr

                    best_valid_acc = valid_acc

                    torch.save(model, experiment_PATH + '/' +
                               'best_model_prune_iteration_' + str(0) + '.pth')

                if es.step(valid_acc):
                    early_stop = True
                    break
        print('Epoch: ' + str(epoch) + ', Train loss: {:.4f}, Train Acc: {:.2f}, Valid loss: {:.4f}, Valid Acc: {:.2f}'.format(
            train_loss, train_acc, valid_loss, valid_acc))
    
    np.save(experiment_PATH + '/stats/train_accs.npy', np.array(train_accs))
    np.save(experiment_PATH + '/stats/train_losses.npy', np.array(train_losses))
    np.save(experiment_PATH + '/stats/valid_accs.npy', np.array(valid_accs))
    np.save(experiment_PATH + '/stats/valid_losses.npy', np.array(valid_losses))
    np.save(experiment_PATH + '/stats/test_accs.npy', np.array(test_accs))
    np.save(experiment_PATH + '/stats/test_losses.npy', np.array(test_losses))

    np.save(experiment_PATH + '/best_train_accs.npy', best_train_accs)
    np.save(experiment_PATH + '/best_train_losses.npy', best_train_losses)
    np.save(experiment_PATH + '/best_valid_accs.npy', best_valid_accs)
    np.save(experiment_PATH + '/best_valid_losses.npy', best_valid_losses)
    np.save(experiment_PATH + '/best_test_accs.npy', best_test_accs)
    np.save(experiment_PATH + '/best_test_losses.npy', best_test_losses)
    np.save(experiment_PATH + '/early_stop_iterations.npy', early_stop_iterations)
    np.save(experiment_PATH + '/unpruned_weights_counts.npy', unpruned_weights_counts)


def train_LTH():
    global optimizer
    global prune_iterations
    global model
    prune_iterations += 1

    print('\nBeginning experiment: ' + experiment_name + '\n')

    best_train_accs = np.zeros((prune_iterations, ))
    best_valid_accs = np.zeros((prune_iterations, ))
    best_test_accs = np.zeros((prune_iterations, ))
    best_train_losses = np.zeros((prune_iterations, ))
    best_valid_losses = np.zeros((prune_iterations, ))
    best_test_losses = np.zeros((prune_iterations, ))
    early_stop_iterations = np.zeros((prune_iterations, ))
    unpruned_weights_counts = np.zeros((prune_iterations, ))

    experiment_PATH = os.getcwd() + '/' + experiment_name

    best_valid_acc = 0
    unpruned_weights_counts[0] = count_all_weights(model)
    for prune_iteration in range(prune_iterations):
        print('\n\nStarting prune iteration: ' + str(prune_iteration) + '\n')

        if prune_iteration != 0:
            # do the pruning here.
            # Before pruning (or after, we can save the mask here, if interested).
            prune_model(model, pruning_rate)

            # Count number of unpruned weights + add to array.
            unpruned_weights_counts[prune_iteration] = count_unpruned_weights(
                model)

            model_init = torch.load(
                experiment_PATH + '/model_init.pth').to(device)
            # with torch.no_grad():
            #     model_init.fc1.weight *= model.fc1.weight_mask.detach().clone()
            #     model_init.fc2.weight *= model.fc2.weight_mask.detach().clone()
            #     model_init.fc_out.weight *= model.fc_out.weight_mask.detach().clone()
            # model = model_init
            reinit_and_apply_mask(model, model_init)
            del model_init

            # Re-initialize the optimizer.
            optimizer = init_optimizer(optimizer_name, model, optimizer_args)

        # Re-init iteration number.
        iteration_nr = 0

        # Re-init best validation acc.
        best_valid_acc = 0

        # Initializing EarlyStopping
        es = EarlyStopping(patience=patience)

        train_acc, train_loss = calculate_accuracy_and_loss(
            model, train_loader, criterion)
        valid_acc, valid_loss = calculate_accuracy_and_loss(
            model, valid_loader, criterion)
        test_acc, test_loss = calculate_accuracy_and_loss(
            model, test_loader, criterion)
        
        train_accs = []
        train_losses = []
        valid_accs = []
        valid_losses = []
        test_accs = []
        test_losses = []

        train_accs.append(train_acc)
        train_losses.append(train_loss)
        valid_accs.append(valid_acc)
        valid_losses.append(valid_loss)
        test_accs.append(test_acc)
        test_losses.append(test_loss)

        print('Stats before training: Train loss: {:.4f}, Train Acc: {:.2f}, Valid loss: {:.4f}, Valid Acc: {:.2f}'.format(
            train_loss, train_acc, valid_loss, valid_acc))

        early_stop = False
        for epoch in range(epochs):
            if early_stop:
                break

            model.train()
            for batch_idx, (imgs, targets) in enumerate(train_loader):

                iteration_nr += 1

                optimizer.zero_grad()
                imgs, targets = imgs.to(device), targets.to(device)

                output = model(imgs)

                train_loss = criterion(output, targets)
                train_loss.backward()

                optimizer.step()

                if iteration_nr % eval_freq == 0:
                    train_acc, train_loss = calculate_accuracy_and_loss(
                        model, train_loader, criterion)
                    valid_acc, valid_loss = calculate_accuracy_and_loss(
                        model, valid_loader, criterion)
                    test_acc, test_loss = calculate_accuracy_and_loss(
                        model, test_loader, criterion)

                    train_accs.append(train_acc)
                    train_losses.append(train_loss)
                    valid_accs.append(valid_acc)
                    valid_losses.append(valid_loss)
                    test_accs.append(test_acc)
                    test_losses.append(test_loss)

                    # Maybe save the best models here, now we're interested only in best accs/losses.
                    if(valid_acc > best_valid_acc):
                        best_train_accs[prune_iteration] = train_acc
                        best_train_losses[prune_iteration] = train_loss
                        best_valid_accs[prune_iteration] = valid_acc
                        best_valid_losses[prune_iteration] = valid_loss
                        best_test_accs[prune_iteration] = test_acc
                        best_test_losses[prune_iteration] = test_loss
                        early_stop_iterations[prune_iteration] = iteration_nr

                        best_valid_acc = valid_acc

                        torch.save(model, experiment_PATH + '/' +
                                    'best_model_prune_iteration_' + str(prune_iteration) + '.pth')

                    if es.step(valid_acc):
                        early_stop = True
                        break

            print('Epoch: ' + str(epoch) + ', Train loss: {:.4f}, Train Acc: {:.2f}, Valid loss: {:.4f}, Valid Acc: {:.2f}'.format(
                train_loss, train_acc, valid_loss, valid_acc))
        
        # After all epochs are finished for the current model, save the relevant data.
        np.save(experiment_PATH + '/stats/train_accs_' + str(prune_iteration) + '.npy', np.array(train_accs))
        np.save(experiment_PATH + '/stats/train_losses_' + str(prune_iteration) + '.npy', np.array(train_losses))
        np.save(experiment_PATH + '/stats/valid_accs_' + str(prune_iteration) + '.npy', np.array(valid_accs))
        np.save(experiment_PATH + '/stats/valid_losses_' + str(prune_iteration) + '.npy', np.array(valid_losses))
        np.save(experiment_PATH + '/stats/test_accs_' + str(prune_iteration) + '.npy', np.array(test_accs))
        np.save(experiment_PATH + '/stats/test_losses_' + str(prune_iteration) + '.npy', np.array(test_losses))

    np.save(experiment_PATH + '/best_train_accs.npy', best_train_accs)
    np.save(experiment_PATH + '/best_train_losses.npy', best_train_losses)
    np.save(experiment_PATH + '/best_valid_accs.npy', best_valid_accs)
    np.save(experiment_PATH + '/best_valid_losses.npy', best_valid_losses)
    np.save(experiment_PATH + '/best_test_accs.npy', best_test_accs)
    np.save(experiment_PATH + '/best_test_losses.npy', best_test_losses)
    np.save(experiment_PATH + '/early_stop_iterations.npy', early_stop_iterations)
    np.save(experiment_PATH + '/unpruned_weights_counts.npy', unpruned_weights_counts)


def train_LTH_flip():
    global optimizer
    global prune_iterations
    global model
    prune_iterations += 1

    print('\nBeginning experiment: ' + experiment_name + '\n')

    best_train_accs = np.zeros((prune_iterations, ))
    best_valid_accs = np.zeros((prune_iterations, ))
    best_test_accs = np.zeros((prune_iterations, ))
    best_train_losses = np.zeros((prune_iterations, ))
    best_valid_losses = np.zeros((prune_iterations, ))
    best_test_losses = np.zeros((prune_iterations, ))
    early_stop_iterations = np.zeros((prune_iterations, ))
    unpruned_weights_counts = np.zeros((prune_iterations, ))

    experiment_PATH = os.getcwd() + '/' + experiment_name
    os.makedirs(experiment_PATH, exist_ok=True)

    os.makedirs(experiment_PATH + '/stats', exist_ok=True)

    best_valid_acc = 0
    unpruned_weights_counts[0] = count_all_weights(model)

    # Need to load model_init + model at the iteration we want.
    model_init = torch.load(os.getcwd() + '/' + flip_LTH_experiment + '/model_init.pth')
    torch.save(model_init, os.getcwd() + '/' + experiment_name + '/model_init.pth')

    model_iteration_k = torch.load(os.getcwd() + '/' + flip_LTH_experiment + '/best_model_prune_iteration_' + str(flip_LTH_prune_iteration) + '.pth')
    model = model_iteration_k

    flip_masks(model)
    reinit_and_apply_mask(model, model_init)

    del model_init
    
    optimizer = init_optimizer(optimizer_name, model, optimizer_args)

    for prune_iteration in range(prune_iterations):
        print('\n\nStarting prune iteration: ' +
                str(prune_iteration) + '\n')

        if prune_iteration != 0:
            # do the pruning here.
            # Before pruning (or after, we can save the mask here, if interested).
            prune_model(model, pruning_rate)

            # Count number of unpruned weights + add to array.
            unpruned_weights_counts[prune_iteration] = count_unpruned_weights(
                model)

            model_init = torch.load(
                os.getcwd() + '/' + experiment_name + '/model_init.pth').to(device)

            reinit_and_apply_mask(model, model_init)
            del model_init

            # Re-initialize the optimizer.
            optimizer = init_optimizer(
                optimizer_name, model, optimizer_args)

        # Re-init iteration number.
        iteration_nr = 0

        # Re-init best validation acc.
        best_valid_acc = 0

        # Initializing EarlyStopping
        es = EarlyStopping(patience=patience)

        train_acc, train_loss = calculate_accuracy_and_loss(
            model, train_loader, criterion)
        valid_acc, valid_loss = calculate_accuracy_and_loss(
            model, valid_loader, criterion)
        test_acc, test_loss = calculate_accuracy_and_loss(
            model, test_loader, criterion)

        train_accs = []
        train_losses = []
        valid_accs = []
        valid_losses = []
        test_accs = []
        test_losses = []

        train_accs.append(train_acc)
        train_losses.append(train_loss)
        valid_accs.append(valid_acc)
        valid_losses.append(valid_loss)
        test_accs.append(test_acc)
        test_losses.append(test_loss)

        print('Stats before training: Train loss: {:.4f}, Train Acc: {:.2f}, Valid loss: {:.4f}, Valid Acc: {:.2f}'.format(
            train_loss, train_acc, valid_loss, valid_acc))

        early_stop = False
        for epoch in range(epochs):
            if early_stop:
                break

            model.train()
            for batch_idx, (imgs, targets) in enumerate(train_loader):
                iteration_nr += 1

                optimizer.zero_grad()
                imgs, targets = imgs.to(device), targets.to(device)

                output = model(imgs)

                train_loss = criterion(output, targets)
                train_loss.backward()

                optimizer.step()

                if iteration_nr % eval_freq == 0:
                    train_acc, train_loss = calculate_accuracy_and_loss(
                        model, train_loader, criterion)
                    valid_acc, valid_loss = calculate_accuracy_and_loss(
                        model, valid_loader, criterion)
                    test_acc, test_loss = calculate_accuracy_and_loss(
                        model, test_loader, criterion)

                    train_accs.append(train_acc)
                    train_losses.append(train_loss)
                    valid_accs.append(valid_acc)
                    valid_losses.append(valid_loss)
                    test_accs.append(test_acc)
                    test_losses.append(test_loss)

                    # Maybe save the best models here, now we're interested only in best accs/losses.
                    if(valid_acc > best_valid_acc):
                        best_train_accs[prune_iteration] = train_acc
                        best_train_losses[prune_iteration] = train_loss
                        best_valid_accs[prune_iteration] = valid_acc
                        best_valid_losses[prune_iteration] = valid_loss
                        best_test_accs[prune_iteration] = test_acc
                        best_test_losses[prune_iteration] = test_loss
                        early_stop_iterations[prune_iteration] = iteration_nr

                        best_valid_acc = valid_acc

                        torch.save(model, experiment_PATH + '/' +
                                    'best_model_prune_iteration_' + str(prune_iteration) + '.pth')

                    if es.step(valid_acc):
                        early_stop = True
                        break

            print('Epoch: ' + str(epoch) + ', Train loss: {:.4f}, Train Acc: {:.2f}, Valid loss: {:.4f}, Valid Acc: {:.2f}'.format(
                train_loss, train_acc, valid_loss, valid_acc))
        
        # After all epochs are finished for the current model, save the relevant data.
        np.save(experiment_PATH + '/stats/train_accs_' + str(prune_iteration) + '.npy', np.array(train_accs))
        np.save(experiment_PATH + '/stats/train_losses_' + str(prune_iteration) + '.npy', np.array(train_losses))
        np.save(experiment_PATH + '/stats/valid_accs_' + str(prune_iteration) + '.npy', np.array(valid_accs))
        np.save(experiment_PATH + '/stats/valid_losses_' + str(prune_iteration) + '.npy', np.array(valid_losses))
        np.save(experiment_PATH + '/stats/test_accs_' + str(prune_iteration) + '.npy', np.array(test_accs))
        np.save(experiment_PATH + '/stats/test_losses_' + str(prune_iteration) + '.npy', np.array(test_losses))

    np.save(experiment_PATH + '/best_train_accs.npy', best_train_accs)
    np.save(experiment_PATH + '/best_train_losses.npy', best_train_losses)
    np.save(experiment_PATH + '/best_valid_accs.npy', best_valid_accs)
    np.save(experiment_PATH + '/best_valid_losses.npy', best_valid_losses)
    np.save(experiment_PATH + '/best_test_accs.npy', best_test_accs)
    np.save(experiment_PATH + '/best_test_losses.npy', best_test_losses)
    np.save(experiment_PATH + '/early_stop_iterations.npy',
            early_stop_iterations)
    np.save(experiment_PATH + '/unpruned_weights_counts.npy',
            unpruned_weights_counts)


# # Setup all variables needed in the train function before calling it.
if experiment_type == "LTH":
    if pruning_criterion == "weight_magnitude":
        prune_model = prune_model_l1_unstructured
    else:
        print("Invalid pruning criterion.")
        sys.exit()
    train_LTH()
elif experiment_type == "SNIP":
    experiment_aux = experiment_name
    if model_name == 'fc':
        for iteration, no_unpruned_weights in enumerate(unpruned_weights_fc):
            if is_iterative_SNIP and iteration < 6:
                continue

            experiment_name = experiment_aux + '_' + str(iteration+1)
            
            # Need to load model_init + model at the iteration we want.
            model = torch.load(os.getcwd() + '/' + snip_LTH_experiment + '/model_init.pth')

            # Save initial model for reference.
            os.makedirs(os.getcwd() + '/' + experiment_name, exist_ok=True)
            torch.save(model, os.getcwd() + '/' + experiment_name + '/model_init.pth')

            # Weight decay? - SNIP does have it with SGD, doesn't have it with SGD.
            #               - LTH seems to have some experiments with weight decay, and some without + unspecified value.
            optimizer = init_optimizer(optimizer_name, model, optimizer_args)

            criterion = nn.CrossEntropyLoss()

            os.makedirs(os.getcwd() + '/' + experiment_name + '/stats', exist_ok=True)

            if is_iterative_SNIP == False:
                train_SNIP(no_unpruned_weights)
            else:
                train_SNIP(unpruned_weights_fc[0:iteration+1])

    elif model_name == 'conv2':
        for iteration, no_unpruned_weights in enumerate(unpruned_weights_conv):
            if is_iterative_SNIP and iteration < 6:
                continue

            experiment_name = experiment_aux + '_' + str(iteration+1)
            # Need to load model_init + model at the iteration we want.
            model = torch.load(os.getcwd() + '/' + snip_LTH_experiment + '/model_init.pth')

            # Save initial model for reference.
            os.makedirs(os.getcwd() + '/' + experiment_name, exist_ok=True)
            torch.save(model, os.getcwd() + '/' + experiment_name + '/model_init.pth')

            # Weight decay? - SNIP does have it with SGD, doesn't have it with SGD.
            #               - LTH seems to have some experiments with weight decay, and some without + unspecified value.
            optimizer = init_optimizer(optimizer_name, model, optimizer_args)

            criterion = nn.CrossEntropyLoss()

            os.makedirs(os.getcwd() + '/' + experiment_name + '/stats', exist_ok=True)

            if is_iterative_SNIP == False: 
                train_SNIP(no_unpruned_weights)
            else:
                train_SNIP(unpruned_weights_conv[0:iteration+1])
    else:
        pass
elif experiment_type == "LTH_flip":
    if pruning_criterion == "weight_magnitude":
        prune_model = prune_model_l1_unstructured
    else:
        print("Invalid pruning criterion.")
        sys.exit()
    train_LTH_flip()
else:
    print("Invalid pruning strategy.")
    sys.exit()
