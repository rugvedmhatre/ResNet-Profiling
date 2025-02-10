"""

Topic  : HPML - Assignment 2
Author : Rugved Mhatre (rrm9598)

"""

# importing libraries
import argparse
import os
import time
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# custom warnings
def custom_formatwarnings(msg, *args, **kwargs):
    return 'WARNING: ' + str(msg) + '\n'

warnings.formatwarning = custom_formatwarnings

# defining the argument parser
parser = argparse.ArgumentParser(
            prog='python lab2.py',
            description='ResNet-18 profiling on CIFAR-10 dataset',
            epilog='------')

parser.add_argument('-d', '--device',
                    choices=['cuda','cpu'], default='cpu',
                    required=False, dest='device',
                    help='select the device for model training (default: cuda)')
parser.add_argument('-dp', '--datapath', default='./data/', 
                    required=False, dest='datapath',
                    help='select the dataset path for training (default: ./data/)')
parser.add_argument('-w', '--workers', type=int,
                    choices=[0, 1, 2, 4, 8, 12, 16], default=2,
                    required=False, dest='workers',
                    help='select the number of workers for data loading (default: 2)')
parser.add_argument('-op', '--optimizer',
                    choices=['sgd', 'sgdnes', 'adagrad', 'adadelta', 'adam'], default='sgd',
                    required=False, dest='optimizer',
                    help='select the optimizer for training (default: sgd)')
parser.add_argument('-v', '--verbose',
                    action='store_true', required=False,
                    dest='verbose',
                    help='if true, all logs will be printed on the console')
parser.add_argument('-q', '--question',
                    choices=['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7'],
                    required=False, dest='question',
                    help='select the assignment question and the code will change correspondingly')
parser.add_argument('-ts', '--torchsummary',
                    action='store_true', required=False,
                    dest='torchsummary',
                    help='if true, it will only print the model summary and exit')

# parsing the arguments
args = parser.parse_args()

if os.path.exists(args.datapath):
    datapath = args.datapath
else:
    datapath = './data/'
    warnings.warn(f"'{args.datapath}' doesn't exist, defaulting to './data/' datapath!")

workers = args.workers
device = args.device

if device == 'cuda':
    if not torch.cuda.is_available():
        device = 'cpu'
        warnings.warn("cuda is not available, running on cpu instead!")

if args.question == 'c1' or args.question == 'c2':
    device = 'cpu'
    workers = 2
    args.optimizer = 'sgd'
    args.verbose = True
elif args.question == 'c3':
    device = 'cpu'
    args.optimizer = 'sgd'
    args.verbose = True
elif args.question == 'c4':
    device = 'cpu'
    args.optimizer = 'sgd'
    args.verbose = True
elif args.question == 'c5':
    workers = 4
    args.optimizer = 'sgd'
    args.verbose = True
elif args.question == 'c6':
    device = 'cuda'
    workers = 4
    args.verbose = True
elif args.question == 'c7':
    device = 'cuda'
    workers = 4
    args.optimizer = 'sgd'
    args.verbose = True

# defining the BasicBlock of the ResNet model
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, batch_norm=True):
        super(BasicBlock, self).__init__()
        self.batch_norm = batch_norm
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        if self.batch_norm:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if self.batch_norm:
            self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if self.batch_norm:
                self.shortcut = nn.Sequential(
                        nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(self.expansion*planes))
            else:
                self.shortcut = nn.Sequential(
                        nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = self.conv1(x)
        if self.batch_norm:
            out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        if self.batch_norm:
            out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# defining the ResNet model
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, batch_norm=True, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.batch_norm = batch_norm

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if self.batch_norm:
            self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, batch_norm=self.batch_norm)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, batch_norm=self.batch_norm)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, batch_norm=self.batch_norm)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, batch_norm=self.batch_norm)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, batch_norm):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, batch_norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        if self.batch_norm:
            out = self.bn1(out)
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# function to create the ResNet18 model
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

# function to create the ResNet18 model without the BatchNormalization layers
def ResNet18woBN():
    return ResNet(BasicBlock, [2, 2, 2, 2], batch_norm=False)

# function to get the train data loader
def get_data_loader(datapath, workers):
    data_statistics = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(*data_statistics, inplace=True)
    ])

    train_dataset = torchvision.datasets.CIFAR10(root=datapath, download=True, transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=workers)
    return train_loader

def main():
    if args.verbose:
        # if args.question: print("Question  :", args.question)
        print("\n== Selected options :")
        print("Device    :", device)
        print("Data Path :", datapath)
        print("Workers   :", workers)
        print("Optimizer :", args.optimizer)

    # creating the dataloader
    if args.verbose: 
        print("\n== Preparing data...")
    train_loader = get_data_loader(datapath, workers)

    # building the model
    if args.verbose: 
        print("\n== Building model...")
    
    if args.question == 'c7':
        if args.verbose:
            print("Using the ResNet-18 model without Batch Norm Layers")
        model = ResNet18woBN()
    else:
        model = ResNet18()
    model = model.to(device)    

    if args.verbose:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {total_params}")

    if args.torchsummary:
        print(model)
        print('\nThere are 20 Conv2D layers in total.')
        
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal trainable parameters: {total_params}")

    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == 'sgdnes':
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=0.1, weight_decay=5e-4)
    elif args.optimizer == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=0.1, weight_decay=5e-4)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)

    if args.torchsummary:
        inputs = torch.randn(1, 3, 32, 32)
        targets = torch.tensor([1])
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        total_gradients = sum(p.grad.numel() for p in model.parameters() if p.grad is not None)
        print(f"\nTotal gradients: {total_gradients}")
        return

    # function to train the model
    def train(epoch):
        if args.verbose: print(f'Epoch: {epoch}')
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        total_data_loading_time = 0
        total_train_time = 0
        
        if device == 'cuda':
            torch.cuda.synchronize()
        epoch_start_time = time.perf_counter()
    
        if device == 'cuda':
            torch.cuda.synchronize()
        data_loading_start_time = time.perf_counter()
    
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if device == 'cuda':
                torch.cuda.synchronize()
            data_loading_end_time = time.perf_counter()
            data_loading_time = data_loading_end_time - data_loading_start_time
            total_data_loading_time += data_loading_time
            
            inputs, targets = inputs.to(device), targets.to(device)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            train_start_time = time.perf_counter()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
            if device == 'cuda':
                torch.cuda.synchronize()
            train_end_time = time.perf_counter()
            train_time = train_end_time - train_start_time
            total_train_time += train_time
    
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if args.verbose and args.question == 'c1':
                print(f'Batch ({batch_idx}/{len(train_loader)})\t: Train Loss : {loss.item():.4f}, Top-1 Train Accuracy : {100. * (predicted.eq(targets).sum().item() / targets.size(0)):.3f}%')
            
            data_loading_start_time = time.perf_counter()
    
        if device == 'cuda':
            torch.cuda.synchronize()
        epoch_end_time = time.perf_counter()
        epoch_total_time = epoch_end_time - epoch_start_time
        
        if args.verbose and (args.question == 'c1' or args.question == 'c2' or args.question == 'c6' or args.question == 'c7'):
            print(f'Train Loss : {train_loss/len(train_loader):.4f}, Top-1 Train Accuracy : {100. * (correct / total):.3f}%') 
            if args.question == 'c2':
                print(f'  Data Loading Time : {total_data_loading_time} seconds')                
                print(f'  Training Time     : {total_train_time} seconds')
                print(f'  Epoch Run Time    : {epoch_total_time} seconds\n')
            if args.question == 'c6':
                print(f'  Training Time     : {total_train_time} seconds')

        return total_data_loading_time, total_train_time, epoch_total_time, train_loss/len(train_loader), 100. * (correct / total)

    # training the model
    if args.verbose:
        print("\n== Training model...")

    all_epochs_total_data_loading_time = 0
    all_epochs_total_train_time = 0
    all_epochs_total_run_time = 0
    average_train_loss = 0
    average_accuracy = 0
    
    for epoch in range(1, 6):
        epoch_data_loading_time, epoch_train_time, epoch_run_time, epoch_train_loss, epoch_accuracy = train(epoch)
        all_epochs_total_data_loading_time += epoch_data_loading_time
        all_epochs_total_train_time += epoch_train_time
        all_epochs_total_run_time += epoch_run_time
        average_train_loss += epoch_train_loss
        average_accuracy += epoch_accuracy

    average_train_loss = average_train_loss / 5.0
    average_accuracy = average_accuracy / 5.0
    
    if args.question == 'c3' or args.question == 'c4':       
        print(f'Total Data Loading Time (for all epochs) : {all_epochs_total_data_loading_time} seconds')
    
    if args.question == 'c4':                                
        print(f'Total Computing Time (for all epochs)    : {all_epochs_total_train_time} seconds')
        print(f'Total Run Time (for all epochs)          : {all_epochs_total_run_time} seconds')

    if args.question == 'c7':
        print(f'\nAverage Train Loss : {average_train_loss:.4f}')
        print(f'Average Accuracy   : {average_accuracy:.3f}%')

    return all_epochs_total_data_loading_time, all_epochs_total_train_time, all_epochs_total_run_time

if __name__ == '__main__':
    if args.question == 'c1':
        main()
    elif args.question == 'c2':
        main()
    elif args.question == 'c3':
        data_loading_times = []
        for w in range(0, 24, 4):
            workers = w            
            data_loading_time, _, _ = main()
            data_loading_times.append(data_loading_time)
        min_loading_time = min(data_loading_times)
        print(f'\nLeast Data Loading Time   : {min_loading_time}')
        print(f'Optimal Number of Workers : {data_loading_times.index(min_loading_time) * 4}')
    elif args.question == 'c4':
        workers = 1
        data_loading_time_1, train_time_1, run_time_1 = main()
        workers = 4
        data_loading_time_4, train_time_4, run_time_4 = main()
        print(f'\n1 Worker Data Loading Time  : {data_loading_time_1} seconds')
        print(f'1 Worker Train Time         : {train_time_1} seconds')
        print(f'1 Worker Run Time           : {run_time_1} seconds')
        print(f'4 Workers Data Loading Time : {data_loading_time_4} seconds')
        print(f'4 Workers Train Time        : {train_time_4} seconds')
        print(f'4 Workers Run Time          : {run_time_4} seconds')
    elif args.question == 'c5':
        device = 'cuda'
        _, _, run_time_gpu = main()
        device = 'cpu'
        _, _, run_time_cpu = main()
        print(f'\nGPU - Average Run Time (over 5 Epochs) : {run_time_gpu/5.0} seconds')
        print(f'CPU - Average Run Time (over 5 Epochs) : {run_time_cpu/5.0} seconds')
    elif args.question == 'c6':
        args.optimizer = 'sgd'
        main()
        args.optimizer = 'sgdnes'
        main()
        args.optimizer = 'adagrad'
        main()
        args.optimizer = 'adadelta'
        main()
        args.optimizer = 'adam'
        main()
    elif args.question == 'c7':
        main()
    else:
        main()
