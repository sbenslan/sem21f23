# ----------------------------------------------------------------------
#
# File: TNNNet.py
#
# Last edited: 10.08.2020        
# 
# Copyright (C) 2020, ETH Zurich and University of Bologna.
#
# Author: Moritz Scherer, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This document wraps HardDense_Model training and evaluation in one script
# For definition of HardDense_Model, see TNNUtils.py or README.md

# For documentation on possible parameters, run TNNNet.py --help

import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
        
import argparse

from tqdm import tqdm
import quantlab_lr_schedulers
from quantlab_indiv import Controller
from quantlab_indiv.inq_ops import INQController, INQLinear, INQConv2d
from quantlab_indiv.ste_ops import STEController, STEActivation

from keras.utils.np_utils import to_categorical

from Thermometers import *
from TNNUtils import *

from tensorboard import program

import numpy as np

# For now this is hardcoded -- might be nice to move to config file
initLR = 1e-2
weightQuantSchedule = {
    '30': 0.2,
    '40': 0.3,
    '50': 0.4,
    '60': 0.5,
    '70': 0.6,
    '80': 0.7,
    '90': 0.8,
    '100': 0.85,
    '115': 0.9,
    '130': 0.95,
    '145': 1.0,
    }


# Parse string arguments to True or False
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Compute the loss given the outputs and criterion
def compute_loss(outputs, labels, criterion):
    loss = criterion(outputs, labels)
    return loss

# Compute the accuracy given outputs and labels
def compute_acc(outputs, labels):
    #print(outputs.shape)
    pred = outputs.reshape(-1,numberOfGestures).max(1)
    squashed_labels = labels.view(-1)
    total = squashed_labels.shape[0]
    correct = pred[1].eq(squashed_labels).sum().item()
    return total, correct

def train(net, epoch, trainloader, l1reg=0, l2reg=0, device='cpu'):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss, correct, total = 0, 0, 0
    
    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = compute_loss(outputs, targets, criterion)

        train_loss += loss

        # Regularize with L1-loss over the WEIGHTS of CNNLAYERS - not over BIASES or OTHER LAYERS
        if(l1reg > 10**(-12)):
            l1_reg = torch.autograd.Variable(torch.FloatTensor(1).to(device), requires_grad=True)
            for L in net.CNNLayers:
                if(isinstance(L,ConvBlock)):
                    l1_reg = l1_reg + torch.norm(list(L.layers[1].parameters())[1],1)
            loss = loss + l1_reg*l1reg

        # Regularize with L2-loss over the WEIGHTS of CNNLAYERS - not over BIASES or OTHER LAYERS
        if(l2reg > 10**(-12)):
            l2_reg = torch.autograd.Variable(torch.FloatTensor(1).to(device), requires_grad=True)
            for L in net.CNNLayers:
                if(isinstance(L,ConvBlock)):
                    l2_reg = l2_reg + torch.norm(list(L.layers[1].parameters())[1],2)
            loss = loss + l2_reg*l2reg

        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    # log to tensorboard
    error_rate = 100.*(total-correct)/total
    lr = optimizer.param_groups[0]['lr']
    tboard.add_scalar("loss/train", train_loss, epoch)
    tboard.add_scalar("error/train", error_rate, epoch)
    tboard.add_scalar("lr", lr, epoch)
    return train_loss, error_rate

def validate(net, epoch, validationloader):
    net.eval()
    val_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(validationloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = compute_loss(outputs, targets, criterion)
            val_loss += loss
            loss = loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # lgo to tensorboard
    error_rate = 100.*(total-correct)/total
    tboard.add_scalar("loss/test", val_loss, epoch)
    tboard.add_scalar("error/test", error_rate, epoch)
    return val_loss, error_rate
        
def test(net, testloader):
    net.eval()
    val_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(testloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = compute_loss(outputs, targets, criterion)
            val_loss += loss
            loss = loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()    
            
    error_rate = 100.*(total-correct)/total
    print('Test -- Loss: %.3f | Err: %.3f%%' % (val_loss, error_rate))        

# Switch for whether to use binary Thermometer or Ternary Thermometer
# Pre-processing is randomcrop, randomflip and thermometer encoding for training,
# Only thermometer encoding for testing
def pre_processing(binary_Thermometer=True):

    if(binary_Thermometer):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(BinaryThermometer(6,5)),
            transforms.Lambda(ThermToTensor()),
        ])

        transform_test = transforms.Compose([
            transforms.Lambda(BinaryThermometer(6,5)),
            transforms.Lambda(ThermToTensor())
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(TernaryThermometer()),
            transforms.Lambda(ThermToTensor()),
        ])

        transform_test = transforms.Compose([
            transforms.Lambda(TernaryThermometer()),
            transforms.Lambda(ThermToTensor())
        ])

    return transform_train, transform_test

if __name__=='__main__':

    batch_size = 128
    num_epochs = 200
    binary_thermometer = True
    session_name = 'default'

    # Keep experiments repeatable, fix all seeds
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    # Use CUDA if available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")    
    
    parser = argparse.ArgumentParser(description="Stimuli generator for")
    parser.add_argument('-b', metavar='BatchSize', dest='batch', type=int, default=batch_size, help='Set the batch size\n')
    parser.add_argument('-T', metavar='ThermometerEncoding', dest='binary_thermometer', type=str2bool, const=True, default=binary_thermometer, nargs='?', help='Set whether to use binary thermometer encoding (default) or ternary thermometer encoding\n')
    parser.add_argument('-p', metavar='Pooling', dest='pooling', type=str2bool, const=True, default=True, nargs='?', help='Set whether to use Max Pooling(default) or Striding')
    parser.add_argument('-t', metavar='Training', dest='training', type=str2bool, const=True, default=True, nargs='?', help='Set whether to Train(default) or Test')
    parser.add_argument('-s', metavar='Sessionname', dest='session_name', type=str, default=session_name, help='Set the session name for logging purposes')
    parser.add_argument('-e', metavar='NumberOfEpochs', dest='num_epochs', type=int, default=num_epochs, help='Set the number of training epochs\n')
    parser.add_argument('-m', metavar='MonitoringEpoch', dest='mon_epoch', type=int, default=10, help='Set the monitoring epoch\n')
    parser.add_argument('-S', metavar='StartEpoch', dest='start_epoch', type=int, default=11, help='Set the activation quantization start epoch\n')
    parser.add_argument('-H', metavar='HardClassifier', dest='hardclass', type=str2bool, const=True, default=True, nargs='?', help='Set whether to use a fixed classifier(default) or trainable')
    parser.add_argument('-q', metavar='QuantizedTraining', dest='quantized', type=str2bool, const=True, default=True, nargs='?', help='Set whether to train quantized(default) or full-precision')
    parser.add_argument('-d', metavar='QuantizationDepth', dest='quantdepth', type=int, default=3, help='Set the quantization depth, i.e. number of possible values per weight\n')
    parser.add_argument('-Q', metavar='QuantizationStrategy', dest='strategy', type=str, default='magnitude', help='Set the quantization strategy')
    parser.add_argument('-w', metavar='WeightDecay', dest='weightdecay', type=float, default=0, help='Set the weight_decay / L2 regularizer')
    parser.add_argument('-L', metavar='L1-Regularizer', dest='reg', type=float, default=0, help='Set the LASSO / L1 regularizer')
    parser.add_argument('-G', metavar='GradientMap', dest='gradmap', type=str2bool, const=True, default=True, nargs='?', help='Return gradient map after testing')
    parser.add_argument('-c', metavar='ChannelDepth', dest='channels', type=int, default=128, help='Set the number of channels in the layers\n')
    args = parser.parse_args()

    print(args)
    
    actQuantStartEpoch = args.start_epoch
    actQuantMonitorEpoch = args.mon_epoch
    transform_train, transform_test = pre_processing(args.binary_thermometer)

    # Use CIFAR-10 dataset
    dataset = torchvision.datasets.CIFAR10(root='./datasets', train=True, download=True, transform=transform_train)
    traindataset, valdataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
    testdataset = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transform_test)

    # Define dataloaders
    trainloader = torch.utils.data.DataLoader(traindataset, batch_size = args.batch, shuffle=True, num_workers=8, pin_memory=True)
    validationloader = torch.utils.data.DataLoader(valdataset, batch_size = args.batch, shuffle=False, num_workers=8, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size = args.batch, shuffle=False, num_workers=8, pin_memory=True)

    if(args.training == True):
        
        net = HardDense_Model(args.channels, 10, actQuantMonitorEpoch, actQuantStartEpoch, weightQuantSchedule, args.hardclass, args.quantized, quantizationDepth=args.quantdepth, strategy=args.strategy, channels=args.channels)
        
        if(args.hardclass):
            net.FixClassifierTernary()
            net.SetClassifierLearning(False)
        else:
            net.SetClassifierLearning(True)
        
        net = net.to(device)
        
        #load_ext tensorboard
        os.makedirs('./logs', exist_ok=True)
        os.makedirs('./models', exist_ok=True)
        
        optimizer = torch.optim.Adam(net.parameters(), lr=initLR)
        criterion = torch.nn.CrossEntropyLoss().to(device)

        tboard = SummaryWriter(log_dir=f'logs/{args.session_name}', max_queue=2)
        tboard.add_graph(net, torch.rand(batch_size, 120, 32, 32).to(device))

        quantControllers = Controller.getControllers(net)

        best_err = 10000
        
        for epoch in range(args.num_epochs):
    
            for ctrlr in quantControllers:
                ctrlr.step_preTraining(epoch, optimizer, tensorboardWriter=tboard)
        
            train(net, epoch, trainloader, l1reg=args.reg, l2reg=args.weightdecay, device=device)
    
            for ctrlr in quantControllers:
                ctrlr.step_preValidation(epoch, tensorboardWriter=tboard)
    
            val_loss, val_error_rate = validate(net, epoch, validationloader)

            if(val_error_rate < best_err and epoch >= 145):
                best_err = val_error_rate
                state = {
                    'net': net.state_dict(),
                    'optim': optimizer.state_dict(),
                    'error_rate': val_error_rate, 
                    'epoch': epoch,
                }
                torch.save(state, f'./models/{args.session_name}-ckpt.pth')
        
    else:

        net = HardDense_Model(args.channels, 10, None, 0, weightQuantSchedule, args.hardclass, args.quantized, quantizationDepth=args.quantdepth, strategy=args.strategy, channels=args.channels)    
        net = net.to(device)

        criterion = torch.nn.CrossEntropyLoss().to(device)
        ckpt = torch.load(f'./models/{args.session_name}-ckpt.pth')
        net.load_state_dict(ckpt['net'])
        test(net, testloader)
        if(args.gradmap):
            net.train()

            x = next(iter(testloader))
            xhat = torch.autograd.Variable(x[0], requires_grad = True)
            compute_loss(net(xhat.to(device)), x[1].to(device), criterion).backward()
            
            np.save(f'./{args.session_name}_gradmap.npy', xhat.grad[0].detach().numpy())

