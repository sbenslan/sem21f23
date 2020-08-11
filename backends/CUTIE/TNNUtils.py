# ----------------------------------------------------------------------
#
# File: TNNUtils.py
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

# This file provides cookie-cutter templates for CIFAR-10 experiments with the CUTIE architecture

import torch
import torch.nn as nn

import numpy as np

from scipy.linalg import *

from quantlab.algorithms.ste import STEController, STEActivation
from quantlab.algorithms.inq import INQController, INQLinear, INQConv2d
from quantlab.algorithms import controller

# A ConvBlock is defined as the following layer stack:
#  torch.nn.Conv2D 1[quantization=false]
#  STEActivation 1[quantization=true]
#  INQConv2D 1[quantization=true]

#  torch.nn.BatchNorm2D 1[quantization=false OR writeback=true]
#  torch.nn.ReLU 1[quantization=false]
#  torch.nn.HardTanh 1[quantization=true OR writeback=true]

# This class is supposed to enable rapid prototyping.
# The HardTanh activation is not really a limitation for ternary neural networks...

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, actQuantStartEpoch, stride=(1,1),quantization=False, quantizationDepth=3, writeback=True, quant_strategy="magnitude"):
        super(ConvBlock, self).__init__()
        
        self.quantization = quantization
        self.quantizationDepth = quantizationDepth
        self.writeback = writeback
        self.stride=stride

        self.quant_strategy = quant_strategy
        self.actQuantStartEpoch = actQuantStartEpoch
        
        self.layers = self.CreateCNN(in_channels, out_channels)

        self.quantActCtrl =  STEController(STEController.get_ste_modules(self.layers))
               
    def CreateCNN(self, in_channels, out_channels):
        cnnlayers = [] 
        if (self.quantization == False):
            cnnlayers += [torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=self.stride, kernel_size=(3,3), padding=(1,1))]
            cnnlayers += [torch.nn.BatchNorm2d(num_features=out_channels, track_running_stats=True)]
            cnnlayers += [torch.nn.ReLU()]
        else:
            cnnlayers += [STEActivation(quant_start_epoch=self.actQuantStartEpoch, 
                                         num_levels=self.quantizationDepth)]
            cnnlayers += [INQConv2d(in_channels, out_channels, stride=self.stride, kernel_size=3, padding=1, 
                                         num_levels=self.quantizationDepth, quant_strategy=self.quant_strategy, 
                                         quant_init_method='uniform-l2-opt')]
            if(self.writeback == True):
                cnnlayers += [torch.nn.BatchNorm2d(num_features=out_channels, track_running_stats=True)]
                cnnlayers += [torch.nn.Hardtanh()]

        return torch.nn.Sequential(*cnnlayers)

    def forward(self, x):
        for i in self.layers:
            x = i(x)
        return x

# A ResBlock is defined as the following layer stack:

# [ConvBlock][ConvBlock]*(x, writeback=True) + ConvBlock(x, writeback=False)

# This class is supposed to enable rapid prototyping.
    
class ResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, depth, quantization=False, quantizationDepth=3, writeback=True):
        super(ResBlock, self).__init__()
        
        self.quantization = quantization
        self.quantizationDepth = quantizationDepth
        self.writeback = writeback
        
        self.main = self.CreateCNN(in_channels, out_channels, depth)
        self.skip = self.CreateBypass(in_channels, out_channels)
        self.merge = self.CreateAfterMerge(out_channels)

        
    def CreateCNN(self, in_channels, out_channels, depth):
        cnnlayers = [] 
        if (self.quantization == False):
            cnnlayers += [torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), padding=(1,1))]
            for i in range(depth-1):
                cnnlayers += [torch.nn.BatchNorm2d(num_features=out_channels, track_running_stats=True)]
                cnnlayers += [torch.nn.ReLU()]
                cnnlayers += [torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,3), padding=(1,1))]
        else:
            cnnlayers += [STEActivation(quant_start_epoch=actQuantStartEpoch, 
                                         num_levels=self.quantizationDepth)]
            cnnlayers += [INQConv2d(in_channels, out_channels, kernel_size=3, padding=1, 
                                         num_levels=self.quantizationDepth, quant_strategy=self.quant_strategy, 
                                         quant_init_method='uniform-l2-opt')]
            for i in range(depth-1):
                cnnlayers += [torch.nn.BatchNorm2d(num_features=out_channels, track_running_stats=True)]
                cnnlayers += [torch.nn.Hardtanh()]
                cnnlayers += [STEActivation(quant_start_epoch=actQuantStartEpoch,  
                                         num_levels=self.quantizationDepth)]
                cnnlayers += [INQConv2d(out_channels, out_channels, kernel_size=3, padding=1, 
                                         num_levels=self.quantizationDepth, quant_strategy=self.quant_strategy, 
                                         quant_init_method='uniform-l2-opt')]

        return torch.nn.Sequential(*cnnlayers)
    
    def CreateBypass(self, in_channels, out_channels):
        bypass = []
        if (self.quantization == False):
            bypass += [torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), padding=(1,1))]
        else:
            bypass += [STEActivation(quant_start_epoch=actQuantStartEpoch, 
                                             num_levels=self.quantizationDepth)]
            bypass += [INQConv2d(in_channels, out_channels, kernel_size=3, padding=1, 
                                             num_levels=self.quantizationDepth, quant_strategy=self.quant_strategy, 
                                             quant_init_method='uniform-l2-opt')]
        return torch.nn.Sequential(*bypass)
    
    def CreateAfterMerge(self, out_channels):
        merge = []
        cnnlayers += [torch.nn.BatchNorm2d(num_features=out_channels, track_running_stats=True)]
        merge += [torch.nn.Hardtanh()]
        return torch.nn.Sequential(*merge)
    
    def forward(self,x):
        bypass = x
        main = x
        for i in self.main:
            main = i(main)
        for i in self.skip:
            bypass = i(bypass)
        x = main + bypass
        if(self.writeback == True):
            for i in self.merge:
                x = i(x)
        return x

# The HardDense_Model is used to have a configurable model to
# 1) Allow comparison of quantization strategies
# 2) Allow comparison of fixed dense layers vs. trainable dense layers
    
class HardDense_Model(torch.nn.Module):
    def __init__(self, numFeatures, numClasses, actQuantStartEpoch, weightQuantSchedule, classTrain=True, quantized=False, quantizationDepth=3, quant_strategy="magnitude", channels=128):
        super(HardDense_Model, self).__init__()

        self.channels = channels
        
        self.numFeatures = numFeatures
        self.numClasses = numClasses
        self.classTrain = classTrain
        self.quantized = quantized
        self.quantizationDepth = quantizationDepth

        self.actQuantStartEpoch = actQuantStartEpoch

        self.quant_strategy = quant_strategy
        
        self.CNNLayers = self.CreateCNN()
        self.HardClassifier = self.CreateClassifier(numFeatures, numClasses)

        self.quantWghtCtrl = INQController(INQController.get_inq_modules(self.CNNLayers),
                                           weightQuantSchedule,
                                           clear_optim_state_on_step=True)
        self.quantActCtrl =  STEController(STEController.get_ste_modules(self.CNNLayers))
    
    def FixClassifierTernary(self):
        for i in self.HardClassifier:
            had = torch.Tensor(np.random.permutation(hadamard(self.numFeatures)[:self.numClasses]))
            if hasattr(i, 'weight'):
                randvec_weights = torch.rand(i.weight.shape)
                i.weight = torch.nn.Parameter(had)
            if hasattr(i, 'bias'):
                randvec_bias = torch.rand(i.bias.shape)
                i.bias = torch.nn.Parameter(torch.zeros(i.bias.shape))
                
    def SetClassifierLearning(self, on=True):
        self.classTrain = on

    def CreateCNN(self):
        cnnlayers = [] 
        cnnlayers += [ConvBlock(120,self.channels,self.actQuantStartEpoch,quantization=self.quantized, writeback=True, quantizationDepth=self.quantizationDepth, quant_strategy=self.quant_strategy)]
        cnnlayers += [ConvBlock(self.channels,self.channels,self.actQuantStartEpoch,quantization=self.quantized, writeback=True, quantizationDepth=self.quantizationDepth, quant_strategy=self.quant_strategy)]
        cnnlayers += [ConvBlock(self.channels,self.channels,self.actQuantStartEpoch,quantization=self.quantized, writeback=True, quantizationDepth=self.quantizationDepth, quant_strategy=self.quant_strategy)]
        cnnlayers += [torch.nn.MaxPool2d(kernel_size=(2,2),stride=(2,2),padding=(0,0))]
        cnnlayers += [ConvBlock(self.channels,self.channels,self.actQuantStartEpoch,quantization=self.quantized, writeback=True, quantizationDepth=self.quantizationDepth, quant_strategy=self.quant_strategy)]
        cnnlayers += [ConvBlock(self.channels,self.channels,self.actQuantStartEpoch,quantization=self.quantized, writeback=True, quantizationDepth=self.quantizationDepth, quant_strategy=self.quant_strategy)]
        cnnlayers += [torch.nn.MaxPool2d(kernel_size=(2,2),stride=(2,2),padding=(0,0))]
        cnnlayers += [ConvBlock(self.channels,self.channels,self.actQuantStartEpoch,quantization=self.quantized, writeback=True, quantizationDepth=self.quantizationDepth, quant_strategy=self.quant_strategy)]
        cnnlayers += [ConvBlock(self.channels,self.channels,self.actQuantStartEpoch,quantization=self.quantized, writeback=True, quantizationDepth=self.quantizationDepth, quant_strategy=self.quant_strategy)]
        cnnlayers += [torch.nn.MaxPool2d(kernel_size=(2,2),stride=(2,2),padding=(0,0))]
        cnnlayers += [ConvBlock(self.channels,self.channels,self.actQuantStartEpoch,quantization=self.quantized, writeback=True, quantizationDepth=self.quantizationDepth, quant_strategy=self.quant_strategy)]
        cnnlayers += [torch.nn.AvgPool2d(kernel_size=(4,4),stride=(4,4),padding=(0,0))]
        return torch.nn.Sequential(*cnnlayers)
    
    def CreateClassifier(self, numFeatures, numClasses):
        classifier = []
        classifier += [torch.nn.Flatten()]
        classifier += [torch.nn.Linear(numFeatures, numClasses)]
        return torch.nn.Sequential(*classifier)
    
    def forward(self, x):
        for i in self.HardClassifier:
            if hasattr(i, 'weight'):
                i.weight.requires_grad = self.classTrain
            if hasattr(i, 'bias'):
                i.bias.requires_grad = self.classTrain
        
        for i in self.CNNLayers:
            x = i(x)
        for i in self.HardClassifier:
            x = i(x)
        return x
