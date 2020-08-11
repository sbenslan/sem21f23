# ----------------------------------------------------------------------
#
# File: TNNExtract.py
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

# This file compiles any valid, sequential combination of ConvBlocks and Pooling units 

import sys

QuantlabRoot = '../..'
sys.path.append(QuantlabRoot)

from TNNNet import *
from Thermometers import *
from TNNUtils import *
from torch.nn import *

expressionList = [ConvBlock, MaxPool2d]

class FusedConvBlock(torch.nn.Module):
    def __init__(self, conv, pool):
        super(FusedConvBlock, self).__init__()
        self.convLayer = conv
        self.poolLayer = pool
        
    def forward(self, x):
        x = self.convLayer(x)
        x = self.poolLayer(x)
        return x

def GetActivations(layer, x):
    if(isinstance(layer, FusedConvBlock)):
        layer = layer.convLayer
    
    return layer.layers[0](x).detach().numpy()

def GetWeights(layer):
    if(isinstance(layer, FusedConvBlock)):
        layer = layer.convLayer
    print(vars(layer.layers[1]))
    return layer.layers[1].weight_frozen.detach().numpy()

def GetThresholds(layer):
    global f
    
    if(isinstance(layer, FusedConvBlock)):
        layer = layer.convLayer
        
    lbias = layer.layers[1].bias.detach().numpy()

    thresholds = [[-0.5, 0.5]]*len(lbias)
  
    if(len(layer.layers)>3):
        #print(vars(layer.layers[2]))
        bbias = layer.layers[2].bias.detach().numpy()
        bweight = layer.layers[2].weight.detach().numpy()
        bmean = layer.layers[2].running_mean.detach().numpy()
        bvar = layer.layers[2].running_var.detach().numpy()
        eps = layer.layers[2].eps
        for i in range(len(lbias)):
            lowerThreshold = thresholds[i][0]
            upperThreshold = thresholds[i][1]

            lowerThreshold = (lowerThreshold - bbias[i])*(np.sqrt(bvar[i]+eps)/abs(bweight[i]))+(bmean[i]-lbias[i])
            upperThreshold = (upperThreshold - bbias[i])*(np.sqrt(bvar[i]+eps)/abs(bweight[i]))+(bmean[i]-lbias[i])
    
            lowerThreshold = np.int((lowerThreshold/abs(lowerThreshold))*np.ceil(abs(lowerThreshold)))
            upperThreshold = np.int((upperThreshold/abs(upperThreshold))*np.ceil(abs(upperThreshold)))

            if (lowerThreshold > upperThreshold):
                x = upperThreshold
                upperThreshold = lowerThreshold
                lowerThreshold = x
            
            thresholds[i] = [lowerThreshold, upperThreshold]
    
    return thresholds
    
class NetParser:
    
    def __init__(self, expressionList=expressionList):
        self.layerList = []
        self.iterator = 0
        self.netList = []
        self.expressionList = expressionList
        
    def unrollToExpression(self, net):

        layerList = []

        if( True in list(map(lambda x: isinstance(net,x),expressionList)) ):
            layerList += [net]
            return layerList
        else:
            if(hasattr(net, 'layers')):
                return self.unrollToExpression(net.layers)
            elif(hasattr(net, 'CNNLayers')):
                return self.unrollToExpression(net.CNNLayers)
            elif(type(net) == torch.nn.Sequential):
                for i in net:
                    layerList += self.unrollToExpression(i)
                return layerList
            else:
                return []

    def look(self):
        if(self.iterator < len(self.netList)):
            return self.netList[self.iterator]
        else:
            return None

    def lookahead(self):
        if(self.iterator+1 < len(self.netList)):
            return self.netList[self.iterator+1]
        else:
            return None

    def consume(self):
        if(self.iterator < len(self.netList)):
            self.iterator = self.iterator + 1
            return self.netList[self.iterator-1]
        else:
            return None

    def parse(self, net):
        
        self.netList = self.unrollToExpression(net)
        layerList = []
        
        while(self.iterator<len(self.netList)):
            currentLayer = self.look()
            if(isinstance(currentLayer, ConvBlock)):
                nextLayer = self.lookahead()
                if(isinstance(nextLayer, torch.nn.MaxPool2d)):
                    conv = self.consume()
                    pool = self.consume()
                    layerList += [FusedConvBlock(conv, pool)]
                else:
                    conv = self.consume()
                    layerList += [conv]
            else:
                print("Conversion error!!!")
                break 
                
        self.iterator = 0
        return layerList

if __name__=='__main__':

    binary_thermometer = True
    session_name = 'default'

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")    
    
    parser = argparse.ArgumentParser(description="Stimuli generator for")
    parser.add_argument('-T', metavar='ThermometerEncoding', dest='binary_thermometer', type=str2bool, const=True, default=binary_thermometer, nargs='?', help='Set whether to use binary thermometer encoding (default) or ternary thermometer encoding\n')
    parser.add_argument('-s', metavar='Sessionname', dest='session_name', type=str, default=session_name, help='Set the session name for logging purposes')
    parser.add_argument('-c', metavar='Channels', dest='channels', type=int, default=128, help='Set the number of channels in the network')
    parser.add_argument('-H', metavar='HardClassifier', dest='hardclass', type=str2bool, const=True, default=True, nargs='?', help='Set whether to use a fixed classifier(default) or trainable')
    args = parser.parse_args()

    net = HardDense_Model(args.channels, 10, 0, weightQuantSchedule, True, True, channels=args.channels)
    session_name = args.session_name

    ckpt = torch.load(f'{QuantlabRoot}/models/{session_name}-ckpt.pth', map_location=device)
    net.load_state_dict(ckpt['net'], strict=False)

    transform_train, transform_test = pre_processing(args.binary_thermometer)
    testdataset = torchvision.datasets.CIFAR10(root=f'{QuantlabRoot}/datasets', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size = 128, shuffle=False, num_workers=8, pin_memory=True)

    batch = next(iter(testloader))
    
    net.eval()
    
    current = torch.unsqueeze(batch[0][0],0)

    parser = NetParser(expressionList)
    FusedList = parser.parse(net)

    print(parser.netList)
    
    acts = []
    weights = []
    thresholds = []
    pooling = []

    currentActs = current
    
    for i in FusedList:
        acts += [GetActivations(i, currentActs)]
        weights += [GetWeights(i)]
        thresholds += [GetThresholds(i)]
        if (isinstance(i, FusedConvBlock)):
            pooling += [1]
        else:
            pooling += [0]
            currentActs = i(currentActs)

    #print(thresholds)
            
    os.makedirs(f'{QuantlabRoot}/models/{session_name}/', exist_ok=True)
    
    np.savez(f'{QuantlabRoot}/models/{session_name}/weights.npz',*weights)
    np.savez(f'{QuantlabRoot}/models/{session_name}/acts.npz',*acts)
    np.savez(f'{QuantlabRoot}/models/{session_name}/thresholds.npz',*thresholds)
    np.savez(f'{QuantlabRoot}/models/{session_name}/pooling.npz',*pooling)


    
