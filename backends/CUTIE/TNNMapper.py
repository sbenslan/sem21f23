# ----------------------------------------------------------------------
#
# File: TNNMapper.py
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

# Maps a compiled HardDense_Model network to CUTIE.
# Requires stimuli path of the CUTIE project

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Thermometers import *
from TNNUtils import *
#from TNNNet import *
#from TNNExtract import *

import sys
import random
import numpy as np
from collections import namedtuple
from io import StringIO
import argparse
from tqdm import tqdm
import json

import sys

accelStimuliPath = '/usr/scratch/fenga6/scheremo/tnn-accel/stimuli'
sys.path.insert(1,accelStimuliPath)

import global_parameters

import gen_activationmemory_full_stimuli as actmemory
import gen_weightmemory_full_stimuli as weightmemory
import gen_ocu_pool_weights_stimuli as ocu
import gen_LUCA_stimuli as LUCA

filename = 'compute_output'

name_stimuli = 'compute_output_stimuli.txt'
name_exp = 'compute_output_exp_responses.txt'
    
f = open(name_stimuli, 'w+')
g = open(name_exp, 'w+')

dep = open('global_parameters.py', 'r').read()
exec(dep)

numbanks = int(k*weight_stagger)

totnumtrits = imagewidth*imageheight*ni
tritsperbank = int(np.ceil(totnumtrits/numbanks))

effectivetritsperword = int(ni/weight_stagger)
physicaltritsperword = int(np.ceil(effectivetritsperword/5))*5
physicalbitsperword = int(physicaltritsperword / 5 * 8)
excessbits = (physicaltritsperword - effectivetritsperword)*2
effectivewordwidth = physicalbitsperword - excessbits
numdecoders = int(physicalbitsperword / 8)

bankdepth = int(np.ceil(tritsperbank/effectivetritsperword))

bankaddressdepth = int(np.ceil(np.log2(bankdepth)))

leftshiftbitwidth = int(np.ceil(np.log2(numbanks)))
splitbitwidth = int(np.ceil(np.log2(weight_stagger)))+1

nibitwidth = int(np.maximum(np.ceil(np.log2(ni)),1))+1
nobitwidth = int(np.maximum(np.ceil(np.log2(no)),1))
imagewidthbitwidth = int(np.maximum(np.ceil(np.log2(imagewidth)),1))+1
imageheightbitwidth = int(np.maximum(np.ceil(np.log2(imageheight)),1))+1

numaddresses = int(numbanks * bankdepth)
memaddressbitwidth = int(np.maximum(np.ceil(np.log2(numaddresses)),1))

leftshiftbitwidth = int(np.ceil(np.log2(numbanks)))
splitbitwidth = int(np.ceil(np.log2(weight_stagger)))+1

rowaddresswidth = int(np.ceil(np.log2(imw)))
matrixaddresswidth = int(np.ceil(np.log2(imw*imh)))
kaddresswidth = int(np.ceil(np.log2(k)))

_output = namedtuple("_outputs", "actmemory_external_acts_o")
_input = namedtuple("_inputs", "actmemory_external_bank_set actmemory_external_we actmemory_external_req actmemory_external_addr actmemory_external_wdata weightmemory_external_bank weightmemory_external_we weightmemory_external_req weightmemory_external_addr weightmemory_external_wdata ocu_thresh_pos ocu_thresh_neg ocu_thresholds_save_enable LUCA_store_to_fifo LUCA_testmode LUCA_imagewidth LUCA_imageheight LUCA_k LUCA_ni LUCA_no LUCA_stride_width LUCA_stride_height LUCA_padding_type LUCA_pooling_enable LUCA_pooling_pooling_type LUCA_pooling_kernel LUCA_pooling_padding_type LUCA_layer_skip_in LUCA_layer_skip_out LUCA_compute_disable")

outputtypes = _output("unsigned")
inputtypes = _input(actmemory.inputtypes.external_bank_set, actmemory.inputtypes.external_we, actmemory.inputtypes.external_req, actmemory.inputtypes.external_addr, actmemory.inputtypes.external_wdata, "unsigned", weightmemory.inputtypes.external_we, weightmemory.inputtypes.external_req, weightmemory.inputtypes.external_addr, weightmemory.inputtypes.external_wdata, ocu.inputtypes.thresh_pos, ocu.inputtypes.thresh_neg, ocu.inputtypes.threshold_store_to_fifo, LUCA.inputtypes.store_to_fifo, LUCA.inputtypes.testmode, LUCA.inputtypes.imagewidth, LUCA.inputtypes.imageheight, LUCA.inputtypes.k, LUCA.inputtypes.ni, LUCA.inputtypes.no, LUCA.inputtypes.stride_width, LUCA.inputtypes.stride_height, LUCA.inputtypes.padding_type, LUCA.inputtypes.pooling_enable, LUCA.inputtypes.pooling_pooling_type, LUCA.inputtypes.pooling_kernel, LUCA.inputtypes.pooling_padding_type, LUCA.inputtypes.skip_in, LUCA.inputtypes.skip_out, LUCA.inputtypes.compute_disable)

outputwidths = _output((physicalbitsperword,(1)))
inputwidths = _input(actmemory.inputwidths.external_bank_set, actmemory.inputwidths.external_we, actmemory.inputwidths.external_req, actmemory.inputwidths.external_addr, actmemory.inputwidths.external_wdata, (nobitwidth,1), weightmemory.inputwidths.external_we, weightmemory.inputwidths.external_req, weightmemory.inputwidths.external_addr, weightmemory.inputwidths.external_wdata, ocu.inputwidths.thresh_pos, ocu.inputwidths.thresh_neg, (1,128), LUCA.inputwidths.store_to_fifo, LUCA.inputwidths.testmode, LUCA.inputwidths.imagewidth, LUCA.inputwidths.imageheight, LUCA.inputwidths.k, LUCA.inputwidths.ni, LUCA.inputwidths.no, LUCA.inputwidths.stride_width, LUCA.inputwidths.stride_height, LUCA.inputwidths.padding_type,  LUCA.inputwidths.pooling_enable, LUCA.inputwidths.pooling_pooling_type, LUCA.inputwidths.pooling_kernel, LUCA.inputwidths.pooling_padding_type, LUCA.inputwidths.skip_in, LUCA.inputwidths.skip_out,LUCA.inputwidths.compute_disable)

cyclenum = 0

pipelinedelay = 1
widthcounter = 0
heightcounter = 0
counting = 1

codebook, orig_codebook = global_parameters.gen_codebook(str(accelStimuliPath+'/decoder_stimuli.txt'), str(accelStimuliPath+'/decoder_exp_responses.txt'))
reverse_codebook = {}

for x,y in codebook.items():
    if y not in reverse_codebook:
        reverse_codebook[y] = x

x_codebook = {}

for x,y in orig_codebook.items():
    if y not in x_codebook:
        x_codebook[y] = x

def format_output(output):

    string = ''
    
    for i in range(output.shape[0]):
        for k in range(output.shape[2]):
            for l in range(output.shape[3]):
                for j in range(output.shape[1]):
                    string += (format_ternary(output[i][j][k][l])) + ' ' 
                #print(string)
                string = ''

                
def double_threshold(x, xmin, xmax):

    xmin = torch.Tensor(xmin)
    xmax = torch.Tensor(xmax)
    
    xmin = torch.unsqueeze(xmin,1)
    xmin = torch.unsqueeze(xmin,2)
    xmin = xmin.repeat(1,x.shape[2],x.shape[3])

    xmax = torch.unsqueeze(xmax,1)
    xmax = torch.unsqueeze(xmax,2)
    xmax = xmax.repeat(1,x.shape[2],x.shape[3])
    
    max_t = x > xmax
    min_t = x < xmin
    
    return (max_t.float() - min_t.float())

def merge_inputs(weightmem, actmem, ocu, LUCA):
    
    retinput = _input(weightmemory.write_addr, weightmemory.write_enable, weightmemory.wdata, actmem.write_enable, actmem.write_enable_bank_set, actmem.write_addr, actmem.wdata, ocu.thresh_pos, ocu.thresh_neg, ocu.threshold_store_to_fifo, LUCA.fifo_empty, LUCA.testmode, LUCA.imagewidth, LUCA.imageheight, LUCA.k, LUCA.ni, LUCA.no, LUCA.stride_width, LUCA.stride_height, LUCA.padding_type, LUCA.compute_disable)

    return retinput

def translate_weights_to_weightmem(weights):
    
    weightmem = np.empty( (int(np.prod(weights.shape)/(ni/weight_stagger)), physicalbitsperword),dtype=int)
    #print('weights')
    #print(weights[0])
    #print(weights.shape)
    weightmemlist = []
    for i in range(weights.shape[0]):
        for n in range(int(weights.shape[1]/int(ni/weight_stagger))):
            for m in range(weights.shape[3]):
                for j in range(weights.shape[2]):
                    word = np.empty(int(ni/weight_stagger))
                    for q in range(int(ni/weight_stagger)):
                        word[q] = weights[i][n*int(ni/weight_stagger)+q][j][m]
                    _word = translate_ternary_sequence(word)
                    #print(_word)
                    weightmemlist.append(translate_binary_string(_word))
    weightmemarray = np.asarray(weightmemlist)
    weightmem = weightmemarray.reshape((int(np.prod(weights.shape)/(ni/weight_stagger)), physicalbitsperword))
    return weightmem
                    
def translate_image_to_actmem(image):
    
    actmem = np.empty((int(np.ceil((image.shape[1])/weight_stagger))*image.shape[2]*image.shape[3], physicalbitsperword), dtype=int)
    actmemlist = []

    for n in range(image.shape[2]):
        for m in range(image.shape[3]):
            for j in range(int(np.floor((image.shape[1])/(ni/weight_stagger)))):
                word = np.empty(int(ni/weight_stagger))
                for i in range(int(ni/weight_stagger)):
                    word[i] = image[0][i+j*int((ni/weight_stagger))][n][m]
                _word = translate_ternary_sequence(word)
                actmemlist.append(translate_binary_string(_word))

    actmemarray = np.asarray(actmemlist)
    actmem = actmemarray.reshape((-1, physicalbitsperword))

    #print(actmem.shape)
    return actmem

def tick(image, net):

    output = net(torch.Tensor(image)).int().numpy()
    #print("Output:")
    #print(format_output(output))
    return output

def translate_binary_string(string):
    ret = np.empty(len(string),dtype=int)
    for i in range(len(string)):
        ret[i] = string[i]

    return(ret)

def translate_ternary_sequence(seq):
    
    string = ''
    _seq = np.copy(seq.reshape(-1))
    
    for i in range(len(_seq)):
        if(int(_seq[i]) == 1):
            string += "01"
        elif(int(_seq[i]) == -1):
            string += "11"
        else:
            string += "00"
            
    string += "00"
    _string = ''
    for i in range(0,int(len(string)),10):
        substr = string[i:i+10]
        _string += reverse_codebook[substr]

    return _string

def makeNet(weights, layer_no,layer_ni,layer_k,strideh,stridew,thresh_pos,thresh_neg,layer_padding,pooling_enable,pooling_kernel,pooling_padding_type):

    #testweights = weights
    testweights = np.pad(weights, pad_width=((0,layer_no-weights.shape[0]),(0,layer_ni-weights.shape[1]),(0,0),(0,0)))
    
    if(pooling_enable == 1):
    
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(layer_ni, layer_no, layer_k, (strideh,stridew))
                with torch.no_grad():
                    self.conv1.weight = nn.Parameter(torch.Tensor(testweights))
                    self.conv1.bias = nn.Parameter(torch.Tensor(np.zeros(layer_no,dtype=int)))
                    if(layer_padding == 1):
                        self.conv1.padding = (int((layer_k-1)/2),int((layer_k-1)/2))
                self.pool1 = nn.MaxPool2d(kernel_size = (pooling_kernel, pooling_kernel))
                self.pool1.kernel_size = (pooling_kernel, pooling_kernel)
                self.pool1.stride = (pooling_kernel, pooling_kernel)
                if(pooling_padding_type == 1):
                    self.pool1.padding = 1
                else:
                    self.pool1.padding = 0
                    
            def forward(self, x):
                x = double_threshold(self.conv1(x),thresh_neg,thresh_pos)
                x = self.pool1(x)
                return x

    else:

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(layer_ni, layer_no, layer_k, (strideh,stridew))
                with torch.no_grad():
                    self.conv1.weight = nn.Parameter(torch.Tensor(testweights))
                    self.conv1.bias = nn.Parameter(torch.Tensor(np.zeros(layer_no,dtype=int)))
                    if(layer_padding == 1):
                        self.conv1.padding = (int((layer_k-1)/2),int((layer_k-1)/2))
            def forward(self, x):
                x = double_threshold(self.conv1(x),thresh_neg,thresh_pos)
                return x

        
    return Net()

def checkImplementation(baseline, mapping, featuremap, device='cpu'):

    layer_ni = mapping.conv1.weight.shape[1]
    image = torch.Tensor(np.pad(featuremap, pad_width=((0,0),(0,layer_ni-featuremap.shape[1]),(0,0),(0,0)))).to(device)
    
    baseres = baseline(torch.Tensor(featuremap).to(device)).detach()
    mapres = mapping(image).detach()

    #print(baseres[0][0])
    
    quantbaseres = ((baseres >= 0.5).int() - (baseres < -0.5).int())
    
    #     print(quantbaseres.shape)
    #     print(mapres.shape)
    
    #     print(quantbaseres[0][0])
    #     print(mapres[0][0])

    errorterm = (torch.abs(quantbaseres-mapres) > 1**(-3)).int()
    
    error = errorterm * baseres
    num_errors = torch.sum(errorterm)
    max_error = torch.max(torch.abs(error))
    
    return error, num_errors, max_error

def mapLayer(acts, weights, pooling, thresholds):
    
    layer_imagewidth = acts.shape[2]
    layer_imageheight = acts.shape[3]

    layer_ni = int(np.ceil(weights.shape[0]/(weight_stagger))*weight_stagger)
    layer_no = int(np.ceil(weights.shape[0]/(weight_stagger))*weight_stagger)
    layer_stridew = 1
    layer_strideh = 1
    layer_k = 3
    layer_thresh_neg = thresholds[0]
    layer_thresh_pos = thresholds[1]
    layer_padding = 1
    
    layer_pooling_enable = pooling
    layer_pooling_pooling_type = 0
    layer_pooling_padding_type = 0
    layer_pooling_kernel = 2

    layer_skip_in = 0
    layer_skip_out = 0

    print(layer_thresh_neg)
    print(layer_thresh_pos)
    
    #image = acts
    image = np.pad(acts, pad_width=((0,0),(0,layer_ni-acts.shape[1]),(0,0),(0,0)))
    
    net = makeNet(weights, int(np.ceil(layer_no/weight_stagger)*weight_stagger), int(np.ceil(layer_ni/weight_stagger)*weight_stagger) ,layer_k, layer_strideh, layer_stridew,
                            layer_thresh_pos,layer_thresh_neg,layer_padding,layer_pooling_enable, layer_pooling_kernel, layer_pooling_padding_type)
    
    actmem = translate_image_to_actmem(image)
    weightmem = translate_weights_to_weightmem(np.copy(net.conv1.weight.detach().numpy()))
    
    inference = image
    inference = tick(inference,net)
        
    if(layer_no < no):
        inf = np.zeros((inference.shape[0],no,inference.shape[2],inference.shape[3]))
        for m in range(inference.shape[0]):
            for i in range(0,layer_no,2):
                if(i<inference.shape[1]):
                    inf[m][i][:][:] = inference[m][i][:][:]
                else:
                    inf[m][i][:][:] = 0
    else:
        inf = inference.copy()
    
    actmemorywrites = int(np.ceil(layer_ni/(ni/weight_stagger))*layer_imagewidth*layer_imageheight)
    weightmemorywrites = int(np.ceil(layer_ni/(ni/weight_stagger))*layer_k*layer_k*layer_no)
    thresholdwrites = layer_no
    
    numwrites = np.maximum(np.maximum(actmemorywrites,weightmemorywrites),thresholdwrites)
    
#     print("actmemorywrites:" + str(actmemorywrites))
#     print("weightmemorywrites:" + str(weightmemorywrites))

    number_active_ocus = int(np.ceil(layer_no/(no/weight_stagger))*(ni/weight_stagger))
    weightmemory_writedepth = int(layer_k*layer_k*np.ceil(layer_ni/(ni/weight_stagger)))
    # Write settings seem okay

    computetime = layer_imagewidth*layer_imageheight*8
    
    for i in tqdm(range(numwrites + computetime)):
        
        actmemory_bank_set = 0
        actmemory_addr = 0
        actmemory_wdata = np.zeros((physicalbitsperword),dtype=int)
        actmemory_write_enable = 0
        actmemory_read_enable = 0
        
        if(i<actmemorywrites):
            actmemory_wdata = actmem[i]
            actmemory_addr = i
            actmemory_write_enable = 1

        weightmemory_read_enable = 0
            
        if(i<weightmemorywrites):
            weightmemory_write_enable = 1
            weightmemory_bank = (int(i/weightmemory_writedepth)%layer_no)
            weightmemory_addr = int(i/(weightmemory_writedepth*layer_no))*weightmemory_writedepth+(i%weightmemory_writedepth)
            weightmemory_wdata = weightmem[i]
        else:
            weightmemory_write_enable = 0
            weightmemory_bank = 0
            weightmemory_addr = 0
            weightmemory_wdata = np.zeros((weightmemory.numbanks,physicalbitsperword),dtype=int)

        ocu_thresh_pos = layer_thresh_pos[-1]
        ocu_thresh_neg = layer_thresh_neg[-1]

        ocu_thresholds_save_enable = np.zeros(no,dtype=int)
        if(i < thresholdwrites):
            ocu_thresh_pos = layer_thresh_pos[i%layer_no]
            ocu_thresh_neg = layer_thresh_neg[i%layer_no]
            ocu_thresholds_save_enable[i%layer_no] = 1

        if(i < numwrites):
            LUCA_store_to_fifo = 0
            LUCA_testmode = 0
            LUCA_imagewidth = layer_imagewidth
            LUCA_imageheight = layer_imageheight
            LUCA_k = layer_k
            LUCA_ni = layer_ni
            LUCA_no = layer_no
            LUCA_stride_height = layer_strideh
            LUCA_stride_width = layer_stridew
            LUCA_padding_type = layer_padding
            LUCA_compute_disable = 1
        elif(i >= numwrites and i < numwrites+1):
            LUCA_store_to_fifo = 0
            LUCA_testmode = 0
            LUCA_imagewidth = layer_imagewidth
            LUCA_imageheight = layer_imageheight
            LUCA_k = layer_k
            LUCA_ni = layer_ni
            LUCA_no = layer_no
            LUCA_stride_height = layer_strideh
            LUCA_stride_width = layer_stridew
            LUCA_padding_type = layer_padding
            LUCA_compute_disable = 0
        elif(i >= numwrites+1 and i < numwrites+computetime):
            LUCA_store_to_fifo = 0
            LUCA_testmode = 0
            LUCA_imagewidth = layer_imagewidth
            LUCA_imageheight = layer_imageheight
            LUCA_k = layer_k
            LUCA_ni = layer_ni
            LUCA_no = layer_no
            LUCA_stride_height = layer_strideh
            LUCA_stride_width = layer_stridew
            LUCA_padding_type = layer_padding
            LUCA_compute_disable = 0
        elif(i >= numwrites+computetime and i < numwrites+computetime+1):
            LUCA_store_to_fifo = 0
            LUCA_testmode = 0
            LUCA_imagewidth = layer_imagewidth
            LUCA_imageheight = layer_imageheight
            LUCA_k = layer_k
            LUCA_ni = layer_ni
            LUCA_no = layer_no
            LUCA_stride_height = layer_strideh
            LUCA_stride_width = layer_stridew
            LUCA_padding_type = layer_padding
            LUCA_compute_disable = 0
        else:
            LUCA_store_to_fifo = 0
            LUCA_testmode = 0
            LUCA_imagewidth = layer_imagewidth
            LUCA_imageheight = layer_imageheight
            LUCA_k = layer_k
            LUCA_ni = layer_ni
            LUCA_no = layer_no
            LUCA_stride_height = layer_strideh
            LUCA_stride_width = layer_stridew
            LUCA_padding_type = layer_padding
            LUCA_compute_disable = 0

        if(i < 1):
            LUCA_store_to_fifo = 1            
            
            
        if(weightmemory_write_enable == 1):
            weightmemory_req = 1
            weightmemory_we = 1
        elif(weightmemory_read_enable == 1):
            weightmemory_req = 1
            weightmemory_we = 0
        else:
            weightmemory_req = 0
            weightmemory_we = 0
            
        if(actmemory_write_enable == 1):
            actmemory_req = 1
            actmemory_we = 1
        elif(actmemory_read_enable == 1):
            actmemory_req = 1
            actmemory_we = 0
        else:
            actmemory_req = 0
            actmemory_we = 0
            
        curr_input = _input(actmemory_bank_set, actmemory_we, actmemory_req, actmemory_addr, actmemory_wdata, weightmemory_bank, weightmemory_we, weightmemory_req, weightmemory_addr, weightmemory_wdata,  ocu_thresh_pos,ocu_thresh_neg,ocu_thresholds_save_enable,LUCA_store_to_fifo,LUCA_testmode,LUCA_imagewidth,LUCA_imageheight,LUCA_k,LUCA_ni,LUCA_no,LUCA_stride_width,LUCA_stride_height,LUCA_padding_type,layer_pooling_enable, layer_pooling_pooling_type, layer_pooling_kernel, layer_pooling_padding_type, layer_skip_in, layer_skip_out, LUCA_compute_disable)
        #print(curr_input)
        f.write("%s \n" % format_input(curr_input))

#     print(inference)
#     print(inference.shape)

    string = ''
    codes = []
    for i in range(inference.shape[0]):
        for k in range(inference.shape[2]):
            for l in range(inference.shape[3]):
                for j in range(int(np.ceil(inference.shape[1]/64))):
                    for q in range(64):
                        string += (format_ternary(inference[i][j*64+q][k][l]))
                    reqlength = int(int(np.ceil(len(string)/130))*130)
                    string = string + '0'*(reqlength-len(string))
                    codes.append(string)
                    string = ''

    encoded_string = ''
    encoded_strings = []
    for string in codes:
        for i in range(0,len(string),10):
            encoded_string += x_codebook[string[i:i+10]]
         
        encoded_strings.append(encoded_string)
        encoded_string = ''

#     print(encoded_strings)

    for string in encoded_strings:
        for i in range(0,len(string),104):
            out = string[i:i+104]
            g.write("%s \n" % out)
            
    return net

if __name__=='__main__':

    session_name = 'default'
    
    parser = argparse.ArgumentParser(description="Stimuli generator for")
    parser.add_argument('-s', metavar='Sessionname', dest='session_name', type=str, default=session_name, help='Set the session name for logging purposes')
    parser.add_argument('-l', metavar='Layer', dest='layer', type=int, default=0, help='Set the layer index')
    args = parser.parse_args()

    _weights = []

    weightsnpzfile = np.load(f'./models/{args.session_name}/weights.npz')
    for i in weightsnpzfile.files:
        _weights.append(weightsnpzfile[i])
        
    weights = np.asarray(_weights[args.layer])

    _acts = []

    actsnpzfile = np.load(f'./models/{args.session_name}/acts.npz',allow_pickle=True)
    for i in actsnpzfile.files:
        _acts.append(actsnpzfile[i])
        
    acts = np.asarray(_acts[args.layer])

    _pooling = []

    poolingnpzfile = np.load(f'./models/{args.session_name}/pooling.npz',allow_pickle=True)
    for i in poolingnpzfile.files:
        _pooling.append(poolingnpzfile[i])
        
    pooling = np.asarray(_pooling[args.layer])

    _thresholds = []

    thresholdsnpzfile = np.load(f'./models/{args.session_name}/thresholds.npz',allow_pickle=True)
    for i in thresholdsnpzfile.files:
        _thresholds.append(thresholdsnpzfile[i])
        
    thresholds = np.asarray(_thresholds[args.layer])
    thresholds = np.transpose(thresholds,(1,0))

    thresholds = np.clip(thresholds, -2**(ocu.inputwidths.thresh_pos[0]-1),2**(ocu.inputwidths.thresh_pos[0]-1)-1)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")    
    device = 'cpu'

    mappednet = mapLayer(acts,weights,pooling,thresholds).to(device)


#     baselinenet = HardDense_Model(128, 10, None, 0, weightQuantSchedule, True, True).to(device)
#     ckpt = torch.load(f'./models/{args.session_name}-ckpt.pth')
#     baselinenet.load_state_dict(ckpt['net'], strict=False)
#     baselinenet.eval()

#     parser = NetParser(expressionList)
#     layerList = parser.parse(baselinenet)

#     errors, num_errors, max_error = checkImplementation(layerList[args.layer], mappednet, acts, device)

#     print(num_errors)
    
#     for i in layerList:
#         if (isinstance(i, FusedConvBlock)):
#             layer = i.convLayer
#         else:
#             layer = i
#         if(len(layer.layers)>3):
#             layer.layers[2].eval()

