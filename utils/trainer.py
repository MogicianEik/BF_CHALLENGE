#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os

torch.backends.cudnn.deterministic = True

class Trainer(object):
    def __init__(self, criterion, optimizer):
        self.criterion = criterion
        self.optimizer = optimizer
    
    def set_train(self, model):
        model.train()
        
    def train(self, sample, model):
        snps, hap1s, hap2s = sample['SNP'], sample['hap1'], sample['hap2']
        # convert data into cuda
        snps = snps.cuda()
        hap1s = hap1s.type('torch.LongTensor').cuda()
        hap2s = hap2s.type('torch.LongTensor').cuda()
        
        # get outputs
        h1, h2 = model.forward(snps)
        loss = min(self.criterion(h1, hap1s) + self.criterion(h2, hap2s), self.criterion(h1, hap2s) + self.criterion(h2, hap1s))
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss
