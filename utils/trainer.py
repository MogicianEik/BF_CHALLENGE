#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.metrics import ConfusionMatrix
import os

torch.backends.cudnn.deterministic = True

# single decoder version, 3 class labels
class Trainer(object):
    def __init__(self, criterion, optimizer, n_class):
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics = ConfusionMatrix(n_class)
    
    def set_train(self, model):
        model.train()
        
    def reset_metrics(self):
        self.metrics.reset()
        
    def get_scores(self):
        score_train = self.metrics.get_scores()
        return score_train
        
    def train(self, sample, model):
        snps, GTs = sample['SNP'], sample['GT']
        # convert data into cuda
        snps = snps.cuda()
        GTs = GTs.type('torch.LongTensor').cuda()
        
        # get outputs
        GT = model.forward(snps)
        loss = self.criterion(GT, GTs)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss
