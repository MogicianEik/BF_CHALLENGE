#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from utils.metrics import ConfusionMatrix
import os

torch.backends.cudnn.deterministic = True

# single decoder version, 3 class labels
class Evaluator(object):
    def __init__(self, n_class, criterion = None,test=False):
        self.criterion = criterion
        self.test = test
        self.metrics = ConfusionMatrix(n_class)
    
    def get_scores(self):
        score_val = self.metrics.get_scores()
        return score_val

    def reset_metrics(self):
        self.metrics.reset()

    def eval_test(self, sample, model):
        with torch.no_grad():
            snps = sample['SNP']
            snps = snps.cuda()
            if not self.test:
                GTs = sample['GT']
                GTs = GTs.type('torch.LongTensor').cuda()
            
            gt = model.forward(snps)
            predicted_gt = torch.argmax(gt, dim=1).cpu().numpy()
            if not self.test:
                loss = self.criterion(gt, GTs)
                return loss, predicted_gt
            else:
                return None, predicted_gt
                
class Predictor(object):
    def __init__(self, label_dict):
        inv_map = {v: k for k, v in label_dict.items()}
        self.label_dict = inv_map

    def eval_test(self, sample, model):
        with torch.no_grad():
            snps = sample['SNP']
            snps = snps.cuda()
            
            gt = model.forward(snps)
            predicted_gt = torch.argmax(gt, dim=1).cpu().numpy()[0]
            return sample['ID'].cpu().numpy()[0][0], predicted_gt
