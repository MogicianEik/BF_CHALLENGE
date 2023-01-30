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
                hap1s = sample['hap1']
                hap2s = sample['hap2']
                hap1s = hap1s.type('torch.LongTensor').cuda()
                hap2s = hap2s.type('torch.LongTensor').cuda()
            
            h1, h2 = model.forward(snps)
            predicted_hap1 = torch.argmax(h1, dim=1).cpu().numpy()
            predicted_hap2 = torch.argmax(h2, dim=1).cpu().numpy()
            if not self.test:
                loss = min(self.criterion(h1, hap1s) + self.criterion(h2, hap2s),
                           self.criterion(h2, hap1s) + self.criterion(h1, hap2s))
                return loss, predicted_hap1, predicted_hap2
            else:
                return None, predicted_hap1, predicted_hap2
                
class Predictor(object):
    def __init__(self, label_dict):
        inv_map = {v: k for k, v in label_dict.items()}
        self.label_dict = inv_map

    def eval_test(self, sample, model):
        with torch.no_grad():
            snps = sample['SNP']
            snps = snps.cuda()
            
            h1, h2 = model.forward(snps)
            predicted_hap1 = torch.argmax(h1, dim=1).cpu().numpy()
            predicted_hap2 = torch.argmax(h2, dim=1).cpu().numpy()
            return sample['ID'].cpu().numpy()[0][0], self.label_dict[predicted_hap1[0]], self.label_dict[predicted_hap2[0]]
