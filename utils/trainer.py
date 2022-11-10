#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.networks import UNaah
from utils.metrics import ConfusionMatrix
import os

torch.backends.cudnn.deterministic = True

class Trainer(object):
    def __init__(self, criterion, optimizer, n_class):
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics = ConfusionMatrix(n_class)
        self.n_class = n_class
    
    def set_train(self, model):
        model.train()

    def get_scores(self):
        score_train = self.metrics.get_scores()
        return score_train

    def reset_metrics(self):
        self.metrics.reset()
        
    def train(self, sample, model, global_fixed):
        images, labels = sample['image'], sample['label'] # PIL images
        ids = sample['id']
        width, height = images[0].size
        
        if width != self.size_g[0] or height != self.size_g[1]:
            images_glb = resize(images, self.size_g) # list of resized PIL images
        else:
            images_glb = list(images)
        
        images_glb = images_transform(images_glb)
        labels_glb = masks_transform(labels)

        if self.mode == 2 or self.mode == 3:
            patches, coordinates, sizes, ratios, label_patches = global2patch(images, labels_glb, self.size_p, ids)
            predicted_patches = np.zeros((len(images), self.n_class)) 
            predicted_ensembles = np.zeros((len(images), self.n_class))
            outputs_global = [ None for i in range(len(images)) ]

        if self.mode == 1:
            # training with only (resized) global image #########################################
            outputs_global, _ = model.forward(images_glb, None, None, None)
            loss = self.criterion(outputs_global, labels_glb)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()
            ##############################################

        if self.mode == 2:
            # training with patches ###########################################
            for i in range(len(images)):
                j = 0
                while j < len(coordinates[i]):
                    patches_var = images_transform(patches[i][j : j+self.sub_batch_size]) # b, c, h, w
                    label_patches_var = masks_transform(label_patches[i][j : j+self.sub_batch_size])             
                    output_ensembles, output_global, output_patches, fmreg_l2 = model.forward(images_glb[i:i+1], patches_var, coordinates[i][j : j+self.sub_batch_size], ratios[i], mode=self.mode, n_patch=len(coordinates[i]))
                    loss = self.criterion(output_patches, label_patches_var) + self.criterion(output_ensembles, label_patches_var) + self.lamb_fmreg * fmreg_l2
                    loss.backward()

                    # patch predictions
                    for pred_patch, pred_ensemble in zip(torch.max(output_patches.data, 1)[1].data, torch.max(output_ensembles.data, 1)[1].data):
                        predicted_patches[i][int(pred_patch)] += 1
                        predicted_ensembles[i][int(pred_ensemble)] += 1

                    j += self.sub_batch_size

                outputs_global[i] = output_global

            outputs_global = torch.cat(outputs_global, dim=0)

            self.optimizer.step()
            self.optimizer.zero_grad()
            #####################################################################################

        if self.mode == 3:
            # train global with help from patches ##################################################
            # go through local patches to collect feature maps
            # collect predictions from patches
           
            for i in range(len(images)):
                j = 0
                while j < len(coordinates[i]):
                    patches_var = images_transform(patches[i][j : j+self.sub_batch_size]) # b, c, h, w
                    _, output_patches = model.module.collect_local_fm(images_glb[i:i+1], patches_var, ratios[i], coordinates[i], [j, j+self.sub_batch_size], len(images), global_model=global_fixed, n_patch_all=len(coordinates[i]))

                    for pred_patch in torch.max(output_patches.data, 1)[1].data:
                        predicted_patches[i][int(pred_patch)] += 1
                
                    j += self.sub_batch_size
            
            # train on global image

            outputs_global, fm_global = model.forward(images_glb, None, None, None, mode=self.mode)

            loss = self.criterion(outputs_global, labels_glb)
            loss.backward(retain_graph=True)

            # fmreg loss
            # generate ensembles & calc loss
            for i in range(len(images)):
                j = 0
                while j < len(coordinates[i]):
                    label_patches_var = masks_transform(label_patches[i][j : j+self.sub_batch_size])
                    patches_var = images_transform(patches[i][j : j+self.sub_batch_size]) # b, c, h, w

                    fl = model.module.generate_local_fm(images_glb[i:i+1], patches_var, ratios[i], coordinates[i], [j, j+self.sub_batch_size], len(images), global_model=global_fixed, n_patch_all=len(coordinates[i]))
                    fg = model.module._crop_global(fm_global[i:i+1], coordinates[i][j:j+self.sub_batch_size], ratios[i])[0]
                    fg = F.interpolate(fg, size=fl.size()[2:], mode='bilinear')
                    output_ensembles = model.module.ensemble(fl, fg)

                    loss = self.criterion(output_ensembles, label_patches_var)# + 0.15 * mse(fl, fg)
                    if i == len(images) - 1 and j + self.sub_batch_size >= len(coordinates[i]):
                        loss.backward()
                    else:
                        loss.backward(retain_graph=True)

                    # ensemble predictions
                    for pred_ensemble in torch.max(output_ensembles.data, 1)[1].data:      
                        predicted_ensembles[i][int(pred_ensemble)] += 1                  

                    j += self.sub_batch_size
          
            self.optimizer.step()
            self.optimizer.zero_grad()

        # global predictions ###########################
        _, predictions_global = torch.max(outputs_global.data, 1)
        self.metrics_global.update(labels_glb, predictions_global)

        if self.mode == 2 or self.mode == 3:
            # patch predictions ###########################
            predictions_local = predicted_patches.argmax(1)
            #self.metrics_local.update(labels_npy, predictions_local)
            self.metrics_local.update(labels_glb, predictions_local)
            ###################################################
            # combined/ensemble predictions ###########################
            predictions = predicted_ensembles.argmax(1)
            self.metrics.update(labels_glb, predictions)
        return loss
