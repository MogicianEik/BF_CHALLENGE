#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from dataset.snp_feature import KFoldTrainDataLoader
from tensorboardX import SummaryWriter
from option import Options
from utils.lr_scheduler import LR_Scheduler
from utils.trainer import Trainer
from utils.evaluator import Evaluator

args = Options().parse()
n_class = args.n_class
kfold = args.kfold
torch.backends.cudnn.deterministic = True

model_path = args.model_path
if not os.path.isdir(model_path): os.mkdir(model_path)
log_path = args.log_path
if not os.path.isdir(log_path): os.mkdir(log_path)
task_name = args.task_name

print(task_name)
###################################
evaluation = args.evaluation
print("evaluation:", evaluation)

###################################
print("preparing datasets and dataloaders......")
batch_size = args.batch_size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# get train, val, test
if not evaluation:
    dataloaders_train, dataloaders_val = KFoldTrainDataLoader(args.train_file, batch_size, kfold)
    for rotation in range(kfold):
        print("creating models, rotation %s"%rotation)
        
        num_epochs = args.num_epochs
        learning_rate = args.lr
        model = UNaah(backbone_name='resnet50',pretrained=True,classes=n_class).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = LR_Scheduler('poly', learning_rate, num_epochs, len(dataloaders_train[rotation]))
        criterion = nn.CrossEntropyLoss()
        writer = SummaryWriter(log_dir=log_path + task_name+'_s'%rotation)
        f_log = open(log_path + task_name+'_s'%rotation + ".log", 'w')
        trainer = Trainer(criterion, optimizer, n_class)
        evaluator = Evaluator(n_class)

        best_pred = 0.0
        for epoch in range(num_epochs):
            trainer.set_train(model)
            optimizer.zero_grad()
            train_loss = 0
            total = 0
            for i_batch, sample_batched in enumerate(dataloaders_train[rotation]):
                scheduler(optimizer, i_batch, epoch, best_pred)
                loss = trainer.train(sample_batched, model)
                total += len(sample_batched['image'])
                train_loss += loss.item()
                score_train = trainer.get_scores()
                if mode == 1: print('[%d/%d] Train loss: %.3f; global acc: %.3f' % ((i_batch + 1)*batch_size, len(dataloaders_train[rotation])*batch_size, train_loss / (i_batch + 1), score_train_global["accuracy"]))
                else: print('[%d/%d] Train loss: %.3f; agg acc: %.3f' % ((i_batch + 1)*batch_size, len(dataloaders_train[rotation])*batch_size, train_loss / (i_batch + 1), score_train_global["accuracy"]))

            score_train, score_train_global, score_train_local = trainer.get_scores()
            trainer.reset_metrics()
            # torch.cuda.empty_cache()

            if epoch % 1 == 0:
                with torch.no_grad():
                    model.eval()
                    print("evaluating...")

                    for i_batch, sample_batched in enumerate(dataloaders_val[rotation]):
                        predictions, predictions_global, predictions_local, outputs_global = evaluator.eval_test(sample_batched, model, global_fixed)
                        prob_scores.append(outputs_global[0][sample_batched['label'][0]])
                        score_val, score_val_global, score_val_local = evaluator.get_scores()
                        # use [1:] since class0 is not considered in deep_globe metric
                        if mode == 1: print('[%d/%d] global acc: %.3f GT: %s' % ((i_batch + 1)*batch_size, len(dataloaders_val[rotation])*batch_size, score_val_global["accuracy"], sample_batched['label']))
                        else: print('[%d/%d] agg acc: %.3f GT: %s' % ((i_batch + 1)*batch_size, len(dataloaders_val[rotation])*batch_size, score_val["accuracy"], sample_batched['label']))
                        
                    score_val = evaluator.get_scores()
                    evaluator.reset_metrics()
                    if score_val["accuracy"] > best_pred:
                        best_pred = score_val["accuracy"]
                        if not evaluation:
                            print("saving model...")
                            torch.save(model.state_dict(), model_path + task_name+'_s'%rotation + ".pth")


                    log = ""
                    log = log + 'epoch [{}/{}] acc: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs, score_train["accuracy"], score_val["accuracy"]) + "\n"
                    log = log + "train: " + str(score_train["accuracy"]) + "\n"
                    log = log + "val acc:" + str(score_val["accuracy"]) + "\n"
                    log += "================================\n"
                    print(log)

                    if evaluation:
                        break

                    f_log.write(log)
                    f_log.flush()
                    writer.add_scalars('accuracy', {'train accuracy': score_train["accuracy"], 'validation accuracy': score_val["accuracy"]}, epoch)

        f_log.close()
