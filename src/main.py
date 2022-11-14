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
from models.unet import unet
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
    dataloaders_train, dataloaders_val, label_encoder_name_mapping = KFoldTrainDataLoader(args.train_file, batch_size, kfold)
    print("Mapping of Label Encoded Classes", label_encoder_name_mapping, sep="\n")
    for rotation in range(kfold):
        print("creating models, rotation %s"%rotation)
        
        num_epochs = args.num_epochs
        learning_rate = args.lr
        model = unet(out_classes=n_class, dimensions=2, monte_carlo_dropout=0.15).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = LR_Scheduler('poly', learning_rate, num_epochs, len(dataloaders_train[rotation]))
        criterion = nn.CrossEntropyLoss()
        writer = SummaryWriter(log_dir=log_path + task_name+'_%s'%rotation)
        f_log = open(log_path + task_name+'_%s'%rotation + ".log", 'w')
        trainer = Trainer(criterion, optimizer)
        evaluator = Evaluator(criterion, evaluation)

        best_pred = float('inf')
        for epoch in range(num_epochs):
            trainer.set_train(model)
            optimizer.zero_grad()
            train_loss = 0
            total = 0
            for i_batch, sample_batched in enumerate(dataloaders_train[rotation]):
                scheduler(optimizer, i_batch, epoch, best_pred)
                loss = trainer.train(sample_batched, model)
                total += len(sample_batched['SNP'])
                train_loss += loss.item()
                print('[%d/%d] Train loss: %.3f' % ((i_batch + 1)*batch_size, len(dataloaders_train[rotation])*batch_size, train_loss / (i_batch + 1)))

            trainer.reset_metrics()

            if epoch % 1 == 0:
                with torch.no_grad():
                    model.eval()
                    print("evaluating...")

                    for i_batch, sample_batched in enumerate(dataloaders_val[rotation]):
                        loss, predicted_hap1, predicted_hap2 = evaluator.eval_test(sample_batched, model)
                        val_loss += loss.item()
                        print('[%d/%d] Val loss: %.3f' % ((i_batch + 1)*batch_size, len(dataloaders_val[rotation])*batch_size, val_loss / (i_batch + 1)))
                        
                    if val_loss < best_pred:
                        best_pred = val_loss                  
                        if not evaluation:
                            print("saving model...")
                            torch.save(model.state_dict(), model_path + task_name + ".pth")


                    log = ""
                    log = log + 'epoch [{}/{}] loss: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs, train_loss, val_loss) + "\n"
                    log += "================================\n"
                    print(log)

                    f_log.write(log)
                    f_log.flush()
                    writer.add_scalars('Loss', {'Train loss': train_loss, 'Validation loss': val_loss}, epoch)

        f_log.close()

# TODO, complete evaluation
