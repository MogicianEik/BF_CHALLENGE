#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import yaml
from tqdm import tqdm
from models.unet_1d import UNET_1D
from dataset.snp_feature import KFoldTrainDataLoader, TestDataLoader, PredictDataLoader
from tensorboardX import SummaryWriter
from option import Options
from utils.lr_scheduler import LR_Scheduler
from utils.trainer import Trainer
from utils.evaluator import Evaluator, Predictor
from torchsummary import summary
from utils.metrics import ConfusionMatrix

# single decoder version
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
prediction = args.prediction
print("prediction:", prediction)

###################################
print("preparing datasets and dataloaders......")
batch_size = args.batch_size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# get train, val, test
if not evaluation and not prediction:
    dataloaders_train, dataloaders_val, label_encoder_name_mapping = KFoldTrainDataLoader(args.train_file, batch_size, n_class, kfold)
    print("Mapping of Label Encoded Classes", label_encoder_name_mapping, sep="\n")
    
    start_time = time.time()
    for rotation in range(kfold):
        print("creating models, rotation %s"%rotation)
                
        num_epochs = args.num_epochs
        learning_rate = args.lr
        model = UNET_1D(n_class,1,128,7,5).cuda() #(input_dim, hidden_layer, kernel_size, depth)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = LR_Scheduler('poly', learning_rate, num_epochs, len(dataloaders_train[rotation]))
        criterion = nn.CrossEntropyLoss()
        writer = SummaryWriter(log_dir=log_path + task_name+'_%s'%rotation)
        f_log = open(log_path + task_name+'_%s'%rotation + ".log", 'w')
        trainer = Trainer(criterion, optimizer, n_class)
        evaluator = Evaluator(n_class, criterion, evaluation)

        best_pred = float('inf')
        for epoch in range(num_epochs):
            trainer.set_train(model)
            optimizer.zero_grad()
            train_loss = 0
            val_loss = 0
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
                        loss, predicted_gt = evaluator.eval_test(sample_batched, model)
                        val_loss += loss.item()
                        print('[%d/%d] Val loss: %.3f' % ((i_batch + 1)*batch_size, len(dataloaders_val[rotation])*batch_size, val_loss / (i_batch + 1)))
                        
                    if val_loss < best_pred:
                        best_pred = val_loss                  
                        if not evaluation:
                            print("saving model...")
                            torch.save(model.state_dict(), model_path + task_name + "_%s.pth" % rotation)

                    elapsed_time = time.time() - start_time 
                    log = ""
                    log = log + 'epoch [{}/{}] loss: train = {:.4f}, val = {:.4f}, time={:.2f}s'.format(epoch+1, num_epochs, train_loss, val_loss, elapsed_time) + "\n"
                    log += "================================\n"
                    print(log)

                    f_log.write(log)
                    f_log.flush()
                    writer.add_scalars('Loss', {'Train loss': train_loss, 'Validation loss': val_loss}, epoch)

        f_log.close()

# TODO, complete evaluation
elif evaluation:
    dataloader_test = TestDataLoader(args.eval_file, args.label_file, batch_size, n_class)
    CM = ConfusionMatrix(n_class)
    start_time = time.time()
    accs = []
    for rotation in range(kfold):
        print("evaluating models, rotation %s"%rotation)
        CM.reset()
        model = UNET_1D(n_class,1,128,7,5).cuda()
        model.load_state_dict(torch.load(model_path + task_name + "_%s.pth" % rotation))
        model.eval()
        evaluator = Evaluator(n_class = n_class, test = evaluation)
        for i_batch, sample_batched in enumerate(dataloader_test):
            _, predicted_gt = evaluator.eval_test(sample_batched, model)
            print(predicted_gt, sample_batched['GT'])
            CM.update(sample_batched['GT'].cpu().numpy(), predicted_gt)
        print('=============================================================')
        accs.append(CM.get_scores()['accuracy'])
    
    for rotation in range(kfold):
        print ('Accuracy of rotation %s: %s'%(rotation, accs[rotation]))

elif prediction:
    dataloader_pred = PredictDataLoader(args.pred_file, batch_size, n_class)
    labels = yaml.load(open(args.label_file, "r"), Loader=yaml.FullLoader)
    print("evaluating unlabeled data, using model: %s" % (model_path + task_name + "_" + args.pred_rotation + ".pth"))
    model = UNET_1D(n_class,1,128,7,5).cuda()
    model.load_state_dict(torch.load(model_path + task_name + "_" + args.pred_rotation + ".pth"))
    model.eval()
    predictor = Predictor(labels['labels'])
    pred_results = {'ID':[], 'Class':[]}
    for i_batch, sample_batched in enumerate(dataloader_pred):
        sample_id, predicted_gt = predictor.eval_test(sample_batched, model)
        pred_results['ID'].append(sample_id)
        pred_results['Class'].append(predicted_gt)
    df = pd.DataFrame.from_dict(pred_results)
    df.to_csv(args.pred_file.split('.')[0] + 'SD_PredResult.csv', index=False)

        
            
