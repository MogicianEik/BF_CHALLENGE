import os
import argparse
import torch

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch Classification')
        # model and dataset 
        parser.add_argument('--n_class', type=int, default=3, help='classification classes')
        parser.add_argument('--train_file', type=str, default='data/train.txt', help='path to train data')
        parser.add_argument('--eval_file', type=str, default='data/val.txt', help='path to evaluation data')
        parser.add_argument('--pred_file', type=str, default='data/pred.txt', help='path to prediction data')
        parser.add_argument('--label_file', type=str, default='data/labels.yaml', help='path to labels for the evaluation')
        parser.add_argument('--model_path', type=str, help='path to trained model')
        parser.add_argument('--log_path', type=str, help='path to log files')
        parser.add_argument('--task_name', type=str, help='task name for naming saved model files and log files')
        parser.add_argument('--evaluation', action='store_true', default=False, help='evaluation only')
        parser.add_argument('--prediction', action='store_true', default=False, help='prediction only')
        parser.add_argument('--pred_rotation', type=str, help='model rotation id used for prediction')
        parser.add_argument('--batch_size', type=int, default=6, help='batch size for number of subjects. Has to be 1 for evaluation and prediction')
        parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
        parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs for training')
        parser.add_argument('--kfold', type=int, default=5, help='number of folds for CV')

        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        return args
