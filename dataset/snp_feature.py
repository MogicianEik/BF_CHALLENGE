import os
import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
import random
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import yaml

class SNPFeature(data.Dataset):
    """input and label dataset for training & validation"""

    def __init__(self, X, GT1, GT2, n_class, transform = None):
        super(SNPFeature, self).__init__()
        """
        Args:
        X: inputs, SNP features, each feature is a binary code.
        GT: haplotype combo code, 0 - two copies are both H1B1G1, 1 - only one H1B1G1, 2 - 0 copies
        transform(callable, optional): Optional transform to be applied on a sample
        """
        self.X = X
        self.GT1 = GT1
        self.GT2 = GT2
        self.n_class = n_class
        self.transform = transform

    def __getitem__(self, index):
        sample = {}
        sample['SNP'] = torch.tensor(self.X[index], dtype=torch.float)
        if self.GT1[index] + self.GT2[index] == 1:
            sample['GT'] = 1
        elif self.GT1[index] + self.GT2[index] > 1:
            sample['GT'] = 2
        else:
            sample['GT'] = 0

        return sample

    def __len__(self):
        return len(self.X)
        
transformer = transforms.Compose([
    transforms.ToTensor(),
])

class SNPFeature_predict(data.Dataset):
    """input for predict dataset"""
    def __init__(self, X, X_id, n_class, transform = None):
        super(SNPFeature_predict, self).__init__()
        """
        Args:
        X: inputs, SNP features, each feature is a binary code.
        X_id: ID for each SNP feature
        transform(callable, optional): Optional transform to be applied on a sample
        """
        self.X = X
        self.id = X_id
        self.n_class = n_class
        self.transform = transform

    def __getitem__(self, index):
        sample = {}
        sample['SNP'] = torch.tensor(self.X[index], dtype=torch.float)
        sample['ID'] = self.id[index]

        return sample

    def __len__(self):
        return len(self.X)
        
def KFoldTrainDataLoader(train_file, batch_size, n_class, k = 5):
    '''
    added a haplotype level data split. H1 and H2 data are splited separatedly and merged.
    Since each patient has two haplotypes, the split uses 1-drop rule. If a patient
    has one H2 label, it will be assigned to the H2 group.
    '''
    
    dataloaders_train = []
    dataloaders_val = []
    # extract inputs and targets
    df = pd.read_csv(train_file, sep = '\t')
    X = df.iloc[:, :-2].values # SNP features
    X = np.pad(X, ((0,0),(0,3500-X.shape[1])), 'constant')# raw resize, make the frame work.
    X = X.reshape(-1,1,X.shape[1]) # Used to be X.shape[1]
    y1 = df.iloc [:, -2].values # hap1
    y2 = df.iloc [:, -1].values # hap2
    # encode the the targets
    label_encoder = LabelEncoder()
    gt1 = label_encoder.fit_transform(y1)
    gt2 = label_encoder.fit_transform(y2)
    label_encoder_name_mapping = dict(zip(label_encoder.classes_,
                                             label_encoder.transform(label_encoder.classes_)))
    kf = KFold(n_splits=k, shuffle=False)
    H1 = []
    H2 = []
    for index in range(len(X)):
        if gt1[index] > 5 or gt2[index] > 5:
            H2.append(index)
        else:
            H1.append(index)
    for H1_index, H2_index in zip(kf.split(H1), kf.split(H2)):
        train_index = list(H1_index[0]) + list(H2_index[0])
        test_index = list(H1_index[1]) + list(H2_index[1])
        dataset_train = SNPFeature(X[train_index], gt1[train_index], gt2[train_index], n_class, transformer)
        dataloader_train = data.DataLoader(dataset=dataset_train, batch_size=batch_size, num_workers=16, shuffle=True, pin_memory=True)
        dataset_val = SNPFeature(X[test_index], gt1[test_index], gt2[test_index], n_class, transformer)
        dataloader_val = data.DataLoader(dataset=dataset_val, batch_size=batch_size, num_workers=16, shuffle=True, pin_memory=True)
        dataloaders_train.append(dataloader_train)
        dataloaders_val.append(dataloader_val)
    return dataloaders_train, dataloaders_val, label_encoder_name_mapping
    
def TestDataLoader(eval_file, label_file, batch_size, n_class):
    df = pd.read_csv(eval_file, sep = '\t')
    labels = yaml.load(open(label_file, "r"), Loader=yaml.FullLoader)
    X = df.iloc[:, :-2].values
    X = np.pad(X, ((0,0),(0,3500-X.shape[1])), 'constant')
    X = X.reshape(-1,1,X.shape[1])
    y1 = df.iloc [:, -2].values # hap1
    y2 = df.iloc [:, -1].values # hap2
    for label in labels['labels']:
        y1[y1 == label] = labels['labels'][label]
        y2[y2 == label] = labels['labels'][label]

    dataset_test = SNPFeature(X, y1, y2, n_class, transformer)
    dataloader_test = data.DataLoader(dataset=dataset_test, batch_size=batch_size, num_workers=16, shuffle=True, pin_memory=True)
    return dataloader_test
    
def PredictDataLoader(pred_file, batch_size, n_class):
    df = pd.read_csv(pred_file, sep = '\t')
    X = df.iloc[:, :-2].values
    X = np.pad(X, ((0,0),(0,3500-X.shape[1])), 'constant')
    X = X.reshape(-1,1,X.shape[1])
    # create a temp pseudo ID for each SNP features
    ids = list(range(X.shape[0]))
    X_id = np.array([ids]).T
    dataset_pred = SNPFeature_predict(X, X_id, n_class, transformer)
    dataloader_pred = data.DataLoader(dataset=dataset_pred, batch_size=batch_size, num_workers=16, shuffle=True, pin_memory=True)
    return dataloader_pred
