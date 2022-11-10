import os
import torch.utils.data as data
import pandas as pd
import random
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

class SNPFeature(data.Dataset):
    """input and label image dataset"""

    def __init__(self, X, GT1, GT2, transform = None):
        super(SNPFeature, self).__init__()
        """
        Args:
        X: inputs, SNP features, each feature is a binary code.
        GT1: haplotype labels
        GT2: haplotype labels
        transform(callable, optional): Optional transform to be applied on a sample
        """
        self.X = X
        self.GT1 = GT1
        self.GT2 = GT2
        self.transform = transform

    def __getitem__(self, index):
        sample = {}
        sample['SNP'] = self.transform(self.X)
        sample['hap1'] = self.transform(self.GT1)
        sample['hap2'] = self.transform(self.GT2)

        return sample

    def __len__(self):
        return len(self.X)
        
transformer = transforms.Compose([
    transforms.ToTensor(),
])
        
def KFoldTrainDataLoader(train_file, batch_size, k = 5):
    dataloaders_train = []
    dataloaders_val = []
    # extract inputs and targets
    df = pd.read_csv(args.train_file, sep = '\t')
    X = df.iloc[:, :-2].values # SNP features
    y1 = df.iloc [:, -2].values # hap1
    y2 = df.iloc [:, -1].values # hap2
    # encode the the targets
    label_encoder = LabelEncoder()
    gt1 = label_encoder.fit_transform(y1)
    gt2 = label_encoder.fit_transform(y2)
    label_encoder_name_mapping = dict(zip(label_encoder.classes_,
                                             label_encoder.transform(label_encoder.classes_)))
    kf = KFold(n_splits=k, shuffle=False)
    for train_index, test_index in kf5.split(list(range(len(X)))):
        dataset_train = SNPFeature(X.iloc[[i for i in train_index], :], y1.iloc[[i for i in train_index], :], y2.iloc[[i for i in train_index], :], transformer)
        dataloader_train = data.DataLoader(dataset=dataset_train, batch_size=batch_size, num_workers=16, shuffle=True, pin_memory=True)
        dataset_val = SNPFeature(X.iloc[[i for i in test_index], :], y1.iloc[[i for i in test_index], :], y2.iloc[[i for i in test_index], :], transformer)
        dataloader_val = data.DataLoader(dataset=dataset_val, batch_size=batch_size, num_workers=16, shuffle=True, pin_memory=True)
        dataloaders_train.append(dataloader_train)
        dataloaders_val.append(dataloader_val)
    return dataloaders_train, dataloaders_val