import os
import pandas as pd 
from torch.utils.data import Dataset

class MnistDataset(Dataset):
    def __init__(self, path, transform = None):
        self.data = pd.read_csv(path)
        self.transform = transform
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        target = self.data.iloc[idx, 0]
        image = self.data.iloc[idx, 1:].values
        image = image.reshape((28, 28))
        if self.transform:
            image = self.transform(image)
        
        return image, target



