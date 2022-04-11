import os
import pandas as pd
import numpy as np
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset

class CustomImageDataset(Dataset):
    def __init__(self, music_dir, other_dir, transform=None, target_transform=None):
        self.train_data = np.load(music_dir)
        self.train_data.append(np.load(other_dir))
        self.labels = [1 for x in range(1, 10500)]
        self.labels.append([0 for x in range(1, 10500)])
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        return self.train_dat[idx], self.labels[idx]

if __name__=="__main__":
    training_data = CustomImageDataset("data/music_data.npy", "data/other_data.npy")
    #train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)