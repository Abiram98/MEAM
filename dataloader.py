import os
import pandas as pd
import numpy as np
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset

class AudioExplorerDataset(Dataset):
    def __init__(self, music_dir, other_dir, transform=None, target_transform=None):
        train_data = np.load(music_dir)
        other_data = np.load(other_dir)
        self.train_data = np.append(train_data,other_data,0 )
        self.labels = [1 for x in range(0, 10500)]
        self.labels.extend([0 for x in range(0, 10500)])
        self.labels = np.array(self.labels)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.train_data[idx], self.labels[idx]

if __name__=="__main__":
    training_data = AudioExplorerDataset("data/music_data.npy", "data/other_data.npy")
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
