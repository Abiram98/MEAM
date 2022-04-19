import os
import pandas as pd
import numpy as np
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset
import torch

class AudioExplorerDataset(Dataset):
    def __init__(self, music_dir, other_dir, transform=None, target_transform=None):
        train_data = np.load(music_dir)
        other_data = np.load(other_dir)
        self.train_data = np.append(train_data,other_data,0 )
        self.train_data = torch.from_numpy(self.train_data)
        self.labels = [1 for x in range(0, 10500)]
        self.labels.extend([0 for x in range(0, 10500)])
        self.labels = torch.from_numpy( np.array(self.labels)).type(torch.FloatTensor)

        self.train_data = torch.reshape(self.train_data, (self.train_data.shape[0],1,self.train_data.shape[1],self.train_data.shape[2]))
        self.labels = torch.reshape(self.labels, (self.labels.shape[0],1))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.train_data[idx], self.labels[idx]

def get_closest_time_step(X, t):
    if (t%X==0):
        return t
    elif (t==0):
        return None
    else:
        return get_closest_time_step(X,t-1)

class AudioExplorerSegmentedDataset(Dataset):
    def __init__(self, music_dir, other_dir,X, transform=None, target_transform=None):
        train_data = np.load(music_dir)
        new_timestep = get_closest_time_step(X,79)
        train_data = train_data[:,:,:new_timestep].reshape(int(new_timestep/X*train_data.shape[0]),30, X)
        train_items = train_data.shape[0]
        #print(sample.shape)
        #exit()
        #train_data = train_data.
        other_data = np.load(other_dir)
        other_data = other_data[:,:,:new_timestep].reshape(int(new_timestep/X*other_data.shape[0]),30, X)
        other_item = other_data.shape[0]
        self.train_data = np.append(train_data,other_data,0 )

        self.train_data = torch.from_numpy(self.train_data)
        print(self.train_data.shape)
        self.labels = [1 for x in range(0, train_items)]
        self.labels.extend([0 for x in range(0, other_item)])
        self.labels = torch.from_numpy( np.array(self.labels)).type(torch.FloatTensor)
        print(self.labels.shape)
        
        self.train_data = torch.reshape(self.train_data, (self.train_data.shape[0],1,self.train_data.shape[1],self.train_data.shape[2]))
        self.labels = torch.reshape(self.labels, (self.labels.shape[0],1))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.train_data[idx], self.labels[idx]

if __name__=="__main__":
    training_data = AudioExplorerSegmentedDataset("data/music_data.npy", "data/other_data.npy", 5)
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
