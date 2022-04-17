import os, sys
from dataloader import AudioExplorerDataset, DataLoader
import torch
import lsh

music_store = lsh.LSHStore()
non_music_store = lsh.LSHStore()
data = AudioExplorerDataset("/data/music_data.npy", "/data/other_data.npy")
batch_size = 64
test_split = 0.2
valid_split = 0.2
data_size = len(data)
test_size = int(test_split * data_size)
valid_size = int(valid_split * data_size)
train_size = data_size - test_size - valid_size
training_data, test_data, valid_data = torch.utils.data.random_split(data, [train_size, test_size, valid_size])
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle = True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle = True)

count = 0
correct = 0
MUSIC_THRES = 0.9

# Inserting data
print("Inserting data...")

for i, data in enumerate(train_dataloader):
    inputs, labels = data

    for j, batch in enumerate(inputs):
        for k, matrix in enumerate(batch):
            if (labels[j][k]):
                music_store.add(lsh.Element(matrix, labels[j][k]))

            else:
                non_music_store.add(lsh.Element(matrix, labels[j][k]))

print("Insertion done")

for i, data in enumerate(test_dataloader):
    inputs, labels = data

    for j, batch in enumerate(inputs):
        for k, matrix in enumerate(batch):
            count += 1
            music_result = music_store.search(matrix)[1]
            non_music_result = non_music_store.search(matrix)[1]

            if (music_result >= non_music_result):
                correct += 1

print("Accuracy: " + str(float(correct) / count))
