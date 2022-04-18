import os, sys
from dataloader import AudioExplorerDataset, DataLoader
import torch
import lsh
from sklearn.metrics import precision_score, recall_score, f1_score

music_store = lsh.LSHStore()
non_music_store = lsh.LSHStore()
data = AudioExplorerDataset("../data/music_data.npy", "../data/other_data.npy")
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
print("Store size: " + str(sys.getsizeof(music_store) + sys.getsizeof(non_music_store)) + " bytes")
print()

precision = 0
recall = 0
f1 = 0

for i, data in enumerate(test_dataloader):
    inputs, labels = data
    outs = list()

    for j, batch in enumerate(inputs):
        for k, matrix in enumerate(batch):
            count += 1
            music_result = music_store.search(matrix)

            if (music_result == None):
                continue

            non_music_result = non_music_store.search(matrix)

            if (non_music_result == None):
                continue

            if (music_result[1] >= non_music_result[1]):
                correct += 1
                outs.append(1)

            else:
                outs.append(0)

    precision += precision_score(torch.reshape(labels, (1,)), outs)
    recall += recall_score(torch.reshape(labels, (1,)), outs)
    f1 += f1_score(torch.reshape(labels, (1,)), outs)

print("Accuracy: " + str(float(correct) / count))
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("F1: " + str(f1))

music_store.mem_optimize()
non_music_store.mem_optimize()
print("Memory optimized store size: " + str(sys.getsizeof(music_store) + sys.getsizeof(non_music_store)) + " bytes")
print()

count = 0
correct = 0
precision = 0
recall = 0
f1 = 0

for i, data in enumerate(test_dataloader):
    inputs, labels = data
    outs = list()

    for j, batch in enumerate(inputs):
        for k, matrix in enumerate(batch):
            count += 1
            music_result = music_store.search(matrix)

            if (music_result == None):
                continue

            non_music_result = non_music_store.search(matrix)

            if (non_music_result == None):
                continue

            if (music_result[1] >= non_music_result[1]):
                correct += 1
                outs.append(1)

            else:
                outs.append(0)

    precision += precision_score(torch.reshape(labels, (1,)), outs)
    recall += recall_score(torch.reshape(labels, (1,)), outs)
    f1 += f1_score(torch.reshape(labels, (1,)), outs)

print("Memory optimized accuracy: " + str(float(correct) / count))
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("F1: " + str(f1))
