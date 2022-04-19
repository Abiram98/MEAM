import os, sys
from dataloader import AudioExplorerDataset, DataLoader
import torch
import lsh
from sklearn.metrics import precision_score, recall_score, f1_score
import data_util as du
import numpy as np
import pickle

music_store = lsh.LSHStore()
non_music_store = lsh.LSHStore()
train_data_music = du.to_list("music_data.npy")
train_data_non_music = du.to_list("other_data.npy")
test_data = list()

for matrix in train_data_music[int(len(train_data_music) * 0.8):-1]:
    test_data.append([matrix, True])

for matrix in train_data_non_music[int(len(train_data_non_music) * 0.8):-1]:
    test_data.append([matrix, False])

train_data_music = train_data_music[0:int(0.8 * len(train_data_music))]
train_data_non_music = train_data_non_music[0:int(0.8 * len(train_data_non_music))]

print("Music size: " + str(len(train_data_music)))
print("Non-music size: " + str(len(train_data_non_music)))
print("Test size: " + str(len(test_data)))

count = 0
correct = 0
MUSIC_THRES = 0.9
#total_length_train = len(train_data_music) + len(train_data_non_music)
total_length_train = 100
total_length_test = len(test_data)

# Inserting data
print("\nInserting data...")

for i, matrix in enumerate(train_data_music):
    if (i == total_length_train / 2):
        break

    print("Inserting: " + str(i + 1) + "/" + str(total_length_train), end = "\r")
    music_store.add(lsh.Element(matrix, str(i)))

for i, matrix in enumerate(train_data_non_music):
    if (i == total_length_train / 2):
        break

    print("Inserting: " + str(int(i + (total_length_train / 2) + 1)) + "/" + str(total_length_train), end = "\r")
    non_music_store.add(lsh.Element(matrix, str(i)))

with open("full_music.db", "wb") as file:
    pickle.dump(music_store, file)

with open("full_non_music.db", "wb") as file:
    pickle.dump(non_music_store, file)

print()
print("Insertion done")
print("Music store element count: " + str(sum(music_store.bucket_sizes())))
print("Non-music store element count: " + str(sum(non_music_store.bucket_sizes())))
print()
print("Max bucket size: " + str(max(max(music_store.bucket_sizes()), max(non_music_store.bucket_sizes()))))
print("Min bucket size: " + str(min(min(music_store.bucket_sizes()), min(non_music_store.bucket_sizes()))))
print()

outs = list()
labels = list()

for i, ground_truth in enumerate(test_data):
    print("Testing progress: " + str((float(i + 1) / total_length_test) * 100) + "%", end = "\r")

    count += 1
    music_result = music_store.search(ground_truth[0])
    non_music_result = non_music_store.search(ground_truth[0])
    labels.append(ground_truth[1])

    if (music_result == None or non_music_result == None):
        continue

    elif (music_result[1] >= non_music_result[1]):
        correct += 1
        outs.append(1)

    else:
        outs.append(0)

print()
print()
print("Accuracy: " + str(float(correct) / count))
print("Precision: " + str(precision_score(labels, outs)))
print("Recall: " + str(recall_score(labels, outs)))
print("F1: " + str(f1_score(labels, outs)))

music_store.mem_optimize()
non_music_store.mem_optimize()
print("Max bucket size: " + str(max(max(music_store.bucket_sizes()), max(non_music_store.bucket_sizes()))))
print("Min bucket size: " + str(min(min(music_store.bucket_sizes()), min(non_music_store.bucket_sizes()))))
print()

with open("appr_music.db", "wb") as file:
    pickle.dump(music_store, file)

with open("appr_non_music.db", "wb") as file:
    pickle.dump(non_music_store, file)

count = 0
correct = 0
labels = list()
outs = list()

for i, ground_truth in enumerate(test_data):
    print("Testing progress: " + str((float(i + 1) / total_length_test) * 100) + "%", end = "\r")

    count += 1
    music_result = music_store.search(ground_truth[0])
    non_music_result = non_music_store.search(ground_truth[0])
    labels.append(ground_truth[1])

    if (music_result == None or non_music_result == None):
        continue

    elif (music_result[1] >= non_music_result[1]):
        correct += 1
        outs.append(1)

    else:
        outs.append(0)

print("Memory optimized accuracy: " + str(float(correct) / count))
print("Precision: " + str(precision_score(labels, outs)))
print("Recall: " + str(recall_score(labels, outs)))
print("F1-score: " + str(f1_score(labels, outs)))
