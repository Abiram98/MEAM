import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import torch.nn.utils.prune as prune

from dataloader import AudioExplorerDataset,DataLoader
import torch.optim as optim
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from models.default_model import CNN

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

batch_size = 64
test_split = 0.2
valid_split = 0.2

with torch.no_grad():
    t = torch.autograd.Variable(torch.Tensor([0.5])).to(device)

data = AudioExplorerDataset("data/music_data.npy", "data/other_data.npy")

data_size = len(data)
test_size = int(test_split*data_size)
valid_size = int(valid_split*data_size)
train_size = data_size - test_size - valid_size

training_data,test_data, valid_data = torch.utils.data.random_split(data,[train_size,test_size, valid_size])

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
cnn = CNN()
cnn.to(device)
summary(cnn, input_size=(batch_size, 1, 30,79))


import warnings
warnings.simplefilter("ignore")

criterion = nn.BCELoss()
optimizer = optim.SGD(cnn.parameters(),lr=0.001, momentum=0.9)
epochs = 100
best_val_f1 = 0
for epoch in range(epochs):
    running_loss = 0.0
    precisions = 0
    recalls = 0
    f1s = 0
    
    for i, data in enumerate(train_dataloader):    
        inputs, labels = data
        #one = inputs[1,1,:,:]
        inputs,labels = inputs.to(device), labels.to(device)
        
        
        optimizer.zero_grad()
        outputs = cnn(inputs)
        outs = torch.reshape((outputs > t).float(),(-1,))

        precisions += precision_score(torch.reshape(labels.cpu(),(-1,)),outs.cpu())
        recalls += recall_score(torch.reshape(labels.cpu(),(-1,)),outs.cpu())
        f1s += f1_score(torch.reshape(labels.cpu(),(-1,)),outs.cpu())

        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 20 ==19:
            print(f'[{epoch+1}, {i+1:5d}] loss: {running_loss/20:.3f} precision: {precisions/i:.3f} recall {recalls/i:.3f} f1 score {f1s/i:.3f}')
            running_loss = 0.0
    precisions_v = 0
    recalls_v = 0
    f1s_v = 0
    with torch.no_grad():
        for i, data in enumerate(valid_dataloader):
            input_valid,labels_valid = data
            input_valid,labels_valid = input_valid.to(device), labels_valid.to(device)
            out_v = torch.reshape((cnn(input_valid) > t).float(),(-1,))
            labels_valid = torch.reshape(labels_valid.float(),(-1,))
            
            

            precisions_v += precision_score(labels_valid.cpu(),out_v.cpu())
            recalls_v += recall_score(labels_valid.cpu(),out_v.cpu())
            f1s_v += f1_score(labels_valid.cpu(),out_v.cpu())
            if i == len(valid_dataloader)-1:
                print(f'[{epoch+1}] val precision: {precisions_v/i:.3f} val recall {recalls_v/i:.3f} val f1 score {f1s_v/i:.3f}')
                if (f1s_v/i) > best_val_f1:
                    print('Achieved a better model')
                    torch.save(cnn.state_dict(), f"{epoch}cnn_weights.pth")
                    best_val_f1 = (f1s_v/i)


print('Finished Training')

with torch.no_grad():
    t = torch.autograd.Variable(torch.Tensor([0.5]))
    precisions_t = 0
    recalls_t = 0
    f1s_t = 0
    for data in test_dataloader:
        inputs,labels = data
        cnn = cnn.cpu()
        outputs = cnn(inputs)
        outs = torch.reshape((outputs > t).float(),(-1,))
        labels = torch.reshape(labels,(-1,))
        precisions_t += precision_score(labels,outs)
        recalls_t += recall_score(labels,outs)
        f1s_t += f1_score(labels,outs)
    
    runs = len(test_dataloader)
    print(f"Test data resuls:\nPrecision {precisions_t/runs:.2f}, Recall {recalls_t/runs:.2f}, F1-score {f1s_t/runs:.2f}")

# print('Performing weight pruning')
# model = cnn
# parameters_to_prune = (
    # (model.conv1, 'weight'),
    # (model.conv2, 'weight'),
    # (model.fc1, 'weight'),
    # (model.fc2, 'weight'),
    # (model.fc3, 'weight'),
# )
# prune.global_unstructured(
    # parameters_to_prune,
    # pruning_method=prune.L1Unstructured,
    # amount=0.3,
# )

# print('Performing quantization')
# model_int8 = torch.quantization.quantize_dynamic(
    # model,
    # {torch.nn.Linear}, 
    # dtype=torch.qint8)
