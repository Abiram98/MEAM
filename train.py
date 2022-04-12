import torch.nn as nn
import torch.nn.functional as F

from dataloader import AudioExplorerDataset,DataLoader
import torch.optim as optim

batch_size = 64
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,6, kernel_size=3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(1360,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,1)
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

import torch
training_data = AudioExplorerDataset("data/music_data.npy", "data/other_data.npy")
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
cnn = CNN()


criterion = nn.BCELoss()
optimizer = optim.SGD(cnn.parameters(),lr=0.001, momentum=0.9)
epochs = 10

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(train_dataloader):
        inputs, labels = data
        
        inputs = torch.reshape(inputs, (inputs.shape[0],1,inputs.shape[1],inputs.shape[2]))
        labels = torch.reshape(labels, (inputs.shape[0],1))
        optimizer.zero_grad()
        outputs = cnn(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 20 ==19:
            print(f'[{epoch+1}, {i+1:5d}] loss: {running_loss/20:.3f}')
            running_loss = 0.0
print('Finished Training')

