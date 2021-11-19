from torchvision import transforms

import model
import data_handler
import torch as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd


train_csv = pd.read_csv('./data/fashion-mnist_train.csv')
test_csv = pd.read_csv('./data/fashion-mnist_test.csv')

train_set = data_handler.FashionDataset(train_csv, transform=transforms.Compose([transforms.ToTensor()]))
test_set = data_handler.FashionDataset(test_csv, transform=transforms.Compose([transforms.ToTensor()]))

trainloader = DataLoader(train_set, batch_size=100)
testloader = DataLoader(train_set, batch_size=100)
model = model.Network()

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 3
print_every = 1000

for e in range(epochs):
    running_loss = 0
    print(f"Epoch: {e + 1}/{epochs}")

    for i, (images, labels) in enumerate(iter(trainloader)):

        # images.resize_(images.size()[0], 784)
        optimizer.zero_grad()

        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % print_every == 0:
            print(f"\tIteration: {i}\t Loss: {running_loss / print_every:.4f}")
            running_loss = 0