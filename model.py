from torch import nn
import torch.nn.functional as F

class Network(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 250)
        self.fc3 = nn.Linear(250, 75)
        self.fc4 = nn.Linear(75, 10)

    def forward(self, x):
        x = x.view(x.shape[0], 28 * 28)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = nn.LogSoftmax(x)
        return x