import torch; print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F


class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 3)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(3, 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(2, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Set the parameters
input_size = 4  # Number of input features
output_size = 1  # Number of output neurons (binary classification)

# Create the neural network
model = SimpleNN(input_size, output_size)

input = torch.ones(size=(1, 4))
output = model(input)
print(output)
target = torch.tensor([[1.0]])

bce_loss = nn.BCEWithLogitsLoss()
loss_1 = bce_loss(output, target)

print(loss_1)

activated_output = F.sigmoid(output)
print(activated_output)

bce_loss = nn.BCELoss()
loss_2 = bce_loss(activated_output, target)

print(loss_2)


