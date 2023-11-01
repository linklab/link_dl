import torch
from torch import nn

# Define the ReLU activation function
relu = nn.ReLU()

# Initialize a weight that will cause the "Dying ReLU" problem
weights = torch.tensor([-2.0, 0.5])  # Negative weights will lead to the problem

# Define an input example
input_data = torch.tensor([1.0, 1.0])

# Calculate the output of the ReLU neuron
output = relu(input_data * weights)

print("Output of the ReLU neuron:", output)
