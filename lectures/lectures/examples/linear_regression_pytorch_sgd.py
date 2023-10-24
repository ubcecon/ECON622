# Example solving linear regression using SGD with a full batch
# And shows how we could implement a more flexible model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from tqdm import tqdm

# Model parameters and simulate data
N = 500  # samples
M = 2
sigma = 0.001
theta = torch.randn(M)
X = torch.randn(N, M)
Y = X @ theta + sigma * torch.randn(N)  # Adding noise

# Prepare it for batching
dataset = TensorDataset(X, Y)    
batch_size = 16
# train_loader is now an iterator that works for loops, etc.
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Residuals/loss
def residuals(model, X, Y):  # batches or full data
    Y_hat = model(X).squeeze()
    return ((Y_hat - Y) ** 2).mean()

## ESTABLISH HYPOTHESIS CLASS
# The "model" for our setup a linear function with an affine "bias" term
model = nn.Linear(M, 1, bias=False)  # random initialization

## CREATE OPTIMIZER, CONNECT TO MODEL PARAMETERS
lr = 0.001  # "learning rate"
num_epochs = 1000  # = steps here
optimizer = optim.SGD( # now actually SGD if batch_size < N
    model.parameters(), lr=lr
)

## "TRAINING LOOP" RUNS OPTIMIZER
 # Using utility to display epoch numbers, could replace with the simple "for epoch"
# for epoch in range(num_epochs):
for epoch in tqdm(range(num_epochs), desc="Epochs"):
    # this will loop until the end of the data, then go to the next epoch
    for X_batch, Y_batch in train_loader:
        optimizer.zero_grad() 
        loss = residuals(model, X_batch, Y_batch)  # primal
        loss.backward()  # backprop/reverse-mode AD
        optimizer.step()  # Update the optimizers internal parameters

## CHECK GENERALIZATION.  EASY HERE GIVEN SIMULATED DATA AND MODEL CLASS
print(f"||theta - theta_hat|| = {torch.norm(theta - model.weight.squeeze())}")
