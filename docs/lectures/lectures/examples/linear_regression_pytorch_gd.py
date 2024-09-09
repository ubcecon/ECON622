# Example solving linear regression using "full batch" gradient descent
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
import numpy as np

# Model parameters and simulate data
N = 500  # samples
M = 2
sigma = 0.001
theta = torch.randn(M)
X = torch.randn(N, M)
Y = X @ theta + sigma * torch.randn(N)  # Adding noise


# Residuals/loss
def residuals(model, X, Y):  # batches or full data
    Y_hat = model(X).squeeze()
    return ((Y_hat - Y) ** 2).mean()


## ESTABLISH HYPOTHESIS CLASS
# The "model" for our setup a linear function with an affine "bias" term
model = nn.Linear(M, 1, bias=False)  # random initialization

## CREATE OPTIMIZER, CONNECT TO MODEL PARAMETERS
# Optimizers hold on to the parameters of the underlying "model"
# All first-order optimizers in JAX and pytorch have this pattern
lr = 0.01  # "learning rate"
num_epochs = 500  # = steps here
print_every = 50
optimizer = optim.SGD(
    model.parameters(), lr=lr
)  # it is GD if using all data each gradient

## "TRAINING LOOP" RUNS OPTIMIZER
for epoch in range(num_epochs):
    optimizer.zero_grad()  # reset gradients for AD since implemented by +=
    loss = residuals(model, X, Y)  # primal
    loss.backward()  # backprop/reverse-mode AD
    optimizer.step()  # Update the optimizers internal parameters

    if epoch % print_every == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Training Loss: {loss.item():.8g}")

## CHECK GENERALIZATION.  EASY HERE GIVEN SIMULATED DATA AND MODEL CLASS
print(f"||theta - theta_hat|| = {torch.norm(theta - model.weight.squeeze())}")
