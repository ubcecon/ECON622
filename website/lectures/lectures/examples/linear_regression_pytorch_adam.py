# Example solving linear regression adding in:
# - train/val/test split
# - early stopping
# - dataloaders
# - Adam instead of SGD
# - tqdm progress bar
# - learning rate scheduler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from tqdm import tqdm

# Model parameters and simulate data
N = 1000  # samples
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
model = nn.Linear(M, 1, bias=False)  # random initialization


## CREATE OPTIMIZER, CONNECT TO MODEL PARAMETERS
lr = 0.001  # "learning rate"
num_epochs = 500  # = steps here
train_prop = 0.7
val_prop = 0.15
batch_size = 16
num_batches = int(np.ceil(N / batch_size))
early_stopping_val_loss = 1e-7
update_lr_every = 20
lr_scheduler_gamma = 0.95

# Using Adam instead of SGD
optimizer = optim.Adam(
    model.parameters(), lr=lr
)  # it is GD if using all data each gradient

# The learning rate scheduler updates the optimizer's learning rate with every step()
scheduler = lr_scheduler.StepLR(
    optimizer, step_size=update_lr_every, gamma=lr_scheduler_gamma
)

# Split the data into training, validation, and test sets
train_size = int(train_prop * N)
val_size = int(val_prop * N)
test_size = N - train_size - val_size
train_data, val_data, test_data = random_split(
    TensorDataset(X, Y), [train_size, val_size, test_size]
)

# Dataloaders will provide batches of the data
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# tqdm is a progress bar which integrates well with loops
pbar = tqdm(range(num_epochs), desc="Epochs")
lowest_val_loss = np.inf  # will track best fit using validation, NOT training loss
for epoch in pbar:  # or "epoch in in range(num_epochs)" if not using tqdm
    model.train()  # Ensure AD is turned on
    train_loss = 0.0  # reset the train_loss, which we accumulate over batches
    for batch_idx, (X_batch, Y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        loss = residuals(model, X_batch, Y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()  # accumulate loss over batches
    train_loss /= num_batches  # average over batches

    # Validation logic.  Could do only sometimes, but here we do it every epoch
    model.eval()  # Turn off AD
    val_loss = 0
    with torch.no_grad():
        for X_val, Y_val in val_loader:
            val_loss += residuals(model, X_val, Y_val)
        val_loss /= num_batches

        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            best_model = model.state_dict()  # store it to load later
    scheduler.step()
    pbar.set_postfix(
        {
            "train_loss": f"{train_loss:.4g}",
            "val_loss": f"{val_loss:.4g}",
            "lr": f"{scheduler.get_last_lr()[0]:.4g}",
        }
    )
    if val_loss < early_stopping_val_loss:
        print(
            f"Early stopping at epoch {epoch} with train_loss = {train_loss:.8g} and val_loss = {val_loss:.8g}"
        )
        break

# Load the best model based on the val_loss
model.load_state_dict(best_model)

# Evaluate on test set
model.eval()
test_loss = 0
with torch.no_grad():
    for X_test, Y_test in test_loader:
        test_loss += residuals(model, X_test, Y_test)
    test_loss /= num_batches

print(
    f"Test Loss: {test_loss:.4g}, ||theta - theta_hat|| = {torch.norm(theta - model.state_dict()['weight'])}"
)
