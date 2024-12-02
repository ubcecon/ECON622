# Takes the baseline version and uses vmap, adds in a learning rate scheduler
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
import optax
import jax_dataloader as jdl
from jax_dataloader.loaders import DataLoaderJAX
from flax import nnx
from typing import List, Optional, Callable
import wandb
import jsonargparse


# My MLP
class MyMLP(nnx.Module):
    def __init__(self, din: int, dout: int, width: int, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(din, width, rngs=rngs)
        self.linear2 = nnx.Linear(width, width, rngs=rngs)
        self.linear3 = nnx.Linear(width, dout, rngs=rngs)

    def __call__(self, x: jax.Array):
        x = self.linear1(x)
        x = nnx.relu(x)
        x = self.linear2(x)
        x = nnx.relu(x)
        x = self.linear3(x)
        return x


def fit_model(
    N: int = 500,
    M: int = 2,
    sigma: float = 0.0001,
    width: int = 128,
    lr: float = 0.001,
    num_epochs: int = 2000,
    batch_size: int = 512,
    seed: int = 42,
    wandb_project: str = "econ622_examples",
    wandb_mode: str = "offline",  # "online", "disabled
):
    if not wandb_mode == "disabled":
        wandb.init(project="survey", mode=wandb_mode)
    rngs = nnx.Rngs(seed)

    theta = random.normal(rngs(), (M,))
    X = random.normal(rngs(), (N, M))
    Y = X @ theta + sigma * random.normal(rngs(), (N,))  # Adding noise

    def residual(model, x, y):
        y_hat = model(x)
        return (y_hat - y) ** 2

    def residuals_loss(model, X, Y):
        return jnp.mean(jax.vmap(residual, in_axes=(None, 0, 0))(model, X, Y))

    model = MyMLP(M, 1, width, rngs=rngs)

    n_params = sum(
        np.prod(x.shape) for x in jax.tree.leaves(nnx.state(model, nnx.Param))
    )
    print(f"Number of parameters: {n_params}")

    optimizer = nnx.Optimizer(model, optax.sgd(lr))

    @nnx.jit
    def train_step(model, optimizer, X, Y):
        def loss_fn(model):
            return residuals_loss(model, X, Y)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)
        return loss

    dataset = jdl.ArrayDataset(X, Y)
    train_loader = DataLoaderJAX(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(num_epochs):
        for X_batch, Y_batch in train_loader:
            loss = train_step(model, optimizer, X_batch, Y_batch)

        if not (wandb_mode == "disabled"):
            wandb.log({"epoch": epoch, "train_loss": loss, "lr": lr})
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, loss {loss}")

    N_test = 200
    X_test = random.normal(rngs(), (N_test, M))
    Y_test = X_test @ theta + sigma * random.normal(rngs(), (N_test,))  # Adding noise

    loss_data = residuals_loss(model, X, Y)
    loss_test = residuals_loss(model, X_test, Y_test)
    print(f"loss(model, X, Y) = {loss_data}, loss(model, X_test, Y_test) = {loss_test}")
    if not (wandb_mode == "disabled"):
        wandb.log(
            {"train_loss": loss_data, "test_loss": loss_test, "num_params": n_params}
        )

    if not wandb_mode == "disabled":
        wandb.finish()


if __name__ == "__main__":
    jsonargparse.CLI(fit_model)
    # Swap with this line to run debugger with different parameters
    # jsonargparse.CLI(fit_model, args=["--num_epochs", "200", "--wandb_mode", "online"])
