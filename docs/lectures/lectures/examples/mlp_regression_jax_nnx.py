# Takes the baseline version and uses vmap, adds in a learning rate scheduler
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
import optax
import jax_dataloader as jdl
from jax_dataloader.loaders import DataLoaderJAX
from flax import nnx

N = 500  # samples
M = 2
sigma = 0.0001
rngs = nnx.Rngs(42)
theta = random.normal(rngs(), (M,))
X = random.normal(rngs(), (N, M))
Y = X @ theta + sigma * random.normal(rngs(), (N,))  # Adding noise

def residual(model, x, y):
    y_hat = model(x)
    return (y_hat - y) ** 2

def residuals_loss(model, X, Y):
    return jnp.mean(jax.vmap(residual, in_axes=(None, 0, 0))(model, X, Y))

# My MLP
class MLP(nnx.Module):
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


model = MLP(M, 1, 128, rngs=rngs)

n_params = sum(np.prod(x.shape) for x in jax.tree.leaves(nnx.state(model, nnx.Param)))
print(f"Number of parameters: {n_params}")

lr = 0.001
optimizer = nnx.Optimizer(model, optax.sgd(lr))

@nnx.jit
def train_step(model, optimizer, X, Y):
    def loss_fn(model):
        return residuals_loss(model, X, Y)
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss


num_epochs = 2000
batch_size = 512
dataset = jdl.ArrayDataset(X, Y)
train_loader = DataLoaderJAX(dataset, batch_size=batch_size, shuffle=True)
for epoch in range(num_epochs):
    for X_batch, Y_batch in train_loader:
        loss = train_step(model, optimizer, X_batch, Y_batch)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, loss {loss}")


N_test = 200
X_test = random.normal(rngs(), (N_test, M))
Y_test = X_test @ theta + sigma * random.normal(rngs(), (N_test,))  # Adding noise


print(f"loss(model, X, Y) = {residuals_loss(model, X, Y)}, loss(model, X_test, Y_test) = {residuals_loss(model, X_test, Y_test)}")