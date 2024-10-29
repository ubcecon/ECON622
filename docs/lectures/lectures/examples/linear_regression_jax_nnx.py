# Takes the baseline version and uses vmap, adds in a learning rate scheduler
import jax
import jax.numpy as jnp
from jax import random
import optax
import jax_dataloader as jdl
from jax_dataloader.loaders import DataLoaderJAX
from flax import nnx

N = 500  # samples
M = 2
sigma = 0.001
rngs = nnx.Rngs(42)
theta = random.normal(rngs(), (M,))
X = random.normal(rngs(), (N, M))
Y = X @ theta + sigma * random.normal(rngs(), (N,))  # Adding noise

def residual(model, x, y):
    y_hat = model(x)
    return (y_hat - y) ** 2

def residuals_loss(model, X, Y):
    return jnp.mean(jax.vmap(residual, in_axes=(None, 0, 0))(model, X, Y))

model = nnx.Linear(M, 1, use_bias=False, rngs=rngs)

lr = 0.001
optimizer = nnx.Optimizer(model, optax.sgd(lr))

@nnx.jit
def train_step(model, optimizer, X, Y):
    def loss_fn(model):
        return residuals_loss(model, X, Y)
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss


num_epochs = 1000
batch_size = 512
dataset = jdl.ArrayDataset(X, Y)
train_loader = DataLoaderJAX(dataset, batch_size=batch_size, shuffle=True)
for epoch in range(num_epochs):
    for X_batch, Y_batch in train_loader:
        loss = train_step(model, optimizer, X_batch, Y_batch)

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch},||theta - theta_hat|| = {jnp.linalg.norm(theta - jnp.squeeze(model.kernel.value))}"
        )

print(f"||theta - theta_hat|| = {jnp.linalg.norm(theta - jnp.squeeze(model.kernel.value))}")