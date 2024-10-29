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

# From https://github.com/google/flax/blob/main/flax/nnx/training/optimizer.py
# HACK! To be replaced when supported by NNX.
def update(self, grads, value = None, value_fn = None):
    gdef, state = nnx.split(self.model, self.wrt)

    def value_fn_wrapped(state):
        model = nnx.merge(gdef, state)
        return value_fn(model)

    updates, new_opt_state = self.tx.update(grads, self.opt_state, state, grad = grads, value = value, value_fn = value_fn_wrapped)


    new_params = optax.apply_updates(state, updates)
    assert isinstance(new_params, nnx.State)

    self.step.value += 1
    nnx.update(self.model, new_params)
    self.opt_state = new_opt_state

# Advantage: a little faster since split, and can use LBFGS due to the hack
lr = 0.001
optimizer = nnx.Optimizer(model,
                          optax.lbfgs(),
                        #optax.sgd(lr),
                          )

@nnx.jit
def train_step(model, optimizer, X, Y):
    def loss_fn(model):
        return residuals_loss(model, X, Y)
    loss, grads =  nnx.value_and_grad(loss_fn)(model)
    # optimizer.update(grads)
    update(optimizer, grads, value = loss, value_fn = loss_fn)
    return loss

num_epochs = 20
batch_size = 1024
dataset = jdl.ArrayDataset(X, Y)
train_loader = DataLoaderJAX(dataset, batch_size=batch_size, shuffle=True)
for epoch in range(num_epochs):
    for X_batch, Y_batch in train_loader:
        loss = train_step(model, optimizer, X_batch, Y_batch)

    if epoch % 2 == 0:
        print(
            f"Epoch {epoch},||theta - theta_hat|| = {jnp.linalg.norm(theta - jnp.squeeze(model.kernel.value))}"
        )

print(f"||theta - theta_hat|| = {jnp.linalg.norm(theta - jnp.squeeze(model.kernel.value))}")
