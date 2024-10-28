# Takes the baseline version and uses vmap, adds in a learning rate scheduler
import jax
import jax.numpy as jnp
from jax import random
import optax
import equinox as eqx
import jax_dataloader as jdl
from jax_dataloader.loaders import DataLoaderJAX


# LLS loss function with vmap
def residual(model, x, y):
    y_hat = model(x)
    return (y_hat - y) ** 2

def residuals(model, X, Y):
    batched_residuals = jax.vmap(residual, in_axes=(None, 0, 0))
    return jnp.mean(batched_residuals(model, X, Y))


# SWITCH OPTIMIZERS HERE!!!!
# reinitialize
#optimizer = optax.sgd(0.001)
optimizer = optax.lbfgs()


N = 500  # samples
M = 2
sigma = 0.0001
key = random.PRNGKey(42)
key, *subkey = random.split(key, num=4)
theta = random.normal(subkey[0], (M,))
X = random.normal(subkey[1], (N, M))
Y = X @ theta + sigma * random.normal(subkey[2], (N,))  # Adding noise


# Hypothesis Class: will start with a linear function, which is randomly initialized
# model is a variable of all parametesr, and supports model(X) calls
key, subkey = random.split(key)
model = eqx.nn.Linear(M, 1, use_bias = False, key = subkey)

# Needs to remove the non-differentiable parts of the "model" object
opt_state = optimizer.init(eqx.filter(model,eqx.is_inexact_array))

@eqx.filter_jit
def make_step(model, opt_state, X, Y):     
  def step_residuals(model):
    return residuals(model, X, Y)
  loss_value, grads = eqx.filter_value_and_grad(step_residuals)(model)
  updates, opt_state = optimizer.update(grads, opt_state, model, value = loss_value, grad = grads, value_fn = step_residuals)
  model = eqx.apply_updates(model, updates)
  return model, opt_state, loss_value

num_epochs = 20
batch_size = 1024#64
dataset = jdl.ArrayDataset(X,Y)
train_loader = DataLoaderJAX(dataset, batch_size = batch_size, shuffle = True)
for epoch in range(num_epochs):
    for X_batch, Y_batch in train_loader:
        model, opt_state, train_loss = make_step(model, opt_state, X_batch, Y_batch)
    
    if epoch % 2 == 0:
        print(f"Epoch {epoch},||theta - theta_hat|| = {jnp.linalg.norm(theta - model.weight)}")

print(f"||theta - theta_hat|| = {jnp.linalg.norm(theta - model.weight)}")


# ## Custom equinox type, like in pytorch
class MyLinear(eqx.Module):
    weight: jax.Array

    def __init__(self, in_size, out_size, key):
        self.weight = jax.random.normal(key, (out_size, in_size))

    # Equivalent to Pytorch's forward
    def __call__(self, x):
        return self.weight @ x

model = MyLinear(M, 1, key = subkey)
opt_state = optimizer.init(eqx.filter(model,eqx.is_inexact_array))

for epoch in range(num_epochs):
    for X_batch, Y_batch in train_loader:
        model, opt_state, train_loss = make_step(model, opt_state, X_batch, Y_batch)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch},||theta - theta_hat|| = {jnp.linalg.norm(theta - model.weight)}")

print(f"||theta - theta_hat|| = {jnp.linalg.norm(theta - model.weight)}")


model = eqx.nn.MLP(M, 1, width_size=128, depth=3, key = subkey)
opt_state = optimizer.init(eqx.filter(model,eqx.is_inexact_array))

for epoch in range(num_epochs):
    for X_batch, Y_batch in train_loader:
        model, opt_state, train_loss = make_step(model, opt_state, X_batch, Y_batch)
    
    if epoch % 100 == 0:
         print(f"Epoch {epoch},train_loss={train_loss}")

# print(f"||theta - theta_hat|| = {jnp.linalg.norm(theta - model.weight)}")