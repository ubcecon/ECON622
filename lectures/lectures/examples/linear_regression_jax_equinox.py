# Takes the baseline version and uses vmap, adds in a learning rate scheduler
import jax
import jax.numpy as jnp
from jax import grad, jit, value_and_grad, vmap
from jax import random
import optax
import equinox as eqx

N = 500  # samples
M = 2
sigma = 0.001
key = random.PRNGKey(42)
# Pattern: split before using key, replace name "key"
key, *subkey = random.split(key, num=4)
theta = random.normal(subkey[0], (M,))
X = random.normal(subkey[1], (N, M))
Y = X @ theta + sigma * random.normal(subkey[2], (N,))  # Adding noise

# Creates an iterable 
def data_loader(key, X, Y, batch_size):
    num_samples = X.shape[0]
    assert num_samples == Y.shape[0]
    indices = jnp.arange(num_samples)
    indices = random.permutation(key, indices)
    # Loop over batches and yield
    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i:i + batch_size]
        yield X[batch_indices], Y[batch_indices]


# Need to randomize our own theta_0 parameters
key, subkey = random.split(key)
theta_0 = random.normal(subkey, (M,))
print(f"theta_0 = {theta_0}, theta = {theta}")

# Probably a way to use `vmap` or `eqx.filter_vmap` here as well

def residual(model, x, y):
    y_hat = model(x)
    return (y_hat - y) ** 2

def residuals(model, X, Y):
    batched_residuals = vmap(residual, in_axes=(None, 0, 0))
    return jnp.mean(batched_residuals(model, X, Y))

# Alternatively could do something like
def residuals_2(model, X, Y):
    Y_hat = vmap(model)(X).squeeze()
    return jax.numpy.mean((Y - Y_hat) ** 2)

# Hypothesis Class: will start with a linear function, which is randomly initialized
# model is a variable of all parametesr, and supports model(X) calls
key, subkey = random.split(key)
model = eqx.nn.Linear(M, 1, use_bias = False, key = subkey)

# reinitialize
lr = 0.001
optimizer = optax.sgd(lr)
# Needs to remove the non-differentiable parts of the "model" object
opt_state = optimizer.init(eqx.filter(model,eqx.is_inexact_array))

@eqx.filter_jit
def make_step(model, opt_state, X, Y):     
  loss_value, grads = eqx.filter_value_and_grad(residuals)(model, X, Y)
  updates, opt_state = optimizer.update(grads, opt_state, model)
  model = eqx.apply_updates(model, updates)
  return model, opt_state, loss_value

num_epochs = 300
batch_size = 64
key, subkey = random.split(key) # will keep same key for shuffling each epoch
for epoch in range(num_epochs):
    key, subkey = random.split(key) # changing key for shuffling each epoch
    train_loader = data_loader(subkey, X, Y, batch_size)
    for X_batch, Y_batch in train_loader:
        model, opt_state, train_loss = make_step(model, opt_state, X_batch, Y_batch)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch},||theta - theta_hat|| = {jnp.linalg.norm(theta - model.weight)}")

print(f"||theta - theta_hat|| = {jnp.linalg.norm(theta - model.weight)}")


## Custom equinox type, like in pytorch
class MyLinear(eqx.Module):
    weight: jax.Array

    def __init__(self, in_size, out_size, key):
        self.weight = jax.random.normal(key, (out_size, in_size))

    # Equivalent to Pytorch's forward
    def __call__(self, x):
        return self.weight @ x

model = MyLinear(M, 1, key = subkey)
opt_state = optimizer.init(eqx.filter(model,eqx.is_inexact_array))

num_epochs = 300
batch_size = 64
key, subkey = random.split(key) # will keep same key for shuffling each epoch
for epoch in range(num_epochs):
    key, subkey = random.split(key) # changing key for shuffling each epoch
    train_loader = data_loader(subkey, X, Y, batch_size)
    for X_batch, Y_batch in train_loader:
        model, opt_state, train_loss = make_step(model, opt_state, X_batch, Y_batch)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch},||theta - theta_hat|| = {jnp.linalg.norm(theta - model.weight)}")

print(f"||theta - theta_hat|| = {jnp.linalg.norm(theta - model.weight)}")