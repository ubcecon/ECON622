# Takes the baseline version and uses vmap, adds in a learning rate scheduler
import jax
import jax.numpy as jnp
from jax import grad, jit, value_and_grad, vmap
from jax import random
import optax

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

# Model: \hat{f}_{\hat{\theta}}(x) = \hat{\theta} \cdot x
def predict(theta, X):
    return jnp.matmul(X, theta) #or jnp.dot(X, theta)

def residual(theta, x, y):
    y_hat = predict(theta, x)
    return (y_hat - y) ** 2

def residuals(theta, X, Y):
    batched_residuals = vmap(residual, in_axes=(None, 0, 0))
    return jnp.mean(batched_residuals(theta, X, Y))


# reinitialize
lr = 0.001
batch_size = 16
scheduler = optax.exponential_decay(
    init_value=lr, 
    transition_steps=1000,
    decay_rate=0.95)
optimizer = optax.sgd(learning_rate=scheduler)
opt_state = optimizer.init(theta_0)


@jax.jit
def make_step(params, opt_state, X, Y):     
  loss_value, grads = jax.value_and_grad(residuals)(params, X, Y)
  updates, opt_state = optimizer.update(grads, opt_state, params)
  params = optax.apply_updates(params, updates)
  return params, opt_state, loss_value

num_epochs = 200  # = steps here
params = theta_0 # initial condition, will update
key, subkey = random.split(key) # will keep same key for shuffling each epoch
for epoch in range(num_epochs):
    key, subkey = random.split(key) # changing key for shuffling each epoch
    train_loader = data_loader(subkey, X, Y, batch_size)
    for X_batch, Y_batch in train_loader:
        params, opt_state, train_loss = make_step(params, opt_state, X_batch, Y_batch)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch},||theta - theta_hat|| = {jnp.linalg.norm(theta - params)}")

print(f"||theta - theta_hat|| = {jnp.linalg.norm(theta - params)}")
