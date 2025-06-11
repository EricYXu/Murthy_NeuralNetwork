"""
Script to gauge the effects scalar multiplicative weight perturbation has on model accuracy. 
"""

# Imports
import network
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace 
import numpy as np
import matplotlib.pyplot as plt

# Iterate over different scalars
lower_scalar = -10
upper_scalar = 11
list_of_scalars = np.arange(lower_scalar,upper_scalar,1).tolist()
unperturbed_accuracy_vec = torch.ones(len(list_of_scalars))
perturbed_accuracy_vec = torch.zeros(len(list_of_scalars))

# Setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") 
m = 100 
n = 50 
seed = 1
network.set_random_mode(True, seed)
init_parameters = SimpleNamespace()
init_parameters.m = m 
init_parameters.n = n 
init_parameters.seed = seed
init_parameters.W = torch.randn(n,m).to(device)
store_image = True

# Odorant-to-measurement perturbation parameters
is_normal = False
is_multiplicative = True
is_correlated = True
stddev = 0

# Creates the network and establishes parameters
enose_network = network.enose(init_parameters)
num_data=10000
training_frac=0.8
is_balanced=True
threshold=0.1
x_hat = threshold
sparsity_index = None
num_training = int(num_data*training_frac)
bounds = (0,1)

# Creates dense dataset
dataset = network.generate_dataset(m, n, device, num_data, training_frac, is_balanced, threshold, bounds, sparsity_index)
train_conc = dataset.train_conc
train_labels = dataset.train_labels
test_conc = dataset.test_conc
test_labels = dataset.test_labels
test_batch_size = 20
test_iteration_count = 100

# Trains the network
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(enose_network.parameters(), lr = 0.001)
lr = 0.001
batch_size = 10
steps = 10000
loss_vec = torch.zeros(steps)
test_interval = 50 
print_interval = 10000 
test_batch_size = 20 
accuracy_vec = torch.zeros(steps // test_interval)
loss_vec, accuracy_vec = network.train_model(
    enose_network,
    criterion,
    optimizer,
    lr,
    train_conc,
    train_labels,
    test_conc,
    test_labels,
    num_data,
    num_training,
    steps,
    batch_size,
    test_interval,
    print_interval,
    test_batch_size,
    test_iteration_count,
    device
)

# Get mean accuracy before perturbation
num_step = len(accuracy_vec)
mean_accuracy = torch.mean(accuracy_vec[num_step//2:-1])
unperturbed_accuracy_vec = mean_accuracy * unperturbed_accuracy_vec

# iterate over different perturbation amounts
original_W = enose_network.W.clone()
print(original_W)

for idx, scalar in enumerate(list_of_scalars):
    print(f"Starting Run with Scalar={scalar}...")

    # Perturb weights
    enose_network.perturb_weights(is_normal, scalar, stddev, is_multiplicative, is_correlated)
    perturbed_accuracy_vec[idx] = torch.mean(network.eval_model_accuracy(enose_network, test_conc, test_labels, num_data, num_training, test_batch_size, test_iteration_count, device))

    # Reset weights to normal
    enose_network.set_weights(original_W)

    print(f"Finished Run with Scalar={scalar}...")

# Saves plot of test accuracy
folder = "./eric_weight_perturb_2_figures"
plt.figure(figsize=(8, 4))
plt.plot(list_of_scalars, unperturbed_accuracy_vec, label='Test Accuracy w/o Perturb')
plt.plot(list_of_scalars, perturbed_accuracy_vec, label='Test Accuracy w/ Perturb')
plt.xlabel("Scalar")
plt.ylabel("Mean Accuracy")
plt.title(f"Mean Accuracy vs. Scalar ({lower_scalar} to {upper_scalar-1}): is_multiplicative={is_multiplicative}, is_correlated={is_correlated}")
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
if store_image:
    plt.savefig(f"{folder}/accuracy_by_perturb_{lower_scalar}_to_{upper_scalar}_is_multiplicative={is_multiplicative}_is_correlated={is_correlated}_m={m}_n={n}.pdf")
plt.close()
